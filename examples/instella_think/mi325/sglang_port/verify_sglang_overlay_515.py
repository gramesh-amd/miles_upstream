#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Validate that the FarSkip SGLang v5.15 overlay landed correctly.

Standalone and inference-only: run this right after
update_sglang_workspace_files.sh, with no dependency on RL / Megatron. Reads
the overlaid sglang root from SGLANG_PYTHON (default /sgl-workspace/sglang/python).
Exit 0 if every check passes, 1 otherwise.
"""
import importlib
import os
import sys

SGLANG = os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python")

ok = True


def check(label, fn):
    global ok
    try:
        fn()
        print("OK   " + label)
    except Exception as e:
        print("FAIL " + label + ": " + str(e))
        ok = False


def _read(rel):
    return open(os.path.join(SGLANG, rel)).read()


def farskip_reference_decoder():
    m = importlib.import_module("sglang.srt.models.deepseek_v2")
    m.DeepseekV2ReferenceFarSkipDecoderLayer


def farskip_overlapped_decoder():
    m = importlib.import_module("sglang.srt.models.deepseek_v2")
    m.DeepseekV2FarSkipDecoderLayer


def instella_arch():
    # The arch file must be importable and register InstellaMoEForCausalLM, else
    # sglang silently falls back to TransformersForCausalLM (no FarSkip).
    m = importlib.import_module("sglang.srt.models.instella_moe")
    assert m.EntryClass.__name__ == "InstellaMoEForCausalLM"


def instella_arch_recognized():
    # model_config must treat InstellaMoEForCausalLM as a deepseek-style MoE arch.
    assert '"InstellaMoEForCausalLM" in self.hf_config.architectures' in _read(
        "sglang/srt/configs/model_config.py"
    )


def gated_attention():
    for rel in (
        "sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py",
        "sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py",
    ):
        assert "if self.gated_attention:" in _read(rel), rel
    assert "self.gate_proj = ColumnParallelLinear(" in _read(
        "sglang/srt/models/deepseek_v2.py"
    )


def weight_loader_safety():
    s = _read("sglang/srt/models/deepseek_common/deepseek_weight_loader.py")
    assert 'weight_loader = getattr(param, "weight_loader", None)' in s
    assert "weight_loader = param.weight_loader" not in s


def communicator_no_all_reduce():
    m = importlib.import_module("sglang.srt.layers.communicator")
    assert hasattr(m.LayerCommunicator, "prepare_mlp_no_all_reduce")


def async_comm_ops():
    m = importlib.import_module("sglang.srt.distributed.communication_op")
    m.CustomNCCLWork
    m.async_tensor_model_parallel_all_reduce
    m.async_reduce_scatter
    d = importlib.import_module("sglang.srt.layers.dp_attention")
    d.dp_gather_partial_async_farskip
    # the upstream two-batch-overlap helper must survive alongside ours
    d.dp_gather_partial_async
    lin = importlib.import_module("sglang.srt.layers.linear")
    lin.AsyncRowParallelLinear
    # deepseek_v2 references these by bare name inside the overlapped FarSkip paths.
    # A missing import is invisible until an overlapped run reaches that line, so
    # assert they resolved into the module namespace.
    dsv2 = importlib.import_module("sglang.srt.models.deepseek_v2")
    for name in (
        "AsyncRowParallelLinear",
        "async_tensor_model_parallel_all_reduce",
        "async_reduce_scatter",
        "dp_gather_partial_async_farskip",
    ):
        assert hasattr(dsv2, name), "deepseek_v2 missing " + name


def pynccl_stream_priority():
    # FarSkip overlap depends on the comm stream being high priority AND the NCCL
    # warmup running on that very stream. A regression here is silent (throughput
    # only), so assert the mechanism, not just the flag.
    s = _read("sglang/srt/distributed/device_communicators/pynccl.py")
    assert 'os.environ.get("FARSKIP_OVERLAPPED_DECODER_LAYER", "0")' in s
    assert 'priority=-1 if comm_high_priority == "1" else 0' in s
    assert "with torch.cuda.stream(self.stream):" in s
    assert "def _resolve_stream(" in s


def unquant_aiter_moe():
    s = _read("sglang/srt/layers/quantization/unquant.py")
    for needle in (
        "def unshuffle_weight(",
        "def _make_aiter_weight_loader(",
        # The helpers above must also be wired up: without the guard a reload
        # double-shuffles, and without the re-attachment the Megatron -> SGLang
        # RL weight broadcast lands on unshuffled weights.
        '"is_shuffled", False',
        "layer.w13_weight.weight_loader = _make_aiter_weight_loader(",
        "layer.w2_weight.weight_loader = _make_aiter_weight_loader(",
    ):
        assert needle in s, needle


def shuffle_invertible():
    import torch
    from aiter.ops.shuffle import shuffle_weight
    from sglang.srt.layers.quantization.unquant import unshuffle_weight

    x = torch.randn(8, 2816, 2048, dtype=torch.bfloat16)
    assert torch.equal(x, unshuffle_weight(shuffle_weight(x.clone(), (16, 16)), (16, 16)))


def mla_decode_shapes():
    # kv_lora_rank 512 + qk_rope_head_dim 32 -> Lk == 544. Without this branch
    # BLOCK_DMODEL rounds up to 1024 against a 512-wide acc and the decode
    # kernel fails to compile.
    assert "elif Lk == 544:" in _read(
        "sglang/srt/layers/attention/triton_ops/decode_attention.py"
    )


def kv_splits():
    # The env var is read in two places (backend + server_args) and consumed by the
    # triton kernel, which v5.15 moved out of triton_backend.py into triton_ops/metadata.py.
    # A partial overlay would silently drop the extra kernel args, so check all three.
    assert "SGLANG_FORCE_KV_SPLITS" in _read(
        "sglang/srt/layers/attention/triton_backend.py"
    )
    assert "SGLANG_FORCE_KV_SPLITS" in _read("sglang/srt/server_args.py")
    assert "force_seqlen_threshold" in _read(
        "sglang/srt/layers/attention/triton_ops/metadata.py"
    )


for mod in (
    "sglang.srt.models.deepseek_v2",
    "sglang.srt.layers.communicator",
):
    check("import " + mod, lambda mod=mod: importlib.import_module(mod))
check("instella_moe arch (InstellaMoEForCausalLM)", instella_arch)
check("instella arch recognized by model_config", instella_arch_recognized)
check("gated MLA attention", gated_attention)
check("weight_loader safety", weight_loader_safety)
check("farskip reference decoder", farskip_reference_decoder)
check("communicator prepare_mlp_no_all_reduce", communicator_no_all_reduce)
check("farskip overlapped decoder", farskip_overlapped_decoder)
check("async comm ops (all_reduce / reduce_scatter / gather / linear)", async_comm_ops)
check("pynccl high-priority comm stream", pynccl_stream_priority)
check("unquant aiter moe", unquant_aiter_moe)
check("unshuffle(shuffle(x))==x", shuffle_invertible)
check("MLA decode Lk=544", mla_decode_shapes)
check("dp-attn kv-splits", kv_splits)

sys.exit(0 if ok else 1)
