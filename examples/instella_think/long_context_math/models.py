"""Model table, sampling arms, budgets and problem sets for the long-context math benchmark.

Every serving and sampling decision lives here so that `run_cell.py` stays model-agnostic.
A wrong entry is usually a silent bad number rather than an error: a model served past its
positional range produces fluent garbage, and a small model served at tp=8 wastes most of a node.

Parallelism: tp is the smallest value that fits the weights plus a 128k KV cache on one GPU, and
dp takes the rest of the node. These models all fit one MI325X, and data parallelism over
independent prompts measured 5.6x the throughput of tp=8 for the 16B MoE.

Context: `native` is the model's real positional range and `advertised` is what its config claims.
They differ for Instella-MoE (65,536 advertised, YaRN factor 40 over 4,096, so ~163k usable), so
serving past the advertised ceiling only needs max_position_embeddings raised. Where `native` falls
short of the served window, `yarn_from` says what to extend from and `rope_override` computes the
minimum factor, because over-extending YaRN degrades short-context behaviour.

Paths are relative to BENCH_MODELS_DIR, which must resolve to the same location on the host and on
every serving node (a shared filesystem).
"""

import json
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("BENCH_MODELS_DIR", "models")).resolve()
OUT_DIR = Path(os.environ.get("BENCH_OUT_DIR", "out/long_context_math")).resolve()
DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", PKG_DIR / "data")).resolve()

# OLMES `aime:2024::olmo3:adapt` / `aime:2025::olmo3:adapt`: zero-shot, temperature 0.6, top_p 0.95,
# no top_k, no penalties. Applied identically to every model in the standardised arm, including
# models whose cards require a penalty; a model that degenerates without one is shown degenerating.
# Its max_gen_toks of 16384 is deliberately not adopted, since the budget is the axis under test.
OLMES_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": -1}
DEFAULT_SAMPLING = OLMES_SAMPLING

_INSTELLA_EXTRA = ["--attention-backend", "triton", "--disable-shared-experts-fusion", "--enable-fp32-lm-head"]
_INSTELLA_ENV = {"FARSKIP_REFERENCE_DECODER_LAYER": "1"}

MODELS = {
    # Instella-MoE-16B-A3B after RL (R1c, step 399). Its production sampling is byte-identical to
    # OLMES_SAMPLING, so these cells are both its recommended and its standardised arm.
    "ours_r1c_399": {
        "path": "hf_R1_drgrpo_220_iter399",
        "tp": 1,
        "dp": 8,
        "native": 163840,
        "advertised": 65536,
        "extra": _INSTELLA_EXTRA,
        "env": _INSTELLA_ENV,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": -1},
        "note": "Instella-MoE-16B-A3B RL step 399",
    },
    # The post-training lineage (SFT -> DPO -> Think | RL), measured on the same budget curve. RL was
    # initialised from DPO. Same architecture and YaRN as ours; only the advertised ceiling differs.
    "instella_sft": {
        "path": "Instella-MoE-16B-A3B-SFT",
        "tp": 1,
        "dp": 8,
        "native": 163840,
        "advertised": 32768,
        "extra": _INSTELLA_EXTRA,
        "env": _INSTELLA_ENV,
        "sampling": OLMES_SAMPLING,
        "note": "Instella-MoE-16B-A3B-SFT",
    },
    "instella_dpo": {
        "path": "Instella-MoE-16B-A3B-DPO",
        "tp": 1,
        "dp": 8,
        "native": 163840,
        "advertised": 32768,
        "extra": _INSTELLA_EXTRA,
        "env": _INSTELLA_ENV,
        "sampling": OLMES_SAMPLING,
        "note": "Instella-MoE-16B-A3B-DPO",
    },
    "instella_think": {
        "path": "Instella-MoE-16B-A3B-Think",
        "tp": 1,
        "dp": 8,
        "native": 163840,
        "advertised": 32768,
        "extra": _INSTELLA_EXTRA,
        "env": _INSTELLA_ENV,
        "sampling": OLMES_SAMPLING,
        "note": "Instella-MoE-16B-A3B-Think",
    },
    "qwen3_30b_a3b_thinking": {
        "path": "qwen3-30b-a3b-thinking-2507",
        # ~60 GB of weights and ~98 KB of KV per token (48 layers x 4 kv heads), so tp=2.
        "tp": 2,
        "dp": 4,
        "native": 262144,
        "advertised": 262144,
        "extra": [],
        "env": {},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "note": "Qwen3-30B-A3B-Thinking-2507",
    },
    # Served from lmsys/gpt-oss-20b-bf16: gfx942 has no MXFP4 compute, and transformers' own
    # dequantize path drops the expert matrices.
    "gpt_oss_20b": {
        "path": "gpt-oss-20b-bf16",
        "tp": 1,
        "dp": 8,
        "native": 131072,
        "advertised": 131072,
        "yarn_from": 4096,
        # The template defaults to "medium". Set in both arms because it selects capability, not sampling.
        "chat_kwargs": {"reasoning_effort": "high"},
        "extra": [],
        "env": {},
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "note": "GPT-OSS-20B (reasoning_effort=high)",
    },
    "qwen3_5_4b": {
        "path": "Qwen3.5-4B",
        "tp": 1,
        "dp": 8,
        "native": 262144,
        "advertised": 262144,
        "extra": [],
        "env": {},
        # Card, "thinking mode for general tasks". Without presence_penalty 78% of its truncated traces
        # at 64k are repetition loops.
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
        "note": "Qwen3.5-4B",
    },
    "gemma4_e4b": {
        "path": "gemma-4-E4B-it",
        "tp": 1,
        "dp": 8,
        "native": 131072,
        "advertised": 131072,
        # The 128k window plus room for the forced answer is 1.8% past native.
        "yarn_from": 131072,
        # The template defaults enable_thinking to False, which silently benchmarks a non-thinking model.
        "chat_kwargs": {"enable_thinking": True},
        "extra": [],
        "env": {},
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "note": "Gemma-4-E4B-it (thinking)",
    },
    # Not in sglang's model registry; served through the transformers fallback.
    "smollm3_3b": {
        "path": "SmolLM3-3B",
        "tp": 1,
        "dp": 8,
        "native": 65536,
        "advertised": 65536,
        "yarn_from": 65536,
        "extra": ["--model-impl", "transformers"],
        "env": {},
        "chat_kwargs": {"enable_thinking": True},
        "sampling": {"temperature": 0.6, "top_p": 0.95},
        "note": "SmolLM3-3B",
    },
}

# Standardised twins: identical weights and serving shape, OLMES sampling, so the two arms can only
# differ in sampling. Models whose card sampling already equals OLMES_SAMPLING need no twin.
MODELS.update(
    {
        f"{key}_olmes": {**MODELS[key], "sampling": OLMES_SAMPLING, "note": f"{MODELS[key]['note']} @ OLMES sampling"}
        for key in ("qwen3_30b_a3b_thinking", "qwen3_5_4b", "gemma4_e4b", "gpt_oss_20b")
    }
)

# Sampling sweep for our own model. Each arm changes one thing from the previous one so an effect can
# be attributed. Tuned beyond our recommended config, so keep these out of cross-model comparisons.
MODELS.update(
    {
        "ours_r1c_399_topk": {
            **MODELS["ours_r1c_399"],
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "note": "ours step 399, top_k=20",
        },
        "ours_r1c_399_pen": {
            **MODELS["ours_r1c_399"],
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
            "note": "ours step 399, top_k=20 + presence_penalty=1.5",
        },
        "ours_r1c_399_t08": {
            **MODELS["ours_r1c_399"],
            "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
            "note": "ours step 399, T=0.8 + top_k=20 + presence_penalty=1.5",
        },
    }
)

# 90k is Polaris's evaluation budget; 128k matches OLMES's max_gen_toks of 131,072.
LENGTHS = {"32k": 32768, "64k": 65536, "90k": 92160, "128k": 131072}

BENCHES = [
    ("aime2024", "aime-2024-olmes"),
    ("aime2025", "aime-2025-olmes"),
    ("hmmt_feb_2025", "hmmt-feb-2025-olmes"),
    ("beyondaime", "beyondaime-olmes"),
]


def model_path(model: dict) -> Path:
    return MODELS_DIR / model["path"]


def num_problems() -> int:
    return sum(sum(1 for line in open(DATA_DIR / f"{fn}.jsonl") if line.strip()) for _, fn in BENCHES)


def _rope_theta(model: dict) -> float | None:
    try:
        return json.loads((model_path(model) / "config.json").read_text()).get("rope_theta")
    except OSError:
        return None


def rope_override(model: dict, ctx: int) -> dict | None:
    """Minimum YaRN extension to reach `ctx`, or None when the model already covers it."""
    if ctx <= model["native"]:
        return None
    base = model.get("yarn_from")
    if base is None:
        raise ValueError(f"{model['note']} cannot reach {ctx} and has no yarn_from")
    scaling = {"rope_type": "yarn", "factor": round(ctx / base, 4), "original_max_position_embeddings": base}
    # transformers 5 builds rope_parameters from rope_scaling, so an override without rope_theta leaves
    # it None and YaRN init fails. Native sglang models read rope_theta from the config and ignore it.
    if "--model-impl" in model.get("extra", []):
        theta = _rope_theta(model)
        if theta is not None:
            scaling["rope_theta"] = theta
    return {"rope_scaling": scaling}
