"""Compute the LM-head logits in fp32, in both the log-prob pass and the gradient pass.

Why this exists
---------------
Instella's logits are large: measured on the DPO checkpoint over 3974 real rollout tokens,
the per-token max |logit| has median 118 (p90 168, max 198). bf16 keeps 8 significand bits,
so its grid step there is 0.5 nats. The trainer multiplies bf16 hidden states by a bf16
`output_layer.weight` and `Float16Module` upcasts the *result* to fp32 afterwards, which
recovers nothing, so every trainer log-prob carries that quantisation.

Reading the head out in fp32 instead, from the identical hidden state, changes the log-probs
by 0.0893 nats on average. For comparison, `train_rollout_logprob_abs_diff` reports 0.0989,
and the same measurement on Qwen3.6-35B-A3B (logit median 24, grid step 0.125) gives 0.0135.
So this readout is essentially the whole train/rollout gap, and it is a property of the
checkpoint's logit scale rather than of the inference stack.

Why it must cover BOTH passes
-----------------------------
An earlier version promoted only the forward-only pass, on the theory that touching the
gradient pass risked the grad buffers. That is wrong, and measurably so. The PPO ratio is
exp(new - old): `old` comes from the forward-only log-prob pass (`rollout_data["log_probs"]`)
and `new` from the training pass, two separate computations of the same quantity. The
readout error is spiky rather than smooth -- median 0.026 but P(>0.2) = 17% and P(>0.28) =
9.9% -- so promoting one side alone would push about a sixth of all tokens outside the
eps_clip band on every step, which is worse than the ~8% `pg_clipfrac` seen today at a point
where the ratio should be exactly 1. Both sides move together or neither does.

Implementation
--------------
The gradient pass cannot simply call `ColumnParallelLinear.forward` with fp32 inputs:
Megatron's fused wgrad kernels take half-precision operands, and its custom autograd
function accumulates into `weight.main_grad`. So the matmul is reimplemented here as an
autograd Function that computes in fp32 and mirrors Megatron's accumulation contract from
`tensor_parallel/layers.py`: add the weight gradient into `main_grad`, set
`grad_added_to_main_grad` and return a dummy weight grad when DDP expects one, and return
None for it otherwise.

Reimplementing the matmul means skipping the tensor-parallel plumbing (sequence-parallel
gather, output gather, bias), so this refuses to install unless the head is genuinely
untouched by tensor parallelism. It fails loudly rather than silently computing the wrong
function.

Cost: an fp32 copy of the head weight (vocab x hidden x 4 bytes) plus an fp32 weight-gradient
temporary during backward. The fp32 logits themselves already exist today, since
`Float16Module` upcasts them.
"""

from __future__ import annotations

import contextlib
from argparse import Namespace
from typing import Any, Iterator, Sequence

import torch


def _find_output_layers(model: Sequence[Any]) -> list[Any]:
    """Collect the `output_layer` of each model chunk, unwrapping DDP/Float16Module."""
    layers = []
    for chunk in model:
        inner = chunk
        # DDP -> Float16Module -> GPTModel; depth varies with configuration, and
        # only the last pipeline stage owns an output_layer at all.
        for _ in range(8):
            if hasattr(inner, "output_layer"):
                break
            if not hasattr(inner, "module"):
                inner = None
                break
            inner = inner.module
        layer = getattr(inner, "output_layer", None) if inner is not None else None
        if layer is not None:
            layers.append(layer)
    return layers


def _assert_no_tensor_parallel(layer: Any) -> None:
    """Refuse to run if the head is sharded or biased, since we bypass that plumbing."""
    from megatron.core import parallel_state

    tp = parallel_state.get_tensor_model_parallel_world_size()
    if tp != 1:
        raise RuntimeError(
            f"fp32_lm_head bypasses tensor-parallel plumbing but tensor parallel size is {tp}. "
            "Either run the head with tp=1 or extend this to gather/reduce like "
            "ColumnParallelLinear does."
        )
    if getattr(layer, "bias", None) is not None and not getattr(layer, "skip_bias_add", False):
        raise RuntimeError("fp32_lm_head does not handle a fused LM-head bias")


class _Fp32Logits(torch.autograd.Function):
    """y = x @ W^T computed in fp32, accumulating dW into Megatron's main_grad."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inp, weight)
        return torch.matmul(inp.float(), weight.float().t())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, weight = ctx.saved_tensors
        grad_output = grad_output.float()

        grad_input = torch.matmul(grad_output, weight.float()).to(inp.dtype)

        # [tokens, vocab]^T @ [tokens, hidden] -> [vocab, hidden], matching weight.
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        i2 = inp.reshape(-1, inp.shape[-1]).float()
        grad_weight_fp32 = torch.matmul(g2.t(), i2)

        main_grad = getattr(weight, "main_grad", None)
        if main_grad is None:
            return grad_input, grad_weight_fp32.to(weight.dtype)

        main_grad.add_(grad_weight_fp32.to(main_grad.dtype))
        if not hasattr(weight, "grad_added_to_main_grad"):
            # Megatron returns None here; autograd treats it as a zero gradient.
            return grad_input, None

        # DDP with overlap_grad_reduce needs a real tensor so its hooks stay on the main
        # backward thread; the contents are ignored because of the flag below.
        weight.grad_added_to_main_grad = True
        dummy = torch.empty(
            main_grad.shape, dtype=inp.dtype, device=main_grad.device, requires_grad=False
        )
        return grad_input, dummy


@contextlib.contextmanager
def fp32_lm_head(args: Namespace, model: Sequence[Any]) -> Iterator[None]:
    """Promote the LM-head matmul to fp32 for the duration of the block.

    Safe to use around both the forward-only log-prob pass and the training pass, and it
    must be used around both so the two sides of the PPO ratio stay comparable.
    """
    if not getattr(args, "fp32_lm_head", False):
        yield
        return

    patched: list[tuple[Any, Any, bool]] = []
    try:
        for layer in _find_output_layers(model):
            _assert_no_tensor_parallel(layer)
            # Distinguish a pre-existing instance attribute from the class method, so
            # the restore below puts the object back exactly as it was found instead
            # of leaving a bound method shadowing the class.
            had_own_forward = "forward" in vars(layer)
            original = layer.forward

            def promoted(
                input_: torch.Tensor,
                weight: torch.Tensor | None = None,
                runtime_gather_output: bool | None = None,
                _layer: Any = layer,
                _original: Any = original,
            ):
                effective = weight if weight is not None else getattr(_layer, "weight", None)
                if effective is None:
                    return _original(input_, weight, runtime_gather_output)
                output = _Fp32Logits.apply(input_, effective)
                output_bias = _layer.bias if getattr(_layer, "skip_bias_add", False) else None
                return output, output_bias

            layer.forward = promoted
            patched.append((layer, original, had_own_forward))
        yield
    finally:
        for layer, original, had_own_forward in patched:
            if had_own_forward:
                layer.forward = original
            else:
                vars(layer).pop("forward", None)
