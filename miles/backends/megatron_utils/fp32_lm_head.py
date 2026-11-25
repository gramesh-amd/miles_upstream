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

Reimplementing the matmul means the tensor-parallel plumbing has to be reproduced rather
than inherited. `_tp_plumbing` and `_maybe_gather_output` do that by composing Megatron's
own autograd-aware collectives around a purely LOCAL fp32 matmul, so the collectives stay
Megatron's: `copy_to_tensor_model_parallel_region` for the backward grad_input all-reduce,
`gather_from_sequence_parallel_region(tensor_parallel_output_grad=False)` for the
sequence-parallel gather plus its reduce-scatter, and
`gather_from_tensor_model_parallel_region` for the optional output re-gather. The local
matmul therefore yields the same `output_parallel` vocab shard ColumnParallelLinear does.
(The original version instead asserted tp==1; TP is now supported.)

A fused LM-head bias is still rejected, because the local matmul has no bias term and
dropping it silently would change the function rather than its precision.

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


def _check_supported(layer: Any) -> None:
    """Reject only what we genuinely cannot reproduce.

    Tensor parallelism IS supported -- see `_tp_plumbing`. A fused LM-head bias is not,
    because the local matmul has no bias term and silently dropping it would change the
    function rather than its precision.
    """
    if getattr(layer, "bias", None) is not None and not getattr(layer, "skip_bias_add", False):
        raise RuntimeError("fp32_lm_head does not handle a fused LM-head bias")


def _tp_world_size() -> int:
    """TP world size, or 1 when the model-parallel groups are not initialised.

    `parallel_state.get_tensor_model_parallel_world_size()` ASSERTS rather than returning 1
    before `initialize_model_parallel`, so calling it bare makes this module unusable (and
    untestable) outside a live distributed run.
    """
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_world_size()
    except (AssertionError, ImportError, AttributeError):
        return 1


def _tp_plumbing(layer: Any, input_: torch.Tensor) -> torch.Tensor:
    """Reproduce ColumnParallelLinear's INPUT-side tensor-parallel handling.

    Mirrors `ColumnParallelLinear.forward` (tensor_parallel/layers.py:985-1030). Three
    things happen there that a bare `x @ W.T` would skip, and only the first would ever
    announce itself:

    1. The weight is sharded over the vocab dim, so the local matmul yields a vocab SLICE.
       That is the correct ColumnParallelLinear contract (`output_parallel`); the optional
       re-gather is handled by the caller.
    2. `copy_to_tensor_model_parallel_region` is identity forward but an ALL-REDUCE in
       backward -- that is how each rank's grad_input is summed over the TP group. Megatron
       does this either here or inside `_forward_impl` when `allreduce_dgrad` is set. Since
       we replace `_forward_impl`, we must always route through it ourselves.
    3. With sequence parallelism the hidden states arrive SCATTERED along the sequence dim
       and `_forward_impl` all-gathers them, reduce-scattering grad_input on the way back.
       `tensor_parallel_output_grad=False` selects exactly that backward.

    These are autograd-aware primitives, so composing them around a purely local matmul
    keeps the collectives in Megatron's hands instead of reimplementing them.
    """
    if _tp_world_size() == 1:
        return input_

    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
    )

    tp_group = getattr(layer, "tp_group", None)
    if getattr(layer, "sequence_parallel", False):
        return gather_from_sequence_parallel_region(
            input_, tensor_parallel_output_grad=False, group=tp_group
        )
    return copy_to_tensor_model_parallel_region(input_, group=tp_group)


def _maybe_gather_output(
    layer: Any, output: torch.Tensor, runtime_gather_output: bool | None
) -> torch.Tensor:
    """Mirror the `gather_output` tail of ColumnParallelLinear.forward (layers.py:1032-1041)."""
    if _tp_world_size() == 1:
        return output
    gather = getattr(layer, "gather_output", False)
    if runtime_gather_output is not None:
        gather = runtime_gather_output
    if not gather:
        return output
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    return gather_from_tensor_model_parallel_region(output, group=getattr(layer, "tp_group", None))


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
            _check_supported(layer)
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
                # input-side TP handling -> local fp32 matmul -> optional output re-gather,
                # matching ColumnParallelLinear.forward's three stages.
                x = _tp_plumbing(_layer, input_)
                output = _Fp32Logits.apply(x, effective)
                output = _maybe_gather_output(_layer, output, runtime_gather_output)
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
