"""Dr. GRPO pg_loss reducer: sum over tokens divided by a CONSTANT (arXiv 2503.20783).

Wired in with --custom-pg-loss-reducer-function-path drgrpo_reducer.get_pg_loss_reducer
(examples/instella_think is on the Ray workers' PYTHONPATH via RUNTIME_ENV_JSON in run_async.sh).

WHAT IT REPLACES. With --calculate-per-token-loss off, miles' default reducer is
sum_of_sample_mean (cp_utils.py): each sample's token sum divided by ITS OWN length, then
loss.py divides by global_batch_size (a sample count, constant per step). So every sample gets
equal total weight and a token in a long sample gets proportionally less. On code that matters
because the long samples are the truncated ones: on the code16k run's dumps (52 steps, 13,312
samples) 28% truncate at the cap, average 1.5x the mean length and score 0.03, so per-sample
weighting under-pushes the largest pool of negatives per token. Dividing by a constant gives
every token the same weight regardless of which sample it sits in.

The flip side, measured on the same dumps: truncated samples' share of gradient mass rises from
29% to 41% under this reducer, so pair it with OVERLONG_FILTER=1 (drops them from the loss and
recentres the survivors; 71% of groups keep a reward spread).

WHY --calculate-per-token-loss MUST STAY OFF. With it on, loss.py returns num_tokens as
Megatron's normalizer, i.e. it divides by the REALISED batch token count, which reintroduces
length dependence (per-token step grows as responses shorten) and no reducer can undo it.
miles' examples/experimental/DrGRPO reducer returns a plain sum in that mode, which makes it
identical to the default. This one asserts instead.

DIVISOR (env DRGRPO_DIVISOR, required). It only rescales the effective learning rate. Setting it
to the run's step-0 mean response length makes the initial step size match the per-sample
reducer on average (sum of len_i/D over the batch equals the sample count). 10686 is the code16k
run's step-0 mean at a 16384 cap; a different cap or checkpoint has a different mean. It is read
from the environment, so it must be in run_async.sh's RUNTIME_ENV_JSON to reach the trainer
actors; a missing value fails here instead of silently falling back to a default.
"""

import os

import torch

from miles.backends.training_utils.parallel import get_parallel_state


def _divisor() -> float:
    raw = os.environ.get("DRGRPO_DIVISOR", "")
    assert raw, "DRGRPO_DIVISOR is not set in the trainer's environment (see RUNTIME_ENV_JSON in run_async.sh)"
    divisor = float(raw)
    assert divisor > 0, f"DRGRPO_DIVISOR must be positive, got {raw}"
    return divisor


def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
):
    """Called by loss_hub/losses.py as f(total_lengths, response_lengths, pg_loss_masks, cptl).

    Only pg_loss uses this reducer; clipfrac, ppo_kl, entropy and KL keep miles' default.
    """
    assert not calculate_per_token_loss, (
        "drgrpo_reducer requires --calculate-per-token-loss OFF (PER_TOKEN_LOSS=0): with it on, "
        "Megatron divides by the realised token count and the denominator is no longer constant."
    )
    # With context parallelism the pg_loss tensor holds only this rank's chunk of each sequence,
    # so splitting it by full response_lengths would misalign tokens and masks.
    cp_size = get_parallel_state().cp.size
    assert cp_size == 1, f"drgrpo_reducer does not support context parallelism (cp size {cp_size})"
    divisor = _divisor()

    def sum_of_constant_mean(x: torch.Tensor) -> torch.Tensor:
        return sum(
            (x_i * loss_mask_i).sum() / divisor
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=True)
        )

    return sum_of_constant_mean
