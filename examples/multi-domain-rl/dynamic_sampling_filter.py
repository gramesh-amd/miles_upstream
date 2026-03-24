"""Combined dynamic sampling filter: zero-std check + prompt retirement.

Identical logic to examples/olmo_poc/dynamic_sampling_filter.py, kept here
for self-containment.

Usage:
    --dynamic-sampling-filter-path \
        examples.multi-domain-rl.dynamic_sampling_filter.check_reward_nonzero_std_and_retirement
"""

import logging
import os

import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

RETIREMENT_THRESHOLD = float(os.environ.get("PROMPT_RETIREMENT_THRESHOLD", "0.9375"))


def check_reward_nonzero_std_and_retirement(
    args, samples: list[Sample], **kwargs
) -> DynamicFilterOutput:
    """Filter groups with zero reward std OR excessively high pass rate."""
    rewards = [sample.get_reward_value(args) for sample in samples]
    rewards_t = torch.tensor(rewards, dtype=torch.float)

    std = rewards_t.std().item()
    if std == 0.0:
        return DynamicFilterOutput(
            keep=False,
            reason=f"zero_std_{round(rewards[0], 1)}",
        )

    if RETIREMENT_THRESHOLD < 1.0:
        pass_rate = (rewards_t > 0.0).float().mean().item()
        if pass_rate >= RETIREMENT_THRESHOLD:
            return DynamicFilterOutput(
                keep=False,
                reason=f"retired_pass_rate_{pass_rate:.2f}",
            )

    return DynamicFilterOutput(keep=True)
