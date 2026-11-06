"""Online difficulty filter: drop groups the CURRENT policy has already mastered.

WHY. An offline prefilter (difficulty_filter.py) removes prompts the DPO base solves more
than 62.5% of the time, but that snapshot goes stale once training works. In the math runs
the raw pass rate doubled (0.20 -> 0.42) by step ~339 and 22.3% of training groups scored
6/8 or 7/8, i.e. about a third of the rollout budget went to prompts already mastered. The
stock check_reward_nonzero_std cannot catch this: it keeps any group whose rewards are not
all identical, so a 7/8 group trains at full generation cost for a nearly zero gradient.

So the threshold is applied every step against the policy's own samples, and prompts
graduate out as they are mastered.

WHAT IT DROPS, with the reason logged as rollout/dynamic_filter/frac_<reason>:
  zero_std_<r>      every reward identical, the stock filter's case (no GRPO gradient).
                    With partial credit (code, IF) this includes e.g. eight 0.5s.
  mastered_<k>of<n> more than ADAPTIVE_FILTER_MAX_PASS of the group fully solved.

"Solved" means reward >= SOLVED_REWARD, NOT reward > 0. Code and IF train on partial credit
(pass rate over tests or constraints), so reward > 0 would count one passing test as a
solve and drop groups that still have most of their signal. SOLVED_REWARD must equal the
task's full reward (its *_REWARD_SCALE); run_code.sh and run_if.sh set it.

THRESHOLD. A group is dropped when its solved fraction is strictly ABOVE the threshold, the
offline filter's convention. At n=8:
  0.625  drops 6/8 and 7/8   (the offline number)
  0.75   drops 7/8           (default)
  0.875  drops nothing more than the stock filter, since 8/8 is already zero_std
Olmo 3 uses 0.625 only offline, against the DPO checkpoint. Online it retires prompts from
resampling at 0.875 (open-instruct no_resampling_pass_rate), which is the 7/8-and-up cut the
0.75 default reproduces here. The math run that copied 0.625 online dropped 19-29% of groups
every step as mastered, while AIME-2024's median per-problem pass rate under that policy was
0.938, i.e. it discarded the band the benchmark lives in.

COST. Strictly more groups are dropped than by the stock filter, so the async worker pulls
more prompts per step and step time rises. Watch rollout/dynamic_filter/frac_kept: if it
collapses the pool has become too easy for the policy, and the fix is re-filtering the data
rather than loosening this.

    --dynamic-sampling-filter-path adaptive_difficulty_filter.adaptive_difficulty_filter
"""

import os

import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput

MAX_PASS_RATE = float(os.environ.get("ADAPTIVE_FILTER_MAX_PASS", "0.75"))
SOLVED_REWARD = float(os.environ.get("SOLVED_REWARD", "1.0"))
_EPS = 1e-6


def _flatten(samples):
    for s in samples:
        if isinstance(s, list):
            yield from _flatten(s)
        else:
            yield s


def adaptive_difficulty_filter(args, samples, **kwargs):
    rewards = [float(s.get_reward_value(args)) for s in _flatten(samples)]
    n = len(rewards)
    if n == 0:
        return DynamicFilterOutput(keep=False, reason="empty_group")

    if not torch.tensor(rewards, dtype=torch.float64).std() > 1e-8:
        return DynamicFilterOutput(keep=False, reason=f"zero_std_{round(rewards[0], 1)}")

    n_solved = sum(1 for r in rewards if r >= SOLVED_REWARD - _EPS)
    if n_solved / n > MAX_PASS_RATE:
        return DynamicFilterOutput(keep=False, reason=f"mastered_{n_solved}of{n}")

    return DynamicFilterOutput(keep=True)
