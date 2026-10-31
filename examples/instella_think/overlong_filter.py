"""Overlong filtering for GRPO, in the drop-and-recentre form rather than the mask form.

Wired in with --custom-reward-post-process-path overlong_filter.overlong_filter_post_process
(examples/instella_think is on the Ray workers' PYTHONPATH, see run_async.sh:715). This replaces
miles' _post_process_rewards, which is the only place that can do BOTH halves of the job: the
loss-mask loop in train_data_conversion.py reads sample.remove_sample AFTER the reward
post-process returns, so one hook can drop the sample from the loss and recentre the advantages of
the survivors in the same pass.

WHY NOT --rollout-sample-filter-path
That flag exists and looks like the right tool, but its own help states the limitation exactly:
"This attribute does not determine whether the sample participates in advantage normalization."
That is the MASK form of overlong filtering -- zero the loss, leave the sample in the group mean --
and it is actively harmful here. Truncated samples carry negative advantage, so deleting them from
the gradient while leaving them in the mean deletes negative push and leaves the survivors net
POSITIVE. Measured on real grouped rollouts (study_tim/overlong_filter_test.py, T=1.0, strict
grader), as signed gradient-mass imbalance:

    filtering                  per-sample   token-level
    none                           +0.0%        -19.9%
    mask loss only                +25.6%        +19.4%     <- what the stock flag would give
    drop + recentre (this)         +0.0%         -7.7%

On the DPO base's 23% truncation rate the mask form reaches +91.6%/+83.5%. A uniform positive push
on sampled tokens is a SHARPENING pressure, and unopposed sharpening is this model's actual failure
mode (mean token probability 0.448 -> 0.743 over 410 steps), so the mask form would accelerate the
very thing we are trying to stop. Dropping and recentring keeps per-sample's exact-zero guarantee.

WHAT THIS IS FOR
Filtering does not fix the normalisation imbalance -- per-sample already makes that exactly zero
for any length distribution. It addresses a different failure, measured on this run's own engines
at step ~170: a punitive truncation reward teaches the policy to GUESS BEFORE THE CAP. Truncations
converted into wrong finished answers roughly one-for-one (23.3% -> 7.5% truncated, 1.2% -> 17.5%
finished-wrong) with no gain in the correct rate, emitted at a median 25687 tokens against the
32768 cap. Removing truncated samples from the loss removes the incentive to do that.

KNOWN COST, AND WHY IT IS LOGGED
Dropping truncated samples turns mixed groups into all-correct groups, which have zero advantage
spread and therefore contribute nothing -- on top of the ~60% of groups the existing dynamic filter
already discards for the same reason. On AIME at T=1.0 this took the base model from 10 usable
groups to 1, and step ~170 from 10 to 7. dapo-math is easier than AIME so the real rate should be
milder, but it is a genuine risk of starving the update, so every call logs the surviving-group
count. Watch it: if usable groups collapse, this filter is costing more signal than the bias it
removes, and the answer is a soft length penalty (DAPO's +3 component) instead of a hard drop.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_TRUNCATED = "truncated"


def _is_truncated(sample) -> bool:
    # Compare by value, not identity: Sample.Status is re-imported in the Ray workers and an
    # `is` check against a different class object would silently be False for every sample,
    # turning this filter into a no-op that still looks like it ran.
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == _TRUNCATED


def _to_groups(args, samples):
    if samples and isinstance(samples[0], list):
        return samples
    n = args.n_samples_per_prompt
    assert len(samples) % n == 0, f"{len(samples)} samples is not a multiple of n_samples_per_prompt={n}"
    return [samples[i : i + n] for i in range(0, len(samples), n)]


def overlong_filter_post_process(args, samples):
    """Return (raw_rewards, advantages) with truncated samples dropped and survivors recentred.

    Signature and contract match miles' _post_process_rewards: both lists are flat and in the
    same order as `samples`.
    """
    # Fail loudly rather than silently changing meaning if the run config moves away from the
    # regime this was written for.
    assert args.advantage_estimator in ("grpo", "gspo"), (
        f"overlong filtering assumes group-relative advantages, got {args.advantage_estimator}"
    )
    assert getattr(args, "rewards_normalization", True), (
        "--rewards-normalization is off, so this hook would be the only thing centring advantages"
    )

    groups = _to_groups(args, samples)
    flat = [s for g in groups for s in g]
    raw_rewards = [s.get_reward_value(args) for s in flat]
    std_norm = getattr(args, "grpo_std_normalization", False)

    advantages: list[float] = []
    n_groups_usable = 0
    n_dropped = 0
    n_groups_emptied = 0

    for group in groups:
        rewards = torch.tensor([s.get_reward_value(args) for s in group], dtype=torch.float)
        keep = torch.tensor([not _is_truncated(s) for s in group], dtype=torch.bool)
        n_dropped += int((~keep).sum())

        # A lone survivor centres to exactly 0, so <2 survivors means the group cannot produce
        # any gradient no matter what we do. Drop it whole so the loss mask reflects reality.
        if int(keep.sum()) < 2:
            n_groups_emptied += 1
            for s in group:
                s.remove_sample = True
            advantages.extend([0.0] * len(group))
            continue

        surviving = rewards[keep]
        centred = rewards - surviving.mean()
        if std_norm:
            centred = centred / (surviving.std() + 1e-6)
        centred = torch.where(keep, centred, torch.zeros_like(centred))

        for s, k in zip(group, keep.tolist(), strict=True):
            if not k:
                s.remove_sample = True
        if bool((centred.abs() > 1e-12).any()):
            n_groups_usable += 1
        advantages.extend(centred.tolist())

    logger.info(
        "overlong_filter: groups=%d usable=%d emptied=%d | samples=%d dropped=%d (%.1f%%)",
        len(groups),
        n_groups_usable,
        n_groups_emptied,
        len(flat),
        n_dropped,
        100.0 * n_dropped / max(len(flat), 1),
    )
    return raw_rewards, advantages
