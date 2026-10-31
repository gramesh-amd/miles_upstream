#!/usr/bin/env python3
"""Log IFEval's LOOSE metric alongside the strict one, per eval dataset.

Wired in with:
    --custom-eval-rollout-log-function-path if_eval_log.log_eval_rollout_data

WHY
miles logs `eval/<name>` as the mean reward, and this recipe's reward is STRICT
all-or-nothing at eval (see if_reward.py). Published IFEval numbers are almost always
LOOSE, so the two are not comparable: measured on this model at step 39, the identical
responses score 73.2% strict and 79.7% loose. Reporting only strict makes the run look
~6 points worse than a history of loose numbers, which is a metric-definition artefact
rather than a regression.

Strict stays the training/eval reward -- it is the harsher and more honest signal. This
only ADDS the comparable numbers, as:

    eval/<name>-loose              prompt-level loose accuracy
    eval/<name>-strict_recomputed  strict, recomputed here (sanity check vs eval/<name>)
    eval/<name>-loose_minus_strict the metric-definition gap
    eval/<name>-inst_loose         instruction-level loose accuracy
    eval/<name>-inst_strict        instruction-level strict accuracy
    eval/<name>-unscorable_frac    samples with no constraints in metadata (must be 0)

The inst_* pair completes the 2x2 that OLMES reports for ifeval/ifbench
(metric_kwargs aggregation_levels ["prompt","inst"] x strictness_levels
["strict","loose"]); the names map to its {prompt,inst}_level_{strict,loose}_acc.
Instruction-level is micro-averaged over constraints, matching IFEval -- so a prompt
carrying six constraints moves it six times as far as a prompt carrying one. That makes
it much less jumpy than the prompt-level number at --n-samples-per-eval-prompt 1, and it
degrades gradually where prompt-level strict is all-or-nothing.

Returns False so miles' default eval logging still runs
(miles/ray/rollout/metrics.py:22-25 treats True as "skip the default").
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics=None) -> bool:
    try:
        import if_reward
        from miles.utils.metric_utils import compute_rollout_step
        from miles.utils.tracking_utils import tracking
    except Exception as exc:  # never let a metrics add-on break the run
        logger.warning("if_eval_log: import failed (%s); skipping loose metrics", exc)
        return False

    log_dict: dict[str, float] = {}
    try:
        for name, payload in data.items():
            samples = payload.get("samples")
            if not samples:
                continue
            strict_vals: list[float] = []
            loose_vals: list[float] = []
            inst_strict = inst_loose = inst_total = 0
            unscorable = 0
            for s in samples:
                got = if_reward.score_counts(s)
                if got is None:
                    unscorable += 1
                    continue
                strict_n, loose_n, n_inst = got
                strict_vals.append(float(strict_n == n_inst))
                loose_vals.append(float(loose_n == n_inst))
                inst_strict += strict_n
                inst_loose += loose_n
                inst_total += n_inst
            if not loose_vals:
                continue
            n = len(loose_vals)
            strict = sum(strict_vals) / n
            loose = sum(loose_vals) / n
            log_dict[f"eval/{name}-loose"] = loose
            log_dict[f"eval/{name}-strict_recomputed"] = strict
            log_dict[f"eval/{name}-loose_minus_strict"] = loose - strict
            log_dict[f"eval/{name}-unscorable_frac"] = unscorable / (n + unscorable)
            if inst_total:
                log_dict[f"eval/{name}-inst_loose"] = inst_loose / inst_total
                log_dict[f"eval/{name}-inst_strict"] = inst_strict / inst_total

        if log_dict:
            # The wandb backend logs without an explicit step and keys off the step field
            # inside the dict (tracking_utils/base.py:57 + define_metric), so including
            # eval/step lands these on the same point as the default eval metrics.
            log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
            logger.info("eval %s loose: %s", rollout_id, log_dict)
            tracking.log(args, log_dict, step_key="eval/step")
    except Exception as exc:
        logger.warning("if_eval_log: loose metric computation failed (%s)", exc, exc_info=True)

    return False  # let miles' default eval logging proceed
