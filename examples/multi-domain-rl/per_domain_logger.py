"""Per-domain reward metrics logging for multi-domain RL.

Logs domain-specific metrics (math, code, ifbench) to WandB and a TSV file.

WandB key convention: ``{metric_group}/{domain}`` so that all domains for
the same metric appear on a single chart (e.g. ``solve_rate/math``,
``solve_rate/code``, ``solve_rate/ifbench`` share one panel).

Usage:
    --custom-rollout-log-function-path \
        examples.multi-domain-rl.per_domain_logger.log_per_domain_metrics
"""

import logging
import os
from collections import defaultdict

import numpy as np

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_pass_rate, compute_rollout_step, has_repetition
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

TSV_HEADER = (
    "step\tdomain\tcount\treward_mean\treward_min\treward_max\tsolve_rate\treward_std\t"
    "resp_len_mean\tresp_len_median\tresp_len_solved\tresp_len_unsolved\t"
    "truncation_rate\trepetition_frac\treward_frac_zero\treward_frac_partial\treward_frac_perfect\n"
)


def _compute_domain_metrics(rewards, resp_lens_all, resp_lens_solved, resp_lens_unsolved,
                            truncated_count, repetition_count):
    if not rewards:
        return {}

    arr = np.array(rewards)
    n = len(rewards)
    return {
        "count": n,
        "reward_mean": arr.mean().item(),
        "reward_min": arr.min().item(),
        "reward_max": arr.max().item(),
        "reward_std": arr.std().item(),
        "solve_rate": (arr > 0.0).mean().item(),
        "resp_len_mean": float(np.mean(resp_lens_all)) if resp_lens_all else 0.0,
        "resp_len_median": float(np.median(resp_lens_all)) if resp_lens_all else 0.0,
        "resp_len_solved": float(np.mean(resp_lens_solved)) if resp_lens_solved else 0.0,
        "resp_len_unsolved": float(np.mean(resp_lens_unsolved)) if resp_lens_unsolved else 0.0,
        "truncation_rate": truncated_count / n,
        "repetition_frac": repetition_count / n,
        "reward_frac_zero": (arr == 0.0).mean().item(),
        "reward_frac_partial": ((arr > 0.0) & (arr < 1.0)).mean().item(),
        "reward_frac_perfect": (arr == 1.0).mean().item(),
    }


def log_per_domain_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Custom rollout log function: per-domain metrics grouped by metric type.

    Returns True to skip default logging.
    """
    if not samples:
        return True

    step = compute_rollout_step(args, rollout_id)

    # --- Group samples by domain ---
    domain_groups = defaultdict(list)
    for s in samples:
        meta = s.metadata if isinstance(s.metadata, dict) else {}
        domain = meta.get("rm_type", "unknown")
        domain_groups[domain].append(s)

    # --- Per-domain metrics ---
    domain_metrics = {}
    for domain, group in domain_groups.items():
        rewards = [s.get_reward_value(args) for s in group if s.reward is not None]
        resp_lens_all = [s.effective_response_length for s in group]
        resp_lens_solved = [s.effective_response_length for s in group if s.get_reward_value(args) > 0]
        resp_lens_unsolved = [s.effective_response_length for s in group if s.get_reward_value(args) <= 0]
        truncated_count = sum(1 for s in group if s.status == Sample.Status.TRUNCATED)
        repetition_count = sum(1 for s in group if has_repetition(s.response))
        domain_metrics[domain] = _compute_domain_metrics(
            rewards, resp_lens_all, resp_lens_solved, resp_lens_unsolved,
            truncated_count, repetition_count,
        )

    # --- Build log dict with metric/{domain} key structure ---
    log_dict = {**(rollout_extra_metrics or {})}

    DOMAIN_KEYS = (
        "reward_mean", "reward_min", "reward_max", "reward_std",
        "solve_rate", "count",
        "resp_len_mean", "resp_len_median", "resp_len_solved", "resp_len_unsolved",
        "truncation_rate", "repetition_frac",
        "reward_frac_zero", "reward_frac_partial", "reward_frac_perfect",
    )
    for domain, m in domain_metrics.items():
        for key in DOMAIN_KEYS:
            log_dict[f"domain/{key}/{domain}"] = m[key]

    # --- Per-domain pass@k ---
    for domain, group in domain_groups.items():
        prompt_groups = defaultdict(list)
        for s in group:
            if s.group_index is not None:
                prompt_groups[s.group_index].append(s)
        if not prompt_groups:
            continue
        group_sizes = [len(g) for g in prompt_groups.values()]
        group_size = group_sizes[0]
        if not (all(gs == group_size for gs in group_sizes) and group_size > 1):
            continue
        flat_rewards = []
        for gi in sorted(prompt_groups.keys()):
            for s in prompt_groups[gi]:
                flat_rewards.append(s.get_reward_value(args))
        for key, val in compute_pass_rate(flat_rewards=flat_rewards, group_size=group_size).items():
            log_dict[f"domain/{key}/{domain}"] = val

    # --- Zero-std group stats (how much data the dynamic filter would drop) ---
    all_prompt_groups = defaultdict(list)
    for s in samples:
        if s.group_index is not None:
            all_prompt_groups[s.group_index].append(s)

    if all_prompt_groups:
        zero_std_all_zero = 0
        zero_std_all_positive = 0
        zero_std_total = 0
        for gidx, grp in all_prompt_groups.items():
            rews = [s.get_reward_value(args) for s in grp]
            if len(set(rews)) == 1:
                zero_std_total += 1
                if rews[0] == 0.0:
                    zero_std_all_zero += 1
                elif rews[0] > 0.0:
                    zero_std_all_positive += 1
        n_groups = len(all_prompt_groups)
        log_dict["zero_std/total_groups"] = n_groups
        log_dict["zero_std/zero_std_count"] = zero_std_total
        log_dict["zero_std/zero_std_frac"] = zero_std_total / n_groups if n_groups else 0.0
        log_dict["zero_std/all_zero_count"] = zero_std_all_zero
        log_dict["zero_std/all_positive_count"] = zero_std_all_positive

    # --- Aggregate (all-domain) metrics ---
    all_rewards = [s.get_reward_value(args) for s in samples if s.reward is not None]
    if all_rewards:
        arr = np.array(all_rewards)
        log_dict["rollout_agg/reward_mean"] = arr.mean().item()
        log_dict["rollout_agg/reward_min"] = arr.min().item()
        log_dict["rollout_agg/reward_max"] = arr.max().item()
        log_dict["rollout_agg/reward_std"] = arr.std().item()
        log_dict["rollout_agg/solve_rate"] = (arr > 0.0).mean().item()

    if all_prompt_groups:
        group_sizes = [len(g) for g in all_prompt_groups.values()]
        group_size = group_sizes[0]
        if all(gs == group_size for gs in group_sizes) and group_size > 1:
            flat_rewards = []
            for gi in sorted(all_prompt_groups.keys()):
                for s in all_prompt_groups[gi]:
                    flat_rewards.append(s.get_reward_value(args))
            for key, val in compute_pass_rate(flat_rewards=flat_rewards, group_size=group_size).items():
                log_dict[f"rollout_agg/{key}"] = val

    resp_lengths = [s.effective_response_length for s in samples]
    if resp_lengths:
        log_dict["rollout_agg/response_length_mean"] = float(np.mean(resp_lengths))
        log_dict["rollout_agg/response_length_median"] = float(np.median(resp_lengths))
        log_dict["rollout_agg/response_length_min"] = float(np.min(resp_lengths))
        log_dict["rollout_agg/response_length_max"] = float(np.max(resp_lengths))
        log_dict["rollout_agg/total_response_tokens"] = int(sum(resp_lengths))
    rep_count = sum(1 for s in samples if has_repetition(s.response))
    log_dict["rollout_agg/repetition_frac"] = rep_count / len(samples) if samples else 0.0

    if rollout_time and rollout_time > 0:
        log_dict["perf/rollout_time_seconds"] = rollout_time
        if resp_lengths:
            log_dict["perf/tokens_per_second"] = sum(resp_lengths) / rollout_time

    # --- Console summary ---
    parts = []
    for domain, m in sorted(domain_metrics.items()):
        parts.append(
            f"{domain}: n={m['count']} r={m['reward_mean']:.3f} "
            f"solve={m['solve_rate']:.3f} "
            f"len={m['resp_len_mean']:.0f}({m['resp_len_median']:.0f}) "
            f"trunc={m['truncation_rate']:.2f} rep={m['repetition_frac']:.2f}"
        )
    summary = " | ".join(parts)
    print(f"[Domain Metrics] step {step}: {summary}", flush=True)

    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")

    # --- TSV dump ---
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir:
        tsv_path = os.path.join(os.path.dirname(dump_dir), "domain_metrics.tsv")
        write_header = not os.path.exists(tsv_path)
        with open(tsv_path, "a") as f:
            if write_header:
                f.write(TSV_HEADER)
            for domain, m in sorted(domain_metrics.items()):
                f.write(
                    f"{step}\t{domain}\t{m['count']}\t{m['reward_mean']:.4f}\t"
                    f"{m['reward_min']:.4f}\t{m['reward_max']:.4f}\t"
                    f"{m['solve_rate']:.4f}\t{m['reward_std']:.4f}\t"
                    f"{m['resp_len_mean']:.1f}\t{m['resp_len_median']:.1f}\t"
                    f"{m['resp_len_solved']:.1f}\t{m['resp_len_unsolved']:.1f}\t"
                    f"{m['truncation_rate']:.4f}\t{m['repetition_frac']:.4f}\t"
                    f"{m['reward_frac_zero']:.4f}\t{m['reward_frac_partial']:.4f}\t"
                    f"{m['reward_frac_perfect']:.4f}\n"
                )

    return True
