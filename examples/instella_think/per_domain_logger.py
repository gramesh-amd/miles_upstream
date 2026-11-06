"""Per-domain rollout metrics, logged alongside miles' own (Olmo 3 style tracking).

Groups each training step's samples by metadata["dataset_source"] (falling back to
metadata["rm_type"], then "all") and logs, under domain/<source>/:
  count, reward_mean, reward_std
  solve_rate        reward >= SOLVED_REWARD, i.e. fully solved. With partial credit (code
                    pass rate, IF constraints) reward > 0 would count one passing test.
  nonzero_rate      reward > 0, which for partial credit is "made any progress"
  resp_len_mean, resp_len_solved, resp_len_unsolved
  truncation_rate   from the engine's finish status. A length test does not work: sglang
                    clamps max_new_tokens to context - prompt, so truncated samples end a
                    prompt-length short of rollout_max_response_len.

Why bother: the Dolci code set mixes sources whose difficulty differs a lot, and the
aggregate rollout/raw_reward cannot show one source saturating while another stalls.

Returns False so miles' default rollout logging still runs; this only adds keys.

    --custom-rollout-log-function-path per_domain_logger.log_per_domain_metrics
"""

import logging
import os
from collections import defaultdict

import numpy as np

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

SOLVED_REWARD = float(os.environ.get("SOLVED_REWARD", "1.0"))
_EPS = 1e-6


def _domain(sample: Sample) -> str:
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    name = meta.get("dataset_source") or meta.get("rm_type") or "all"
    return str(name).replace("/", "_")


def _reward(args, sample: Sample) -> float | None:
    if sample.reward is None:
        return None
    try:
        return float(sample.get_reward_value(args))
    except (TypeError, ValueError, KeyError):
        return None


def _length(sample: Sample) -> int:
    return int(sample.effective_response_length or sample.response_length or 0)


def _domain_stats(args, samples: list[Sample]) -> dict[str, float]:
    rows = [
        (r, _length(s), s.status == Sample.Status.TRUNCATED) for s in samples if (r := _reward(args, s)) is not None
    ]
    if not rows:
        return {}
    rewards = np.array([r for r, _, _ in rows])
    lengths = np.array([n for _, n, _ in rows])
    solved = rewards >= SOLVED_REWARD - _EPS
    return {
        "count": len(rows),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "solve_rate": float(solved.mean()),
        "nonzero_rate": float((rewards > 0).mean()),
        "resp_len_mean": float(lengths.mean()),
        "resp_len_solved": float(lengths[solved].mean()) if solved.any() else 0.0,
        "resp_len_unsolved": float(lengths[~solved].mean()) if (~solved).any() else 0.0,
        "truncation_rate": float(np.mean([t for _, _, t in rows])),
    }


def log_per_domain_metrics(rollout_id, args, samples: list[Sample], rollout_extra_metrics, rollout_time) -> bool:
    try:
        by_domain: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_domain[_domain(s)].append(s)

        log_dict: dict[str, float] = {}
        parts = []
        for domain, group in sorted(by_domain.items()):
            m = _domain_stats(args, group)
            if not m:
                continue
            log_dict |= {f"domain/{domain}/{k}": v for k, v in m.items()}
            parts.append(
                f"{domain}: n={m['count']} solve={m['solve_rate']:.3f} "
                f"len={m['resp_len_mean']:.0f} trunc={m['truncation_rate']:.2f}"
            )
        if log_dict:
            step = compute_rollout_step(args, rollout_id)
            logger.info(f"[domain metrics] step {step}: {' | '.join(parts)}")
            log_dict["rollout/step"] = step
            tracking.log(args, log_dict, step_key="rollout/step")
    except Exception:
        # Observability only: never take the rollout down over a metric.
        logger.exception("per_domain_logger failed; skipping domain metrics this step")
    return False
