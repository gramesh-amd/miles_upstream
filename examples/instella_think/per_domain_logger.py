"""Per-domain rollout metrics, logged alongside miles' own (Olmo 3 style tracking).

Every training sample is put in one domain per grouping key and the stats of each domain are
logged under rollout/domain/<key>/<domain>/ (rollout/* is bound to the rollout/step axis in W&B).
The keys come from PER_DOMAIN_KEYS (comma-separated, default "route,src,src_route"):
  route      metadata["route"]   e.g. function / stdio (the code training data)
  src        metadata["src"]     e.g. acecoder / klear / nemotron / synthetic2
  src_route  "<src>_<route>"
  source     metadata["dataset_source"], else metadata["rm_type"] (the old single grouping)
A sample without the field goes to domain "other"; a key no sample has is not logged.

Stats per domain:
  count             samples generated and graded since the previous log
  reward_mean       training reward (binary or pass rate, as trained)
  solve_rate        reward >= SOLVED_REWARD, i.e. fully solved
  nonzero_rate      reward > 0
  pass_rate_mean    mean fraction of tests passed (code rows; 0 when the think gate or code
                    extraction rejected the answer), informative under the binary reward
  resp_len_mean, resp_len_solved, resp_len_unsolved
  truncation_rate   from the engine's finish status. A length test does not work: sglang
                    clamps max_new_tokens to context - prompt, so truncated samples end a
                    prompt-length short of rollout_max_response_len.
  group_all_solved, group_none_solved
                    share of prompts (complete groups) whose samples all / none solved: the
                    groups that carry no gradient under GRPO and that the dynamic filter drops
  trained_count     samples of the domain in the batch handed to the trainer

Scope. The trainer batch (the hook's `samples`) is what is left after the dynamic filter, which
under the binary reward drops every all-pass and all-fail group, so its solve rate is biased.
The stats therefore come from every graded training sample: the reward function calls note()
(code_reward.batched_reward does) and the hook drains that window. With a reward function that
does not call note(), the stats fall back to the trainer batch.

Returns False so miles' default rollout logging still runs; this only adds keys.

    --custom-rollout-log-function-path per_domain_logger.log_per_domain_metrics
"""

import logging
import os
import threading
from collections import defaultdict

import numpy as np

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

SOLVED_REWARD = float(os.environ.get("SOLVED_REWARD", "1.0"))
KEYS = [k.strip() for k in os.environ.get("PER_DOMAIN_KEYS", "route,src,src_route").split(",") if k.strip()]
_WINDOW_MAX = int(os.environ.get("PER_DOMAIN_WINDOW_MAX", "200000"))
_EPS = 1e-6


def _meta(sample) -> dict:
    meta = getattr(sample, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _domain(meta: dict, key: str) -> str | None:
    if key == "src_route":
        src, route = meta.get("src"), meta.get("route")
        name = f"{src}_{route}" if src is not None and route is not None else None
    elif key == "source":
        name = meta.get("dataset_source") or meta.get("rm_type")
    else:
        name = meta.get(key)
    return None if name is None else str(name).replace("/", "_")


def _record(sample, reward) -> dict | None:
    """What the stats need from one graded sample, taken right away (the Sample object lives on)."""
    if reward is None:
        return None
    try:
        reward = float(reward)
    except (TypeError, ValueError):
        return None
    meta = _meta(sample)
    rate = meta.get("code_pass_rate")
    if not isinstance(rate, (int, float)):
        rate = 0.0 if "code_pass_rate" not in meta and meta.get("rm_type") == "code" else None
    return {
        "domains": {k: _domain(meta, k) for k in KEYS},
        "reward": reward,
        "pass_rate": None if rate is None else float(rate),
        "length": int(getattr(sample, "effective_response_length", 0) or getattr(sample, "response_length", 0) or 0),
        "truncated": getattr(sample, "status", None) == Sample.Status.TRUNCATED,
        "group": getattr(sample, "group_index", None),
    }


class _Window:
    """Records of the graded training samples since the last drain (thread-safe, bounded)."""

    def __init__(self, cap: int):
        self._cap, self._lock, self._items, self.dropped = cap, threading.Lock(), [], 0

    def add(self, rec: dict) -> None:
        with self._lock:
            if len(self._items) >= self._cap:
                self._items.pop(0)
                self.dropped += 1
            self._items.append(rec)

    def drain(self) -> list[dict]:
        with self._lock:
            items, self._items = self._items, []
            return items


WINDOW = _Window(_WINDOW_MAX)


def note(samples, rewards) -> None:
    """Called by the reward function with the training samples it just graded and their rewards
    (unscaled or scaled alike, as long as SOLVED_REWARD matches). Never raises."""
    try:
        for s, r in zip(samples, rewards, strict=False):
            if (rec := _record(s, r)) is not None:
                WINDOW.add(rec)
    except Exception:
        logger.exception("per_domain_logger.note failed")


def _stats(recs: list[dict], n_per_prompt: int) -> dict[str, float]:
    rewards = np.array([r["reward"] for r in recs])
    lengths = np.array([r["length"] for r in recs])
    solved = rewards >= SOLVED_REWARD - _EPS
    out = {
        "count": len(recs),
        "reward_mean": float(rewards.mean()),
        "solve_rate": float(solved.mean()),
        "nonzero_rate": float((rewards > 0).mean()),
        "resp_len_mean": float(lengths.mean()),
        "resp_len_solved": float(lengths[solved].mean()) if solved.any() else 0.0,
        "resp_len_unsolved": float(lengths[~solved].mean()) if (~solved).any() else 0.0,
        "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
    }
    rates = [r["pass_rate"] for r in recs if r["pass_rate"] is not None]
    if rates:
        out["pass_rate_mean"] = float(np.mean(rates))
    groups = defaultdict(list)
    for r, ok in zip(recs, solved, strict=True):
        if r["group"] is not None:
            groups[r["group"]].append(bool(ok))
    complete = [g for g in groups.values() if len(g) >= n_per_prompt]
    if complete:
        out["group_all_solved"] = float(np.mean([all(g) for g in complete]))
        out["group_none_solved"] = float(np.mean([not any(g) for g in complete]))
    return out


def _flatten(samples) -> list:
    out = []
    for item in samples or []:
        out.extend(item if isinstance(item, (list, tuple)) else [item])
    return out


def log_per_domain_metrics(rollout_id, args, samples: list[Sample], rollout_extra_metrics, rollout_time) -> bool:
    try:
        trained = _flatten(samples)
        recs = WINDOW.drain()
        if not recs:  # the reward function does not call note(): use the trainer batch
            recs = [rec for s in trained if (rec := _record(s, _reward(args, s))) is not None]
        n_per_prompt = int(getattr(args, "n_samples_per_prompt", 1) or 1)

        log_dict: dict[str, float] = {}
        parts = []
        for key in KEYS:
            by_domain: dict[str, list[dict]] = defaultdict(list)
            for r in recs:
                if (d := r["domains"].get(key)) is not None:
                    by_domain[d].append(r)
            if not by_domain:
                continue
            trained_counts = defaultdict(int)
            for s in trained:
                trained_counts[_domain(_meta(s), key) or "other"] += 1
            for domain in sorted(set(by_domain) | set(trained_counts)):
                m = _stats(by_domain[domain], n_per_prompt) if by_domain.get(domain) else {"count": 0}
                m["trained_count"] = trained_counts.get(domain, 0)
                log_dict |= {f"rollout/domain/{key}/{domain}/{k}": v for k, v in m.items()}
                if key == KEYS[0] and m["count"]:
                    parts.append(
                        f"{domain}: n={m['count']} solve={m['solve_rate']:.3f} "
                        f"pass={m.get('pass_rate_mean', float('nan')):.3f} trunc={m['truncation_rate']:.2f}"
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


def _reward(args, sample: Sample) -> float | None:
    if sample.reward is None:
        return None
    try:
        return float(sample.get_reward_value(args))
    except (TypeError, ValueError, KeyError):
        return None
