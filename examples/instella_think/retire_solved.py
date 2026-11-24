"""Retire prompts the policy solved on every sample of their previous visit, and bring them back at
RETIRE_RETURN_FRAC of draws (AceReason / Skywork-OR1 epoch-wise removal of solved prompts; MiMo
re-samples ~10% easy data so they are not forgotten).

WHY. A group scored 8/8 has zero advantage and is dropped by the dynamic filter, but only after its
8 rollouts were generated and graded. In one code run about 30% of draws were AceCoder 8/8 groups. Skipping those prompts when they are DRAWN moves the compute to prompts that still teach.

Three pieces, all in the RolloutManager process (it builds the data source and runs the fully-async
rollout), sharing this module's state:
    --dynamic-sampling-filter-path retire_solved.retire_solved_filter
        records each group's outcome (all >= RETIRE_SOLVED_MIN_REWARD -> retired, anything else ->
        active again), then defers to RETIRE_INNER_DYNAMIC_FILTER (default: the stock
        check_reward_nonzero_std) for keep/drop.
    --data-source-path retire_solved.RetireSolvedDataSource
        skips retired prompts at draw time, except a RETIRE_RETURN_FRAC share of them.
    ROLLOUT_LOG_HOOKS += retire_solved.log_rollout_data
        rollout/retire/{n_retired, skipped, returned, newly_retired, reactivated}.
run_async.sh wires all three under RETIRE_SOLVED=1.

The worker thread draws prompts while the rollout loop runs the filter, hence _LOCK. The retired set
and the RNG of the return draws are saved next to miles' dataset state at every checkpoint
(rollout/retired_solved_<rollout_id>.pt) and restored on resume. Prompts are keyed by
metadata["custom_id"] (falling back to the prompt text).
"""

import collections
import hashlib
import json
import logging
import os
import random
import threading

import torch

from miles.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.utils.misc import load_function

logger = logging.getLogger(__name__)

RETURN_FRAC = float(os.environ.get("RETIRE_RETURN_FRAC", "0.1"))
SOLVED_MIN_REWARD = float(os.environ.get("RETIRE_SOLVED_MIN_REWARD", "0.999"))
_INNER_PATH = os.environ.get("RETIRE_INNER_DYNAMIC_FILTER") or (
    "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
)

_LOCK = threading.Lock()
_RETIRED: set[str] = set()
_COUNTS = collections.Counter()  # since the previous rollout log
_INNER = None


def _flatten(samples):
    for s in samples:
        if isinstance(s, list):
            yield from _flatten(s)
        else:
            yield s


def _key(sample) -> str:
    meta = getattr(sample, "metadata", None)
    if isinstance(meta, dict) and meta.get("custom_id") is not None:
        return str(meta["custom_id"])
    prompt = getattr(sample, "prompt", "")
    text = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True, default=str)
    return "h:" + hashlib.sha1(text.encode()).hexdigest()


def retire_solved_filter(args, samples, **kwargs):
    global _INNER
    if _INNER is None:
        _INNER = load_function(_INNER_PATH)
    group = list(_flatten(samples))
    if group:
        rewards = [s.get_reward_value(args) for s in group]
        key = _key(group[0])
        solved = all(r is not None and r >= SOLVED_MIN_REWARD for r in rewards)
        with _LOCK:
            if solved and key not in _RETIRED:
                _RETIRED.add(key)
                _COUNTS["newly_retired"] += 1
            elif not solved and key in _RETIRED:
                _RETIRED.discard(key)
                _COUNTS["reactivated"] += 1
    return call_dynamic_filter(_INNER, args, samples, **kwargs)


def _state_path(directory: str, rollout_id) -> str:
    return os.path.join(directory, f"rollout/retired_solved_{rollout_id}.pt")


class RetireSolvedDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        self._rng = random.Random(getattr(args, "rollout_seed", 0) or 0)

    def get_samples(self, num_samples: int):
        out = self._get_samples_from_buffer(num_samples)
        need = num_samples - len(out)
        # At most one pass over the dataset is skipped per call: once nearly everything is retired,
        # retired prompts are served instead of scanning forever.
        budget = len(self.dataset) if self.dataset is not None else 0
        skipped = returned = 0
        while need > 0:
            for group in RolloutDataSource.get_samples(self, need):
                key = _key(group[0])
                with _LOCK:
                    retired = key in _RETIRED
                    skip = retired and skipped < budget and self._rng.random() >= RETURN_FRAC
                if skip:
                    skipped += 1
                    continue
                out.append(group)
                need -= 1
                returned += retired
        if skipped or returned:
            with _LOCK:
                _COUNTS["skipped"] += skipped
                _COUNTS["returned"] += returned
        return out

    def save(self, rollout_id):
        super().save(rollout_id)
        if not self.args.rollout_global_dataset:
            return
        path = _state_path(self.args.save, rollout_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOCK:
            torch.save({"retired": sorted(_RETIRED), "rng_state": self._rng.getstate()}, path)

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if not self.args.rollout_global_dataset or self.args.load is None:
            return
        path = _state_path(self.args.load, rollout_id)
        if not os.path.exists(path):
            logger.warning(f"no retired-prompt state under {path}: starting with none retired")
            return
        state = torch.load(path)
        with _LOCK:
            _RETIRED.clear()
            _RETIRED.update(state.get("retired", []))
            if state.get("rng_state") is not None:
                self._rng.setstate(state["rng_state"])
        logger.info(f"restored {len(_RETIRED)} retired prompts from {path}")


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics=None, rollout_time=None) -> bool:
    """Adds rollout/retire/* to miles' own rollout log (rollout_extra_metrics), so no extra W&B call."""
    with _LOCK:
        counts = dict(_COUNTS)
        _COUNTS.clear()
        n_retired = len(_RETIRED)
    metrics = {"rollout/retire/n_retired": n_retired}
    metrics |= {
        f"rollout/retire/{k}": counts.get(k, 0) for k in ("skipped", "returned", "newly_retired", "reactivated")
    }
    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(metrics)
    else:
        try:
            from miles.utils.metric_utils import compute_rollout_step
            from miles.utils.tracking_utils import tracking

            tracking.log(
                args, metrics | {"rollout/step": compute_rollout_step(args, rollout_id)}, step_key="rollout/step"
            )
        except Exception as exc:  # never let a metrics add-on break the run
            logger.warning("retire_solved logging failed (%s)", exc)
    logger.info("retire_solved %s: n_retired=%d %s", rollout_id, n_retired, counts)
    return False
