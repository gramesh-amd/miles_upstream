"""Runs several rollout log hooks, since miles takes one --custom-rollout-log-function-path.

run_async.sh uses it when more than one hook is on (per_domain_logger.py with PER_DOMAIN_METRICS=1
and CUSTOM_ROLLOUT_LOG_PATH, e.g. run_code.sh's code_reward.log_rollout_data); passing the flag
twice would keep only the last one. ROLLOUT_LOG_HOOKS is the comma-separated list, reaching the
rollout workers through the runtime env.

    --custom-rollout-log-function-path rollout_log_chain.log_rollout_chain

Every hook runs, in order, even if an earlier one raised. Returns True (skip miles' default
logging) only if some hook did.
"""

import logging
import os

from miles.utils.misc import load_function

logger = logging.getLogger(__name__)

_HOOKS = None


def _hooks():
    global _HOOKS
    if _HOOKS is None:
        paths = [p.strip() for p in os.environ.get("ROLLOUT_LOG_HOOKS", "").split(",") if p.strip()]
        _HOOKS = [(p, load_function(p)) for p in paths]
        logger.info("rollout log hooks: %s", paths)
    return _HOOKS


def log_rollout_chain(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    skip = False
    for path, fn in _hooks():
        try:
            skip |= bool(fn(rollout_id, args, samples, rollout_extra_metrics, rollout_time))
        except Exception:
            logger.exception("rollout log hook %s failed", path)
    return skip
