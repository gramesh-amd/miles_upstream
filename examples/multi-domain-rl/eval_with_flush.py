"""Custom eval/rollout function that flushes SGLang engine caches between eval datasets.

Workaround for triton attention memory corruption bug on ROCm that causes
'Memory access fault' crashes during sustained inference. The corruption is
cumulative -- by flushing the KV cache between eval datasets, each dataset
starts with a clean engine state.

Usage: --eval-function-path examples.multi-domain-rl.eval_with_flush.generate_rollout
"""

import asyncio
import logging
from argparse import Namespace
from typing import Any

import httpx
import sglang_router
from packaging.version import parse

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.sglang_rollout import eval_rollout_single_dataset
from miles.rollout.sglang_rollout import generate_rollout as _upstream_generate_rollout
from miles.utils.async_utils import run
from miles.utils.http_utils import get

logger = logging.getLogger(__name__)


async def _get_engine_urls(args: Namespace) -> list[str]:
    """Get individual SGLang engine URLs from the router."""
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers"
        )
        return response["urls"]
    else:
        response = await get(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"
        )
        return [worker["url"] for worker in response["workers"]]


async def _flush_all_engines(args: Namespace) -> bool:
    """Flush KV cache on all SGLang engines to reset memory state.

    Returns True on success, False if any engine failed to flush.
    """
    try:
        urls = await _get_engine_urls(args)
    except Exception as e:
        logger.warning(f"Could not get engine URLs for flush: {e}")
        return False

    logger.info(f"Flushing KV cache on {len(urls)} engines: {urls}")
    all_ok = True

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for url in urls:
            flushed = False
            for attempt in range(10):
                try:
                    resp = await client.get(f"{url}/flush_cache")
                    if resp.status_code == 200:
                        flushed = True
                        break
                    logger.info(f"flush_cache {url}: status {resp.status_code}, retrying...")
                except Exception as e:
                    logger.warning(f"flush_cache attempt {attempt + 1} for {url}: {e}")
                await asyncio.sleep(1)
            if not flushed:
                logger.error(f"Failed to flush cache for {url} after 10 attempts")
                all_ok = False

    if all_ok:
        logger.info("All engines flushed successfully")
    return all_ok


async def _eval_with_flush(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    """Evaluate datasets sequentially with cache flush between each."""
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    datasets = getattr(args, "eval_datasets", []) or []
    results: dict[str, dict[str, list[Any]]] = {}

    for i, dataset_cfg in enumerate(datasets):
        logger.info(f"Eval dataset {i + 1}/{len(datasets)}: {dataset_cfg.name}")
        r = await eval_rollout_single_dataset(args, rollout_id, dataset_cfg)
        results.update(r)

        if i < len(datasets) - 1:
            try:
                await _flush_all_engines(args)
            except Exception as e:
                logger.warning(f"flush_cache between datasets failed: {e}; continuing")

    return RolloutFnEvalOutput(data=results)


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Drop-in replacement for miles.rollout.sglang_rollout.generate_rollout.

    For training (evaluation=False): delegates to the upstream generate_rollout.
    For evaluation (evaluation=True): runs datasets sequentially with KV cache
    flush between each to prevent cumulative triton memory corruption.
    """
    if not evaluation:
        return _upstream_generate_rollout(args, rollout_id, data_source, evaluation=False)

    return run(_eval_with_flush(args, rollout_id))
