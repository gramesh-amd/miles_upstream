import asyncio
import atexit
import logging
import queue
import threading
import time

import aiohttp

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.data_source import DataSource
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def group_oldest_weight_version(group: list[Sample]) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [s.oldest_weight_version for s in group if s.oldest_weight_version is not None]
    return min(versions) if versions else None


class _CachedWeightVersion:
    """Throttled query for the current engine weight version.

    The endpoint that reports ``weight_version`` has drifted across SGLang
    versions and differs between the router and the engines. On this build the
    router serves it at ``/get_model_info`` (``/model_info`` is 404 there),
    while some builds/engines use ``/model_info`` or ``/get_weight_version``.
    Try them in order and use the first that returns a numeric version. If none
    resolve, warn ONCE and loudly -- a silently-None version disables the whole
    staleness filter (nothing is ever recycled, so max_weight_staleness becomes
    a no-op and rejected_ratio is a meaningless 0).
    """

    _ENDPOINTS = ("/get_model_info", "/model_info", "/get_weight_version")

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query: float = 0.0
        self._warned = False

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        for endpoint in self._ENDPOINTS:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(base + endpoint, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
            except Exception as e:
                logger.debug(f"Weight-version query to {endpoint} failed: {e}")
                continue
            version = data.get("weight_version") if isinstance(data, dict) else None
            if version is not None and str(version).isdigit():
                self._value = int(version)
                self._last_query = now
                return self._value
        if not self._warned:
            self._warned = True
            logger.warning(
                "Could not read weight_version from any of "
                f"{self._ENDPOINTS} at {base}; the staleness filter is DISABLED "
                "(no groups will be recycled and --max-weight-staleness is a no-op)."
            )
        return self._value


_cached_version = _CachedWeightVersion()


# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer: DataSource):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer: DataSource):
        if args.async_max_concurrent_samples is not None:
            client_capacity = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )
            if args.async_max_concurrent_samples > client_capacity:
                print(
                    f"--async-max-concurrent-samples ({args.async_max_concurrent_samples}) exceeds the "
                    f"client concurrency cap ({client_capacity}); the excess queues on the semaphore"
                )
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        active_tasks = set()
        if self.args.async_max_concurrent_samples is not None:
            max_concurrent_tasks = max(1, self.args.async_max_concurrent_samples // self.args.n_samples_per_prompt)
        else:
            max_concurrent_tasks = self.args.rollout_batch_size
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                # If active task count hasn't reached limit, try to get new data and start tasks
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                result = done_task.result()
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer: DataSource) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True
    stale_groups_recycled = 0
    aborted_groups_returned = 0
    dynamic_filter_dropped = 0
    staleness_values = []

    use_staleness_filter = getattr(args, "max_weight_staleness", None) is not None

    # DAPO dynamic sampling: drop groups the filter rejects (e.g. zero-std groups
    # where every trajectory is all-correct or all-wrong, which contribute a ~0
    # advantage / no gradient). Dropped groups are NOT recycled -- the continuously
    # running worker just keeps pulling fresh prompts, which is the async
    # equivalent of DAPO's over-sampling. No-op unless the flag is set.
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )
    if dynamic_filter is not None:
        print(f"Dynamic sampling filter enabled: {args.dynamic_sampling_filter_path}")

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")
    if use_staleness_filter:
        print(f"Staleness filter enabled: max_weight_staleness={args.max_weight_staleness}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Query current engine version once per collection batch (cached/throttled)
        current_engine_version = None
        if use_staleness_filter:
            current_engine_version = await _cached_version.get(args)

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    for s in group:
                        s.reset_for_retry()
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                aborted_groups_returned += 1
                # don't count as processed for training
                continue

            # Staleness filter: discard groups whose oldest weight version is too far behind
            oldest = group_oldest_weight_version(group)
            if oldest is not None and current_engine_version is not None:
                staleness = current_engine_version - oldest
                staleness_values.append(staleness)
                if staleness > args.max_weight_staleness:
                    try:
                        for s in group:
                            s.reset_for_retry()
                        data_buffer.add_samples([group])
                    except Exception as e:
                        logger.warning(f"Failed to recycle stale group {group_id}: {e}")
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group {group_id} "
                        f"(oldest_version={oldest}, current={current_engine_version}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                    # don't count as processed for training
                    continue

            # DAPO dynamic sampling filter (e.g. reject zero-std groups). Unlike
            # stale/aborted groups these are discarded, not recycled: the prompt
            # produced a degenerate all-same-reward group, so we drop it and let
            # the worker over-sample fresh prompts instead.
            if dynamic_filter is not None:
                filter_output = call_dynamic_filter(dynamic_filter, args, group)
                if not filter_output.keep:
                    dynamic_filter_dropped += 1
                    logger.info(f"Dropped group {group_id} by dynamic filter (reason={filter_output.reason})")
                    continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
    if stale_groups_recycled > 0 or staleness_values:
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
        print(
            f"Staleness stats: recycled={stale_groups_recycled}, "
            f"avg_staleness={avg_staleness:.1f}, "
            f"max_staleness={max(staleness_values) if staleness_values else 0}"
        )

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    # Surface queue-rejection stats to wandb. These ride the rollout/step axis
    # via log_rollout_data's rollout_extra_metrics channel. "rejected" = groups
    # pulled from the completed queue but NOT forwarded to training: recycled
    # (too stale) + aborted. Ratio is over all groups considered this step.
    accepted_groups = len(data)
    considered = accepted_groups + stale_groups_recycled + aborted_groups_returned + dynamic_filter_dropped
    metrics = {
        "rollout/async/accepted_groups": accepted_groups,
        "rollout/async/recycled_stale_groups": stale_groups_recycled,
        "rollout/async/aborted_groups": aborted_groups_returned,
        "rollout/async/dynamic_filter_dropped": dynamic_filter_dropped,
        "rollout/async/rejected_ratio": ((stale_groups_recycled + aborted_groups_returned) / considered)
        if considered > 0
        else 0.0,
        "rollout/async/stale_recycled_ratio": (stale_groups_recycled / considered) if considered > 0 else 0.0,
        "rollout/async/dynamic_filter_dropped_ratio": (dynamic_filter_dropped / considered) if considered > 0 else 0.0,
        "perf/async/queue_size_end": worker.get_queue_size(),
        "perf/async/rollout_duration_s": duration,
    }
    if staleness_values:
        metrics["rollout/async/avg_staleness"] = sum(staleness_values) / len(staleness_values)
        metrics["rollout/async/max_staleness"] = max(staleness_values)

    data = sorted(data, key=lambda group: group[0].index)
    return data, metrics


def generate_rollout_fully_async(args, rollout_id, data_buffer: DataSource, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples, metrics = run(generate_rollout_async(args, rollout_id, data_buffer))
    return RolloutFnTrainOutput(samples=completed_samples, metrics=metrics)


# Register exit cleanup function

atexit.register(stop_global_worker)
