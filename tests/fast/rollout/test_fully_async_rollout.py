"""Batch assembly in examples/fully_async/fully_async_rollout.py with a fake worker (no engines, no Ray)."""

import asyncio
import importlib.util
import queue
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.types import Sample

N = 2


def load_rollout_module():
    module_path = Path(__file__).resolve().parents[3] / "examples" / "fully_async" / "fully_async_rollout.py"
    spec = importlib.util.spec_from_file_location("test_fully_async_rollout_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeWorker:
    def __init__(self):
        self.output_queue = queue.Queue()
        self.pending_groups = deque()

    def complete(self, *groups):
        for group in groups:
            self.output_queue.put((group[0].index, group))

    def get_completed_groups(self):
        out = []
        while not self.output_queue.empty():
            out.append(self.output_queue.get_nowait())
        return out

    def get_queue_size(self):
        return self.output_queue.qsize()


class FakeBuffer:
    def __init__(self):
        self.added = []

    def add_samples(self, groups):
        self.added.extend(groups)


def make_group(index, version=0, status=Sample.Status.COMPLETED):
    return [
        Sample(
            index=index,
            prompt=f"p{index}",
            response="r",
            reward=float(j),
            status=status,
            weight_versions=[str(version)],
        )
        for j in range(N)
    ]


@pytest.fixture
def rollout(monkeypatch):
    module = load_rollout_module()
    worker = FakeWorker()
    monkeypatch.setattr(module, "get_global_worker", lambda args, data_buffer: worker)
    return module, worker


def make_args(**kwargs):
    defaults = dict(
        rollout_global_dataset=True, rollout_batch_size=2, max_weight_staleness=None, dynamic_sampling_filter_path=None
    )
    return SimpleNamespace(**{**defaults, **kwargs})


def step(module, args, buffer, rollout_id=0):
    return asyncio.run(asyncio.wait_for(module.generate_rollout_async(args, rollout_id, buffer), timeout=10))


def indices(data):
    return [group[0].index for group in data]


def test_groups_beyond_the_batch_carry_over_in_completion_order(rollout):
    module, worker = rollout
    args, buffer = make_args(), FakeBuffer()
    trained, carried = [], []

    worker.complete(*(make_group(i) for i in (4, 0, 3, 1, 2)))
    for rollout_id, new in enumerate([(), (5, 6), (), (7,)]):
        worker.complete(*(make_group(i) for i in new))
        data, metrics = step(module, args, buffer, rollout_id)
        trained.append(indices(data))
        carried.append(metrics.get("rollout/async/carried_over_groups"))

    # Batches are sorted by index; the leftovers of each step lead the next one.
    assert trained == [[0, 4], [1, 3], [2, 5], [6, 7]]
    assert carried == [3, 3, 1, 0]
    assert not worker.pending_groups and not buffer.added


def test_carried_over_groups_are_checked_for_staleness_when_consumed(rollout, monkeypatch):
    module, worker = rollout
    args, buffer = make_args(max_weight_staleness=1), FakeBuffer()
    engine_version = {"v": 0}

    async def current_version(_args):
        return engine_version["v"]

    monkeypatch.setattr(module._cached_version, "get", current_version)
    worker.complete(*(make_group(i, version=0) for i in range(3)))
    data, _ = step(module, args, buffer, 0)
    assert indices(data) == [0, 1]

    engine_version["v"] = 2
    worker.complete(make_group(3, version=2), make_group(4, version=2))
    data, metrics = step(module, args, buffer, 1)
    assert indices(data) == [3, 4]
    assert metrics["rollout/async/recycled_stale_groups"] == 1
    assert [g[0].index for g in buffer.added] == [2] and buffer.added[0][0].reward is None


def test_aborted_leftover_is_recycled_once_and_not_trained(rollout):
    module, worker = rollout
    args, buffer = make_args(), FakeBuffer()
    worker.complete(make_group(0), make_group(1), make_group(2, status=Sample.Status.ABORTED))
    data, _ = step(module, args, buffer, 0)
    assert indices(data) == [0, 1] and not buffer.added

    worker.complete(make_group(3), make_group(4))
    data, metrics = step(module, args, buffer, 1)
    assert indices(data) == [3, 4]
    assert metrics["rollout/async/aborted_groups"] == 1
    assert [g[0].index for g in buffer.added] == [2]
    assert not worker.pending_groups


class FakeDataSource(FakeBuffer):
    """Serves recycled groups first, then fresh prompts with increasing indices."""

    def __init__(self):
        super().__init__()
        self.next_index = 0

    def get_samples(self, num_samples):
        out = []
        for _ in range(num_samples):
            if self.added:
                out.append(self.added.pop(0))
            else:
                out.append(make_group(self.next_index))
                self.next_index += 1
        return out


def test_backlog_stays_bounded_when_production_outruns_consumption(monkeypatch):
    module = load_rollout_module()
    engine_version = {"v": 0}
    attempts = []

    async def generate(args, group, sampling_params, evaluation):
        attempts.append(id(group))
        for s in group:
            s.weight_versions = [str(engine_version["v"])]
        return group

    async def current_version(_args):
        return engine_version["v"]

    monkeypatch.setattr(module, "generate_and_rm_group", generate)
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    monkeypatch.setattr(module._cached_version, "get", current_version)

    # 6 groups in flight per poll against a consumer that takes 2 per step.
    args = make_args(
        max_weight_staleness=4,
        async_max_concurrent_samples=6 * N,
        n_samples_per_prompt=N,
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        async_data_buffer_capacity_factor=2.0,
    )
    source = FakeDataSource()
    worker = module.AsyncRolloutWorker(args, source)
    worker.poll_interval_s = 0.002
    monkeypatch.setattr(module, "get_global_worker", lambda args, data_buffer: worker)
    worker.start()

    max_concurrent = args.async_max_concurrent_samples // N
    trained, max_backlog, staleness = [], 0, []
    try:
        for rollout_id in range(40):
            time.sleep(0.02)
            max_backlog = max(max_backlog, worker.buffered_groups())
            data, metrics = step(module, args, source, rollout_id)
            staleness += [engine_version["v"] - group[0].oldest_weight_version for group in data]
            trained += data
            engine_version["v"] += 1
    finally:
        worker.stop()

    assert worker.buffer_capacity == 4
    assert max_backlog <= worker.buffer_capacity + max_concurrent
    assert max(staleness) <= args.max_weight_staleness
    assert len({group[0].index for group in trained}) == len(trained)
    # Every generated group is trained, recycled to the source, or still buffered; none is lost.
    leftover = [g for _, g in worker.pending_groups] + [g for _, g in worker.get_completed_groups()]
    assert len(attempts) == len(trained) + len(source.added) + len(leftover)
    assert {id(g) for g in trained}.isdisjoint(id(g) for g in leftover)


def test_frac_mastered_counts_prefixed_reasons(rollout, monkeypatch):
    from miles.rollout.filter_hub.base_types import DynamicFilterOutput

    module, worker = rollout
    reasons = {0: "overlong_mastered_7of8", 1: "mastered_8of8", 2: "zero_std_0.0"}

    def dynamic_filter(args, group, **kwargs):
        reason = reasons.get(group[0].index)
        return DynamicFilterOutput(keep=reason is None, reason=reason)

    monkeypatch.setattr(module, "load_function", lambda path: dynamic_filter)
    args, buffer = make_args(dynamic_sampling_filter_path="f"), FakeBuffer()
    worker.complete(*(make_group(i) for i in range(5)))
    data, metrics = step(module, args, buffer)
    assert indices(data) == [3, 4]
    assert metrics["rollout/dynamic_filter/frac_mastered"] == 2 / 5
