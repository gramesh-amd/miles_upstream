"""Batch assembly in examples/fully_async/fully_async_rollout.py with a fake worker (no engines, no Ray)."""

import asyncio
import importlib.util
import queue
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
