"""Tests for retire_solved.py on a fake dataset (no tokenizer, no Ray, no GPU).

    python3 test_retire_solved.py          # or: python3 -m pytest test_retire_solved.py
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retire_solved as rs  # noqa: E402

from miles.rollout import data_source  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

N = 8
STOCK = "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"


class FakeDataset:
    def __init__(self, path, **kwargs):
        self.samples = [Sample(prompt=f"p{i}", metadata={"custom_id": f"id{i}"}) for i in range(int(path))]

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch_id):
        pass


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(data_source, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(data_source, "load_processor", lambda *a, **k: None)
    monkeypatch.setattr(data_source, "Dataset", FakeDataset)
    monkeypatch.setattr(rs, "_INNER_PATH", STOCK)
    monkeypatch.setattr(rs, "_INNER", None)
    monkeypatch.setattr(rs, "RETURN_FRAC", 0.1)
    rs._RETIRED.clear()
    rs._COUNTS.clear()
    yield
    rs._RETIRED.clear()
    rs._COUNTS.clear()


def make_source(tmp_path, n_prompts=100, seed=0):
    args = SimpleNamespace(
        rollout_global_dataset=True, hf_checkpoint=None, chat_template_path=None, dump_details=None,
        prompt_data=str(n_prompts), rollout_max_prompt_len=None, input_key="prompt", multimodal_keys=None,
        label_key="label", metadata_key="metadata", tool_key=None, apply_chat_template=False,
        apply_chat_template_kwargs=None, rollout_seed=seed, rollout_shuffle=False, buffer_filter_path=None,
        n_samples_per_prompt=N, save=str(tmp_path), load=str(tmp_path), reward_key=None,
    )
    return args, rs.RetireSolvedDataSource(args)


def group_with(sample, n_correct, n_truncated=0):
    group = []
    for j in range(N):
        s = Sample(prompt=sample.prompt, metadata=dict(sample.metadata))
        s.reward = 1.0 if j < n_correct else 0.0
        s.status = Sample.Status.TRUNCATED if j >= N - n_truncated else Sample.Status.COMPLETED
        group.append(s)
    return group


def cid(group) -> int:
    return int(group[0].metadata["custom_id"][2:])


def visit_all(args, src, n_prompts=100):
    """One epoch: even ids solved 8/8, odd ids 3/8."""
    for group in src.get_samples(n_prompts):
        i = cid(group)
        out = rs.retire_solved_filter(args, group_with(group[0], N if i % 2 == 0 else 3))
        assert out.keep == (i % 2 == 1), (i, out)


def test_retires_only_all_solved_groups(tmp_path):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    assert rs._RETIRED == {f"id{i}" for i in range(0, 100, 2)}
    assert rs._COUNTS["newly_retired"] == 50
    # A second 8/8 visit does not count again.
    rs.retire_solved_filter(args, group_with(src.dataset.samples[0], N))
    assert rs._COUNTS["newly_retired"] == 50


def test_min_reward_threshold(tmp_path, monkeypatch):
    args, src = make_source(tmp_path)
    group = group_with(src.dataset.samples[0], N)
    group[0].reward = 0.99
    rs.retire_solved_filter(args, group)
    assert not rs._RETIRED
    monkeypatch.setattr(rs, "SOLVED_MIN_REWARD", 0.9)
    rs.retire_solved_filter(args, group)
    assert rs._RETIRED == {"id0"}


def test_reactivates_on_any_failure(tmp_path):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    rs.retire_solved_filter(args, group_with(src.dataset.samples[0], N - 1))
    assert "id0" not in rs._RETIRED
    assert rs._COUNTS["reactivated"] == 1


def test_return_fraction(tmp_path):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    drawn = [cid(src.get_samples(1)[0]) for _ in range(2000)]
    # Retired prompts are half the pool; each is drawn with probability RETURN_FRAC.
    share = sum(i % 2 == 0 for i in drawn) / len(drawn)
    expected = 0.1 / (1 + 0.1)
    assert abs(share - expected) < 0.025, share
    assert rs._COUNTS["returned"] == sum(i % 2 == 0 for i in drawn)
    assert rs._COUNTS["skipped"] > 0


def test_return_frac_zero_and_one(tmp_path, monkeypatch):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    monkeypatch.setattr(rs, "RETURN_FRAC", 0.0)
    assert sorted(cid(g) for g in src.get_samples(50)) == list(range(1, 100, 2))
    monkeypatch.setattr(rs, "RETURN_FRAC", 1.0)
    assert sorted(cid(g) for g in src.get_samples(100)) == list(range(100))


def test_buffer_served_first_without_retirement_check(tmp_path):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    recycled = group_with(src.dataset.samples[0], 0)
    src.add_samples([recycled])
    assert src.get_samples(1)[0] is recycled


def test_all_retired_terminates(tmp_path):
    args, src = make_source(tmp_path, n_prompts=50)
    rs._RETIRED.update(s.metadata["custom_id"] for s in src.dataset.samples)
    out = src.get_samples(50)
    assert len(out) == 50 and len({id(g) for g in out}) == 50
    assert 0 < rs._COUNTS["skipped"] <= 50
    assert rs._COUNTS["returned"] == 50


def test_save_load_roundtrip_with_rng(tmp_path):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    src.get_samples(37)
    src.save(19)
    saved = set(rs._RETIRED)
    expected = [cid(src.get_samples(1)[0]) for _ in range(300)]

    rs._RETIRED.clear()
    _, resumed = make_source(tmp_path, seed=123)
    resumed.load(19)
    assert rs._RETIRED == saved
    assert [cid(resumed.get_samples(1)[0]) for _ in range(300)] == expected


def test_load_without_rng_state_keeps_seeded_rng(tmp_path):
    args, src = make_source(tmp_path)
    os.makedirs(tmp_path / "rollout")
    rs.torch.save({"retired": ["id1", "id2"]}, rs._state_path(str(tmp_path), 5))
    before = src._rng.getstate()
    src.load(5)
    assert rs._RETIRED == {"id1", "id2"}
    assert src._rng.getstate() == before


def test_load_missing_state_is_a_noop(tmp_path):
    args, src = make_source(tmp_path)
    rs._RETIRED.add("id3")
    src.load(-1)
    assert rs._RETIRED == {"id3"}


def test_chains_to_inner_filter(tmp_path, monkeypatch):
    args, src = make_source(tmp_path)
    calls = []
    sentinel = rs.call_dynamic_filter(None)

    def inner(a, samples, **kwargs):
        calls.append((len(samples), kwargs))
        return sentinel

    monkeypatch.setattr(rs, "_INNER", inner)
    assert rs.retire_solved_filter(args, group_with(src.dataset.samples[0], N), extra=1) is sentinel
    assert calls == [(N, {"extra": 1})]
    assert rs._RETIRED == {"id0"}


def test_chains_to_overlong_dynamic_filter(tmp_path, monkeypatch):
    args, src = make_source(tmp_path)
    monkeypatch.setattr(rs, "_INNER_PATH", "overlong_filter.overlong_dynamic_filter")
    monkeypatch.setenv("OVERLONG_INNER_DYNAMIC_FILTER", STOCK)
    p0, p1, p2 = src.dataset.samples[:3]

    # 4 correct + 4 truncated (reward 0): not retired; the survivors are all correct, so dropped.
    out = rs.retire_solved_filter(args, group_with(p0, 4, n_truncated=4))
    assert not out.keep and out.reason == "overlong_zero_std_1.0"
    assert not rs._RETIRED

    # 8/8: retired, and dropped by the inner filter as zero-std.
    out = rs.retire_solved_filter(args, group_with(p1, N))
    assert not out.keep and rs._RETIRED == {"id1"}

    # 3 correct, 3 wrong, 2 truncated: kept on the survivors' spread.
    assert rs.retire_solved_filter(args, group_with(p2, 3, n_truncated=2)).keep


def test_log_rollout_data_drains_counts(tmp_path):
    args, src = make_source(tmp_path)
    visit_all(args, src)
    src.get_samples(50)
    metrics = {}
    assert rs.log_rollout_data(1, args, [], metrics, 0.0) is False
    assert metrics["rollout/retire/n_retired"] == 50
    assert metrics["rollout/retire/newly_retired"] == 50 and metrics["rollout/retire/skipped"] > 0
    metrics = {}
    rs.log_rollout_data(2, args, [], metrics, 0.0)
    assert metrics["rollout/retire/newly_retired"] == 0 and metrics["rollout/retire/skipped"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
