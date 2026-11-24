"""Tests for prune_checkpoints.parse_scores on sample eval log lines (numpy >= 2 reprs, nan / inf).

    python3 test_prune_checkpoints.py          # or: python3 -m pytest test_prune_checkpoints.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prune_checkpoints as pc  # noqa: E402

METRICS = ("eval/ifeval", "eval/ifbench")
LINES = {
    "eval 10: {'eval/ifeval': 0.5, 'eval/ifbench': 0.25}": 0.375,
    "eval 20: {'eval/ifeval': np.float64(0.6), 'eval/ifbench': np.float32(0.2), 'n': np.int64(541)}": 0.4,
    "eval 30: {'eval/ifeval': np.float64(0.7), 'eval/truncated': np.True_, 'eval/ok': np.False_}": 0.7,
    "eval 40: {'eval/ifeval': np.float64(nan), 'eval/ifbench': np.float64(0.3)}": 0.3,
    "eval 50: {'eval/ifeval': np.float64(inf), 'eval/ifbench': -inf, 'eval/x': nan, 'eval/nan_rate': 0.1}": None,
    "eval 60: {'eval/ifeval': np.float64(0.9), 'eval/per_set': [np.float64(0.1), nan], 'ok': np.bool_(True)}": 0.9,
}


def test_parse_scores():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "train.log"
        log.write_text("\n".join(["noise line"] + list(LINES)) + "\n")
        got = pc.parse_scores([log], METRICS)
    want = {int(line.split()[1].rstrip(":")): v for line, v in LINES.items() if v is not None}
    assert got.keys() == want.keys(), (got, want)
    for k, v in want.items():
        assert abs(got[k] - v) < 1e-6, (k, got[k], v)


if __name__ == "__main__":
    test_parse_scores()
    print("ok")
