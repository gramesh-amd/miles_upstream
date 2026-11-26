#!/usr/bin/env python3
"""Paired per-problem bootstrap between two cells.

    python3 paired.py qwen3_30b_a3b_thinking_64k ours_r1c_399_64k

Both cells answered the same problems, so resampling problems (not samples) cancels the
question-sampling noise in the difference. Comparing two independent confidence intervals instead
hides real differences behind wide overlapping bands.
"""

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import is_correct  # noqa: E402
from models import OUT_DIR  # noqa: E402


def per_problem(tag: str) -> dict[tuple, float]:
    """Mean strict accuracy per (benchmark, problem) over that problem's samples."""
    hit, tot = defaultdict(int), defaultdict(int)
    with gzip.open(OUT_DIR / f"gen_{tag}.jsonl.gz", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if "error" in d:
                continue
            k = (d["bench"], d["pi"])
            tot[k] += 1
            hit[k] += int(is_correct(d["response"], d["label"]))
    return {k: hit[k] / tot[k] for k in tot}


def main(a: str, b: str, iters: int = 10000) -> None:
    A, B = per_problem(a), per_problem(b)
    keys = sorted(set(A) & set(B))
    if not keys:
        sys.exit("no overlapping problems")
    diff = [A[k] - B[k] for k in keys]
    obs = sum(diff) / len(diff)
    rng = random.Random(0)
    boots = sorted(sum(diff[rng.randrange(len(diff))] for _ in diff) / len(diff) for _ in range(iters))
    lo, hi = boots[int(0.025 * iters)], boots[int(0.975 * iters)]
    wins, ties = sum(d > 0 for d in diff), sum(d == 0 for d in diff)
    print(f"{a}  vs  {b}")
    print(f"  problems compared : {len(keys)}")
    print(f"  mean paired diff  : {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  per-problem record: {wins} win / {len(keys) - wins - ties} loss / {ties} tie")
    print(f"  verdict           : {'SIGNIFICANT' if lo > 0 or hi < 0 else 'not distinguishable'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
