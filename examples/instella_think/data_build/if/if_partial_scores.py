#!/usr/bin/env python3
"""Per-constraint (partial) scores for every prefiltered prompt. Run in a container.

For each of the 26k prompts x 8 saved rollouts, records how many constraints each
rollout satisfied, so the pool can be re-cut by partial score instead of the
all-or-nothing pass rate the first filter used.

Writes out/if_prefilter/partial_scores.jsonl:
  key, domain, nc, binary_pass (k/8), partial[8] (fraction satisfied per rollout),
  per_constraint_pass {id: k/8}, unclosed (rollouts with no </think>)
"""
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("IF_IFBENCH_DIR", "/workspace/repos/IFBench")
os.environ.setdefault("NLTK_DATA", "/workspace/out/nltk_data")
os.environ["IF_THINK_PREFILL"] = "1"
sys.path.insert(0, "/workspace/instella_think")


class S:
    def __init__(self, prompt, label):
        self.prompt, self.label, self.metadata = prompt, label, {}


def work(line):
    import if_reward as R
    try:
        g = json.loads(line)
    except json.JSONDecodeError:
        return None
    ids, kws = R._constraints_of(S(g["prompt"], g["label"]), {})
    if not ids:
        return None
    partial, per_c, unclosed, full, mat = [], {i: 0 for i in ids}, 0, 0, []
    for resp in g["responses"]:
        if "</think>" not in resp or not resp.strip():
            unclosed += 1
            partial.append(0.0)
            mat.append([0] * len(ids))
            continue
        ans = R._clean_answer(resp)
        ok = [bool(ans) and R._check_one(i, kw, ans, g["prompt"]) for i, kw in zip(ids, kws)]
        mat.append([int(o) for o in ok])
        for i, o in zip(ids, ok):
            per_c[i] += int(o)
        partial.append(sum(ok) / len(ok))
        full += int(all(ok))
    n = len(g["responses"])
    return json.dumps({
        "key": g["key"], "domain": g["domain"], "nc": g["nc"],
        "binary_pass": full / n, "partial": partial,
        "per_constraint_pass": {i: c / n for i, c in per_c.items()},
        "ids": ids, "mat": mat,
        "unclosed": unclosed,
    })


if __name__ == "__main__":
    lines = []
    for f in sorted(glob.glob(os.environ.get("GEN_GLOB", "/workspace/out/if_prefilter/gen_g*.jsonl.gz"))):
        try:
            with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
                for line in fh:
                    lines.append(line)
        except (EOFError, OSError):
            pass  # shard still being appended: keep what was read
    out = os.environ.get("OUT", "/workspace/out/if_prefilter/partial_scores_mat.jsonl")
    with Pool(int(os.environ.get("NPROC", "24"))) as p, open(out, "w") as fo:
        n = 0
        for r in p.imap_unordered(work, lines, chunksize=64):
            if r:
                fo.write(r + "\n")
                n += 1
    print(f"wrote {n} -> {out}")
