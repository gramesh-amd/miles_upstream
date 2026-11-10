#!/usr/bin/env python3
"""Shards for a multi-node prefilter (prefilter.sbatch). Runs on the login node.

    prefilter_shards.py split <K> [INPUT]   # INPUT (default the code training set) -> K shards
    prefilter_shards.py status              # graded / total per shard
    prefilter_shards.py merge [OUTPUT]      # all shards done -> one filtered set + summed stats

Shards are round-robin by line (the set is shuffled by the run anyway) and live in
$DATA_ROOT/prefilter/shards/shard_NN.jsonl; each shard's filter writes shard_NN.filtered.jsonl,
its .cache.jsonl (one graded prompt per line after a config header) and, when complete, .stats.json.
The merged set defaults to $DATA_ROOT/prefilter/dolci-code-train.filtered.jsonl, the path
README "5b" trains from (CODE_TRAIN=...).
"""

import glob
import json
import os
import sys

DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.join(
    os.environ.get("LUX_ROOT", f"/lustre/orion/ven136/scratch/{os.environ.get('USER', '')}/instella"), "data")
SHARDS = os.path.join(DATA_ROOT, "prefilter", "shards")


def lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def split(k, src):
    os.makedirs(SHARDS, exist_ok=True)
    if glob.glob(os.path.join(SHARDS, "shard_*.jsonl")):
        sys.exit(f"{SHARDS} already has shards; remove them to re-split")
    outs = [open(os.path.join(SHARDS, f"shard_{i:02d}.jsonl"), "w") for i in range(k)]
    n = 0
    with open(src) as f:
        for n, line in enumerate(f, 1):
            outs[(n - 1) % k].write(line)
    for o in outs:
        o.close()
    print(f"{n} prompts from {src} -> {k} shards in {SHARDS}")


def shard_inputs():
    return sorted(p for p in glob.glob(os.path.join(SHARDS, "shard_*.jsonl"))
                  if not p.endswith((".filtered.jsonl", ".cache.jsonl")))


def status():
    tot_done = tot = 0
    for p in shard_inputs():
        out = p[:-len(".jsonl")] + ".filtered.jsonl"
        total = lines(p)
        done = max(0, lines(out + ".cache.jsonl") - 1) if os.path.exists(out + ".cache.jsonl") else 0
        tot_done, tot = tot_done + done, tot + total
        print(f"{os.path.basename(p)}: {done}/{total}{'  complete' if os.path.exists(out + '.stats.json') else ''}")
    print(f"total: {tot_done}/{tot} ({100.0 * tot_done / max(tot, 1):.1f}%)")


def merge(dst):
    inputs = shard_inputs()
    missing = [p for p in inputs if not os.path.exists(p[:-len(".jsonl")] + ".filtered.jsonl.stats.json")]
    if missing:
        sys.exit(f"{len(missing)} shards not complete, e.g. {missing[:3]}")
    kept = 0
    hist = {}
    with open(dst, "w") as out:
        for p in inputs:
            base = p[:-len(".jsonl")] + ".filtered.jsonl"
            with open(base) as f:
                for line in f:
                    out.write(line)
                    kept += 1
            stats = json.load(open(base + ".stats.json"))
            for key, v in (stats.get("histogram") or {}).items():
                hist[key] = hist.get(key, 0) + v
    total = sum(lines(p) for p in inputs)
    json.dump({"prompts": total, "kept": kept, "histogram": hist}, open(dst + ".stats.json", "w"), indent=1)
    print(f"merged {len(inputs)} shards: kept {kept}/{total} -> {dst}\nhistogram (n_correct: prompts): {hist}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "split":
        split(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_ROOT, "code-train", "dolci-code-train.jsonl"))
    elif cmd == "status":
        status()
    elif cmd == "merge":
        merge(sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_ROOT, "prefilter", "dolci-code-train.filtered.jsonl"))
    else:
        sys.exit(__doc__)
