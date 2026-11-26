#!/usr/bin/env python3
"""Recompute which prompts are still ungraded, and cut fresh chunks for them.

Called by grade_supervisor.sh on every pass.

IDENTITY. A prompt is keyed by "<source>:<line>" -- source file plus its line
number in that file. The grader's cache records only {i, n_correct,
n_truncated}, where i is the ROW INDEX IN ITS INPUT FILE, so the input file is
the only thing that maps a grade back to a prompt. Every chunk file is
therefore kept forever and never rewritten; a chunk's sidecar .key file stores
the ordered prompt keys for exactly that reason.

IN-PROGRESS WORK IS EXCLUDED. A chunk that has been claimed is treated as
spoken for even if its grades are not in yet, so a second node is never handed
prompts another node is mid-way through. That is what prevents the duplicate
cache records seen on g33.

The legacy shards from before the supervisor (shard_g33/g34/g35 for DAPO and
dolci_gap/shard_g15/g32) are read the same way, so their completed work counts
and is not regraded.
"""
import argparse
import glob
import json
import os

ROOT = "/it-share-prj2-1/data/goramesh-instella-rl"
SOURCES = {
    "dapo": f"{ROOT}/out/analysis_1012/dapo/dapo_17k_en.jsonl",
    "dolci": f"{ROOT}/out/analysis_1012/dolci_filter/pool_30k.jsonl",
}
# Pre-supervisor assignments: (chunk-key file or source, input path, cache glob)
LEGACY = [
    ("dapo", f"{ROOT}/out/analysis_1012/dapo/grades/shard_{n}.jsonl",
     f"{ROOT}/out/analysis_1012/dapo/grades/graded_{n}.jsonl.cache.jsonl") for n in ("g33", "g34", "g35")
] + [
    ("dolci", f"{ROOT}/out/analysis_1012/dolci_gap/shard_{n}.jsonl",
     f"{ROOT}/out/analysis_1012/dolci_gap/graded_{n}.jsonl.cache.jsonl") for n in ("g15", "g32")
]


def load_keys(src, path):
    """Ordered prompt keys for an input file, matched back to its source."""
    if not os.path.exists(path):
        return []
    base = [json.loads(l) for l in open(SOURCES[src], errors="replace")]
    index = {}
    for i, r in enumerate(base):
        index.setdefault(" ".join(str(r["prompt"][0]["content"]).split())[:300], i)
    out = []
    for l in open(path, errors="replace"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            out.append(None)
            continue
        k = " ".join(str(r["prompt"][0]["content"]).split())[:300]
        i = index.get(k)
        out.append(f"{src}:{i}" if i is not None else None)
    return out


def graded_from(keys, cache):
    done = set()
    if not os.path.exists(cache):
        return done
    for l in open(cache, errors="replace"):
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        i = d.get("i")
        if i is not None and 0 <= i < len(keys) and keys[i]:
            done.add(keys[i])
    return done


ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--chunk", type=int, default=600)
ap.add_argument("--live", default="", help="comma-separated nodes whose grader is ALIVE")
a = ap.parse_args()
chunks = os.path.join(a.work, "chunks")
claims = os.path.join(a.work, "claims")
os.makedirs(chunks, exist_ok=True)
os.makedirs(claims, exist_ok=True)

done, spoken_for = set(), set()

# ALREADY-RECOVERED dolci grades. 27239 unique prompts were graded months ago on
# this same DPO checkpoint and recovered from the bucket; without this the
# supervisor would regrade the entire 30k pool -- ~90 node-hours of pure waste.
# grades_30k.jsonl is keyed by gidx, which IS the dolci line number.
REC = f"{ROOT}/out/analysis_1012/dolci_filter/grades_30k.jsonl"
if os.path.exists(REC):
    n_rec = 0
    for l in open(REC, errors="replace"):
        try:
            g = json.loads(l)["gidx"]
        except Exception:  # noqa: BLE001
            continue
        done.add(f"dolci:{int(g)}")
        n_rec += 1
    print(f"  recovered dolci grades: {n_rec} lines -> {len(done)} unique")

# legacy shards. Their COMPLETED grades count as done, and because a grader may
# still be working through them, EVERY prompt in those shards is also marked
# spoken-for so a second node is never handed the same prompt. This is the
# duplicate-record failure that hit g33.
LIVE = {x for x in a.live.split(",") if x}
for src, inp, cache in LEGACY:
    keys = load_keys(src, inp)
    done |= graded_from(keys, cache)
    # Only reserve a legacy shard while its grader is actually RUNNING. These
    # nodes get reclaimed without warning; if a dead node's shard stayed
    # reserved forever its ungraded prompts could never be picked up again.
    node = inp.rsplit("_", 1)[-1].split(".")[0]
    if node in LIVE:
        spoken_for |= {k for k in keys if k}

# supervisor chunks: completed grades plus everything in a CLAIMED chunk
for kf in sorted(glob.glob(os.path.join(chunks, "chunk_*.key"))):
    cid = os.path.basename(kf)[:-4]
    keys = [l.strip() or None for l in open(kf)]
    done |= graded_from(keys, os.path.join(chunks, f"graded_{cid}.jsonl.cache.jsonl"))
    if os.path.exists(os.path.join(claims, cid)):
        spoken_for |= {k for k in keys if k}

remaining = []
for src, path in SOURCES.items():
    n = sum(1 for _ in open(path, errors="replace"))
    for i in range(n):
        k = f"{src}:{i}"
        if k not in done and k not in spoken_for:
            remaining.append(k)

print(f"  graded={len(done)} in_progress={len(spoken_for)} remaining={len(remaining)}")
if not remaining:
    print("  nothing left to chunk")
    raise SystemExit(0)

# cut new chunks only for what is not already sitting in an unclaimed chunk
already = set()
for kf in glob.glob(os.path.join(chunks, "chunk_*.key")):
    cid = os.path.basename(kf)[:-4]
    if not os.path.exists(os.path.join(claims, cid)):
        already |= {l.strip() for l in open(kf) if l.strip()}
todo = [k for k in remaining if k not in already]
if not todo:
    print(f"  {len(already)} prompts already sit in unclaimed chunks; none cut")
    raise SystemExit(0)

cache = {s: [json.loads(l) for l in open(p, errors="replace")] for s, p in SOURCES.items()}
nxt = len(glob.glob(os.path.join(chunks, "chunk_*.jsonl")))
made = 0
for off in range(0, len(todo), a.chunk):
    part = todo[off:off + a.chunk]
    cid = f"chunk_{nxt + made:04d}"
    with open(os.path.join(chunks, f"{cid}.jsonl"), "w") as fh, \
         open(os.path.join(chunks, f"{cid}.key"), "w") as kh:
        for k in part:
            s, i = k.split(":")
            fh.write(json.dumps(cache[s][int(i)]) + "\n")
            kh.write(k + "\n")
    made += 1
print(f"  cut {made} new chunk(s) of up to {a.chunk}")
