#!/usr/bin/env python3
"""Build data/mix_v4_math.jsonl.

THE SELECTION RULE: keep everything the model does NOT already solve perfectly.
Drop 8/8, keep 0/8 through 7/8.

This is a deliberate change from the 1/8-7/8 band I had been sizing against, and
three independent lines of evidence point the same way:

  * POLARIS ablated exactly this on DeepScaleR-40K with R1-Distill-Qwen-7B.
    Full set (easy-heavy, J-shaped) -> worse. Remove 8/8 only (26K, mirrored-J)
    -> BEST. Aggressively keep only <=4/8 (19K) -> worse again. Over-filtering
    the easy side past the 8/8 cut actively hurt.
    https://hkunlp.github.io/blog/2025/Polaris/
  * OLMO 3 filters only the easy side offline (drop pass rate > 0.625) and
    leans on in-training dynamic sampling for the hard side.
  * OUR OWN BEST ARM already did this. Arm A trained on dolci, which retained
    its 0/8 prompts; it was the strongest arm on AIME. Excluding 0/8 now would
    depart from the only configuration of ours that has worked.

0/8 prompts are not free -- they burn 8 rollouts for zero advantage until the
policy improves enough to crack them -- but the online dynamic-sampling filter
already discards zero-variance groups, so the cost is rollout time rather than
corrupted gradient, and the prompts become live as the policy improves.

ZEBRALOGIC IS EXCLUDED. Polaris is 26K DeepScaleR + 27K AReaL-boba, and
AReaL-boba carries ZebraLogic logic-grid puzzles ("There are N houses, numbered
1 to N...", answers like 'tennis' and 'tall'). That is 5.0% of our pool and it
is not competition math; training on it spends AIME budget on constraint
puzzles. Polaris publishes no per-row source tag, so this is matched on the
problem text and the finding is recorded rather than assumed.

OMEGA IS EXCLUDED, unchanged from the previous build.
"""
import json
import os
import re
from collections import Counter, defaultdict

ROOT = "/it-share-prj2-1/data/goramesh-instella-rl"
A = f"{ROOT}/out/analysis_1012"
OUT = os.environ.get("OUT", f"{ROOT}/data/mix_v4_math.jsonl")
# Keep anything not perfectly solved. Set MAXK=7 to also drop 8/8 (default) and
# MINK to raise the hard-side floor if we ever want the strict band back.
MINK = int(os.environ.get("MINK", "0"))
MAXK = int(os.environ.get("MAXK", "7"))
DROP_OMEGA = os.environ.get("DROP_OMEGA", "1") == "1"

ZEBRA = re.compile(
    r"there are \d+ houses, numbered|each house is occupied by a different person",
    re.I)


def norm(s):
    return " ".join(str(s).lower().split())


def getp(r):
    p = r.get("prompt")
    if isinstance(p, list) and p and isinstance(p[0], dict):
        return p[0].get("content", "")
    return p if isinstance(p, str) else ""


records = []        # (source, prompt_obj, label, n_correct)
stats = defaultdict(Counter)

# ---- 1. dolci: pool_30k joined to the recovered grades by global index -----
# grades_30k carries gidx into pool_30k. mixv4_band.py verifies this join by
# label equality; it is re-verified here because a silent off-by-one would
# attach every difficulty to the wrong prompt.
pool = [json.loads(l) for l in open(f"{A}/dolci_filter/pool_30k.jsonl")]
ok = bad = 0
for l in open(f"{A}/dolci_filter/grades_30k.jsonl"):
    g = json.loads(l)
    i = g.get("gidx")
    if i is None or not (0 <= i < len(pool)):
        continue
    p = pool[i]
    if str(p.get("label", "")).strip() != str(g.get("label", "")).strip():
        bad += 1
        continue
    ok += 1
    src = p.get("metadata", {}).get("dataset_source", "?")
    if DROP_OMEGA and "omega" in src.lower():
        stats[src]["dropped_omega"] += 1
        continue
    k = int(round(float(g["pass_rate"]) * 8))
    stats[src]["graded"] += 1
    if MINK <= k <= MAXK:
        records.append((src, p.get("prompt"), p.get("label"), k))
        stats[src]["kept"] += 1
rate = ok / max(ok + bad, 1)
print(f"dolci join: {ok} label matches, {bad} mismatches ({rate:.1%})")
assert rate > 0.95, "gidx -> pool_30k join is wrong; difficulties would be misattached"

# ---- 2 & 3. shard+cache grading runs (DAPO, Polaris) ----------------------
def add_shards(name, gdir, nodes, zebra_filter=False):
    for n in nodes:
        shard = f"{gdir}/shard_{n}.jsonl"
        cache = f"{gdir}/graded_{n}.jsonl.cache.jsonl"
        if not (os.path.exists(shard) and os.path.exists(cache)):
            continue
        recs = [json.loads(l) for l in open(shard)]
        seen = {}
        for l in open(cache):
            try:
                c = json.loads(l)
            except json.JSONDecodeError:
                continue
            seen[c["i"]] = c          # last write wins; duplicates have occurred
        for i, c in seen.items():
            if i >= len(recs):
                continue
            r = recs[i]
            stats[name]["graded"] += 1
            if zebra_filter and ZEBRA.search(getp(r)):
                stats[name]["dropped_zebralogic"] += 1
                continue
            k = int(c["n_correct"])
            if MINK <= k <= MAXK:
                records.append((name, r.get("prompt"), r.get("label"), k))
                stats[name]["kept"] += 1


NODES = ["g2", "g15", "g32", "g33", "g34", "g35"]
add_shards("dapo_17k_en", f"{A}/dapo/grades", NODES)
add_shards("polaris_band1_5", f"{A}/polaris/grades", NODES, zebra_filter=True)

# ---- dedup across every source -------------------------------------------
# The previous build found 2,743 dolci<->DAPO duplicates, and Polaris
# transitively contains DAPO and Open-Reasoner-Zero content via AReaL-boba, so
# cross-source collision is expected rather than hypothetical.
seen, final = set(), []
dups = Counter()
for src, prompt, label, k in records:
    key = norm(getp({"prompt": prompt}))
    if not key or label in (None, ""):
        continue
    if key in seen:
        dups[src] += 1
        continue
    seen.add(key)
    final.append({"prompt": prompt, "label": str(label),
                  "metadata": {"dataset_source": src, "n_correct": k}})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    for r in final:
        fh.write(json.dumps(r) + "\n")

print(f"\nrule: keep {MINK}/8 .. {MAXK}/8  (drop 8/8 = perfectly solved)")
print(f"{'source':<26} {'graded':>8} {'kept':>8} {'dups':>7} {'notes'}")
for s in sorted(stats, key=lambda x: -stats[x]["kept"]):
    c = stats[s]
    extra = " ".join(f"{k}={v}" for k, v in c.items()
                     if k.startswith("dropped"))
    print(f"  {s[:24]:<24} {c['graded']:>8} {c['kept']:>8} {dups[s]:>7}  {extra}")
print(f"\nmix_v4: {len(final):,} prompts -> {OUT}")

h = Counter(r["metadata"]["n_correct"] for r in final)
print("\ndifficulty distribution (should be mirrored-J: hard-leaning):")
for k in range(9):
    c = h.get(k, 0)
    bar = "#" * int(60 * c / max(h.values()))
    print(f"  {k}/8 {c:>6} {100*c/len(final):>5.1f}% {bar}")
zero = h.get(0, 0)
print(f"\n  0/8 share {100*zero/len(final):.1f}%  (no gradient until the policy improves)")
print(f"  mean pass rate {sum(k*v for k, v in h.items())/8/len(final):.3f}")

nums = sorted(int(x) for x in (str(r["label"]).strip() for r in final)
              if re.fullmatch(r"\d{1,9}", x))
if nums:
    import statistics
    print(f"\nanswers: integer {100*len(nums)/len(final):.1f}%  "
          f"median {statistics.median(nums):.0f}  "
          f"in 0-999 {100*sum(1 for x in nums if x<=999)/len(nums):.1f}%")
