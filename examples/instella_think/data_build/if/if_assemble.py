#!/usr/bin/env python3
"""Assemble the final IF training set from scored prefilter shards.

Filters on pass rate, then RE-STRATIFIES the survivors, because difficulty
filtering is not uniform across cells: easy 1-constraint prompts are culled
far more often than 5-constraint ones, and domains differ too. Stratifying
before filtering (which is what I first proposed) produces a set whose
advertised balance no longer holds. Olmo's own order is source -> filter ->
mix, i.e. balance last.

THRESHOLDS. Drop pass rate > 0.625 only -- Olmo's rule, and Olmo's ONLY rule:
"remove all samples that the model easily solves (that is, those with a pass
rate greater than 62.5%)". Polaris likewise removes only the 8/8 cases.
Neither drops 0/8.

An earlier version of this script also dropped 0/8 as "no GRPO signal". On this
data that removed 78% of the set, because these IFBench-Train constraints are
genuinely hard for our DPO -- new:copy_span_idx (exact character-index spans)
and count:counting_composition fail 100% of the time, and the measured IFBench
score of 0.21 says the same thing. Zero-variance groups cost nothing anyway:
active sampling skips them online until the policy improves enough to produce
variance, which is exactly the prompts we want it growing into.
"""
import argparse, collections, glob, json
from pathlib import Path

ROOT = Path("/it-share-prj2-1/data/goramesh-instella-rl")
SCORED = ROOT / "out/if_prefilter"

DOMAIN_TARGET = {"chat": 0.20, "flan": 0.18, "math": 0.14, "code": 0.14,
                 "other": 0.14, "safety": 0.10, "multiling": 0.10}
NC_TARGET = {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pass", type=float, default=0.625)
    ap.add_argument("--min-pass", type=float, default=0.0)   # keep 0/8 by default
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--out", default="data/if/if_train.jsonl")
    a = ap.parse_args()

    scored = {}
    for f in glob.glob(str(SCORED / "scored_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            scored[r["key"]] = r
    if not scored:
        raise SystemExit("no scored shards found")

    cand = {json.loads(l)["metadata"]["key"]: json.loads(l)
            for l in (ROOT / "data/if/if_candidates.jsonl").open()}

    total = len(scored)
    dist = collections.Counter(r["n_pass"] for r in scored.values())
    print(f"scored prompts: {total:,}")
    print("pass-rate distribution (of 8):",
          {k: f"{v} ({v/total:.1%})" for k, v in sorted(dist.items())})

    keep = [r for r in scored.values()
            if a.min_pass <= r["pass_rate"] <= a.max_pass]
    print(f"\nsurvivors after {a.min_pass}<=rate<={a.max_pass}: {len(keep):,} "
          f"({len(keep)/total:.1%})")
    print("  dropped too easy (>%.3f): %d" % (a.max_pass,
          sum(1 for r in scored.values() if r["pass_rate"] > a.max_pass)))
    print("  dropped 0/8 (no signal) : %d" %
          sum(1 for r in scored.values() if r["n_pass"] == 0))

    # survival by cell, which is the number that invalidated pre-filter balance
    by = collections.defaultdict(lambda: [0, 0])
    for r in scored.values():
        c = by[(r["domain"], r["nc"])]
        c[0] += 1
        if a.min_pass <= r["pass_rate"] <= a.max_pass:
            c[1] += 1
    print("\nsurvival by domain:")
    dom = collections.defaultdict(lambda: [0, 0])
    for (d, _), (t, s) in by.items():
        dom[d][0] += t; dom[d][1] += s
    for d, (t, s) in sorted(dom.items()):
        print(f"  {d:10s} {s:5,}/{t:5,}  {s/max(t,1):5.1%}")
    print("survival by constraint count:")
    nc = collections.defaultdict(lambda: [0, 0])
    for (_, n), (t, s) in by.items():
        nc[n][0] += t; nc[n][1] += s
    for n, (t, s) in sorted(nc.items()):
        print(f"  {n} constraint{'s' if n>1 else ' '} {s:5,}/{t:5,}  {s/max(t,1):5.1%}")

    # re-stratify survivors to target, limited by the scarcest cell
    pool = collections.defaultdict(list)
    for r in keep:
        pool[(r["domain"], r["nc"])].append(r)
    # APPROXIMATE balance, not exact. Sizing every cell to the scarcest one
    # collapsed a 3,642-survivor set to 890: one thin cell (5-constraint
    # safety) dictated all 35. Instead take what each cell has up to its
    # target, then redistribute the shortfall across cells that still have
    # supply, preferring deeper constraint counts.
    final, short = [], 0
    for d, ds in DOMAIN_TARGET.items():
        for n, ns in NC_TARGET.items():
            want = int(a.target * ds * ns)
            got = pool[(d, n)][:want]
            final.extend(got)
            short += want - len(got)
    if short > 0:
        taken = {r["key"] for r in final}
        spare = sorted((r for cell in pool.values() for r in cell if r["key"] not in taken),
                       key=lambda r: -r["nc"])
        final.extend(spare[:short])
        print(f"\nredistributed {min(short, len(spare)):,} from under-filled cells "
              f"(deepest constraint counts first)")

    dest = ROOT / a.out
    with dest.open("w") as fh:
        for r in final:
            c = cand[r["key"]]
            fh.write(json.dumps({**c, "metadata": {**c["metadata"],
                                                   "pass_rate": r["pass_rate"]}}) + "\n")
    n = len(final)
    print(f"\nwrote {n:,} -> {dest}")
    print("  domain mix:", {k: f"{v/n:.1%}" for k, v in
                            sorted(collections.Counter(r['domain'] for r in final).items())})
    print("  constraint mix:", {k: f"{v/n:.1%}" for k, v in
                                sorted(collections.Counter(r['nc'] for r in final).items())})
    print(f"  mean pass rate: {sum(r['pass_rate'] for r in final)/max(n,1):.3f}")


if __name__ == "__main__":
    main()
