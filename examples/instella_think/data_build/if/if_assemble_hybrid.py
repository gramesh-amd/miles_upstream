#!/usr/bin/env python3
"""Assemble the IF2 training set for the hybrid reward
r = (1-w)*fraction_satisfied + w*all_satisfied  (if_reward.py, IF_STRICT_WEIGHT=w).

Selection, per prompt, from the 8 prefilter rollouts in partial_scores_mat.jsonl:
  1. signal: the 8 hybrid rewards are not all equal (otherwise GRPO advantage is 0);
  2. not easy: strict pass rate <= --max-pass (Olmo 3's 62.5% rule);
  3. satisfiable: a prompt with 0/8 strict passes is dropped if it pairs constraints
     that IFEvalG's own INSTRUCTION_CONFLICTS forbids (e.g. copy_span_idx, whose
     checker demands the whole response equal a span, next to anything else), or if
     it contains a constraint type that passes < --dead-type-rate of rollouts
     pool-wide. Prompts that passed strictly at least once are kept regardless --
     they are demonstrably satisfiable (Precision-over-Diversity's learnability rule).
Then re-stratify by (domain, nc) exactly as if_assemble.py does.
"""
import argparse, collections, json
from pathlib import Path

from if_assemble import DOMAIN_TARGET, NC_TARGET, ROOT

MAT = ROOT / "out/if_prefilter/partial_scores_mat.jsonl"
CONFLICTS = ROOT / "out/if_prefilter/conflict_keys.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-weight", type=float, default=0.5)
    ap.add_argument("--max-pass", type=float, default=0.625)
    ap.add_argument("--dead-type-rate", type=float, default=0.02)
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--out", default="data/if/if_train_hybrid.jsonl")
    ap.add_argument("--mat", default=str(MAT))
    ap.add_argument("--candidates", default="data/if/if_candidates.jsonl")
    ap.add_argument("--conflicts", default=str(CONFLICTS), help="'' when candidates are pre-screened")
    a = ap.parse_args()
    w = a.strict_weight

    rows = [json.loads(l) for l in open(a.mat)]
    conflict = set(json.load(open(a.conflicts))) if a.conflicts else set()

    tot, ok = collections.Counter(), collections.Counter()
    for r in rows:
        for j, cid in enumerate(r["ids"]):
            for m in r["mat"]:
                tot[cid] += 1
                ok[cid] += m[j]
    dead = {c for c in tot if ok[c] / tot[c] < a.dead_type_rate}
    print("dead constraint types (<%.0f%% pass):" % (100 * a.dead_type_rate), sorted(dead))

    drop = collections.Counter()
    keep = []
    for r in rows:
        hyb = [(1 - w) * sum(m) / len(m) + w * float(all(m)) for m in r["mat"]]
        r["hybrid_mean"] = sum(hyb) / len(hyb)
        if max(hyb) - min(hyb) < 1e-9:
            drop["no_variance"] += 1
            continue
        if r["binary_pass"] > a.max_pass:
            drop["too_easy"] += 1
            continue
        if r["binary_pass"] == 0 and r["key"] in conflict:
            drop["conflict_never_passed"] += 1
            continue
        if r["binary_pass"] == 0 and dead & set(r["ids"]):
            drop["dead_type_never_passed"] += 1
            continue
        keep.append(r)
    print(f"pool {len(rows):,} -> kept {len(keep):,}; dropped {dict(drop)}")
    print("  kept with >=1 strict pass:", sum(r["binary_pass"] > 0 for r in keep))

    pool = collections.defaultdict(list)
    for r in keep:
        pool[(r["domain"], r["nc"])].append(r)
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
        print(f"redistributed {min(short, len(spare)):,} from under-filled cells")

    cand = {json.loads(l)["metadata"]["key"]: json.loads(l)
            for l in (ROOT / a.candidates).open()}
    dest = ROOT / a.out
    with dest.open("w") as fh:
        for r in final:
            c = {k: v for k, v in cand[r["key"]].items() if k != "rendered"}
            fh.write(json.dumps({**c, "metadata": {**c["metadata"],
                                                   "pass_rate": r["binary_pass"],
                                                   "hybrid_mean": r["hybrid_mean"]}}) + "\n")
    n = len(final)
    print(f"\nwrote {n:,} -> {dest}")
    print("  domain mix:", {k: f"{v/n:.1%}" for k, v in
                            sorted(collections.Counter(r['domain'] for r in final).items())})
    print("  constraint mix:", {k: f"{v/n:.1%}" for k, v in
                                sorted(collections.Counter(r['nc'] for r in final).items())})
    print(f"  mean strict pass {sum(r['binary_pass'] for r in final)/n:.3f}, "
          f"mean hybrid {sum(r['hybrid_mean'] for r in final)/n:.3f}, "
          f"0/8-strict share {sum(r['binary_pass']==0 for r in final)/n:.1%}")


if __name__ == "__main__":
    main()
