#!/usr/bin/env python3
"""Item 11: rows never solved in v6 (0 full solves over all kept rollouts) that were also 0/8 at
the v6 prefilter. Seeded sample of 40 (20 Klear, 20 S2) dumped with execution evidence for hand
diagnosis. Out: work/never_solved_dump.txt, never_solved_population.csv
"""
import collections, csv, os, pickle, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from decide import seg_results

OUTD = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio"
W = f"{OUTD}/work"


def main():
    vs = pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))
    A = pickle.load(open(f"{W}/analysis.pkl", "rb"))
    pools = pickle.load(open(f"{W}/pool.pkl", "rb"))
    R = pickle.load(open(f"{W}/results.pkl", "rb"))
    DEC = pickle.load(open(f"{W}/decisions.pkl", "rb"))
    rbr = collections.defaultdict(dict)
    for (r, j, k), res in R.items():
        rbr[r][(k, j)] = res
    pop = [v for v in vs if v.get("in_v6") and v.get("v6_solved", 0) == 0
           and (v.get("prefilter") or {}).get("n_correct", -1) == 0]
    with open(f"{OUTD}/never_solved_population.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["custom_id", "src", "v6_rollouts", "v6_max_reward", "decision", "reason"])
        for v in pop:
            d = DEC.get(v["row"], {})
            w.writerow([v["custom_id"], v["src"], v.get("v6_n"), v.get("v6_max"), d.get("status"), d.get("reason", "")])
    print("population", collections.Counter(v["src"] for v in pop))
    rnd = random.Random(0)
    sample = []
    for s in ("klear", "synthetic2"):
        sample += rnd.sample([v for v in pop if v["src"] == s], 20)
    with open(f"{W}/never_solved_dump.txt", "w") as f:
        for n, v in enumerate(sample):
            a = A[v["row"]]
            segr = seg_results(pools[v["row"]], rbr.get(v["row"], {}))
            f.write(f"\n{'=' * 100}\n#{n} {v['custom_id']} src={v['src']} v6_n={v.get('v6_n')} v6_max={v.get('v6_max')} "
                    f"fmt={a['fmt']} tol={a['tol']} case={a['case_stmt']} multi={bool(a['multi_hits'])} img={a['image']}\n")
            f.write(f"decision: {DEC.get(v['row'])}\n")
            f.write(v["prompt"][:3000] + ("\n...[cut]" if len(v["prompt"]) > 3000 else "") + "\n")
            f.write(f"-- tests ({len(v['label'])}):\n")
            for t in v["label"][:3]:
                f.write(f"   {str(t)[:300]!r}\n")
            f.write("-- refs:\n")
            per_test = collections.Counter()
            for j, sr in segr.items():
                o = sr.get("orig") or sr.get("to_assert") or []
                for i, r in enumerate(o):
                    per_test[i] += r["ex"]
                rates = sum(r["ex"] for r in o) / max(1, len(o))
                to = sum(r["rc"] in ("to", "skip") for r in o)
                f.write(f"   ref{j} {v['refs'][j]['origin']} pass={rates:.2f} timeouts={to}\n")
                bad = [(i, r) for i, r in enumerate(o) if not r["ex"] and r.get("out") is not None]
                for i, r in bad[:2]:
                    exp = v["label"][i]["output"] if isinstance(v["label"][i], dict) else v["label"][i]
                    f.write(f"      t{i} got={r['out'][:100]!r} exp={str(exp)[:100]!r} cs={r.get('cs')} fl={r.get('fl')}\n")
            f.write(f"   per-test ref passes: {dict(sorted(per_test.items()))}\n")
    print("wrote", len(sample))


if __name__ == "__main__":
    main()
