#!/usr/bin/env python3
"""Unverified kept rows where >= 2 references pass every other test and print the SAME output on the
one(s) they fail: candidate wrong labels. Majority agreement is not a proof, so these are only listed
(work/blocking_candidates.csv) and fed to hand brute-force checks (bruteforce/)."""
import collections, csv, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
from decide import seg_results

OUTD = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio"
W = f"{OUTD}/work"


def main():
    vs = {v["row"]: v for v in pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))}
    pools = pickle.load(open(f"{W}/pool.pkl", "rb"))
    R = pickle.load(open(f"{W}/results.pkl", "rb"))
    DEC = pickle.load(open(f"{W}/decisions.pkl", "rb"))
    rbr = collections.defaultdict(dict)
    for (r, j, k), res in R.items():
        rbr[r][(k, j)] = res
    out = []
    for r, d in DEC.items():
        if d.get("status") != "kept" or d.get("verified") or vs[r]["kind"] != "stdio":
            continue
        segr = seg_results(pools[r], rbr.get(r, {}))
        vec = {j: sr["orig"] for j, sr in segr.items() if "orig" in sr}
        n = len(vs[r]["label"])
        for ti in range(n):
            agree = collections.Counter()
            for j, rs in vec.items():
                if not rs[ti]["ex"] and rs[ti].get("h") and all(rs[k]["ex"] for k in range(n) if k != ti):
                    agree[rs[ti]["h"]] += 1
            if agree and agree.most_common(1)[0][1] >= 2:
                h = agree.most_common(1)[0][0]
                got = next(rs[ti]["out"] for rs in vec.values() if rs[ti].get("h") == h)
                out.append(dict(custom_id=vs[r]["custom_id"], src=vs[r]["src"], test=ti, n_tests=n,
                                n_refs_agree=agree[h], label=vs[r]["label"][ti]["output"][:120], refs_out=got[:120],
                                input_len=len(vs[r]["label"][ti]["input"])))
    with open(f"{OUTD}/blocking_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0].keys()) if out else ["custom_id"])
        w.writeheader()
        w.writerows(out)
    print(len(out), "candidate blocking tests in", len({o["custom_id"] for o in out}), "rows")
    for o in sorted(out, key=lambda o: o["input_len"])[:25]:
        print(o)


if __name__ == "__main__":
    main()
