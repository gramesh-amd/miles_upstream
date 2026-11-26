#!/usr/bin/env python3
"""Re-run assert conversions whose candidate tests changed after analyze.py was revised (JSON-decoding
rule). Replaces the to_assert segment in pool.pkl and the matching (row, ref, 'assert') results; every
reference of the row is tried (stop at the first full pass)."""
import collections, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
import sandbox

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/work"


def main():
    vs = {v["row"]: v for v in pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))}
    A = pickle.load(open(f"{W}/analysis.pkl", "rb"))
    pools = pickle.load(open(f"{W}/pool.pkl", "rb"))
    R = pickle.load(open(f"{W}/results.pkl", "rb"))
    jobs = []
    for r, a in A.items():
        if vs[r]["kind"] != "stdio":
            continue
        new = a["conv_cands"].get("to_assert")
        old = next((s for s in pools[r] if s[0] == "to_assert"), None)
        if (new[1] if new else None) == (old[2] if old else None):
            continue
        pools[r] = [s for s in pools[r] if s[0] != "to_assert"] + ([("to_assert", "assert", new[1])] if new else [])
        for k in [k for k in R if k[0] == r and k[2] == "assert"]:
            del R[k]
        if new:
            segs = [s for s in pools[r] if s[1] == "assert"]
            jobs.append(dict(row=r, refs=list(enumerate(x["code"] for x in vs[r]["refs"])), segs=segs, need=1))
    print("changed rows", len(jobs), flush=True)
    for o in sandbox.run_rows(jobs, int(os.environ.get("PROCS", "24"))):
        R.update(o)
    pickle.dump(pools, open(f"{W}/pool.pkl", "wb"))
    pickle.dump(R, open(f"{W}/results.pkl", "wb"))


if __name__ == "__main__":
    main()
