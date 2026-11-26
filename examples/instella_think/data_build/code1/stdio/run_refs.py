#!/usr/bin/env python3
"""Run candidate references against each row's test pool in the sandbox.

Pool per row (stdio part, one job per reference): statement samples, to_stdin conversion, extra
uncapped tests (rows with < 3 tests), then the trained label tests. Assert-kind pools (Klear's 8
assert rows, to_assert conversions) are separate jobs.
Round 1: the first 4 references (v6 solved > OLMo > v6 near-miss).  Round 2: remaining references
for rows where no reference passed the trained or converted tests yet.

Out: work/pool.pkl {row: {segments}}, work/results.pkl {(row, ref_idx, pool_kind): res list}
"""
import collections, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import sandbox

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/work"
PROCS = int(os.environ.get("PROCS", "24"))


def build_pools(vs, A):
    pools = {}
    for v in vs:
        a = A[v["row"]]
        segs = []  # (name, kind, tests)
        if v["kind"] == "stdio":
            orig = [{"input": t["input"], "output": t["output"]} for t in v["label"]]
            ins = {" ".join(t["input"].split()) for t in orig}
            segs.append(("sample", "stdio", a["samples"][:5]))
            if "to_stdin" in a["conv_cands"]:
                segs.append(("to_stdin", "stdio", a["conv_cands"]["to_stdin"][1]))
            extra = []
            if len(orig) < 3 and v["full_tests"] and isinstance(v["full_tests"], list):
                extra = [t for t in v["full_tests"][len(orig):] if " ".join(str(t["input"]).split()) not in ins][:5]
            segs.append(("extra", "stdio", extra))
            segs.append(("orig", "stdio", orig))
        else:
            segs.append(("orig", "assert", list(v["label"])))
        if "to_assert" in a["conv_cands"]:
            segs.append(("to_assert", "assert", a["conv_cands"]["to_assert"][1]))
        pools[v["row"]] = [s for s in segs if s[2]]
    return pools


def jobs_for(row, refs, idxs, pool):
    out = []
    for kind in ("stdio", "assert"):
        segs = [s for s in pool if s[1] == kind]
        if not segs:
            continue
        tests = [t for s in segs for t in s[2]]
        for j in idxs:
            out.append(dict(key=(row, j, kind), code=refs[j]["code"], kind=kind, tests=tests))
    return out


def ordered(refs):
    pri = {"v6": 0, "olmo": 1, "v6near": 2}
    return sorted(range(len(refs)), key=lambda j: (pri[refs[j]["origin"]], j))


def passed_target(row, pool, res_by_ref):
    """True if some reference passes all tests of 'orig' or a conversion segment."""
    for (kind, j), res in res_by_ref.items():
        off = 0
        for name, k, tests in [s for s in pool if s[1] == kind]:
            seg = res[off: off + len(tests)]
            off += len(tests)
            if name in ("orig", "to_stdin", "to_assert") and all(r["ex"] for r in seg):
                return True
    return False


def main():
    vs = pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))
    A = pickle.load(open(f"{W}/analysis.pkl", "rb"))
    pools = build_pools(vs, A)
    byrow = {v["row"]: v for v in vs}
    results = {}
    if os.path.exists(f"{W}/results.pkl"):
        results = pickle.load(open(f"{W}/results.pkl", "rb"))
        old = pickle.load(open(f"{W}/pool.pkl", "rb"))
        for r in {k[0] for k in results}:
            pools[r] = old[r]  # results are aligned to the pool they were run on
    pickle.dump(pools, open(f"{W}/pool.pkl", "wb"))
    # rows a statement-level check drops regardless of tests are not executed
    skip = {r for r, a in A.items() if a["empty_stmt"] or a["non_english"] or a["interactive"]
            or a["other_lang"] or a["image"] or a["multi_hits"]}
    print("skipping statement-dropped rows:", len(skip), flush=True)
    byrow = {r: v for r, v in byrow.items() if r not in skip}
    order = {r: ordered(byrow[r]["refs"]) for r in byrow}
    done_rows = {k[0] for k in results}
    for rnd, sl in ((1, slice(0, 4)), (2, slice(4, None))):
        jobs = []
        for r, v in byrow.items():
            if not pools[r]:
                continue
            have = {(k, j): res for (rr, j, k), res in results.items() if rr == r}
            if rnd == 1 and r in done_rows:
                continue
            if rnd == 2 and passed_target(r, pools[r], have):
                continue
            idxs = [j for j in order[r][sl] if not any((r, j, k) in results for k in ("stdio", "assert"))]
            if idxs:
                jobs.append(dict(row=r, refs=[(j, v["refs"][j]["code"]) for j in idxs], segs=pools[r],
                                 need=1))
        jobs.sort(key=lambda j: -sum(len(s[2]) for s in j["segs"]))
        print(f"round {rnd}: {len(jobs)} row jobs", flush=True)
        t0 = time.time()
        for i, o in enumerate(sandbox.run_rows(jobs, PROCS)):
            results.update(o)
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                pickle.dump(results, open(f"{W}/results.pkl.tmp", "wb"))
                os.replace(f"{W}/results.pkl.tmp", f"{W}/results.pkl")
                print(f"  round {rnd}: {i + 1}/{len(jobs)} rows  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
