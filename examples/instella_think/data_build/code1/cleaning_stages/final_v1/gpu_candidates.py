"""Candidate lists for the GPU repairs documented in GPU_REPAIRS.md (nothing here runs a model or code).

Writes final_v1/gpu_candidates/*.jsonl. Each line: custom_id, src, route, reason, prompt (messages, unchanged),
tests (the current cleaned tests, or the v6 label when the row never reached cleaning), refs (up to 16:
origin, code, passes_existing_tests when known), olmo_passrate, plus per-list fields. `priority` marks the
rows that meet the list's entry criteria in GPU_REPAIRS.md.

    python3 gpu_candidates.py
"""
import collections
import csv
import json
import os
import pickle
import sys

sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/scripts")
from analyze import norm_in  # noqa: E402
from decide import seg_results, all_ok  # noqa: E402
from stdio_common import FINAL, V7, WORK  # noqa: E402

OUT = f"{FINAL}/gpu_candidates"
MAXREFS = 16


def main():
    os.makedirs(OUT, exist_ok=True)
    src_rows = [json.loads(l) for l in open(f"{V7}/input/code-train/dolci-code-train.jsonl")]
    ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
    v6 = {i["custom_id"]: (i, r) for i, r in zip(ids, src_rows)}
    drops = {r["custom_id"]: r["reason"] for r in csv.DictReader(open(f"{FINAL}/dropped.csv"))}
    final_ids = {json.loads(l)["metadata"]["custom_id"] for l in open(f"{FINAL}/train.jsonl")}
    S1 = pickle.load(open(f"{WORK}/assemble1.pkl", "rb"))
    P2 = pickle.load(open(f"{WORK}/phase2.pkl", "rb"))
    small = pickle.load(open(f"{WORK}/stdio_rows_small.pkl", "rb"))
    by_cid = {v["custom_id"]: v for v in small.values()}
    A = pickle.load(open(f"{V7}/stdio/work/analysis.pkl", "rb"))
    pools = pickle.load(open(f"{V7}/stdio/work/pool.pkl", "rb"))
    R = pickle.load(open(f"{V7}/stdio/work/results.pkl", "rb"))
    rbr = collections.defaultdict(dict)
    for (r, j, k), res in R.items():
        rbr[r][(k, j)] = res
    del R
    CL = pickle.load(open(f"{V7}/stdio/work/dup_clusters.pkl", "rb"))["cluster"]
    final_clusters = {CL[by_cid[c]["row"]] for c in final_ids if c in by_cid}
    audit = {a["custom_id"]: a for a in pickle.load(open("/tmp/v7fn/audit.pkl", "rb"))}
    dec_st = pickle.load(open(f"{WORK}/stdio/decisions.pkl", "rb"))

    def tests_of(cid):
        for pool in (S1["rows"], {k: v[0] for k, v in S1["drops"].items()}):
            if cid in pool:
                return pool[cid]["tests"]
        if cid in P2["cands"]:
            return P2["cands"][cid]["tests"]
        return json.loads(v6[cid][1]["label"])

    def stdio_refs(cid):
        v = by_cid[cid]
        segr = seg_results(pools.get(v["row"], []), rbr.get(v["row"], {}))
        out = []
        for j, r in enumerate(v["refs"]):
            sr = segr.get(j)
            out.append(dict(origin=r["origin"], code=r["code"],
                            passes_existing_tests=(all_ok(sr.get("orig")) if sr and "orig" in sr else None)))
        out.sort(key=lambda x: (x["passes_existing_tests"] is not True, x["origin"] != "v6"))
        return out[:MAXREFS]

    def fn_refs(cid):
        a = audit.get(cid)
        if a is None:
            return []
        out = []
        for i, x in enumerate(a["refs"]):
            if x["prog"] != "ok":
                continue
            out.append(dict(origin=x["kind"], code=a["ref_codes"][i], passes_existing_tests=set(x["v"]) == {"1"}))
        uniq, seen = [], set()
        for x in sorted(out, key=lambda x: not x["passes_existing_tests"]):
            if x["code"] not in seen:
                seen.add(x["code"])
                uniq.append(x)
        return uniq[:MAXREFS]

    def base(cid, reason, route):
        i, r = v6[cid]
        return dict(custom_id=cid, src=i["src"], route=route, reason=reason, prompt=r["prompt"],
                    olmo_passrate=i["olmo_passrate"], tests=tests_of(cid))

    lists = collections.defaultdict(list)
    for cid, reason in drops.items():
        i, r = v6[cid]
        rs = reason.split(":")[0].split(" (")[0]
        cur = tests_of(cid)
        stdio = bool(cur) and isinstance(cur[0], dict)
        if rs == "S3_multi_answer_no_checker" or reason.startswith("S_label_differs_from_statement_sample:refs_follow_label"):
            v = by_cid[cid]
            b = base(cid, reason, "stdio")
            refs = stdio_refs(cid)
            nuniq = len({norm_in(t["input"]) for t in b["tests"] if isinstance(t, dict)})
            match = any(x["passes_existing_tests"] for x in refs)
            dup = CL.get(v["row"]) in final_clusters
            b.update(refs=refs, n_unique_inputs=nuniq, multi_phrase=(A[v["row"]]["multi_hits"] or [""])[0][:200],
                     ref_matches_label=match, duplicate_of_final_row=dup,
                     priority=nuniq >= 5 and len(v["refs"]) >= 2 and (match or float(i["olmo_passrate"] or 0) > 0) and not dup)
            lists["checker_stdio_multi_answer"].append(b)
        elif rs in ("S_lt3_tests_unaugmentable", "S_lt3_tests") or (rs == "P6_lt5_tests" and stdio):
            b = base(cid, reason, "stdio")
            refs = stdio_refs(cid)
            passing = sum(bool(x["passes_existing_tests"]) for x in refs)
            if rs == "P6_lt5_tests":
                passing = max(passing, 1)
            b.update(refs=refs, n_refs_passing_existing=passing,
                     priority=passing >= 1 and len(by_cid[cid]["refs"]) >= 3
                     and CL.get(by_cid[cid]["row"]) not in final_clusters)
            lists["testgen_stdio"].append(b)
        elif rs in ("F3_constant_passes_all", "F_no_tests_left_after_cleaning") or (rs == "P6_lt5_tests" and not stdio):
            b = base(cid, reason, "function")
            refs = fn_refs(cid) if cid in audit else stdio_refs(cid)
            distinct = len({x["code"] for x in refs if x["passes_existing_tests"]})
            if rs == "P6_lt5_tests":
                distinct = max(distinct, 1)
            b.update(refs=refs, n_distinct_refs_passing_existing=distinct, priority=distinct >= 2)
            lists["testgen_function"].append(b)
        elif rs.startswith("P5_unverified") or rs == "F_multi_answer_no_checker_unverified" or rs == "F_unverified_hidden_state":
            route = "stdio" if stdio else "function"
            b = base(cid, reason, route)
            b.update(refs=(fn_refs(cid) if cid in audit else stdio_refs(cid) if cid in by_cid else []),
                     priority=rs.startswith("P5_unverified") and "hidden_state" not in rs)
            lists["teacher_solve_unverified"].append(b)
    summary = {}
    for name, L in lists.items():
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for x in L:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary[name] = dict(rows=len(L), priority=sum(bool(x["priority"]) for x in L),
                             by_src=dict(collections.Counter(x["src"] for x in L)),
                             priority_by_src=dict(collections.Counter(x["src"] for x in L if x["priority"])),
                             by_reason=dict(collections.Counter(x["reason"].split(":")[0].split(" (")[0] for x in L)))
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
