"""Stage 3 (no execution): one candidate per Codeforces problem, and the static dedup against final_v1.

Copies of one CF problem (Klear and S2 copies, image and multi-answer copies) become one group:
  * prompt: a multi-answer copy's original prompt if there is one (its statement has no image), else the
    image copy with the most own tests (its Open-R1-swapped prompt). That copy's custom_id is the row id;
    the other copies are `duplicate_of_recovered`;
  * tests: the winner's cleaned tests, then each other copy's tests with new inputs (a copy whose shared input
    has a different expected output is skipped, final_v1 merge rule; for checker rows only new inputs are taken),
    then Open-R1 official tests with new inputs, up to 32 tests, each under 1 MB;
  * references: every copy's references plus accepted CF Python submissions.
Dedup against final_v1 (the copy with more tests is kept; a tie keeps final_v1):
  * same CF id (decontam/date_check.csv ref_id of the final_v1 row), or the same v7 S7 duplicate cluster, or
  * MinHash 5-gram Jaccard >= 0.7 after LaTeX normalisation with equal digit sequences (assemble.minhash_pairs).

    python3 groups.py  ->  /tmp/v7v2/groups.pkl
"""
import collections
import csv
import json
import pickle
import re

from v2common import FINAL1, MAX_TEST_BYTES, MAX_TESTS_TOTAL, V7, WORK, load_server_module
from analyze import norm_in  # noqa: E402
from assemble import latex_norm, minhash_pairs  # noqa: E402

ces = load_server_module()
PRI = {"v6": 0, "olmo": 1, "cf_ok": 2, "v6near": 3}


def same_out(a, b, ci):
    return ces._lines_equal_with_decimal(a, b, None, True) if ci else ces._lines_equal_with_decimal(a, b)


def main():
    R = pickle.load(open(f"{WORK}/cands.pkl", "rb"))
    DC = pickle.load(open(f"{WORK}/decontam.pkl", "rb"))
    fin_cl = pickle.load(open(f"{WORK}/cand_rows.pkl", "rb"))["fin_cl"]
    live = {c: r for c, r in R.items() if not r["drop"] and not DC[c]["hits"]}
    by_pid = collections.defaultdict(list)
    for c, r in live.items():
        by_pid[r["pid"]].append(r)
    groups = {}
    for pid, copies in by_pid.items():
        copies.sort(key=lambda r: (r["task"] != "checker", -len(r["tests"]), r["cid"]))
        win = copies[0]
        route = win["route"]
        ci = bool(win["opts"].get("case_insensitive")) or any(r["opts"].get("case_insensitive") for r in copies)
        tests = list(win["tests"])
        seen = {norm_in(t["input"]): t["output"] for t in tests}
        merged_from, skipped = {}, {}
        for r in copies[1:]:
            local, conflict = [], False
            for t in r["tests"]:
                k = norm_in(t["input"])
                if k in seen:
                    if route == "stdio" and not same_out(t["output"], seen[k], True):
                        conflict = True
                        break
                    continue
                local.append(t)
            if conflict:
                skipped[r["cid"]] = "conflicting_output"
                continue
            room = MAX_TESTS_TOTAL - len(tests)
            local = local[:max(room, 0)]
            for t in local:
                seen[norm_in(t["input"])] = t["output"]
            if local:
                merged_from[r["cid"]] = len(local)
                tests += local
        n_win_own = len(win["tests"])
        official = []
        for t in win["official_added"] + [t for r in copies[1:] for t in r["official_added"]]:
            if len(tests) + len(official) >= MAX_TESTS_TOTAL:
                break
            k = norm_in(t["input"])
            if k in seen or len(t["input"]) + len(t["output"]) > MAX_TEST_BYTES:
                continue
            seen[k] = t["output"]
            official.append(t)
        # yes/no written in more than one case across the merged tests -> the grader's YES/NO folding
        yn = {tok for t in tests + official for tok in str(t["output"]).split() if tok.lower() in ("yes", "no")}
        if len({x.lower() for x in yn}) < len(yn):
            ci = True
        opts = dict(win["opts"])
        if route == "stdio" and ci:
            opts["case_insensitive"] = True
        refs, seen_code = [], set()
        for r in copies:
            for x in r["refs"]:
                key = " ".join(x["code"].split())
                if key not in seen_code:
                    seen_code.add(key)
                    refs.append(dict(x, from_cid=r["cid"]))
        refs.sort(key=lambda x: (PRI.get(x["origin"], 2), -(x.get("reward") or 0)))
        groups[pid] = dict(pid=pid, cid=win["cid"], task=win["task"], src=win["src"], route=route, prompt=win["prompt"],
                           checker=win["checker"], opts=opts, own_tests=tests, n_win_own=n_win_own,
                           official=official, merged_from=merged_from, merge_skipped=skipped,
                           copies=[r["cid"] for r in copies], refs=refs, cf_wa=win["cf_wa"], cf_ok=win["cf_ok"],
                           date=win["date"], title=win["title"], rating=win["rating"], metadata=win["metadata"],
                           s7_clusters={r["s7_cluster"] for r in copies}, swap=win.get("swap"),
                           image_copy=any(r["task"] == "statement" for r in copies),
                           tasks=sorted({r["task"] for r in copies}))
    # ---- static dedup against final_v1
    dc = {r["custom_id"]: r for r in csv.DictReader(open(f"{V7}/decontam/date_check.csv"))}
    fin = [json.loads(l) for l in open(f"{FINAL1}/train.jsonl")]
    fin_by_pid = collections.defaultdict(list)
    for r in fin:
        c = r["metadata"]["custom_id"]
        if dc.get(c, {}).get("ref") == "cf":
            fin_by_pid[dc[c]["ref_id"]].append(c)
    fin_ntests = {r["metadata"]["custom_id"]: r["metadata"]["n_tests"] for r in fin}
    pids = sorted(groups)
    texts = [latex_norm(r["prompt"][0]["content"]) for r in fin] + [latex_norm(groups[p]["prompt"]) for p in pids]
    digits = ["".join(re.findall(r"\d", t)) for t in texts]
    nf = len(fin)
    mh = collections.defaultdict(list)
    for i, j, jac in minhash_pairs(texts):
        a, b = min(i, j), max(i, j)
        if a < nf <= b and digits[a] == digits[b]:
            mh[pids[b - nf]].append((fin[a]["metadata"]["custom_id"], round(jac, 3)))
        elif a >= nf and b >= nf and digits[a] == digits[b]:
            print("minhash pair inside recovered (different CF ids):", pids[a - nf], pids[b - nf], round(jac, 3))
    for p, g in groups.items():
        dup = set(fin_by_pid.get(p, []))
        dup |= {fin_cl[k] for k in g["s7_clusters"] if k in fin_cl}
        dup |= {c for c, _ in mh.get(p, [])}
        g["v1_dups"] = sorted(dup)
        g["v1_dup_how"] = dict(cf_id=sorted(fin_by_pid.get(p, [])),
                               s7=sorted(fin_cl[k] for k in g["s7_clusters"] if k in fin_cl), minhash=mh.get(p, []))
        g["v1_dup_max_tests"] = max((fin_ntests[c] for c in dup), default=0)
    pickle.dump(groups, open(f"{WORK}/groups.pkl", "wb"))
    c = collections.Counter((g["route"], bool(g["v1_dups"])) for g in groups.values())
    print("groups", len(groups), "from rows", len(live), dict(c))
    print("v1 dup by", collections.Counter(k for g in groups.values() for k, v in g["v1_dup_how"].items() if v))
    print("n tests (own+merged, official)", collections.Counter(min(len(g["own_tests"]) + len(g["official"]), 33) // 4 * 4
                                                                for g in groups.values()))
    print("merged copies", sum(len(g["merged_from"]) for g in groups.values()), "skipped", sum(len(g["merge_skipped"]) for g in groups.values()))


if __name__ == "__main__":
    main()
