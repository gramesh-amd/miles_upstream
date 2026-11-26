"""Stage 3: prompt dedup across rows, then write fixed.jsonl / dropped.csv / stats.json.

    python3 finalize.py
"""
import collections
import csv
import json
import os
import pickle
import re

import testlib as tl

WORK = "/tmp/v7fn"
OUT = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function"


def norm_prompt(p):
    return re.sub(r"\s+", " ", p.lower()).strip()


def shingles(p, n=5):
    w = re.findall(r"[a-z0-9_]+", p.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def fix_codes(d, a):
    f = set()
    log = a["prep"]["log"]
    if log.get("dup_exact") or log.get("dup_ast"):
        f.add("dedup_tests")
    if log.get("syntax_error"):
        f.add("drop_syntax_error")
    if log.get("stdlib_alias_tests"):
        f.add("stdlib_alias")
    if log.get("setup_merged_away") or any("dep_merged" in i["fix"] for i in d.get("test_info", [])):
        f.add("dep_merged")
    for k in ("drop_dep_unresolved", "drop_dep_state", "drop_vacuous", "drop_noassert", "prune_stub_tests", "drop_broken_test"):
        if d["fixes"].get(k):
            f.add(k)
    if d.get("float_tests"):
        f.add("float_tol")
    if d.get("order_tests"):
        f.add("any_order")
    if d.get("sig_added"):
        f.add("add_signature")
    return sorted(f)


if __name__ == "__main__":
    rows = {r["row"]: r for r in pickle.load(open(f"{WORK}/rows.pkl", "rb"))}
    audit = {a["row"]: a for a in pickle.load(open(f"{WORK}/audit.pkl", "rb"))}
    D = pickle.load(open(f"{WORK}/decide.pkl", "rb"))
    # rows whose recorded reference failed through the grader's own route (verify_grader.py), re-checked
    # with the other references (recheck_grader_failures.py): switch reference, or demote to unverified
    if os.path.exists(f"{WORK}/recheck_fail.pkl"):
        rf = pickle.load(open(f"{WORK}/recheck_fail.pkl", "rb"))
        for d in D:
            if d["row"] in rf:
                good, _ = rf[d["row"]]
                if good is not None:
                    d["verify_ref"] = good
                else:
                    d["verified"] = False
                    d["nondeterministic"] = True
    dropped = []
    kept = []
    for d in D:
        if d["drop"]:
            dropped.append((d, d["drop"]))
            continue
        sig = d.get("sig")
        if sig is not None:
            if not sig.get("arity_ok") or any(x["kind"] == "name" for x in sig["details"]):
                dropped.append((d, "interface_missing_unrecoverable"))
                continue
            d["sig_added"] = True
        if d.get("hidden_state") and not d["verified"]:
            dropped.append((d, "unverified_hidden_state"))
            continue
        if d["multi_answer"] and not d["verified"] and not d.get("order_tests"):
            dropped.append((d, "multi_answer_no_checker_unverified"))
            continue
        kept.append(d)

    # prompt dedup: exact after whitespace/case normalisation, then near-dup (same tested names, 5-gram J >= .85)
    def rank(d):
        return (not d["verified"], -len(d["tests_final"]), d["stub_floor_post"], d["row"])

    groups = collections.defaultdict(list)
    for d in kept:
        groups[norm_prompt(rows[d["row"]]["prompt"])].append(d)
    survivors = []
    for g in groups.values():
        g.sort(key=rank)
        survivors.append(g[0])
        for x in g[1:]:
            dropped.append((x, "dup_prompt_exact"))
            x["dup_of"] = g[0]["custom_id"]
    byfn = collections.defaultdict(list)
    for d in survivors:
        fns = set()
        for t in d["tests_final"]:
            tr = tl.parse(t)
            if tr is not None:
                fns |= set(tl.called_free(tr))
        byfn[frozenset(fns)].append(d)
    gone = set()
    near_pairs = []
    for key, g in byfn.items():
        if len(g) < 2:
            continue
        g.sort(key=rank)
        sh = [shingles(rows[d["row"]]["prompt"]) for d in g]
        for i in range(len(g)):
            if g[i]["row"] in gone:
                continue
            for j in range(i + 1, len(g)):
                if g[j]["row"] in gone:
                    continue
                inter = len(sh[i] & sh[j])
                jac = inter / max(1, len(sh[i] | sh[j]))
                if jac >= 0.85:
                    gone.add(g[j]["row"])
                    g[j]["dup_of"] = g[i]["custom_id"]
                    near_pairs.append((g[i]["custom_id"], g[j]["custom_id"], round(jac, 3)))
    final = []
    for d in survivors:
        if d["row"] in gone:
            dropped.append((d, "dup_prompt_near"))
        else:
            final.append(d)
    final.sort(key=lambda d: d["row"])

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/fixed.jsonl", "w") as f:
        for d in final:
            r = rows[d["row"]]
            a = audit[d["row"]]
            prompt = r["prompt"]
            if d.get("sig_added"):
                prompt = prompt.rstrip() + "\n\n" + d["sig"]["sentence"]
            meta = dict(r["metadata"])
            meta.update(n_tests=len(d["tests_final"]), custom_id=r["custom_id"], src=r["src"],
                        fixes=fix_codes(d, a), verified=bool(d["verified"]),
                        flags={"example_tests": d["n_example_tests"],
                               "meaningful_tests_if_examples_held_out": d["n_meaningful_heldout"],
                               "multi_answer_statement": bool(d["multi_answer"]),
                               "stub_floor": round(d["stub_floor_post"], 3),
                               "nondeterministic_tests": bool(d.get("nondeterministic")),
                               "random_task": bool(re.search(r"\brandom", r["prompt"], re.I))})
            if d["fixes"].get("drop_broken_test"):
                meta["broken_test_evidence"] = d["evidence"]["blocking"]
            if not d["verified"]:
                meta["unverified_pattern"] = d.get("fail_pattern")
            f.write(json.dumps({"prompt": [{"role": "user", "content": prompt}],
                                "label": json.dumps(d["tests_final"], ensure_ascii=False),
                                "metadata": meta}, ensure_ascii=False) + "\n")
    with open(f"{OUT}/dropped.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["custom_id", "src", "reason", "detail"])
        for d, why in sorted(dropped, key=lambda x: x[0]["row"]):
            det = d.get("dup_of", "")
            if why == "constant_passes_all":
                det = d.get("stub_best", "")
            w.writerow([d["custom_id"], d["src"], why, det])
    pickle.dump({"final": final, "dropped": dropped, "near_pairs": near_pairs}, open(f"{WORK}/final.pkl", "wb"))
    c = collections.Counter()
    for d, why in dropped:
        c[(d["src"], "drop:" + why)] += 1
    for d in final:
        c[(d["src"], "kept")] += 1
        c[(d["src"], "kept_verified")] += bool(d["verified"])
    for k, v in sorted(c.items()):
        print(k, v)
    print("near-dup pairs", len(near_pairs))
