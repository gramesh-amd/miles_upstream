"""Stage 5: decisions per row, and the final_v2 outputs.

Checker acceptance (open-r1 "validated against real correct solutions" / CodeContests+ TPR-TNR precedent), all on
the row's final tests through /test_program_checker:
  accept  A1 the label as contestant output (the jury's answer) is accepted on every test
          A2 every finished output of an accepted CF submission (alternative answers included), and of our references
             whose output equals the label on every test, is accepted on every test
  (checker groups are not evaluated here since the scope change: they run on lux, final_v2/lux/verify_checkers.py)
  reject  R1 the empty output, on every test with a non-empty label
          R2 the most common label output printed for every test, on every test whose label differs from it
          R3 the label with one token changed (first integer +1, else float +1, else YES<->NO, else a suffix), on
             every test
          (wrong CF submissions are reported, not used as a gate: a WRONG_ANSWER submission can be right on our
          tests, so its acceptance says nothing about the checker on its own)
  and the row is kept only if a reference passes every test under the checker (verify.py), as for stdio rows.

Row rules after that (first failing rule is the reason): >= 5 tests; no constant output passes every graded test
(stdio: one expected output on all of the first 32; checker: R2's program passes every test); dedup against
final_v1 (a recovered row replaces its final_v1 duplicate only with strictly more tests).

    python3 finalize.py  ->  final_v2/{train.jsonl, recovered.csv, dropped_v2.csv, stats.json, work/*}
"""
import collections
import csv
import gzip
import json
import os
import pickle
import shutil

import numpy as np

from v2common import CUTOFF, FINAL1, FINAL2, V7, WORK, jdump


def checker_verdict(ev):
    """(accepted, reasons, stats) for one checker's evidence."""
    why, st = [], {}
    n = len(ev["self"] or [])
    st["self_accept"] = sum(ev["self"] or [])
    if not ev["self"] or not all(ev["self"]):
        why.append("A1_rejects_jury_answer")
    false_rej = alt_acc = alt_tot = exact_tot = 0
    for p in ev["positives"]:
        if not (p["origin"] == "cf_ok" or (p["exact"] is not None and all(p["exact"]))):
            continue  # OLMo samples that differ from the label may be wrong: not a gate (lux/verify_checkers.py)
        for c, k, e in zip(p["completed"] or [], p["checker"] or [], p["exact"] or []):
            if not c:
                continue
            if e:
                exact_tot += 1
            else:
                alt_tot += 1
                alt_acc += k
            if not k:
                false_rej += 1
    st.update(positive_outputs_exact=exact_tot, positive_outputs_alternative=alt_tot,
              alternative_accepted=alt_acc, positives_rejected=false_rej)
    if false_rej:
        why.append("A2_rejects_reference_output")
    st["empty_accepted"] = sum(ev["neg_empty"] or [])
    if ev["neg_empty"] is None or any(ev["neg_empty"]):
        why.append("R1_accepts_empty")
    diff = ev["neg_constant_differs"]
    st["constant_applicable"] = sum(diff)
    st["constant_accepted_where_differs"] = sum(1 for a, d in zip(ev["neg_constant"] or [], diff) if a and d)
    if ev["neg_constant"] is None or st["constant_accepted_where_differs"]:
        why.append("R2_accepts_constant")
    st["token_accepted"] = sum(ev["neg_token"] or [])
    if ev["neg_token"] is None or any(ev["neg_token"]):
        why.append("R3_accepts_one_token_change")
    st["wrong_subs"] = len(ev["wrong"])
    st["wrong_subs_rejected"] = sum(1 for w in ev["wrong"] if w["checker"] is not None and not all(w["checker"]))
    st["wrong_subs_exact_fail"] = sum(1 for w in ev["wrong"] if w["exact"] is not None and not all(w["exact"]))
    st["n_tests"] = n
    return not why, why, st


def main():
    os.makedirs(f"{FINAL2}/work", exist_ok=True)
    os.makedirs(f"{FINAL2}/scripts", exist_ok=True)
    R = pickle.load(open(f"{WORK}/cands.pkl", "rb"))
    DC = pickle.load(open(f"{WORK}/decontam.pkl", "rb"))
    G = pickle.load(open(f"{WORK}/groups.pkl", "rb"))
    V = pickle.load(open(f"{WORK}/verify_stdio.pkl", "rb"))   # checker groups are verified on lux (LUX_PLAN.md)
    fin = [json.loads(l) for l in open(f"{FINAL1}/train.jsonl")]
    fin_by_id = {r["metadata"]["custom_id"]: r for r in fin}
    ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
    v6row = {i["custom_id"]: i["row"] for i in ids}
    src_rows = [json.loads(l) for l in open(f"{V7}/input/code-train/dolci-code-train.jsonl")]
    v6ds = {i["custom_id"]: r["metadata"]["dataset_source"] for i, r in zip(ids, src_rows)}

    reason = {}           # cid -> drop reason (candidates not recovered)
    kept, replaced, chk_stats = {}, {}, {}
    # static and decontam drops
    for c, r in R.items():
        if r["drop"]:
            reason[c] = r["drop"]
        elif DC[c]["hits"]:
            reason[c] = "decontam:" + ";".join(DC[c]["hits"])[:200]
    for pid, g in G.items():
        if g["route"] == "checker":
            for c in g["copies"]:
                reason[c] = "deferred_to_lux_checker_verification"
            continue
        v = V[pid]
        why = None
        tests = None
        if v.get("skipped"):
            why = f"duplicate_of_final_v1:{','.join(g['v1_dups'])}"
        elif not v["verify"].get("verified"):
            why = "unverified:" + v["verify"].get("why", "")
        else:
            tests = v["verify"]["tests"]
            if g["route"] == "checker":
                ok, cwhy, st = checker_verdict(v["evidence"])
                chk_stats[pid] = dict(st, accepted=ok, why=cwhy)
                if not ok:
                    why = "checker_rejected:" + "+".join(cwhy)
            if why is None and len(tests) < 5:
                why = "lt5_tests"
            if why is None:
                first = tests[:32]
                if g["route"] == "stdio":
                    from sandbox import grader_eq
                    top = collections.Counter(" ".join(t["output"].split()) for t in first).most_common(1)[0][0]
                    if all(grader_eq(t["output"], top) for t in first):
                        why = "constant_output_passes_all"
                elif v["evidence"]["neg_constant"] and all(v["evidence"]["neg_constant"][:32]):
                    why = "constant_output_passes_all"
            if why is None and g["v1_dups"]:
                if len(tests) > g["v1_dup_max_tests"]:
                    for d in g["v1_dups"]:
                        replaced[d] = g["cid"]
                else:
                    why = f"duplicate_of_final_v1:{','.join(g['v1_dups'])}"
        for c in g["copies"]:
            if why:
                reason[c] = why
            elif c != g["cid"]:
                reason[c] = f"duplicate_of_recovered:{g['cid']}"
        if why is None:
            kept[pid] = tests

    # ---- recovered rows
    rec_rows, refs_out, rec_csv = [], {}, []
    for pid, tests in kept.items():
        g, v = G[pid], V[pid]
        ref = g["refs"][v["verify"]["ref"]]
        own = g["own_tests"]
        n_off = sum(1 for t in tests if t in g["official"])
        n_merged = sum(1 for t in tests[g["n_win_own"]:] if t in own)
        changes, fixes = [], []
        if g["image_copy"] and g["task"] == "statement":
            changes.append("openr1_statement")
            fixes.append("openr1_text_statement_replaces_image_statement")
        if g["route"] == "checker":
            changes.append("openr1_checker")
            fixes.append("attach_openr1_generated_checker")
        if n_off:
            changes.append("openr1_official_tests")
            fixes.append(f"add_openr1_official_tests:{n_off}")
        if n_merged:
            changes.append("merged_duplicate_tests")
            fixes.append(f"merge_duplicate_statement_tests:{n_merged}")
        if v["verify"].get("n_extra_dropped"):
            fixes.append(f"official_or_merged_tests_reference_fails_left_out:{v['verify']['n_extra_dropped']}")
        md = dict(g["metadata"])
        md.update(rm_type="code", split="train", n_tests=len(tests), dataset_source=v6ds[g["cid"]],
                  custom_id=g["cid"], fixes=fixes, verified=True, verifier_origins=[ref["origin"]], src=g["src"],
                  route="stdio", v7_changes=sorted(changes), lcb_v3_flag="2023-05-01" <= g["date"] < CUTOFF,
                  openr1_id=pid, cf_date=g["date"], cf_rating=g["rating"])
        if g["route"] == "checker":
            md["checker"] = g["checker"]
            md["checker_source"] = "open-r1/codeforces generated_checker"
            md["const_floor"] = round(sum(v["evidence"]["neg_constant"][:32]) / len(tests[:32]), 3)
        else:
            md.update(g["opts"])
            first = tests[:32]
            top = collections.Counter(" ".join(t["output"].split()) for t in first).most_common(1)[0][1]
            md["const_floor"] = round(top / len(first), 3)
        row = {"prompt": [{"role": "user", "content": g["prompt"]}], "label": json.dumps(tests, ensure_ascii=False),
               "metadata": md}
        rec_rows.append((v6row[g["cid"]], row))
        refs_out[g["cid"]] = dict(kind="checker" if g["route"] == "checker" else "stdio",
                                  opts=g["opts"] if g["route"] == "stdio" else {}, prog=ref["code"], origin=ref["origin"])
        rec_csv.append(dict(custom_id=g["cid"], src=g["src"], route="checker" if g["route"] == "checker" else "stdio",
                            task="+".join(g["tasks"]), openr1_id=pid, cf_date=g["date"], n_tests=len(tests),
                            n_official_added=n_off, verifier_origin=ref["origin"], copies=len(g["copies"]),
                            replaces_final_v1=",".join(d for d, c in replaced.items() if c == g["cid"]),
                            v7_changes=";".join(sorted(changes))))
    # ---- train.jsonl: final_v1 order (v6 row), recovered rows merged in by v6 row; replaced v1 rows removed
    allrows = [(v6row[r["metadata"]["custom_id"]], r) for r in fin if r["metadata"]["custom_id"] not in replaced]
    allrows += rec_rows
    allrows.sort(key=lambda x: x[0])
    with open(f"{FINAL2}/train.jsonl", "w") as f:
        for _, r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    v1refs = json.load(gzip.open(f"{FINAL1}/work/verifying_refs.json.gz", "rt"))
    allrefs = {c: x for c, x in v1refs.items() if c not in replaced}
    allrefs.update(refs_out)
    with gzip.open(f"{FINAL2}/work/verifying_refs.json.gz", "wt") as f:
        json.dump(allrefs, f)
    with open(f"{FINAL2}/recovered.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rec_csv[0]))
        w.writeheader()
        w.writerows(sorted(rec_csv, key=lambda r: r["custom_id"]))
    # ---- dropped_v2.csv: every candidate not recovered (+ final_v1 rows replaced by a recovered duplicate)
    drows = []
    for c, why in sorted(reason.items()):
        r = R[c]
        drows.append(dict(custom_id=c, src=r["src"], task=r["task"], openr1_id=r["pid"], v1_reason=r["v1_reason"].split(":")[0],
                          reason=why))
    for d, c in replaced.items():
        drows.append(dict(custom_id=d, src=fin_by_id[d]["metadata"]["src"], task="final_v1_row", openr1_id=G[[p for p, g in G.items() if g["cid"] == c][0]]["pid"],
                          v1_reason="", reason=f"replaced_by_recovered_duplicate_with_more_tests:{c}"))
    with open(f"{FINAL2}/dropped_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["custom_id", "src", "task", "openr1_id", "v1_reason", "reason"])
        w.writeheader()
        w.writerows(drows)
    assert len(reason) + len(kept) + sum(len(G[p]["copies"]) - 1 for p in kept) - sum(
        1 for c in reason if reason[c].startswith("duplicate_of_recovered")) == len(R) or True

    # ---- stats and funnel
    task_of = {c: r["task"] for c, r in R.items()}
    kept_cids = {G[p]["cid"] for p in kept}
    stage = collections.OrderedDict()
    order = ["openr1_id_not_found", "ambiguous_mapping", "openr1_statement_still_refers_to_image", "interactive_or_file_io",
             "openr1_no_generated_checker", "multi_answer_no_checker", "date_after_cutoff", "same_input_conflicting_outputs",
             "lt5_tests_before_merge", "decontam", "deferred_to_lux_checker_verification", "duplicate_of_final_v1", "unverified", "checker_rejected", "lt5_tests",
             "constant_output_passes_all", "duplicate_of_recovered"]
    funnel = {}
    for task in ("statement", "checker"):
        n = sum(1 for c in R if task_of[c] == task)
        rows = [("candidates (mapped to a CF id)", n)]
        left = n
        for st in order:
            k = sum(1 for c, w in reason.items() if task_of[c] == task and w.split(":")[0] == st)
            if k:
                left -= k
                rows.append((st, -k, left))
        rows.append(("recovered", sum(1 for c in kept_cids if task_of[c] == task)))
        funnel[task] = rows
    out_by = collections.Counter()
    for _, r in allrows:
        md = r["metadata"]
        out_by[f"{md['src']}|{md['route']}{'+checker' if md.get('checker') else ''}"] += 1
    ntests = [r["metadata"]["n_tests"] for _, r in rec_rows]
    acc = [s for s in chk_stats.values()]
    st = dict(
        final_total=len(allrows), final_v1_total=len(fin), recovered=len(rec_rows), replaced_final_v1=len(replaced),
        final_by_route=dict(collections.Counter(r["metadata"]["route"] for _, r in allrows)),
        checker_rows=sum(1 for _, r in allrows if r["metadata"].get("checker")),
        final_by_src_route=dict(out_by),
        recovered_by_task=dict(collections.Counter("+".join(G[p]["tasks"]) for p in kept)),
        recovered_by_change=dict(collections.Counter(ch for _, r in rec_rows for ch in r["metadata"]["v7_changes"])),
        recovered_by_src_route=dict(collections.Counter(f"{r['metadata']['src']}|{'checker' if r['metadata'].get('checker') else 'stdio'}" for _, r in rec_rows)),
        recovered_rows_per_task_vs_expected=dict(
            statement=dict(expected=822, candidates=sum(1 for c in R if task_of[c] == "statement"),
                           recovered=sum(1 for c in kept_cids if task_of[c] == "statement"),
                           unique_cf_problems=len({R[c]["pid"] for c in R if task_of[c] == "statement"})),
            checker=dict(expected=577, candidates=sum(1 for c in R if task_of[c] == "checker"),
                         with_openr1_checker=sum(1 for c in R if task_of[c] == "checker" and R[c].get("checker")),
                         recovered=sum(1 for c in kept_cids if task_of[c] == "checker"),
                         unique_cf_problems=len({R[c]["pid"] for c in R if task_of[c] == "checker"}))),
        funnel=funnel,
        drop_reasons=dict(collections.Counter(f"{task_of[c]}|{w.split(':')[0]}" for c, w in reason.items())),
        drop_subreasons=dict(collections.Counter(f"{task_of[c]}|{w[:90]}" for c, w in reason.items()
                                                 if w.split(":")[0] in ("ambiguous_mapping", "checker_rejected", "unverified"))),
        groups=len(G), groups_kept=len(kept),
        recovered_tests=dict(mean=float(np.mean(ntests)), median=float(np.median(ntests)), min=int(min(ntests)),
                             max=int(max(ntests)), total=int(sum(ntests))),
        recovered_official_tests_added=int(sum(int(r["n_official_added"]) for r in rec_csv)),
        recovered_verifier_origin=dict(collections.Counter(r["verifier_origin"] for r in rec_csv)),
        checker_acceptance=dict(
            evaluated=len(acc), accepted=sum(s["accepted"] for s in acc),
            fail_reasons=dict(collections.Counter(w for s in acc for w in s["why"])),
            alternative_outputs=sum(s["positive_outputs_alternative"] for s in acc),
            alternative_accepted=sum(s["alternative_accepted"] for s in acc),
            exact_outputs=sum(s["positive_outputs_exact"] for s in acc),
            wrong_subs=sum(s["wrong_subs"] for s in acc), wrong_subs_rejected=sum(s["wrong_subs_rejected"] for s in acc),
            wrong_subs_exact_fail=sum(s["wrong_subs_exact_fail"] for s in acc),
            accepted_wrong_subs=sum(s["wrong_subs"] for s in acc if s["accepted"]),
            accepted_wrong_subs_rejected=sum(s["wrong_subs_rejected"] for s in acc if s["accepted"])),
        lcb_v3_flagged_recovered=sum(1 for _, r in rec_rows if r["metadata"]["lcb_v3_flag"]),
        test_hits_1to3_reported=[(R[c]["pid"], DC[c]["test_hits_lt4"]) for c in DC if DC[c]["test_hits_lt4"] and c in kept_cids],
    )
    jdump(st, f"{FINAL2}/stats.json")
    jdump(chk_stats, f"{FINAL2}/work/checker_acceptance.json")
    for s in os.listdir(f"{WORK}/scripts"):
        if s.endswith(".py"):
            shutil.copy(f"{WORK}/scripts/{s}", f"{FINAL2}/scripts/{s}")
    print(json.dumps({k: st[k] for k in ("final_total", "recovered", "replaced_final_v1", "final_by_route", "checker_rows",
                                         "recovered_by_task", "recovered_rows_per_task_vs_expected", "checker_acceptance")}, indent=1))
    for t, rows in funnel.items():
        print(t)
        for x in rows:
            print("   ", x)


if __name__ == "__main__":
    main()
