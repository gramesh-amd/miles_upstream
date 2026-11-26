"""Before/after numbers for REPORT.md -> stats.json, rewrites.csv, unverified.csv."""
import ast
import collections
import csv
import json
import pickle
import re

import testlib as tl
from decide import CONSTS, word_in

WORK = "/tmp/v7fn"
OUT = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function"

rows = {r["row"]: r for r in pickle.load(open(f"{WORK}/rows.pkl", "rb"))}
A = {a["row"]: a for a in pickle.load(open(f"{WORK}/audit.pkl", "rb"))}
D = {d["row"]: d for d in pickle.load(open(f"{WORK}/decide.pkl", "rb"))}
F = pickle.load(open(f"{WORK}/final.pkl", "rb"))
final = {d["row"]: d for d in F["final"]}
dropped = {d["row"]: why for d, why in F["dropped"]}
S = collections.defaultdict(collections.Counter)
ast_Try, ast_Pass, ast_walk = ast.Try, ast.Pass, ast.walk
EXIT_RE = re.compile(r"\b(sys\.exit|os\._exit|exit\(|quit\()")

for row, r in rows.items():
    s = r["src"]
    a = A[row]
    d = D[row]
    c = S[s]
    c["rows_before"] += 1
    c["tests_before"] += len(r["tests"])
    oi = list(dict.fromkeys(a["orig_ix"]))
    T = a["tests"]
    okv = [x["v"] for x in a["refs"] if x["prog"] == "ok"]
    # 5. verification before (any reference passes every original test)
    c["verified_before"] += any(all(v[j] == "1" for j in oi) for v in okv)
    c["has_runnable_ref"] += bool(okv)
    # 2. stub floor before
    fl = max(sum(a["stubs"][k][j] == "1" for j in a["orig_ix"]) / len(a["orig_ix"]) for k in CONSTS)
    c["stub_ge50_before"] += fl >= 0.5
    c["stub_full_before"] += fl >= 1.0
    # never solved (v6 logged rewards, rows that were in v6)
    if r["in_v6"]:
        c["in_v6"] += 1
        c["v6_never_solved"] += not any(x >= 0.999 for x in r["v6_rewards"])
    c["olmo_pass0"] += (r["olmo_passrate"] or 0) == 0
    # 1. interface name missing (original tests)
    fns = []
    for t in r["tests"]:
        tr = tl.parse(t)
        if tr is not None:
            fns += [f for f in tl.free_names(tr) if f not in tl.MODULES and f not in tl.SYMBOLS]
    fns = list(dict.fromkeys(fns))
    miss = [f for f in fns if not word_in(f, r["prompt"])]
    c["name_missing_before"] += bool(miss)
    # 4. duplicates / dependencies
    lg = a["prep"]["log"]
    c["rows_dup_tests"] += bool(lg.get("dup_exact") or lg.get("dup_ast"))
    c["dup_tests_removed"] += lg.get("dup_exact", 0) + lg.get("dup_ast", 0)
    c["rows_dep_merged"] += bool(lg.get("dep_merged"))
    c["rows_dep_unresolved"] += bool(lg.get("dep_unresolvable"))
    c["rows_dep_state"] += bool(d.get("fixes", {}).get("drop_dep_state"))
    c["rows_stdlib_alias"] += bool(lg.get("stdlib_alias_tests"))
    c["syntax_error_tests"] += lg.get("syntax_error", 0)
    # 8. hazards on original tests
    hz = collections.Counter()
    for j in oi:
        t = T[j]
        tr = tl.parse(t)
        if tr is None:
            continue
        if not tl.has_assert(tr):
            hz["no_assert"] += 1
        if a["stubs"]["empty_program"][j] == "1":
            hz["passes_empty_program"] += 1
        if all(a["stubs"][k][j] == "1" for k in CONSTS):
            hz["passes_every_constant"] += 1
        if EXIT_RE.search(t):
            hz["exit_in_test"] += 1
        if any(isinstance(n, ast_Try) and any(all(isinstance(b, ast_Pass) for b in h.body) for h in n.handlers)
               for n in ast_walk(tr)):
            hz["try_except_pass"] += 1
    for k, v in hz.items():
        c["haz_tests_" + k] += v
        c["haz_rows_" + k] += 1
    c["refs_exit0"] += sum(x["prog"] in ("exit0", "exit0_hard") for x in a["refs"])
    c["refs_total"] += len(a["refs"])
    c["rows_with_exit0_ref"] += any(x["prog"] in ("exit0", "exit0_hard") for x in a["refs"])
    # outcomes
    if row in final:
        f = final[row]
        c["kept"] += 1
        c["tests_after"] += len(f["tests_final"])
        c["verified_after"] += bool(f["verified"])
        c["verified_by_rename"] += bool(f.get("verified_by_rename"))
        c["nondeterministic"] += bool(f.get("nondeterministic"))
        c["stub_ge50_after"] += f["stub_floor_post"] >= 0.5
        c["sig_added"] += bool(f.get("sig_added"))
        c["sig_added_verified"] += bool(f.get("sig_added")) and bool(f["verified"])
        c["example_rows"] += f["n_example_tests"] > 0
        c["example_tests"] += f["n_example_tests"]
        c["example_heldout_lt3"] += f["n_example_tests"] > 0 and f["n_meaningful_heldout"] < 3
        c["tests_if_examples_held_out"] += len(f["tests_final"]) - f["n_example_tests"]
        c["float_rows"] += f["float_tests"] > 0
        c["float_tests"] += f["float_tests"]
        c["float_needed_rows"] += f["float_needed"] > 0
        c["order_rows"] += f["order_tests"] > 0
        c["order_needed_rows"] += f["order_needed"] > 0
        c["pruned_rows"] += bool(f["fixes"].get("prune_stub_tests"))
        c["pruned_tests"] += f["fixes"].get("prune_stub_tests", 0)
        c["vacuous_rows"] += bool(f["fixes"].get("drop_vacuous"))
        c["vacuous_tests"] += f["fixes"].get("drop_vacuous", 0)
        c["noassert_tests_dropped"] += f["fixes"].get("drop_noassert", 0)
        c["broken_test_dropped"] += f["fixes"].get("drop_broken_test", 0)
        c["dep_state_tests_dropped"] += f["fixes"].get("drop_dep_state", 0)
        c["multi_answer_kept"] += bool(f["multi_answer"])
        c["random_task_kept"] += bool(re.search(r"\brandom", r["prompt"], re.I))
        if not f["verified"]:
            c["never_solved_after"] += 1
        c["olmo_pass0_after"] += (r["olmo_passrate"] or 0) == 0
        if r["in_v6"]:
            c["in_v6_after"] += 1
            c["v6_never_solved_after_logged"] += not any(x >= 0.999 for x in r["v6_rewards"])
    else:
        c["dropped"] += 1
        c["drop:" + dropped.get(row, "?")] += 1

out = {s: dict(c) for s, c in S.items()}
json.dump(out, open(f"{OUT}/stats.json", "w"), indent=1, sort_keys=True)

with open(f"{OUT}/rewrites.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["custom_id", "src", "kind", "needed_by_a_reference", "original_test", "rewritten_test"])
    for row, f in sorted(final.items()):
        a = A[row]
        for new, info in zip(f["tests_final"], f["test_info"]):
            for kind, key in (("float_tol", "float_needed"), ("any_order", "order_needed")):
                if kind in info["fix"]:
                    tail = new.split("\n")[-1]
                    w.writerow([f["custom_id"], f["src"], kind, bool(info.get(key)), a["tests"][info["base"]][:300],
                                tail[:300]])

with open(f"{OUT}/unverified.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["custom_id", "src", "n_tests", "n_refs_runnable", "best_ref_pass_frac", "best_ref_failing_tests",
                "tests_no_ref_passes", "blocking_candidate", "hand_verdict", "nondeterministic"])
    for row, f in sorted(final.items()):
        if f["verified"]:
            continue
        fp = f.get("fail_pattern") or {}
        b = f["evidence"].get("blocking") or {}
        w.writerow([f["custom_id"], f["src"], len(f["tests_final"]), f.get("n_refs_ok", 0), round(fp.get("best_frac", 0), 3),
                    fp.get("n_block_tests_best", ""), fp.get("tests_no_ref_passes", ""), bool(b),
                    b.get("hand_verdict") or "", bool(f.get("nondeterministic"))])
for s, c in sorted(S.items()):
    print(s, json.dumps(dict(sorted(c.items()))))
