#!/usr/bin/env python3
"""Per-row decisions from statement analysis + sandbox results. Writes fixed.jsonl, dropped.csv,
multi_answer.csv, stats.json and per-row diagnostics (work/decisions.pkl).

Order: statement drops -> test-format repair -> sample contradictions -> slow tests -> augmentation
(< 3 tests) -> constant floor -> empties -> verification -> dedup (most tests wins).
"""
import ast, collections, csv, json, os, pickle, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from sandbox import grader_eq, case_eq, float_eq
from analyze import norm_in

OUTD = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio"
W = f"{OUTD}/work"
SLOW = 5.0  # grader timeout
NONSTDIN = {"pylist_of_lines", "fn_args_list_repr", "fn_args_wrapped_out", "fn_args_json", "gfg_literal",
            "stdin_but_fn_statement", "empty_input"}
FLOATY = re.compile(r"^-?\d+\.\d+$")


PROOFS = json.load(open(f"{OUTD}/bruteforce/proofs.json"))


def _strlist(s):
    s = str(s).strip()
    if not s.startswith("["):
        return False
    try:
        x = ast.literal_eval(s)
    except Exception:
        return False
    return isinstance(x, list) and all(isinstance(e, str) for e in x)


def clean_sample_output(o):
    """cut explanation prose that the statement parser absorbed after a blank line"""
    lines = o.split("\n")
    for k, l in enumerate(lines):
        if l.strip():
            continue
        nxt = next((x for x in lines[k + 1:] if x.strip()), None)
        if nxt is None:
            break
        if len(re.findall(r"[A-Za-z]{2,}", nxt)) >= 3 or nxt.rstrip().endswith(":"):
            return "\n".join(lines[:k]).strip()
    return o.strip()


def sample_ok(r, out_clean, out_raw):
    if r.get("ex") and out_clean == out_raw.strip():
        return True
    o = r.get("out")
    return o is not None and len(o) < 300 and r.get("rc") == 0 and grader_eq(o, out_clean)


def seg_results(pool, res_by_ref):
    """{ref: {seg_name: [res...]}}"""
    out = collections.defaultdict(dict)
    for (kind, j), res in res_by_ref.items():
        off = 0
        for name, k, tests in [s for s in pool if s[1] == kind]:
            out[j][name] = res[off: off + len(tests)]
            off += len(tests)
    return out


def all_ok(rs):
    return bool(rs) and all(r["ex"] for r in rs)


def expected_of_assert(s):
    m = re.search(r"==\s*(.+?)\s*$", s)
    if m:
        return m.group(1)
    m = re.search(r"-\s*(.+?)\)\s*<=", s)
    return m.group(1) if m else s


def const_floor(kind, tests):
    """share of tests a single constant answer passes, and the dominant group"""
    if not tests:
        return 1.0, []
    if kind == "stdio":
        groups = []
        for i, t in enumerate(tests):
            for g in groups:
                if grader_eq(t["output"], tests[g[0]]["output"]):
                    g.append(i)
                    break
            else:
                groups.append([i])
    else:
        gm = collections.defaultdict(list)
        for i, t in enumerate(tests):
            gm[expected_of_assert(t)].append(i)
        groups = list(gm.values())
    groups.sort(key=len, reverse=True)
    return len(groups[0]) / len(tests), groups


def main():
    vs = pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))
    A = pickle.load(open(f"{W}/analysis.pkl", "rb"))
    pools = pickle.load(open(f"{W}/pool.pkl", "rb"))
    R = pickle.load(open(f"{W}/results.pkl", "rb"))
    D = pickle.load(open(f"{W}/dup_clusters.pkl", "rb"))["cluster"]
    rbr = collections.defaultdict(dict)
    for (r, j, k), res in R.items():
        rbr[r][(k, j)] = res

    S = collections.Counter()           # defect counters, keyed (src, name)
    kept, dropped, multi_rows, diag = {}, [], [], {}

    def drop(v, reason, **kw):
        dropped.append(dict(custom_id=v["custom_id"], row=v["row"], src=v["src"], reason=reason))
        S[(v["src"], "drop:" + reason.split(":")[0])] += 1
        diag[v["row"]] = dict(status="dropped", reason=reason, **kw)

    for v in vs:
        a, src, row = A[v["row"]], v["src"], v["row"]
        pool = pools[row]
        segr = seg_results(pool, rbr.get(row, {}))
        refs = v["refs"]
        S[(src, "rows")] += 1
        fixes = []

        # --- per-ref diagnostics on the trained tests (case/float/ws-only failures) ---
        for j, sr in segr.items():
            o = sr.get("orig")
            if not o or v["kind"] != "stdio":
                continue
            fails = [r for r in o if not r["ex"]]
            if fails and all(r.get("cs") for r in fails):
                S[(src, "refs_fail_only_case")] += 1
                S[(src, "rows_ref_fail_only_case")] += 0
            if fails and all(r.get("fl") for r in fails):
                S[(src, "refs_fail_only_float")] += 1
            if fails and all(r.get("ws") or r.get("tk") for r in fails):
                S[(src, "refs_fail_only_ws")] += 1
        rows_case_only = any(
            (lambda f: f and all(r.get("cs") for r in f))([r for r in sr.get("orig", []) if not r["ex"]])
            for sr in segr.values()) and v["kind"] == "stdio"
        S[(src, "rows_with_case_only_ref_fail")] += rows_case_only
        S[(src, "rows_with_case_only_ref_fail_and_stmt")] += rows_case_only and bool(a["case_stmt"])

        # --- 6. statement-level drops ---
        if a["empty_stmt"]:
            drop(v, "empty_statement"); continue
        if a["non_english"]:
            drop(v, "non_english_statement"); continue
        if a["interactive"]:
            drop(v, "interactive"); continue
        if a["other_lang"]:
            drop(v, "other_language_request:" + a["other_lang"][:40]); continue
        if a["image"]:
            drop(v, "image_placeholder" + ("_tag" if a["image_tags"] else "_phrase")); continue
        # --- 3. multi-answer ---
        if a["multi_hits"]:
            ver = any(all_ok(sr.get("orig")) or all_ok(sr.get("to_stdin")) or all_ok(sr.get("to_assert")) for sr in segr.values()) \
                if segr else "not_executed"
            multi_rows.append(dict(custom_id=v["custom_id"], src=src, phrase=a["multi_hits"][0][:160],
                                   a_ref_passes_exact=ver, n_tests=len(v["label"])))
            drop(v, "multi_answer_no_checker"); continue

        # --- 1. test format ---
        kind, tests, seg = v["kind"], [dict(t) if isinstance(t, dict) else t for t in v["label"]], "orig"
        orig_ok = any(all_ok(sr.get("orig")) for sr in segr.values())
        if a["fmt"] in NONSTDIN:
            S[(src, "nonstdin_rows")] += 1
            S[(src, "nonstdin:" + a["fmt"])] += 1
            if orig_ok:
                S[(src, "nonstdin_kept_as_is_ref_passes")] += 1
            else:
                choice = None
                for name in (("to_stdin", "to_assert") if a["io_fmt"] else ("to_assert", "to_stdin")):
                    if name in a["conv_cands"] and any(all_ok(sr.get(name)) for sr in segr.values()):
                        choice = name
                        break
                if choice is None:
                    why = "no_interface_or_format" if not a["conv_cands"] else "conversion_unverified"
                    drop(v, f"nonstdin_tests_{why}:{a['fmt']}"); continue
                kind, tests = a["conv_cands"][choice]
                tests = list(tests)
                seg = choice
                fixes.append(choice)
                S[(src, "nonstdin_fixed_" + choice)] += 1

        # tests at this point; per-test passing refs
        def ref_vecs(name):
            return {j: sr[name] for j, sr in segr.items() if name in sr}

        vec = ref_vecs(seg)
        n = len(tests)
        keep_idx = list(range(n))
        # statement samples exactly as executed (pool order), with absorbed prose cut off
        samples = [dict(t, clean=clean_sample_output(t["output"])) for t in next((s[2] for s in pool if s[0] == "sample"), [])]

        # brute-force proofs (bruteforce/proofs.json): relabel proven-wrong labels, drop invalid inputs
        if seg == "orig":
            for p in PROOFS.get(v["custom_id"], []):
                ti = p["test"]
                if p["verdict"] == "label_wrong":
                    tests[ti] = dict(tests[ti], output=p["brute"] + "\n")
                    vec = {j: [dict(r) for r in rs] for j, rs in vec.items()}
                    for rs in vec.values():
                        r = rs[ti]
                        r["ex"] = bool(r.get("out") is not None and len(r["out"]) < 300 and r.get("rc") == 0
                                       and grader_eq(r["out"], tests[ti]["output"]))
                    fixes.append("relabel_bruteforce_proven")
                    S[(src, "bruteforce_label_wrong_relabelled")] += 1
                elif p["verdict"] == "invalid_input":
                    keep_idx = [i for i in keep_idx if i != ti]
                    fixes.append("drop_constraint_violating_test")
                    S[(src, "bruteforce_invalid_input_test_dropped")] += 1
                else:
                    S[(src, "bruteforce_label_confirmed")] += 1
            # Python list literals mixed into a stdin test set (input and output both list-of-str reprs)
            if kind == "stdio" and a["fmt"] == "stdin":
                bad = [i for i, t in enumerate(tests) if _strlist(t["input"]) and _strlist(t["output"])]
                if bad:
                    keep_idx = [i for i in keep_idx if i not in bad]
                    fixes.append(f"drop_python_literal_tests:{len(bad)}")
                    S[(src, "rows_with_python_literal_tests_in_stdin_set")] += 1

        # --- 8a. identical tests collapse; same input with conflicting outputs must be resolved ---
        if kind == "stdio":
            by_in = collections.defaultdict(list)
            for i, t in enumerate(tests):
                by_in[norm_in(t["input"])].append(i)
            drop_i, conflict = set(), False
            smp_out = {norm_in(s["input"]): s["clean"] for s in samples}
            for key, ii in by_in.items():
                if len(ii) < 2:
                    continue
                variants = []
                for i in ii:
                    for g in variants:
                        if grader_eq(tests[i]["output"], tests[g[0]]["output"]):
                            g.append(i)
                            break
                    else:
                        variants.append([i])
                for g in variants:
                    drop_i.update(g[1:])  # exact repeats
                if len(variants) > 1:
                    S[(src, "inconsistent_duplicate_input_rows")] += 1
                    others = [k for k in range(n) if k not in ii]
                    good = []
                    for g in variants:
                        i = g[0]
                        refs_ok = [j for j, rs in vec.items() if rs[i]["ex"] and all(rs[k]["ex"] for k in others)]
                        stmt_ok = key in smp_out and grader_eq(tests[i]["output"], smp_out[key])
                        if refs_ok or stmt_ok:
                            good.append(g)
                    if len(good) == 1:
                        drop_i.update(i for g in variants if g is not good[0] for i in g)
                        drop_i.discard(good[0][0])
                        fixes.append("drop_inconsistent_duplicate_input_test")
                        diag.setdefault(row, {})["inconsistent"] = [(tests[g[0]]["output"][:40], g in good) for g in variants]
                    else:
                        conflict = True
            if conflict:
                drop(v, "same_input_conflicting_outputs_unresolved"); continue
            if drop_i:
                n_rep = len(drop_i)
                keep_idx = [i for i in keep_idx if i not in drop_i]
                if "drop_inconsistent_duplicate_input_test" not in fixes:
                    fixes.append(f"dedupe_identical_tests:{n_rep}")
                S[(src, "rows_with_repeated_tests")] += 1

        # --- 2. sample contradictions (stdio only) ---
        if kind == "stdio" and samples:
            svec = ref_vecs("sample")
            bad_tests = []
            for si, smp in enumerate(samples):
                so = smp["clean"]
                if not so:
                    continue
                for ti, t in enumerate(tests):
                    if ti not in keep_idx or norm_in(t["input"]) != norm_in(smp["input"]):
                        continue
                    if grader_eq(t["output"], so):
                        continue
                    if case_eq(t["output"], so) or float_eq(t["output"], so, 1e-4) \
                            or t["output"].split() == so.split():
                        S[(src, "sample_vs_label_format_only")] += 1
                        continue
                    # who is right? refs that pass all *other* tests decide
                    others_ok = [j for j, rs in vec.items() if all(rs[k]["ex"] for k in keep_idx if k != ti)]
                    stmt_side = [j for j in others_ok if svec.get(j) and sample_ok(svec[j][si], so, smp["output"])]
                    label_side = [j for j in others_ok if vec[j][ti]["ex"]]
                    bad_tests.append(dict(ti=ti, si=si, stmt=len(stmt_side), label=len(label_side),
                                          label_out=t["output"][:80], stmt_out=so[:80]))
            if bad_tests:
                S[(src, "sample_contradiction_rows")] += 1
                diag.setdefault(row, {})["contradiction"] = bad_tests
                if all(b["stmt"] > 0 and b["label"] == 0 for b in bad_tests):
                    drop_ti = {b["ti"] for b in bad_tests}
                    keep_idx = [i for i in keep_idx if i not in drop_ti]
                    fixes.append("drop_label_contradicting_statement_sample")
                    S[(src, "sample_contradiction_label_wrong_test_dropped")] += 1
                elif all(b["label"] > 0 and b["stmt"] == 0 for b in bad_tests):
                    # references reproduce the label, not the statement: an unflagged multi-answer
                    # problem (or a statement typo); exact match would reject the statement's own answer
                    drop(v, "label_differs_from_statement_sample:refs_follow_label", contradiction=bad_tests); continue
                else:
                    drop(v, "label_differs_from_statement_sample:unresolved", contradiction=bad_tests); continue

        # --- 9. slow tests: no reference finishes in 5 s although one passes every other test ---
        slow = []
        nofin = [ti for ti in keep_idx
                 if [rs[ti] for rs in vec.values() if ti < len(rs) and rs[ti]["rc"] != "skip"]
                 and all(rs[ti]["rc"] in ("to", "skip") or rs[ti].get("t", 0) > SLOW for rs in vec.values() if ti < len(rs))]
        if nofin:
            S[(src, "rows_with_test_no_reference_finishes_5s")] += 1
            S[(src, "tests_no_reference_finishes_5s")] += len(nofin)
        for ti in list(keep_idx):
            runs = [rs[ti] for rs in vec.values() if ti < len(rs)]
            if not runs:
                continue
            finished = [r for r in runs if r["rc"] not in ("to", "skip") and r.get("t", 0) <= SLOW]
            if finished:
                continue
            if any(r["rc"] in ("to", "skip") or r.get("t", 0) > SLOW for r in runs):
                others_ok = any(all(r["ex"] for k, r in enumerate(rs) if k != ti and k in keep_idx) for rs in vec.values())
                if others_ok:
                    slow.append(ti)
        if slow:
            S[(src, "slow_test_rows")] += 1
            S[(src, "slow_tests")] += len(slow)
            if len(keep_idx) - len(slow) >= 3:
                keep_idx = [i for i in keep_idx if i not in slow]
                fixes.append("drop_tests_no_reference_finishes_5s")
            else:
                drop(v, "huge_tests_no_reference_finishes"); continue

        # --- 8. augmentation when < 3 tests: uncapped Dolci tests, then statement samples ---
        aug = []
        if len(keep_idx) < 3 and kind == "stdio":
            S[(src, "lt3_tests_rows")] += 1
            have = {norm_in(tests[i]["input"]) for i in keep_idx}
            verifiers = [j for j, rs in vec.items() if all(rs[i]["ex"] for i in keep_idx)]
            for name in ("extra", "sample"):
                seg_tests = next((s[2] for s in pool if s[0] == name), [])
                for k, t in enumerate(seg_tests):
                    if len(keep_idx) + len(aug) >= 3 + 2:
                        break
                    if norm_in(t["input"]) in have:
                        continue
                    if name == "sample":
                        ok = samples[k]["clean"] and any(segr[j].get(name) and sample_ok(segr[j][name][k], samples[k]["clean"], t["output"]) for j in verifiers)
                        out_t = samples[k]["clean"]
                    else:
                        ok = any(segr[j].get(name) and segr[j][name][k]["ex"] for j in verifiers)
                        out_t = t["output"]
                    if ok:
                        aug.append(dict(input=t["input"], output=out_t))
                        have.add(norm_in(t["input"]))
                        fixes.append("add_" + ("uncapped_dolci_test" if name == "extra" else "statement_sample_test"))
            if len(keep_idx) + len(aug) < 3:
                drop(v, "lt3_tests_unaugmentable"); continue
            S[(src, "lt3_tests_augmented")] += 1
        elif len(keep_idx) < 3:
            drop(v, "lt3_tests"); continue

        final = [tests[i] for i in keep_idx] + aug
        # pass vectors on final tests for verification: original tests by index, aug verified by construction
        def ref_passes_final(j):
            rs = vec.get(j)
            return bool(rs) and all(rs[i]["ex"] for i in keep_idx)

        # --- 7. constant-output floor ---
        floor, groups = const_floor(kind, final)
        if floor >= 0.5:
            S[(src, "const_floor_ge50")] += 1
            # thin the dominant group(s) until no single answer passes >= 50%, keeping >= 3 tests
            idx = list(range(len(final)))
            while True:
                sub = [final[i] for i in idx]
                f, gs = const_floor(kind, sub)
                if f < 0.5 or len(idx) <= 3:
                    break
                dom = [idx[k] for k in gs[0]]
                # drop the dominant test with the shortest input (least discriminative)
                victim = min(dom, key=lambda i: len(str(final[i].get("input", final[i])) if kind == "stdio" else final[i]))
                idx.remove(victim)
            f, _ = const_floor(kind, [final[i] for i in idx])
            if f >= 0.5:
                drop(v, f"constant_output_floor_{floor:.2f}"); continue
            removed = len(final) - len(idx)
            keep_final = set(idx)
            # map back for verification bookkeeping
            n_orig_kept = len(keep_idx)
            keep_idx = [keep_idx[k] for k in range(n_orig_kept) if k in keep_final]
            aug = [aug[k - n_orig_kept] for k in range(n_orig_kept, len(final)) if k in keep_final]
            final = [final[i] for i in sorted(keep_final)]
            fixes.append(f"thin_constant_tests:{removed}")
            S[(src, "const_floor_thinned")] += 1

        # --- 8b. empty expected outputs ---
        if kind == "stdio":
            ne = sum(1 for t in final if not str(t["output"]).strip())
            if ne:
                S[(src, "rows_with_empty_output_tests")] += 1
            crlf = sum("\r" in str(t["input"]) for t in final)
            if crlf:
                S[(src, "rows_with_crlf_input")] += 1
            ws_out = sum(any(l.rstrip("\r") != l.rstrip() for l in str(t["output"]).split("\n")) for t in final)
            if ws_out:
                S[(src, "rows_with_trailing_space_in_output_lines(grader_strips:harmless)")] += 1
            inner = sum(bool(re.search(r"\S[ \t]{2,}\S", str(t["output"]))) for t in final)
            if inner:
                S[(src, "rows_with_inner_multispace_output")] += 1

        # --- verification ---
        verifiers = [j for j in vec if ref_passes_final(j)]
        verified = bool(verifiers)
        if not verified and kind == "stdio":
            # 8c. inner-whitespace layout the grader compares literally on non-numeric lines
            for j, rs in vec.items():
                sub = [rs[i] for i in keep_idx]
                if sub and all(r["ex"] or r.get("ws") for r in sub) and any(not r["ex"] for r in sub):
                    for k, i in enumerate(keep_idx):
                        if not rs[i]["ex"]:
                            final[k] = dict(final[k], output="\n".join(" ".join(l.split()) for l in str(final[k]["output"]).split("\n")))
                    fixes.append("collapse_inner_whitespace_in_expected")
                    S[(src, "ws_normalized_rows")] += 1
                    verifiers = [j]
                    verified = True
                    break
        relaxed = False
        if not verified and kind == "stdio":
            for j, rs in vec.items():
                sub = [rs[i] for i in keep_idx]
                if sub and all(r["ex"] or r.get("cs") and a["case_stmt"] or r.get("fl") and a["tol"] for r in sub):
                    relaxed = True
        md = dict(v["metadata"])
        md["n_tests"] = len(final)
        md["custom_id"] = v["custom_id"]
        md["fixes"] = fixes
        md["verified"] = verified
        if relaxed:
            md["verified_relaxed"] = True
        md["verifier_origins"] = sorted({refs[j]["origin"] for j in verifiers})
        # --- 4/5 flags ---
        if kind == "stdio" and a["tol"] is not None:
            outs = [t["output"] for t in final]
            if any(FLOATY.match(tok) for o in outs for tok in o.split()):
                md["float_tol"] = a["tol"] if isinstance(a["tol"], float) else 1e-6
                if a["tol"] == "unspecified":
                    md["float_tol_unspecified_value"] = True
                S[(src, "float_tol_flagged")] += 1
        if kind == "stdio" and a["case_stmt"]:
            md["case_insensitive"] = True
            S[(src, "case_insensitive_flagged")] += 1
        rec = dict(prompt=[{"role": "user", "content": v["prompt"]}], label=json.dumps(final, ensure_ascii=False), metadata=md)
        kept[row] = dict(rec=rec, v=v, n=len(final), verified=verified, n_ver=len(verifiers))
        diag.setdefault(row, {}).update(status="kept", fixes=fixes, verified=verified)

    # --- 10. dedup: keep the copy with the most tests (then verified, then more verifying refs) ---
    clusters = collections.defaultdict(list)
    for row, k in kept.items():
        clusters[D[row]].append(row)
    for cid, rows in clusters.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: (-kept[r]["n"], -kept[r]["verified"], -kept[r]["n_ver"], r))
        win = rows[0]
        for r in rows[1:]:
            v = kept[r]["v"]
            S[(v["src"], "dup_dropped")] += 1
            dropped.append(dict(custom_id=v["custom_id"], row=r, src=v["src"],
                                reason=f"duplicate_statement_of:{kept[win]['v']['custom_id']}"))
            diag[r] = dict(status="dropped", reason="duplicate")
            del kept[r]

    with open(f"{OUTD}/fixed.jsonl", "w") as f:
        for row in sorted(kept):
            f.write(json.dumps(kept[row]["rec"], ensure_ascii=False) + "\n")
    with open(f"{OUTD}/dropped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["custom_id", "reason", "src", "row"])
        w.writeheader()
        for d in sorted(dropped, key=lambda d: d["row"]):
            w.writerow(d)
    with open(f"{OUTD}/multi_answer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["custom_id", "src", "phrase", "a_ref_passes_exact", "n_tests"])
        w.writeheader()
        w.writerows(multi_rows)
    for row, k in kept.items():
        s = k["v"]["src"]
        S[(s, "kept")] += 1
        S[(s, "kept_verified")] += k["verified"]
        S[(s, "kept_relaxed_only")] += bool(k["rec"]["metadata"].get("verified_relaxed"))
        for fx in k["rec"]["metadata"]["fixes"]:
            S[(s, "fix:" + fx.split(":")[0])] += 1
    stats = {f"{a}|{b}": c for (a, b), c in sorted(S.items())}
    json.dump(stats, open(f"{OUTD}/stats.json", "w"), indent=1)
    pickle.dump(diag, open(f"{W}/decisions.pkl", "wb"))
    for k, c in stats.items():
        print(f"{k:70s} {c}")


if __name__ == "__main__":
    main()
