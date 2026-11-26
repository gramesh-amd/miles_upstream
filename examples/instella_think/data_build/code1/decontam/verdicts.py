#!/usr/bin/env python3
"""Assign a verdict to every hand-checked (train row, eval problem) pair and write the deliverables.

Verdicts
  contaminated        same problem: same task and interface as the eval item (eval solution passes the row's tests,
                      a training reference passes the eval tests or the statement reproduces the eval item), or the
                      row shares the eval's hidden tests wholesale.
  paraphrase          same problem idea, but the statement and the tests differ (renamed function, added twist,
                      different constraints/format).
  paraphrase_generic  a textbook one-liner (is_prime, median, power, circle area, ...) restated; collides with
                      the eval item only because the task is generic. Not dropped (see REPORT.md).
  false_positive      different problem; generic phrasing, edge-case tests or a reused function name.
For HE-Pro / MBPP-Pro, a match to the embedded raw (HumanEval / MBPP) sub-problem is reported with the suffix
"_raw_subproblem"; the Pro-specific new function is the tested part.

Rule (function-style sets, hand-checked; overrides in HAND below):
  A = fraction of the row's tests passed by the eval canonical solution, B = training references passing eval tests.
  A >= .85 and same function name and (B > 0 or no reference available)  -> contaminated (paraphrase_generic if
                                                                            the eval item is a textbook one-liner)
  A >= .85 otherwise                                                     -> paraphrase / paraphrase_generic
  else                                                                   -> false_positive
"""
import os, pickle, re
import numpy as np
import pandas as pd

D = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam"
W = f"{D}/work"
AC = "rlvr_acecoder_filtered_filtered-request-"
KL = "klear-code-rlvr_filtered-request-"
NE6 = "llama-nemotron-rlvr-difficulty-6_filtered-request-"

GENERIC_HE = {f"HumanEval/{i}" for i in (8, 9, 12, 13, 23, 28, 29, 30, 31, 34, 35, 42, 45, 47, 48, 52, 53, 55, 58, 60,
                                         61, 77, 90, 96, 126)}


def is_generic(eval_set, eval_id):
    return eval_set in ("mbppplus", "mbpppro") or eval_id in GENERIC_HE


# (eval_id, custom_id) -> verdict ; eval_id "*" applies to every eval item of that row
HAND = {
    # Codeforces: Russian Technocup 2016 problem 646B == CF 2010 C1/C2 (Testing Round 19, 2024-08-28); 49/24 tests shared
    ("2010/C1", KL + "1-56"): "contaminated", ("2010/C1", KL + "139-95"): "contaminated",
    ("2010/C1", KL + "97-100"): "contaminated", ("2010/C2", KL + "1-56"): "contaminated",
    ("2010/C2", KL + "139-95"): "contaminated", ("2010/C2", KL + "97-100"): "contaminated",
    # LCB v5 3659 (count paths with XOR k, 2024-12-21) re-hosts CF 1006F with the same examples, other constraints
    ("3659", KL + "135-25"): "paraphrase", ("3659", KL + "94-70"): "paraphrase",
    # HumanEval: training row restates the HE item with HE-specific content but a different name
    ("HumanEval/63", AC + "189-54"): "contaminated",
    ("HumanEval/26", AC + "187-16"): "contaminated", ("HumanEval/71", AC + "213-16"): "contaminated",
    ("HumanEval/95", AC + "66-100"): "contaminated",
    # same name, A high, but the statement asks for different behaviour (tests do not exercise it)
    ("HumanEval/5", AC + "124-32"): "paraphrase", ("HumanEval/5", AC + "230-92"): "paraphrase",
    ("HumanEval/10", AC + "246-38"): "paraphrase", ("HumanEval/3", AC + "199-40"): "paraphrase",
    # evolved variants with A < .85 (HumanEval-derived twist)
    ("HumanEval/1", AC + "219-61"): "paraphrase", ("HumanEval/154", AC + "264-54"): "paraphrase",
    ("HumanEval/154", AC + "134-46"): "paraphrase", ("HumanEval/154", AC + "183-45"): "paraphrase",
    ("HumanEval/3", AC + "139-95"): "paraphrase", ("HumanEval/5", AC + "267-94"): "paraphrase",
    ("HumanEval/5", AC + "60-83"): "paraphrase", ("HumanEval/5", AC + "215-73"): "paraphrase",
    ("HumanEval/5", AC + "232-34"): "paraphrase",
    ("HumanEval/57", AC + "237-15"): "paraphrase", ("HumanEval/108", AC + "245-34"): "paraphrase",
    ("HumanEval/12", AC + "235-41"): "paraphrase", ("HumanEval/128", AC + "228-66"): "paraphrase",
    ("HumanEval/129", AC + "265-46"): "paraphrase", ("HumanEval/129", AC + "163-56"): "paraphrase",
    ("HumanEval/26", AC + "115-61"): "paraphrase", ("HumanEval/54", AC + "173-35"): "paraphrase",
    ("HumanEval/75", AC + "186-87"): "paraphrase", ("HumanEval/91", AC + "239-12"): "paraphrase",
    ("HumanEval/95", AC + "87-16"): "paraphrase", ("HumanEval/95", AC + "219-100"): "paraphrase",
    ("HumanEval/95", AC + "140-46"): "paraphrase", ("HumanEval/95", AC + "180-50"): "paraphrase",
    ("HumanEval/95", AC + "246-30"): "paraphrase", ("HumanEval/36", AC + "175-20"): "paraphrase",
    ("HumanEval/116", AC + "175-44"): "paraphrase",
    ("mbppplus/610", AC + "69-42"): "paraphrase", ("mbppplus/757", AC + "146-64"): "paraphrase",
    ("mbppplus/641", AC + "124-6"): "paraphrase",
    ("humanevalpro/48", AC + "47-50"): "paraphrase",
    # generic restatements with A < .85
    ("HumanEval/47", AC + "392-33"): "false_positive", ("HumanEval/55", AC + "125-63"): "paraphrase_generic",
    ("HumanEval/31", AC + "134-26"): "paraphrase_generic", ("HumanEval/31", AC + "179-78"): "paraphrase_generic",
    ("HumanEval/96", NE6 + "11-33"): "paraphrase_generic", ("HumanEval/96", AC + "247-76"): "paraphrase_generic",
    ("HumanEval/53", AC + "42-22"): "paraphrase_generic", ("HumanEval/61", AC + "533-51"): "paraphrase_generic",
    ("HumanEval/58", AC + "86-78"): "false_positive",
    ("mbppplus/583", AC + "89-38"): "paraphrase_generic", ("mbppplus/560", AC + "459-69"): "paraphrase_generic",
    ("mbpppro/317", AC + "371-97"): "paraphrase_generic", ("mbpppro/357", AC + "371-97"): "paraphrase_generic",
    ("mbpppro/60", AC + "371-97"): "paraphrase_generic",
    # A high by coincidence (different task)
    ("HumanEval/10", AC + "203-65"): "false_positive", ("HumanEval/10", AC + "194-66"): "false_positive",
    ("HumanEval/10", NE6 + "9-22"): "false_positive", ("HumanEval/10", AC + "70-14"): "false_positive",
    ("HumanEval/58", AC + "90-47"): "false_positive", ("HumanEval/61", AC + "4-3"): "false_positive",
    ("HumanEval/54", AC + "261-55"): "false_positive", ("HumanEval/48", NE6 + "20-98"): "false_positive",
    ("mbppplus/244", AC + "238-72"): "false_positive", ("mbppplus/292", AC + "207-25"): "false_positive",
    ("mbppplus/568", AC + "479-70"): "false_positive", ("mbppplus/631", AC + "14-84"): "false_positive",
    ("mbppplus/743", AC + "204-45"): "false_positive", ("mbppplus/71", AC + "239-92"): "false_positive",
    ("mbppplus/141", AC + "239-92"): "false_positive", ("mbppplus/105", AC + "226-76"): "false_positive",
    ("mbppplus/65", AC + "226-76"): "false_positive", ("mbppplus/97", AC + "176-6"): "false_positive",
    ("mbppplus/97", AC + "126-79"): "false_positive", ("mbppplus/16", AC + "115-76"): "false_positive",
    ("mbppplus/447", AC + "145-39"): "false_positive",
    ("humanevalpro/74", AC + "73-60"): "false_positive", ("humanevalpro/100", AC + "200-25"): "false_positive",
    ("mbpppro/41", AC + "126-79"): "false_positive", ("mbpppro/41", AC + "147-54"): "false_positive",
    ("mbpppro/75", AC + "214-27"): "false_positive", ("mbpppro/149", AC + "376-1"): "false_positive",
    ("mbpppro/154", AC + "226-76"): "false_positive", ("mbpppro/183", AC + "217-77"): "false_positive",
    ("mbpppro/183", AC + "232-23"): "false_positive", ("mbpppro/183", AC + "434-95"): "false_positive",
    ("mbpppro/338", AC + "574-98"): "false_positive", ("mbpppro/374", AC + "434-95"): "false_positive",
    ("mbpppro/266", "llama-nemotron-rlvr-difficulty-6_filtered-request-2-35"): "false_positive",
    ("mbpppro/266", "llama-nemotron-rlvr-difficulty-6_filtered-request-15-45"): "false_positive",
}
# rows whose every HE/10 match is an evolved make_palindrome variant (ignore case / non-alnum etc.)
HE10_VARIANT_DEFAULT = "paraphrase"


def main():
    train = pickle.load(open(f"{W}/train.pkl", "rb"))
    ev = pickle.load(open(f"{W}/eval.pkl", "rb"))
    allp = pd.read_parquet(f"{W}/all_pairs.parquet")
    cand = pd.read_parquet(f"{W}/candidates.parquet")
    ex = pd.read_parquet(f"{W}/exec_verify.parquet")
    num = pd.read_parquet(f"{W}/num_pairs.parquet")

    # pairs to adjudicate: detector candidates + execution hits from the TF-IDF top-10 + numeric-example hits
    exhit = ex[(ex.A >= 0.85) | (ex.B_pass > 0)][["ti", "ei"]]
    numhit = num[(num.tfidf >= 0.3)][["ti", "ei"]]
    raw = pd.read_csv(f"{W}/raw_ngram_check.csv")
    raw = raw[((raw.n == 8) & (raw.raw_ratio >= 0.2)) | ((raw.n == 13) & (raw.raw_ratio >= 0.1))]
    cid2ti = {t["custom_id"]: i for i, t in enumerate(train)}
    key2ei = {(e["eval_id"], e["part"]): j for j, e in enumerate(ev)}
    rawhit = pd.DataFrame({"ti": [cid2ti[c] for c in raw.best], "ei": [key2ei[(a, b)] for a, b in zip(raw.eval_id, raw.part)]})
    keys = pd.concat([cand[["ti", "ei"]], exhit, numhit, rawhit]).drop_duplicates()
    m = keys.merge(allp, on=["ti", "ei"], how="left").merge(
        ex[["ti", "ei", "A", "A_pass", "A_n", "B_pass", "B_tried", "train_func", "eval_entry", "origin"]],
        on=["ti", "ei"], how="left").merge(num[["ti", "ei", "num8"]].rename(columns={}), on=["ti", "ei"], how="left")
    m["custom_id"] = [train[i]["custom_id"] for i in m.ti]
    m["src"] = [train[i]["src"] for i in m.ti]
    m["eval_set"] = [ev[j]["eval_set"] for j in m.ei]
    m["eval_id"] = [ev[j]["eval_id"] for j in m.ei]
    m["part"] = [ev[j]["part"] for j in m.ei]
    m["sets"] = [",".join(ev[j]["sets"]) for j in m.ei]
    for c in ("A", "B_pass", "B_tried", "num8", "ng13", "ng8", "jac5", "tfidf", "exact", "func_score", "call",
              "callname", "io", "assert_"):
        m[c] = m[c].fillna(0)
    m["idref"] = m["idref"].fillna("")
    m["train_func"] = m["train_func"].fillna("")

    # pro raw parts are the HE/MBPP item verbatim: reuse that verdict
    he_name = {e["eval_id"]: next(iter(e["funcs"]), None) for e in ev if e["eval_set"] in ("humanevalplus", "mbppplus")}
    name_to_id = {}
    for k, v in he_name.items():
        if v:
            name_to_id.setdefault(v, k)

    def base_id(r):
        if r.part != "raw":
            return r.eval_id
        if r.eval_set == "humanevalpro":
            return "HumanEval/" + r.eval_id.split("/")[1]
        fn = next(iter(ev[r.ei]["funcs"]), None)
        return name_to_id.get(fn, r.eval_id)

    m["base_id"] = m.apply(base_id, axis=1)

    def rule(r):
        key = (r.base_id, r.custom_id)
        if key in HAND:
            return HAND[key]
        if r.eval_set in ("lcb", "codeforces", "cruxeval"):
            return "false_positive"
        gen = is_generic("mbppplus" if r.base_id.startswith("mbppplus") else r.eval_set, r.base_id)
        same_name = bool(r.eval_entry) and r.eval_entry in r.train_func.split(",")
        if r.A >= 0.85:
            if same_name and (r.B_pass > 0 or r.B_tried == 0):
                return "paraphrase_generic" if gen else "contaminated"
            return "paraphrase_generic" if gen else "paraphrase"
        if r.base_id == "HumanEval/10" and "palindrom" in train[r.ti]["text"].lower():
            return HE10_VARIANT_DEFAULT
        return "false_positive"

    # raw parts: evaluate the rule against the base HE/MBPP pair (same row, same text) when available
    m["verdict"] = m.apply(rule, axis=1)
    base = m[m.part != "raw"].set_index(["custom_id", "eval_id"]).verdict.to_dict()
    raw = m.part == "raw"
    m.loc[raw, "verdict"] = [base.get((c, b), v) for c, b, v in zip(m[raw].custom_id, m[raw].base_id, m[raw].verdict)]
    m.loc[raw & (m.verdict != "false_positive"), "verdict"] = m.loc[raw & (m.verdict != "false_positive"), "verdict"] + "_raw_subproblem"

    def detector(r):
        opts = [("exact", r.exact, 1), ("ng13", r.ng13, .2), ("ng8", r.ng8, .5), ("jac5", r.jac5, .3),
                ("test_io", r.io, 1), ("test_call_samename", r.callname, 1), ("test_assert", r.assert_, 1),
                ("exec_A", r.A, .85), ("func_sig", r.func_score, .75), ("num_examples", r.num8, 1),
                ("tfidf", r.tfidf, .5), ("test_call", r.call, 3)]
        hit = [f"{n}={v:.3g}" for n, v, t in opts if v >= t]
        if r.idref:
            hit.append("idref:" + r.idref)
        return ";".join(hit) or f"tfidf={r.tfidf:.2f}"

    m["detector"] = m.apply(detector, axis=1)
    m["test_frac"] = (np.maximum(m.io, m.call) / m.n_eval_tests.fillna(0).clip(lower=1)).fillna(0)
    m["score"] = m[["ng13", "ng8", "jac5", "tfidf", "A", "test_frac"]].max(axis=1).round(3)
    # one row per eval set membership (LCB problems can belong to v3/v5/v6)
    rows = []
    for r in m.itertuples():
        sets = r.sets.split(",")
        if r.eval_set == "cruxeval":
            sets = ["cruxeval_o", "cruxeval_i"]
        for s in sets:
            rows.append(dict(custom_id=r.custom_id, src=r.src, eval_set=s, eval_id=r.eval_id, part=r.part,
                             detector=r.detector, score=r.score, verdict=r.verdict, ng13=round(r.ng13, 3),
                             ng8=round(r.ng8, 3), jac5=round(r.jac5, 3), tfidf=round(r.tfidf, 3),
                             exec_A=f"{int(r.A_pass) if r.A_pass == r.A_pass else 0}/{int(r.A_n) if r.A_n == r.A_n else 0}",
                             exec_B=f"{int(r.B_pass)}/{int(r.B_tried)}", train_func=r.train_func,
                             eval_entry=r.eval_entry if isinstance(r.eval_entry, str) else ""))
    out = pd.DataFrame(rows)
    # HumanEval-Pro item i embeds HumanEval/i verbatim as its raw sub-problem
    he_c = out[(out.eval_set == "humanevalplus") & (out.verdict == "contaminated")]
    have = set(zip(out[out.eval_set == "humanevalpro"].custom_id, out[out.eval_set == "humanevalpro"].eval_id))
    extra = he_c.assign(eval_set="humanevalpro", eval_id="humanevalpro/" + he_c.eval_id.str.split("/").str[1], part="raw",
                        verdict="contaminated_raw_subproblem")
    extra = extra[[(a, b) not in have for a, b in zip(extra.custom_id, extra.eval_id)]]
    out = pd.concat([out[~out.set_index(["custom_id", "eval_id"]).index.isin(list(zip(extra.custom_id, extra.eval_id)))], extra])
    out = out.sort_values(["verdict", "eval_set", "eval_id", "custom_id"])
    out.to_csv(f"{D}/contaminated.csv", index=False)
    drop = sorted(set(out[out.verdict.str.startswith("contaminated")].custom_id))
    with open(f"{D}/drop_list.txt", "w") as f:
        f.write("\n".join(drop) + "\n")
    para = out[out.verdict.str.startswith("paraphrase")]
    para[["custom_id", "eval_set", "eval_id", "verdict"]].drop_duplicates().to_csv(f"{D}/paraphrase_list.csv", index=False)
    pd.DataFrame({"custom_id": drop, "reason": [
        "eval contamination: " + ";".join(sorted(set(out[(out.custom_id == c) & out.verdict.str.startswith("contaminated")].eval_set + ":" +
                                                    out[(out.custom_id == c) & out.verdict.str.startswith("contaminated")].eval_id)))
        for c in drop]}).to_csv(f"{D}/dropped.csv", index=False)
    print("pairs adjudicated", len(m), "rows in csv", len(out), "drop", len(drop))
    s = out.drop_duplicates(["custom_id", "eval_set", "eval_id"]).groupby(["eval_set", "verdict"]).agg(
        pairs=("custom_id", "size"), train_rows=("custom_id", "nunique"), eval_items=("eval_id", "nunique"))
    print(s.to_string())
    s.to_csv(f"{W}/verdict_summary.csv")


if __name__ == "__main__":
    main()
