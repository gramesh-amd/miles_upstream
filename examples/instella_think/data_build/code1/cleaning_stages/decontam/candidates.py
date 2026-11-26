#!/usr/bin/env python3
"""Merge detector outputs into one (train row, eval problem) candidate table and dump side-by-side views.

Hand-check thresholds (a pair is a candidate if ANY holds):
  ng13 >= 0.20 | ng8 >= 0.50 | jac5 >= 0.30 | exact >= 1 | tfidf >= 0.50
  func_score >= 0.75 (same name and same arity)
  callname >= 1 (same function name AND same args/expected) | call >= 3 | io >= 1 | assert >= 1
  idref in {cf_url, cf_ref, lc_url, ac_url, ac_id}  (direct id of an eval problem)
  title (eval title phrase in prompt) AND tfidf >= 0.25
"""
import pickle, sys
import numpy as np
import pandas as pd

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/work"
train = pickle.load(open(f"{W}/train.pkl", "rb"))
ev = pickle.load(open(f"{W}/eval.pkl", "rb"))

text = pd.read_parquet(f"{W}/text_pairs.parquet")
tests = pd.read_parquet(f"{W}/test_pairs.parquet")
func = pd.read_parquet(f"{W}/func_pairs.parquet")
ids = pd.read_parquet(f"{W}/id_pairs.parquet")
ids = ids.groupby(["ti", "ei"]).agg(idref=("idref", lambda s: ",".join(sorted(set(s)))),
                                     idmatch=("idmatch", lambda s: " | ".join(sorted(set(s))))).reset_index()
m = text.merge(tests, on=["ti", "ei"], how="outer").merge(func, on=["ti", "ei"], how="outer") \
    .merge(ids, on=["ti", "ei"], how="outer")
for c in ["ng13", "ng13_train", "ng8", "ng8_train", "jac5", "tfidf", "exact", "call_label", "call_full", "io_label",
          "io_full", "callname_label", "callname_full", "assert_label", "assert_full", "func_score"]:
    m[c] = m[c].fillna(0)
m["idref"] = m["idref"].fillna("")

# text scores for pairs found only by non-text detectors: compute tfidf/ngram on the fly (cheap: few pairs)
sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/scripts")
from detect import tok, ngrams  # noqa: E402


def quick(ti, ei):
    a, b = tok(train[ti]["text"]), tok(ev[ei]["text"])
    out = {}
    for n in (13, 8, 5):
        A, B = ngrams(a, n), ngrams(b, n)
        out[f"q{n}"] = len(A & B) / max(1, len(B))
        if n == 5:
            out["qjac5"] = len(A & B) / max(1, len(A | B))
    return out


q = pd.DataFrame([quick(t, e) for t, e in zip(m.ti, m.ei)])
m = pd.concat([m.reset_index(drop=True), q], axis=1)
m["call"] = np.maximum(m.call_label, m.call_full)
m["callname"] = np.maximum(m.callname_label, m.callname_full)
m["io"] = np.maximum(m.io_label, m.io_full)
m["assert_"] = np.maximum(m.assert_label, m.assert_full)
direct = m.idref.str.contains("cf_url|cf_ref|lc_url|ac_url|ac_id")
m["cand"] = (m.ng13 >= .2) | (m.ng8 >= .5) | (m.jac5 >= .3) | (m.exact >= 1) | (m.tfidf >= .5) | \
            (m.func_score >= .75) | (m.callname >= 1) | (m.call >= 3) | (m.io >= 1) | (m.assert_ >= 1) | direct | \
            (m.idref.str.contains("title") & (m.tfidf >= .25))
m["custom_id"] = [train[i]["custom_id"] for i in m.ti]
m["src"] = [train[i]["src"] for i in m.ti]
m["eval_set"] = [ev[i]["eval_set"] for i in m.ei]
m["eval_id"] = [ev[i]["eval_id"] for i in m.ei]
m["part"] = [ev[i]["part"] for i in m.ei]
m["sets"] = [",".join(ev[i]["sets"]) for i in m.ei]
m["date"] = [ev[i]["date"] for i in m.ei]
m.to_parquet(f"{W}/all_pairs.parquet")
c = m[m.cand].copy()
print("all pairs", len(m), "candidates", len(c), "unique train rows", c.ti.nunique())
print(c.groupby("eval_set").agg(pairs=("ti", "size"), rows=("ti", "nunique"), evals=("eval_id", "nunique")))


def reason(r):
    rs = []
    for k, t in (("ng13", .2), ("ng8", .5), ("jac5", .3), ("tfidf", .5)):
        if r[k] >= t:
            rs.append(f"{k}={r[k]:.2f}")
    if r.exact:
        rs.append("exact")
    if r.func_score >= .75:
        rs.append(f"func={r.func}({r.func_score})")
    if r.callname:
        rs.append(f"callname={int(r.callname)}")
    if r.call >= 3:
        rs.append(f"call={int(r.call)}")
    if r.io:
        rs.append(f"io={int(r.io)}")
    if r.assert_:
        rs.append(f"assert={int(r.assert_)}")
    if r.idref:
        rs.append(f"id={r.idref}:{r.idmatch}")
    return "; ".join(rs)


c["why"] = c.apply(reason, axis=1)
c = c.sort_values(["eval_set", "eval_id", "tfidf"], ascending=[True, True, False])
c.to_parquet(f"{W}/candidates.parquet")
for es, g in c.groupby("eval_set"):
    with open(f"{W}/review_{es}.md", "w") as f:
        for k, r in enumerate(g.itertuples()):
            e, t = ev[r.ei], train[r.ti]
            f.write(f"\n## [{k}] {r.eval_id} ({r.part}, {r.sets}) <-> {r.custom_id} ({r.src})\n")
            f.write(f"why: {r.why} | tfidf={r.tfidf:.2f} q13={r.q13:.2f} q8={r.q8:.2f} qjac5={r.qjac5:.2f} "
                    f"call={int(r.call)} io={int(r.io)} func={r.func}\n")
            f.write("EVAL: " + e["text"][:900].replace("\n", " ") + "\n")
            f.write("TRAIN: " + t["text"][:900].replace("\n", " ") + "\n")
            lab = t["tests_label"]
            f.write("TRAIN_TESTS: " + str(lab[:2])[:300] + "\n")
