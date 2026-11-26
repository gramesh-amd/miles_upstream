#!/usr/bin/env python3
"""Assemble statement-level corpus (no test blobs) for all candidate stdio sources + our v7 rows.

Reuses the v5 stdio audit (/tmp/datacand/norm: statements; audit/all_rows.parquet: flags, dates, test counts)
and adds sources the v5 audit did not load (deepmind/code_contests, BAAI/TACO, INTELLECT-3-RL code).
Output: work/corpus.parquet  (one row per candidate problem row), work/ours.parquet (v7 + Dolci rows).
"""
import glob, json, re, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/tmp/datacand")
from common import text_flags  # noqa: E402

N = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/new_sources"
DC = "/tmp/datacand"
V7 = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data"
CF_DATES = {str(c["id"]): pd.to_datetime(c["startTimeSeconds"], unit="s").strftime("%Y-%m-%d")
            for c in json.load(open(f"{N}/raw/cf_contest_list.json"))["result"] if c.get("startTimeSeconds")}

RE_CF_URL = re.compile(r"codeforces\.com/(?:contest|problemset/problem|gym)/(\d+)/(?:problem/)?([A-Z]\d?)", re.I)
RE_AT_URL = re.compile(r"atcoder\.jp/contests/[^/]+/tasks/([a-z0-9_]+)", re.I)


def key_from_url(u):
    if not isinstance(u, str):
        return None
    m = RE_CF_URL.search(u)
    if m:
        return f"cf:{m.group(1)}/{m.group(2).upper()}"
    m = RE_AT_URL.search(u)
    if m:
        return f"at:{m.group(1).lower()}"
    return None


A_COLS = ["pid", "source", "sub", "fn_style", "f_stdin_format", "interactive", "image", "multi_answer_text",
          "float_tol_text", "file_io", "needs_checker", "checker", "case_insensitive", "float_checker",
          "n_tests_known", "date_final", "date_upper", "date_source", "url", "title", "extra", "ccplus_verified",
          "tpr1", "tnr1", "in_dolci", "fp_const_all", "r1_solve_ocr2", "proxy_solve", "rating", "n_py_solutions"]
A = pd.read_parquet(f"{DC}/norm/all_rows.parquet", columns=A_COLS)

parts = []
for f in ["openr1_cf", "ccplus", "hardtests", "deepcoder", "taco_verified", "klear", "synthetic2", "skywork",
          "nemotron_rl", "eurus", "rstar"]:
    t = pq.read_table(f"{DC}/norm/{f}.parquet", columns=["pid", "statement"]).to_pandas()
    parts.append(t)
S = pd.concat(parts).drop_duplicates("pid")
df = A.merge(S, on="pid", how="left")
# rStar synthetic_rl: 398K LLM-written problems without dates or reference outputs; keep a 30K sample for rates
syn = df.source == "rstar_synthetic_rl"
df = pd.concat([df[~syn], df[syn].sample(30000, random_state=0)])
df["sample_weight"] = np.where(df.source == "rstar_synthetic_rl", 398107 / 30000, 1.0)


def pkey(r):
    k = key_from_url(r.url)
    if k:
        return k
    p = r.pid
    if r.source == "openr1_cf_verifiable":
        c, i = p.split("/")[1:3]
        return f"cf:{c}/{i}"
    if r.source == "ccplus":
        try:
            plat = json.loads(r.extra).get("platform")
        except Exception:
            plat = None
        m = re.match(r"ccplus/(\d+)_([A-Z]\d?)$", p)
        if plat == "Codeforces" and m:
            return f"cf:{m.group(1)}/{m.group(2)}"
    return None


df["pkey"] = [pkey(r) for r in df.itertuples()]

# ---------------------------------------------------------------- new sources (not in v5 audit)
new = []
cc = pd.read_parquet(f"{DC}/meta/cc_train_meta.parquet")
for r in cc.itertuples():
    k = f"cf:{r.cf_contest_id}/{r.cf_index}" if r.cf_contest_id and r.cf_contest_id > 0 else None
    d = CF_DATES.get(str(r.cf_contest_id)) if k else None
    new.append(dict(pid=f"code_contests/{r.name}", source="deepmind_code_contests", sub=str(r.source),
                    statement=r.description, pkey=k, date_final=d, date_upper=d or "2021-12-31",
                    date_source="own" if d else "vintage_bound", rating=r.cf_rating or np.nan,
                    file_io=bool(r.input_file), n_tests_known=np.nan))
tc = pd.read_parquet(f"{DC}/meta/taco_train_meta.parquet")
for i, r in enumerate(tc.itertuples()):
    k = key_from_url(r.url)
    d = r.date if isinstance(r.date, str) and r.date[:2] in ("19", "20") else None
    if k and k.startswith("cf:"):
        d = d or CF_DATES.get(k[3:].split("/")[0])
    new.append(dict(pid=f"taco/{i}", source="baai_taco", sub=str(r.source), statement=r.question, pkey=k,
                    date_final=d, date_upper=d or "2023-03-31", date_source="own" if d else "vintage_bound",
                    image=bool(r.picture_num and str(r.picture_num) not in ("0", "None", "nan")),
                    n_tests_known=np.nan, url=r.url))
i3 = pd.read_parquet(f"{N}/raw/intellect3/code/train-00000-of-00001.parquet")
for i, r in enumerate(i3.itertuples()):
    info = json.loads(r.info)
    try:
        t = json.loads(info.get("tests") or "{}")
        nt = len(t.get("inputs", [])) if isinstance(t, dict) else len(t)
        fn = bool(isinstance(t, dict) and t.get("fn_name"))
    except Exception:
        nt, fn = np.nan, False
    new.append(dict(pid=f"intellect3/{i}", source="intellect3_code", sub=info.get("source"), statement=r.question,
                    n_tests_known=nt, fn_style=fn, proxy_solve=getattr(r, "_3", np.nan)))
nd = pd.DataFrame(new)
for c, v in text_flags("").items():
    if c in nd:
        nd[c] = nd[c].fillna(False) | [text_flags(s)[c] for s in nd.statement]
    else:
        nd[c] = [text_flags(s)[c] for s in nd.statement]
nd["fn_style"] = nd.get("fn_style", False)
nd["fn_style"] = nd["fn_style"].fillna(False).astype(bool) | nd["fn_statement"]
nd["needs_checker"] = nd.multi_answer_text
nd["f_stdin_format"] = ~nd.fn_style
nd["sample_weight"] = 1.0
df = pd.concat([df, nd.drop(columns=["fn_statement"])], ignore_index=True)
df["statement"] = df.statement.fillna("")
for c in ["interactive", "image", "multi_answer_text", "float_tol_text", "file_io", "needs_checker", "fn_style",
          "case_insensitive", "float_checker", "ccplus_verified", "in_dolci"]:
    df[c] = df[c].fillna(False).astype(bool)
df["f_stdin_format"] = df.f_stdin_format.fillna(True).astype(bool)
for c in ["extra", "checker", "url", "title", "sub", "date_final", "date_upper", "date_source", "fp_const_all"]:
    df[c] = df[c].astype("string")
df.to_parquet(f"{N}/work/corpus.parquet")
print(df.source.value_counts().to_string())

# ---------------------------------------------------------------- ours
rows = []
for kind in ["stdio", "function"]:
    for line in open(f"{V7}/{kind}/fixed.jsonl"):
        d = json.loads(line)
        rows.append(dict(oid=d["metadata"]["custom_id"], set=f"v7_{kind}", text=d["prompt"][-1]["content"]))
ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
for idr, line in zip(ids, open(f"{V7}/input/code-train/dolci-code-train.jsonl")):
    d = json.loads(line)
    rows.append(dict(oid=idr["custom_id"], set=f"dolci_{idr['src']}", text=d["prompt"][-1]["content"]))
o = pd.DataFrame(rows)
o.to_parquet(f"{N}/work/ours.parquet")
print(o.set.value_counts().to_string())
