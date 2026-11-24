#!/usr/bin/env python3
"""Decontaminate the assembled mix against the eval sets.

    python3 decontam.py OUT/train_pre_decontam.jsonl --eval-items eval_items.parquet \
        --eval-hashes eval_hashes.jsonl --out-dir OUT

Eval corpus:
  --eval-items   one row per eval statement part: eval_set, eval_id, text, platform, item ("<set>|<id>")
  --eval-hashes  one JSON object per eval problem: eval_set, eval_id, platform, title,
                 io [[h, n], ...] (hidden stdin/stdout pairs), calls [[fname, h, n], ...] (hidden function
                 calls), with h = common.h64 of the normalised pair and n its length in characters
Detectors and actions:
  ng8 >= 0.5 / ng13 >= 0.2 containment of the eval item in one train statement (shared >= 3), boilerplate
      n-grams (in >= 5 eval problems or >= 100 train rows) removed                              -> drop
  Codeforces contest/index, AtCoder task or LeetCode slug equal to an eval item                 -> drop
  identical normalised stdin/stdout pairs with an LCB / Codeforces eval item (input >= 10 chars, pair in
      < 3 eval items): >= 4 shared -> drop, 1-3 reported (short shared pairs are edge-case collisions)
  function rows: identical (args, expected) calls with the eval's function name: >= 1 for HumanEval / MBPP
      and their Pro variants, >= 3 for LCB / CRUXEval (LeetCode names like minimumOperations recur)
      -> drop; other overlaps reported
  date >= --cutoff (start of the LCB v5 window)                                                  -> drop
Writes OUT/decontam_hits.csv and OUT/decontam_drop.txt. Rows are streamed once; only their n-grams and
hashes are kept.
"""
import argparse
import ast
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import h64, read_jsonl, statement_of, tok

WRAPPERS = {"set", "sorted", "list", "tuple", "frozenset", "str", "int", "float", "round", "abs", "len", "bool"}
CF_URL = re.compile(r"codeforces\.com/(?:contest|problemset/problem|gym)/(\d+)/(?:problem/)?([A-Za-z]\d?)")
AT_URL = re.compile(r"atcoder\.jp/contests/[^/]+/tasks/([a-z0-9_]+)")


def norm_io(s):
    s = str(s).replace("\r", "")
    return "\n".join(ln.rstrip() for ln in s.strip().split("\n"))


def canon(v):
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(canon(x) for x in v)) + "}"
    if isinstance(v, dict):
        return "{" + ",".join(sorted(canon(k) + ":" + canon(x) for k, x in v.items())) + "}"
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return repr(int(v))
    return repr(v)


def lit(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in WRAPPERS \
            and len(node.args) == 1 and not node.keywords:
        return ast.literal_eval(node.args[0])
    return ast.literal_eval(node)


def grams(ws, n):
    if len(ws) < n:
        return np.zeros(0, np.uint64)
    return np.unique(np.fromiter((h64(" ".join(ws[i:i + n])) for i in range(len(ws) - n + 1)), np.uint64))


def fn_calls(label, fn):
    """(fname, args canon, expected canon) of fn_build.py asserts: `_r = fn(args)\\nassert (...) == expected`."""
    out = []
    for s in label:
        try:
            t = ast.parse(s)
            args = [lit(x) for x in t.body[0].value.args]
            out.append((fn, canon(args), canon(lit(t.body[1].test.comparators[0]))))
        except Exception:
            continue
    return out


def features(r):
    """What the detectors need from one row."""
    m = r["metadata"]
    ws = tok(statement_of(r))
    label = json.loads(r["label"]) if isinstance(r["label"], str) else r["label"]
    if isinstance(label, dict):
        label = label.get("tests") or []
    io, calls = [], []
    if label and isinstance(label[0], dict):
        seen = set()
        for t in label:
            i, o = norm_io(t.get("input", "")), norm_io(t.get("output", ""))
            if len(i) < 10:
                continue
            h = h64(i + "\x00" + o)
            if h not in seen:
                seen.add(h)
                io.append(h)
    elif label and m.get("fn_name") and str(m.get("custom_id", "")).startswith("v8_"):
        calls = [(fn, h64(a + "\x00" + x)) for fn, a, x in fn_calls(label, m["fn_name"]) if len(a) + len(x) >= 12]
    return {"cid": m["custom_id"], "g": (grams(ws, 8), grams(ws, 13)), "url": m.get("url") or "",
            "upstream_pid": m.get("upstream_pid") or "", "openr1_id": m.get("openr1_id") or "",
            "io": io, "is_io": bool(label and isinstance(label[0], dict)),
            "calls": calls, "date": m.get("date") or m.get("date_eff") or m.get("cf_date")}


def ngram_hits(rows, E):
    hits = []
    FE = [(grams(tok(t or ""), 8), grams(tok(t or ""), 13)) for t in E.text]
    items = E.item.values
    uitem = {k: i for i, k in enumerate(pd.unique(items))}
    prob = np.array([uitem[k] for k in items])
    for ni, (n, thr) in enumerate(((8, 0.5), (13, 0.2))):
        vocab = {}
        for f in FE:
            for g in f[ni].tolist():
                vocab.setdefault(g, len(vocab))
        er, ec = [], []
        for i, f in enumerate(FE):
            for g in f[ni].tolist():
                er.append(i)
                ec.append(vocab[g])
        Em = sp.csr_matrix((np.ones(len(er), np.float32), (er, ec)), shape=(len(E), len(vocab)))
        Pm = sp.csr_matrix((np.ones(len(E)), (prob, np.arange(len(E)))), shape=(len(uitem), len(E))) @ Em
        df_e = np.asarray((Pm > 0).sum(0)).ravel()
        tr, tc = [], []
        for i, r in enumerate(rows):
            for g in r["g"][ni].tolist():
                j = vocab.get(g)
                if j is not None:
                    tr.append(i)
                    tc.append(j)
        Tm = sp.csr_matrix((np.ones(len(tr), np.float32), (tr, tc)), shape=(len(rows), len(vocab)))
        df_t = np.asarray(Tm.sum(0)).ravel()
        keep = ~((df_e >= 5) | (df_t >= 100))
        Em_nb = Em @ sp.diags(keep.astype(np.float32))
        esz = np.asarray(Em_nb.sum(1)).ravel()
        S = (Tm @ Em_nb.T).tocoo()
        cov = S.data / np.maximum(esz[S.col], 1)
        sel = (cov >= thr) & (S.data >= 3)
        for d, e, c, s in sorted(zip(S.row[sel], S.col[sel], cov[sel], S.data[sel], strict=True), key=lambda x: (x[0], -x[1])):
            hits.append(dict(custom_id=rows[d]["cid"], det=f"ng{n}", item=items[e], score=round(float(c), 3),
                             shared=int(s), action="drop"))
        print(f"ng{n}: hits {int(sel.sum())}, max coverage {float(cov.max()) if len(cov) else 0:.3f}", flush=True)
    return hits


def id_hits(rows, E, ev):
    ek = collections.defaultdict(list)
    for r in E.itertuples():
        if r.eval_set in ("codeforces", "openr1_cf_test", "deepcoder_cf_test") or (r.eval_set == "lcb" and r.platform == "codeforces"):
            ek["cf:" + r.eval_id.replace("_", "/")].append(r.item)
        elif r.eval_set == "lcb" and r.platform == "atcoder":
            ek["at:" + r.eval_id.lower()].append(r.item)
    for e in ev:
        if e["eval_set"] == "lcb" and e.get("platform") == "leetcode" and e.get("title"):
            ek["lc:" + e["title"].lower()].append(f"lcb|{e['eval_id']}")
    hits = []
    for r in rows:
        keys = set()
        # carry-over rows from open-r1 Codeforces keep their problem id only here (no url, no upstream_pid)
        if r["openr1_id"]:
            keys.add("cf:" + r["openr1_id"])
        for s in (r["url"], r["upstream_pid"]):
            for a, b in CF_URL.findall(s):
                keys.add(f"cf:{a}/{b.upper()}")
            for t in AT_URL.findall(s):
                keys.add("at:" + t.lower())
        if r["upstream_pid"].startswith("openr1_cf/"):
            keys.add("cf:" + r["upstream_pid"].split("/", 1)[1])
        if r["cid"].startswith("v8_leetcode_"):
            keys.add("lc:" + r["cid"][len("v8_leetcode_"):].lower())
        for k in sorted(keys):
            for it in ek.get(k, []):
                hits.append(dict(custom_id=r["cid"], det="id", item=it, score=1.0, shared=0, action="drop"))
    return hits


def test_hits(rows, ev):
    idx, cidx = collections.defaultdict(set), collections.defaultdict(set)
    for e in ev:
        it = f"{e['eval_set']}|{e['eval_id']}"
        if e["eval_set"] in ("lcb", "codeforces"):
            for h, n in e["io"]:
                if n >= 10:
                    idx[h].add(it)
        for fname, h, n in e["calls"]:
            if n >= 12:
                cidx[h].add((fname, it))
    idx = {h: sorted(s) for h, s in idx.items() if len(s) < 3}
    cidx = {h: sorted(s) for h, s in cidx.items()}
    hits = []
    for r in rows:
        if r["is_io"]:
            cnt = collections.Counter(it for h in r["io"] for it in idx.get(h, ()))
            for it, k in cnt.items():
                hits.append(dict(custom_id=r["cid"], det="test_io", item=it, score=k, shared=k,
                                 action="drop" if k >= 4 else "report"))
        elif r["calls"]:
            same, anyc = collections.Counter(), collections.Counter()
            for fn, h in r["calls"]:
                for ename, it in cidx.get(h, ()):
                    anyc[it] += 1
                    if ename == fn:
                        same[it] += 1
            for it, k in anyc.items():
                # every confirmed function-style contamination kept the eval's function name; different-name
                # overlaps were trivial calls (f(1) == True)
                he_mbpp = not it.startswith(("lcb|", "cruxeval|"))
                act = "drop" if same[it] >= 3 or (same[it] and he_mbpp) else "report"
                hits.append(dict(custom_id=r["cid"], det="test_call", item=it, score=k, shared=same[it], action=act))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train")
    ap.add_argument("--eval-items", required=True)
    ap.add_argument("--eval-hashes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cutoff", default="2024-08-01")
    a = ap.parse_args()
    E = pd.read_parquet(a.eval_items)
    ev = list(read_jsonl(a.eval_hashes))
    rows = [features(r) for r in read_jsonl(a.train)]
    hits = ngram_hits(rows, E) + id_hits(rows, E, ev) + test_hits(rows, ev)
    for r in rows:
        if r["date"] and str(r["date"])[:10] >= a.cutoff:
            hits.append(dict(custom_id=r["cid"], det="date", item=str(r["date"])[:10], score=0, shared=0, action="drop"))
    H = pd.DataFrame(hits, columns=["custom_id", "det", "item", "score", "shared", "action"])
    os.makedirs(a.out_dir, exist_ok=True)
    H.to_csv(os.path.join(a.out_dir, "decontam_hits.csv"), index=False)
    drop = sorted(set(H[H.action == "drop"].custom_id))
    with open(os.path.join(a.out_dir, "decontam_drop.txt"), "w") as f:
        f.write("".join(x + "\n" for x in drop))
    print("hits by detector/action:", H.groupby(["det", "action"]).size().to_dict())
    print("rows dropped:", len(drop), "of", len(rows))


if __name__ == "__main__":
    sys.exit(main())
