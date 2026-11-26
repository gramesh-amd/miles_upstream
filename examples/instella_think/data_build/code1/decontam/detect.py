#!/usr/bin/env python3
"""Run all decontamination detectors (train rows x eval problems).

Text detectors (word tokens, LaTeX commands stripped, lowercase):
  ng13 / ng8 : |shared n-grams| / |eval n-grams| (decon/Tulu-style containment of the eval item in one train row),
               plus |shared| / |train n-grams|. Boilerplate n-grams (in >=5 distinct eval problems or >=100 train
               rows) are excluded from numerator and denominator.
  jac5       : exact Jaccard of word 5-gram shingle sets (what MinHash estimates).
  exact      : normalised eval statement equals / is a contiguous substring of the normalised train prompt.
  tfidf      : cosine of sublinear TF-IDF (word 1+2-grams); paraphrase candidate generator only.
Non-text detectors:
  func       : tested-function name (+ parameter names / count) equal to an HE/MBPP/Pro/LeetCode entry point.
  test_io    : identical (stdin, stdout) pair (normalised whitespace), non-trivial.
  test_call  : identical (args, expected) of a function call (container-insensitive), non-trivial.
  test_assert: identical assert string (whitespace removed).
  idref      : CF contest/problem ids, AtCoder task ids, LeetCode slugs/URLs, eval titles in the prompt.
Outputs work/pairs.parquet, work/per_train.parquet, work/per_eval.parquet, work/ref_pairs.parquet
"""
import collections, json, math, os, pickle, re, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

WORK = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/work"
LATEX_CMD = re.compile(r"\\(?:texttt|mathrm|text|textbf|mathbf|operatorname)\{([^}]*)\}")
LATEX = re.compile(r"\\[A-Za-z]+")
WORD = re.compile(r"[a-z0-9]+")


def tok(s):
    s = LATEX_CMD.sub(r" \1 ", s)
    s = LATEX.sub(" ", s)
    return WORD.findall(s.lower())


def ngrams(ws, n):
    return {hash(" ".join(ws[i:i + n])) for i in range(len(ws) - n + 1)}


def build_matrix(sets_list, vocab):
    rows, cols = [], []
    for i, s in enumerate(sets_list):
        for g in s:
            j = vocab.get(g)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(sets_list), len(vocab)))


def overlap(train_tok, eval_tok, eval_keys, n, boiler_eval=5, boiler_train=100, keep=None):
    """Return dense-chunk iterator stats: ratio_eval (non-boilerplate), ratio_train, shared counts."""
    E = [ngrams(w, n) for w in eval_tok]
    vocab = {}
    for s in E:
        for g in s:
            if g not in vocab:
                vocab[g] = len(vocab)
    df_e = np.zeros(len(vocab), np.int32)
    seen_key = collections.defaultdict(set)
    for k, s in zip(eval_keys, E):
        for g in s:
            seen_key[g].add(k)
    for g, ks in seen_key.items():
        df_e[vocab[g]] = len(ks)
    T_all_sizes = np.array([max(0, len(w) - n + 1) for w in train_tok], np.float32)
    T = [ngrams(w, n) for w in train_tok]
    Tm = build_matrix(T, vocab)
    df_t = np.asarray(Tm.sum(0)).ravel()
    boiler = (df_e >= boiler_eval) | (df_t >= boiler_train)
    mask = sp.diags((~boiler).astype(np.float32))
    Em = build_matrix(E, vocab)
    Em_nb = Em @ mask
    e_size_nb = np.asarray(Em_nb.sum(1)).ravel()
    e_size_all = np.array([len(s) for s in E], np.float32)
    return Tm, Em, Em_nb, e_size_nb, e_size_all, T_all_sizes, boiler.mean()


def tfidf(docs_tok):
    vocab, rows, cols, vals = {}, [], [], []
    for i, ws in enumerate(docs_tok):
        c = collections.Counter(ws + [a + "_" + b for a, b in zip(ws, ws[1:])])
        for t, v in c.items():
            j = vocab.setdefault(t, len(vocab))
            rows.append(i); cols.append(j); vals.append(1 + math.log(v))
    M = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(docs_tok), len(vocab)))
    df = np.bincount(M.indices, minlength=len(vocab))
    idf = np.log((1 + len(docs_tok)) / (1 + df)) + 1
    M = M @ sp.diags(idf.astype(np.float32))
    norms = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9
    return sp.diags(1 / norms) @ M


def text_pairs(train, evals, tag, thr):
    ttok = [tok(t["text"]) for t in train]
    etok = [tok(e["text"]) for e in evals]
    ekeys = [e.get("eval_id") or e.get("ref_id") for e in evals]
    res = {}
    for n in (13, 8, 5):
        Tm, Em, Em_nb, esz_nb, esz_all, tsz, bfrac = overlap(ttok, etok, ekeys, n)
        print(f"[{tag}] n={n} vocab={Em.shape[1]} boilerplate_frac={bfrac:.3f}", flush=True)
        res[n] = (Tm, Em, Em_nb, esz_nb, esz_all, tsz)
    TF = tfidf(ttok + etok)
    TFt, TFe = TF[:len(train)], TF[len(train):]
    tnorm = [" " + " ".join(w) + " " for w in ttok]
    enorm = [" " + " ".join(w) + " " for w in etok]
    out = []
    per_eval = {k: np.zeros(len(evals), np.float32) for k in ("ng13", "ng8", "jac5", "tfidf")}
    per_eval_arg = {k: np.full(len(evals), -1, np.int64) for k in per_eval}
    per_train = {k: np.zeros(len(train), np.float32) for k in ("ng13", "ng8", "jac5", "tfidf", "ng13_train")}
    per_train_arg = {k: np.full(len(train), -1, np.int64) for k in per_train}
    CH = 1500
    for a in range(0, len(train), CH):
        b = min(len(train), a + CH)
        m = {}
        for n in (13, 8):
            Tm, Em, Em_nb, esz_nb, esz_all, tsz = res[n]
            shared = (Tm[a:b] @ Em_nb.T).toarray()
            m[f"ng{n}"] = np.where(esz_nb[None, :] > 0, shared / np.maximum(esz_nb[None, :], 1), 0)
            m[f"ng{n}_train"] = shared / np.maximum(tsz[a:b, None], 1)
        Tm, Em, Em_nb, esz_nb, esz_all, tsz = res[5]
        sh5 = (Tm[a:b] @ Em.T).toarray()
        m["jac5"] = sh5 / np.maximum(tsz[a:b, None] + esz_all[None, :] - sh5, 1)
        m["r5all"] = sh5 / np.maximum(esz_all[None, :], 1)
        m["tfidf"] = (TFt[a:b] @ TFe.T).toarray()
        for k in per_eval:
            col = m[k].max(0)
            arg = m[k].argmax(0) + a
            upd = col > per_eval[k]
            per_eval[k][upd] = col[upd]
            per_eval_arg[k][upd] = arg[upd]
        for k in per_train:
            per_train[k][a:b] = m[k].max(1)
            per_train_arg[k][a:b] = m[k].argmax(1)
        sel = (m["ng13"] >= thr["ng13"]) | (m["ng8"] >= thr["ng8"]) | (m["jac5"] >= thr["jac5"]) | \
              (m["tfidf"] >= thr["tfidf"]) | (m["r5all"] >= 0.95)
        ii, jj = np.nonzero(sel)
        for i, j in zip(ii, jj):
            gi = a + i
            ex = 0
            if m["r5all"][i, j] >= 0.95 and len(etok[j]) >= 5:
                if enorm[j] == tnorm[gi]:
                    ex = 2
                elif enorm[j] in tnorm[gi]:
                    ex = 1
            out.append((gi, j, float(m["ng13"][i, j]), float(m["ng13_train"][i, j]), float(m["ng8"][i, j]),
                        float(m["ng8_train"][i, j]), float(m["jac5"][i, j]), float(m["tfidf"][i, j]), ex,
                        len(etok[j])))
        print(f"[{tag}] rows {b}/{len(train)} pairs {len(out)}", flush=True)
    pairs = pd.DataFrame(out, columns=["ti", "ei", "ng13", "ng13_train", "ng8", "ng8_train", "jac5", "tfidf",
                                       "exact", "eval_len"])
    return pairs, per_eval, per_eval_arg, per_train, per_train_arg


# ------------------------------------------------------------------ tests
def test_pairs(train, evals):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_corpus import parse_assert, norm_io, norm_assert, io_key, call_hash
    io_idx, call_idx, as_idx = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)
    for j, e in enumerate(evals):
        for h, _ in e["io"]:
            io_idx[h].add(j)
        for c in e["calls"]:
            call_idx[c[1]].add(j)
        for s in e["asserts"]:
            as_idx[s].add(j)
    call_names = {j: {c[1]: c[0] for c in e["calls"]} for j, e in enumerate(evals)}

    def nontriv(n, js, minlen):
        return n >= minlen and len({evals[j]["eval_id"] for j in js}) < 3
    rec = collections.defaultdict(lambda: collections.Counter())
    for i, t in enumerate(train):
        for which, tests in (("label", t["tests_label"]), ("full", t["tests_full"])):
            if not isinstance(tests, list):
                continue
            seen = set()
            for x in tests:
                if isinstance(x, dict):
                    h, n = io_key(norm_io(x["input"]), norm_io(x["output"]))
                    if h in seen:
                        continue
                    seen.add(h)
                    js = io_idx.get(h)
                    if js and nontriv(n, js, 10):
                        for j in js:
                            rec[(i, j)][f"io_{which}"] += 1
                elif isinstance(x, str):
                    s = norm_assert(x)
                    for j in as_idx.get(s, ()):
                        if s not in seen:
                            rec[(i, j)][f"assert_{which}"] += 1
                    pk = parse_assert(x)
                    seen.add(s)
                    if pk:
                        _, h, n = call_hash(pk)
                        if ("c", h) in seen:
                            continue
                        seen.add(("c", h))
                        js = call_idx.get(h)
                        if js and nontriv(n, js, 12):
                            for j in js:
                                rec[(i, j)][f"call_{which}"] += 1
                                if call_names[j].get(h) == pk[0]:
                                    rec[(i, j)][f"callname_{which}"] += 1
    rows = []
    for (i, j), c in rec.items():
        ntest = len(evals[j]["io"]) + len(evals[j]["calls"]) + len(evals[j]["asserts"])
        rows.append(dict(ti=i, ei=j, n_eval_tests=ntest, **c))
    df = pd.DataFrame(rows).fillna(0)
    print("test-overlap pairs", len(df))
    return df


GENERIC = {"f", "solve", "main", "solution", "func", "function", "foo", "helper", "test", "run", "answer", "calc",
           "compute", "process", "check", "result", "get", "fn"}


def func_pairs(train, evals):
    idx = collections.defaultdict(list)
    for j, e in enumerate(evals):
        for name, params in e["funcs"].items():
            if name.lower() in GENERIC or name.startswith("_"):
                continue
            idx[name].append((j, params))
    rows = []
    for i, t in enumerate(train):
        for name, params in t["funcs"].items():
            for j, ep in idx.get(name, ()):
                if params and ep and tuple(params) == tuple(ep):
                    sc = 1.0
                elif params and ep and len(params) == len(ep):
                    sc = 0.75
                else:
                    sc = 0.5
                rows.append(dict(ti=i, ei=j, func=name, func_score=sc, train_params=",".join(params),
                                 eval_params=",".join(ep)))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("func_score", ascending=False).drop_duplicates(["ti", "ei"])
    print("func pairs", len(df))
    return df


# ------------------------------------------------------------------ ids / titles
ID_PATTERNS = {
    "cf_url": re.compile(r"codeforces\.(?:com|ru)/(?:contest|problemset/problem|gym)/(\d+)/(?:problem/)?([A-Za-z]\d?)", re.I),
    "cf_ref": re.compile(r"\b(?:codeforces|CF)\s*(?:round\s*)?#?\s*(\d{3,4})\s*(?:problem\s*)?([A-H]\d?)\b"),
    "cf_round": re.compile(r"codeforces\s+(?:round|global round|educational round|beta round)[^\n]{0,30}", re.I),
    "lc_url": re.compile(r"leetcode\.(?:com|cn)/problems/([a-z0-9\-]+)", re.I),
    "lc_mention": re.compile(r"\bleetcode\b", re.I),
    "ac_url": re.compile(r"atcoder\.jp/contests/([a-z0-9_\-]+)(?:/tasks/([a-z0-9_]+))?", re.I),
    "ac_id": re.compile(r"\b(abc|arc|agc)[ _-]?(\d{3})(?:[ _-]([a-h]))?\b", re.I),
    "ac_name": re.compile(r"atcoder\s+(beginner|regular|grand)\s+contest\s*(\d+)", re.I),
    "url_any": re.compile(r"https?://[^\s)>\]]+", re.I),
}


def id_pairs(train, evals, ttok_norm):
    lcb_by_id = {e["eval_id"]: j for j, e in enumerate(evals) if e["eval_set"] == "lcb"}
    cf_by_id = {e["eval_id"]: j for j, e in enumerate(evals) if e["eval_set"] == "codeforces"}
    # titles: >=3 content words
    titles = collections.defaultdict(list)
    for j, e in enumerate(evals):
        if e["eval_set"] in ("lcb", "codeforces") and e.get("title"):
            ti = re.sub(r"^[A-Z]\d?\.\s+", "", e["title"])
            ti = ti.replace("-", " ") if e["platform"] == "leetcode" else ti
            ws = tok(ti)
            if len(ws) >= 3:
                titles[" " + " ".join(ws) + " "].append(j)
    rows, hits = [], []
    for i, t in enumerate(train):
        text = t["text"]
        for name, pat in ID_PATTERNS.items():
            for m in pat.finditer(text):
                hits.append(dict(ti=i, kind=name, match=m.group(0)[:120]))
                g = m.groups()
                cand = []
                if name in ("cf_url", "cf_ref"):
                    pid = f"{g[0]}/{g[1].upper()}"
                    cand += [cf_by_id[pid]] if pid in cf_by_id else []
                    q = f"{g[0]}_{g[1].upper()}"
                    cand += [lcb_by_id[q]] if q in lcb_by_id else []
                if name == "lc_url":
                    cand += [j for j, e in enumerate(evals) if e["platform"] == "leetcode" and e["title"] == g[0].lower()]
                if name == "ac_url" and g[1]:
                    cand += [lcb_by_id[g[1].lower()]] if g[1].lower() in lcb_by_id else []
                if name == "ac_id" and g[2]:
                    q = f"{g[0].lower()}{g[1]}_{g[2].lower()}"
                    cand += [lcb_by_id[q]] if q in lcb_by_id else []
                for j in cand:
                    rows.append(dict(ti=i, ei=j, idref=name, idmatch=m.group(0)[:80]))
        s = ttok_norm[i]
        for ti_, js in titles.items():
            if ti_ in s:
                for j in js:
                    rows.append(dict(ti=i, ei=j, idref="title", idmatch=ti_.strip()))
    print("id/title pairs", len(rows), "raw id hits", len(hits))
    return pd.DataFrame(rows), pd.DataFrame(hits)


if __name__ == "__main__":
    stages = set(sys.argv[1:]) or {"text", "tests", "func", "ids", "ref"}
    train = pickle.load(open(f"{WORK}/train.pkl", "rb"))
    evals = pickle.load(open(f"{WORK}/eval.pkl", "rb"))
    if "text" in stages:
        thr = dict(ng13=0.10, ng8=0.25, jac5=0.15, tfidf=0.45)
        pairs, pe, pea, pt, pta = text_pairs(train, evals, "eval", thr)
        pairs.to_parquet(f"{WORK}/text_pairs.parquet")
        pd.DataFrame({**pe, **{k + "_arg": v for k, v in pea.items()}}).to_parquet(f"{WORK}/per_eval_text.parquet")
        pd.DataFrame({**pt, **{k + "_arg": v for k, v in pta.items()}}).to_parquet(f"{WORK}/per_train_text.parquet")
    if "tests" in stages:
        test_pairs(train, evals).to_parquet(f"{WORK}/test_pairs.parquet")
    if "func" in stages:
        func_pairs(train, evals).to_parquet(f"{WORK}/func_pairs.parquet")
    if "ids" in stages:
        tn = [" " + " ".join(tok(t["text"])) + " " for t in train]
        idp, hits = id_pairs(train, evals, tn)
        idp.to_parquet(f"{WORK}/id_pairs.parquet")
        hits.to_parquet(f"{WORK}/id_hits.parquet")
    if "ref" in stages:
        refs = []
        for f in ("ref_cf.pkl", "ref_lc.pkl"):
            refs += pickle.load(open(f"{WORK}/{f}", "rb"))
        rp, *_ = text_pairs(train, refs, "ref", dict(ng13=0.2, ng8=0.4, jac5=0.3, tfidf=0.6))
        rp.to_parquet(f"{WORK}/ref_pairs.parquet")
        pickle.dump(refs, open(f"{WORK}/refs.pkl", "wb"))
