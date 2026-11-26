#!/usr/bin/env python3
"""Eval decontamination + dedup against our rows + cross-source clusters.

Tokeniser and eval corpus are the v7 decontam ones (decontam/scripts/detect.py tok; decontam/work/eval.pkl:
LCB all versions, Codeforces eval 422, HE+/MBPP+, HE-Pro/MBPP-Pro raw+new, CRUXEval), plus the open-r1 CF test
split (468) and DeepCoder's CF test set (408) from the v5 eval corpus.

Eval side (Tulu 3 / OLMo 3 style, identical to v7 REPORT.md): ng8 = |shared 8-grams| / |eval 8-grams|, ng13 same
with 13-grams; boilerplate n-grams (in >= 5 eval problems or >= 100 corpus rows) removed from both. Flags:
ng8 >= 0.5 (drop rule) and ng13 >= 0.2 (recall flag, also dropped here since there is no hand check yet).
ID side: Codeforces contest/index or AtCoder task id equal to an eval item.

Dedup side: 8-gram hashes sub-sampled 1/4 (hash mod 4 == 0), n-grams in > 100 rows dropped, containment
|shared| / min(|A|, |B|) >= 0.5 with min(|A|,|B|) >= 8 sampled grams; plus equal CF/AtCoder ids. Union-find.
Outputs work/decontam.parquet (per corpus row), work/clusters.parquet (uid -> cluster), work/edges.parquet.
"""
import os, pickle, re, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Pool

N = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/new_sources"
DEC = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam"
sys.path.insert(0, f"{DEC}/scripts")
from detect import tok  # noqa: E402


def grams(ws, n):
    if len(ws) < n:
        return np.zeros(0, np.int64)
    return np.unique(np.fromiter((hash(" ".join(ws[i:i + n])) for i in range(len(ws) - n + 1)), np.int64))


def feat(text):
    ws = tok(text or "")
    return grams(ws, 8), grams(ws, 13)


def main():
    C = pd.read_parquet(f"{N}/work/corpus.parquet", columns=["pid", "source", "statement", "pkey"])
    O = pd.read_parquet(f"{N}/work/ours.parquet")
    ev = pickle.load(open(f"{DEC}/work/eval.pkl", "rb"))
    E = pd.DataFrame([dict(eval_set=e["eval_set"], eval_id=str(e["eval_id"]), part=e["part"], text=e["text"],
                           platform=e.get("platform"), date=e.get("date")) for e in ev])
    v5e = pd.read_parquet("/tmp/datacand/norm/eval.parquet")
    v5e = v5e[v5e.bench.isin(["openr1_cf_test", "deepcoder_cf_test"])]
    E = pd.concat([E, pd.DataFrame(dict(eval_set=v5e.bench, eval_id=v5e.eval_id.str.split("/", n=1).str[1],
                                        part="statement", text=v5e.text, platform="codeforces", date=v5e.date))],
                  ignore_index=True)
    E["item"] = E.eval_set + "|" + E.eval_id
    E.to_parquet(f"{N}/work/eval_items.parquet")
    print("eval items", E.groupby("eval_set").size().to_dict(), flush=True)

    docs = list(C.statement) + list(O.text)
    uid = list(C.pid) + list("ours:" + O.set + ":" + O.oid)
    nC = len(C)
    with Pool(30) as p:
        F = p.map(feat, docs, chunksize=500)
        FE = p.map(feat, list(E.text), chunksize=50)
    print("featurised", len(F), flush=True)

    # ------------------------------------------------ eval containment (sparse)
    res = {}
    for ni, n in enumerate((8, 13)):
        vocab = {}
        for f in FE:
            for g in f[ni].tolist():
                vocab.setdefault(g, len(vocab))
        items = E.item.values
        uitem = {k: i for i, k in enumerate(pd.unique(items))}
        er, ec = [], []
        for i, f in enumerate(FE):
            for g in f[ni].tolist():
                er.append(i); ec.append(vocab[g])
        Em = sp.csr_matrix((np.ones(len(er), np.float32), (er, ec)), shape=(len(E), len(vocab)))
        # eval df counted over distinct eval problems
        prob = np.array([uitem[k] for k in items])
        Pm = sp.csr_matrix((np.ones(len(E)), (prob, np.arange(len(E)))), shape=(len(uitem), len(E))) @ Em
        df_e = np.asarray((Pm > 0).sum(0)).ravel()
        tr, tc = [], []
        for i, f in enumerate(F):
            for g in f[ni].tolist():
                j = vocab.get(g)
                if j is not None:
                    tr.append(i); tc.append(j)
        Tm = sp.csr_matrix((np.ones(len(tr), np.float32), (tr, tc)), shape=(len(F), len(vocab)))
        df_t = np.asarray(Tm[:nC].sum(0)).ravel()
        keep = ~((df_e >= 5) | (df_t >= 100))
        Em_nb = Em @ sp.diags(keep.astype(np.float32))
        esz = np.asarray(Em_nb.sum(1)).ravel()
        S = (Tm @ Em_nb.T).tocoo()
        cov = S.data / np.maximum(esz[S.col], 1)
        res[n] = pd.DataFrame(dict(doc=S.row, e=S.col, shared=S.data, cov=cov))
        print(f"n={n} vocab={len(vocab)} boiler={1 - keep.mean():.3f} pairs={len(S.data)}", flush=True)
    hits = []
    for n, thr in ((8, 0.5), (13, 0.2)):
        r = res[n]
        r = r[(r["cov"] >= thr) & (r.shared >= 3)]
        hits.append(r.assign(det=f"ng{n}"))
    H = pd.concat(hits)
    H["uid"] = np.array(uid)[H.doc.values]
    H["eval_set"] = E.eval_set.values[H.e.values]
    H["eval_id"] = E.eval_id.values[H.e.values]
    H["eval_text"] = E.text.values[H.e.values].astype(str)
    H["doc_text"] = [docs[d][:600] for d in H.doc.values]
    H["eval_text"] = H.eval_text.str[:600]
    H.drop(columns=["doc", "e"]).to_parquet(f"{N}/work/text_hits.parquet")
    print("text hits", H.groupby(["det", "eval_set"]).size().to_dict(), flush=True)

    # per-doc max coverage
    mx = {}
    for n in (8, 13):
        r = res[n]
        g = r.sort_values("cov").drop_duplicates("doc", keep="last")
        a = np.zeros(len(F)); a[g.doc.values] = g["cov"].values
        b = np.full(len(F), "", object); b[g.doc.values] = E.item.values[g.e.values]
        mx[f"ng{n}_max"], mx[f"ng{n}_arg"] = a, b

    # ID hits
    ek = set()
    for r in E.itertuples():
        if r.eval_set in ("codeforces", "openr1_cf_test"):
            ek.add(("cf:" + r.eval_id.replace("_", "/"), r.item))
        elif r.eval_set == "lcb" and r.platform == "codeforces":
            ek.add(("cf:" + r.eval_id.replace("_", "/"), r.item))
        elif r.eval_set == "lcb" and r.platform == "atcoder":
            ek.add(("at:" + r.eval_id.lower(), r.item))
    ekd = {}
    for k, it in ek:
        ekd.setdefault(k, []).append(it)
    pk = list(C.pkey) + [None] * len(O)
    id_hit = np.array(["|".join(ekd[k]) if k in ekd else "" for k in pk], object)
    print("id hits", int((id_hit != "").sum()), flush=True)

    # ------------------------------------------------ dedup (sampled 8-gram containment)
    hs, ds = [], []
    for i, f in enumerate(F):
        g = f[0][(f[0] & 3) == 0]
        hs.append(g); ds.append(np.full(len(g), i, np.int32))
    hs = np.concatenate(hs); ds = np.concatenate(ds)
    o = np.argsort(hs, kind="stable"); hs, ds = hs[o], ds[o]
    b = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1], True])
    sz = np.diff(b)
    nsamp = np.bincount(ds, minlength=len(F))
    ok_g = (sz >= 2) & (sz <= 100)
    print("sampled grams", len(hs), "multi-doc groups", int(ok_g.sum()), "pair occ", int((sz[ok_g] * (sz[ok_g] - 1) // 2).sum()), flush=True)
    # non-boilerplate sampled size per doc
    bad = np.repeat(sz > 100, sz)
    nsamp_nb = np.bincount(ds[~bad], minlength=len(F))
    acc = []
    starts = b[:-1][ok_g]; sizes = sz[ok_g]
    for k in np.unique(sizes):
        st = starts[sizes == k]
        M = ds[st[:, None] + np.arange(k)[None, :]]  # groups x k
        iu, ju = np.triu_indices(k, 1)
        a, c = M[:, iu].ravel().astype(np.int64), M[:, ju].ravel().astype(np.int64)
        lo, hi = np.minimum(a, c), np.maximum(a, c)
        key = (lo << 20) | hi
        u, cnt = np.unique(key, return_counts=True)
        acc.append((u, cnt))
    keys = np.concatenate([u for u, _ in acc]); cnts = np.concatenate([c for _, c in acc])
    o = np.argsort(keys, kind="stable"); keys, cnts = keys[o], cnts[o]
    bb = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    keys = keys[bb]; cnts = np.add.reduceat(cnts, bb)
    lo, hi = (keys >> 20).astype(np.int64), (keys & ((1 << 20) - 1)).astype(np.int64)
    mn = np.minimum(nsamp_nb[lo], nsamp_nb[hi])
    cont = cnts / np.maximum(mn, 1)
    sel = (cont >= 0.5) & (mn >= 8)
    edges = pd.DataFrame(dict(a=lo[sel], b=hi[sel], cont=cont[sel], shared=cnts[sel]))
    # id edges
    kdf = pd.DataFrame(dict(k=pk, i=np.arange(len(pk)))).dropna()
    first = kdf.groupby("k").i.transform("min")
    ide = pd.DataFrame(dict(a=first.values, b=kdf.i.values, cont=1.0, shared=-1))
    ide = ide[ide.a != ide.b]
    edges = pd.concat([edges, ide], ignore_index=True)
    print("text edges", int(sel.sum()), "id edges", len(ide), flush=True)

    par = np.arange(len(F))

    def find(x):
        r = x
        while par[r] != r:
            r = par[r]
        while par[x] != r:
            par[x], x = r, par[x]
        return r
    for a, c in zip(edges.a.values, edges.b.values):
        ra, rc = find(a), find(c)
        if ra != rc:
            par[max(ra, rc)] = min(ra, rc)
    cl = np.array([find(i) for i in range(len(F))])
    edges["ua"] = np.array(uid)[edges.a.values]; edges["ub"] = np.array(uid)[edges.b.values]
    edges.to_parquet(f"{N}/work/edges.parquet")
    out = pd.DataFrame(dict(uid=uid, cluster=cl, nsamp=nsamp, nsamp_nb=nsamp_nb, id_hit=id_hit, **mx))
    out.to_parquet(f"{N}/work/overlap.parquet")
    vc = pd.Series(cl).value_counts()
    print("clusters", len(vc), "largest", vc.head(5).to_dict(), flush=True)


if __name__ == "__main__":
    main()
