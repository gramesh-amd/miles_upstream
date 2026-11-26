"""Stage 2 (no execution): decontamination of every candidate that survived prepare.py.

Eval corpus = v7 decontam/work/eval.pkl (LCB all 1,055 problems = every version, the 422-problem Codeforces eval,
HE+/MBPP+, HE-Pro/MBPP-Pro, CRUXEval-I/O) + the open-r1 CF test split (468, as in new_sources).
Detectors, a hit drops the row:
  id     the CF id equals an eval item (Codeforces eval, LCB codeforces problems, open-r1 CF test split)
  ng8 / ng13   eval-side 8-gram coverage >= 0.5 / 13-gram >= 0.2 of the new prompt, boilerplate n-grams removed
               (in >= 5 eval problems or >= 100 train rows; same rule as new_sources/scripts/overlap.py)
  tests  >= 4 identical normalised (stdin, stdout) pairs shared with one LCB / CF-eval problem's hidden tests,
         over all of the row's tests (own + merged + Open-R1 official); 1-3 shared pairs are reported only
         (v7: false positives share 1-3 edge-case tests)
  cluster  new_sources cluster-level hit for the open-r1 problem or the Dolci row (text, id or test overlap anywhere
           in the cluster: this is what catches CF 646B reused as eval 2010/C1-C2)
Also the date cutoff (open-r1 contest_start < 2024-08-01) is re-checked.

    python3 decontam.py  ->  /tmp/v7v2/decontam.pkl
"""
import collections
import hashlib
import json
import pickle
import sys

import pandas as pd

from v2common import CUTOFF, FINAL1, V7, WORK
from analyze import body_of  # noqa: E402

sys.path.insert(0, f"{V7}/decontam/scripts")
from detect import tok  # noqa: E402

NS = f"{V7}/new_sources"
HEADLINE = {"lcb", "codeforces", "openr1_cf_test", "humanevalplus", "mbppplus", "humanevalpro", "mbpppro", "cruxeval"}


def sh(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def norm_io(s):
    s = str(s).replace("\r", "")
    return "\n".join(ln.rstrip() for ln in s.strip().split("\n"))


def grams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    R = pickle.load(open(f"{WORK}/cands.pkl", "rb"))
    live = {c: r for c, r in R.items() if not r["drop"]}
    E = pd.read_parquet(f"{NS}/work/eval_items.parquet")
    E = E[E.eval_set.isin(HEADLINE)]
    # ---- id
    eval_cf = {}
    for r in E.itertuples():
        if r.platform == "codeforces":
            eval_cf.setdefault(r.eval_id.replace("_", "/"), set()).add(r.item)
    # ---- text n-grams (eval side coverage, boilerplate removed)
    feats = {n: [grams(tok(t or ""), n) for t in E.text] for n in (8, 13)}
    items = list(E.item)
    train_docs = [body_of(json.loads(l)["prompt"][0]["content"]) for l in open(f"{FINAL1}/train.jsonl")]
    docs = {c: body_of(r["prompt"]) for c, r in live.items()}
    out = {c: dict(hits=[], test_hits_lt4=[]) for c in live}
    for n, thr in ((8, 0.5), (13, 0.2)):
        prob_df = collections.Counter()
        per_item = collections.defaultdict(set)
        for it, g in zip(items, feats[n]):
            per_item[it] |= g
        for g in per_item.values():
            prob_df.update(g)
        evocab = set(prob_df)
        tdf = collections.Counter()
        for d in train_docs + list(docs.values()):
            tdf.update(grams(tok(d), n) & evocab)
        boiler = {g for g, k in prob_df.items() if k >= 5} | {g for g, k in tdf.items() if k >= 100}
        index = collections.defaultdict(set)
        esz = {}
        for it, g in per_item.items():
            g = g - boiler
            esz[it] = len(g)
            for x in g:
                index[x].add(it)
        for c, d in docs.items():
            shared = collections.Counter()
            for x in grams(tok(d), n):
                for it in index.get(x, ()):
                    shared[it] += 1
            for it, k in shared.items():
                cov = k / max(esz[it], 1)
                if cov >= thr and k >= 3:
                    out[c]["hits"].append(f"ng{n}:{it}:{cov:.2f}")
    # ---- test I/O overlap with LCB / CF-eval hidden tests
    ev = pickle.load(open(f"{V7}/decontam/work/eval.pkl", "rb"))
    idx = collections.defaultdict(set)
    for e in ev:
        if e["eval_set"] in ("lcb", "codeforces"):
            io = e["io"] if isinstance(e["io"], list) else eval(e["io"])
            for h, n in io:
                if n >= 10:
                    idx[h].add(f"{e['eval_set']}|{e['eval_id']}")
    idx = {h: s for h, s in idx.items() if len(s) < 3}
    # ---- new_sources cluster-level hits (text / id / tests >= 4 anywhere in the cluster)
    ov = pd.read_parquet(f"{NS}/work/overlap.parquet", columns=["uid", "cluster", "id_hit"])
    th = pd.read_parquet(f"{NS}/work/text_hits.parquet", columns=["uid", "eval_set", "eval_id", "det"])
    th = th[th.eval_set.isin(HEADLINE)]
    tt = pd.read_parquet(f"{NS}/work/test_hits.parquet")
    tt = tt[tt.shared >= 4]
    uhit = collections.defaultdict(set)
    for r in th.itertuples():
        uhit[r.uid].add(f"{r.det}:{r.eval_set}|{r.eval_id}")
    for r in ov[ov.id_hit != ""].itertuples():
        uhit[r.uid].add("id:" + r.id_hit)
    for r in tt.itertuples():
        uhit[r.uid].add(f"tests{r.shared}:{r.item}")
    ucl = dict(zip(ov.uid, ov.cluster))
    clhit = collections.defaultdict(set)
    for u, h in uhit.items():
        if u in ucl:
            clhit[ucl[u]] |= h
    for c, r in live.items():
        o = out[c]
        if r["pid"] in eval_cf:
            o["hits"].append("id:" + ",".join(sorted(eval_cf[r["pid"]])))
        cnt = collections.Counter()
        seen = set()
        for t in r["tests"] + r["official_added"]:
            i, s = norm_io(t["input"]), norm_io(t["output"])
            h = sh(i + "\x00" + s)
            if len(i) < 10 or h in seen:
                continue
            seen.add(h)
            for it in idx.get(h, ()):
                cnt[it] += 1
        for it, k in cnt.items():
            (o["hits"] if k >= 4 else o["test_hits_lt4"]).append(f"tests{k}:{it}")
        for u in (f"openr1_cf/{r['pid']}", f"ours:dolci_{r['src']}:{c}"):
            if u in ucl and clhit.get(ucl[u]):
                o["hits"].append(f"cluster({u.split(':')[0]}):" + ";".join(sorted(clhit[ucl[u]]))[:200])
        o["openr1_cluster"] = ucl.get(f"openr1_cf/{r['pid']}")
        o["dolci_cluster"] = ucl.get(f"ours:dolci_{r['src']}:{c}")
        if not r["date"] or r["date"] >= CUTOFF:
            o["hits"].append("date_after_cutoff")
    pickle.dump(out, open(f"{WORK}/decontam.pkl", "wb"))
    bad = {c: o for c, o in out.items() if o["hits"]}
    print("candidates", len(live), "contaminated", len(bad), "unique pids", len({live[c]["pid"] for c in bad}))
    print(collections.Counter(h.split(":")[0].split("(")[0] for o in bad.values() for h in o["hits"]))
    for c, o in list(bad.items())[:30]:
        print(live[c]["pid"], o["hits"][:3])
    print("test hits 1-3 (reported only):", sum(bool(o["test_hits_lt4"]) for o in out.values()),
          "eg", [(live[c]["pid"], o["test_hits_lt4"][:2]) for c, o in out.items() if o["test_hits_lt4"]][:6])
    print("cluster coverage: openr1", sum(o["openr1_cluster"] is not None for o in out.values()),
          "dolci", sum(o["dolci_cluster"] is not None for o in out.values()), "of", len(out))


if __name__ == "__main__":
    main()
