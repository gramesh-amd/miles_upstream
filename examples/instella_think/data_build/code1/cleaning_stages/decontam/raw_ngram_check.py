#!/usr/bin/env python3
"""Sanity check without the boilerplate filter: raw max 13-/8-gram containment per eval set (Tulu/decon style)."""
import collections, pickle, sys
import pandas as pd
sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/scripts")
from detect import tok

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/work"
train = pickle.load(open(f"{W}/train.pkl", "rb"))
ev = pickle.load(open(f"{W}/eval.pkl", "rb"))
out = []
for n in (13, 8):
    E = [{" ".join(w[i:i + n]) for w in [tok(e["text"])] for i in range(len(w) - n + 1)} for e in ev]
    idx = collections.defaultdict(list)
    for j, s in enumerate(E):
        for g in s:
            idx[g].append(j)
    best = [0.0] * len(ev)
    arg = [-1] * len(ev)
    for i, t in enumerate(train):
        w = tok(t["text"])
        c = collections.Counter()
        for g in {" ".join(w[k:k + n]) for k in range(len(w) - n + 1)}:
            for j in idx.get(g, ()):
                c[j] += 1
        for j, k in c.items():
            r = k / len(E[j])
            if r > best[j]:
                best[j], arg[j] = r, i
    for j, e in enumerate(ev):
        out.append(dict(n=n, eval_set=e["eval_set"], eval_id=e["eval_id"], part=e["part"], raw_ratio=best[j],
                        best=train[arg[j]]["custom_id"] if arg[j] >= 0 else ""))
df = pd.DataFrame(out)
df.to_csv(f"{W}/raw_ngram_check.csv", index=False)
print(df.groupby(["n", "eval_set"]).raw_ratio.agg(["max", lambda s: (s >= .5).sum(), lambda s: (s >= .2).sum()]).round(3))
print(df.sort_values("raw_ratio", ascending=False).head(12).to_string())
