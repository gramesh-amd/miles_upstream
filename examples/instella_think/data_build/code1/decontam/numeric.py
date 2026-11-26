#!/usr/bin/env python3
"""Format-agnostic example detector: shared runs of 8 consecutive numbers between statements.

Catches problems re-hosted with a new statement format but the same worked examples (e.g. a Codeforces stdin
example re-used as a LeetCode-style `grid = [[...]], k = 11` example). Numbers are extracted in order from the
statement text; an 8-number run counts if it has >= 4 distinct values and appears in < 3 eval problems and
< 20 training rows (otherwise it is a generic sequence such as 1 2 3 4 5 6 7 8).
Output: work/num_pairs.parquet (ti, ei, num8 = number of shared runs)
"""
import collections, hashlib, pickle, re
import pandas as pd

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/work"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
N = 8


def runs(text):
    xs = NUM.findall(text)
    out = set()
    for i in range(len(xs) - N + 1):
        w = xs[i:i + N]
        if len(set(w)) >= 4:
            out.add(int.from_bytes(hashlib.blake2b(" ".join(w).encode(), digest_size=8).digest(), "little"))
    return out


def main():
    train = pickle.load(open(f"{W}/train.pkl", "rb"))
    ev = pickle.load(open(f"{W}/eval.pkl", "rb"))
    idx = collections.defaultdict(set)
    for j, e in enumerate(ev):
        for h in runs(e["text"]):
            idx[h].add(j)
    tr_runs = [runs(t["text"]) & idx.keys() for t in train]
    df_t = collections.Counter(h for s in tr_runs for h in s)
    rec = collections.Counter()
    for i, s in enumerate(tr_runs):
        for h in s:
            js = idx[h]
            if len({ev[j]["eval_id"] for j in js}) < 3 and df_t[h] < 20:
                for j in js:
                    rec[(i, j)] += 1
    df = pd.DataFrame([(i, j, n) for (i, j), n in rec.items()], columns=["ti", "ei", "num8"])
    df.to_parquet(f"{W}/num_pairs.parquet")
    df["eval_set"] = [ev[j]["eval_set"] for j in df.ei]
    print(len(df), df.groupby("eval_set").agg(pairs=("ti", "size"), evals=("ei", "nunique"), max=("num8", "max")))


if __name__ == "__main__":
    main()
