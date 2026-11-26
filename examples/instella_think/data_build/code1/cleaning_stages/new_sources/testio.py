#!/usr/bin/env python3
"""Test-overlap detector (no execution): identical normalised (stdin, stdout) pairs shared with LCB / Codeforces-eval
hidden tests. Same hashing and triviality rule as v7 decontam/scripts/detect.py (input >= 10 chars, pair in < 3 eval
problems). Train tests: open-r1/codeforces official_tests (all train problems) + inline tests in /tmp/datacand/norm.
Output work/test_hits.parquet (uid, eval item, shared)."""
import collections, glob, hashlib, json, pickle
import pandas as pd
import pyarrow.parquet as pq

N = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/new_sources"
DEC = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam"


def sh(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def norm_io(s):
    s = str(s).replace("\r", "")
    return "\n".join(l.rstrip() for l in s.strip().split("\n"))


def io_key(i, o):
    return sh(i + "\x00" + o), len(i)


ev = pickle.load(open(f"{DEC}/work/eval.pkl", "rb"))
idx = collections.defaultdict(set)
for e in ev:
    if e["eval_set"] in ("lcb", "codeforces"):
        io = e["io"] if isinstance(e["io"], list) else eval(e["io"])
        for h, n in io:
            if n >= 10:
                idx[h].add(f"{e['eval_set']}|{e['eval_id']}")
idx = {h: s for h, s in idx.items() if len(s) < 3}
print("eval io keys", len(idx), flush=True)


def scan(uid, tests, out):
    c = collections.Counter()
    seen = set()
    for t in tests or []:
        if not isinstance(t, dict):
            continue
        i, o = t.get("input"), t.get("output")
        if isinstance(i, list):
            i = "\n".join(map(str, i))
        if isinstance(o, list):
            o = "\n".join(map(str, o))
        if i is None or o is None:
            continue
        h, n = io_key(norm_io(i), norm_io(o))
        if h in seen or n < 10:
            continue
        seen.add(h)
        for it in idx.get(h, ()):
            c[it] += 1
    for it, k in c.items():
        out.append(dict(uid=uid, item=it, shared=k, n_tests=len(seen)))


out = []
for f in sorted(glob.glob("/tmp/datacand/raw/open-r1__codeforces/data/train-*.parquet")):
    d = pd.read_parquet(f, columns=["id", "official_tests"])
    for r in d.itertuples():
        scan(f"openr1_cf/{r.id}", list(r.official_tests) if r.official_tests is not None else [], out)
print("openr1 done", len(out), flush=True)
for f in ["deepcoder", "taco_verified", "klear", "synthetic2", "skywork", "nemotron_rl", "eurus", "rstar", "ccplus",
          "hardtests"]:
    p = pq.ParquetFile(f"/tmp/datacand/norm/{f}.parquet")
    for rg in range(p.num_row_groups):
        t = p.read_row_group(rg, columns=["pid", "tests"]).to_pandas()
        for r in t.itertuples():
            tests = r.tests
            if isinstance(tests, str):
                try:
                    tests = json.loads(tests)
                except Exception:
                    tests = None
            if tests is None:
                continue
            scan(r.pid, list(tests), out)
    print(f, "done", len(out), flush=True)
T = pd.DataFrame(out)
T.to_parquet(f"{N}/work/test_hits.parquet")
print(T[T.shared >= 3].sort_values("shared", ascending=False).head(40).to_string())
print(T.groupby(pd.cut(T.shared, [0, 1, 2, 3, 5, 10, 1000])).size())
