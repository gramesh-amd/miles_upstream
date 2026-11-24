#!/usr/bin/env python3
"""Download one upstream source at a pinned revision, keeping only the problems in the union's clusters.

    python3 fetch.py --source hardtests_tests --union union.csv.gz --clusters clusters.parquet --work WORK

Each parquet shard is downloaded, filtered to the wanted ids, written to
WORK/extract/<source>/<shard>.jsonl.gz and deleted, so peak disk stays near one shard per worker.
A shard whose output exists is skipped, so the command is resumable, and readers take the outputs in
file-name order, so row order does not depend on download timing. --part k/n splits the shards
across hosts that share WORK.

The test-heavy sources are trimmed on the fly to what the build can use (the sandbox receives every
stored test, so each problem is capped at --max-tests tests and --max-bytes bytes, the 8 largest
inputs first because select_tests always takes them; a single test over --max-one bytes is skipped):
  openr1_tests     generated_tests cleaned and capped; official_tests dropped (openr1_verifiable has them)
  hardtests_tests  test_cases decoded into a plain list and capped; only output_judging_function kept
  ccplus_1x        test_cases capped; submissions reduced to <= 16 Python 3 ones that compile
"""
import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from common import clean_tests, decode_hardtests, is_python, load_union

REVISIONS = {
    "open-r1/codeforces": "fbe3f6e903ee854eec2e69e9d96d0306cde59baf",
    "open-r1/codeforces-submissions": "07e7bf31f284b0e90e02dfc7ab675d5821e1c068",
    "sigcp/hardtests_problems": "06393e565b836f072fb5b9f18cad668e641337cd",
    "sigcp/hardtests_tests": "8c6c4fa10732762dd38d30c62523abe6b6f81750",
    "ByteDance-Seed/Code-Contests-Plus": "96c850540fade31d384a25766461e0da6b08f5fc",
    "agentica-org/DeepCoder-Preview-Dataset": "177913a7bd43791646ef6a43645caa3c871ab3db",
}

# repo, file prefix, id column, union source whose cluster members are wanted, columns to keep (None = all).
# Sources without an id column are kept whole: DeepCoder pids are <file>/<row index>.
SOURCES = {
    "openr1_verifiable": ("open-r1/codeforces", "verifiable/", "id", "openr1_cf_verifiable", None),
    "openr1_tests": ("open-r1/codeforces", "verifiable_tests/", "id", "openr1_cf_verifiable",
                     ["id", "official_tests", "official_tests_complete", "generated_checker", "input_mode",
                      "generated_tests"]),
    "hardtests_problems": ("sigcp/hardtests_problems", "data/", "pid", "hardtests", None),
    "hardtests_tests": ("sigcp/hardtests_tests", "data/", "pid", "hardtests", None),
    "ccplus_1x": ("ByteDance-Seed/Code-Contests-Plus", "ccplus_1x/", "id", "ccplus", None),
    "cf_subs": ("open-r1/codeforces-submissions", "data/", "problem_id", "openr1_cf_verifiable", None),
    "deepcoder_taco": ("agentica-org/DeepCoder-Preview-Dataset", "taco/", None, None, None),
    "deepcoder_pi": ("agentica-org/DeepCoder-Preview-Dataset", "primeintellect/", None, None, None),
}


def keep_row(src, r):
    # submissions are only reference candidates and wrong-answer probes
    if src == "cf_subs":
        return "py" in (r.get("programmingLanguage") or "").lower() and r.get("verdict") in ("OK", "WRONG_ANSWER")
    return True


def cap(tests, a):
    tests = [t for t in clean_tests(tests) if len(t["input"]) + len(t["output"]) <= a.max_one]
    big = sorted(range(len(tests)), key=lambda i: -len(tests[i]["input"]))[:8]
    order = big + [i for i in range(len(tests)) if i not in set(big)]
    out, b = [], 0
    for i in order:
        t = tests[i]
        sz = len(t["input"]) + len(t["output"])
        if len(out) >= a.max_tests or (out and b + sz > a.max_bytes):
            continue
        out.append(t)
        b += sz
    return out


def slim(src, r, a):
    if src == "openr1_tests":
        r["generated_tests"] = cap(r.get("generated_tests") or [], a)
        r.pop("official_tests", None)
    elif src == "hardtests_tests":
        try:
            r["test_cases"] = cap(decode_hardtests(r["test_cases"]), a)
        except Exception:
            r["test_cases"] = []
        kit = r.get("test_cases_kit") or {}
        r["test_cases_kit"] = {"output_judging_function": kit.get("output_judging_function")}
        r.pop("mapping", None)
    elif src == "ccplus_1x":
        r["test_cases"] = cap(r.get("test_cases") or [], a)
        for key in ("correct_submissions", "incorrect_submissions"):
            r[key] = [s for s in (r.get(key) or []) if s.get("language") == "py3" and is_python(s.get("code"))][:16]
    return r


def jsonable(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def read_rows(pf, rg, hit, cols):
    try:
        full = pf.read_row_group(rg, columns=cols)
        # row by row: take() concatenates and overflows 32-bit string offsets on multi-GB test columns
        return [full.slice(i, 1).to_pylist()[0] for i in hit]
    except pa.ArrowNotImplementedError:
        # a nested column too large for one array: stream small batches
        want, got, base = set(hit), {}, 0
        for b in pf.iter_batches(batch_size=4, row_groups=[rg], columns=cols):
            for j in range(b.num_rows):
                if base + j in want:
                    got[base + j] = b.slice(j, 1).to_pylist()[0]
            base += b.num_rows
        return [got[i] for i in hit]


def process(a, fname, ids, out_path):
    repo, _, idcol, _, cols = SOURCES[a.source]
    local = hf_hub_download(repo, fname, repo_type="dataset", revision=REVISIONS[repo],
                            local_dir=os.path.join(a.work, "dl", a.source))
    pf = pq.ParquetFile(local)
    names = pf.schema_arrow.names
    cols = [c for c in (cols or names) if c in names]
    kept, offset = 0, 0
    with gzip.open(out_path + ".tmp", "wt", compresslevel=3) as g:
        for rg in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if idcol is None:
                hit = list(range(n))
                rows = pf.read_row_group(rg, columns=cols).to_pylist()
            else:
                col = pf.read_row_group(rg, columns=[idcol]).column(0).to_pylist()
                hit = [i for i, x in enumerate(col) if x in ids]
                rows = read_rows(pf, rg, hit, cols) if hit else []
            for i, r in zip(hit, rows, strict=True):
                if not keep_row(a.source, r):
                    continue
                r["_file"], r["_row"] = fname, offset + i
                g.write(json.dumps(jsonable(slim(a.source, r, a)), ensure_ascii=False) + "\n")
                kept += 1
            offset += n
    os.replace(out_path + ".tmp", out_path)
    os.remove(local)
    print(f"{fname}: kept {kept}", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--union", required=True, help="union CSV (pid, source, cluster, ...)")
    ap.add_argument("--clusters", required=True, help="cluster members (pid, source, cluster), parquet or CSV")
    ap.add_argument("--work", required=True, help="work dir; output goes to WORK/extract/<source>/")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--part", default="0/1", help="k/n: take the shards with index %% n == k")
    ap.add_argument("--max-tests", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=8 << 20)
    ap.add_argument("--max-one", type=int, default=2 << 20)
    a = ap.parse_args()
    k, n = map(int, a.part.split("/"))
    repo, prefix, idcol, union_src, _ = SOURCES[a.source]
    ids = set()
    if idcol is not None:
        _, members, _ = load_union(a.union, a.clusters)
        ids = {p.split("/", 1)[1] for p in members[members.source == union_src].pid}
    out_dir = os.path.join(a.work, "extract", a.source)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=REVISIONS[repo])
                   if f.startswith(prefix) and f.endswith(".parquet"))
    todo = []
    for i, f in enumerate(files):
        out = os.path.join(out_dir, f.replace("/", "__")[:-len(".parquet")] + ".jsonl.gz")
        if i % n == k and not os.path.exists(out):
            todo.append((f, out))
    print(f"{a.source}: {len(files)} shards, {len(todo)} to do, {len(ids)} wanted ids", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        total = sum(ex.map(lambda t: process(a, t[0], ids, t[1]), todo))
    print(f"{a.source}: kept {total} rows", flush=True)


if __name__ == "__main__":
    sys.exit(main())
