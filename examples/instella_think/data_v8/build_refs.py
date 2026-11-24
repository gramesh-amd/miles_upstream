#!/usr/bin/env python3
"""Reference solutions from two whole-file sources, at pinned revisions.

    python3 build_refs.py --union union.csv.gz --clusters clusters.parquet \
        --codewars-pool taco_codewars_fn.dedup.jsonl.gz --work WORK

  WORK/refs/taco_verified.jsonl.gz  likaixin/TACO-verified rows of the union's clusters (candidates for
                                    stdio_cands.py taco and cluster references for every other source)
  WORK/refs/codewars_refs.json      {Codewars pool uid: [<= 6 Python solutions]}: the pool rows carry no
                                    gold solution, so they are matched to codeparrot/apps on the kata URL
"""
import argparse
import collections
import gzip
import json
import os
import sys
import warnings

from huggingface_hub import hf_hub_download

from common import is_python, load_union, read_jsonl

TACO_VERIFIED = ("likaixin/TACO-verified", "9e68575e01938f016199eda6f07d1a7ce96b35e1", "taco_verified.json")
APPS = ("codeparrot/apps", "21e74ddf8de1a21436da12e3e653065c5213e9d1", ("train.jsonl", "test.jsonl"))
TACO_FIELDS = ("id", "question", "solutions", "input_output", "source", "url", "date", "difficulty")
MAX_CODEWARS_REFS = 6


def taco_verified(a, dl, out):
    u, members, _ = load_union(a.union, a.clusters)
    want = set(members.pid) | set(u.pid)
    repo, rev, fname = TACO_VERIFIED
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev, local_dir=dl)
    n = 0
    with gzip.open(out + ".tmp", "wt") as g:
        for d in json.load(open(path)):
            if f"taco_verified/{d['id']}" in want:
                g.write(json.dumps({k: d.get(k) for k in TACO_FIELDS}, ensure_ascii=False) + "\n")
                n += 1
    os.replace(out + ".tmp", out)
    print(f"taco_verified: {n} rows of the union's clusters -> {out}")


def codewars_refs(a, dl, out):
    repo, rev, files = APPS
    by_url = collections.defaultdict(list)
    for fname in files:
        for r in read_jsonl(hf_hub_download(repo, fname, repo_type="dataset", revision=rev, local_dir=dl)):
            if "codewars.com" in (r.get("url") or "") and r.get("solutions"):
                by_url[r["url"]] += json.loads(r["solutions"])
    refs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for r in read_jsonl(a.codewars_pool):
            m = r["metadata"]
            sols = [s for s in by_url.get(m.get("url"), []) if is_python(s)][:MAX_CODEWARS_REFS]
            if sols:
                refs[m["uid"]] = sols
    json.dump(refs, open(out, "w"))
    print(f"codewars_refs: {len(refs)} pool rows with an APPS solution -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--union", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--codewars-pool", required=True, help="the Codewars function pool (fn_build.py input)")
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    out = os.path.join(a.work, "refs")
    os.makedirs(out, exist_ok=True)
    taco_verified(a, os.path.join(a.work, "dl", "taco_verified"), os.path.join(out, "taco_verified.jsonl.gz"))
    codewars_refs(a, os.path.join(a.work, "dl", "apps"), os.path.join(out, "codewars_refs.json"))


if __name__ == "__main__":
    sys.exit(main())
