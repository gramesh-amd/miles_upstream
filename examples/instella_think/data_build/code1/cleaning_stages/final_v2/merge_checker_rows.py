#!/usr/bin/env python3
"""Merge the lux checker verification (verify_checkers.py accepted_rows.jsonl) into final_v2. Stdlib only, no execution.

  * a row whose CF id is already in train.jsonl (metadata.openr1_id of a recovered row, or the final_v1 row's
    decontam/date_check.csv ref_id), or whose prompt is a MinHash duplicate (5-gram Jaccard >= 0.7 after LaTeX
    normalisation, equal digit sequences: the final_v1 rule) of a train row, is skipped unless it has strictly more
    tests, in which case it replaces that row (verify_checkers.py already applied this to final_v1 duplicates);
  * train.jsonl is rewritten in v6 row order; the previous file is kept as train.pre_checker_merge.jsonl;
  * work/verifying_refs.json.gz gains the checker rows' references; recovered.csv gains their lines; stats.json gains
    a "checker_merge" section (REPORT.md is not edited: add the lux report's numbers by hand).

    python3 merge_checker_rows.py --accepted /tmp/v7_checkers/accepted_rows.jsonl [--final-v2 DIR] [--dry-run]
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import random
import re
import shutil

V7 = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data"
KLEAR_HEAD = "Solve the following coding problem using the programming language python:"
TAILS = ["Write Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.",
         "Now solve the problem and return the code.", "Now solve the problem by providing the code."]


def body_of(p):  # stdio/scripts/analyze.body_of
    b = p[len(KLEAR_HEAD):] if p.startswith(KLEAR_HEAD) else p
    for t in TAILS:
        i = b.rfind(t)
        if i >= 0:
            b = b[:i]
    b = re.sub(r"The input will be given via stdin and the output should be printed to stdout by your code\.\s*$", "", b.strip())
    b = re.sub(r"The input will be stdin and you should print your solution to stdout\s*$", "", b.strip())
    return b.strip()


def latex_norm(t):  # final_v1/scripts/assemble.latex_norm
    t = body_of(t).replace("$", " ")
    t = re.sub(r"\\(left|right|cdot|times|ldots|dots|cdots|quad|qquad|[,;!])", " ", t)
    t = re.sub(r"\\([A-Za-z]+)", r" \1 ", t)
    return re.sub(r"[{}^_\\]", "", t).lower()


def shingles(t):
    w = re.findall(r"[a-z0-9]+", t)
    return {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}


def minhash_sig(s, a, b, P=2 ** 31 - 1):
    hs = [int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF for x in s] or [0]
    return tuple(min((h * ai + bi) % P for h in hs) for ai, bi in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--final-v2", default=f"{V7}/final_v2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    F = a.final_v2
    train = [json.loads(l) for l in open(f"{F}/train.jsonl")]
    acc = [json.loads(l) for l in open(a.accepted)]
    ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
    v6row = {i["custom_id"]: i["row"] for i in ids}
    dc = {r["custom_id"]: r for r in csv.DictReader(open(f"{V7}/decontam/date_check.csv"))}

    def cf_of(r):
        md = r["metadata"]
        if md.get("openr1_id"):
            return md["openr1_id"]
        d = dc.get(md["custom_id"], {})
        return d.get("ref_id") if d.get("ref") == "cf" else None

    by_cf = collections.defaultdict(list)
    for r in train:
        if cf_of(r):
            by_cf[cf_of(r)].append(r["metadata"]["custom_id"])
    rng = random.Random(0)
    A = [rng.randint(1, 2 ** 31 - 2) for _ in range(128)]
    B = [rng.randint(0, 2 ** 31 - 2) for _ in range(128)]
    bands = collections.defaultdict(list)
    sets, digits, ntest = {}, {}, {}
    for r in train:
        c = r["metadata"]["custom_id"]
        t = latex_norm(r["prompt"][0]["content"])
        sets[c], digits[c], ntest[c] = shingles(t), "".join(re.findall(r"\d", t)), r["metadata"]["n_tests"]
        sig = minhash_sig(sets[c], A, B)
        for k in range(16):
            bands[(k, sig[k * 8:(k + 1) * 8])].append(c)
    remove, add, log = set(), [], collections.Counter()
    refs_new, rec_lines = {}, []
    for x in acc:
        row = x["row"]
        md = row["metadata"]
        c = md["custom_id"]
        if c in sets:
            log["already_in_train"] += 1
            continue
        t = latex_norm(row["prompt"][0]["content"])
        s, dg = shingles(t), "".join(re.findall(r"\d", t))
        sig = minhash_sig(s, A, B)
        cand = {o for k in range(16) for o in bands.get((k, sig[k * 8:(k + 1) * 8]), [])}
        dups = {o for o in cand if digits[o] == dg and len(s & sets[o]) / max(1, len(s | sets[o])) >= 0.7}
        dups |= set(by_cf.get(md["openr1_id"], []))
        dups -= set(md.get("replaces_final_v1") or [])
        if dups and md["n_tests"] <= max(ntest[o] for o in dups):
            log["skipped_duplicate_of_train_row"] += 1
            continue
        remove |= dups | set(md.get("replaces_final_v1") or [])
        md.pop("replaces_final_v1", None)
        add.append(row)
        refs_new[c] = x["ref"]
        rec_lines.append(dict(custom_id=c, src=md["src"], route="checker", task="checker(lux)", openr1_id=md["openr1_id"],
                              cf_date=md["cf_date"], n_tests=md["n_tests"],
                              n_official_added=sum(int(f.split(":")[1]) for f in md["fixes"] if f.startswith("add_openr1_official_tests")),
                              verifier_origin=x["ref"]["origin"], copies="", replaces_final_v1=",".join(sorted(remove & dups)),
                              v7_changes=";".join(md["v7_changes"])))
    log["added"], log["removed_replaced"] = len(add), len(remove)
    print(dict(log))
    if a.dry_run:
        return
    shutil.copy(f"{F}/train.jsonl", f"{F}/train.pre_checker_merge.jsonl")
    rows = [r for r in train if r["metadata"]["custom_id"] not in remove] + add
    rows.sort(key=lambda r: v6row[r["metadata"]["custom_id"]])
    with open(f"{F}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    refs = json.load(gzip.open(f"{F}/work/verifying_refs.json.gz", "rt"))
    for c in remove:
        refs.pop(c, None)
    refs.update(refs_new)
    with gzip.open(f"{F}/work/verifying_refs.json.gz", "wt") as f:
        json.dump(refs, f)
    with open(f"{F}/recovered.csv", "a", newline="") as f:
        if rec_lines:
            csv.DictWriter(f, list(rec_lines[0])).writerows(rec_lines)
    st = json.load(open(f"{F}/stats.json"))
    st["checker_merge"] = dict(log, final_total=len(rows), checker_rows=sum(1 for r in rows if r["metadata"].get("checker")),
                               removed=sorted(remove))
    json.dump(st, open(f"{F}/stats.json", "w"), indent=1)
    print("train.jsonl rows", len(rows))


if __name__ == "__main__":
    main()
