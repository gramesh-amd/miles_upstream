#!/usr/bin/env python3
"""train_pre_decontam.jsonl minus decontam_drop.txt -> train.jsonl, with schema checks and a stats file.

    python3 finalize.py --pre OUT/train_pre_decontam.jsonl --drop OUT/decontam_drop.txt --out OUT/train.jsonl

Checks every kept row: unique custom_id, a non-empty user prompt, and >= 3 tests for new rows; a failed
check exits non-zero and leaves --out untouched (rows go to <out>.tmp, renamed at the end). Writes
<out dir>/final_stats.json (rows per origin and source).
"""
import argparse
import collections
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--drop", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.drop) as f:
        drop = {ln.strip() for ln in f if ln.strip()}
    S = collections.Counter()
    seen = set()
    tmp = a.out + ".tmp"
    try:
        with open(tmp, "w") as fo, open(a.pre) as fi:
            for line in fi:
                r = json.loads(line)
                md = r["metadata"]
                cid = md["custom_id"]
                if cid in drop:
                    S["decontam_dropped"] += 1
                    continue
                if cid in seen:
                    raise SystemExit(f"duplicate custom_id {cid}")
                seen.add(cid)
                if not (r["prompt"] and r["prompt"][0]["role"] == "user" and r["prompt"][0]["content"].strip()):
                    raise SystemExit(f"{cid}: no user prompt")
                tests = json.loads(r["label"]) if isinstance(r["label"], str) else r["label"]
                if len(tests) < 3 and md.get("v8_origin") != "code1_carryover":
                    raise SystemExit(f"{cid}: {len(tests)} tests, a new row needs >= 3")
                S[f"rows|{md.get('v8_origin')}|{md.get('src')}"] += 1
                S["rows"] += 1
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, a.out)
    S["drop_list_size"] = len(drop)
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "final_stats.json"), "w") as f:
        json.dump(dict(S), f, indent=1)
    for k, v in sorted(S.items()):
        print(f"{k:55s} {v}")


if __name__ == "__main__":
    sys.exit(main())
