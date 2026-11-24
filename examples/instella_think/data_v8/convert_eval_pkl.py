#!/usr/bin/env python3
"""Convert a pickled eval corpus (a list of per-problem dicts) into decontam.py's --eval-hashes JSONL.

    python3 convert_eval_pkl.py eval.pkl eval_hashes.jsonl

The pickle is read with a data-only unpickler, so nothing in it can be instantiated or called. Only the
fields decontam.py reads are kept: eval_set, eval_id, platform, title, io [[h, n]], calls [[fname, h, n]].
"""
import argparse
import json
import sys

from common import DataOnlyUnpickler

FIELDS = ("eval_set", "eval_id", "platform", "title", "io", "calls")


def convert(src: str, dst: str) -> int:
    with open(src, "rb") as f:
        ev = DataOnlyUnpickler(f).load()
    with open(dst, "w") as f:
        for e in ev:
            f.write(json.dumps({k: e.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")
    return len(ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    print(f"{convert(a.src, a.dst)} eval problems -> {a.dst}")


if __name__ == "__main__":
    sys.exit(main())
