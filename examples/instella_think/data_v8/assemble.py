#!/usr/bin/env python3
"""Assemble the mix: carry-over rows + new stdio rows + new function rows, deduplicated and shuffled.

    python3 assemble.py --build-dir WORK/build --carry carry.jsonl [--status status.parquet] --out-dir OUT

  carry-over  --carry, one row per problem (custom_ids with an "__os<k>" oversample suffix are dropped),
              stamped with code1_status / code1_last_outcome from --status (None without it)
  new stdio   BUILD/verified_*.jsonl (stdio_verify.py shards)
  new fn      BUILD/fn_rows.jsonl (fn_build.py)
A new row whose statement is contained in a carry-over statement is dropped: sampled word 8-grams
(hash & 3 == 0), grams in > 100 carry rows ignored, containment >= 0.5 over min(|grams|) >= 8 against
the 3 carry rows sharing the most grams. AceCoder carry rows are ordered problems still mixed in the
previous run first (status mixed_active, last outcome trained), then the rest, each shuffled, and
capped at --ace-cap (0 = all). Everything is then shuffled with --seed.

Writes OUT/train_pre_decontam.jsonl, OUT/new_vs_carry_dups.csv, OUT/assemble_stats.json. Rows are
indexed by byte offset and streamed, so memory stays small whatever the size of the labels.

Reproducing a recorded build: --hash legacy --order legacy --duplicates last --dedup-list <its
new_vs_carry_dups.csv>. The original build sampled grams with Python's salted hash(), so its dedup
set cannot be recomputed; the list pins it (the recomputed matches are only reported). legacy order
feeds the shuffle in shard-file order; sorted order (the default) sorts by custom_id first, so the
output depends only on the set of input rows.
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import sys

from common import h64, statement_of, tok

DEDUP_N, DEDUP_BIG, DEDUP_MIN, DEDUP_THR, DEDUP_TOP = 8, 100, 8, 0.5, 3


class Rows:
    """JSONL rows addressed by (file, byte offset)."""

    def __init__(self):
        self.files = []

    def scan(self, path):
        self.files.append(open(path, "rb"))
        k, off = len(self.files) - 1, 0
        for line in self.files[k]:
            yield (k, off), json.loads(line)
            off += len(line)

    def read(self, ref):
        f = self.files[ref[0]]
        f.seek(ref[1])
        return json.loads(f.readline())


def sgrams(text, hash_fn):
    ws = tok(text)
    g = {hash_fn(" ".join(ws[i:i + DEDUP_N])) for i in range(max(0, len(ws) - DEDUP_N + 1))}
    return {x for x in g if x & 3 == 0}


def find_dups(carry, new, hash_fn):
    """{new custom_id: (carry custom_id, containment)} for new statements contained in a carry-over one."""
    inv, cg = collections.defaultdict(list), []
    for i, c in enumerate(carry):
        g = sgrams(c["stmt"], hash_fn)
        cg.append(len(g))
        for x in g:
            inv[x].append(i)
    big = {x for x, v in inv.items() if len(v) > DEDUP_BIG}
    dups = {}
    for r in new:
        g = sgrams(r["stmt"], hash_fn) - big
        cnt = collections.Counter(j for x in g for j in inv.get(x, ()))
        for j, c in cnt.most_common(DEDUP_TOP):
            mn = min(len(g), cg[j])
            if mn >= DEDUP_MIN and c / mn >= DEDUP_THR:
                dups[r["cid"]] = (carry[j]["cid"], round(c / mn, 2))
                break
    return dups


def load_status(path):
    """{custom_id: (status, last_outcome)} from a parquet or CSV table keyed by custom_id (column or index)."""
    import pandas as pd

    st = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "custom_id" in st.columns:
        st = st.set_index("custom_id")
    st = st[["status", "last_outcome"]].astype(object)
    st = st.where(st.notna(), None)
    return dict(zip(st.index, zip(st.status, st.last_outcome, strict=True), strict=True))


def merge_new(scans, duplicates):
    """New rows by custom_id. A custom_id in several files is an error unless duplicates == "last": then the
    file that sorts last wins, at the position of the first occurrence."""
    new, seen = {}, collections.defaultdict(list)
    for path, it in scans:
        for ref, r in it:
            m = r["metadata"]
            cid = m["custom_id"]
            seen[cid].append(os.path.basename(path))
            new[cid] = {"cid": cid, "ref": ref, "src": m.get("src"), "stmt": statement_of(r)}
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        msg = "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(dup.items()))
        if duplicates == "error":
            raise SystemExit(f"custom_ids in several verified files (pass --duplicates last to keep the last): {msg}")
        print(f"duplicate custom_ids, kept the last file's row: {msg}")
    return list(new.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", required=True, help="verified_*.jsonl and fn_rows.jsonl")
    ap.add_argument("--carry", required=True, help="carry-over rows (JSONL, training schema)")
    ap.add_argument("--status", help="per-problem status of the carry-over rows in the previous run (optional)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ace-cap", type=int, default=0, help="keep at most this many AceCoder rows (0 = all)")
    ap.add_argument("--hash", choices=("blake2b", "legacy"), default="blake2b",
                    help="8-gram hash of the dedup; legacy = Python's salted hash(), needs --dedup-list")
    ap.add_argument("--order", choices=("sorted", "legacy"), default="sorted")
    ap.add_argument("--duplicates", choices=("error", "last"), default="error")
    ap.add_argument("--dedup-list", help="pin the dedup to a recorded new_vs_carry_dups.csv")
    a = ap.parse_args()
    if a.hash == "legacy" and not a.dedup_list:
        ap.error("--hash legacy needs --dedup-list (the salted hash cannot reproduce a recorded build)")
    hash_fn = hash if a.hash == "legacy" else h64
    S = collections.Counter()
    status = load_status(a.status) if a.status else {}
    rows = Rows()

    carry = []
    for ref, r in rows.scan(a.carry):
        m = r["metadata"]
        cid = m["custom_id"]
        if "__os" in cid:
            S["code1_oversample_copy_dropped"] += 1
            continue
        st, last = status.get(cid, (None, None))
        carry.append({"cid": cid, "ref": ref, "src": m.get("src"), "stmt": statement_of(r),
                      "status": st, "last": last, "origin": "code1_carryover"})
    S["code1_status_missing"] = sum(c["status"] is None for c in carry)
    if a.order == "sorted":
        carry.sort(key=lambda c: c["cid"])
    rng = random.Random(a.seed)
    ace = [c for c in carry if c["src"] == "acecoder"]
    other = [c for c in carry if c["src"] != "acecoder"]
    tier1 = [c for c in ace if c["status"] == "mixed_active" and c["last"] == "trained"]
    rest = [c for c in ace if not (c["status"] == "mixed_active" and c["last"] == "trained")]
    rng.shuffle(tier1)
    rng.shuffle(rest)
    ace_keep = (tier1 + rest)[:a.ace_cap] if a.ace_cap else tier1 + rest
    S["acecoder_available"], S["acecoder_tier1"], S["acecoder_kept"] = len(ace), len(tier1), len(ace_keep)
    carry = other + ace_keep

    paths = sorted(glob.glob(os.path.join(a.build_dir, "verified_*.jsonl")))
    new = merge_new([(p, rows.scan(p)) for p in paths], a.duplicates)
    S["new_stdio"] = len(new)
    fn = [{"cid": r["metadata"]["custom_id"], "ref": ref, "src": r["metadata"].get("src"), "stmt": statement_of(r)}
          for ref, r in rows.scan(os.path.join(a.build_dir, "fn_rows.jsonl"))]
    S["new_leetcode"] = len(fn)
    new += fn
    if a.order == "sorted":
        new.sort(key=lambda r: r["cid"])
    for r in new:
        r["origin"] = "v8_new"

    found = find_dups(carry, new, hash_fn)
    if a.dedup_list:
        with open(a.dedup_list) as f:
            pinned = {r["new_id"]: (r["carry_id"], r["containment"]) for r in csv.DictReader(f)}
        print(f"dedup pinned to {a.dedup_list}: {len(pinned)} ids, {len(set(pinned) - {r['cid'] for r in new})} "
              f"not among the new rows; recomputed with --hash {a.hash}: {len(found)} matches, "
              f"{len(set(found) ^ set(pinned))} disagree with the list")
        dups = pinned
    else:
        dups = found
    kept_new = [r for r in new if r["cid"] not in dups]
    dup_rows = [(r["cid"], *dups[r["cid"]]) for r in new if r["cid"] in dups]
    S["new_dup_of_carryover_dropped"] = len(dup_rows)

    out = carry + kept_new
    if a.order == "sorted":
        out.sort(key=lambda r: r["cid"])
    rng.shuffle(out)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "train_pre_decontam.jsonl"), "w") as f:
        for e in out:
            r = rows.read(e["ref"])
            if e["origin"] == "code1_carryover":
                r["metadata"]["code1_status"] = e["status"]
                r["metadata"]["code1_last_outcome"] = e["last"]
            r["metadata"]["v8_origin"] = e["origin"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out_dir, "new_vs_carry_dups.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["new_id", "carry_id", "containment"])
        w.writerows(dup_rows)
    by = collections.Counter((e["origin"], e["src"]) for e in out)
    S.update({f"final|{o}|{s}": c for (o, s), c in by.items()})
    S["final_rows"] = len(out)
    with open(os.path.join(a.out_dir, "assemble_stats.json"), "w") as f:
        json.dump(dict(S), f, indent=1)
    for k, v in sorted(S.items()):
        print(f"{k:50s} {v}")


if __name__ == "__main__":
    sys.exit(main())
