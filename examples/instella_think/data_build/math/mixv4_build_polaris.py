#!/usr/bin/env python3
"""Build the Polaris band 1-5 grading pool -- the prompts that close mix_v4's gap.

WHY BAND 1-5. Polaris ships `difficulty` as "k/8", a pass count on its own
reference model, and it is the first external difficulty label MEASURED to track
our pass count (Spearman +0.514 over 400 stratified prompts graded on this DPO
checkpoint). Our measured usable rate per bucket is an inverted U -- bucket 0 is
52% zero-of-eight, bucket 7 is 56% eight-of-eight, and both ends give an
identically zero GRPO advantage. Band 1-5 holds 27,483 prompts at a measured
58.1% usable.

WHY NOT THE CHEAPEST BAND. Band 3-4 scores higher (62.0% usable, 49 node-hours
vs 52) but needs 98.5% of its own pool to close the gap, and every bucket rate
here is measured at n=50, i.e. roughly +-7 percentage points. Band 1-5 pays three
extra node-hours for a 2.5x larger pool to absorb that error.

SIZING. We need 6,321 usable to reach 20,000 from the 13,679 in hand. At 58.1%
that is 10,880 prompts to grade; OVERSAMPLE (default 1.15x) covers the chance
that the real rate lands a few points low, and grading is resumable per prompt
so an unused tail costs nothing.

DEDUP IS AGAINST THE PROMPTS WE ALREADY HOLD, not just within Polaris. The last
build found 2,743 dolci<->DAPO duplicates; a prompt graded twice is a prompt paid
for twice and weighted twice in training.
"""
import io
import json
import os
import random
import re
import urllib.request

import pyarrow.parquet as pq

ROOT = os.environ.get("INSTELLA_RL_ROOT", "/it-share-prj2-1/data/goramesh-instella-rl")
OUT = f"{ROOT}/out/analysis_1012/polaris"
REPO = "POLARIS-Project/Polaris-Dataset-53K"
BAND = {int(x) for x in os.environ.get("BAND", "1,2,3,4,5").split(",")}
NEED = int(os.environ.get("NEED", "10880"))
OVER = float(os.environ.get("OVER", "1.15"))
JACCARD = 0.45
SEED = 20261016
os.makedirs(OUT, exist_ok=True)


def norm(s):
    return " ".join(str(s).lower().split())


def toks(s):
    return set(re.findall(r"[a-z0-9]+", norm(s)))


# ---- AIME, for decontamination -------------------------------------------
aime = []
for f in ("aime-2024-olmes", "aime-2025-olmes"):
    aime += [json.loads(l) for l in open(f"{ROOT}/data/{f}.jsonl")]
aime_txt = [norm(r["prompt"][0]["content"]) for r in aime]
aime_tok = [toks(t) for t in aime_txt]
print(f"AIME reference prompts: {len(aime_txt)}")

# ---- everything already in the mix, for dedup -----------------------------
HELD = [
    f"{ROOT}/data/mix_v2_math.jsonl",
    f"{ROOT}/data/mix_v3_aime.jsonl",
    f"{ROOT}/out/analysis_1012/dolci_filter/pool_30k.jsonl",
    f"{ROOT}/out/analysis_1012/dapo/dapo_17k_en.jsonl",
]
held = set()
for p in HELD:
    if not os.path.exists(p):
        print(f"  dedup source MISSING (skipped): {p}")
        continue
    n0 = len(held)
    with open(p) as fh:
        for l in fh:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            pr = r.get("prompt")
            if isinstance(pr, list) and pr and isinstance(pr[0], dict):
                pr = pr[0].get("content", "")
            if isinstance(pr, str) and pr.strip():
                held.add(norm(pr))
    print(f"  dedup source {os.path.basename(p):<46} +{len(held)-n0:>6} (total {len(held)})")

# ---- fetch Polaris --------------------------------------------------------
idx = json.loads(urllib.request.urlopen(
    f"https://huggingface.co/api/datasets/{REPO}/parquet", timeout=60).read())
cfg = "default" if "default" in idx else sorted(idx)[0]
split = "train" if "train" in idx[cfg] else sorted(idx[cfg])[0]
recs = []
for u in idx[cfg][split]:
    recs += pq.read_table(io.BytesIO(
        urllib.request.urlopen(u, timeout=1800).read())).to_pylist()
print(f"\nPolaris rows fetched: {len(recs)}")

PROMPT_KEYS = ("problem", "question", "prompt", "query", "messages")
ANS_KEYS = ("answer", "solution", "ground_truth", "label", "final_answer",
            "expected_answer", "gt")


def get_prompt(rec):
    for k in PROMPT_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v[0].get("content", "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def get_answer(rec):
    for k in ANS_KEYS:
        if k in rec and rec[k] not in (None, ""):
            v = rec[k]
            if isinstance(v, list) and v:
                v = v[0]
            return str(v)
    rm = rec.get("reward_model")
    if isinstance(rm, dict):
        v = rm.get("ground_truth")
        if isinstance(v, list) and v:
            v = v[0]
        if v:
            return str(v)
    return None


def bucket(rec):
    d = rec.get("difficulty")
    if d is None:
        return None
    m = re.match(r"^\s*(\d+)\s*/\s*8\s*$", str(d))
    if m:
        return int(m.group(1))
    try:
        return int(round(float(d)))
    except (TypeError, ValueError):
        return None


random.seed(SEED)
random.shuffle(recs)

want = int(NEED * OVER)
kept, seen = [], set()
drop = {"off_band": 0, "no_label": 0, "no_prompt": 0, "no_answer": 0,
        "aime": 0, "dup_internal": 0, "dup_held": 0}
for r in recs:
    if len(kept) >= want:
        break
    b = bucket(r)
    if b is None:
        drop["no_label"] += 1
        continue
    if b not in BAND:
        drop["off_band"] += 1
        continue
    p = get_prompt(r)
    if not str(p).strip():
        drop["no_prompt"] += 1
        continue
    a = get_answer(r)
    if a is None or not a.strip():
        drop["no_answer"] += 1
        continue
    n = norm(p)
    if n in seen:
        drop["dup_internal"] += 1
        continue
    if n in held:
        drop["dup_held"] += 1
        continue
    if n in aime_txt or any(n in x or x in n for x in aime_txt):
        drop["aime"] += 1
        continue
    tk = toks(p)
    if max((len(tk & x) / max(len(tk | x), 1) for x in aime_tok), default=0.0) >= JACCARD:
        drop["aime"] += 1
        continue
    seen.add(n)
    kept.append({"prompt": [{"role": "user", "content": str(p)}],
                 "label": a.strip(),
                 "metadata": {"dataset_source": "polaris",
                              "published_difficulty": b}})

path = f"{OUT}/pool_band1_5.jsonl"
with open(path, "w") as fh:
    for r in kept:
        fh.write(json.dumps(r) + "\n")

from collections import Counter  # noqa: E402
print(f"\nkept {len(kept)} (target {NEED} x {OVER} = {want})   dropped={drop}")
print("  per bucket:", dict(sorted(Counter(r["metadata"]["published_difficulty"]
                                           for r in kept).items())))
nums = sorted(int(x) for x in (str(r["label"]).strip() for r in kept)
              if re.fullmatch(r"\d{1,6}", x))
if nums:
    import statistics
    print(f"  answers: integer {100*len(nums)/len(kept):.1f}%  median {statistics.median(nums):.0f}"
          f"  in 0-999 {100*sum(1 for x in nums if x<=999)/len(nums):.1f}%")
print(f"\n-> {path}")
print(f"   expected usable at the measured 58.1%: ~{int(0.581*len(kept)):,}")
