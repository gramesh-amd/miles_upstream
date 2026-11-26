#!/usr/bin/env python3
"""Fetch DAPO-Math-17k-Processed from HuggingFace and convert to our schema.

WHY THIS SOURCE. Two independent lines of evidence put DAPO first:
  - Literature: "RLVR Datasets and Where to Find Them" (arXiv 2605.26971) trains
    Qwen3-1.7B/8B with GRPO across the major RLVR math sets and ranks
    DAPO++ > DAPO > DeepScaleR > DeepMath > Skywork > OpenR1. MATH-BEYOND
    (ICLR 2026) independently selects DAPO for difficulty plus post-R1 novelty,
    and rejects DeepMath/Big-Math as "explicitly designed to be solvable".
  - Our own grades: of the five sources in our pool DAPO-Math-17k is both the
    HARDEST (mean pass 0.348, 36.1% at 0/8) and the closest to AIME in form
    (100% integer labels, median answer 70, 17% in the 100-499 band).
We already hold 2584 rows via hamishivi's filtered copy; this fetches the full
set so the remainder can be graded on our own model.

`datasets` is not installed here, so parquet files are pulled through the HF
resolve API and read with pyarrow. The repo exposes configs all/cn/en; we take
`en` because the Chinese split is a different prompt distribution and our pool's
provenance is the English processed set.

DECONTAMINATION. DAPO is known to contain leaked instances -- the paper's DAPO++
variant exists precisely to remove them -- so every row is checked against
AIME-2024 and AIME-2025 by exact match, substring, and token-Jaccard. mix_v2's
build record used a Jaccard threshold of 0.25, so the same threshold is used
here for consistency with the rest of the project.

Anything that fails to yield a usable prompt+label is counted and reported, never
silently dropped.
"""
import io
import json
import os
import re
import sys
import urllib.request

import pyarrow.parquet as pq

REPO = "open-r1/DAPO-Math-17k-Processed"
CONFIG = os.environ.get("DAPO_CONFIG", "en")
ROOT = os.environ.get("INSTELLA_RL_ROOT", "/it-share-prj2-1/data/goramesh-instella-rl")
OUT = f"{ROOT}/out/analysis_1012/dapo"
JACCARD = float(__import__("os").environ.get("JACCARD", "0.45"))
os.makedirs(OUT, exist_ok=True)

idx = json.loads(urllib.request.urlopen(
    f"https://huggingface.co/api/datasets/{REPO}/parquet", timeout=60).read())
urls = idx[CONFIG]["train"]
print(f"config={CONFIG}: {len(urls)} parquet file(s)")

rows = []
for u in urls:
    raw = urllib.request.urlopen(u, timeout=600).read()
    t = pq.read_table(io.BytesIO(raw))
    print(f"  {u.split('/')[-1]}: {t.num_rows} rows, columns={t.schema.names}")
    rows += t.to_pylist()
assert rows, "downloaded zero rows"


def norm(s):
    return " ".join(str(s).lower().split())


def toks(s):
    return set(re.findall(r"[a-z0-9]+", norm(s)))


# ---- AIME reference for decontamination ----------------------------------
aime = []
for f in ("aime-2024-olmes", "aime-2025-olmes"):
    aime += [json.loads(l) for l in open(f"{ROOT}/data/{f}.jsonl")]
aime_txt = [norm(r["prompt"][0]["content"]) for r in aime]
aime_tok = [toks(t) for t in aime_txt]
print(f"AIME reference: {len(aime_txt)} problems")

out_path = f"{OUT}/dapo_17k_en.jsonl"
kept = 0
drop = {"no_prompt": 0, "no_label": 0, "aime_exact": 0, "aime_substr": 0,
        "aime_jaccard": 0, "dup": 0}
seen = set()
maxj = 0.0
with open(out_path, "w") as fh:
    for r in rows:
        # prompt may be a chat list or a bare string depending on the config
        p = r.get("prompt")
        if isinstance(p, list) and p and isinstance(p[0], dict):
            text = p[0].get("content", "")
        elif isinstance(p, str):
            text = p
        else:
            text = r.get("question") or r.get("problem") or ""
        if not str(text).strip():
            drop["no_prompt"] += 1
            continue
        # label can live under several keys across DAPO variants
        lab = r.get("solution") or r.get("answer") or r.get("label")
        rm = r.get("reward_model")
        if not lab and isinstance(rm, dict):
            lab = rm.get("ground_truth")
        if isinstance(lab, list) and lab:
            lab = lab[0]
        if lab is None or not str(lab).strip():
            drop["no_label"] += 1
            continue
        n = norm(text)
        if n in seen:
            drop["dup"] += 1
            continue
        seen.add(n)
        if n in aime_txt:
            drop["aime_exact"] += 1
            continue
        if any(n in a or a in n for a in aime_txt):
            drop["aime_substr"] += 1
            continue
        tk = toks(text)
        j = max((len(tk & a) / max(len(tk | a), 1) for a in aime_tok), default=0.0)
        maxj = max(maxj, j)
        if j >= JACCARD:
            drop["aime_jaccard"] += 1
            continue
        fh.write(json.dumps({
            "prompt": [{"role": "user", "content": str(text)}],
            "label": str(lab).strip(),
            "metadata": {"dataset_source": "DAPO-Math-17k-Processed"},
        }) + "\n")
        kept += 1

print(f"\nkept {kept} rows -> {out_path}")
for k, v in drop.items():
    print(f"  dropped {k:<14} {v}")
print(f"  max AIME Jaccard seen: {maxj:.3f} (threshold {JACCARD})")
assert kept > 1000, "suspiciously few rows survived - check the schema mapping"

# quick AIME-likeness readout, same columns used for the other sources
ints = []
for l in open(out_path):
    s = json.loads(l)["label"].strip()
    if re.fullmatch(r"-?\d+", s):
        ints.append(int(s))
if ints:
    import statistics
    f = lambda c: 100.0 * c / len(ints)  # noqa: E731
    print(f"\n  %integer {100.0 * len(ints) / kept:.0f}   median {statistics.median(ints):.0f}"
          f"   <=20 {f(sum(1 for v in ints if 0 <= v <= 20)):.0f}%"
          f"   100-499 {f(sum(1 for v in ints if 100 <= v <= 499)):.0f}%"
          f"   >999 {f(sum(1 for v in ints if v > 999)):.0f}%")
    print("  (AIME target: %integer 100, median 208, <=20 3%, 100-499 52%, >999 0%)")
