#!/usr/bin/env python3
"""Candidate set for the second IF prefilter. Runs IN-CONTAINER.

Differences from if_build_mix.py:
  * only prompts whose constraints are mutually satisfiable: no pair forbidden by
    IFEvalG's INSTRUCTION_CONFLICTS (45.6% of the first pool violated it) and no
    constraint type that never passed in the first prefilter;
  * drawn from the full 95k pool, including prompts the first prefilter saw --
    that pass sampled without the chat template at T=1.0, so its numbers are not
    reused;
  * every row carries `rendered`, the exact text training feeds SGLang
    (single user turn, tokenizer chat template, add_generation_prompt=True), so
    the prefilter measures the input distribution RL will see.
"""
import argparse, ast, collections, json, random, re, sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/instella_think")
sys.path.insert(0, "/workspace/scripts")
from IFEvalG import instructions_registry as reg  # noqa: E402

SRC = "/workspace/data/if/IF_multi_constraints_upto5/data/train-00000-of-00001.parquet"
TOK = "/workspace/models/Instella-MoE-16B-A3B-DPO"
DOMAIN_TARGET = {"chat": 0.20, "flan": 0.18, "math": 0.14, "code": 0.14,
                 "other": 0.14, "safety": 0.10, "multiling": 0.10}
NC_TARGET = {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.20}
PATTERNS = [
    ("math", ["numinamath", "open_math", "gsm8k", "math_", "personas_math"]),
    ("code", ["codealpaca", "heval", "evol_code"]),
    ("safety", ["wildjailbreak", "wildguard"]),
    ("multiling", ["aya"]),
    ("chat", ["wildchat", "oasst", "no_robots"]),
    ("flan", ["flan_v2"]),
]
DEAD = {"new:copy_span_idx", "count:counting_composition", "keywords:keyword_specific_position"}


def domain(src):
    s = src.lower()
    for name, pats in PATTERNS:
        if any(p in s for p in pats):
            return name
    return "other"


def satisfiable(ids):
    if DEAD & set(ids):
        return False
    conf = reg.INSTRUCTION_CONFLICTS
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if b in conf.get(a, ()) or a in conf.get(b, ()):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="/workspace/data/if/if_candidates2.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(SRC)
    df["gt"] = df["ground_truth"].map(lambda s: ast.literal_eval(s)[0])
    df["ids"] = df["gt"].map(lambda e: list(e["instruction_id"]))
    df = df[df["ids"].map(satisfiable)].copy()
    df["nc"] = df["ids"].map(len)
    df["prompt"] = df["messages"].map(lambda m: m[0]["content"])
    df["src"] = df["key"].map(
        lambda k: re.sub(r"[_-]?\d+$", "",
                         (str(k)[len("ai2-adapt-dev/"):] if str(k).startswith("ai2-adapt-dev/")
                          else str(k))).lower())
    df["dom"] = df["src"].map(domain)
    print(f"satisfiable pool: {len(df):,}")

    freq = collections.Counter(i for ids in df["ids"] for i in ids)
    df["rarity"] = df["ids"].map(lambda ids: min(freq[i] for i in ids))
    picked, shortfall = [], 0
    for dom, ds in DOMAIN_TARGET.items():
        for nc, ns in NC_TARGET.items():
            want = round(a.n * ds * ns)
            cell = df[(df["dom"] == dom) & (df["nc"] == nc)]
            take = cell if len(cell) <= want else cell.nsmallest(want, "rarity")
            shortfall += max(0, want - len(cell))
            picked.append(take)
    out = pd.concat(picked).drop_duplicates(subset="key")
    if shortfall > 0:
        rest = df[~df["key"].isin(set(out["key"])) & (df["nc"] >= 3)]
        out = pd.concat([out, rest.nsmallest(min(shortfall, len(rest)), "rarity")])
        print(f"redistributed {shortfall:,} to deeper cells")
    out = out.sample(frac=1.0, random_state=a.seed)

    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    with open(a.out, "w") as fh:
        for _, r in out.iterrows():
            rendered = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                               tokenize=False, add_generation_prompt=True)
            fh.write(json.dumps({
                "prompt": r["prompt"], "label": r["ground_truth"], "rendered": rendered,
                "metadata": {"rm_type": "ifbench", "key": r["key"], "domain": r["dom"],
                             "n_constraints": int(r["nc"])},
            }) + "\n")
    n = len(out)
    print(f"wrote {n:,} -> {a.out}")
    print("domain mix:", {k: f"{v/n:.1%}" for k, v in sorted(collections.Counter(out['dom']).items())})
    print("constraint mix:", {k: f"{v/n:.1%}" for k, v in sorted(collections.Counter(out['nc']).items())})
    print("rendered example tail:", repr(rendered[-120:]))


if __name__ == "__main__":
    main()
