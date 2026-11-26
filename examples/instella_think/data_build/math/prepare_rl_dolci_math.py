"""Export the math (RLVR) subset of Dolci-Think-RL as a miles-rollout JSONL.

OLMo 3-style math RL: AI2's own RL prompt set, graded by the OLMo 3 MathVerifier.

AMD's prepare_rl_dolci.py splits Dolci into "if" vs "general", where general lumps
math_rlvr together with code_rlvr and general_mix -- there is no math-only path.
This filters to math_rlvr only, and deliberately REUSES their `rm_type_for` and
`to_record` (imported, not copied) so the record format cannot drift from theirs.

`metadata.rm_type` must come out as "math": reward_model.py::_score_one dispatches
on it and returns 0.0 with a warning for anything it does not recognise, so a
wrong tag here silently trains on all-zero reward.

Usage:
    python3 prepare_rl_dolci_math.py --output /path/math_prompts.jsonl
    python3 prepare_rl_dolci_math.py --output /path/smoke.jsonl --max-total 256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

# Reuse AMD's record builder rather than reimplementing it.
_MOPD = os.path.expanduser("~/Primus-Instella-Internal/rl/examples/mopd")
if _MOPD not in sys.path:
    sys.path.insert(0, _MOPD)
from prepare_rl_dolci import rm_type_for, to_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="allenai/Dolci-Think-RL-7B")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-total", type=int, default=0, help="0 = all math rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    print(f"loaded {args.dataset}:{args.split}  rows={len(ds)}", flush=True)

    rows = []
    sources = {}
    unwrapped = 0
    multi = 0
    for ex in ds:
        src = ex.get("dataset_source")
        if rm_type_for(src) != "math":
            continue
        sources[src] = sources.get(src, 0) + 1
        rec = to_record(
            ex.get("prompt"),
            ex.get("ground_truth"),
            ex.get("constraint"),
            src,
        )
        # Dolci stores ground_truth as a single-element LIST. reward_model.py does
        # math_reward(answer, str(label)), so a list arrives at the verifier as the
        # literal "['99']" and never matches -> every reward 0, GRPO sees no
        # advantage, and the run trains on nothing while looking healthy.
        # Unwrap to a scalar string here.
        label = rec["label"]
        if isinstance(label, (list, tuple)):
            if len(label) == 1:
                rec["label"] = str(label[0])
                unwrapped += 1
            else:
                # Keep multi-answer labels visible rather than silently picking one.
                multi += 1
                rec["label"] = str(label[0]) if label else ""
        rows.append(rec)
    if unwrapped:
        print(f"unwrapped {unwrapped} single-element list labels -> scalar strings")
    if multi:
        print(f"WARNING: {multi} labels had multiple answers; kept the first only")

    print(f"math rows: {len(rows)}")
    for src, n in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"   {n:7d}  {src}")

    if args.max_total and args.max_total < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_total]
        print(f"subsampled to {len(rows)} (seed={args.seed})")

    # Fail loudly rather than train on zero reward.
    bad = [r for r in rows if r["metadata"]["rm_type"] != "math"]
    if bad:
        sys.exit(f"ERROR: {len(bad)} records are not rm_type=math")
    unlabeled = sum(1 for r in rows if not r["label"])
    if unlabeled:
        print(f"WARNING: {unlabeled} records have an empty label and will score 0")
    nonstr = [r for r in rows if not isinstance(r["label"], str)]
    if nonstr:
        sys.exit(f"ERROR: {len(nonstr)} labels are not plain strings; they would "
                 f"reach the verifier via str() and never match")

    # Sanity-check the reward end to end on a few records, so a format change in
    # either the data or the verifier fails here and not 3 hours into training.
    try:
        sys.path.insert(0, os.path.expanduser(
            "~/Primus-Instella-Internal/rl/examples/if_rl"))
        import olmo3_math_verify as verifier

        checked = ok = 0
        for r in rows[:20]:
            lbl = r["label"]
            checked += 1
            if verifier.verify(r"\boxed{%s}" % lbl, lbl):
                ok += 1
        print(f"verifier self-check: {ok}/{checked} labels grade True when boxed")
        if checked and ok == 0:
            sys.exit("ERROR: verifier matched none of the labels; check label format")
    except ImportError:
        print("NOTE: olmo3_math_verify not importable here; skipped verifier self-check")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {args.output}")

    if rows:
        s = rows[0]
        print("\nsample record:")
        print(f"  prompt[0].content: {s['prompt'][0]['content'][:160]!r}")
        print(f"  label            : {s['label']!r}")
        print(f"  metadata         : {s['metadata']}")


if __name__ == "__main__":
    main()
