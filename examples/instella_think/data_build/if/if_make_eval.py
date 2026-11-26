#!/usr/bin/env python3
"""Emit IFEval (541) and IFBench_test (300) in the eval format run_async.sh reads.

if_reward.py takes constraints from EITHER metadata.instruction_id_list or a
Dolci-style `label`; these sources already carry the flat list form, so write it
straight into metadata and leave `label` as the same spec for _constraints_of's
fallback path.

metadata.rm_type MUST be "ifbench": _score_one routes on it, and an absent
rm_type scores 0.0 for every sample, which reads as total model failure rather
than as a wiring bug.

Also stamp metadata.eval=1 -- _is_eval() keys strict (all-or-nothing) scoring off
it, so without it the eval numbers would be partial-credit and would not compare
against anything published.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/if"
SRC = [
    ("ifeval", ROOT / "data/if/eval/IFEval/ifeval_input_data.jsonl"),
    ("ifbench", ROOT / "repos/IFBench/data/IFBench_test.jsonl"),
]


def coerce(v):
    """kwargs/instruction_id_list arrive as python-literal strings in IFBench."""
    if isinstance(v, str):
        import ast

        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return v or []


for name, path in SRC:
    if not path.exists():
        print(f"MISSING {path}")
        continue
    rows = [json.loads(l) for l in path.open() if l.strip()]
    out = OUT / f"eval_{name}.jsonl"
    n = 0
    with out.open("w") as fh:
        for i, r in enumerate(rows):
            ids = coerce(r.get("instruction_id_list"))
            kws = coerce(r.get("kwargs"))
            if not ids:
                continue
            # Strip explicit nulls: the checkers take **kwargs and a None for an
            # unused slot raises inside build_description, which _check_one
            # swallows as False -- a silent wrong answer, not an error.
            kws = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kws]
            while len(kws) < len(ids):
                kws.append({})
            prompt = r.get("prompt", "")
            fh.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "label": [{"instruction_id": ids, "kwargs": kws}],
                        "metadata": {
                            "rm_type": "ifbench",
                            "eval": 1,
                            "instruction_id_list": ids,
                            "kwargs": kws,
                            "prompt_text": prompt,
                            "record_id": int(r.get("key", i) or i),
                        },
                    }
                )
                + "\n"
            )
            n += 1
    print(f"{name}: {n} -> {out}")
