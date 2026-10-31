#!/usr/bin/env python3
"""Build the instruction-following train/eval JSONLs for run_if.sh.

    # look before you leap -- prints the real schema and a few raw rows, writes nothing
    python3 examples/instella_think/prepare_if_data.py inspect --source dolci

    # write the files run_if.sh expects
    python3 examples/instella_think/prepare_if_data.py build --out-dir ~/data

Output schema, matching what if_reward.py reads and what miles' --input-key/--label-key
expect (miles/utils/data.py:203-249):

    {"prompt":   [{"role": "user", "content": "<the instruction>"}],
     "label":    "<json of {instruction_id, kwargs}>",      # human-readable, not parsed
     "metadata": {"instruction_id_list": [...], "kwargs": [{...}],
                  "prompt_text": "<the instruction>", "rm_type": "ifbench",
                  "dataset_source": ..., "passrate": ...}}

`metadata` is the load-bearing field: miles passes it through untouched, and if_reward.py
reads instruction_id_list/kwargs/prompt_text from it. `label` is written for debuggability
only -- if_reward.py parses it solely as a fallback and counts when it has to.

WHY THIS SCRIPT IS PARANOID ABOUT THE SOURCE SCHEMA
The upstream Instella IF example assumed `label` literal-evals to
[{"instruction_id": [...], "kwargs": [...]}], while its own data generator
(Instella-MoE/rl/examples/mopd/prepare_rl_dolci.py:79-93) wrote `label = ground_truth`
and put the constraint under `metadata.constraint` instead. Those two only agree if
Dolci's `ground_truth` happens to BE the constraint spec. Rather than inherit that
assumption, this script auto-detects the encoding, and refuses to write anything if it
cannot map a row to (instruction_id_list, kwargs) -- a mismapped constraint is not a
crash, it is a silent zero reward on every sample, which is much more expensive to find.

Every emitted constraint id is also checked against the vendored IFEvalG registry, since
an id the checker does not know scores 0 forever.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))

# An IFEval instruction id, e.g. "detectable_format:json_format". Used to tell a real id
# from a serialized blob that merely happens to contain a colon.
_ID_RE = re.compile(r"^[a-z_]+:[a-zA-Z_0-9]+$")

SOURCES = {
    # Instella's own IF RL mixture. IF rows are tagged in `dataset_source`.
    "dolci": {
        "repo": "allenai/Dolci-Think-RL-7B",
        "kind": "dataset",
        "filter_col": "dataset_source",
        "filter_substr": "IF_multi_constraints",
    },
    # Eval: the original Google IFEval prompts (541), already in the right shape.
    "ifeval": {"repo": "google/IFEval", "kind": "file", "file": "ifeval_input_data.jsonl"},
    # Eval: IFBench's held-out set, harder/newer constraints.
    "ifbench": {"repo": "allenai/IFBench_test", "kind": "dataset"},
}


# --- registry -------------------------------------------------------------------
def load_registry_ids() -> set[str]:
    """Constraint ids the reward can actually check: IFEvalG plus IFBench's extension.

    Both must be counted or validation is misleading. IFEvalG alone flags every IFBench
    prompt as unknown (IFBench uses a newer constraint family), which would push anyone
    towards --drop-unknown and silently discard the entire IFBench eval set.
    """
    ids: set[str] = set()
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        from IFEvalG import instructions_registry

        ids |= set(instructions_registry.INSTRUCTION_DICT.keys())
    except ImportError as exc:
        print(
            f"WARNING: IFEvalG not importable ({exc}); skipping id validation.\n"
            "         Install: pip install langdetect absl-py immutabledict nltk",
            file=sys.stderr,
        )

    ifbench_dir = os.environ.get("IF_IFBENCH_DIR", "")
    if os.path.isdir(ifbench_dir):
        try:
            if ifbench_dir not in sys.path:
                sys.path.insert(0, ifbench_dir)
            from ifbench import instructions_registry as ifb  # type: ignore

            ids |= set(ifb.INSTRUCTION_DICT.keys())
        except Exception as exc:
            print(f"WARNING: IFBench registry not importable ({exc}).", file=sys.stderr)
    return ids


# --- loading --------------------------------------------------------------------
# Dolci carries pre-tokenized columns that are megabytes per row and useless here.
# Materializing 102k rows with these as python dicts costs tens of GB and minutes; dropping
# them at the Arrow level first makes a full scan cheap.
_DROP_COLS = ("input_ids", "input_ids_prompt", "attention_mask", "labels", "outputs")


def _rows_from_hf(spec: dict, limit: int | None = None) -> list[dict]:
    """Load rows as plain dicts, already filtered to this source's subset.

    `limit` caps the number of MATCHING rows returned, not how far we scan. That matters
    because Dolci is grouped by `dataset_source`: the first 4000 rows are all
    math_rlvr_mixture_dpo, so a naive head-of-dataset scan finds zero IF rows and looks
    like the filter is broken.
    """
    repo = spec["repo"]
    if spec["kind"] == "file":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo, filename=spec["file"], repo_type="dataset")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        return rows[:limit] if limit else rows

    try:
        from datasets import load_dataset

        ds = load_dataset(repo, split="train")
        drop = [c for c in _DROP_COLS if c in ds.column_names]
        if drop:
            ds = ds.remove_columns(drop)
        col, sub = spec.get("filter_col"), spec.get("filter_substr")
        if col and col in ds.column_names:
            n_before = len(ds)
            ds = ds.filter(lambda b: [sub in (v or "") for v in b[col]], batched=True)
            print(f"  filter {col}~{sub}: {n_before} -> {len(ds)} rows")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [dict(r) for r in ds]
    except ImportError:
        pass

    # No `datasets`: read the parquet shards directly.
    from huggingface_hub import HfApi, hf_hub_download

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            f"Need `datasets` or `pyarrow` to read {repo} ({exc}).\n"
            "Run this inside the training container, or: pip install pyarrow"
        ) from exc

    files = sorted(
        s.rfilename for s in HfApi().dataset_info(repo).siblings if s.rfilename.endswith(".parquet")
    )
    col, sub = spec.get("filter_col"), spec.get("filter_substr")
    rows: list[dict] = []
    for fn in files:
        path = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset")
        table = pq.read_table(path)
        keep = [c for c in table.column_names if c not in _DROP_COLS]
        shard = table.select(keep).to_pylist()
        if col:
            shard = [r for r in shard if sub in str(r.get(col) or "")]
        rows.extend(shard)
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


def _select(rows: list[dict], spec: dict) -> list[dict]:
    col, sub = spec.get("filter_col"), spec.get("filter_substr")
    if not col:
        return rows
    return [r for r in rows if sub in str(r.get(col) or "")]


# --- constraint extraction ------------------------------------------------------
def _parse_maybe(v):
    """Parse a JSON or python-literal string; pass anything else through."""
    if not isinstance(v, str):
        return v
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(v)
        except Exception:
            continue
    return v


def _spec_to_pair(obj) -> tuple[list, list] | None:
    """Map one constraint object to (instruction_id_list, kwargs_list).

    Dolci nests this two deep: `ground_truth` is a LIST holding a single STRING, and that
    string is a python-literal list of dicts, e.g.
        ["[{'instruction_id': ['detectable_format:json_format'], 'kwargs': [None]}]"]
    So a string element must be parsed before it can be judged, and only a string that
    fails to parse AND looks like "family:name" is an actual instruction id. Treating any
    colon-bearing string as an id (the obvious shortcut) swallows that whole repr as one
    bogus id, which then fails every registry lookup and rewards 0 forever.
    """
    if isinstance(obj, str):
        parsed = _parse_maybe(obj)
        if not isinstance(parsed, str):
            return _spec_to_pair(parsed)
        return ([obj.strip()], [{}]) if _ID_RE.match(obj.strip()) else None
    if isinstance(obj, dict):
        ids = (
            obj.get("instruction_id_list")
            or obj.get("instruction_id")
            or obj.get("instruction_ids")
        )
        if isinstance(ids, str):
            ids = [ids]
        if not ids:
            return None
        kw = obj.get("kwargs") or obj.get("args") or []
        if isinstance(kw, dict):
            kw = [kw]
        kw = [_parse_maybe(k) or {} for k in kw]
        return list(ids), list(kw)
    if isinstance(obj, list) and obj:
        ids: list = []
        kws: list = []
        for o in obj:
            pair = _spec_to_pair(o)
            if pair:
                ids += pair[0]
                kws += pair[1]
        return (ids, kws) if ids else None
    return None


def extract_constraints(row: dict) -> tuple[list, list, str]:
    """Return (ids, kwargs, which_field_worked). Empty ids means unmappable."""
    # Ordered by how explicit the field is, so a real instruction_id_list always wins.
    for field in ("metadata", "constraint", "ground_truth", "label", "constraints"):
        if field not in row or row[field] in (None, "", [], {}):
            continue
        val = row[field]
        if field == "metadata":
            val = _parse_maybe(val)
            if isinstance(val, dict) and val.get("instruction_id_list"):
                pair = _spec_to_pair(val)
                if pair:
                    return pair[0], pair[1], field
            continue
        pair = _spec_to_pair(val)
        if pair:
            ids, kws = pair
            if len(kws) != len(ids):
                kws = (kws + [{}] * len(ids))[: len(ids)]
            return ids, kws, field
    # Already-flat IFEval/IFBench shape.
    if row.get("instruction_id_list"):
        ids = list(row["instruction_id_list"])
        kws = [_parse_maybe(k) or {} for k in (row.get("kwargs") or [])]
        kws = (kws + [{}] * len(ids))[: len(ids)]
        return ids, kws, "instruction_id_list"
    return [], [], ""


def prompt_of(row: dict) -> str:
    for key in ("prompt", "instruction", "question", "text"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            # Dolci prompts carry a literal "user: " prefix on some rows.
            return v[len("user: ") :] if v.startswith("user: ") else v
        if isinstance(v, list) and v:  # chat-formatted
            for turn in reversed(v):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    return str(turn.get("content") or "")
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for turn in reversed(msgs):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content") or "")
    return ""


# --- commands -------------------------------------------------------------------
def cmd_inspect(a) -> int:
    spec = SOURCES[a.source]
    rows = _rows_from_hf(spec, limit=a.scan)
    print(f"=== {spec['repo']} : loaded {len(rows)} rows (limit {a.scan}) ===")
    if not rows:
        return 1
    print("\ncolumns:")
    for k, v in rows[0].items():
        s = json.dumps(v, default=str)
        print(f"  {k:22} {type(v).__name__:10} {s[:110]}")

    sel = _select(rows, spec)
    print(f"\nafter filter {spec.get('filter_col')}~{spec.get('filter_substr')}: {len(sel)} rows")
    if spec.get("filter_col"):
        print("dataset_source values seen (top 10):")
        for name, n in Counter(str(r.get(spec["filter_col"])) for r in rows).most_common(10):
            print(f"  {n:7d}  {name}")

    print("\n=== constraint extraction on the first rows ===")
    reg = load_registry_ids()
    for row in sel[: a.show]:
        ids, kws, field = extract_constraints(row)
        print(f"\n  via field: {field or '<NONE - UNMAPPABLE>'}")
        print(f"  prompt : {prompt_of(row)[:160]!r}")
        for raw in ("constraint", "constraint_type", "ground_truth"):
            if raw in row:
                print(f"  {raw:15}: {json.dumps(row[raw], default=str)[:220]}")
        print(f"  -> ids : {ids}")
        print(f"  -> kw  : {json.dumps(kws, default=str)[:220]}")
        if reg:
            bad = [i for i in ids if i not in reg]
            print(f"  -> unknown to registry: {bad if bad else 'none'}")
    return 0


def _convert(rows: list[dict], reg: set[str], drop_unknown: bool, tag: str, split: str):
    out, stats = [], Counter()
    unknown: Counter = Counter()
    for row in rows:
        stats["seen"] += 1
        ids, kws, field = extract_constraints(row)
        prompt = prompt_of(row)
        if not ids:
            stats["no_constraints"] += 1
            continue
        if not prompt:
            stats["no_prompt"] += 1
            continue
        stats[f"via_{field}"] += 1
        bad = [i for i in ids if reg and i not in reg]
        if bad:
            unknown.update(bad)
            stats["has_unknown_id"] += 1
            if drop_unknown:
                continue
        out.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "label": json.dumps({"instruction_id": ids, "kwargs": kws}),
                "metadata": {
                    "rm_type": "ifbench",
                    "domain": "if",
                    # if_reward.py scores strictly when this says "eval". It cannot use
                    # miles' `evaluation` kwarg: miles never forwards it to the reward
                    # (sglang_rollout.py:354,363,396), so it is always False.
                    "split": split,
                    "instruction_id_list": ids,
                    "kwargs": kws,
                    "prompt_text": prompt,
                    "dataset_source": str(row.get("dataset_source") or tag),
                    "passrate": row.get("passrate"),
                    "n_constraints": len(ids),
                },
            }
        )
        stats["written"] += 1
    return out, stats, unknown


def cmd_build(a) -> int:
    reg = load_registry_ids()
    if reg:
        print(f"registry: {len(reg)} constraint types implemented by ./IFEvalG")
    os.makedirs(a.out_dir, exist_ok=True)
    rc = 0

    targets = [
        ("dolci", os.path.join(a.out_dir, "if-train", "dolci-if-train.jsonl"), a.max_train, "train"),
        ("ifeval", os.path.join(a.out_dir, "ifeval", "ifeval-eval.jsonl"), None, "eval"),
        ("ifbench", os.path.join(a.out_dir, "ifbench", "ifbench-eval.jsonl"), None, "eval"),
    ]
    if a.only:
        targets = [t for t in targets if t[0] in a.only]

    for name, out_path, limit, split in targets:
        spec = SOURCES[name]
        print(f"\n=== {name}: {spec['repo']} ===")
        try:
            rows = _select(_rows_from_hf(spec), spec)
        except Exception as exc:
            print(f"  FAILED to load: {type(exc).__name__}: {exc}")
            rc = 1
            continue
        if limit:
            rows = rows[:limit]
        recs, stats, unknown = _convert(rows, reg, a.drop_unknown, name, split)
        print(f"  {dict(stats)}")
        if unknown:
            print(f"  unknown constraint ids: {dict(unknown.most_common(10))}")

        # A source that maps almost nothing means the schema moved; do not write a file
        # that would train on 12 prompts or reward nothing.
        if stats["seen"] and stats["written"] / stats["seen"] < a.min_yield:
            print(
                f"  REFUSING to write {out_path}: yield "
                f"{stats['written']}/{stats['seen']} < --min-yield {a.min_yield}.\n"
                f"  Run `inspect --source {name}` and fix extract_constraints()."
            )
            rc = 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} -> {out_path}")

        nconstraints = Counter(len(r["metadata"]["instruction_id_list"]) for r in recs)
        print(f"  constraints/prompt: {dict(sorted(nconstraints.items()))}")
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="print source schema + extraction dry-run")
    i.add_argument("--source", choices=sorted(SOURCES), default="dolci")
    i.add_argument("--scan", type=int, default=2000, help="rows to load")
    i.add_argument("--show", type=int, default=3, help="rows to dump")
    i.set_defaults(fn=cmd_inspect)

    b = sub.add_parser("build", help="write train + eval JSONLs")
    b.add_argument("--out-dir", default=os.path.expanduser("~/data"))
    b.add_argument("--max-train", type=int, default=None)
    b.add_argument("--only", nargs="+", choices=sorted(SOURCES))
    b.add_argument("--drop-unknown", action="store_true", help="skip rows with unsupported ids")
    b.add_argument("--min-yield", type=float, default=0.5, help="refuse to write below this map rate")
    b.set_defaults(fn=cmd_build)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
