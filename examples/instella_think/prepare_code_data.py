#!/usr/bin/env python3
"""Build code RL data from Dolci-Think-RL's code_rlvr subset.

    python3 examples/instella_think/prepare_code_data.py inspect        # schema + stats, writes nothing
    python3 examples/instella_think/prepare_code_data.py build          # train JSONL

Output rows match the IF data, with metadata.rm_type = "code":
    {"prompt": [{"role": "user", "content": ...}],
     "label": "<JSON list of {input, output}>",
     "metadata": {"rm_type": "code", "split": "train"|"eval", "n_tests": N, "dataset_source": ...}}

Dolci nests `ground_truth` as a list holding one string: JSON stdin/stdout cases, JSON
assert strings, or (about 3k rows) base64(zlib(pickle)) of the stdin/stdout cases.
`label` here is always plain JSON, capped to --max-tests cases.

Evals are the OLMo 3 code benchmarks (prepare_code_benchmarks.py). --n-eval N still carves
a seeded N-row hold-out out of train into code-eval/dolci-code-heldout.jsonl, stamped
split=eval so code_reward.py scores it all-or-nothing.

Run where the HF cache is shared (HF_HOME on NFS) and memory is not capped, i.e.
inside the training container rather than on the login node. --out-dir defaults to
$DATA_ROOT, which paths.sh sets; pass it explicitly when running without paths.sh.
"""

import argparse
import base64
import collections
import io
import json
import os
import pickle
import random
import zlib

REPO = "allenai/Dolci-Think-RL-7B"
SOURCE_SUBSTR = "code_rlvr"
USER_PREFIX = "user: "


def load_rows():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.endswith(".parquet"))
    rows = []
    for fn in files:
        path = hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset")
        table = pq.read_table(path, columns=["prompt", "ground_truth", "dataset_source"])
        rows.extend(r for r in table.to_pylist() if SOURCE_SUBSTR in (r["dataset_source"] or ""))
    return rows


class _DataOnlyUnpickler(pickle.Unpickler):
    """Plain containers and scalars only: refusing every global means no class or
    function in the payload can be instantiated or called."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _decode(s: str):
    """JSON, or (for ~3k rows) base64(zlib(pickle)) of the same list of cases."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    if s.startswith("eJ"):
        raw = zlib.decompress(base64.b64decode(s))
        v = _DataOnlyUnpickler(io.BytesIO(raw)).load()
        return json.loads(v) if isinstance(v, str) else v
    raise ValueError("not JSON or compressed pickle")


def parse_tests(gt):
    v = gt[0] if isinstance(gt, list) and len(gt) == 1 and isinstance(gt[0], str) else gt
    try:
        if isinstance(v, str):
            v = _decode(v)
        # One more level: a list holding a single JSON document of cases.
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
            try:
                inner = _decode(v[0])
                if isinstance(inner, (list, dict)):
                    v = inner
            except (ValueError, pickle.UnpicklingError, zlib.error):
                pass
    except (ValueError, pickle.UnpicklingError, zlib.error):
        return None, "unparseable"
    if isinstance(v, list) and v and all(isinstance(t, dict) and "input" in t and "output" in t for t in v):
        return [{"input": str(t["input"]), "output": str(t["output"])} for t in v], "stdio"
    if isinstance(v, list) and v and all(isinstance(t, str) for t in v):
        return v, "assert"
    if isinstance(v, dict) and v.get("test"):
        return v, "function"
    return None, "unknown"


def cap_tests(tests, max_tests: int, max_bytes: int):
    """Keep the first max_tests cases within a byte budget, always at least one."""
    if not isinstance(tests, list):
        return tests
    kept, size = [], 0
    for t in tests[:max_tests]:
        n = len(json.dumps(t, ensure_ascii=False))
        if kept and size + n > max_bytes:
            break
        kept.append(t)
        size += n
    return kept


def to_record(row, tests, split):
    content = row["prompt"]
    if isinstance(content, str) and content.startswith(USER_PREFIX):
        content = content[len(USER_PREFIX):]
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": json.dumps(tests, ensure_ascii=False),
        "metadata": {
            "rm_type": "code",
            "split": split,
            "n_tests": len(tests) if isinstance(tests, list) else 1,
            "dataset_source": row["dataset_source"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "build"])
    ap.add_argument("--out-dir", default=os.environ.get("DATA_ROOT"), required="DATA_ROOT" not in os.environ)
    ap.add_argument("--n-eval", type=int, default=0, help="rows held out of train as an eval set")
    ap.add_argument("--min-tests", type=int, default=1)
    ap.add_argument("--max-tests", type=int, default=32,
                    help="cases stored per row; match code_reward.py's CODE_MAX_TESTS")
    ap.add_argument("--max-label-bytes", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows()
    kinds = collections.Counter()
    n_tests = []
    parsed = []
    for r in rows:
        tests, kind = parse_tests(r["ground_truth"])
        kinds[kind] += 1
        if tests is not None:
            n = len(tests) if isinstance(tests, list) else 1
            n_tests.append(n)
            if n >= args.min_tests:
                parsed.append((r, cap_tests(tests, args.max_tests, args.max_label_bytes)))

    n_tests.sort()
    q = lambda p: n_tests[min(len(n_tests) - 1, int(p * len(n_tests)))] if n_tests else 0
    print(f"{REPO} rows matching '{SOURCE_SUBSTR}': {len(rows)}")
    print(f"  test formats: {dict(kinds)}")
    print(f"  tests per row: min={q(0)} p25={q(.25)} median={q(.5)} p90={q(.9)} max={q(1.0)}")
    tails = collections.Counter(str(r['prompt']).rstrip()[-160:].split("\n")[-1][:80] for r, _ in parsed[:2000])
    print("  most common last prompt lines (first 2000 rows):")
    for line, c in tails.most_common(5):
        print(f"    {c:5d}  {line!r}")

    if args.cmd == "inspect":
        return

    rng = random.Random(args.seed)
    rng.shuffle(parsed)
    eval_rows, train_rows = parsed[: args.n_eval], parsed[args.n_eval:]
    train_path = os.path.join(args.out_dir, "code-train", "dolci-code-train.jsonl")
    eval_path = os.path.join(args.out_dir, "code-eval", "dolci-code-heldout.jsonl")
    outputs = [(train_path, train_rows, "train")] + ([(eval_path, eval_rows, "eval")] if eval_rows else [])
    for path, items, split in outputs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r, tests in items:
                f.write(json.dumps(to_record(r, tests, split), ensure_ascii=False) + "\n")
        print(f"wrote {len(items):6d} rows -> {path}")


if __name__ == "__main__":
    main()
