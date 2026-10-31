#!/usr/bin/env python3
"""Builds OLMo 3's code benchmarks as miles eval sets, scored by code_reward.py.

    HumanEval+         evalplus/humanevalplus               164 problems
    MBPP+              evalplus/mbppplus                    378 problems
    LiveCodeBench v3   livecodebench/code_generation_lite   612 problems (release_v3)

Prompts, answer extraction, program assembly, tests and timeouts follow oe-eval's
`::olmo3:adapt` task configs (allenai/olmes, oe_eval/configs/tasks.py), which is what the
OLMo 3 Think results (arXiv 2512.13961, Table 16) were measured with. The sampling half of
that protocol (10 samples, temperature 0.6, top_p 0.95, 32768 tokens) is the eval config
run_code.sh passes when CODE_BENCH=1.

LiveCodeBench ships ~5 GB of tests, so its rows reference per-problem test files instead
of embedding them; code_exec_server.py loads them from --tests-dir. That directory is
<out>/tests and must be readable on the code-exec host.

    python3 prepare_code_benchmarks.py --out-dir "${DATA_ROOT}/code-bench"

Needs pandas, pyarrow and jinja2 (all in the miles container).
"""

import argparse
import base64
import hashlib
import json
import os
import pickle
import re
import urllib.request
import zlib

import jinja2
import pandas as pd

# Pinned dataset revisions and the sha256 of each file at that revision.
FILES = {
    "humanevalplus.parquet": (
        "evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        "data/test-00000-of-00001-5973903632b82d40.parquet"),
    "mbppplus.parquet": (
        "evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        "data/test-00000-of-00001-d5781c9c51e02795.parquet"),
    # release_v3 = test.jsonl + test2.jsonl + test3.jsonl (code_generation_lite.py).
    "lcb_test.jsonl": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "test.jsonl"),
    "lcb_test2.jsonl": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "test2.jsonl"),
    "lcb_test3.jsonl": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "test3.jsonl"),
}

# oe-eval renders templates with this environment (oe_eval/tasks/utils.py).
JINJA = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined,
                           keep_trailing_newline=True)

HE_TEMPLATE = (
    "Complete the following function:\n{{prompt}}\nProvide CONCISE reasoning on how to arrive at the "
    "answer, and make sure to finish the response with the following, where (CODE) is the code for "
    "the complete function:\n\nHere is the completed function:\n\n```python\n(CODE)\n```"
)
MBPP_TEMPLATE = (
    "{{code_prompt}}\n\nProvide CONCISE reasoning on how to arrive at the answer, and make sure to "
    "finish the response with the following:\n\nHere is the completed function:\n\n```python\n(CODE)\n```"
    "\nwhere (CODE) is the code for the complete function."
)
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests."
)
LCB_TEMPLATE = (
    "### Question:\n{{problem_statement}}\n\n### Format:\nProvide CONCISE reasoning on how to arrive at "
    "the answer in the <think> </think> tag.\n{{format_instruction}}\n\n### Answer: (use the provided "
    "format with backticks)\n\n"
)
LCB_STARTER = ("You will use the following starter code to write the solution to the problem and "
               "enclose your code within delimiters.")
LCB_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
    "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```"
)
# lcb_runner / oe_eval testing_util.import_string, prepended to every LiveCodeBench program.
LCB_IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
    "import json\nsys.setrecursionlimit(50000)\n"
)
EVALPLUS_TIMEOUT = 20.0  # metric_kwargs.timeout, whole program
LCB_TIMEOUT = 6.0        # per test


def fetch(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    for name, (repo, rev, path) in FILES.items():
        dst = os.path.join(raw_dir, name)
        meta = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://huggingface.co/api/datasets/{repo}/paths-info/{rev}",
            data=f"paths={path}".encode())))
        want = meta[0]["lfs"]["oid"]
        if not os.path.exists(dst) or _sha256(dst) != want:
            print(f"downloading {repo}@{rev[:8]}/{path}")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}", dst + ".part")
            os.replace(dst + ".part", dst)
            if _sha256(dst) != want:
                raise SystemExit(f"{dst}: sha256 mismatch after download")
        print(f"ok {name} ({repo}@{rev[:8]})")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _row(bench, task_id, messages, label):
    return {"prompt": messages, "label": json.dumps(label, ensure_ascii=False),
            "metadata": {"rm_type": "code", "split": "eval", "bench": bench, "task_id": task_id}}


def humanevalplus(raw_dir):
    for d in pd.read_parquet(os.path.join(raw_dir, "humanevalplus.parquet")).to_dict("records"):
        content = JINJA.from_string(HE_TEMPLATE).render(prompt=d["prompt"])
        yield _row("humanevalplus", d["task_id"], [{"role": "user", "content": content}], {
            "bench": "humanevalplus", "extract": "olmes", "route": "script", "timeout": EVALPLUS_TIMEOUT,
            "prefix": d["prompt"],
            "test": d["test"] + f"\ncheck({d['entry_point']})",
        })


def mbppplus(raw_dir):
    for d in pd.read_parquet(os.path.join(raw_dir, "mbppplus.parquet")).to_dict("records"):
        signature = d["code"].split(":")[0] + ":"
        content = JINJA.from_string(MBPP_TEMPLATE).render(code_prompt=d["prompt"] + signature)
        yield _row("mbppplus", d["task_id"], [{"role": "user", "content": content}], {
            "bench": "mbppplus", "extract": "olmes", "route": "script", "timeout": EVALPLUS_TIMEOUT,
            # oe-eval's "hacky way to get in the original imports if any" (codex_mbpp.py)
            "prefix": signature + "\n    '''doc_string'''\n\n",
            "test": d["test"],
        })


def _private_tests(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def livecodebench(raw_dir, tests_dir):
    os.makedirs(os.path.join(tests_dir, "lcb_v3"), exist_ok=True)
    for name in ("lcb_test.jsonl", "lcb_test2.jsonl", "lcb_test3.jsonl"):
        with open(os.path.join(raw_dir, name)) as f:
            for line in f:
                d = json.loads(line)
                public = json.loads(d["public_test_cases"]) if d.get("public_test_cases") else []
                private = _private_tests(d["private_test_cases"]) if d.get("private_test_cases") else []
                tests = [{"input": t["input"], "output": t["output"]} for t in public + private]
                fn_name = (json.loads(d["metadata"]) if d.get("metadata") else {}).get("func_name")
                if d.get("starter_code", "").strip():
                    fmt = LCB_STARTER + f"\n```python\n{d['starter_code']}\n```"
                else:
                    fmt = LCB_STDIN
                content = JINJA.from_string(LCB_TEMPLATE).render(
                    problem_statement=d["question_content"], format_instruction=fmt)
                ref = "lcb_v3/" + re.sub(r"[^A-Za-z0-9_.-]", "_", d["question_id"]) + ".json"
                with open(os.path.join(tests_dir, ref), "w") as tf:
                    json.dump(tests, tf, ensure_ascii=False)
                label = {"bench": "lcb_v3", "extract": "lcb", "timeout": LCB_TIMEOUT,
                         "tests_ref": ref, "n_tests": len(tests)}
                if fn_name:
                    # grade_call_based: import_string + "\n\n" + code
                    label.update(route="call", fn_name=fn_name, prefix=LCB_IMPORTS + "\n\n")
                else:
                    # grade_stdio: make_function puts import_string + "\n" ahead of the code
                    label.update(route="stdio", prefix=LCB_IMPORTS + "\n")
                yield _row("lcb_v3", d["question_id"],
                           [{"role": "system", "content": LCB_SYSTEM}, {"role": "user", "content": content}],
                           label)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", default=None, help="downloads land here (default <out-dir>/raw)")
    args = ap.parse_args()
    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw")
    os.makedirs(args.out_dir, exist_ok=True)
    fetch(raw_dir)
    tests_dir = os.path.join(args.out_dir, "tests")
    for name, rows in (("humanevalplus", humanevalplus(raw_dir)), ("mbppplus", mbppplus(raw_dir)),
                       ("lcb_v3", livecodebench(raw_dir, tests_dir))):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        n = 0
        with open(path + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        os.replace(path + ".tmp", path)
        print(f"wrote {path}: {n} rows")


if __name__ == "__main__":
    main()
