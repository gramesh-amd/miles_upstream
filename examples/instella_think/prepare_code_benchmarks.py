#!/usr/bin/env python3
"""Builds code benchmarks as miles eval sets, scored by code_reward.py.

OLMo 3's sets (oe-eval `::olmo3:adapt`):
    humanevalplus      evalplus/humanevalplus               164 problems
    mbppplus           evalplus/mbppplus                    378 problems
    lcb_v3             livecodebench/code_generation_lite   612 problems (release_v3)
Sets used by recent reasoning-model reports (same prompts / grading where they overlap):
    lcb_v5             release_v5, 2024-08-01 <= contest_date < 2025-02-01   279 problems
    lcb_v6             test6.jsonl (the release_v6 increment)                175 problems
    lcb_v6_strict      test6.jsonl, 2025-02-01 <= contest_date < 2025-05-01  131 problems
    humanevalpro       CodeEval-Pro/humaneval-pro (self-invoking)            164 problems
    mbpppro            CodeEval-Pro/mbpp-pro                                 378 problems
    cruxeval_o         cruxeval-org/cruxeval, output prediction              800 problems
    cruxeval_i         cruxeval-org/cruxeval, input prediction               800 problems
    codeforces         open-r1/codeforces `verifiable` test split            422 problems
                       (Aug 2024 - Feb 2025, stdio, 45 with a checker; only with --codeforces)

Prompts, answer extraction, program assembly, tests and timeouts for OLMo 3's sets follow
oe-eval's task configs (allenai/olmes, oe_eval/configs/tasks.py), which is what the OLMo 3
Think results (arXiv 2512.13961, Table 16) were measured with. The sampling half of that
protocol (10 samples, temperature 0.6, top_p 0.95, 32768 tokens) is the eval config
run_code.sh passes. lcb_v5/v6 reuse lcb_v3's prompt and grading unchanged; the Pro sets use
the HumanEval+ answer format; CRUXEval uses its CoT prompts with an [ANSWER] tag; Codeforces
uses the LiveCodeBench stdin prompt and open-r1's (piston) judging rules.

LiveCodeBench and Codeforces ship GBs of tests, so their rows reference per-problem test
files instead of embedding them; code_exec_server.py loads them from --tests-dir. That
directory is <out>/tests and must be readable on the code-exec host.

    python3 prepare_code_benchmarks.py --out-dir "${DATA_ROOT}/code-bench"
    python3 prepare_code_benchmarks.py --out-dir ... --sets lcb_v5 lcb_v6 --codeforces

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
    # release_vN = test.jsonl + ... + testN.jsonl (code_generation_lite.py); test4-6 are 2.0 GB.
    **{f"lcb_test{s}.jsonl": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
                              f"test{s}.jsonl") for s in ("", "2", "3", "4", "5", "6")},
    "humanevalpro.parquet": (
        "CodeEval-Pro/humaneval-pro", "cd078f93d57d1902b5c3e4ae330166b2ca0e0e80",
        "data/train-00000-of-00001.parquet"),
    "mbpppro.parquet": (
        "CodeEval-Pro/mbpp-pro", "50f18448e09a8383226e1a5cd3654d2a454fe333",
        "data/train-00000-of-00001.parquet"),
    "cruxeval.jsonl": ("cruxeval-org/cruxeval", "b96af0450242eb4da433032b90998f25588a5d0f", "test.jsonl"),
}
CF_REPO, CF_REV = "open-r1/codeforces", "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
CF_FILES = {"cf_verifiable_test.parquet": (CF_REPO, CF_REV, "verifiable/test-00000-of-00001.parquet")}
# Which sets need which raw files (fetch only what is built).
NEEDS = {
    "humanevalplus": ["humanevalplus.parquet"], "mbppplus": ["mbppplus.parquet"],
    "lcb_v3": ["lcb_test.jsonl", "lcb_test2.jsonl", "lcb_test3.jsonl"],
    "lcb_v5": [f"lcb_test{s}.jsonl" for s in ("", "2", "3", "4", "5")],
    "lcb_v6": ["lcb_test6.jsonl"], "lcb_v6_strict": ["lcb_test6.jsonl"],
    "humanevalpro": ["humanevalpro.parquet"], "mbpppro": ["mbpppro.parquet"],
    "cruxeval_o": ["cruxeval.jsonl"], "cruxeval_i": ["cruxeval.jsonl"],
    "codeforces": ["cf_verifiable_test.parquet"],
}
DEFAULT_SETS = ["humanevalplus", "mbppplus", "lcb_v3", "lcb_v5", "lcb_v6", "lcb_v6_strict",
                "humanevalpro", "mbpppro", "cruxeval_o", "cruxeval_i"]

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

# HumanEval Pro / MBPP Pro (arXiv 2412.21199): the base problem plus a harder problem whose
# solution must call it. Same answer format as HE_TEMPLATE, so the olmes extractor applies.
PRO_TEMPLATE = (
    "Complete the following two functions. The second function must call the first one.\n"
    "```python\n{{raw_problem}}\n\n{{new_problem}}\n```\nProvide CONCISE reasoning on how to arrive at the "
    "answer, and make sure to finish the response with the following, where (CODE) is the code for "
    "both complete functions, including any imports they need:\n\nHere is the completed function:\n\n"
    "```python\n(CODE)\n```"
)
PRO_PREFIX = "from typing import *\nimport math\n"

# CRUXEval CoT prompts (facebookresearch/cruxeval, prompts.py make_cot_{output,input}_prompt),
# minus the trailing "[THOUGHT]" cue, since a thinking model reasons in its think block.
CRUX_O_TEMPLATE = (
    "You are given a Python function and an assertion containing an input to the function. Complete the "
    "assertion with a literal (no unsimplified expressions, no function calls) containing the output when "
    "executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT "
    "output any extra information. Execute the program step by step before arriving at an answer, and "
    "provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the "
    "examples.\n\n"
    "[PYTHON]\ndef f(s):\n    s = s + s\n    return \"b\" + s + \"a\"\nassert f(\"hi\") == ??\n[/PYTHON]\n"
    "[THOUGHT]\nLet's execute the code step by step:\n\n"
    "1. The function f is defined, which takes a single argument s.\n"
    "2. The function is called with the argument \"hi\", so within the function, s is initially \"hi\".\n"
    "3. Inside the function, s is concatenated with itself, so s becomes \"hihi\".\n"
    "4. The function then returns a new string that starts with \"b\", followed by the value of s (which is "
    "now \"hihi\"), and ends with \"a\".\n"
    "5. The return value of the function is therefore \"bhihia\".\n[/THOUGHT]\n"
    "[ANSWER]\nassert f(\"hi\") == \"bhihia\"\n[/ANSWER]\n\n"
    "[PYTHON]\n{{code}}\nassert f({{input}}) == ??\n[/PYTHON]"
)
CRUX_I_TEMPLATE = (
    "You will be given a function f and an output in the form f(??) == output. Your task is to find any "
    "input such that executing f on the input leads to the given output. There may be multiple answers, but "
    "only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] "
    "tags. Express your answer as a passing assertion containing the input and the given output.\n\n"
    "[PYTHON]\ndef f(x):\n    return x + 1\nassert f(??) == 17\n[/PYTHON]\n"
    "[THOUGHT]\nTo find an input such that executing f on the input leads to the given output, we can work "
    "backwards from the given assertion. We know that f(??) == 17. \n\n"
    "Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. \n"
    "[/THOUGHT]\n[ANSWER]\nassert f(16) == 17\n[/ANSWER]\n\n"
    "[PYTHON]\n{{code}}\nassert f(??) == {{output}}\n[/PYTHON]"
)
# cruxeval evaluation/utils_execute.py BASE_IMPORTS, without numpy.
CRUX_IMPORTS = (
    "from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat\n"
    "from copy import deepcopy\nfrom string import ascii_lowercase\n"
    "from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt\n"
    "from collections import defaultdict, deque, Counter\n"
    "from bisect import bisect, bisect_left, bisect_right, insort\n"
    "from heapq import heappush, heappop, heapify, merge\nfrom functools import reduce, cache, lru_cache\n"
    "from random import randrange, shuffle\nfrom operator import itemgetter, sub\n"
    "from re import search as re_search\nfrom os.path import commonprefix\n"
    "from typing import List, Tuple, Dict, Set, Optional, Callable, Union, Any\n"
    "import string\nimport re\nimport collections\nimport math\nimport sys\nimport random\n"
)
CRUX_TIMEOUT = 10.0

# Codeforces: Python gets a multiple of the C++ time limit (open-r1 judges with the raw limit
# because it mostly grades C++). Tests per problem are capped so the bench sandbox's
# --max-tests never truncates a row.
CF_TL_MULT, CF_TL_MIN, CF_TL_MAX = 3.0, 6.0, 20.0
CF_MAX_TESTS = 64
CF_MAX_TEST_MB, CF_MAX_PROBLEM_MB = 8, 64


def fetch(raw_dir, names):
    os.makedirs(raw_dir, exist_ok=True)
    for name in names:
        _fetch_one(os.path.join(raw_dir, name), *{**FILES, **CF_FILES}[name])
        print(f"ok {name}")


def _fetch_one(dst, repo, rev, path, quiet=False):
    meta = json.load(urllib.request.urlopen(urllib.request.Request(
        f"https://huggingface.co/api/datasets/{repo}/paths-info/{rev}",
        data=f"paths={path}".encode())))
    # LFS files are checked by sha256, small git-stored files by their git blob sha1.
    lfs = meta[0].get("lfs")
    want, digest = (lfs["oid"], _sha256) if lfs else (meta[0]["oid"], _git_sha1)
    if not os.path.exists(dst) or digest(dst) != want:
        if not quiet:
            print(f"downloading {repo}@{rev[:8]}/{path}")
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}", dst + ".part")
        os.replace(dst + ".part", dst)
        if digest(dst) != want:
            raise SystemExit(f"{dst}: checksum mismatch after download")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha1(path):
    h = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        h.update(f.read())
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


# (bench, raw files, contest_date window [start, end), tests directory). lcb_v6_strict is a
# subset of lcb_v6 and shares its test files.
LCB_SETS = {
    "lcb_v3": (NEEDS["lcb_v3"], None, None, "lcb_v3"),
    "lcb_v5": (NEEDS["lcb_v5"], "2024-08-01", "2025-02-01", "lcb_v5"),
    "lcb_v6": (NEEDS["lcb_v6"], None, None, "lcb_v6"),
    "lcb_v6_strict": (NEEDS["lcb_v6"], "2025-02-01", "2025-05-01", "lcb_v6"),
}
EXPECTED_ROWS = {"lcb_v3": 612, "lcb_v5": 279, "lcb_v6": 175, "lcb_v6_strict": 131,
                 "humanevalpro": 164, "mbpppro": 378, "cruxeval_o": 800, "cruxeval_i": 800}


def livecodebench(raw_dir, tests_dir, bench="lcb_v3"):
    files, start, end, tests_bench = LCB_SETS[bench]
    os.makedirs(os.path.join(tests_dir, tests_bench), exist_ok=True)
    for name in files:
        with open(os.path.join(raw_dir, name)) as f:
            for line in f:
                d = json.loads(line)
                date = d["contest_date"][:10]
                if (start and date < start) or (end and date >= end):
                    continue
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
                ref = tests_bench + "/" + re.sub(r"[^A-Za-z0-9_.-]", "_", d["question_id"]) + ".json"
                with open(os.path.join(tests_dir, ref), "w") as tf:
                    json.dump(tests, tf, ensure_ascii=False)
                label = {"bench": bench, "extract": "lcb", "timeout": LCB_TIMEOUT,
                         "tests_ref": ref, "n_tests": len(tests)}
                if fn_name:
                    # grade_call_based: import_string + "\n\n" + code
                    label.update(route="call", fn_name=fn_name, prefix=LCB_IMPORTS + "\n\n")
                else:
                    # grade_stdio: make_function puts import_string + "\n" ahead of the code
                    label.update(route="stdio", prefix=LCB_IMPORTS + "\n")
                row = _row(bench, d["question_id"],
                           [{"role": "system", "content": LCB_SYSTEM}, {"role": "user", "content": content}],
                           label)
                if bench != "lcb_v3":
                    row["metadata"].update(difficulty=d["difficulty"], platform=d["platform"], contest_date=date)
                yield row


def _imports_of(src):
    return "".join(ln + "\n" for ln in src.splitlines() if re.match(r"(from\s+\S+\s+)?import\s", ln))


def codeeval_pro(raw_dir, bench):
    """HumanEval Pro / MBPP Pro. The answer must hold both functions; the prefix only adds the
    base problem's imports and the usual typing/math names, so a model that leaves out the base
    function fails, as in the paper's harness (which runs the extracted code plus test_code)."""
    for d in pd.read_parquet(os.path.join(raw_dir, f"{bench}.parquet")).to_dict("records"):
        raw_problem, new_problem = d["raw_problem"].strip("\n"), d["new_problem"].strip("\n")
        content = JINJA.from_string(PRO_TEMPLATE).render(raw_problem=raw_problem, new_problem=new_problem)
        yield _row(bench, f"{bench}/{d['id']}", [{"role": "user", "content": content}], {
            "bench": bench, "extract": "olmes", "route": "script", "timeout": EVALPLUS_TIMEOUT,
            "prefix": PRO_PREFIX + _imports_of(d["raw_problem"]),
            "test": d["test_code"],
        })


def pro_canonical(d):
    """The dataset's reference: both problems with their solutions, as one program."""
    return f"{d['raw_problem']}{d['raw_solution']}\n\n{d['new_problem']}{d['new_solution']}\n"


def cruxeval(raw_dir, mode):
    """CRUXEval-O / -I, graded like cruxeval's evaluate_generations: `assert <gold output> ==
    <prediction>`, with the prediction a literal (O) or an f(...) call (I). code_reward's
    "crux" extractor turns the [ANSWER] assertion into `__pred = (...)`."""
    bench = f"cruxeval_{mode}"
    with open(os.path.join(raw_dir, "cruxeval.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            tmpl = CRUX_O_TEMPLATE if mode == "o" else CRUX_I_TEMPLATE
            content = JINJA.from_string(tmpl).render(code=d["code"], input=d["input"], output=d["output"])
            yield _row(bench, d["id"], [{"role": "user", "content": content}], {
                "bench": bench, "extract": "crux", "crux_mode": mode, "route": "script",
                "timeout": CRUX_TIMEOUT, "prefix": CRUX_IMPORTS + "\n" + d["code"] + "\n\n",
                "test": f"assert {d['output']} == __pred",
            })


def _cf_statement(d):
    parts = [f"Execution time limit: {d['time_limit']} seconds\nMemory limit: {d['memory_limit']} MB\n",
             f"# Problem\n{d['description']}"]
    for key, title in (("input_format", "Input Format"), ("output_format", "Output Format")):
        if d.get(key):
            parts.append(f"## {title}\n{d[key]}")
    ex = "".join(f"```input\n{e['input'].strip()}\n```\n```output\n{e['output'].strip()}\n```\n"
                 for e in (d.get("examples") if d.get("examples") is not None else []))
    if ex:
        parts.append("## Examples\n" + ex.rstrip("\n"))
    if d.get("note"):
        parts.append(f"## Note\n{d['note']}")
    return "\n\n".join(parts)


def _crlf(s):
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _cf_cap(tests, max_test_mb, max_problem_mb):
    """Drops tests over max_test_mb, then the largest ones until the problem fits in
    max_problem_mb: the sandbox parses and caches a problem's whole test file in memory."""
    size = [len(t["input"]) + len(t["output"]) for t in tests]
    keep = [i for i, s in enumerate(size) if s <= max_test_mb * 1e6]
    for i in sorted(keep, key=lambda i: -size[i]):
        if sum(size[j] for j in keep) <= max_problem_mb * 1e6:
            break
        keep.remove(i)
    return [tests[i] for i in sorted(keep)]


def codeforces(raw_dir, tests_dir, max_tests=CF_MAX_TESTS, max_test_mb=CF_MAX_TEST_MB,
               max_problem_mb=CF_MAX_PROBLEM_MB):
    """open-r1/codeforces `verifiable` test split (contests 2024-08 .. 2025-02; open-r1 asks
    not to train on it). Tests = R1-generated hard tests from
    generated_tests/test_cases_<contest>.parquet, then official ones (complete or < 400
    chars), deduplicated by input; capped at max_tests and by size (_cf_cap). Rows with a
    generated_checker are judged by it (route "checker"), the rest by piston's token diff."""
    probs = pd.read_parquet(os.path.join(raw_dir, "cf_verifiable_test.parquet"))
    probs = probs[probs["interaction_format"].fillna("").str.len() == 0]
    gen_dir = os.path.join(raw_dir, "cf_generated")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(os.path.join(tests_dir, "codeforces"), exist_ok=True)
    for contest, group in probs.groupby("contest_id", sort=True):
        gen = {}
        if (group["generated_tests"] > 0).any():
            name = f"test_cases_{int(contest):04d}.parquet"
            dst = os.path.join(gen_dir, name)
            _fetch_one(dst, CF_REPO, CF_REV, f"generated_tests/{name}", quiet=True)
            g = pd.read_parquet(dst, columns=["problem_id", "input", "output", "test_i"]).sort_values("test_i")
            for pid, rows in g.groupby("problem_id"):
                gen[pid] = list(zip(rows["input"], rows["output"]))
        for d in group.sort_values("index").to_dict("records"):
            seen, tests = set(), []
            for src in (gen.get(d["id"], []), [(t["input"], t["output"]) for t in d["official_tests"]]):
                for i, o in src:
                    i, o = _crlf(i), _crlf(o)
                    if i not in seen:
                        seen.add(i)
                        tests.append({"input": i, "output": o})
            n_all = len(tests)
            tests = _cf_cap(tests[:max_tests], max_test_mb, max_problem_mb)
            if not tests:
                continue
            ref = "codeforces/" + d["id"].replace("/", "_") + ".json"
            with open(os.path.join(tests_dir, ref), "w") as tf:
                json.dump(tests, tf, ensure_ascii=False)
            content = JINJA.from_string(LCB_TEMPLATE).render(problem_statement=_cf_statement(d),
                                                             format_instruction=LCB_STDIN)
            timeout = min(CF_TL_MAX, max(CF_TL_MIN, CF_TL_MULT * float(d["time_limit"])))
            label = {"bench": "codeforces", "extract": "lcb", "route": "checker", "timeout": timeout,
                     "tests_ref": ref, "n_tests": len(tests), "prefix": LCB_IMPORTS + "\n"}
            if d.get("generated_checker"):
                label["checker"] = d["generated_checker"]
            row = _row("codeforces", d["id"],
                       [{"role": "system", "content": LCB_SYSTEM}, {"role": "user", "content": content}],
                       label)
            rating = d.get("rating")
            row["metadata"].update(rating=None if pd.isna(rating) else int(rating), contest_id=str(contest),
                                   time_limit=float(d["time_limit"]), checker=bool(d.get("generated_checker")),
                                   tests_dropped=n_all - len(tests))
            yield row


def build(name, raw_dir, tests_dir, cf_max_tests=CF_MAX_TESTS, cf_max_test_mb=CF_MAX_TEST_MB,
          cf_max_problem_mb=CF_MAX_PROBLEM_MB):
    if name == "humanevalplus":
        return humanevalplus(raw_dir)
    if name == "mbppplus":
        return mbppplus(raw_dir)
    if name in LCB_SETS:
        return livecodebench(raw_dir, tests_dir, name)
    if name in ("humanevalpro", "mbpppro"):
        return codeeval_pro(raw_dir, name)
    if name in ("cruxeval_o", "cruxeval_i"):
        return cruxeval(raw_dir, name[-1])
    if name == "codeforces":
        return codeforces(raw_dir, tests_dir, cf_max_tests, cf_max_test_mb, cf_max_problem_mb)
    raise SystemExit(f"unknown set {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", default=None, help="downloads land here (default <out-dir>/raw)")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, choices=list(NEEDS))
    ap.add_argument("--codeforces", action="store_true",
                    help="also build codeforces (downloads ~9 GB of generated tests)")
    ap.add_argument("--cf-max-tests", type=int, default=CF_MAX_TESTS)
    ap.add_argument("--cf-max-test-mb", type=float, default=CF_MAX_TEST_MB)
    ap.add_argument("--cf-max-problem-mb", type=float, default=CF_MAX_PROBLEM_MB)
    args = ap.parse_args()
    sets = list(dict.fromkeys(args.sets + (["codeforces"] if args.codeforces else [])))
    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw")
    os.makedirs(args.out_dir, exist_ok=True)
    fetch(raw_dir, list(dict.fromkeys(f for s in sets for f in NEEDS[s])))
    tests_dir = os.path.join(args.out_dir, "tests")
    for name in sets:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        n = 0
        with open(path + ".tmp", "w") as f:
            for r in build(name, raw_dir, tests_dir, args.cf_max_tests, args.cf_max_test_mb,
                           args.cf_max_problem_mb):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        if name in EXPECTED_ROWS and n != EXPECTED_ROWS[name]:
            raise SystemExit(f"{name}: {n} rows, expected {EXPECTED_ROWS[name]}")
        os.replace(path + ".tmp", path)
        print(f"wrote {path}: {n} rows")


if __name__ == "__main__":
    main()
