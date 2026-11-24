#!/usr/bin/env python3
"""Function-call slice: LeetCodeDataset (class Solution) and Codewars (top-level def) rows in the training
schema, verified on the sandbox.

    CODE_EXEC_API_URL=http://host:port/<token> python3 fn_build.py \
        --leetcode-pool leetcode_call.jsonl.gz --leetcode-refs leetcode_refs.jsonl.gz \
        --codewars-pool codewars_fn.jsonl.gz --codewars-refs WORK/refs/codewars_refs.json --out-dir WORK/build

Pool rows (one per problem, already decontaminated against LCB / HumanEval / MBPP and capped at 32 tests)
are LiveCodeBench-style prompts with a `call` label: {"route": "call", "fn_name", "tests": [{"input": one JSON
value per line, "output": JSON}]}, metadata uid / tier / date / difficulty / url. The training reward has no
call route, so each test becomes an assert with LCB's call semantics (a returned tuple compared as a list,
==), and the prompt is re-rendered as statement + LCB's starter-code instruction + starter, without LCB's
eval scaffolding. Rows that need ListNode / TreeNode helpers are dropped.

A reference (LeetCode: the audited row's `ref`; Codewars: build_refs.py's APPS solutions, <= 6 tried) runs
with LCB's imports prepended at REF_T. If it fails some tests the problem is kept with only the passed ones
when >= 80% (and >= 5) pass. Stubs must not pass every test. Writes OUT/fn_rows.jsonl and OUT/fn_dropped.csv.
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

import common
from common import RECIPE, read_jsonl, starter_prompt

sys.path.insert(0, RECIPE)
from prepare_code_benchmarks import LCB_IMPORTS, LCB_STARTER  # noqa: E402

# Half of training's 5 s, so a verified reference is not borderline at training time.
REF_T = 2.5
MAX_REFS = 6
MAX_ASSERT_BYTES = 200_000
MIN_TESTS = 5
REF_KEEP_FRAC = 0.8
Q_RE = re.compile(r"### Question:\n(.*?)\n\n### Format:", re.S)
STARTER_RE = re.compile(r"```python\n(.*?)```", re.S)
DATASET_SOURCE = {"leetcode": "newfacade/LeetCodeDataset", "codewars": "PrimeIntellect/verifiable-coding-problems"}


def split_prompt(prompt):
    user = prompt[-1]["content"]
    q = Q_RE.search(user)
    blocks = STARTER_RE.findall(user)
    if not q or not blocks:
        return None, None
    return q.group(1).strip(), blocks[-1].rstrip()


def to_assert(fn, cls, tc):
    args = [json.loads(x) for x in tc["input"].strip("\n").split("\n")]
    expected = json.loads(tc["output"].strip())
    call = f"Solution().{fn}" if cls else fn
    return (f"_r = {call}({', '.join(repr(a) for a in args)})\n"
            f"assert (list(_r) if type(_r) is tuple else _r) == {expected!r}")


def candidates(a, dropped):
    lc_ref = {r["uid"]: r.get("ref") for r in read_jsonl(a.leetcode_refs)}
    with open(a.codewars_refs) as f:
        cw_ref = json.load(f)
    rows = []
    for src, path in (("leetcode", a.leetcode_pool), ("codewars", a.codewars_pool)):
        for r in read_jsonl(path):
            m = r["metadata"]
            uid = m["uid"]
            lab = r["label"] if isinstance(r["label"], dict) else json.loads(r["label"])
            if lab.get("route") != "call":
                dropped.append((uid, src, "needs_listnode_treenode_helpers"))
                continue
            stmt, starter = split_prompt(r["prompt"])
            if stmt is None:
                dropped.append((uid, src, "prompt_unparsed"))
                continue
            cls = "class Solution" in starter
            try:
                tests = [to_assert(lab["fn_name"], cls, t) for t in lab["tests"]]
            except Exception as e:
                dropped.append((uid, src, f"test_convert:{type(e).__name__}"))
                continue
            tests = [t for t in tests if len(t) <= MAX_ASSERT_BYTES]
            if len(tests) < MIN_TESTS:
                dropped.append((uid, src, "lt5_tests"))
                continue
            ref = [lc_ref[uid]] if src == "leetcode" and lc_ref.get(uid) else cw_ref.get(uid, [])
            # tier B: LeetCode problems from LCB v3's date window that LCB did not sample; kept and flagged
            meta = {"rm_type": "code", "split": "train", "n_tests": len(tests), "dataset_source": DATASET_SOURCE[src],
                    "custom_id": f"v8_{src}_{uid.split('/', 1)[1]}", "src": src, "route": "function",
                    "fn_name": lab["fn_name"], "fixes": ["v8_call_to_assert", "v8_lcb_starter_prompt"],
                    "lcb_v3_flag": src == "leetcode" and m.get("tier") == "B", "date": m.get("date"),
                    "difficulty": m.get("difficulty"), "statement": stmt, "starter_code": starter, "upstream_uid": uid}
            rows.append({"prompt": [{"role": "user", "content": starter_prompt(stmt, starter, LCB_STARTER)}],
                         "label": json.dumps(tests, ensure_ascii=False), "metadata": meta, "_ref": ref})
    return rows


def check(r):
    """("ok", best stub pass rate) or (drop reason, None); trims r's label to the reference's passed tests."""
    if not r["_ref"]:
        return "no_reference", None
    best = None
    for ref in r["_ref"][:MAX_REFS]:
        res, _ = common.run(LCB_IMPORTS + "\n" + ref, r, timeout=REF_T)
        if res and (best is None or sum(res) > sum(best)):
            best = res
        if res and all(res):
            break
    if best is None:
        return "sandbox_error", None
    if min(best) == 0:
        tests = json.loads(r["label"])
        kept = [t for t, x in zip(tests, best, strict=True) if x]
        if len(kept) < REF_KEEP_FRAC * len(tests) or len(kept) < MIN_TESTS:
            return f"reference_fails_{len(best) - sum(best)}of{len(best)}", None
        r["label"] = json.dumps(kept, ensure_ascii=False)
        r["metadata"]["n_tests"] = len(kept)
        r["metadata"]["fixes"].append(f"drop_tests_reference_fails_80pct_rule:{len(tests) - len(kept)}")
    fn = r["metadata"]["fn_name"]
    cls = "class Solution" in r["metadata"]["starter_code"]
    stubs = ["", "def _x():\n    pass\n",
             f"class Solution:\n    def {fn}(self, *a, **k):\n        return None\n" if cls
             else f"def {fn}(*a, **k):\n    return None\n"]
    stub_best = 0.0
    for s in stubs:
        sr, _ = common.run(s, r, timeout=REF_T)
        if sr:
            stub_best = max(stub_best, sum(sr) / len(sr))
    if stub_best >= 1.0:
        return "stub_passes_all", stub_best
    return "ok", stub_best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leetcode-pool", required=True)
    ap.add_argument("--leetcode-refs", required=True, help="JSONL of {uid, ref}")
    ap.add_argument("--codewars-pool", required=True)
    ap.add_argument("--codewars-refs", required=True, help="build_refs.py output")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dropped = []
    rows = candidates(a, dropped)
    print("candidates", dict(collections.Counter(r["metadata"]["src"] for r in rows)), flush=True)
    keep = []
    for r, (status, stub) in zip(rows, common.pmap(check, rows, a.workers), strict=True):
        if status != "ok":
            dropped.append((r["metadata"]["upstream_uid"], r["metadata"]["src"], status))
            continue
        r["metadata"]["verified"] = True
        r["metadata"]["stub_floor"] = stub
        r.pop("_ref")
        keep.append(r)
    with open(os.path.join(a.out_dir, "fn_rows.jsonl"), "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out_dir, "fn_dropped.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "src", "reason"])
        w.writerows(dropped)
    print("kept", dict(collections.Counter(r["metadata"]["src"] for r in keep)))
    print("dropped", dict(collections.Counter((d[1], d[2].split(":")[0]) for d in dropped)))


if __name__ == "__main__":
    sys.exit(main())
