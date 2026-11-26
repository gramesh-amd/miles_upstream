"""End-to-end sanity check of final_v1/train.jsonl through the grader server (training options).

  * 30 random rows with their verifying reference: binary reward must be 1 (every test passes), both on the
    full label and on the first 32 tests that training grades;
  * 30 other random rows with trivial stubs (assert route: every tested name returns None / 0 / '' / [] /
    False / True / its first argument, plus an empty program; stdio route: empty program, `print(0)`, and
    printing the row's most common expected output): binary reward must be 0 for every stub.

    python3 sanity.py  ->  final_v1/work/sanity.json
"""
import collections
import gzip
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function/scripts")
import testlib as tl  # noqa: E402
import gclient  # noqa: E402
from stdio_common import FINAL  # noqa: E402

CONST = {"None": "None", "zero": "0", "empty_str": "''", "empty_list": "[]", "false": "False", "true": "True",
         "echo": "a[0] if a else None"}


def stubs_for(row, tests):
    if row["metadata"]["route"] == "stdio" and isinstance(tests[0], dict):
        common = collections.Counter(str(t["output"]).strip() for t in tests).most_common(1)[0][0]
        return {"empty_program": "pass\n", "print_0": "print(0)\n", "print_most_common_output": f"print({common!r})\n"}
    names = []
    for t in tests:
        tr = tl.parse(t)
        if tr is not None:
            for f in tl.called_free(tr):
                if f not in names and not f.startswith("_v7_"):
                    names.append(f)
    out = {"empty_program": "pass\n"}
    for k, ret in CONST.items():
        src = []
        for n in names:
            if n[:1].isupper():
                src.append(f"class {n}:\n    def __init__(self, *a, **k):\n        pass\n"
                           f"    def __getattr__(self, _n):\n        return lambda *a, **k: (lambda a: {ret})(a)\n")
            else:
                src.append(f"def {n}(*a, **k):\n    return {ret}\n")
        out[k] = "".join(src) or "pass\n"
    return out


def main():
    rows = [json.loads(l) for l in open(f"{FINAL}/train.jsonl")]
    refs = json.load(gzip.open(f"{FINAL}/work/verifying_refs.json.gz", "rt"))
    rng = random.Random(20261118)
    pick = rng.sample(range(len(rows)), 60)
    ref_rows, stub_rows = [rows[i] for i in pick[:30]], [rows[i] for i in pick[30:]]

    def ref_check(row):
        cid = row["metadata"]["custom_id"]
        tests = json.loads(row["label"])
        r = refs[cid]
        kind = "assert" if isinstance(tests[0], str) else "stdio"
        opts = gclient.stdio_opts(row["metadata"])
        res = gclient.run(kind, r["prog"], tests, opts)
        return dict(custom_id=cid, route=row["metadata"]["route"], n=len(tests), passed=sum(res or []),
                    binary=int(bool(res) and all(res)), binary_first32=int(bool(res) and all(res[:32])))

    def stub_check(row):
        cid = row["metadata"]["custom_id"]
        tests = json.loads(row["label"])
        kind = "assert" if isinstance(tests[0], str) else "stdio"
        opts = gclient.stdio_opts(row["metadata"])
        out = {}
        for k, prog in stubs_for(row, tests).items():
            res = gclient.run(kind, prog, tests[:32], opts)
            out[k] = dict(frac=round(sum(res) / len(res), 3), binary=int(all(res)))
        return dict(custom_id=cid, route=row["metadata"]["route"], n=len(tests),
                    max_binary=max(v["binary"] for v in out.values()),
                    max_partial=max(v["frac"] for v in out.values()), stubs=out)

    with ThreadPoolExecutor(16) as ex:
        R = list(ex.map(ref_check, ref_rows))
        St = list(ex.map(stub_check, stub_rows))
    summary = dict(ref_rows=len(R), ref_binary_pass=sum(r["binary"] for r in R),
                   ref_binary_pass_first32=sum(r["binary_first32"] for r in R),
                   ref_routes=dict(collections.Counter(r["route"] for r in R)),
                   stub_rows=len(St), stub_any_binary_pass=sum(s["max_binary"] for s in St),
                   stub_routes=dict(collections.Counter(s["route"] for s in St)),
                   stub_max_partial_mean=round(sum(s["max_partial"] for s in St) / len(St), 3),
                   stub_max_partial_max=max(s["max_partial"] for s in St))
    json.dump(dict(summary=summary, ref=R, stub=St), open(f"{FINAL}/work/sanity.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in R:
        if not r["binary"]:
            print("REF FAIL", r)
    for s in St:
        if s["max_binary"]:
            print("STUB PASS", s)


if __name__ == "__main__":
    main()
