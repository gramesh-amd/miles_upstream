"""Verification for rows whose tested function name is missing from the prompt and whose references all
picked a different name. For each reference, every top-level function it defines is tried as the
tested name (`<tested> = <candidate>` appended); a reference that then passes every final test verifies
both the tests and the signature (its parameter list becomes the stated signature).

    python3 rename_verify.py [workers]  ->  /tmp/v7fn/rename.pkl  {row: {name: (ref index, cand, args)}}
"""
import ast
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import sandbox
import testlib as tl

WORK = "/tmp/v7fn"


def job(args):
    row, missing, codes, tests = args
    found = {}
    for name in missing:
        for ri, code in enumerate(codes):
            tree = tl.parse(code)
            if tree is None:
                continue
            cands = [st for st in tree.body if isinstance(st, ast.FunctionDef) and not st.name.startswith("_")]
            for st in cands[:6]:
                prog = code + f"\n\n{name} = {st.name}\n"
                out = sandbox.run(prog, tests, timeout=5.0)
                if out["prog"] == "ok" and all(r["rc"] == 0 for r in out["res"]):
                    found[name] = (ri, st.name, tl.render_args(st.args))
                    break
            if name in found:
                break
    return row, found


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    D = pickle.load(open(f"{WORK}/decide.pkl", "rb"))
    A = {a["row"]: a for a in pickle.load(open(f"{WORK}/audit.pkl", "rb"))}
    jobs = []
    for d in D:
        if d["drop"] or d.get("verified") or not d.get("sig"):
            continue
        missing = [x["name"] for x in d["sig"]["details"] if x["kind"] in ("def", "name")]
        if len(missing) != 1:
            continue
        a = A[d["row"]]
        codes = [a["ref_codes"][i] for i, x in enumerate(a["refs"]) if x["prog"] == "ok"]
        jobs.append((d["row"], missing, codes[:12], d["tests_final"]))
    print("rows", len(jobs), flush=True)
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for row, found in ex.map(job, jobs, chunksize=1):
            if found:
                res[row] = found
    pickle.dump(res, open(f"{WORK}/rename.pkl", "wb"))
    print("verified by rename", len(res))
