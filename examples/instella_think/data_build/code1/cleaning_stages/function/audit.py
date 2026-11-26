"""Stage 1: static test cleaning + execution of every reference and stub on original and candidate tests.

    python3 audit.py [n_rows|0] [workers]   ->  /tmp/v7fn/audit.pkl

Per row, candidate tests are built from the uncapped Dolci tests:
  dedupe (exact, then AST-equal) -> drop unparsable -> alias stdlib names the test uses without importing
  -> merge tests that read names assigned by an earlier test into one self-contained test.
Variants (float-tolerant, order-insensitive) are executed alongside so the decision stage can pick them.
Every program runs through sandbox.run (grader semantics: program + test as __main__, exit 0 = pass).
Result codes per test: 1 pass, A AssertionError, N NameError, Y TypeError, T timeout, X program-level
failure, E other exception.
"""
import collections
import hashlib
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import sandbox
import testlib as tl

WORK = "/tmp/v7fn"
MAX_REFS = 32
STUB_CONSTS = {"None": "None", "zero": "0", "empty_str": "''", "empty_list": "[]", "false": "False",
               "true": "True", "echo": None}


def code_char(r, prog):
    if r["rc"] == 0:
        return "1"
    if prog not in ("ok",):
        return "X"
    e = r.get("exc") or ""
    return {"AssertionError": "A", "NameError": "N", "TypeError": "Y", "TimeoutError": "T",
            "Timeout": "T"}.get(e, "E")


def stub_programs(fns):
    progs = {}
    for k, v in STUB_CONSTS.items():
        body = "return a[0] if a else (next(iter(k.values())) if k else None)" if v is None else f"return {v}"
        progs[k] = "".join(f"def {f}(*a, **k):\n    {body}\n" for f in fns) or "pass\n"
    progs["empty_program"] = "pass\n"
    return progs


def ref_order(refs):
    def key(x):
        if x["kind"] == "v6" and x["score"] >= 0.999:
            return (0, 0)
        if x["kind"] == "ace87k":
            return (1, 0)
        if x["kind"] == "olmo":
            return (2, 0)
        return (3, -(x["score"] or 0))
    return sorted(refs, key=key)[:MAX_REFS]


def prep(row):
    log = collections.Counter()
    prompt = row["prompt"]
    full = row["tests"]  # uncapped Dolci tests differ on only 17 rows; the training label is the base
    refs = ref_order(row["refs"])
    ref_top = set()
    for x in refs:
        tr = tl.parse(x["code"])
        if tr is not None:
            for st in tr.body:
                if isinstance(st, (ast_FunctionDef, ast_ClassDef)):
                    ref_top.add(st.name)
    # a. exact dedupe
    seen, t1 = set(), []
    for t in full:
        if t in seen:
            log["dup_exact"] += 1
            continue
        seen.add(t)
        t1.append(t)
    # b/c. parse + AST dedupe
    t2, keys = [], set()
    for t in t1:
        tree = tl.parse(t)
        if tree is None:
            log["syntax_error"] += 1
            continue
        k = tl.norm_key(tree)
        if k is not None and k in keys:
            log["dup_ast"] += 1
            continue
        keys.add(k)
        t2.append(t)
    # d. stdlib aliasing
    t3, aliased = [], set()
    for t in t2:
        tree = tl.parse(t)
        cand = [n for n in tl.free_names(tree) if (n in tl.MODULES or n in tl.SYMBOLS) and n not in ref_top
                and not re.search(r"(class|def)\s+" + re.escape(n) + r"\b", prompt)]
        if cand:
            t_new, done = tl.alias_stdlib(t, cand)
            if done:
                aliased |= set(done)
                log["stdlib_alias_tests"] += 1
                t = t_new
        t3.append(t)
    # e. cross-test name dependencies
    trees = [tl.parse(t) for t in t3]
    assigned = [tl.top_assigned(tr) for tr in trees]
    frees = [set(tl.free_names(tr)) for tr in trees]
    is_setup = [not tl.has_assert(tr) for tr in trees]
    merged_from = [None] * len(t3)
    consumed = set()
    forward = set()
    for i in range(len(t3)):
        need = {n for n in frees[i] if any(n in assigned[j] for j in range(len(t3)) if j != i)}
        if not need:
            continue
        chain, todo = set(), list(need)
        ok = True
        while todo:
            n = todo.pop()
            js = [j for j in range(i) if n in assigned[j] and j not in chain]
            if not js:
                if not any(n in assigned[j] for j in chain):
                    ok = False
                continue
            j = max(js)
            chain.add(j)
            todo += [m for m in frees[j] if any(m in assigned[q] for q in range(j))]
        if not ok:
            forward.add(i)
            continue
        merged_from[i] = sorted(chain)
    final = []
    for i, t in enumerate(t3):
        if merged_from[i]:
            consumed |= {j for j in merged_from[i] if is_setup[j]}
    for i, t in enumerate(t3):
        if i in consumed and is_setup[i]:
            log["setup_merged_away"] += 1
            continue
        if i in forward:
            log["dep_unresolvable"] += 1
            final.append({"t": t, "fix": ["dep_unresolved"], "src": i})
            continue
        if merged_from[i]:
            log["dep_merged"] += 1
            t = "\n".join(t3[j] for j in merged_from[i]) + "\n" + t
            final.append({"t": t, "fix": ["dep_merged"], "src": i})
        else:
            final.append({"t": t, "fix": [], "src": i})
    any_order = bool(tl.ANY_ORDER_RE.search(prompt))
    multi = bool(tl.MULTI_ANS_RE.search(prompt))
    for c in final:
        c["float"] = tl.float_rewrite(c["t"])
        c["order"] = tl.order_rewrite(c["t"]) if any_order else None
    ftrees = [tl.parse(c["t"]) for c in final]
    fns = []
    for tr in ftrees + [tl.parse(t) for t in row["tests"]]:
        if tr is None:
            continue
        for f in tl.called_free(tr):
            if f not in fns and not f.startswith("_v7_"):
                fns.append(f)
    iface = []
    for tr in ftrees:
        for f in tl.free_names(tr):
            if f not in iface and not f.startswith("_v7_"):
                iface.append(f)
    return {"log": log, "final": final, "refs": refs, "fns": fns, "iface": iface, "any_order": any_order,
            "multi": multi, "aliased": sorted(aliased)}


import ast  # noqa: E402
ast_FunctionDef, ast_ClassDef = (ast.FunctionDef, ast.AsyncFunctionDef), ast.ClassDef


def audit_row(row):
    try:
        return _audit_row(row)
    except Exception as e:
        return {"row": row["row"], "custom_id": row["custom_id"], "src": row["src"], "error": repr(e)[:300]}


def _audit_row(row):
    t0 = time.time()
    p = prep(row)
    tests, idx = [], {}

    def add(t):
        if t not in idx:
            idx[t] = len(tests)
            tests.append(t)
        return idx[t]

    orig_ix = [add(t) for t in row["tests"]]
    for c in p["final"]:
        c["ix"] = add(c["t"])
        c["fix_ix"] = add(c["float"]) if c["float"] else None
        c["ord_ix"] = add(c["order"]) if c["order"] else None
    res_refs = []
    for x in p["refs"]:
        out = sandbox.run(x["code"], tests, timeout=5.0)
        res_refs.append({"kind": x["kind"], "score": x["score"], "prog": out["prog"],
                         "v": "".join(code_char(r, out["prog"]) for r in out["res"]),
                         "h": hashlib.sha1(x["code"].encode()).hexdigest()[:12]})
    res_stub = {}
    for k, prog in stub_programs(p["fns"]).items():
        out = sandbox.run(prog, tests, timeout=5.0)
        res_stub[k] = "".join(code_char(r, out["prog"]) for r in out["res"])
    # state dependency on the original tests: best reference, sequential vs isolated
    seq = None
    orig_set = list(dict.fromkeys(orig_ix))
    best = None
    for i, rr in enumerate(res_refs):
        if rr["prog"] != "ok":
            continue
        s = sum(rr["v"][j] == "1" for j in orig_set)
        if best is None or s > best[0]:
            best = (s, i)
    if best is not None and best[0] < len(orig_set):
        out = sandbox.run(p["refs"][best[1]]["code"], [tests[j] for j in orig_set], timeout=5.0, seq=True)
        seq = {"ref": best[1], "v": "".join(code_char(r, out["prog"]) for r in out["res"]), "ix": orig_set}
    return {"row": row["row"], "custom_id": row["custom_id"], "src": row["src"], "tests": tests,
            "orig_ix": orig_ix, "prep": {k: v for k, v in p.items() if k != "refs"},
            "ref_codes": [x["code"] for x in p["refs"]], "refs": res_refs, "stubs": res_stub, "seq": seq,
            "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 26
    rows = pickle.load(open(f"{WORK}/rows.pkl", "rb"))
    if n:
        import random
        rows = random.Random(0).sample(rows, n)
    part = f"{WORK}/audit{'_' + str(n) if n else ''}.parts"
    done = {}
    if os.path.exists(part):
        with open(part, "rb") as f:
            while True:
                try:
                    r = pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    break
                done[r["row"]] = r
    print("resuming with", len(done), flush=True)
    todo = [r for r in rows if r["row"] not in done]
    todo.sort(key=lambda r: -len(r["refs"]) * len(r["tests"]))
    os.makedirs(sandbox.SCRATCH, exist_ok=True)
    from concurrent.futures import as_completed
    t = time.time()
    with ProcessPoolExecutor(workers) as ex, open(part, "ab") as fp:
        futs = [ex.submit(audit_row, r) for r in todo]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            pickle.dump(r, fp)
            fp.flush()
            done[r["row"]] = r
            if i % 500 == 0:
                print(i, f"{time.time() - t:.0f}s", flush=True)
    out = [done[r["row"]] for r in rows]
    pickle.dump(out, open(f"{WORK}/audit{'_' + str(n) if n else ''}.pkl", "wb"))
    print("done", len(out), "errors", sum("error" in r for r in out), f"{time.time() - t:.0f}s")
