"""Stage 2: per-row decisions from audit.pkl, plus the extra executions they need (signature stubs,
probes of reference outputs on blocking tests).

    python3 decide.py  ->  /tmp/v7fn/decide.pkl

Order per row (each step works on the output of the previous one):
  1. pick the test version: float-tolerant variant for exact float compares; order-insensitive variant when
     the statement says any order;
  2. drop tests whose names come from a later test (unresolvable dependency) and tests that only pass when
     state from an earlier test leaks in (sequential pass, isolated fail for the best reference);
  3. drop vacuous tests: no assert at all, or every constant stub (and so any program that defines the
     name) passes them;
  4. stub floor on the remaining tests: a constant passing all -> drop row; best stub >= 50% -> prune
     every stub-passable test;
  5. broken test: >= 2 distinct references fail exactly the same single test, no reference passes it,
     and the near-miss references agree on one output for it -> drop that test (evidence recorded);
  6. fewer than 3 meaningful (non-stub-passable) tests -> drop row;
  7. tested names missing from the prompt -> append an `Implement ...` line with the signature taken from
     the passing references (name/arity checked by a signature stub);
  8. flags: prompt examples reused as tests, multi-answer statements, unverified.
Prompt dedup across rows is done in finalize.py.
"""
import ast
import collections
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import sandbox
import testlib as tl

WORK = "/tmp/v7fn"
OUT = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function"
# Blocking tests read by hand (handcheck_blocking.csv): W = the statement proves the test wrong (drop the
# test), U = the test encodes a convention the statement does not fix (drop the row), M = the test is right.
HAND = {}
if os.path.exists(f"{OUT}/handcheck_blocking.csv"):
    import csv
    HAND = {r["custom_id"]: r["verdict"] for r in csv.DictReader(open(f"{OUT}/handcheck_blocking.csv"))}
RENAME = pickle.load(open(f"{WORK}/rename.pkl", "rb")) if os.path.exists(f"{WORK}/rename.pkl") else {}
HIDDEN_RE = re.compile(r"(predefined|pre-defined|simulated (database|db)|mock(ed)? (model|data|database)|"
                       r"hard-?coded (list|set|mapping)|existing (users?|records?) (list|set))", re.I)
CONSTS = ["None", "zero", "empty_str", "empty_list", "false", "true", "echo"]


def ok_refs(a):
    return [i for i, x in enumerate(a["refs"]) if x["prog"] == "ok"]


def word_in(name, text):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text) is not None


def plan(a, row):
    """Decisions that need no new execution. Returns dict with chosen tests and pending execution."""
    T = a["tests"]
    P = a["prep"]
    refs = ok_refs(a)
    V = {i: a["refs"][i]["v"] for i in refs}
    S = a["stubs"]
    d = {"fixes": collections.Counter(), "flags": {}, "drop": None, "evidence": {}}
    # 1. versions
    cur = []  # list of (test index, info)
    order_ok = tl.output_order_free(row["prompt"])
    for c in P["final"]:
        k = c["ix"]
        info = {"src": c["src"], "fix": list(c["fix"]), "base": c["ix"]}
        if c["fix_ix"] is not None:
            k = c["fix_ix"]
            info["fix"].append("float_tol")
            info["float_needed"] = any(V[i][c["ix"]] != "1" and V[i][c["fix_ix"]] == "1" for i in refs)
        if c["ord_ix"] is not None and k == c["ix"] and order_ok:
            k = c["ord_ix"]
            info["fix"].append("any_order")
            info["order_needed"] = any(V[i][c["ix"]] != "1" and V[i][c["ord_ix"]] == "1" for i in refs)
        cur.append((k, info))
    n0 = len(cur)
    # 2. dependencies
    keep = []
    for k, info in cur:
        if "dep_unresolved" in info["fix"]:
            d["fixes"]["drop_dep_unresolved"] += 1
            continue
        keep.append((k, info))
    cur = keep
    if a["seq"]:
        s = a["seq"]
        rv = a["refs"][s["ref"]]["v"]
        # a test that no reference passes in isolation but the best one passes after the earlier tests ran
        leaky = {j for kk, j in enumerate(s["ix"]) if s["v"][kk] == "1" and rv[j] != "1"
                 and not any(V[i][j] == "1" for i in refs)}
        if leaky:
            orig_t = {T[j] for j in leaky}
            before = len(cur)
            cur = [(k, info) for k, info in cur if T[info["base"]] not in orig_t]
            d["fixes"]["drop_dep_state"] += before - len(cur)
            d["flags"]["state_dependency"] = True
    # 3. vacuous
    keep = []
    for k, info in cur:
        tr = tl.parse(T[k])
        if tr is not None and not tl.has_assert(tr):
            d["fixes"]["drop_noassert"] += 1
            d["flags"].setdefault("noassert_tests", []).append(T[k][:200])
            continue
        if all(S[s][k] == "1" for s in CONSTS):
            d["fixes"]["drop_vacuous"] += 1
            d["flags"].setdefault("vacuous_tests", []).append(T[k][:200])
            continue
        keep.append((k, info))
    cur = keep
    if not cur:
        d["drop"] = "no_tests_left_after_cleaning"
        return d, cur
    # 4. stub floor
    rates = {s: sum(S[s][k] == "1" for k, _ in cur) / len(cur) for s in CONSTS}
    best = max(rates.values())
    d["stub_floor_pre"] = best
    d["stub_best"] = max(rates, key=rates.get)
    if best >= 1.0:
        d["drop"] = "constant_passes_all"
        return d, cur
    stubpass = {k for k, _ in cur if any(S[s][k] == "1" for s in CONSTS)}
    if best >= 0.5:
        pruned = [(k, info) for k, info in cur if k not in stubpass]
        if len(pruned) < 3:
            d["drop"] = "stub_floor_ge50_unprunable"
            return d, cur
        d["fixes"]["prune_stub_tests"] += len(cur) - len(pruned)
        cur = pruned
    # 5. broken single test
    ks = [k for k, _ in cur]
    full = [i for i in refs if all(V[i][k] == "1" for k in ks)]
    d["n_refs_ok"] = len(refs)
    d["n_refs_full_pre"] = len(full)
    d["pending_probe"] = None
    if not full and refs:
        fails = {}
        for i in refs:
            f = [k for k in ks if V[i][k] != "1"]
            fails[i] = f
        single = collections.defaultdict(list)
        for i, f in fails.items():
            if len(f) == 1:
                single[f[0]].append(i)
        if single:
            k_blk, near = max(single.items(), key=lambda x: len(x[1]))
            distinct = {a["refs"][i]["h"] for i in near}
            if len(distinct) >= 2 and not any(V[i][k_blk] == "1" for i in refs):
                d["pending_probe"] = {"k": k_blk, "near": near}
        d["fail_pattern"] = {
            "best_frac": max(1 - len(f) / len(ks) for f in fails.values()),
            "n_block_tests_best": min(len(f) for f in fails.values()),
            "tests_no_ref_passes": sum(not any(V[i][k] == "1" for i in refs) for k in ks),
            "n_tests": len(ks),
        }
    d["n0"] = n0
    return d, cur


def probe_expr(test):
    tree = tl.parse(test)
    p = tl.eq_parts(tree)
    if p:
        return ast.unparse(p[0])
    if tree is not None and len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
        t = tree.body[0].test
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            return ast.unparse(t.left)
    return None


def signature_lines(a, row, cur, passing, override=None):
    """Missing interface names -> (sentence, details). Signatures come from the passing references."""
    prompt = row["prompt"]
    T = a["tests"]
    trees = [tl.parse(T[k]) for k, _ in cur]
    called, freeall = [], []
    for tr in trees:
        if tr is None:
            continue
        for f in tl.called_free(tr):
            if f not in called and not f.startswith("_v7_"):
                called.append(f)
        for f in tl.free_names(tr):
            if f not in freeall and not f.startswith("_v7_"):
                freeall.append(f)
    missing = [f for f in freeall if not word_in(f, prompt)]
    if not missing:
        return None
    codes = [a["ref_codes"][i] for i in passing] or [a["ref_codes"][i] for i in ok_refs(a)]
    parts, details = [], []
    for f in missing:
        sigs = collections.Counter()
        kind = None
        if override and f in override:
            sigs[("def", override[f][2])] += 1
            codes_f = []
        else:
            codes_f = codes
        for c in codes_f:
            fd = tl.ref_def(c, f)
            if fd is not None:
                sigs[("def", tl.render_args(fd.args))] += 1
                continue
            cd = tl.ref_class(c, f)
            if cd is not None:
                init = next((s for s in cd.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None)
                ia = tl.render_args(init.args, drop_self=True) if init is not None else ""
                sigs[("class", ia)] += 1
        if sigs:
            (kind, args), _ = sigs.most_common(1)[0]
        else:
            pos, kws, star = tl.call_shapes(trees, f)
            if f in called and len(pos) == 1 and not kws and not star:
                kind, args = "def", ", ".join(f"arg{i + 1}" for i in range(next(iter(pos))))
            else:
                kind, args = "name", None
        if kind == "def":
            parts.append(f"`def {f}({args})`")
        elif kind == "class":
            parts.append(f"`class {f}`" + (f" (constructor `{f}({args})`)" if args else ""))
        else:
            parts.append(f"`{f}`")
        details.append({"name": f, "kind": kind, "args": args, "from_refs": bool(sigs), "n_sig": sum(sigs.values()),
                        "sig_variants": len(sigs)})
    sentence = "Implement " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
    return sentence, details


def sig_stub(details, others):
    """Missing names get the derived signature (functions raise a sentinel once their arguments bind);
    names the prompt already states get permissive stand-ins, so only the new signature is under test."""
    src = ["class V7ArityOK(Exception):\n    pass\n"]
    for x in details:
        if x["kind"] == "def":
            src.append(f"def {x['name']}({x['args']}):\n    raise V7ArityOK()\n")
        elif x["kind"] == "class":
            src.append(f"class {x['name']}:\n    def __init__(self{', ' + x['args'] if x['args'] else ''}):\n        pass\n")
    for o in others:
        src.append(f"def {o}(*a, **k):\n    return None\n")
    return "".join(src)


_BIND = ("positional argument", "keyword argument", "takes", "missing", "unexpected")


def arity_verdict(out, details):
    names = [x["name"] for x in details]
    bad = []
    for r in out["res"]:
        m = r.get("msg") or ""
        if r.get("exc") == "TypeError" and any(n in m for n in names) and any(b in m for b in _BIND):
            bad.append(m)
        if r.get("exc") == "NameError" and any(f"'{n}'" in m for n in names):
            bad.append(m)
    return out["prog"] == "ok" and not bad, bad[:3]


def finish(job):
    a, row = job
    if "error" in a:
        return {"row": a["row"], "custom_id": a["custom_id"], "src": a["src"], "drop": "audit_error",
                "fixes": collections.Counter(), "evidence": {"error": a["error"]}}
    d, cur = plan(a, row)
    T = a["tests"]
    d["row"] = a["row"]
    d["custom_id"] = a["custom_id"]
    d["src"] = a["src"]
    if d["drop"]:
        return d
    refs = ok_refs(a)
    V = {i: a["refs"][i]["v"] for i in refs}
    # 5b. probe near-miss references on the blocking test
    pp = d.pop("pending_probe", None)
    if pp:
        k = pp["k"]
        ex = probe_expr(T[k])
        vals = collections.Counter()
        if ex:
            for i in pp["near"][:6]:
                out = sandbox.run(a["ref_codes"][i], [{"probe": ex}], timeout=5.0)
                r = out["res"][0]
                vals[r.get("val", "<" + str(r.get("exc")) + ">")] += 1
        consensus = len(vals) == 1 and not next(iter(vals)).startswith("<")
        ev = {"test": T[k], "n_near": len(pp["near"]), "distinct_near": len({a["refs"][i]["h"] for i in pp["near"]}),
              "kinds": sorted({a["refs"][i]["kind"] for i in pp["near"]}), "ref_values": dict(vals),
              "consensus": consensus, "n_refs": len(refs)}
        d["evidence"]["blocking"] = ev
        hand = HAND.get(a["custom_id"])
        ev["hand_verdict"] = hand
        if hand == "U":
            d["drop"] = "underspecified_hand_checked"
            return d
        if hand == "W":
            cur = [(kk, info) for kk, info in cur if kk != k]
            d["fixes"]["drop_broken_test"] += 1
    ks = [k for k, _ in cur]
    # 6. meaningful tests
    meaningful = [k for k in ks if not any(a["stubs"][s][k] == "1" for s in CONSTS)]
    if len(meaningful) < 3:
        d["drop"] = "lt3_meaningful_tests"
        d["n_meaningful"] = len(meaningful)
        return d
    passing = [i for i in refs if all(V[i][k] == "1" for k in ks)]
    d["verified"] = bool(passing)
    d["n_refs_full"] = len(passing)
    d["passing_kinds"] = sorted({a["refs"][i]["kind"] for i in passing})
    d["verify_ref"] = passing[0] if passing else None
    d["verify_suffix"] = ""
    override = None
    if not passing and a["row"] in RENAME:
        override = RENAME[a["row"]]
        (name, (ri, cand, args)), = override.items()
        okl = ok_refs(a)
        d["verified"] = True
        d["verified_by_rename"] = True
        d["n_refs_full"] = 1
        d["passing_kinds"] = ["renamed:" + a["refs"][okl[ri]]["kind"]]
        d["verify_ref"] = okl[ri]
        d["verify_suffix"] = f"\n\n{name} = {cand}\n"
    # 7. signature line
    sl = signature_lines(a, row, cur, passing, override)
    d["sig"] = None
    if sl:
        sentence, details = sl
        d["sig"] = {"sentence": sentence, "details": details}
        missing = {x["name"] for x in details}
        others = set()
        for k in ks:
            tr = tl.parse(T[k])
            if tr is not None:
                others |= {f for f in tl.free_names(tr) if not f.startswith("_v7_")}
        others -= missing
        if all(x["kind"] in ("def", "class") for x in details):
            out = sandbox.run(sig_stub(details, sorted(others)), [T[k] for k in ks], timeout=5.0)
            d["sig"]["stub_prog"] = out["prog"]
            d["sig"]["stub_exc"] = collections.Counter(r.get("exc") for r in out["res"]).most_common(3)
            d["sig"]["arity_ok"], d["sig"]["arity_bad"] = arity_verdict(out, details)
        else:
            d["sig"]["arity_ok"] = False
        # references that define exactly this interface and pass every test
        d["sig"]["verified"] = bool(d["verified"]) and all(x["from_refs"] for x in details) and d["sig"]["arity_ok"]
    # 8. flags
    pn = tl._norm(row["prompt"])
    fns = set()
    for k in ks:
        tr = tl.parse(T[k])
        if tr is not None:
            fns |= set(tl.called_free(tr))
    leak = [k for k in ks if tl.example_leak(T[k], pn, fns)]
    d["n_example_tests"] = len(leak)
    d["n_meaningful_heldout"] = len([k for k in meaningful if k not in leak])
    d["hidden_state"] = bool(HIDDEN_RE.search(row["prompt"]))
    d["multi_answer"] = a["prep"]["multi"]
    d["any_order"] = a["prep"]["any_order"]
    d["tests_final"] = [T[k] for k in ks]
    d["test_info"] = [info for _, info in cur]
    d["float_tests"] = sum("float_tol" in info["fix"] for _, info in cur)
    d["float_needed"] = sum(bool(info.get("float_needed")) for _, info in cur)
    d["order_tests"] = sum("any_order" in info["fix"] for _, info in cur)
    d["order_needed"] = sum(bool(info.get("order_needed")) for _, info in cur)
    d["stub_floor_post"] = max(sum(a["stubs"][s][k] == "1" for k in ks) / len(ks) for s in CONSTS)
    return d


if __name__ == "__main__":
    audit = pickle.load(open(os.environ.get("AUDIT", f"{WORK}/audit.pkl"), "rb"))
    rows = {r["row"]: r for r in pickle.load(open(f"{WORK}/rows.pkl", "rb"))}
    jobs = [(a, rows[a["row"]]) for a in audit]
    out = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 24) as ex:
        for i, d in enumerate(ex.map(finish, jobs, chunksize=16)):
            out.append(d)
    pickle.dump(out, open(os.environ.get("DECIDE", f"{WORK}/decide.pkl"), "wb"))
    c = collections.Counter()
    for d in out:
        c[(d["src"], "drop:" + str(d["drop"]))] += 1
        for k, v in d["fixes"].items():
            c[(d["src"], k)] += v
    for k, v in sorted(c.items()):
        print(k, v)
