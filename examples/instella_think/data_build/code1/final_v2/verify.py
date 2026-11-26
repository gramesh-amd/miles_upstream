"""Stage 4 (grader server, 16 concurrent requests): verify every group, and collect the checker evidence.

Per group (groups.pkl), with the training options (stdio: float_tol / case_insensitive; 5 s per test):
  verification: references in order (v6, OLMo by Dolci reward, accepted CF Python submissions, v6 near-miss; up to
      10) on all tests (own + merged + Open-R1 official), stop at the first that passes every test. If none does,
      the first reference that passes the winner's own tests keeps the official / merged tests it passes (the
      final_v1 rule for merged and uncapped tests); a failed official test is left out.
  checker rows only, on the final tests (checker = Open-R1 generated_checker):
      self      the label output as the contestant output (the jury answer must be accepted)
      refs      up to 3 more of our references and up to 3 accepted CF submissions: per test, `completed`
                (program exits 0 in time, via a checker of ours that accepts anything), `checker` (the verdict)
                and `exact` (stdio route, plain compare)
      negatives empty output; the most common label output on every test (constant); the label with one token
                changed (first integer +1, else first float +1, else YES<->NO, else a suffix); up to 3 wrong
                CF submissions (WRONG_ANSWER verdict)

    VERIFY_ROUTES=stdio python3 verify.py  ->  /tmp/v7v2/verify_stdio.pkl   (resumable; partial in verify_stdio.part.pkl)
    Checker groups (VERIFY_ROUTES=checker) run on lux instead: final_v2/lux/verify_checkers.py.
"""
import collections
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from v2common import WORK
import gclient2 as gc

MAX_REFS = 10


def perturb(out):
    s = str(out)
    m = re.search(r"(?<![\w.])-?\d+(?![\w.])", s)
    if m:
        return s[:m.start()] + str(int(m.group(0)) + 1) + s[m.end():], "int+1"
    m = re.search(r"(?<![\w.])-?\d+\.\d+(?![\w.])", s)
    if m:
        d = len(m.group(0).split(".")[1])
        return s[:m.start()] + f"{float(m.group(0)) + 1:.{d}f}" + s[m.end():], "float+1"
    m = re.search(r"\S+", s)
    if not m:
        return "0\n", "empty->0"
    flip = {"YES": "NO", "NO": "YES", "Yes": "No", "No": "Yes", "yes": "no", "no": "yes"}
    tok = m.group(0)
    return s[:m.start()] + flip.get(tok, tok + "x") + s[m.end():], ("yes/no" if tok in flip else "suffix")


def verify_tests(g, own, extra):
    route, opts, chk = g["route"], g["opts"], g["checker"]
    tried, own_pass = [], None
    for j, ref in enumerate(g["refs"][:MAX_REFS]):
        res = gc.run_route(route, ref["code"], own + extra, opts, chk, stop_on_fail=True)
        if res is None:
            return dict(verified=False, why="grader_unreachable")
        tried.append(ref["origin"])
        if all(res):
            return dict(verified=True, ref=j, tests=own + extra, n_extra_kept=len(extra), tried=tried)
        if own_pass is None and all(res[:len(own)]):
            own_pass = j
    if own_pass is None:
        return dict(verified=False, why="no_reference_passes", tried=tried)
    res = gc.run_route(route, g["refs"][own_pass]["code"], extra, opts, chk) if extra else []
    if res is None:
        return dict(verified=False, why="grader_unreachable")
    kept = [t for t, ok in zip(extra, res) if ok]
    return dict(verified=True, ref=own_pass, tests=own + kept, n_extra_kept=len(kept), n_extra_dropped=len(extra) - len(kept),
                tried=tried)


def checker_evidence(g, tests, ver_ref):
    chk = g["checker"]
    labels = [t["output"] for t in tests]
    ev = {}
    ev["self"] = gc.run_checker(gc.table_program(tests, labels), tests, chk)
    common = collections.Counter(" ".join(o.split()) for o in labels).most_common(1)[0][0]
    pert = [perturb(o) for o in labels]
    ev["neg_empty"] = gc.run_checker(gc.table_program(tests, [""] * len(tests)), tests, chk)
    ev["neg_constant"] = gc.run_checker(gc.table_program(tests, [common + "\n"] * len(tests)), tests, chk)
    ev["neg_constant_differs"] = [" ".join(o.split()) != common for o in labels]
    ev["neg_token"] = gc.run_checker(gc.table_program(tests, [p for p, _ in pert]), tests, chk)
    ev["neg_token_kind"] = collections.Counter(k for _, k in pert)
    ev["positives"] = []
    ours = [r for i, r in enumerate(g["refs"][:MAX_REFS]) if r["origin"] != "cf_ok" and i != ver_ref][:3]
    cfok = [r for i, r in enumerate(g["refs"]) if r["origin"] == "cf_ok" and i != ver_ref][:3]
    for r in ours + cfok:
        ev["positives"].append(dict(origin=r["origin"],
                                    completed=gc.run_checker(r["code"], tests, gc.COMPLETED_PROBE),
                                    checker=gc.run_checker(r["code"], tests, chk),
                                    exact=gc.run("stdio", r["code"], tests, {})))
    ev["wrong"] = []
    for code in g["cf_wa"][:3]:
        ev["wrong"].append(dict(checker=gc.run_checker(code["code"], tests, chk),
                                exact=gc.run("stdio", code["code"], tests, {})))
    return ev


def job(item):
    pid, g = item
    t0 = time.time()
    out = dict(pid=pid)
    if g["v1_dups"] and len(g["own_tests"]) + len(g["official"]) <= g["v1_dup_max_tests"]:
        out["skipped"] = "duplicate_of_final_v1_with_as_many_tests"
        return pid, out
    win_own = g["own_tests"][:g["n_win_own"]]
    extra = g["own_tests"][g["n_win_own"]:] + g["official"]
    v = verify_tests(g, win_own, extra)
    out["verify"] = v
    if v.get("verified") and g["route"] == "checker":
        out["evidence"] = checker_evidence(g, v["tests"], v["ref"])
    out["secs"] = round(time.time() - t0, 1)
    return pid, out


def main():
    G = pickle.load(open(f"{WORK}/groups.pkl", "rb"))
    routes = set(os.environ.get("VERIFY_ROUTES", "stdio,checker").split(","))
    tag = "_".join(sorted(routes))
    G = {p: g for p, g in G.items() if g["route"] in routes}
    part = f"{WORK}/verify_{tag}.part.pkl"
    res = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
    todo = sorted(((p, g) for p, g in G.items() if p not in res), key=lambda kv: kv[1]["route"] != "checker")
    if len(sys.argv) > 1:
        todo = todo[:int(sys.argv[1])]
    print("groups", len(G), "done", len(res), "todo", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        for i, (pid, o) in enumerate(ex.map(job, todo)):
            res[pid] = o
            if (i + 1) % 20 == 0:
                pickle.dump(res, open(part, "wb"))
                print(i + 1, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(res, open(part, "wb"))
    if len(res) == len(G):
        pickle.dump(res, open(f"{WORK}/verify_{tag}.pkl", "wb"))
    c = collections.Counter((G[p]["route"], o.get("skipped") or o["verify"].get("why") or "verified") for p, o in res.items())
    for k, n in sorted(c.items()):
        print(k, n)


if __name__ == "__main__":
    main()
