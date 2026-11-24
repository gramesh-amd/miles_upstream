"""Offline tests for eval_code_ckpt.py's grading wrapper and summary; no server needed.

    python3 test_eval_code_ckpt.py

The sandbox side is checked by regrading a finished run: --regrade-from <run>/gen.jsonl.gz into
another --out, then compare its graded.jsonl.gz with the run's per (ds, pi, k).
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_reward  # noqa: E402
import eval_code_ckpt as ev  # noqa: E402


def check(name, got, want):
    ok = got == want if not isinstance(want, float) else got is not None and abs(got - want) < 1e-9
    print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" want {want!r}"))
    return ok


def brute_pass_at_k(n, c, k):
    """Probability that k of n samples drawn without replacement include a correct one."""
    draws = list(itertools.combinations([1] * c + [0] * (n - c), k))
    return sum(any(d) for d in draws) / len(draws)


def rec(ds, pi, k, correct, ntok=100, truncated=False, reason=None):
    return {"ds": ds, "pi": pi, "k": k, "correct": correct, "ntok": ntok, "truncated": truncated,
            "reason": reason, "think_closed": True, "multi_think_close": False, "route": "stdio"}


def test_pass_at_k():
    return all(check(f"pass@{k} n={n} c={c}", ev.pass_at_k(n, c, k), brute_pass_at_k(n, c, k))
               for n in (1, 4, 10) for c in range(n + 1) for k in sorted({1, 2, n}) if k <= n)


def test_summary():
    # lcb_v5: 3 problems solved 4/4, 2/4, 0/4; lcb_v6: 1 problem solved 1/4
    rs = [rec("lcb_v5", pi, k, k < c) for pi, c in enumerate((4, 2, 0)) for k in range(4)]
    rs += [rec("lcb_v6", 0, k, k < 1, truncated=k == 3) for k in range(4)]
    s = ev.summarize(rs, 4)
    ok = check("lcb_v5 pass@1", s["lcb_v5"]["pass@1"], 0.5)
    ok &= check("lcb_v5 pass@4", s["lcb_v5"]["pass@4"], round(2 / 3, 4))
    ok &= check("lcb_v5 solved_all / solved_none", (s["lcb_v5"]["solved_all"], s["lcb_v5"]["solved_none"]),
                (round(1 / 3, 4), round(1 / 3, 4)))
    ok &= check("lcb_v6 truncated", s["lcb_v6"]["truncated"], 0.25)
    ok &= check("lcb_v6 acc_when_finished", s["lcb_v6"]["acc_when_finished"], round(1 / 3, 4))
    # weighted by problem count (3 and 1), not the mean of the two sets' pass@1 (0.375)
    ok &= check("lcb_v5v6", s["lcb_v5v6"], round((0.5 * 3 + 0.25 * 1) / 4, 4))
    ok &= check("no lcb_v5v6 without both sets", "lcb_v5v6" in ev.summarize(rs[:12], 4), False)
    return ok


def test_grade():
    label = '{"bench": "lcb_v6", "route": "stdio", "extract": "lcb", "timeout": 6, "n_tests": 1, "tests_ref": "x"}'
    answer = "<think>\nplan\n</think>\n```python\nprint(1)\n```\n"
    real, sleep = code_reward.bench_pass, ev.time.sleep
    calls = []
    ev.time.sleep = lambda s: None
    try:
        code_reward.bench_pass = lambda code, lab: calls.append(code) or True
        r = ev.grade(code_reward, {"response": answer}, label)
        ok = check("graded correct", (r["correct"], r["reason"], "response" in r), (True, None, False))
        ok &= check("code sent", calls, ["print(1)"])
        calls.clear()
        code_reward.bench_pass = lambda code, lab: calls.append(code)
        r = ev.grade(code_reward, {"response": answer}, label)
        ok &= check("sandbox down: 3 tries, then api_error", (len(calls), r["correct"], r["reason"]),
                    (3, False, "api_error"))
        r = ev.grade(code_reward, {"response": "<think>x</think> no fence"}, label)
        ok &= check("no fenced code: no_code", (r["correct"], r["reason"]), (False, "no_code"))
        r = ev.grade(code_reward, {"response": "a</think>b</think>\n```\nx\n```\n```\nprint(2)\n```"}, label)
        ok &= check("multi </think> flagged", r["multi_think_close"], True)
    finally:
        code_reward.bench_pass, ev.time.sleep = real, sleep
    return ok


def main():
    ok = True
    for t in (test_pass_at_k, test_summary, test_grade):
        print(t.__name__)
        ok &= t()
    print("ALL OK" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
