"""final_v1's gclient plus the checker route (/test_program_checker, open-r1 / piston Codeforces judge) and
lookup-table programs that print a fixed output per test (used to feed known outputs to a checker).
Every program and every checker runs only through the grader server."""
import json

import gclient
from gclient import TIMEOUT, _chunks, _post, run, stdio_opts  # noqa: F401

# A checker of our own (no dataset code): accepts whatever the program printed, so on the checker route the
# verdict is 1 iff the program exited 0 within the time limit.
COMPLETED_PROBE = "print(1)\n"


def run_checker(program, tests, checker, stop_on_fail=False):
    """0/1 per test on /test_program_checker (None if the server could not be reached)."""
    out = []
    for chunk in _chunks(tests):
        body = {"program": program, "tests": [{"input": str(t["input"]), "output": str(t["output"])} for t in chunk],
                "max_execution_time": TIMEOUT, "stop_on_fail": stop_on_fail}
        if checker:
            body["checker"] = checker
        res = _post("/test_program_checker", body, len(chunk))
        if res is None:
            return None
        out += res
        if stop_on_fail and not all(res):
            out += [0] * (len(tests) - len(out))
            break
    return out


def run_route(route, program, tests, opts=None, checker=None, stop_on_fail=False):
    if route == "checker":
        return run_checker(program, tests, checker, stop_on_fail)
    return run("stdio", program, tests, opts or {}, stop_on_fail)


def table_program(tests, outputs):
    """A program printing outputs[i] when its stdin is tests[i]['input'] (keyed by the whitespace-normalised input)."""
    import hashlib
    table = {}
    for t, o in zip(tests, outputs):
        k = hashlib.sha1(" ".join(str(t["input"]).split()).encode("utf-8", "replace")).hexdigest()
        table[k] = o
    return ("import sys, hashlib\nT = " + json.dumps(table) + "\n"
            "k = hashlib.sha1(' '.join(sys.stdin.read().split()).encode('utf-8', 'replace')).hexdigest()\n"
            "sys.stdout.write(T.get(k, ''))\n")


__all__ = ["gclient", "run", "run_checker", "run_route", "table_program", "stdio_opts", "COMPLETED_PROBE"]
