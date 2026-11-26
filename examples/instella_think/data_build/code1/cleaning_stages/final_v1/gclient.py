"""Client for the sandboxed grader server (code_exec_server.py). All untrusted code in this build (reference
solutions, stubs, candidates) runs only through it, with the training grader's options:
assert route strict=True, name="tmp_sol" (code_reward.train_options with CODE_STRICT_GRADING=1);
stdio route float_tol / case_insensitive from the row metadata; 5 s per test (CODE_EXEC_TIMEOUT).

URL: $V7_GRADER_URL (default http://127.0.0.1:18987/testtoken). The token is never printed.
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("V7_GRADER_URL", "http://127.0.0.1:18987/testtoken")
TIMEOUT = 5.0
CHUNK = 32  # server --max-tests is 64; training reads the first 32 (CODE_MAX_TESTS)


def _post(path, body, n):
    data = json.dumps(body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT * n + 120) as r:
                res = json.loads(r.read())["results"]
            if len(res) == n:
                return res
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


MAX_BYTES = 16 << 20  # server rejects bodies over 64 MB


def _chunks(tests):
    cur, size = [], 0
    for t in tests:
        n = len(str(t)) if isinstance(t, str) else len(str(t.get("input", ""))) + len(str(t.get("output", "")))
        if cur and (len(cur) >= CHUNK or size + n > MAX_BYTES):
            yield cur
            cur, size = [], 0
        cur.append(t)
        size += n
    if cur:
        yield cur


def run(kind, program, tests, opts=None, stop_on_fail=False):
    """0/1 per test (None if the server could not be reached). kind: 'assert' or 'stdio'."""
    opts = opts or {}
    out = []
    for chunk in _chunks(tests):
        if kind == "assert":
            body = {"program": program, "tests": [str(t) for t in chunk], "max_execution_time": TIMEOUT,
                    "strict": True, "name": "tmp_sol", "stop_on_fail": stop_on_fail}
            path = "/test_program"
        else:
            body = {"program": program, "tests": [{"input": str(t["input"]), "output": str(t["output"])} for t in chunk],
                    "max_execution_time": TIMEOUT, "stop_on_fail": stop_on_fail}
            if opts.get("float_tol") is not None:
                body["float_tol"] = float(opts["float_tol"])
            if opts.get("case_insensitive"):
                body["case_insensitive"] = True
            path = "/test_program_stdio"
        res = _post(path, body, len(chunk))
        if res is None:
            return None
        out += res
        if stop_on_fail and not all(res):
            out += [0] * (len(tests) - len(out))
            break
    return out


def stdio_opts(meta):
    return {k: meta[k] for k in ("float_tol", "case_insensitive") if meta.get(k) is not None}
