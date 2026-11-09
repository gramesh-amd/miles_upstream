"""Unit-test reward for code RL on Instella-MoE (--custom-rm-path code_reward.batched_reward).

Grading follows Instella-MoE's reward_model.compute_code_reward: strip the think block,
take the last fenced ```python block (or the whole answer if unfenced), and run it
against the row's tests. Test formats, all JSON in `label`:
    [{"input": ..., "output": ...}, ...]   stdin/stdout, decimal-aware line compare
    ["assert f(1) == 2", ...]              each assert appended to the program
    {"test": "def test_x(): ...", ...}     pytest-style functions, each called once
    {"test": "def check(c): ...", "entry_point": "f"}   HumanEval-style check(f)
    {"bench": ..., "route": ..., ...}      a benchmark row from prepare_code_benchmarks.py,
                                           graded by oe-eval's rules (see _score_bench)

Programs run on the code-exec service (code_exec_server.py) at CODE_EXEC_API_URL, never
on the training node. Set CODE_EXEC_API_URL=local only for offline debugging.

Training reward is the pass rate (CODE_PARTIAL_CREDIT=1) or all-or-nothing. Eval is
always all-or-nothing, i.e. pass@1. Like if_reward.py, the eval/train split is read from
metadata.split because miles calls a custom reward without its `evaluation` kwarg.

A 0 reward is ambiguous, so every non-model cause is counted and logged as a CODE_REWARD
line every CODE_LOG_EVERY samples:
    api_error       the service did not answer (down, overloaded, wrong token)
    no_tests        label had no parseable tests -- a data problem
    think_gate_fail no clean <think>...</think> (CODE_REQUIRE_THINK=1)
    no_code         nothing that looks like code after the think block
api_error must stay at 0; if it is not, the reward is measuring the network.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time

import requests

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("CODE_EXEC_API_URL", "").strip().rstrip("/")
_LOCAL = _API_URL.lower() == "local"
REWARD_SCALE = float(os.environ.get("CODE_REWARD_SCALE", "1.0"))
_PARTIAL_CREDIT = os.environ.get("CODE_PARTIAL_CREDIT", "1") == "1"
_REQUIRE_THINK = os.environ.get("CODE_REQUIRE_THINK", "1") == "1"
_TIMEOUT = float(os.environ.get("CODE_EXEC_TIMEOUT", "5"))
# Tests per sample. Dolci rows carry up to hundreds; past a few dozen the extra tests
# rarely change the verdict but multiply sandbox load.
_MAX_TESTS = int(os.environ.get("CODE_MAX_TESTS", "32"))
_CONCURRENCY = int(os.environ.get("CODE_REWARD_CONCURRENCY", "64"))
_LOG_EVERY = int(os.environ.get("CODE_LOG_EVERY", "200"))
_HTTP_RETRIES = int(os.environ.get("CODE_EXEC_API_RETRIES", "2"))

# Benchmark rows (prepare_code_benchmarks.py) can go to their own service, so their heavy
# tests never queue behind training rewards.
_BENCH_API_URL = os.environ.get("CODE_BENCH_API_URL", "").strip().rstrip("/") or _API_URL

_FENCE_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n?(.*?)```", re.DOTALL)
# oe-eval codex_humanevalplus / mbppplus ::olmo3:adapt answer_format_regex and answer_regexes.
_OLMES_FORMAT_RE = re.compile(r"(?s)Here is the completed function:\n*```(?:python\n)?\n*(.*?)\s*```")
_OLMES_ANSWER_RE = re.compile(r"(?s)```(?:python\n)?\n*(.*?)\s*```")
_SPECIAL_TOKEN_RE = re.compile(r"<｜[^｜]*｜>")
_TEST_FUNC_RE = re.compile(r"^def (test_\w+)\s*\(", re.MULTILINE)
_ASSERT_FUNC_RE = re.compile(r"assert\s+(\w+)\s*\(")


class _Stats:
    KEYS = ("api_error", "no_tests", "think_gate_fail", "no_code")

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.n = 0
        self.reward_sum = 0.0
        self.all_pass = 0
        self.counts = {k: 0 for k in self.KEYS}

    def note(self, key):
        with self.lock:
            self.counts[key] += 1

    def note_sample(self, raw, all_pass):
        with self.lock:
            self.n += 1
            self.reward_sum += raw
            self.all_pass += int(all_pass)
            if _LOG_EVERY and self.n % _LOG_EVERY == 0:
                self._dump_locked()

    def _dump_locked(self):
        c = " ".join(f"{k}={v}" for k, v in self.counts.items())
        logger.info("CODE_REWARD scored=%d mean_reward=%.4f all_pass=%.4f | %s",
                    self.n, self.reward_sum / max(1, self.n), self.all_pass / max(1, self.n), c)

    def dump(self):
        with self.lock:
            self._dump_locked()


STATS = _Stats()


def _has_clean_think(response: str) -> bool:
    return response.lstrip().startswith("<think>") and "</think>" in response


def _answer_of(response: str) -> str:
    if "</think>" in response:
        response = response.split("</think>", 1)[-1]
    return _SPECIAL_TOKEN_RE.sub("", response).strip()


def extract_code(answer: str) -> str:
    blocks = _FENCE_RE.findall(answer)
    return blocks[-1].strip() if blocks else answer.strip()


def _parse_tests(label):
    tests = label
    for _ in range(3):
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except (json.JSONDecodeError, TypeError):
                return None
        elif isinstance(tests, list) and len(tests) == 1 and isinstance(tests[0], str):
            try:
                inner = json.loads(tests[0])
            except (json.JSONDecodeError, TypeError):
                break
            if not isinstance(inner, (list, dict)):
                break
            tests = inner
        else:
            break
    return tests if isinstance(tests, (list, dict)) and tests else None


# --- execution ------------------------------------------------------------------
_session = None
_session_lock = threading.Lock()


def _http():
    global _session
    with _session_lock:
        if _session is None:
            s = requests.Session()
            a = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=max(8, _CONCURRENCY * 2))
            s.mount("http://", a)
            s.mount("https://", a)
            _session = s
    return _session


def _run_remote(program: str, tests: list, stdio: bool):
    """0/1 per test, or None if the service could not be reached."""
    path = "/test_program_stdio" if stdio else "/test_program"
    body = {"program": program, "tests": tests, "max_execution_time": _TIMEOUT}
    return _post(_API_URL, path, body, len(tests), min(300.0, _TIMEOUT * len(tests) + 30.0))


def _post(url: str, path: str, body: dict, n_tests: int, read_timeout: float):
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            r = _http().post(url + path, json=body, timeout=(3.0, read_timeout))
            if r.status_code == 200:
                res = r.json().get("results")
                if isinstance(res, list) and len(res) == n_tests:
                    return [1 if x in (1, True) else 0 for x in res]
                logger.warning("code-exec %s returned a malformed body: %r", path, r.text[:200])
                return None
            if 400 <= r.status_code < 500:
                logger.warning("code-exec %s HTTP %d (check CODE_EXEC_API_URL / token): %r",
                               path, r.status_code, r.text[:200])
                return None
        except (requests.RequestException, ValueError) as e:
            if attempt == _HTTP_RETRIES:
                logger.warning("code-exec %s failed after %d attempts: %s", path, attempt + 1, e)
                return None
        time.sleep(0.5 * (attempt + 1))
    return None


def _run_local(program: str, tests: list, stdio: bool):
    out = []
    for tc in tests:
        try:
            if stdio:
                r = subprocess.run([sys.executable, "-c", program], input=str(tc.get("input", "")),
                                   capture_output=True, text=True, timeout=_TIMEOUT)
                ok = r.returncode == 0 and _lines_equal(r.stdout, str(tc.get("output", "")))
            else:
                r = subprocess.run([sys.executable, "-c", f"{program}\n\n{tc}\n"],
                                   capture_output=True, text=True, timeout=_TIMEOUT)
                ok = r.returncode == 0
        except Exception:
            ok = False
        out.append(int(ok))
    return out


def _lines_equal(a: str, b: str) -> bool:
    from decimal import Decimal, InvalidOperation

    al = [x.strip() for x in a.strip().split("\n")]
    bl = [x.strip() for x in b.strip().split("\n")]
    if len(al) != len(bl):
        return False
    for x, y in zip(al, bl):
        if x == y:
            continue
        try:
            if [Decimal(t) for t in x.split()] != [Decimal(t) for t in y.split()]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


def _run(program, tests, stdio):
    if _LOCAL:
        return _run_local(program, tests, stdio)
    if not _API_URL:
        return None
    return _run_remote(program, tests, stdio)


def pass_rate(code: str, tests) -> float | None:
    """Fraction of tests passed, or None when the tests could not be run at all."""
    if isinstance(tests, dict):
        test_str = str(tests.get("test", "") or "")
        if not test_str:
            return 0.0
        funcs = [m.group(1) for m in _TEST_FUNC_RE.finditer(test_str)]
        if funcs:
            blocks = re.split(r"(?=^def test_\w+\s*\()", test_str, flags=re.MULTILINE)
            prelude = blocks[0] if not blocks[0].lstrip().startswith("def test_") else ""
            fn_blocks = [b for b in blocks if b.lstrip().startswith("def test_")]
            cases = [f"{prelude}\n{b}\n{_TEST_FUNC_RE.search(b).group(1)}()\n" for b in fn_blocks]
        else:
            entry = tests.get("entry_point") or ""
            cases = [f"{test_str}\ncheck({entry})\n"] if entry and "check(" in test_str else [test_str]
        res = _run(code, cases[:_MAX_TESTS], stdio=False)
        return None if res is None else sum(res) / len(cases)

    if isinstance(tests[0], dict):
        cases = [{"input": t.get("input", ""), "output": t.get("output", "")}
                 for t in tests if isinstance(t, dict)]
        if not cases:
            return 0.0
        capped = cases[:_MAX_TESTS]
        res = _run(code, capped, stdio=True)
        return None if res is None else sum(res) / len(capped)

    cases = [str(t) for t in tests][:_MAX_TESTS]
    res = _run(code, cases, stdio=False)
    return None if res is None else sum(res) / len(cases)


# --- benchmarks (oe-eval ::olmo3:adapt protocol) ------------------------------------
def _bench_answer(response: str) -> str:
    """oe-eval's r1_style output processor: drop everything up to the LAST </think>. With no
    </think> at all the whole response is searched, so there is no think gate here."""
    answer = re.sub(r"(?ms).*</think>", "", response)
    answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer)
    answer = re.sub(r"(?ms)</answer>\s*$", "", answer)
    return _SPECIAL_TOKEN_RE.sub("", answer)


def _bench_code(answer: str, style: str) -> str:
    if style == "lcb":
        # lcb_runner extract_code (OpenAIChat style): lines between the last two ``` lines
        lines = answer.split("\n")
        fences = [i for i, line in enumerate(lines) if "```" in line]
        return "\n".join(lines[fences[-2] + 1:fences[-1]]) if len(fences) >= 2 else ""
    # oe_eval.tasks.utils.extract_answer: last format match, else last plain fence
    found = _OLMES_FORMAT_RE.findall(answer)
    code = found[-1] if found else ""
    if code == "":
        found = _OLMES_ANSWER_RE.findall(answer)
        code = found[-1] if found else ""
    return code


def bench_pass(code: str, label: dict) -> bool | None:
    """All-tests verdict for a benchmark row, or None when the service could not answer."""
    timeout = float(label["timeout"])
    body = {"program": label.get("prefix", "") + code, "max_execution_time": timeout, "stop_on_fail": True}
    route = label["route"]
    if route == "script":
        # oe-eval exec()s HumanEval+/MBPP+ programs with fresh globals, so __name__ is not
        # "__main__" and a model's own `if __name__ == "__main__":` block never runs.
        path, n_tests = "/test_program", 1
        body.update(tests=[label["test"]], name="tmp_sol")
    else:
        path = "/test_program_call" if route == "call" else "/test_program_stdio"
        n_tests = int(label["n_tests"])
        body["tests_ref"] = label["tests_ref"]
        if route == "call":
            body["fn_name"] = label["fn_name"]
    if _LOCAL or not _BENCH_API_URL:
        return None
    res = _post(_BENCH_API_URL, path, body, n_tests, min(3600.0, (timeout + 1.0) * n_tests + 60.0))
    return None if res is None else all(res)


def _score_bench(response: str, label: dict) -> float:
    code = _bench_code(_bench_answer(response), label.get("extract", "olmes"))
    if not code.strip():
        STATS.note("no_code")
        STATS.note_sample(0.0, False)
        return 0.0
    ok = bench_pass(code, label)
    if ok is None:
        STATS.note("api_error")
        STATS.note_sample(0.0, False)
        return 0.0
    STATS.note_sample(float(ok), ok)
    return float(ok) * REWARD_SCALE


# --- scoring --------------------------------------------------------------------
def _is_eval(meta: dict, evaluation_kwarg: bool) -> bool:
    return evaluation_kwarg or str(meta.get("split", "")).lower() == "eval"


def _gated_pass_rate(response: str, tests, evaluation: bool) -> tuple[float | None, str | None]:
    """Pass rate of a non-benchmark response behind the reward's gates.

    Returns (rate, None) when the program ran, (0.0, reason) when a gate rejected it, and
    (None, "api_error") when the sandbox did not answer.
    """
    if _REQUIRE_THINK:
        ok = ("</think>" in response) if evaluation else _has_clean_think(response)
        if not ok:
            return 0.0, "think_gate_fail"
    code = extract_code(_answer_of(response))
    if not code:
        return 0.0, "no_code"
    rate = pass_rate(code, tests)
    if rate is None:
        return None, "api_error"
    return rate, None


def train_all_pass(response: str, label) -> tuple[bool | None, str | None]:
    """Whether a training response passes every test, under the training reward's gates.

    Used by difficulty_filter.py to count solves. (None, "api_error") means the sandbox did
    not answer, which a caller must retry rather than count as a failure; (False, "no_tests")
    means the row itself is ungradable.
    """
    tests = _parse_tests(label)
    if tests is None:
        return False, "no_tests"
    rate, reason = _gated_pass_rate(response or "", tests, evaluation=False)
    if rate is None:
        return None, reason
    return rate >= 1.0 - 1e-9, reason


def _score_one(sample, evaluation: bool = False) -> float:
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    response = getattr(sample, "response", None) or ""
    evaluation = _is_eval(meta, evaluation)

    tests = _parse_tests(getattr(sample, "label", None))
    if tests is None:
        STATS.note("no_tests")
        STATS.note_sample(0.0, False)
        return 0.0
    if isinstance(tests, dict) and "bench" in tests:
        return _score_bench(response, tests)

    rate, reason = _gated_pass_rate(response, tests, evaluation)
    if reason is not None:
        STATS.note(reason)
        STATS.note_sample(0.0, False)
        return 0.0

    all_pass = rate >= 1.0 - 1e-9
    raw = rate if (_PARTIAL_CREDIT and not evaluation) else float(all_pass)
    STATS.note_sample(raw, all_pass)
    return raw * REWARD_SCALE


# Grading blocks a thread for the whole sandbox round trip (tens of seconds for a LiveCodeBench
# sample). asyncio.to_thread would share the loop's default pool, at most 32 threads, so miles'
# per-sample calls could never grade more than 32 samples at once whatever CODE_REWARD_CONCURRENCY
# says; evals then starve the engines. A pool of its own makes CODE_REWARD_CONCURRENCY the cap.
_EXECUTOR = None


def _executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max(1, _CONCURRENCY), thread_name_prefix="code-reward")
    return _EXECUTOR


async def batched_reward(args, samples, **kwargs):
    """Entry point for both call shapes miles uses (single Sample and list[Sample])."""
    evaluation = bool(kwargs.get("evaluation", False))
    loop = asyncio.get_running_loop()
    if not isinstance(samples, (list, tuple)):
        return await loop.run_in_executor(_executor(), _score_one, samples, evaluation)
    if not samples:
        return []
    sem = asyncio.Semaphore(max(1, _CONCURRENCY))

    async def _one(s):
        async with sem:
            return await loop.run_in_executor(_executor(), _score_one, s, evaluation)

    return await asyncio.gather(*[_one(s) for s in samples])


# --- self-test ------------------------------------------------------------------
def health() -> dict | None:
    if _LOCAL:
        return {"ok": True, "mode": "local"}
    if not _API_URL:
        return None
    try:
        r = _http().get(_API_URL + "/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


if __name__ == "__main__":
    # python3 examples/instella_think/code_reward.py   (needs CODE_EXEC_API_URL)
    class _S:
        def __init__(self, response, label, metadata=None):
            self.response, self.label, self.metadata = response, label, metadata or {}

    logging.basicConfig(level=logging.INFO)
    h = health()
    if h is None:
        raise SystemExit(f"code-exec service not reachable at CODE_EXEC_API_URL={_API_URL or '(unset)'}")
    print(f"service: {h}")

    stdio = json.dumps([{"input": "2 3\n", "output": "5\n"}, {"input": "1 1\n", "output": "2\n"},
                        {"input": "4 4\n", "output": "8\n"}, {"input": "0 7\n", "output": "7\n"}])
    good = "<think>add them</think>\n```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    half = "<think>r</think>\n```python\na, b = map(int, input().split())\nprint(a + b if a != b else 0)\n```"
    asserts = json.dumps(["assert double(2) == 4", "assert double(0) == 0"])
    pytest_style = json.dumps({"test": "def test_a():\n    assert inc(1) == 2\n\ndef test_b():\n    assert inc(5) == 7\n"})
    part = (lambda r: r) if _PARTIAL_CREDIT else (lambda r: 0.0)  # all-or-nothing scores a partial pass 0
    cases = [
        (good, stdio, {}, 1.0, "stdio all pass"),
        (half, stdio, {}, part(0.5), "stdio half pass, partial credit"),
        (half, stdio, {"split": "eval"}, 0.0, "eval is all-or-nothing"),
        ("```python\nprint(5)\n```", stdio, {}, 0.0 if _REQUIRE_THINK else part(0.25), "think gate"),
        ("<think>r</think>no code here", stdio, {}, 0.0, "prose is not a program"),
        ("<think>r</think>```python\ndef double(x): return 2*x\n```", asserts, {}, 1.0, "assert list"),
        ("<think>r</think>```python\ndef inc(x): return x+1\n```", pytest_style, {}, part(0.5), "pytest functions"),
        ("<think>r</think>```python\nwhile True: pass\n```", stdio, {}, 0.0, "infinite loop"),
        (good, json.dumps([stdio]), {}, 1.0, "Dolci shape: list holding one JSON string"),
    ]
    ok = True
    for resp, label, meta, want, why in cases:
        got = asyncio.run(batched_reward(None, [_S(resp, label, meta)]))[0]
        flag = "ok " if abs(got - want * REWARD_SCALE) < 1e-6 else "FAIL"
        ok &= flag == "ok "
        print(f"  {flag} reward={got:.3f} want={want * REWARD_SCALE:.3f}  ({why})")
    STATS.dump()
    if STATS.counts["api_error"]:
        print("  FAIL api_error > 0: the service dropped requests")
        ok = False
    raise SystemExit(0 if ok else 1)
