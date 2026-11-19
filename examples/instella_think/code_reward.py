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

Checker rows (problems with several correct outputs): the stdin/stdout list plus a Python
checker, either as metadata.checker = "<python source>" next to the usual label (what the v7
data uses) or self-contained as label = {"tests": [{"input", "output"}, ...], "checker": "..."}.
They go to the service's /test_program_checker route (open-r1's Codeforces judge): the checker
runs sandboxed as `checker.py input.txt correct_output.txt solution_output.txt` and a test
passes iff it exits 0 printing 1 (or 100). A checker that crashes or prints anything else fails
the test. Reward mode, think gate, CODE_MAX_TESTS, CODE_EXEC_TIMEOUT and the hack monitor are
the same as for stdin/stdout rows; float_tol / case_insensitive do not apply (the checker decides).

Programs run on the code-exec service (code_exec_server.py) at CODE_EXEC_API_URL, never
on the training node. Set CODE_EXEC_API_URL=local only for offline debugging.

Training reward, CODE_REWARD_MODE:
    binary   (default) 1 iff pass rate >= CODE_REWARD_PASS_THRESHOLD, else 0. The default
             threshold 1.0 means every test passes; 0.99 only differs from it when
             CODE_MAX_TESTS >= 100 (at 32 tests the best non-perfect rate is 31/32 = 0.969).
    partial  the pass rate itself. The legacy CODE_PARTIAL_CREDIT=1/0 selects partial/binary
             when CODE_REWARD_MODE is unset.
Eval is always all-or-nothing at 1.0, i.e. pass@1, whatever the mode. Like if_reward.py, the
eval/train split is read from metadata.split because miles calls a custom reward without its
`evaluation` kwarg. The think gate and the other 0-reward cases below apply in both modes.

The raw pass rate of every graded sample is kept in sample.metadata["code_pass_rate"] and,
through log_rollout_data, logged as rollout/code/pass_rate_mean and rollout/code/all_pass_rate.

A 0 reward is ambiguous, so every non-model cause is counted and logged as a CODE_REWARD
line every CODE_LOG_EVERY samples:
    api_error       the service did not answer (down, overloaded, wrong token)
    no_tests        label had no parseable tests -- a data problem
    think_gate_fail no clean <think>...</think> (CODE_REQUIRE_THINK=1)
    no_code         nothing that looks like code after the think block
api_error must stay at 0; if it is not, the reward is measuring the network.

Reward-hack monitoring (never changes a reward): every graded sample gets
sample.metadata["code_hack"], booleans from the server's per-request flags and a static scan
of the code (hack_flags), and training samples are also kept in WINDOW (CODE_HACK_WINDOW_MAX). With
--custom-rollout-log-function-path code_reward.log_rollout_data, both are logged per rollout
step as rollout/code_hack/* rates. `solved` there means raw reward >= 0.99, so it is the
all-pass rate under the default binary reward and the near-perfect rate under partial.
"""

import ast
import asyncio
import collections
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from decimal import Decimal, InvalidOperation

import requests

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("CODE_EXEC_API_URL", "").strip().rstrip("/")
_LOCAL = _API_URL.lower() == "local"
REWARD_SCALE = float(os.environ.get("CODE_REWARD_SCALE", "1.0"))
REWARD_MODES = ("binary", "partial")


def _reward_mode() -> str:
    mode = os.environ.get("CODE_REWARD_MODE", "").strip().lower()
    legacy = os.environ.get("CODE_PARTIAL_CREDIT", "").strip()
    if legacy not in ("", "0", "1"):
        raise ValueError(f"CODE_PARTIAL_CREDIT must be 0 or 1, got {legacy!r}")
    legacy_mode = {"1": "partial", "0": "binary"}.get(legacy)
    if mode and mode not in REWARD_MODES:
        raise ValueError(f"CODE_REWARD_MODE must be one of {REWARD_MODES}, got {mode!r}")
    if mode and legacy_mode and mode != legacy_mode:
        raise ValueError(f"CODE_REWARD_MODE={mode} contradicts CODE_PARTIAL_CREDIT={legacy}; unset one")
    return mode or legacy_mode or "binary"


REWARD_MODE = _reward_mode()
PASS_THRESHOLD = float(os.environ.get("CODE_REWARD_PASS_THRESHOLD", "1.0"))
if not 0.0 < PASS_THRESHOLD <= 1.0:
    raise ValueError(f"CODE_REWARD_PASS_THRESHOLD must be in (0, 1], got {PASS_THRESHOLD}")
_REQUIRE_THINK = os.environ.get("CODE_REQUIRE_THINK", "1") == "1"
_TIMEOUT = float(os.environ.get("CODE_EXEC_TIMEOUT", "5"))
# Tests per sample. Dolci rows carry up to hundreds; past a few dozen the extra tests
# rarely change the verdict but multiply sandbox load.
_MAX_TESTS = int(os.environ.get("CODE_MAX_TESTS", "32"))
_CONCURRENCY = int(os.environ.get("CODE_REWARD_CONCURRENCY", "64"))
_LOG_EVERY = int(os.environ.get("CODE_LOG_EVERY", "200"))
_HTTP_RETRIES = int(os.environ.get("CODE_EXEC_API_RETRIES", "2"))
_STRICT = os.environ.get("CODE_STRICT_GRADING", "1") == "1"
# Training samples kept for the next rollout log; bounds memory when the log hook is off.
_WINDOW_MAX = int(os.environ.get("CODE_HACK_WINDOW_MAX", "100000"))

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
        self.pass_rate_sum = 0.0
        self.all_pass = 0
        self.counts = {k: 0 for k in self.KEYS}

    def note(self, key):
        with self.lock:
            self.counts[key] += 1

    def note_sample(self, raw, all_pass, rate=0.0):
        with self.lock:
            self.n += 1
            self.reward_sum += raw
            self.pass_rate_sum += rate
            self.all_pass += int(all_pass)
            if _LOG_EVERY and self.n % _LOG_EVERY == 0:
                self._dump_locked()

    def _dump_locked(self):
        c = " ".join(f"{k}={v}" for k, v in self.counts.items())
        n = max(1, self.n)
        logger.info("CODE_REWARD mode=%s scored=%d mean_reward=%.4f mean_pass_rate=%.4f all_pass=%.4f | %s",
                    REWARD_MODE, self.n, self.reward_sum / n, self.pass_rate_sum / n, self.all_pass / n, c)

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
_STDIO_OPTS = ("float_tol", "case_insensitive")
_ASSERT_OPTS = ("strict", "name")
_CHECKER_TIMEOUT = 10.0  # must equal code_exec_server.CHECKER_TIMEOUT
_INT_RE = re.compile(r"[+-]?\d+")
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


def _run_remote(program: str, tests: list, stdio: bool, opts=None, flags=None):
    """0/1 per test, or None if the service could not be reached. The server's per-test
    outcome counts (early_exit, strict_reject, timeout) are added into `flags`."""
    checker = (opts or {}).get("checker")
    if checker:
        body = {"program": program, "tests": tests, "max_execution_time": _TIMEOUT, "checker": checker}
        read_timeout = min(300.0, (_TIMEOUT + _CHECKER_TIMEOUT) * len(tests) + 30.0)
        return _post(_API_URL, "/test_program_checker", body, len(tests), read_timeout, flags)
    path = "/test_program_stdio" if stdio else "/test_program"
    body = {"program": program, "tests": tests, "max_execution_time": _TIMEOUT}
    if opts:
        body.update({k: v for k, v in opts.items() if k in (_STDIO_OPTS if stdio else _ASSERT_OPTS)})
    return _post(_API_URL, path, body, len(tests), min(300.0, _TIMEOUT * len(tests) + 30.0), flags)


def _post(url: str, path: str, body: dict, n_tests: int, read_timeout: float, flags=None):
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            r = _http().post(url + path, json=body, timeout=(3.0, read_timeout))
            if r.status_code == 200:
                got = r.json()
                res = got.get("results")
                if isinstance(res, list) and len(res) == n_tests:
                    if flags is not None and isinstance(got.get("flags"), dict):
                        for k, v in got["flags"].items():
                            if isinstance(v, int):
                                flags[k] = flags.get(k, 0) + v
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


def _checker_local(checker: str, tc: dict, out: str) -> bool:
    import tempfile

    def no_blank(s):
        return "".join(ln + "\n" for ln in s.replace("\r", "").split("\n") if ln != "")

    with tempfile.TemporaryDirectory() as d:
        files = {"checker.py": checker, "input.txt": str(tc.get("input", "")).replace("\r", ""),
                 "correct_output.txt": no_blank(str(tc.get("output", ""))), "solution_output.txt": no_blank(out)}
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        r = subprocess.run([sys.executable, "checker.py", "input.txt", "correct_output.txt", "solution_output.txt"],
                           cwd=d, capture_output=True, text=True, timeout=_CHECKER_TIMEOUT)
    return r.returncode == 0 and "".join(r.stdout.split()) in ("1", "100")


def _run_local(program: str, tests: list, stdio: bool, opts=None):
    checker = (opts or {}).get("checker")
    out = []
    for tc in tests:
        try:
            if checker:
                r = subprocess.run([sys.executable, "-c", program], input=str(tc.get("input", "")),
                                   capture_output=True, text=True, timeout=_TIMEOUT)
                ok = r.returncode == 0 and _checker_local(checker, tc, r.stdout)
            elif stdio:
                r = subprocess.run([sys.executable, "-c", program], input=str(tc.get("input", "")),
                                   capture_output=True, text=True, timeout=_TIMEOUT)
                cmp = {k: v for k, v in (opts or {}).items() if k in _STDIO_OPTS}
                ok = r.returncode == 0 and _lines_equal(r.stdout, str(tc.get("output", "")), **cmp)
            else:
                r = subprocess.run([sys.executable, "-c", f"{program}\n\n{tc}\n"],
                                   capture_output=True, text=True, timeout=_TIMEOUT)
                ok = r.returncode == 0
        except Exception:
            ok = False
        out.append(int(ok))
    return out


def _tokens_equal(x: str, y: str, float_tol, case_insensitive: bool) -> bool:
    if x == y or (case_insensitive and x.lower() == y.lower() and y.lower() in ("yes", "no")):
        return True
    try:
        dx, dy = Decimal(x), Decimal(y)
        if dx == dy:
            return True
    except (InvalidOperation, ValueError):
        return False
    if float_tol is None or not (dx.is_finite() and dy.is_finite()):
        return False
    if _INT_RE.fullmatch(x) and _INT_RE.fullmatch(y):
        return False
    return abs(dx - dy) <= Decimal(str(float_tol)) * max(Decimal(1), abs(dy))


def _lines_equal(a: str, b: str, float_tol=None, case_insensitive=False) -> bool:
    """Mirror of code_exec_server._lines_equal_with_decimal (that module need not be importable
    here); test_code_hack_monitor.py checks that the two agree."""
    al = [x.strip() for x in a.strip().split("\n")]
    bl = [x.strip() for x in b.strip().split("\n")]
    if len(al) != len(bl):
        return False
    for x, y in zip(al, bl):
        if x == y:
            continue
        xt, yt = x.split(), y.split()
        if float_tol is None and not case_insensitive:
            try:
                if [Decimal(t) for t in xt] != [Decimal(t) for t in yt]:
                    return False
            except (InvalidOperation, ValueError):
                return False
            continue
        if len(xt) != len(yt) or not all(_tokens_equal(p, q, float_tol, case_insensitive) for p, q in zip(xt, yt)):
            return False
    return True


def _run(program, tests, stdio, opts=None, flags=None):
    if _LOCAL:
        return _run_local(program, tests, stdio, opts)
    if not _API_URL:
        return None
    return _run_remote(program, tests, stdio, opts, flags)



def split_checker(tests, meta: dict):
    """(tests, checker source or None) for a parsed label; (None, None) when a row names a checker
    that cannot be used (not a non-empty string, or tests that are not stdin/stdout cases)."""
    checker = None
    if isinstance(tests, dict) and "checker" in tests:
        checker, tests = tests.get("checker"), tests.get("tests")
    elif meta.get("checker") is not None:
        checker = meta["checker"]
    if checker is None:
        return tests, None
    if not (isinstance(checker, str) and checker.strip() and isinstance(tests, list) and tests
            and all(isinstance(t, dict) for t in tests)):
        return None, None
    return tests, checker


def train_options(meta: dict) -> dict:
    """Grader options for training rows; evals keep the benchmarks' own semantics.
    stdio: per-row comparison options set by the data pipeline when the statement allows
    them (metadata.float_tol, metadata.case_insensitive; YES/NO folding only).
    asserts (CODE_STRICT_GRADING=1): every value an assert compares or truth-tests is reduced to
    builtin values (code_exec_server.strict_assert), so objects with rigged dunders fail; and the
    program is not run as __main__, so a `if __name__ == "__main__":` block neither runs nor can
    exit before the tests."""
    opts = {"strict": True, "name": "tmp_sol"} if _STRICT else {}
    if meta.get("float_tol") is not None:
        opts["float_tol"] = float(meta["float_tol"])
    if meta.get("case_insensitive"):
        opts["case_insensitive"] = True
    return opts


def pass_rate(code: str, tests, opts=None, flags=None) -> float | None:
    """Fraction of tests passed, or None when the tests could not be run at all.
    `opts` are grader options (train_options, or {"checker": source} for checker rows, which
    sends stdin/stdout tests to /test_program_checker); `flags` collects the server's outcome counts."""
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
        capped = cases[:_MAX_TESTS]
        res = _run(code, capped, stdio=False, opts=opts, flags=flags)
        return None if res is None else sum(res) / len(capped)

    if isinstance(tests[0], dict):
        cases = [{"input": t.get("input", ""), "output": t.get("output", "")}
                 for t in tests if isinstance(t, dict)]
        if not cases:
            return 0.0
        capped = cases[:_MAX_TESTS]
        res = _run(code, capped, stdio=True, opts=opts, flags=flags)
        return None if res is None else sum(res) / len(capped)

    cases = [str(t) for t in tests][:_MAX_TESTS]
    res = _run(code, cases, stdio=False, opts=opts, flags=flags)
    return None if res is None else sum(res) / len(cases)


# --- benchmarks (oe-eval ::olmo3:adapt protocol) ------------------------------------
def _bench_answer(response: str) -> str:
    """oe-eval's r1_style output processor: drop everything up to the LAST </think>. With no
    </think> at all the whole response is searched, so there is no think gate here."""
    answer = re.sub(r"(?ms).*</think>", "", response)
    answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer)
    answer = re.sub(r"(?ms)</answer>\s*$", "", answer)
    return _SPECIAL_TOKEN_RE.sub("", answer)


_CRUX_ANSWER_RE = re.compile(r"(?s)\[ANSWER\](.*?)\[/ANSWER\]")
# Calls a CRUXEval-O literal may contain; anything else (f(...), sys.exit(0), ...) is not a
# literal and would let the prediction run code ahead of the assert.
_CRUX_LITERAL_CALLS = {"set", "frozenset", "tuple", "list", "dict", "str", "int", "float", "bool",
                       "bytes", "complex", "range"}


def _crux_assert(text: str):
    """The last `assert ...` statement (or bare expression) in text, as (module, node)."""
    text = re.sub(r"(?m)^\s*```(?:python|py)?\s*$", "", text).strip()
    for cand in (text, *reversed([ln.strip() for ln in text.splitlines() if ln.strip()])):
        try:
            tree = ast.parse(cand)
        except (SyntaxError, ValueError):
            continue
        stmts = [s for s in tree.body if isinstance(s, (ast.Assert, ast.Expr))]
        if stmts:
            return cand, stmts[-1]
    return None, None


def _crux_pred(answer: str, mode: str) -> str:
    """CRUXEval: the predicted output literal (mode "o") or the f(...) call (mode "i") from
    the last [ANSWER]...[/ANSWER] (else the last `assert` line), as `__pred = (...)`.
    cruxeval itself splits the assertion on "=="; parsing it keeps "==" inside strings intact."""
    found = _CRUX_ANSWER_RE.findall(answer)
    if found:
        text = found[-1]
    else:
        lines = [ln for ln in answer.splitlines() if ln.strip().startswith("assert")]
        if not lines:
            return ""
        text = lines[-1]
    src, node = _crux_assert(text)
    if node is None:
        return ""
    expr = node.test if isinstance(node, ast.Assert) else node.value
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Eq):
        expr = expr.comparators[0] if mode == "o" else expr.left
    if mode == "o":
        for n in ast.walk(expr):
            if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in _CRUX_LITERAL_CALLS):
                return ""
            if isinstance(n, (ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
                return ""
    elif not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "f"):
        return ""
    seg = ast.get_source_segment(src, expr)
    return f"__pred = ({seg})\n" if seg else ""


def _bench_code(answer: str, style: str, label: dict | None = None) -> str:
    if style == "crux":
        return _crux_pred(answer, (label or {}).get("crux_mode", "o"))
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
        path = {"call": "/test_program_call", "checker": "/test_program_checker"}.get(route, "/test_program_stdio")
        n_tests = int(label["n_tests"])
        body["tests_ref"] = label["tests_ref"]
        if route == "call":
            body["fn_name"] = label["fn_name"]
        if route == "checker" and label.get("checker"):
            body["checker"] = label["checker"]
    if _LOCAL or not _BENCH_API_URL:
        return None
    res = _post(_BENCH_API_URL, path, body, n_tests, min(3600.0, (timeout + 1.0) * n_tests + 60.0))
    return None if res is None else all(res)


def _score_bench(response: str, label: dict) -> float:
    code = _bench_code(_bench_answer(response), label.get("extract", "olmes"), label)
    if not code.strip():
        STATS.note("no_code")
        STATS.note_sample(0.0, False)
        return 0.0
    ok = bench_pass(code, label)
    if ok is None:
        STATS.note("api_error")
        STATS.note_sample(0.0, False)
        return 0.0
    STATS.note_sample(float(ok), ok, float(ok))
    return float(ok) * REWARD_SCALE


# --- reward-hack monitoring -----------------------------------------------------
# Informational only: nothing here changes a reward. Static flags come from scan_code, the
# rest from the server's per-request outcome counts.
SCAN_KEYS = ("touch_frames", "touch_builtins", "helper_tamper", "read_own_file", "exit_call", "sys_modules",
             "eq_true", "class_assign", "trace_hook", "atexit", "signal", "ctypes", "subprocess")
SERVER_KEYS = ("early_exit", "strict_reject", "timeout")
HACK_KEYS = SERVER_KEYS + SCAN_KEYS

_FRAME_CALLS = {"sys._getframe", "inspect.stack", "inspect.currentframe", "inspect.getouterframes",
                "inspect.trace", "gc.get_objects", "gc.get_referrers"}
_FRAME_NAMES = {"f_back", "f_locals", "f_globals", "tb_frame", "_getframe", "currentframe",
                "get_referrers", "get_objects"}
_OWN_FILES = ("solution.py", "_case.json")
_OPENERS = {"open", "io.open", "codecs.open", "tokenize.open", "pathlib.Path", "Path"}
_MUTATORS = {"update", "pop", "popitem", "setdefault", "clear", "__setitem__", "__delitem__", "__setattr__",
             "__delattr__"}
_PROC_MODULES = {"subprocess", "pty", "multiprocessing"}
_PROC_CALLS = ("os.system", "os.popen", "os.exec", "os.spawn", "os.posix_spawn", "os.fork")
# helper_tamper: what the strict tests' own helpers and builtins run on (the residual attacks in
# test_code_exec_adversarial.py): the helpers by name, function internals, module-level rebinding
# of builtins the tests call, and attributes of imported modules (sys excepted: stdin/stdout swaps
# are common and sys has its own keys).
_TEST_HELPERS = {"__v7canon__", "_v7_close"}
_FUNC_INTERNALS = {"__code__", "__defaults__", "__kwdefaults__", "__globals__", "__closure__"}
_TEST_BUILTINS = {"all", "any", "abs", "len", "isinstance", "issubclass", "sorted", "sum", "min", "max", "type",
                  "round", "list", "set", "tuple", "dict", "str", "int", "float", "bool", "repr", "zip", "map"}


def scan_code(code: str) -> dict:
    """Booleans for constructs that only a grader exploit needs (SCAN_KEYS). Code that does
    not parse gets all False: it cannot pass a test anyway."""
    flags = dict.fromkeys(SCAN_KEYS, False)
    try:
        tree = ast.parse(code)
    except Exception:
        return flags
    alias, mods, names, strs, mutated = {}, set(), set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.split(".")[0]
                mods.add(top)
                alias[a.asname or top] = a.name if a.asname else top
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
            for a in n.names:
                alias[a.asname or a.name] = f"{n.module}.{a.name}"

    def dotted(e):
        # "sys._getframe" for sys._getframe / from sys import _getframe / import sys as s; s._getframe
        if isinstance(e, ast.Name):
            return alias.get(e.id, e.id)
        if isinstance(e, ast.Attribute):
            base = dotted(e.value)
            return base and f"{base}.{e.attr}"
        if isinstance(e, ast.Subscript):
            return dotted(e.value)
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "vars" and len(e.args) == 1:
            base = dotted(e.args[0])
            return base and f"{base}.__dict__"
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "globals" and not e.args:
            return "__globals__"
        return None

    calls, attr_stores, tamper = set(), set(), False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            tamper |= n.name in _TEST_BUILTINS
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, (ast.Lambda, ast.Name, ast.Attribute)):
            tamper |= any(isinstance(t, ast.Name) and t.id in _TEST_BUILTINS
                          for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            tamper |= any((a.asname or a.name) in _TEST_BUILTINS for a in n.names)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
            if isinstance(n.ctx, ast.Store) and n.id == "__builtins__":
                mutated.add("__builtins__")
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                mutated.add(dotted(n.value) or "")
                attr_stores.add(dotted(n.value) or "")
                flags["class_assign"] |= n.attr == "__class__"
                tamper |= n.attr in _FUNC_INTERNALS
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)):
            mutated.add(dotted(n.value) or "")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            strs.add(n.value)
        elif isinstance(n, ast.Call):
            fn = dotted(n.func) or ""
            calls.add(fn)
            if isinstance(n.func, ast.Attribute) and n.func.attr in _MUTATORS:
                mutated.add(dotted(n.func.value) or "")
            if fn.rpartition(".")[2] in ("setattr", "delattr", "__setattr__", "__delattr__") and n.args:
                mutated.add(dotted(n.args[0]) or "")
                attr_stores.add(dotted(n.args[0]) or "")
                tamper |= any(isinstance(a, ast.Constant) and a.value in _FUNC_INTERNALS for a in n.args[1:2])
                flags["class_assign"] |= any(isinstance(a, ast.Constant) and a.value == "__class__" for a in n.args)
            if fn in _OPENERS and n.args:
                a = n.args[0]
                flags["read_own_file"] |= (isinstance(a, ast.Name) and a.id == "__file__") or dotted(a) == "sys.argv"
            flags["exit_call"] |= fn in ("exit", "quit", "sys.exit", "os._exit", "os.abort", "os.kill")
        elif isinstance(n, ast.Global):
            tamper |= bool(_TEST_BUILTINS & set(n.names))
        elif isinstance(n, ast.Raise) and n.exc is not None:
            exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
            flags["exit_call"] |= isinstance(exc, ast.Name) and exc.id in ("SystemExit", "KeyboardInterrupt")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ("__eq__", "__ne__"):
            rets = {r.value.value if isinstance(r.value, ast.Constant) else None
                    for r in ast.walk(n) if isinstance(r, ast.Return)}
            flags["eq_true"] |= rets == {n.name == "__eq__"}
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda) \
                and isinstance(n.value.body, ast.Constant) and n.value.body.value in (True, False):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    flags["eq_true"] |= (t.id, n.value.body.value) in (("__eq__", True), ("__ne__", False))

    mods |= {s for s in strs if s in ("atexit", "signal", "ctypes", "_ctypes") or s in _PROC_MODULES}
    flags["touch_frames"] = bool(calls & _FRAME_CALLS or (names | strs) & _FRAME_NAMES)
    flags["touch_builtins"] = any(m.split(".")[0] in ("builtins", "__builtins__") for m in mutated)
    imported = {v.split(".")[0] for v in alias.values()} - {"sys"}
    flags["helper_tamper"] = bool(tamper or (names | strs) & _TEST_HELPERS or "__globals__" in mutated
                                  or flags["touch_builtins"]
                                  or any(b.split(".")[0] in imported for b in attr_stores if b))
    flags["read_own_file"] |= any(s.endswith(_OWN_FILES) for s in strs)
    flags["exit_call"] |= "_exit" in strs
    flags["sys_modules"] = any(m == "sys.modules" or m.startswith("sys.modules.") for m in mutated)
    flags["trace_hook"] = bool({"sys.settrace", "sys.setprofile", "threading.settrace", "threading.setprofile"} & calls
                               or {"settrace", "setprofile"} & strs)
    flags["atexit"] = "atexit" in mods
    flags["signal"] = "signal" in mods
    flags["ctypes"] = bool({"ctypes", "_ctypes"} & mods)
    flags["subprocess"] = bool(_PROC_MODULES & mods or any(c.startswith(_PROC_CALLS) for c in calls))
    return flags


def hack_flags(code: str, server: dict, solved: bool) -> dict:
    flags = {k: bool(server.get(k, 0)) for k in SERVER_KEYS}
    flags.update(scan_code(code))
    flags["solved"] = bool(solved)
    return flags


class _Window:
    """(pass rate, hack_flags) of the training samples scored since the last rollout log, newest
    CODE_HACK_WINDOW_MAX kept. Sample metadata alone would miss groups the dynamic filter drops,
    and those include every group a hack solves 8/8 (and, under the binary reward, every all-pass
    or all-fail group)."""

    def __init__(self, maxlen: int):
        self.lock = threading.Lock()
        self.items = collections.deque(maxlen=maxlen)

    def add(self, rate: float, flags: dict):
        with self.lock:
            self.items.append((rate, flags))

    def drain(self) -> list:
        with self.lock:
            out = list(self.items)
            self.items.clear()
        return out


WINDOW = _Window(_WINDOW_MAX)


def pass_rate_metrics(rates: list) -> dict:
    """Over graded samples (think-gate, no-code and api_error samples never reach the grader):
    pass_rate_mean is the mean raw pass rate, the partial-credit reward; all_pass_rate is the
    fraction passing every test, the binary reward at threshold 1.0."""
    n = len(rates)
    if not n:
        return {}
    return {"n_graded": n, "pass_rate_mean": sum(rates) / n,
            "all_pass_rate": sum(r >= 1.0 - 1e-9 for r in rates) / n}


def hack_metrics(flags: list) -> dict:
    """Rates over a list of hack_flags dicts. `solved_<k>_rate` is among samples with raw
    reward >= 0.99, i.e. how often a flagged program is also one that wins."""
    n = len(flags)
    if not n:
        return {}
    solved = [f for f in flags if f.get("solved")]
    any_keys = tuple(k for k in HACK_KEYS if k != "timeout")
    out = {"n_scored": n, "solved_rate": len(solved) / n}
    for k in HACK_KEYS:
        out[f"{k}_rate"] = sum(bool(f.get(k)) for f in flags) / n
        out[f"solved_{k}_rate"] = sum(bool(f.get(k)) for f in solved) / len(solved) if solved else 0.0
    out["any_rate"] = sum(any(f.get(k) for k in any_keys) for f in flags) / n
    out["solved_any_rate"] = sum(any(f.get(k) for k in any_keys) for f in solved) / len(solved) if solved else 0.0
    return out


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics=None, rollout_time=None) -> bool:
    """--custom-rollout-log-function-path code_reward.log_rollout_data. Logs, on the rollout/step
    axis (only rollout/* is bound to it in W&B):
        rollout/code_hack/<k>          every training sample scored since the previous log,
                                       including groups the dynamic filter dropped
        rollout/code_hack/trained/<k>  the batch handed to the trainer (sample.metadata)
        rollout/code/<k>               pass_rate_metrics, same two scopes (rollout/code/trained/<k>)
    Returns False so miles' default rollout logging still runs."""
    try:
        from miles.utils.metric_utils import compute_rollout_step
        from miles.utils.tracking_utils import tracking

        trained, trained_rates = [], []
        for group in samples or []:
            for s in group if isinstance(group, (list, tuple)) else [group]:
                meta = getattr(s, "metadata", None)
                if not isinstance(meta, dict):
                    continue
                if isinstance(meta.get("code_hack"), dict):
                    trained.append(meta["code_hack"])
                if isinstance(meta.get("code_pass_rate"), (int, float)):
                    trained_rates.append(float(meta["code_pass_rate"]))
        window = WINDOW.drain()
        log_dict = {f"rollout/code_hack/{k}": v for k, v in hack_metrics([f for _, f in window]).items()}
        log_dict |= {f"rollout/code_hack/trained/{k}": v for k, v in hack_metrics(trained).items()}
        log_dict |= {f"rollout/code/{k}": v for k, v in pass_rate_metrics([r for r, _ in window]).items()}
        log_dict |= {f"rollout/code/trained/{k}": v for k, v in pass_rate_metrics(trained_rates).items()}
        if log_dict:
            log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
            logger.info("code_hack %s: %s", rollout_id, log_dict)
            tracking.log(args, log_dict, step_key="rollout/step")
    except Exception as exc:  # never let a metrics add-on break the run
        logger.warning("code_hack logging failed (%s)", exc, exc_info=True)
    return False


# --- scoring --------------------------------------------------------------------
def _is_eval(meta: dict, evaluation_kwarg: bool) -> bool:
    return evaluation_kwarg or str(meta.get("split", "")).lower() == "eval"


def reward_from_pass_rate(rate: float, evaluation: bool) -> float:
    """Unscaled reward for a graded sample (see the module docstring for the modes)."""
    if evaluation:
        return float(rate >= 1.0 - 1e-9)
    if REWARD_MODE == "partial":
        return rate
    return float(rate >= PASS_THRESHOLD - 1e-9)


def _gated_pass_rate(response: str, tests, evaluation: bool, opts=None, flags=None):
    """(pass rate, reason, code) of a non-benchmark response behind the reward's gates:
    (rate, None, code) when the program ran, (0.0, reason, "") when the think gate or code
    extraction rejected it, and (None, "api_error", code) when the sandbox did not answer."""
    if _REQUIRE_THINK:
        ok = ("</think>" in response) if evaluation else _has_clean_think(response)
        if not ok:
            return 0.0, "think_gate_fail", ""
    code = extract_code(_answer_of(response))
    if not code:
        return 0.0, "no_code", ""
    rate = pass_rate(code, tests, opts, flags)
    if rate is None:
        return None, "api_error", code
    return rate, None, code


def _grader_options(meta: dict, checker, evaluation: bool):
    return {"checker": checker} if checker else None if evaluation else train_options(meta)


def train_all_pass(response: str, label, metadata=None) -> tuple[bool | None, str | None]:
    """Whether a training response earns the full training reward, through the same grading as
    _score_one (think gate, checker rows, strict asserts, per-row stdio options from metadata).

    Used by difficulty_filter.py to count solves. "Full reward" is pass rate >=
    CODE_REWARD_PASS_THRESHOLD under CODE_REWARD_MODE=binary and every test passing under
    partial (where any passing test earns some reward). (None, "api_error") means the sandbox
    did not answer, which a caller must retry rather than count as a failure; (False,
    "no_tests") means the row itself is ungradable.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    tests = _parse_tests(label)
    if tests is None or (isinstance(tests, dict) and "bench" in tests):
        return False, "no_tests"
    tests, checker = split_checker(tests, meta)
    if tests is None:
        return False, "no_tests"
    rate, reason, _ = _gated_pass_rate(response or "", tests, False, _grader_options(meta, checker, False))
    if rate is None:
        return None, reason
    need = PASS_THRESHOLD if REWARD_MODE == "binary" else 1.0
    return rate >= need - 1e-9, reason


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
    tests, checker = split_checker(tests, meta)
    if tests is None:
        STATS.note("no_tests")
        STATS.note_sample(0.0, False)
        return 0.0

    server = {}
    rate, reason, code = _gated_pass_rate(response, tests, evaluation, _grader_options(meta, checker, evaluation),
                                          server)
    if reason is not None:
        STATS.note(reason)
        STATS.note_sample(0.0, False)
        return 0.0

    raw = reward_from_pass_rate(rate, evaluation)
    STATS.note_sample(raw, rate >= 1.0 - 1e-9, rate)
    flags = hack_flags(code, server, raw >= 0.99)
    if isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata["code_hack"] = flags
        sample.metadata["code_pass_rate"] = rate
    if not evaluation:
        WINDOW.add(rate, flags)
    return raw * REWARD_SCALE


# Grading blocks a thread for the whole sandbox round trip (tens of seconds for a LiveCodeBench
# sample). asyncio.to_thread would share the loop's default pool, at most 32 threads, so miles'
# per-sample calls could never grade more than 32 samples at once whatever CODE_REWARD_CONCURRENCY
# says; evals then starve the engines. A pool of its own makes CODE_REWARD_CONCURRENCY the cap.
_EXECUTOR = None
_EXECUTOR_LOCK = threading.Lock()


def _executor():
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max(1, _CONCURRENCY), thread_name_prefix="code-reward")
    return _EXECUTOR


async def batched_reward(args, samples, **kwargs):
    """Entry point for both call shapes miles uses (single Sample and list[Sample])."""
    evaluation = bool(kwargs.get("evaluation", False))
    loop = asyncio.get_running_loop()
    if not isinstance(samples, (list, tuple)):
        reward = await loop.run_in_executor(_executor(), _score_one, samples, evaluation)
        _note_domains([samples], [reward], evaluation)
        return reward
    if not samples:
        return []
    sem = asyncio.Semaphore(max(1, _CONCURRENCY))

    async def _one(s):
        async with sem:
            return await loop.run_in_executor(_executor(), _score_one, s, evaluation)

    rewards = await asyncio.gather(*[_one(s) for s in samples])
    _note_domains(samples, rewards, evaluation)
    return rewards


def _note_domains(samples, rewards, evaluation: bool) -> None:
    """Hand graded training samples to per_domain_logger, so its per-domain stats include the
    groups the dynamic filter drops (PER_DOMAIN_METRICS=0 turns it off)."""
    if evaluation or os.environ.get("PER_DOMAIN_METRICS", "1") != "1":
        return
    try:
        import per_domain_logger

        keep = [(s, r) for s, r in zip(samples, rewards) if not _is_eval(getattr(s, "metadata", None) or {}, False)]
        per_domain_logger.note([s for s, _ in keep], [r for _, r in keep])
    except Exception as exc:  # metrics only
        logger.warning("per-domain note failed (%s)", exc)


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
    # (response, label, metadata, (binary reward, partial reward), pass rate or None, why)
    cases = [
        (good, stdio, {}, (1.0, 1.0), 1.0, "stdio all pass"),
        (half, stdio, {}, (0.0, 0.5), 0.5, "stdio half pass"),
        (half, stdio, {"split": "eval"}, (0.0, 0.0), 0.5, "eval is all-or-nothing in both modes"),
        ("```python\nprint(5)\n```", stdio, {}, (0.0, 0.0 if _REQUIRE_THINK else 0.25),
         None if _REQUIRE_THINK else 0.25, "think gate"),
        ("<think>r</think>no code here", stdio, {}, (0.0, 0.0), 0.0, "prose is not a program"),
        ("<think>r</think>```python\ndef double(x): return 2*x\n```", asserts, {}, (1.0, 1.0), 1.0, "assert list"),
        ("<think>r</think>```python\ndef inc(x): return x+1\n```", pytest_style, {}, (0.0, 0.5), 0.5,
         "pytest functions"),
        ("<think>r</think>```python\nwhile True: pass\n```", stdio, {}, (0.0, 0.0), 0.0, "infinite loop"),
        (good, json.dumps([stdio]), {}, (1.0, 1.0), 1.0, "Dolci shape: list holding one JSON string"),
    ]
    # checker rows: "print two non-negative integers that sum to n"; the jury answers differ from n 0
    pair_checker = ("import sys\nn = int(open(sys.argv[1]).read().split()[0])\nt = open(sys.argv[3]).read().split()\n"
                    "ok = len(t) == 2 and all(x.isdigit() for x in t) and int(t[0]) + int(t[1]) == n\n"
                    "print(1 if ok else 0)\n")
    pairs = [{"input": "5\n", "output": "2 3\n"}, {"input": "10\n", "output": "5 5\n"},
             {"input": "1\n", "output": "0 1\n"}, {"input": "7\n", "output": "3 4\n"}]
    pair_label, pair_meta = json.dumps(pairs), {"checker": pair_checker}
    alt = "<think>r</think>\n```python\nn = int(input())\nprint(n, 0)\n```"
    cases += [
        (alt, pair_label, pair_meta, (1.0, 1.0), 1.0, "checker accepts an answer other than the jury's"),
        (alt, pair_label, {}, (0.0, 0.0), 0.0, "same answer without the checker: plain compare rejects it"),
        (alt, json.dumps({"tests": pairs, "checker": pair_checker}), {}, (1.0, 1.0), 1.0,
         "checker inside the label"),
        ("<think>r</think>\n```python\nprint(1, 1)\n```", pair_label, pair_meta, (0.0, 0.0), 0.0,
         "checker rejects a wrong answer"),
        ("<think>r</think>\n```python\nn = int(input())\nprint(n, 0) if n % 2 else print(1, 1)\n```", pair_label,
         pair_meta, (0.0, 0.75), 0.75, "checker row, 3 of 4 tests"),
        (alt, pair_label, {"checker": "raise RuntimeError('broken checker')\n"}, (0.0, 0.0), 0.0,
         "checker crash fails every test"),
        (alt, pair_label, {"checker": "print('accepted')\n"}, (0.0, 0.0), 0.0, "checker output other than 1 fails"),
        (alt, pair_label, dict(pair_meta, split="eval"), (1.0, 1.0), 1.0, "checker row in eval"),
        ("```python\nn = int(input())\nprint(n, 0)\n```", pair_label, pair_meta,
         (0.0, 0.0) if _REQUIRE_THINK else (1.0, 1.0), None if _REQUIRE_THINK else 1.0, "think gate on a checker row"),
        ("<think>r</think>\n```python\nwhile True: pass\n```", pair_label, pair_meta, (0.0, 0.0), 0.0,
         "checker row, infinite loop"),
        (alt, pair_label, {"checker": 7}, (0.0, 0.0), None, "unusable checker: no_tests"),
    ]
    ok = True
    env_mode, env_threshold = REWARD_MODE, PASS_THRESHOLD
    print(f"env: CODE_REWARD_MODE={env_mode} CODE_REWARD_PASS_THRESHOLD={env_threshold}")
    for REWARD_MODE in REWARD_MODES:
        PASS_THRESHOLD = 1.0
        for resp, label, meta, wants, want_rate, why in cases:
            want = wants[REWARD_MODES.index(REWARD_MODE)]
            s = _S(resp, label, dict(meta))
            got = asyncio.run(batched_reward(None, [s]))[0]
            rate = s.metadata.get("code_pass_rate")
            good_rate = rate == want_rate if want_rate is None else rate is not None and abs(rate - want_rate) < 1e-6
            flag = "ok " if abs(got - want * REWARD_SCALE) < 1e-6 and good_rate else "FAIL"
            ok &= flag == "ok "
            print(f"  {flag} [{REWARD_MODE}] reward={got:.3f} want={want * REWARD_SCALE:.3f} "
                  f"pass_rate={rate} want={want_rate}  ({why})")
    REWARD_MODE, PASS_THRESHOLD = "binary", 0.5
    got = asyncio.run(batched_reward(None, [_S(half, stdio)]))[0]
    flag = "ok " if abs(got - REWARD_SCALE) < 1e-6 else "FAIL"
    ok &= flag == "ok "
    print(f"  {flag} [binary, threshold 0.5] reward={got:.3f} want={REWARD_SCALE:.3f}  (stdio half pass)")
    REWARD_MODE, PASS_THRESHOLD = env_mode, env_threshold
    WINDOW.drain()
    STATS.dump()
    if STATS.counts["api_error"]:
        print("  FAIL api_error > 0: the service dropped requests")
        ok = False
    raise SystemExit(0 if ok else 1)
