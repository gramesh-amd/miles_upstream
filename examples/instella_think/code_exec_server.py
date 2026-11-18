#!/usr/bin/env python3
"""Code-execution service for the code RL reward (code_reward.py).

Implements the API the Instella reward client already speaks:

    POST <prefix>/test_program        {"program", "tests": [str], "max_execution_time"}
    POST <prefix>/test_program_stdio  {"program", "tests": [{"input","output"}], "max_execution_time"}
    POST <prefix>/test_program_call   {"program", "tests": [{"input","output"}], "fn_name", ...}
    POST <prefix>/test_program_checker {"program", "tests": [{"input","output"}], "checker"?, ...}
        -> {"results": [0|1, ...]}   one entry per test, in order
    GET  <prefix>/health              -> {"ok": true, ...}

Optional request fields, used by the benchmark evals (prepare_code_benchmarks.py):
    "tests_ref": path under --tests-dir, sent instead of "tests"
    "stop_on_fail": skip (and fail) tests not yet started once one has failed
    "name": the program's __name__ (default "__main__")
/test_program_call follows LiveCodeBench's call-based grading: the program is imported as
a module and fn_name (on Solution() if the code defines it) is called per test.
/test_program_checker follows open-r1's Codeforces judge (piston packages/codeforces run):
blank lines are dropped from both outputs; with a "checker" (Python source), it runs as
`checker.py input.txt correct_output.txt solution_output.txt` under the same guard and the
test passes iff it exits 0 printing 1 or 100; without one, outputs are compared token by token
with yes/no case-insensitive.

Standard library only, so it runs on a bare CPU host (python3 >= 3.8).

<prefix> is a secret path segment read from --token-file. The reward client builds
URLs as CODE_EXEC_API_URL + path, so pointing CODE_EXEC_API_URL at
http://host:port/<token> authenticates every request with no client change.
Anything else gets a 404.

ISOLATION -- read before trusting it. The intended host has no Docker and denies
unprivileged user namespaces, so programs run as the service's own user with:
  * a fresh `python3 -I` per test, in its own session, killed by process group on
    timeout, under rlimits on CPU, address space, file size, and open files;
  * a permanent audit hook (sys.addaudithook cannot be removed from Python) that
    blocks sockets, subprocess/exec/fork, signals, ctypes, deletes/renames/chmod,
    writes outside the per-test scratch dir, and reads under /home, /shared_nfs,
    /root, /mnt.
That stops accidental damage from model-written code. It is not a security
boundary against code written to escape it (a C extension can bypass audit
hooks), so do not expose this service beyond the training cluster.
"""

import argparse
import collections
import functools
import hashlib
import json
import os
import re
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Executed in the child before the user program. Kept as a separate file (written
# once at startup) so the user program's tracebacks and line numbers are its own.
GUARD_SOURCE = r'''
import os, sys, builtins

_SCRATCH = os.path.realpath(os.environ.get("CODE_EXEC_SCRATCH", "/nonexistent"))
_NO_READ = ("/home", "/shared_nfs", "/root", "/mnt")
# The interpreter's own tree (a venv under /home, say) must stay readable or imports fail.
_OWN = tuple({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
_BLOCKED = {
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
    "socket.getaddrinfo", "socket.gethostbyname", "socket.__new__",
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "os.kill", "os.killpg", "pty.spawn",
    "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
    "os.chmod", "os.chown", "os.truncate", "os.link", "os.symlink",
    "shutil.rmtree", "shutil.move", "shutil.chown",
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function", "ctypes.addressof",
    "ctypes.cdata", "ctypes.create_string_buffer", "ctypes.string_at",
    "sys.addaudithook", "sys.setprofile", "sys.settrace",
    "webbrowser.open",
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

def _inside(path, root):
    return path == root or path.startswith(root + os.sep)

def _hook(event, args):
    # `import ctypes` (and so `import numpy`) dlopens the running process itself. That
    # loads nothing new, and symbol lookup and foreign calls stay blocked.
    if event == "ctypes.dlopen" and args and args[0] is None:
        return
    if event in _BLOCKED or event.startswith("os.exec") or event.startswith("os.spawn"):
        raise PermissionError("sandbox: %s is not allowed" % event)
    if event == "open":
        path, mode, flags = args
        if not isinstance(path, (str, bytes)):
            return
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        real = os.path.realpath(path)
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                  (isinstance(flags, int) and flags & _WRITE_FLAGS)
        if writing and not _inside(real, _SCRATCH) and real not in ("/dev/null",):
            raise PermissionError("sandbox: write outside scratch: %s" % real)
        if any(_inside(real, p) for p in _NO_READ) and not _inside(real, _SCRATCH) \
                and not any(_inside(real, p) for p in _OWN):
            raise PermissionError("sandbox: read not allowed: %s" % real)

sys.addaudithook(_hook)
del _hook

_prog = sys.argv[1]
with open(_prog) as _f:
    _src = _f.read()
sys.argv = [_prog] + sys.argv[2:]
_code = compile(_src, "solution.py", "exec")
_g = {"__name__": os.environ.get("CODE_EXEC_NAME", "__main__"), "__file__": "solution.py",
      "__builtins__": builtins}


def _run(code, g, must_finish):
    # must_finish: the per-run token arrives on stdin's first line and is consumed here, so the
    # program can neither read it nor create the done file; the file is written only if exec
    # returns. Kept in this frame, not module globals, so `import __main__` cannot reach it
    # (frame introspection still could: same limit as EvalPlus / open-instruct).
    if must_finish:
        done = os.path.join(_SCRATCH, "_done_" + sys.stdin.readline().strip())
        oopen, oclose, oflags = os.open, os.close, os.O_WRONLY | os.O_CREAT
    exec(code, g)
    if must_finish:
        oclose(oopen(done, oflags, 0o600))


_run(_code, _g, os.environ.get("CODE_EXEC_MUST_FINISH") == "1")
del _run
'''

_TLS = threading.local()
STATS_LOCK = threading.Lock()
STATS = {"requests": 0, "tests": 0, "passed": 0, "timeouts": 0, "errors": 0, "early_exits": 0, "busy_s": 0.0}


def _bump(**kw):
    with STATS_LOCK:
        for k, v in kw.items():
            STATS[k] += v


_INT_RE = re.compile(r"[+-]?\d+")


def _tokens_equal(x: str, y: str, float_tol, case_insensitive: bool) -> bool:
    # case_insensitive folds YES/NO only (open-r1 / Codeforces convention), not names like Alice.
    if x == y or (case_insensitive and x.lower() == y.lower() and y.lower() in ("yes", "no")):
        return True
    try:
        dx, dy = Decimal(x), Decimal(y)
    except (InvalidOperation, ValueError):
        return False
    if dx == dy:
        return True
    if float_tol is None or not (dx.is_finite() and dy.is_finite()):
        return False
    if _INT_RE.fullmatch(x) and _INT_RE.fullmatch(y):
        return False
    # Codeforces-style: absolute or relative error within float_tol.
    return abs(dx - dy) <= Decimal(str(float_tol)) * max(Decimal(1), abs(dy))


def _lines_equal_with_decimal(a: str, b: str, float_tol=None, case_insensitive=False) -> bool:
    """Same comparison as the Instella reward's local path: per-line, whitespace
    stripped, with a Decimal fallback so 1.0 == 1 == 1.00. float_tol and
    case_insensitive are per-row options for statements that allow them (training only)."""
    a_lines = [ln.strip() for ln in a.strip().split("\n")]
    b_lines = [ln.strip() for ln in b.strip().split("\n")]
    if len(a_lines) != len(b_lines):
        return False
    for al, bl in zip(a_lines, b_lines):
        if al == bl:
            continue
        at, bt = al.split(), bl.split()
        if float_tol is None and not case_insensitive:
            # eval path: every token of a differing line must be numeric (LCB rule).
            try:
                if [Decimal(t) for t in at] != [Decimal(t) for t in bt]:
                    return False
            except (InvalidOperation, ValueError):
                return False
            continue
        if len(at) != len(bt) or not all(_tokens_equal(x, y, float_tol, case_insensitive) for x, y in zip(at, bt)):
            return False
    return True


# strict (training) equality: both sides of `assert A == B` pass through __canon, which
# reduces values to builtin types, so an object whose __eq__ always returns True cannot pass.
# Every type it relies on is taken from literals and bound as a default when the prelude is
# defined, so rebinding builtins (type, isinstance, ...) in the solution does not reach it.
# Heap types (any class defined in Python) must keep identity equality; C types with their
# own equality (Decimal, datetime, deque, slice, ...) have their referents checked the same way.
CANON_PRELUDE = r'''
def __canon(x, _d=0, _T=().__class__.__class__, _O=().__class__.__mro__[-1],
            _B=True.__class__, _I=(0).__class__, _F=(0.0).__class__, _C=(0j).__class__,
            _S="".__class__, _Y=b"".__class__, _L=[].__class__, _U=().__class__,
            _D={}.__class__, _ST={0}.__class__, _FS=None, _W=None, _R=None,
            _TD=().__class__.__class__.__dict__):
    if _d > 200:
        raise TypeError("value nested too deeply")
    t = _T(x)
    if x is None or t is _B or t is _I or t is _F or t is _C or t is _S or t is _Y:
        return x
    mro = _TD["__mro__"].__get__(t)
    for k in mro:
        if _TD["__flags__"].__get__(k) & 512 == 0 and _TD["__module__"].__get__(k) == "numpy":
            return __canon(x.tolist(), _d + 1)
    for k in mro:
        if k is _B:
            return _I.__bool__(x)
        if k is _I:
            return _I.__int__(x)
        if k is _F:
            return _F.__float__(x)
        if k is _C:
            return _C.__add__(x, 0j)
        if k is _S:
            return _S.__str__(x)
        if k is _Y:
            return b"".join((x,))
        if k is _D:
            return {__canon(a, _d + 1): __canon(v, _d + 1) for a, v in _D.items(x)}
        if k is _L:
            return [__canon(v, _d + 1) for v in _L.__iter__(x)]
        if k is _U:
            return _U(__canon(v, _d + 1) for v in _U.__iter__(x))
        if k is _ST or k is _FS:
            return _FS(__canon(v, _d + 1) for v in k.__iter__(x))
    identity = True
    for name in ("__eq__", "__ne__"):
        for k in mro:
            kd = _TD["__dict__"].__get__(k)
            if name in kd:
                if _TD["__flags__"].__get__(k) & 512 or _T(kd[name]) is not _W:
                    raise TypeError("user-defined equality in strict comparison")
                identity = identity and k is _O
                break
    if not identity and t is not _T:
        for r in _R(x):
            __canon(r, _d + 1)
    return x
import gc as __v7_gc
__canon.__defaults__ = __canon.__defaults__[:-4] + (
    [k for k in ().__class__.__mro__[-1].__subclasses__()
     if ().__class__.__class__.__dict__["__name__"].__get__(k) == "frozenset"
     and not ().__class__.__class__.__dict__["__flags__"].__get__(k) & 512][0],
    ().__class__.__mro__[-1].__dict__["__eq__"].__class__,
    __v7_gc.get_referents, ().__class__.__class__.__dict__)
del __v7_gc
'''


def strict_assert(test: str) -> str:
    """`assert A == B[, msg]` and `assert _v7_close(A, B)` (the v7 data's tolerant float
    compare) -> both operands through __canon; any other test is left as is."""
    import ast

    try:
        tree = ast.parse(test)
    except SyntaxError:
        return test
    if not hasattr(ast, "unparse"):
        return test
    wrap = lambda e: ast.Call(func=ast.Name(id="__canon", ctx=ast.Load()), args=[e], keywords=[])
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.NotEq)):
            t.left, t.comparators = wrap(t.left), [wrap(t.comparators[0])]
            changed = True
        elif isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id == "_v7_close" \
                and len(t.args) == 2 and not t.keywords:
            t.args = [wrap(a) for a in t.args]
            changed = True
    if not changed:
        return test
    return CANON_PRELUDE + "\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _finished(d: str, token: str) -> bool:
    return os.path.exists(os.path.join(d, f"_done_{token}"))


_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _drop_blank_lines(s: str) -> str:
    return "".join(ln + "\n" for ln in s.replace("\r", "").split("\n") if ln != "")


def cf_tokens_equal(out: str, expected: str) -> bool:
    """piston's Codeforces diff: whitespace-insensitive, YES/NO case-insensitive."""
    def norm(s):
        return _YES_NO_RE.sub(lambda m: m.group(1).upper(), s.replace("\r", "")).split()
    return norm(out) == norm(expected)


class Runner:
    def __init__(self, workers: int, mem_mb: int, python: str, scratch_root: str):
        self.sem = threading.BoundedSemaphore(workers)
        # Tests of one request fan out here; the semaphore, not the pool size, bounds
        # how many processes actually run, so a big request cannot starve small ones.
        self.pool = ThreadPoolExecutor(max_workers=workers * 8)
        self.mem_bytes = mem_mb * 1024 * 1024
        self.python = python
        self.scratch_root = scratch_root
        os.makedirs(scratch_root, exist_ok=True)
        # named by content so servers of different versions can share a scratch root
        tag = hashlib.sha256(GUARD_SOURCE.encode()).hexdigest()[:12]
        self.guard = os.path.join(scratch_root, f"_guard_{tag}.py")
        tmp = f"{self.guard}.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(GUARD_SOURCE)
        os.replace(tmp, self.guard)
        self.env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

    def _limits(self, timeout: float):
        cpu = int(timeout) + 1
        mem = self.mem_bytes

        def apply():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.nice(5)

        return apply

    def run_one(self, program: str, stdin: str, timeout: float, name: str = "__main__", case=None,
                files=None, args=(), must_finish=False):
        """Returns (returncode or None on timeout, stdout). `name` is the program's
        __name__; `case`, if given, is written to _case.json beside it; `files` maps
        file names to contents written beside it; `args` follow it on the command line.
        With must_finish, a clean exit only counts if control reached the end of the
        program: exiting early (sys.exit(0) before appended tests run) returns -1."""
        token = secrets.token_hex(16) if must_finish else None
        if token:
            stdin = token + "\n" + stdin
        with self.sem:
            # A stop_on_fail request's tests queue here together; once one has failed, the
            # rest skip instead of running.
            failed = getattr(_TLS, "failed", None)
            if failed is not None and failed.is_set():
                return -1, ""
            t0 = time.monotonic()
            d = tempfile.mkdtemp(prefix="t-", dir=self.scratch_root)
            try:
                prog = os.path.join(d, "solution.py")
                with open(prog, "w") as f:
                    f.write(program)
                if case is not None:
                    with open(os.path.join(d, "_case.json"), "w") as f:
                        json.dump(case, f)
                for fname, content in (files or {}).items():
                    with open(os.path.join(d, fname), "w") as f:
                        f.write(content)
                env = dict(self.env, HOME=d, TMPDIR=d, CODE_EXEC_SCRATCH=d, CODE_EXEC_NAME=name,
                           CODE_EXEC_MUST_FINISH="1" if token else "0")
                p = subprocess.Popen(
                    [self.python, "-I", self.guard, prog, *args],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    cwd=d, env=env, preexec_fn=self._limits(timeout), close_fds=True,
                )
                try:
                    out, _ = p.communicate(stdin.encode("utf-8", "replace"), timeout=timeout)
                    rc = p.returncode
                    if token and rc == 0 and not _finished(d, token):
                        _bump(early_exits=1)
                        rc = -1
                    return rc, out.decode("utf-8", "replace")
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # Bounded: a process that survives SIGKILL (e.g. stuck in uninterruptible I/O)
                    # would otherwise pin this worker forever, and a sandbox that loses workers this
                    # way stalls every request queued behind them.
                    try:
                        p.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        _bump(errors=1)
                    _bump(timeouts=1)
                    return None, ""
            except Exception:
                _bump(errors=1)
                return -1, ""
            finally:
                shutil.rmtree(d, ignore_errors=True)
                _bump(busy_s=time.monotonic() - t0)

    def _map(self, one, tests, stop_on_fail):
        """Runs one(tc) per test. With stop_on_fail, tests not yet started once any test
        has failed are reported failed without running: same all-pass verdict, less work."""
        if not stop_on_fail:
            return list(self.pool.map(one, tests))
        failed = threading.Event()

        def guarded(tc):
            if failed.is_set():
                return 0
            _TLS.failed = failed
            try:
                r = one(tc)
            finally:
                _TLS.failed = None
            if not r:
                failed.set()
            return r

        return list(self.pool.map(guarded, tests))

    def assert_tests(self, program, tests, timeout, name="__main__", stop_on_fail=False, strict=False):
        if strict:
            tests = [strict_assert(tc) for tc in tests]

        def one(tc):
            rc, _ = self.run_one(f"{program}\n\n{tc}\n", "", timeout, name=name, must_finish=True)
            return 1 if rc == 0 else 0

        return self._map(one, tests, stop_on_fail)

    def stdio_tests(self, program, tests, timeout, stop_on_fail=False, float_tol=None, case_insensitive=False):
        def one(tc):
            if not isinstance(tc, dict):
                return 0
            rc, out = self.run_one(program, str(tc.get("input", "")), timeout)
            return 1 if rc == 0 and _lines_equal_with_decimal(
                out, str(tc.get("output", "")), float_tol, case_insensitive) else 0

        return self._map(one, tests, stop_on_fail)

    def call_tests(self, program, tests, timeout, fn_name, stop_on_fail=False, strict=False):
        """LiveCodeBench call-based grading (lcb_runner testing_util.grade_call_based): the
        program is imported, not run as __main__; inputs are one JSON value per line."""
        target = "Solution()" if "class Solution" in program else None
        program = program + "\n\n" + CALL_HARNESS.format(fn=fn_name, target=target, strict=bool(strict))

        def one(tc):
            if not isinstance(tc, dict):
                return 0
            case = {"input": str(tc.get("input", "")), "output": str(tc.get("output", ""))}
            rc, _ = self.run_one(program, "", timeout, name="tmp_sol", case=case, must_finish=True)
            return 1 if rc == 0 else 0

        return self._map(one, tests, stop_on_fail)

    def judge_output(self, out, tc, checker=None):
        """open-r1 / piston Codeforces verdict for a program's stdout on one test."""
        if not checker:
            return 1 if cf_tokens_equal(out, str(tc.get("output", ""))) else 0
        files = {"input.txt": str(tc.get("input", "")).replace("\r", ""),
                 "correct_output.txt": _drop_blank_lines(str(tc.get("output", ""))),
                 "solution_output.txt": _drop_blank_lines(out)}
        rc, verdict = self.run_one(checker, "", CHECKER_TIMEOUT, files=files,
                                   args=("input.txt", "correct_output.txt", "solution_output.txt"))
        return 1 if rc == 0 and "".join(verdict.split()) in ("1", "100") else 0

    def checker_tests(self, program, tests, timeout, checker=None, stop_on_fail=False):
        def one(tc):
            if not isinstance(tc, dict):
                return 0
            rc, out = self.run_one(program, str(tc.get("input", "")), timeout)
            return self.judge_output(out, tc, checker) if rc == 0 else 0

        return self._map(one, tests, stop_on_fail)


CHECKER_TIMEOUT = 10.0  # piston runs checker.py under `timeout 10`


# Appended to call-based programs. Names are dunder-prefixed so the solution's own
# globals (it may star-import json, sys, io...) cannot shadow them.
CALL_HARNESS = r'''
def __grade():
    import json as __j
    import builtins as __b
    with __b.open("_case.json") as __f:
        __c = __j.load(__f)
    __obj = {target}
    __m = getattr(__obj, {fn!r}) if __obj is not None else globals()[{fn!r}]
    __p = __m(*[__j.loads(__l) for __l in __c["input"].split("\n")])
    if isinstance(__p, tuple):
        __p = list(__p)
    if {strict}:
        # open-instruct's JSON round trip: only plain JSON values can compare equal.
        __p = __j.loads(__j.dumps(__p, allow_nan=False))
    if not (__p == __j.loads(__c["output"])):
        raise SystemExit(1)
__grade()
'''


def make_tests_loader(tests_dir):
    """Resolves a request's "tests_ref" (a path relative to --tests-dir) to its test list.
    Benchmarks like LiveCodeBench carry up to ~150 MB of tests per problem, too much to
    send with every sample, so they stay on disk here."""
    if not tests_dir:
        return None
    root = os.path.realpath(tests_dir)
    locks = collections.defaultdict(threading.Lock)
    locks_guard = threading.Lock()

    # Parsing one of these files holds the GIL for seconds and stalls the whole server, so a
    # cache that misses under load (several nodes scoring many problems at once) starves every
    # request and they time out. Keep plenty of them: the sandbox host has the memory.
    @functools.lru_cache(maxsize=int(os.environ.get("CODE_EXEC_TESTS_CACHE", "256")))
    def parse(path):
        with open(path) as f:
            tests = json.load(f)
        if not isinstance(tests, list):
            raise ValueError("tests_ref file must hold a JSON list")
        return tests

    def load(ref):
        path = os.path.realpath(os.path.join(root, ref))
        if not path.startswith(root + os.sep):
            raise ValueError("tests_ref escapes --tests-dir")
        # One parse per file: concurrent misses on the same file wait for the first.
        with locks_guard:
            lock = locks[path]
        with lock:
            return parse(path)

    return load


def make_handler(runner: Runner, prefix: str, max_tests: int, max_timeout: float, started: float,
                 load_tests=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self):
            if not self.path.startswith(prefix + "/"):
                return None
            return self.path[len(prefix):]

        def do_GET(self):
            if self._route() != "/health":
                return self._send(404, {"error": "not found"})
            with STATS_LOCK:
                stats = dict(STATS)
            self._send(200, {"ok": True, "uptime_s": round(time.time() - started),
                             "workers": runner.sem._initial_value, **stats})

        def do_POST(self):
            route = self._route()
            if route not in ("/test_program", "/test_program_stdio", "/test_program_call", "/test_program_checker"):
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 64 << 20:
                    return self._send(413, {"error": "payload too large"})
                req = json.loads(self.rfile.read(n))
                program = req["program"]
                if "tests_ref" in req:
                    if load_tests is None:
                        raise ValueError("tests_ref needs the server's --tests-dir")
                    tests = load_tests(str(req["tests_ref"]))
                else:
                    tests = req["tests"]
                if not isinstance(program, str) or not isinstance(tests, list):
                    raise ValueError("program must be str, tests a list")
                timeout = min(float(req.get("max_execution_time", 10)), max_timeout)
                stop_on_fail = bool(req.get("stop_on_fail", False))
                fn_name = req.get("fn_name")
                if route == "/test_program_call" and not (isinstance(fn_name, str) and fn_name.isidentifier()):
                    raise ValueError("/test_program_call needs an identifier fn_name")
                name = req.get("name", "__main__")
                if not (isinstance(name, str) and name.isidentifier()):
                    raise ValueError("name must be an identifier")
                checker = req.get("checker") or None
                if checker is not None and not isinstance(checker, str):
                    raise ValueError("checker must be Python source")
                float_tol = req.get("float_tol")
                float_tol = None if float_tol is None else float(float_tol)
                case_insensitive = bool(req.get("case_insensitive", False))
                strict = bool(req.get("strict", False))
            except (OSError, ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            total = len(tests)
            if total > max_tests:
                tests = tests[:max_tests]
            padded = total - len(tests)
            if route == "/test_program":
                res = runner.assert_tests(program, [str(t) for t in tests], timeout, name, stop_on_fail, strict)
            elif route == "/test_program_stdio":
                res = runner.stdio_tests(program, tests, timeout, stop_on_fail, float_tol, case_insensitive)
            elif route == "/test_program_checker":
                res = runner.checker_tests(program, tests, timeout, checker, stop_on_fail)
            else:
                res = runner.call_tests(program, tests, timeout, fn_name, stop_on_fail, strict)
            _bump(requests=1, tests=len(res), passed=sum(res))
            # Tests beyond the cap count as failed so the client's length check holds.
            self._send(200, {"results": res + [0] * padded})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18950)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.code-exec-token"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4),
                    help="concurrent test processes (default: cpus - 4, leaving room for other users)")
    ap.add_argument("--mem-mb", type=int, default=2048, help="address-space limit per test")
    ap.add_argument("--max-tests", type=int, default=64, help="tests run per request; the rest count as failed")
    ap.add_argument("--max-timeout", type=float, default=10.0, help="cap on per-test seconds")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tests-dir", default="",
                    help="root for requests that name their tests by tests_ref (benchmark evals)")
    ap.add_argument("--scratch", default=os.path.join(tempfile.gettempdir(), f"code-exec-{os.getuid()}"))
    ap.add_argument("--stats-every", type=float, default=60.0)
    args = ap.parse_args()

    if not os.path.exists(args.token_file):
        old = os.umask(0o077)
        with open(args.token_file, "w") as f:
            f.write(secrets.token_hex(16) + "\n")
        os.umask(old)
    token = open(args.token_file).read().strip()
    prefix = "/" + token

    runner = Runner(args.workers, args.mem_mb, args.python, args.scratch)
    started = time.time()
    srv = ThreadingHTTPServer((args.host, args.port),
                              make_handler(runner, prefix, args.max_tests, args.max_timeout, started,
                                           make_tests_loader(args.tests_dir)))
    srv.daemon_threads = True

    def report():
        while True:
            time.sleep(args.stats_every)
            with STATS_LOCK:
                s = dict(STATS)
            print(f"[{time.strftime('%F %T')}] requests={s['requests']} tests={s['tests']} "
                  f"passed={s['passed']} timeouts={s['timeouts']} errors={s['errors']} "
                  f"busy={s['busy_s']:.0f}s", flush=True)

    threading.Thread(target=report, daemon=True).start()
    print(f"[{time.strftime('%F %T')}] code-exec on {args.host}:{args.port} workers={args.workers} "
          f"mem={args.mem_mb}MB max_tests={args.max_tests} python={args.python}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
