#!/usr/bin/env python3
"""Code-execution service for the code RL reward (code_reward.py).

Implements the API the Instella reward client already speaks:

    POST <prefix>/test_program        {"program", "tests": [str], "max_execution_time"}
    POST <prefix>/test_program_stdio  {"program", "tests": [{"input","output"}], "max_execution_time"}
    POST <prefix>/test_program_call   {"program", "tests": [{"input","output"}], "fn_name", ...}
        -> {"results": [0|1, ...]}   one entry per test, in order
    GET  <prefix>/health              -> {"ok": true, ...}

Optional request fields, used by the benchmark evals (prepare_code_benchmarks.py):
    "tests_ref": path under --tests-dir, sent instead of "tests"
    "stop_on_fail": skip (and fail) tests not yet started once one has failed
    "name": the program's __name__ (default "__main__")
/test_program_call follows LiveCodeBench's call-based grading: the program is imported as
a module and fn_name (on Solution() if the code defines it) is called per test.

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
import functools
import json
import os
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
sys.argv = [_prog]
_code = compile(_src, "solution.py", "exec")
_g = {"__name__": os.environ.get("CODE_EXEC_NAME", "__main__"), "__file__": "solution.py",
      "__builtins__": builtins}
exec(_code, _g)
'''

STATS_LOCK = threading.Lock()
STATS = {"requests": 0, "tests": 0, "passed": 0, "timeouts": 0, "errors": 0, "busy_s": 0.0}


def _bump(**kw):
    with STATS_LOCK:
        for k, v in kw.items():
            STATS[k] += v


def _lines_equal_with_decimal(a: str, b: str) -> bool:
    """Same comparison as the Instella reward's local path: per-line, whitespace
    stripped, with a Decimal fallback so 1.0 == 1 == 1.00."""
    a_lines = [ln.strip() for ln in a.strip().split("\n")]
    b_lines = [ln.strip() for ln in b.strip().split("\n")]
    if len(a_lines) != len(b_lines):
        return False
    for al, bl in zip(a_lines, b_lines):
        if al == bl:
            continue
        try:
            if [Decimal(t) for t in al.split()] != [Decimal(t) for t in bl.split()]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


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
        self.guard = os.path.join(scratch_root, "_guard.py")
        with open(self.guard, "w") as f:
            f.write(GUARD_SOURCE)
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

    def run_one(self, program: str, stdin: str, timeout: float, name: str = "__main__", case=None):
        """Returns (returncode or None on timeout, stdout). `name` is the program's
        __name__; `case`, if given, is written to _case.json beside it."""
        with self.sem:
            t0 = time.monotonic()
            d = tempfile.mkdtemp(prefix="t-", dir=self.scratch_root)
            try:
                prog = os.path.join(d, "solution.py")
                with open(prog, "w") as f:
                    f.write(program)
                if case is not None:
                    with open(os.path.join(d, "_case.json"), "w") as f:
                        json.dump(case, f)
                env = dict(self.env, HOME=d, TMPDIR=d, CODE_EXEC_SCRATCH=d, CODE_EXEC_NAME=name)
                p = subprocess.Popen(
                    [self.python, "-I", self.guard, prog],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    cwd=d, env=env, preexec_fn=self._limits(timeout), close_fds=True,
                )
                try:
                    out, _ = p.communicate(stdin.encode("utf-8", "replace"), timeout=timeout)
                    return p.returncode, out.decode("utf-8", "replace")
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.communicate()
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
            r = one(tc)
            if not r:
                failed.set()
            return r

        return list(self.pool.map(guarded, tests))

    def assert_tests(self, program, tests, timeout, name="__main__", stop_on_fail=False):
        def one(tc):
            rc, _ = self.run_one(f"{program}\n\n{tc}\n", "", timeout, name=name)
            return 1 if rc == 0 else 0

        return self._map(one, tests, stop_on_fail)

    def stdio_tests(self, program, tests, timeout, stop_on_fail=False):
        def one(tc):
            if not isinstance(tc, dict):
                return 0
            rc, out = self.run_one(program, str(tc.get("input", "")), timeout)
            return 1 if rc == 0 and _lines_equal_with_decimal(out, str(tc.get("output", ""))) else 0

        return self._map(one, tests, stop_on_fail)

    def call_tests(self, program, tests, timeout, fn_name, stop_on_fail=False):
        """LiveCodeBench call-based grading (lcb_runner testing_util.grade_call_based): the
        program is imported, not run as __main__; inputs are one JSON value per line."""
        target = "Solution()" if "class Solution" in program else None
        program = program + "\n\n" + CALL_HARNESS.format(fn=fn_name, target=target)

        def one(tc):
            if not isinstance(tc, dict):
                return 0
            case = {"input": str(tc.get("input", "")), "output": str(tc.get("output", ""))}
            rc, _ = self.run_one(program, "", timeout, name="tmp_sol", case=case)
            return 1 if rc == 0 else 0

        return self._map(one, tests, stop_on_fail)


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

    @functools.lru_cache(maxsize=8)
    def load(ref):
        path = os.path.realpath(os.path.join(root, ref))
        if not path.startswith(root + os.sep):
            raise ValueError("tests_ref escapes --tests-dir")
        with open(path) as f:
            tests = json.load(f)
        if not isinstance(tests, list):
            raise ValueError("tests_ref file must hold a JSON list")
        return tests

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
            if route not in ("/test_program", "/test_program_stdio", "/test_program_call"):
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
            except (OSError, ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            total = len(tests)
            if total > max_tests:
                tests = tests[:max_tests]
            padded = total - len(tests)
            if route == "/test_program":
                res = runner.assert_tests(program, [str(t) for t in tests], timeout, name, stop_on_fail)
            elif route == "/test_program_stdio":
                res = runner.stdio_tests(program, tests, timeout, stop_on_fail)
            else:
                res = runner.call_tests(program, tests, timeout, fn_name, stop_on_fail)
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
