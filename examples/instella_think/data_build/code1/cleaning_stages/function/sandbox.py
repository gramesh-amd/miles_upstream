"""Sandboxed batch execution that mirrors the training assert route.

Training grades each assert as its own process: `program + "\\n\\n" + test` run as __main__ with empty
stdin; pass = exit code 0 (code_exec_server.Runner.assert_tests). Spawning one interpreter per
(program, test) pair is too slow for ~170K reference programs, so here one sandboxed interpreter
executes the program once and then fork()s a child per test. Each child starts from the exact
post-program state, which is what a fresh process would see, so cross-test state cannot leak.
Exit semantics follow the grader: SystemExit(0/None) counts as a pass, both inside the program (all
tests pass, the grader's main-guard hole) and inside a test.

Isolation: separate `python3 -I` process in its own session and temp dir, rlimits (CPU, 2 GB address
space, 16 MB files, 64 fds), empty environment, and an audit hook copied from code_exec_server that blocks
sockets, subprocess/exec, signals to other processes, deletes, writes outside the temp dir, and reads
under /home, /root, /mnt, /shared_nfs, /it-share*. fork is allowed only for the harness itself.

run(program, tests, timeout=5.0, seq=False) -> list of per-test dicts {rc, exc[, val]} or None.
A test is either a string (an assert snippet) or {"probe": expr}, which returns repr(eval(expr)).
"""
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

PY = sys.executable
SCRATCH = os.environ.get("V7_SCRATCH", "/tmp/v7fn/scratch")

HARNESS = r'''
import os, sys, builtins, json, signal, time
_SCR = os.path.realpath(os.getcwd())
_NO_READ = ("/home", "/shared_nfs", "/root", "/mnt", "/it-share-prj2-1", "/it-share")
_OWN = tuple({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
_BLOCKED = {
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.__new__", "subprocess.Popen", "os.system", "os.posix_spawn",
    "os.forkpty", "os.kill", "os.killpg", "pty.spawn", "os.remove", "os.unlink", "os.rmdir",
    "os.rename", "os.replace", "os.chmod", "os.chown", "os.truncate", "os.link", "os.symlink",
    "shutil.rmtree", "shutil.move", "shutil.chown", "ctypes.dlsym", "ctypes.call_function",
    "ctypes.addressof", "ctypes.cdata", "ctypes.create_string_buffer", "ctypes.string_at",
    "sys.addaudithook", "sys.setprofile", "sys.settrace", "webbrowser.open",
}
_WF = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_FORK_OK = [False]
def _inside(p, r):
    return p == r or p.startswith(r + os.sep)
def _hook(ev, args, _ok=_FORK_OK):
    if ev == "ctypes.dlopen":
        if args and args[0] is None:
            return
        raise PermissionError("sandbox: dlopen")
    if ev == "os.fork":
        if _ok[0]:
            return
        raise PermissionError("sandbox: fork")
    if ev in _BLOCKED or ev.startswith("os.exec") or ev.startswith("os.spawn"):
        raise PermissionError("sandbox: %s" % ev)
    if ev == "open":
        path, mode, flags = args
        if not isinstance(path, (str, bytes)):
            return
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        real = os.path.realpath(path)
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & _WF)
        if w and not _inside(real, _SCR) and real != "/dev/null":
            raise PermissionError("sandbox: write %s" % real)
        if any(_inside(real, p) for p in _NO_READ) and not _inside(real, _SCR) and not any(_inside(real, p) for p in _OWN):
            raise PermissionError("sandbox: read %s" % real)
with open("case.json") as _f:
    _case = json.load(_f)
_out_fd = os.dup(1)
_dn = os.open("/dev/null", os.O_WRONLY)
os.dup2(_dn, 1); os.dup2(_dn, 2)
sys.addaudithook(_hook)
del _hook
_T = float(_case["timeout"])
def _alarm(*a):
    raise TimeoutError("v7 timeout")
signal.signal(signal.SIGALRM, _alarm)
def _emit(obj):
    os.write(_out_fd, (json.dumps(obj) + "\n").encode())
_g = {"__name__": "__main__", "__file__": "solution.py", "__builtins__": builtins}
_t0 = time.monotonic()
_prog_status = "ok"
try:
    signal.setitimer(signal.ITIMER_REAL, _T)
    exec(compile(_case["program"], "solution.py", "exec"), _g)
except SystemExit as e:
    _prog_status = "exit0" if (e.code is None or e.code == 0) else "exit_nz"
except BaseException as e:
    _prog_status = "error:" + type(e).__name__
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
_used = time.monotonic() - _t0
_emit({"prog": _prog_status, "prog_s": round(_used, 3)})
if _prog_status != "ok":
    os._exit(0)
_budget = max(0.2, _T - _used - 0.05)
def _rc_of(e):
    c = e.code
    if c is None:
        return 0
    if isinstance(c, int):
        return c & 255
    return 1
if _case.get("seq"):
    res = []
    for _t in _case["tests"]:
        r = {"rc": 0, "exc": None}
        try:
            signal.setitimer(signal.ITIMER_REAL, _budget)
            exec(compile(_t, "<test>", "exec"), _g)
        except SystemExit as e:
            r = {"rc": _rc_of(e), "exc": "SystemExit"}
        except BaseException as e:
            r = {"rc": 1, "exc": type(e).__name__}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        res.append(r)
    _emit({"res": res})
    os._exit(0)
res = []
for _t in _case["tests"]:
    rfd, wfd = os.pipe()
    _FORK_OK[0] = True
    try:
        pid = os.fork()
    finally:
        _FORK_OK[0] = False
    if pid == 0:
        os.close(rfd)
        r = {"rc": 0, "exc": None}
        try:
            signal.setitimer(signal.ITIMER_REAL, _budget)
            if isinstance(_t, dict):
                v = eval(compile(_t["probe"], "<probe>", "eval"), _g)
                signal.setitimer(signal.ITIMER_REAL, 0)
                try:
                    r["val"] = repr(v)[:400]
                except BaseException as e:
                    r["val"] = "<repr failed %s>" % type(e).__name__
            else:
                exec(compile(_t, "<test>", "exec"), _g)
                signal.setitimer(signal.ITIMER_REAL, 0)
        except SystemExit as e:
            r = {"rc": _rc_of(e), "exc": "SystemExit"}
        except BaseException as e:
            r = {"rc": 1, "exc": type(e).__name__}
            try:
                r["msg"] = str(e)[:160]
            except BaseException:
                pass
        try:
            os.write(wfd, json.dumps(r).encode()[:2000])
        except BaseException:
            pass
        os._exit(0)
    os.close(wfd)
    deadline = time.monotonic() + _budget + 1.5
    data = b""
    got = None
    while True:
        try:
            p, st = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            break
        if p:
            got = st
            break
        if time.monotonic() > deadline:
            try:
                os.kill(pid, 9)
            except BaseException:
                pass
            os.waitpid(pid, 0)
            break
        time.sleep(0.002)
    try:
        while True:
            b = os.read(rfd, 4096)
            if not b:
                break
            data += b
    except BaseException:
        pass
    os.close(rfd)
    try:
        r = json.loads(data.decode()) if data else {"rc": 1, "exc": "Crash" if got is not None else "Timeout"}
    except BaseException:
        r = {"rc": 1, "exc": "Crash"}
    res.append(r)
_emit({"res": res})
os._exit(0)
'''


def _limits(cpu):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (4096, 4096))
    return apply


_HPATH = None


def _harness_path():
    global _HPATH
    if _HPATH is None:
        os.makedirs(SCRATCH, exist_ok=True)
        p = os.path.join(SCRATCH, f"_harness_{os.getpid()}.py")
        with open(p, "w") as f:
            f.write(HARNESS)
        _HPATH = p
    return _HPATH


def run(program, tests, timeout=5.0, seq=False):
    """Returns {"prog": status, "res": [...]} ; on a harness crash, all tests fail."""
    if not tests:
        return {"prog": "ok", "res": []}
    d = tempfile.mkdtemp(prefix="t-", dir=SCRATCH if os.path.isdir(SCRATCH) else None)
    try:
        with open(os.path.join(d, "case.json"), "w") as f:
            json.dump({"program": program, "tests": tests, "timeout": timeout, "seq": seq}, f)
        # wall budget: program + every test at most `timeout`, plus slack
        wall = timeout * (len(tests) + 1) + 10
        p = subprocess.Popen([PY, "-I", _harness_path()], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, cwd=d, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                                                                     "HOME": d, "TMPDIR": d,
                                                                     "PYTHONHASHSEED": "0",
                                                                     "OMP_NUM_THREADS": "1"},
                             preexec_fn=_limits(int(timeout) + 2), close_fds=True)
        try:
            out, _ = p.communicate(timeout=wall)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, _ = p.communicate()
        prog, res = None, None
        for line in out.decode("utf-8", "replace").splitlines():
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "prog" in o:
                prog = o["prog"]
            if "res" in o:
                res = o["res"]
        if prog is None:
            # died before reporting: os._exit inside the program, a hard crash, or a kill
            prog = "exit0_hard" if p.returncode == 0 else "crash"
        if res is None:
            ok = prog in ("exit0", "exit0_hard")
            res = [{"rc": 0 if ok else 1, "exc": "ProgExit" if ok else prog} for _ in tests]
        if len(res) < len(tests):
            res = res + [{"rc": 1, "exc": "Missing"}] * (len(tests) - len(res))
        return {"prog": prog, "res": res}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def passvec(out):
    return [1 if r["rc"] == 0 else 0 for r in out["res"]]


if __name__ == "__main__":
    prog = "def f(x):\n    return x * 2\n"
    tests = ["assert f(2) == 4", "assert f(3) == 7", "assert g(1) == 2", "import sys; sys.exit(0)",
             "while True: pass", {"probe": "f(21)"}, "x = 1", "assert x == 1"]
    print(run(prog, tests, timeout=2))
    print(run(prog, tests[:2] + tests[6:], timeout=2, seq=True))
    print(run("import sys\nif __name__ == '__main__':\n    sys.exit(0)\n", ["assert False"], timeout=2))
    print(run("import os\nos._exit(0)\n", ["assert False"], timeout=2))
    print(run("import socket\nsocket.socket()\n", ["assert False"], timeout=2))
    print(run("import os\nos.fork()\n", ["assert True"], timeout=2))
    print(run("open('/home/x')\n", ["assert True"], timeout=2))
    print(run("x=[0]*(10**10)\n", ["assert True"], timeout=2))
