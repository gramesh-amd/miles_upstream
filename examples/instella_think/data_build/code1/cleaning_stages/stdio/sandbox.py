"""Sandboxed execution with the grader's own isolation (code_exec_server.Runner: fresh `python3 -I`
per test, audit-hook guard, rlimits on CPU/AS/FSIZE/NOFILE, temp dir, no network), fanned out over a
multiprocessing pool. Each job = one program against a list of tests; per test we keep enough to
re-judge offline under several comparison rules (exact grader rule, case-fold, float tolerance,
whitespace tokens) without re-running.
"""
import hashlib, json, math, multiprocessing as mp, os, sys, tempfile, time
from decimal import Decimal, InvalidOperation

sys.path.insert(0, "/tmp/wt_lux/examples/instella_think")
import code_exec_server as ces  # noqa: E402

# Wall cap is above the grader's 5 s because the host is shared (load ~35); elapsed time is kept
# so "slow" (> 5 s) is judged offline.
TIMEOUT = 10.0
_runner = None


def _init():
    global _runner
    scratch = os.path.join(tempfile.gettempdir(), f"v7stdio-{os.getuid()}-{os.getpid()}")
    _runner = ces.Runner(workers=1, mem_mb=2048, python=sys.executable, scratch_root=scratch)


def grader_eq(out, exp):
    return ces._lines_equal_with_decimal(out, exp)


def _lines(s):
    return [x.strip() for x in s.strip().split("\n")]


def case_eq(out, exp):
    return grader_eq(out.lower(), exp.lower())


def ws_eq(out, exp):
    """token-equal ignoring all whitespace layout inside lines (line count must match)"""
    a, b = _lines(out), _lines(exp)
    return len(a) == len(b) and all(x.split() == y.split() for x, y in zip(a, b))


def tokens_eq(out, exp):
    """token-equal ignoring line structure entirely"""
    return out.split() == exp.split()


def float_eq(out, exp, tol=1e-6):
    a, b = _lines(out), _lines(exp)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        xs, ys = x.split(), y.split()
        if len(xs) != len(ys):
            return False
        for p, q in zip(xs, ys):
            if p == q:
                continue
            try:
                fp, fq = float(p), float(q)
            except ValueError:
                return False
            if not (math.isfinite(fp) and math.isfinite(fq)):
                return False
            if abs(fp - fq) > tol * max(1.0, abs(fq)):
                return False
    return True


def judge(out, exp):
    return dict(ex=grader_eq(out, exp), cs=case_eq(out, exp), ws=ws_eq(out, exp),
                tk=tokens_eq(out, exp), fl=float_eq(out, exp))


def run_job(job):
    """job: dict(key, code, kind stdio|assert, tests, abort_after=2)
    returns dict(key, res=[{rc, t, ex, cs, ws, tk, fl, out}])  (assert kind: ex only)"""
    code, kind, tests = job["code"], job["kind"], job["tests"]
    res, n_to = [], 0
    for tc in tests:
        if n_to >= job.get("abort_after", 2):
            res.append(dict(rc="skip", t=0.0, ex=False))
            continue
        t0 = time.monotonic()
        if kind == "stdio":
            rc, out = _runner.run_one(code, str(tc["input"]), TIMEOUT)
        else:
            rc, out = _runner.run_one(f"{code}\n\n{tc}\n", "", TIMEOUT)
        dt = time.monotonic() - t0
        if rc is None:
            n_to += 1
            res.append(dict(rc="to", t=dt, ex=False))
            continue
        if kind == "stdio":
            r = judge(out, str(tc["output"])) if rc == 0 else dict(ex=False)
            r.update(rc=rc, t=round(dt, 3))
            if not r["ex"]:
                r["out"] = out[:300]
                r["h"] = hashlib.md5(" ".join(out.split()).encode()).hexdigest()[:12]
        else:
            r = dict(rc=rc, t=round(dt, 3), ex=rc == 0)
        res.append(r)
    return dict(key=job["key"], res=res)


def run_row(job):
    """job: dict(row, refs=[(ref_idx, code)], segs=[(name, kind, tests)], need=2)
    Runs references in order; stops once `need` references pass every test of a target segment
    (orig / to_stdin / to_assert). Returns {(row, ref_idx, kind): res}."""
    out = {}
    n_ok = 0
    for j, code in job["refs"]:
        ok = False
        for kind in ("stdio", "assert"):
            segs = [s for s in job["segs"] if s[1] == kind]
            if not segs:
                continue
            tests = [t for s in segs for t in s[2]]
            res = run_job(dict(key=None, code=code, kind=kind, tests=tests))["res"]
            out[(job["row"], j, kind)] = res
            off = 0
            for name, _, ts in segs:
                seg = res[off: off + len(ts)]
                off += len(ts)
                if name in ("orig", "to_stdin", "to_assert") and all(r["ex"] for r in seg):
                    ok = True
        n_ok += ok
        if n_ok >= job.get("need", 2):
            break
    return out


def run_rows(jobs, procs=24):
    with mp.Pool(procs, initializer=_init) as pool:
        yield from pool.imap_unordered(run_row, jobs, chunksize=1)


def run_all(jobs, procs=30, log_every=2000):
    out = []
    t0 = time.time()
    with mp.Pool(procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            out.append(r)
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{len(jobs)} jobs  {time.time() - t0:.0f}s", flush=True)
    return out
