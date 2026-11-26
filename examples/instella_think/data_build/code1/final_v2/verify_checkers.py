#!/usr/bin/env python3
"""Verify the Open-R1 generated checkers for the v7 multi-answer rows on a lux node (no GPU needed). Stdlib only.

Starts its own grader server (code_exec_server.py, copied here from instella-code-rl-lux-v7) on 127.0.0.1 with many
workers and its own scratch dir, then for every candidate in checker_candidates.jsonl.gz, through
/test_program_checker only (dataset checkers and reference solutions are untrusted and never run outside it):

  mapping   the winner's prepare.py mapping check must be ok (shared sample/test, constraint numbers, title, time
            limit); A1 below (the jury answer accepted on every test) re-checks the mapping against the checker
  verify    references in order (v6, OLMo, accepted CF submissions, v6 near-miss; up to 10) on own + official tests,
            first one passing every test wins; otherwise the first reference passing the winner's own tests keeps
            the official/merged tests it passes
  accept    A1 label as contestant output accepted on every test
            A2 every finished output of up to 3 accepted CF submissions (alternative answers included), and of our
               references whose output equals the label on every test, is accepted on every test; outputs of our
               other references (OLMo samples can be wrong) are reported only
  reject    R1 empty output (tests with a non-empty label), R2 the most common label output printed everywhere
            (tests whose label differs from it), R3 the label with one token changed (int +1 / float +1 / YES<->NO /
            suffix): each rejected on every test. Wrong CF submissions are reported, not a gate.
  rows      >= 5 tests; the R2 constant program must not pass every test; a final_v1 duplicate (CF id, S7 cluster or
            MinHash) is replaced only by a row with strictly more tests

Outputs in --out-dir: accepted_rows.jsonl (final_v2 train schema, metadata.checker set), results.jsonl (evidence
per CF problem), report.md, report.json. Merge with merge_checker_rows.py.

    python3 verify_checkers.py --workers 100 --out-dir /tmp/v7_checkers   (see ../../final_v1/LUX_PLAN.md and the
    repo's examples/instella_think/HANDOFF.md)

code_exec_server.py here must be byte-identical to instella-code-rl-lux-v7's examples/instella_think/code_exec_server.py:
the script refuses to start if its sha256 is not SERVER_SHA256 (re-copy it and update the constant), and the
repo's test_grader.sh fails if the two files differ.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 5.0          # CODE_EXEC_TIMEOUT used in training
CHUNK = 32
MAX_REFS = 10
CUTOFF = "2024-08-01"
URL = None
SERVER_SHA256 = "def246c0b34f4e00eb8946c0915c999a70d8fa8d6f8bc39243e425e8775bc2af"
COMPLETED_PROBE = "print(1)\n"   # our own checker: accepts anything, i.e. 1 iff the program finished cleanly


def post(path, body, n):
    data = json.dumps(body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=(TIMEOUT + 10) * n + 120) as r:
                res = json.loads(r.read())["results"]
            if len(res) == n:
                return res
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def run(route, program, tests, checker=None, stop_on_fail=False):
    out = []
    for i in range(0, len(tests), CHUNK):
        chunk = tests[i:i + CHUNK]
        body = {"program": program, "tests": [{"input": str(t["input"]), "output": str(t["output"])} for t in chunk],
                "max_execution_time": TIMEOUT, "stop_on_fail": stop_on_fail}
        if route == "checker" and checker:
            body["checker"] = checker
        res = post("/test_program_checker" if route == "checker" else "/test_program_stdio", body, len(chunk))
        if res is None:
            return None
        out += res
        if stop_on_fail and not all(res):
            out += [0] * (len(tests) - len(out))
            break
    return out


def table_program(tests, outputs):
    table = {}
    for t, o in zip(tests, outputs):
        table[hashlib.sha1(" ".join(str(t["input"]).split()).encode("utf-8", "replace")).hexdigest()] = o
    return ("import sys, hashlib\nT = " + json.dumps(table) + "\n"
            "k = hashlib.sha1(' '.join(sys.stdin.read().split()).encode('utf-8', 'replace')).hexdigest()\n"
            "sys.stdout.write(T.get(k, ''))\n")


def perturb(out):
    s = str(out)
    m = re.search(r"(?<![\w.])-?\d+(?![\w.])", s)
    if m:
        return s[:m.start()] + str(int(m.group(0)) + 1) + s[m.end():]
    m = re.search(r"(?<![\w.])-?\d+\.\d+(?![\w.])", s)
    if m:
        d = len(m.group(0).split(".")[1])
        return s[:m.start()] + f"{float(m.group(0)) + 1:.{d}f}" + s[m.end():]
    m = re.search(r"\S+", s)
    if not m:
        return "0\n"
    flip = {"YES": "NO", "NO": "YES", "Yes": "No", "No": "Yes", "yes": "no", "no": "yes"}
    return s[:m.start()] + flip.get(m.group(0), m.group(0) + "x") + s[m.end():]


def verify(c):
    chk = c["checker"]
    own = c["own_tests"][:c["n_win_own"]]
    extra = c["own_tests"][c["n_win_own"]:] + c["official"]
    own_pass, tried = None, []
    for j, ref in enumerate(c["refs"][:MAX_REFS]):
        res = run("checker", ref["code"], own + extra, chk, stop_on_fail=True)
        if res is None:
            return dict(verified=False, why="grader_unreachable")
        tried.append(ref["origin"])
        if all(res):
            return dict(verified=True, ref=j, tests=own + extra, tried=tried, n_extra_dropped=0)
        if own_pass is None and all(res[:len(own)]):
            own_pass = j
    if own_pass is None:
        return dict(verified=False, why="no_reference_passes", tried=tried)
    res = run("checker", c["refs"][own_pass]["code"], extra, chk) if extra else []
    if res is None:
        return dict(verified=False, why="grader_unreachable")
    kept = [t for t, ok in zip(extra, res) if ok]
    return dict(verified=True, ref=own_pass, tests=own + kept, tried=tried, n_extra_dropped=len(extra) - len(kept))


def evidence(c, tests, ver_ref):
    chk = c["checker"]
    labels = [t["output"] for t in tests]
    common = collections.Counter(" ".join(o.split()) for o in labels).most_common(1)[0][0]
    ev = dict(self=run("checker", table_program(tests, labels), tests, chk),
              neg_empty=run("checker", table_program(tests, [""] * len(tests)), tests, chk),
              neg_constant=run("checker", table_program(tests, [common + "\n"] * len(tests)), tests, chk),
              neg_constant_differs=[" ".join(o.split()) != common for o in labels],
              neg_token=run("checker", table_program(tests, [perturb(o) for o in labels]), tests, chk),
              positives=[], wrong=[])
    ours = [r for i, r in enumerate(c["refs"][:MAX_REFS]) if r["origin"] != "cf_ok" and i != ver_ref][:3]
    cfok = [r for i, r in enumerate(c["refs"]) if r["origin"] == "cf_ok" and i != ver_ref][:3]
    for r in ours + cfok:
        ev["positives"].append(dict(origin=r["origin"], completed=run("checker", r["code"], tests, COMPLETED_PROBE),
                                    checker=run("checker", r["code"], tests, chk),
                                    exact=run("stdio", r["code"], tests)))
    for w in c["cf_wa"][:3]:
        ev["wrong"].append(dict(checker=run("checker", w["code"], tests, chk), exact=run("stdio", w["code"], tests)))
    return ev


def verdict(ev):
    why = []
    if not ev["self"] or not all(ev["self"]):
        why.append("A1_rejects_jury_answer")
    alt = alt_ok = bad = untrusted_rej = 0
    for p in ev["positives"]:
        # trusted: accepted CF submissions, and our references whose output equals the label on every test
        trusted = p["origin"] == "cf_ok" or (p["exact"] is not None and all(p["exact"]))
        for done, k, e in zip(p["completed"] or [], p["checker"] or [], p["exact"] or []):
            if not done:
                continue
            if trusted:
                bad += not k
                if not e:
                    alt += 1
                    alt_ok += k
            else:
                untrusted_rej += not k
    if bad:
        why.append("A2_rejects_reference_output")
    if ev["neg_empty"] is None or any(ev["neg_empty"]):
        why.append("R1_accepts_empty")
    if ev["neg_constant"] is None or any(a and d for a, d in zip(ev["neg_constant"], ev["neg_constant_differs"])):
        why.append("R2_accepts_constant")
    if ev["neg_token"] is None or any(ev["neg_token"]):
        why.append("R3_accepts_one_token_change")
    wa_rej = sum(1 for w in ev["wrong"] if w["checker"] is not None and not all(w["checker"]))
    return why, dict(alternative_outputs=alt, alternative_accepted=alt_ok, reference_outputs_rejected=bad,
                     untrusted_reference_outputs_rejected=untrusted_rej,
                     wrong_subs=len(ev["wrong"]), wrong_subs_rejected=wa_rej)


def job(c):
    t0 = time.time()
    out = dict(pid=c["pid"], custom_id=c["custom_id"], copies=c["copies"])
    if not c["mapping"]["ok"]:
        out["drop"] = "ambiguous_mapping"
    elif not c["date"] or c["date"] >= CUTOFF:
        out["drop"] = "date_after_cutoff"
    elif c["v1_dups"] and len(c["own_tests"]) + len(c["official"]) <= c["v1_dup_max_tests"]:
        out["drop"] = "duplicate_of_final_v1"
    if out.get("drop"):
        return out, None
    v = verify(c)
    out["verify"] = {k: v.get(k) for k in ("verified", "why", "ref", "tried", "n_extra_dropped")}
    if not v["verified"]:
        out["drop"] = "unverified"
        return out, None
    tests = v["tests"]
    ev = evidence(c, tests, v["ref"])
    why, st = verdict(ev)
    out.update(checker_why=why, checker_stats=st, n_tests=len(tests))
    if why:
        out["drop"] = "checker_rejected:" + "+".join(why)
    elif len(tests) < 5:
        out["drop"] = "lt5_tests"
    elif all(ev["neg_constant"][:32]):
        out["drop"] = "constant_output_passes_all"
    out["secs"] = round(time.time() - t0, 1)
    if out.get("drop"):
        return out, None
    ref = c["refs"][v["ref"]]
    md = dict(c["metadata"])
    n_off = sum(1 for t in tests if t in c["official"])
    changes = ["openr1_checker"] + (["openr1_statement"] if c["image_copy"] and "checker" not in c["tasks"] else []) \
        + (["openr1_official_tests"] if n_off else [])
    md.update(rm_type="code", split="train", n_tests=len(tests), custom_id=c["custom_id"], verified=True,
              verifier_origins=[ref["origin"]], src=c["src"], route="stdio", v7_changes=sorted(changes),
              fixes=["attach_openr1_generated_checker"] + ([f"add_openr1_official_tests:{n_off}"] if n_off else []),
              lcb_v3_flag="2023-05-01" <= c["date"] < CUTOFF, openr1_id=c["pid"], cf_date=c["date"],
              cf_rating=c["rating"], checker=c["checker"], checker_source="open-r1/codeforces generated_checker",
              const_floor=round(sum(ev["neg_constant"][:32]) / len(tests[:32]), 3), replaces_final_v1=c["v1_dups"])
    row = {"prompt": [{"role": "user", "content": c["prompt"]}], "label": json.dumps(tests, ensure_ascii=False),
           "metadata": md}
    return out, dict(row=row, ref=dict(kind="checker", opts={}, prog=ref["code"], origin=ref["origin"]))


def main():
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(HERE, "checker_candidates.jsonl.gz"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=100, help="grader processes")
    ap.add_argument("--concurrency", type=int, default=0, help="concurrent requests (default workers // 2)")
    ap.add_argument("--port", type=int, default=18999)
    ap.add_argument("--python", default=sys.executable, help="interpreter the grader runs programs with")
    ap.add_argument("--scratch", default="", help="grader scratch dir (default <out-dir>/scratch, removed at the end)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with open(os.path.join(HERE, "code_exec_server.py"), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != SERVER_SHA256:
        sys.exit(f"FATAL: {HERE}/code_exec_server.py has sha256 {sha}, expected {SERVER_SHA256}. Copy the server "
                 "from instella-code-rl-lux-v7 and update SERVER_SHA256.")
    os.makedirs(a.out_dir, exist_ok=True)
    scratch = a.scratch or os.path.join(a.out_dir, "scratch")
    tok = os.path.join(a.out_dir, "grader.token")
    old = os.umask(0o077)
    with open(tok, "w") as f:
        f.write(secrets.token_hex(16) + "\n")
    os.umask(old)
    srv = subprocess.Popen([a.python, os.path.join(HERE, "code_exec_server.py"), "--host", "127.0.0.1", "--port",
                            str(a.port), "--workers", str(a.workers), "--token-file", tok, "--python", a.python,
                            "--scratch", scratch], stdout=open(os.path.join(a.out_dir, "server.log"), "w"),
                           stderr=subprocess.STDOUT)
    URL = f"http://127.0.0.1:{a.port}/" + open(tok).read().strip()
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(URL + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        cands = [json.loads(l) for l in gzip.open(a.cands, "rt")]
        if a.limit:
            cands = cands[:a.limit]
        t0 = time.time()
        results, rows = [], []
        with ThreadPoolExecutor(a.concurrency or max(1, a.workers // 2)) as ex:
            for i, (out, row) in enumerate(ex.map(job, cands)):
                results.append(out)
                if row:
                    rows.append(row)
                if (i + 1) % 20 == 0:
                    print(f"{i + 1}/{len(cands)} accepted={len(rows)} {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(a.out_dir, "accepted_rows.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(a.out_dir, "results.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        reasons = collections.Counter((r.get("drop") or "accepted").split(":")[0] for r in results)
        fails = collections.Counter(w for r in results for w in r.get("checker_why", []))
        st = [r["checker_stats"] for r in results if "checker_stats" in r]
        rep = dict(candidates=len(cands), rows_covered=sum(len(c["copies"]) for c in cands), accepted=len(rows),
                   outcomes=dict(reasons), checker_fail_reasons=dict(fails), checkers_evaluated=len(st),
                   alternative_outputs=sum(s["alternative_outputs"] for s in st),
                   alternative_accepted=sum(s["alternative_accepted"] for s in st),
                   wrong_subs=sum(s["wrong_subs"] for s in st), wrong_subs_rejected=sum(s["wrong_subs_rejected"] for s in st),
                   wall_s=round(time.time() - t0), workers=a.workers)
        json.dump(rep, open(os.path.join(a.out_dir, "report.json"), "w"), indent=1)
        with open(os.path.join(a.out_dir, "report.md"), "w") as f:
            f.write("# Open-R1 checker verification (lux)\n\n")
            for k, v in rep.items():
                f.write(f"- **{k}**: {v}\n")
        print(json.dumps(rep, indent=1))
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
        shutil.rmtree(scratch, ignore_errors=True)
        os.remove(tok)


if __name__ == "__main__":
    main()
