#!/usr/bin/env python3
"""Stage 1 of the stdio build: one normalised candidate per union problem, no sandbox.

    python3 stdio_cands.py openr1 hardtests ccplus taco --union union.csv.gz --clusters clusters.parquet \
        --extract WORK/extract --taco-verified WORK/refs/taco_verified.jsonl.gz --out-dir WORK/build

For every union pid of the source (one representative per cluster) writes OUT/cands_<source>.jsonl.gz:
    pid, source, cluster, statement (cleaned, see clean_*), tests (cleaned, deduped, <= 200 / 8 MB),
    checker (Python source or None) + checker_kind, refs (Python reference programs: own source first,
    then other copies of the same cluster), wa (Python wrong-answer programs, for the false-positive check),
    flags (text flags), meta (date, rating, platform, url, time limit, the union row, ...).
`taco` also covers the DeepCoder TACO and PrimeIntellect subsets.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sqlite3
import sys

import pandas as pd

from common import clean_tests, decode_hardtests, is_python, load_union, read_jsonl, text_flags

STORE_CAP, STORE_BYTES = 200, 8 << 20
MAX_REFS, MAX_WA = 8, 4
UNION_SOURCES = {"openr1": ("openr1_cf_verifiable",), "hardtests": ("hardtests",), "ccplus": ("ccplus",),
                 "taco": ("taco_verified", "deepcoder_primeintellect", "deepcoder_taco")}


def store(tests):
    out, b = [], 0
    for t in tests[:STORE_CAP]:
        sz = len(t["input"]) + len(t["output"])
        if out and b + sz > STORE_BYTES:
            break
        out.append(t)
        b += sz
    return out


class KV:
    """pid -> JSON value on disk, so a source's tests are never all in memory at once."""

    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT)")

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, json.dumps(v, ensure_ascii=False)))

    def commit(self):
        self.db.commit()

    def get(self, k, default=None):
        row = self.db.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else default


# ---------------------------------------------------------------- statements
_EXP = re.compile(r"(≤|≥|<|>|=|·|×|\*|\\le|\\leq|\\cdot)(\s*)10(1[0-8]|[5-9])(?![\d.,])")
_MOD = re.compile(r"\b10(9|8)\s*\+\s*(7|9|3)\b")


def fix_lost_exponents(s):
    """Codeforces statements scraped without <sup>: '≤ 105' for 10^5, '109 + 7' for 10^9 + 7. Only exponents
    5..18 after a comparison or multiplication sign, where a literal 105..1018 bound is implausible."""
    n0 = s
    s = _EXP.sub(lambda m: f"{m.group(1)}{m.group(2)}10^{m.group(3)}", s)
    s = _MOD.sub(lambda m: f"10^{m.group(1)} + {m.group(2)}", s)
    return s, s != n0


def tex_cf(s):
    """Codeforces' MathJax delimiters ($$$x$$$ inline, $$$$$$x$$$$$$ display) -> standard $x$ / $$x$$."""
    return (s or "").replace("$$$$$$", "$$").replace("$$$", "$")


def examples_block(exs):
    if not exs:
        return ""
    head = "Examples" if len(exs) > 1 else "Example"
    body = "\n\n\n".join(f"Input\n\n{e['input'].rstrip()}\n\n\nOutput\n\n{e['output'].rstrip()}" for e in exs)
    return f"\n\n{head}\n\n{body}"


def clean_openr1(r):
    parts = [tex_cf(r["description"]).strip(), "Input\n\n" + tex_cf(r["input_format"]).strip(),
             "Output\n\n" + tex_cf(r["output_format"]).strip()]
    st = "\n\n".join(parts) + examples_block(r.get("examples") or [])
    if r.get("note"):
        st += "\n\nNote\n\n" + tex_cf(r["note"]).strip()
    fixes = []
    if "$" not in st:
        st, ch = fix_lost_exponents(st)
        if ch:
            fixes.append("v8_lost_exponent_fix")
    return st, fixes


_URL_LINE = re.compile(r"^\s*\[problemUrl\]:.*$\n?", re.M)


def clean_hardtests(text):
    return _URL_LINE.sub("", text or "").strip(), []


_KLEAR_HEAD = re.compile(r"^Solve the following coding problem using the programming language python:\s*")


def clean_plain(text):
    st = _KLEAR_HEAD.sub("", (text or "").strip())
    st, ch = fix_lost_exponents(st)
    return st.strip(), (["v8_lost_exponent_fix"] if ch else [])


# ---------------------------------------------------------------- build
class Builder:
    def __init__(self, a):
        self.a = a
        self.u, _, self.members = load_union(a.union, a.clusters)
        self.rep = self.u.set_index("pid")
        self.all_members = {p for ps in self.members.values() for p in ps}
        self.stats = collections.Counter()
        self.ht_sols, self.cc_sols, self.cc_wa, self.tv_sols = {}, {}, {}, {}
        self.cf_ok, self.cf_wa = self.cf_subs_index()
        self.tv_rows = {}
        for d in read_jsonl(a.taco_verified):
            p = f"taco_verified/{d['id']}"
            if p in self.all_members:
                sols = d.get("solutions")
                if isinstance(sols, str):
                    try:
                        sols = json.loads(sols)
                    except Exception:
                        sols = []
                self.tv_sols[p] = [s for s in (sols or []) if is_python(s)][:MAX_REFS]
            if p in self.rep.index:
                self.tv_rows[p] = d

    def jl(self, name):
        """Rows of one fetched source, shards in file-name order."""
        for path in sorted(glob.glob(os.path.join(self.a.extract, name, "*.jsonl.gz"))):
            yield from read_jsonl(path)

    def cf_subs_index(self):
        wanted = {p.split("/", 1)[1] for ps in self.members.values() for p in ps if p.startswith("openr1_cf/")}
        ok, wa = collections.defaultdict(list), collections.defaultdict(list)
        for r in self.jl("cf_subs"):
            pid = r["problem_id"]
            if pid not in wanted:
                continue
            code = r.get("source") or ""
            if not is_python(code):
                continue
            if r["verdict"] == "OK" and len(ok[pid]) < MAX_REFS:
                ok[pid].append(code)
            elif r["verdict"] == "WRONG_ANSWER" and len(wa[pid]) < MAX_WA and (r.get("passedTestCount") or 0) >= 1:
                wa[pid].append(code)
        return ok, wa

    def ht_load(self):
        if self.ht_sols:
            return
        for r in self.jl("hardtests_problems"):
            sols = [s["code"] for s in (r.get("solutions") or [])
                    if str(s.get("language")).lower() in ("python3", "python") and is_python(s.get("code"))]
            self.ht_sols["hardtests/" + r["pid"]] = sols[:MAX_REFS]

    def cc_load(self):
        if self.cc_sols:
            return
        for r in self.jl("ccplus_1x"):
            p = "ccplus/" + r["id"]
            self.cc_sols[p] = [s["code"] for s in (r.get("correct_submissions") or [])
                               if s.get("language") == "py3" and is_python(s.get("code"))][:MAX_REFS]
            self.cc_wa[p] = [s["code"] for s in (r.get("incorrect_submissions") or [])
                             if s.get("language") == "py3" and is_python(s.get("code"))][:MAX_WA]

    def cluster_refs(self, pid, cl):
        """Own-source references first, then other copies in the cluster."""
        own, other, wa = [], [], []
        for m in [pid] + [x for x in self.members.get(cl, []) if x != pid]:
            bucket = own if m == pid else other
            if m.startswith("openr1_cf/"):
                k = m.split("/", 1)[1]
                bucket += self.cf_ok.get(k, [])
                wa += self.cf_wa.get(k, [])
            elif m.startswith("hardtests/"):
                self.ht_load()
                bucket += self.ht_sols.get(m, [])
            elif m.startswith("ccplus/"):
                self.cc_load()
                bucket += self.cc_sols.get(m, [])
                wa += self.cc_wa.get(m, [])
            elif m in self.tv_sols:
                bucket += self.tv_sols[m]
        seen, refs = set(), []
        for c in own + other:
            if c not in seen:
                seen.add(c)
                refs.append(c)
        return refs[:MAX_REFS * 2], wa[:MAX_WA * 2]

    def emit(self, f, pid, source, statement, tests, checker, checker_kind, meta, fixes):
        cl = int(self.rep.loc[pid, "cluster"])
        refs, wa = self.cluster_refs(pid, cl)
        tests = store(clean_tests(tests))
        union = {k: (None if pd.isna(v) else v) for k, v in self.rep.loc[pid].to_dict().items()}
        rec = {"pid": pid, "source": source, "cluster": cl, "statement": statement, "tests": tests,
               "n_tests_clean": len(tests), "checker": checker, "checker_kind": checker_kind,
               "refs": refs, "wa": wa, "flags": text_flags(statement), "fixes": fixes,
               "meta": {**meta, "union": union}}
        f.write(json.dumps(rec, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        self.stats[(source, "written")] += 1
        self.stats[(source, "has_ref")] += bool(refs)

    def openr1(self, f, pids):
        gen = KV(os.path.join(self.a.out_dir, "kv_openr1.sqlite"))
        for r in self.jl("openr1_tests"):
            p = "openr1_cf/" + r["id"]
            if p in pids:
                gen.put(p, store(clean_tests(r.get("generated_tests") or [])))
        gen.commit()
        for r in self.jl("openr1_verifiable"):
            p = "openr1_cf/" + r["id"]
            if p not in pids:
                continue
            st, fixes = clean_openr1(r)
            off = r.get("official_tests") or []
            g = gen.get(p, [])
            chk = r.get("generated_checker") or None
            self.emit(f, p, "openr1", st, off + g, chk, "openr1_generated" if chk else None,
                      {"date": pd.to_datetime(int(r["contest_start"]), unit="s").strftime("%Y-%m-%d"),
                       "rating": r.get("rating"), "platform": "codeforces", "time_limit": r.get("time_limit"),
                       "cf_id": r["id"], "n_official": len(off), "official_complete": r.get("official_tests_complete"),
                       "n_generated": len(g), "input_mode": r.get("input_mode"),
                       "url": f"https://codeforces.com/contest/{r['contest_id']}/problem/{r['index']}"}, fixes)

    def hardtests(self, f, pids):
        tests = KV(os.path.join(self.a.out_dir, "kv_hardtests.sqlite"))
        for r in self.jl("hardtests_tests"):
            p = "hardtests/" + r["pid"]
            if p not in pids:
                continue
            try:
                tc = r["test_cases"] if isinstance(r["test_cases"], list) else decode_hardtests(r["test_cases"])
            except Exception as e:
                self.stats[("hardtests", f"test_decode_error:{type(e).__name__}")] += 1
                tc = []
            j = (r.get("test_cases_kit") or {}).get("output_judging_function") or None
            tests.put(p, {"tests": store(clean_tests(tc)), "judge": j})
        tests.commit()
        for r in self.jl("hardtests_problems"):
            p = "hardtests/" + r["pid"]
            if p not in pids:
                continue
            st, fixes = clean_hardtests(r["question_content"])
            tj = tests.get(p, {"tests": [], "judge": None})
            self.emit(f, p, "hardtests", st, tj["tests"], tj["judge"], "hardtests_judge" if tj["judge"] else None,
                      {"date": r.get("contest_date"), "platform": r.get("platform"), "url": r.get("url"),
                       "title": r.get("question_title"), "time_limit": r.get("time_limit"),
                       "difficulty": r.get("difficulty_ratings")}, fixes)

    def ccplus(self, f, pids):
        for r in self.jl("ccplus_1x"):
            p = "ccplus/" + r["id"]
            if p not in pids:
                continue
            st, fixes = clean_plain(r.get("description"))
            self.emit(f, p, "ccplus", st, r.get("test_cases") or [], None, None,
                      {"platform": r.get("source"), "title": r.get("title"), "time_limit": r.get("time_limit"),
                       "tpr": r.get("true_positive_rate"), "tnr": r.get("true_negative_rate"),
                       "cc_checker_src": (r.get("checker") or "")[:20000]}, fixes)

    def taco(self, f, pids):
        for p, d in self.tv_rows.items():
            io_ = d.get("input_output")
            io_ = json.loads(io_) if isinstance(io_, str) else (io_ or {})
            if io_.get("fn_name"):
                self.stats[("taco", "fn_style_skipped")] += 1
                continue
            tests = [{"input": i, "output": o} for i, o in zip(io_.get("inputs") or [], io_.get("outputs") or [], strict=False)]
            st, fixes = clean_plain(d.get("question"))
            self.emit(f, p, "taco", st, tests, None, None,
                      {"platform": d.get("source"), "url": d.get("url"), "date": d.get("date"),
                       "difficulty": d.get("difficulty")}, fixes)
        for sub, prefix in (("taco", "deepcoder/taco/"), ("pi", "deepcoder/primeintellect/")):
            want = {p for p in pids if p.startswith(prefix)}
            if not want:
                continue
            for r in self.jl(f"deepcoder_{sub}"):
                # pid: deepcoder/<subset>/<shard index part of the file name>/<row index in the file>
                p = f"{prefix}{r['_file'][-20:-8]}/{r['_row']}"
                if p not in want:
                    continue
                tj = r.get("tests")
                tj = json.loads(tj) if isinstance(tj, str) else tj
                if isinstance(tj, dict):
                    if tj.get("fn_name"):
                        continue
                    tests = [{"input": i, "output": o} for i, o in zip(tj.get("inputs") or [], tj.get("outputs") or [], strict=False)]
                else:
                    tests = [{"input": t.get("input"), "output": t.get("output")} for t in (tj or [])
                             if t.get("type", "stdin_stdout") == "stdin_stdout"]
                sols = r.get("solutions")
                sols = json.loads(sols) if isinstance(sols, str) else (sols or [])
                own = []
                for c in sols:
                    m = re.search(r"```(?:python|py|Python3|python3)?\n(.*?)```", c or "", re.S)
                    c = m.group(1) if m else c
                    if is_python(c):
                        own.append(c)
                self.tv_sols[p] = own[:MAX_REFS]
                st, fixes = clean_plain(r.get("problem"))
                self.emit(f, p, "deepcoder_" + sub, st, tests, None, None, {"platform": None}, fixes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", choices=sorted(UNION_SOURCES))
    ap.add_argument("--union", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--extract", required=True, help="fetch.py output (WORK/extract)")
    ap.add_argument("--taco-verified", required=True, help="build_refs.py output (WORK/refs/taco_verified.jsonl.gz)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    b = Builder(a)
    for s in a.sources:
        pids = set(b.u[b.u.source.isin(UNION_SOURCES[s])].pid)
        out = os.path.join(a.out_dir, f"cands_{s}.jsonl.gz")
        with gzip.open(out + ".tmp", "wt") as f:
            getattr(b, s)(f, pids)
        os.replace(out + ".tmp", out)
        print(s, {k[1]: v for k, v in b.stats.items() if k[0] in (s, "deepcoder_taco", "deepcoder_pi")}, flush=True)


if __name__ == "__main__":
    sys.exit(main())
