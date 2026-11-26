#!/usr/bin/env python3
"""Execution check of function-style candidates (HE+, MBPP+, HE-Pro, MBPP-Pro).

For each candidate (train row, eval problem):
  A = fraction of the TRAIN row's tests passed by the EVAL canonical solution
      (A == 1 -> the training reward is maximised by exactly the eval behaviour: same problem)
  B = does a training reference solution (OLMo sample from Dolci `outputs`, or our v6 rollout) pass the EVAL tests
Untrusted code runs in a subprocess: no network (unshare -rn), temp dir, CPU/AS/FSIZE/NOFILE limits, timeout.
Output: work/exec_verify.parquet
"""
import glob, json, os, pickle, re, resource, site, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pyarrow.parquet as pq

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/work"
EB = "/tmp/evalbuild/raw"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import parse_assert  # noqa: E402

PRE = "from typing import *\nimport math, re, collections, itertools, functools, heapq, string, sys\n" \
      "from collections import *\nfrom itertools import *\nfrom functools import *\n"


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


USERSITE = site.getusersitepackages()  # numpy for the HE+/MBPP+ harnesses lives here; -I would hide it


def run(code, timeout=30):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.py")
        open(p, "w").write(f"import sys\nsys.path.append({USERSITE!r})\n" + code)
        try:
            r = subprocess.run(["unshare", "-rn", sys.executable, "-I", p], cwd=d, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=limits, env={"PATH": "/usr/bin:/bin", "HOME": d})
            return r.returncode, r.stdout[-4000:], r.stderr[-1500:]
        except subprocess.TimeoutExpired:
            return -9, "", "timeout"


def count_asserts_program(solution, asserts, alias):
    body = [PRE, solution, alias, "__n=0", "__ok=0"]
    for a in asserts:
        body.append("__n+=1\ntry:\n    " + a.replace("\n", "\n    ") + "\n    __ok+=1\nexcept BaseException:\n    pass")
    body.append("print('RESULT', __ok, __n)")
    return "\n".join(body)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def main():
    FSETS = ["humanevalplus", "mbppplus", "humanevalpro", "mbpppro"]
    cand = pd.read_parquet(f"{W}/candidates.parquet")
    cand = cand[cand.eval_set.isin(FSETS)][["ti", "ei", "custom_id", "eval_set", "eval_id", "part"]].copy()
    cand["origin"] = "candidate"
    train = pickle.load(open(f"{W}/train.pkl", "rb"))
    ev = pickle.load(open(f"{W}/eval.pkl", "rb"))
    # execution-based paraphrase search: top-K TF-IDF neighbours of every function-style eval item
    from detect import tfidf, tok
    import numpy as np
    K = int(os.environ.get("TOPK", 10))
    tis = [i for i, t in enumerate(train) if any(isinstance(a, str) for a in t["tests_label"])]
    eis = [j for j, e in enumerate(ev) if e["eval_set"] in FSETS]
    M = tfidf([tok(train[i]["text"]) for i in tis] + [tok(ev[j]["text"]) for j in eis])
    S = (M[len(tis):] @ M[:len(tis)].T).toarray()
    extra = []
    for a, j in enumerate(eis):
        for b in np.argsort(-S[a])[:K]:
            i = tis[b]
            extra.append(dict(ti=i, ei=j, custom_id=train[i]["custom_id"], eval_set=ev[j]["eval_set"],
                              eval_id=ev[j]["eval_id"], part=ev[j]["part"], origin=f"tfidf_top{K}"))
    extra = pd.DataFrame(extra)
    cand = pd.concat([cand, extra]).drop_duplicates(["ti", "ei"], keep="first")
    prev_path = f"{W}/exec_verify.parquet"
    prev = None
    if os.environ.get("RERUN_SUBSET") and os.path.exists(prev_path):
        prev = pd.read_parquet(prev_path)
        keep = prev[(prev.A >= 0.5) | (prev.origin == "candidate")][["ti", "ei"]]
        cand = cand.merge(keep, on=["ti", "ei"])
    print("exec jobs", len(cand), cand.origin.value_counts().to_dict(), flush=True)
    he = pd.read_parquet(f"{EB}/humanevalplus.parquet").set_index("task_id")
    mb = pd.read_parquet(f"{EB}/mbppplus.parquet")
    mb["eid"] = "mbppplus/" + mb.task_id.astype(str)
    mb = mb.set_index("eid")
    pro = {n: pd.read_parquet(f"{EB}/{n}.parquet").assign(eid=lambda d: n + "/" + d.id.astype(str)).set_index("eid")
           for n in ("humanevalpro", "mbpppro")}

    need = set(cand.custom_id)
    outs = {}
    for f in sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--allenai--Dolci-Think-RL-7B/snapshots/*/data/*.parquet"))):
        t = pq.read_table(f, columns=["custom_id", "outputs"]).to_pandas()
        t = t[t.custom_id.isin(need)]
        for cid, o in zip(t.custom_id, t.outputs):
            outs[cid] = [x for x in (o if o is not None else [])]

    def eval_solution(r):
        if r.eval_set == "humanevalplus":
            x = he.loc[r.eval_id]
            return x.prompt + x.canonical_solution, x.entry_point, x.test + f"\ncheck({x.entry_point})\nprint('EVALPASS')"
        if r.eval_set == "mbppplus":
            x = mb.loc[r.eval_id]
            names = [k[0] for k in (parse_assert(a) for a in list(x.test_list)) if k]
            return x.code, names[0] if names else None, x.test + "\nprint('EVALPASS')"
        x = pro[r.eval_set].loc[r.eval_id]
        sol = x.raw_problem + x.raw_solution + "\n" + x.new_problem + x.new_solution
        names = [k[0] for k in (parse_assert(a) for a in x.test_code.split("\n") if a.strip().startswith("assert")) if k]
        entry = names[0] if names else None
        if r.part == "raw":
            m = re.search(r"def\s+(\w+)\s*\(", x.raw_problem)
            return sol, m.group(1) if m else None, None
        return sol, entry, x.test_code + "\nprint('EVALPASS')"

    jobs = []
    for r in cand.itertuples():
        t = train[r.ti]
        asserts = [a for a in t["tests_label"] if isinstance(a, str)]
        tnames = []
        for a in asserts:
            k = parse_assert(a)
            if k and k[0] not in tnames:
                tnames.append(k[0])
        sol, entry, etest = eval_solution(r)
        jobs.append((r, t, asserts, tnames, sol, entry, etest))

    def work(job):
        r, t, asserts, tnames, sol, entry, etest = job
        res = dict(ti=r.ti, ei=r.ei, custom_id=r.custom_id, eval_set=r.eval_set, eval_id=r.eval_id, part=r.part,
                   origin=r.origin, eval_entry=entry, train_func=",".join(tnames))
        alias = "\n".join(f"{n} = {entry}" for n in tnames if entry and n != entry)
        if asserts and entry:
            rc, out, err = run(count_asserts_program(sol, asserts, alias))
            m = re.search(r"RESULT (\d+) (\d+)", out)
            res["A_pass"], res["A_n"] = (int(m.group(1)), int(m.group(2))) if m else (0, len(asserts))
        else:
            res["A_pass"], res["A_n"] = 0, len(asserts)
        res["A"] = res["A_pass"] / max(1, res["A_n"])
        # B: training references vs eval tests
        res["B_tried"], res["B_pass"] = 0, 0
        if etest and entry and tnames and (res["A"] >= 0.5 or r.origin == "candidate"):
            alias_b = f"\n{entry} = {tnames[0]}\n" if tnames[0] != entry else "\n"
            for o in outs.get(r.custom_id, [])[:8]:
                code = extract_code(o)
                if not code:
                    continue
                rc, out, err = run(count_asserts_program(code, asserts, ""))
                m = re.search(r"RESULT (\d+) (\d+)", out)
                if not m or m.group(1) != m.group(2):
                    continue  # use only references that solve the training row
                res["B_tried"] += 1
                rc, out, err = run(PRE + code + alias_b + etest, timeout=60)
                if "EVALPASS" in out:
                    res["B_pass"] += 1
                if res["B_tried"] >= 3:
                    break
        return res

    with ThreadPoolExecutor(24) as ex:
        rows = list(ex.map(work, jobs))
    df = pd.DataFrame(rows)
    if prev is not None:
        prev = prev.set_index(["ti", "ei"])
        new = df.set_index(["ti", "ei"])
        prev = prev[~prev.index.isin(new.index)]
        df = pd.concat([prev, new]).reset_index()
    df.to_parquet(prev_path)
    print(df.groupby(df.eval_id.str.split("/").str[0]).agg(n=("A", "size"), A1=("A", lambda s: (s >= .999).sum()),
                                                            Bany=("B_pass", lambda s: (s > 0).sum()),
                                                            Btried=("B_tried", lambda s: (s > 0).sum())))


if __name__ == "__main__":
    main()
