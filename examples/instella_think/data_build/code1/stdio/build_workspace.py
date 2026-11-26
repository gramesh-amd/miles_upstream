#!/usr/bin/env python3
"""Collect every in-scope (stdio-source) row with its tests and candidate reference programs.

Out: work/rows.jsonl.gz, one record per in-scope row:
  row, custom_id, src, olmo_passrate, prompt, label (capped tests as trained),
  full_tests (uncapped Dolci ground truth), kind (stdio|assert),
  refs: [{origin: olmo|v6, code, reward(v6 only)}]   deduplicated by code text
Joins: Dolci parquet by custom_id; v6 rollouts by problem_id -> exact (prompt, label) of problems.jsonl.gz.
"""
import ast, glob, gzip, json, os, sys, collections
import pyarrow.parquet as pq

sys.path.insert(0, "/tmp/wt_lux/examples/instella_think")
from code_reward import extract_code, _answer_of  # grader's own extraction
from prepare_code_data import parse_tests

V7 = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data"
OUT = f"{V7}/stdio/work/rows.jsonl.gz"
SCOPE = {"klear", "synthetic2"}

rows = [json.loads(l) for l in open(f"{V7}/input/code-train/dolci-code-train.jsonl")]
ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
recs = {}
for r, i in zip(rows, ids):
    lab = json.loads(r["label"])
    stdio = isinstance(lab, list) and lab and isinstance(lab[0], dict)
    if i["src"] in SCOPE or stdio:
        recs[i["row"]] = dict(row=i["row"], custom_id=i["custom_id"], src=i["src"], olmo_passrate=i["olmo_passrate"],
                              prompt=r["prompt"][0]["content"], label=lab, metadata=r["metadata"],
                              kind="stdio" if stdio else "assert", refs=[], full_tests=None)
print("in scope", len(recs), collections.Counter(v["src"] for v in recs.values()))
by_cid = collections.defaultdict(list)
for v in recs.values():
    by_cid[v["custom_id"]].append(v)


def add_ref(v, origin, code, reward=None):
    code = code.strip()
    if not code or any(x["code"] == code for x in v["refs"]):
        return
    v["refs"].append(dict(origin=origin, code=code, reward=reward))


fs = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--allenai--Dolci-Think-RL-7B/snapshots/*/data/*.parquet")))
for f in fs:
    for r in pq.read_table(f, columns=["custom_id", "outputs", "ground_truth"]).to_pylist():
        vs = by_cid.get(r["custom_id"])
        if not vs:
            continue
        full, _ = parse_tests(r["ground_truth"])
        for v in vs:
            v["full_tests"] = full
            for o in r["outputs"] or []:
                add_ref(v, "olmo", extract_code(_answer_of(o or "")))
print("with full tests", sum(v["full_tests"] is not None for v in recs.values()))

# v6: problem_id -> (prompt, label)
key2rows = collections.defaultdict(list)
for v in recs.values():
    key2rows[(v["prompt"].strip(), json.dumps(v["label"], ensure_ascii=False))].append(v)
pid2rows = {}
for l in gzip.open("/tmp/v6roll/problems.jsonl.gz", "rt"):
    d = json.loads(l)
    p = d["prompt"]
    p = p.split("<｜User｜>", 1)[-1].rsplit("<｜Assistant｜>", 1)[0].strip()
    lab = d["label"]
    try:
        lab = json.dumps(json.loads(lab), ensure_ascii=False)
    except Exception:
        pass
    m = key2rows.get((p, lab))
    if m:
        pid2rows[d["problem_id"]] = m
        md = d["metadata"] if isinstance(d["metadata"], dict) else ast.literal_eval(d["metadata"])
        for v in m:
            v["in_v6"] = True
            v["prefilter"] = md.get("prefilter")
print("v6 problems matched to in-scope rows", len(pid2rows))
nsol = collections.Counter()
seen_pid = set()
for f in sorted(glob.glob("/tmp/v6roll/samples/step_*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        d = json.loads(l)
        m = pid2rows.get(d["problem_id"])
        if not m:
            continue
        seen_pid.add(d["problem_id"])
        rw = float(d["reward"])
        for v in m:
            v.setdefault("v6_n", 0)
            v["v6_n"] += 1
            v.setdefault("v6_max", 0.0)
            v["v6_max"] = max(v["v6_max"], rw)
            v.setdefault("v6_solved", 0)
            v["v6_solved"] += int(rw >= 1 - 1e-9)
        # solved rollouts are references; keep up to 6 near-misses for test-level diagnosis
        if rw >= 1 - 1e-9 or rw >= 0.5:
            code = extract_code(_answer_of(d["response"]))
            for v in m:
                if rw >= 1 - 1e-9 and sum(x["origin"] == "v6" for x in v["refs"]) < 8:
                    add_ref(v, "v6", code, rw)
                elif rw < 1 - 1e-9 and sum(x["origin"] == "v6near" for x in v["refs"]) < 4:
                    add_ref(v, "v6near", code, rw)
print("v6 pids seen in samples", len(seen_pid))
with gzip.open(OUT, "wt") as g:
    for k in sorted(recs):
        g.write(json.dumps(recs[k], ensure_ascii=False) + "\n")
c = collections.Counter((v["src"], bool(v.get("v6_n"))) for v in recs.values())
print(c)
print("refs per row", collections.Counter(min(len(v["refs"]), 10) for v in recs.values()))
