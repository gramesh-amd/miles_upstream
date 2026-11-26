"""Join every assert-style training row with its full Dolci tests and all candidate reference programs.

    python3 build_inputs.py  ->  /tmp/v7fn/rows.pkl

Per row: row (index in dolci-code-train.jsonl), custom_id, src, prompt, tests (training label, capped),
full_tests (uncapped Dolci ground_truth), olmo_passrate, refs: list of {code, kind, score}.
Reference kinds: olmo (Dolci outputs), v6 (v6 rollouts; score = logged reward), ace87k (AceCode-87K stored
inferences with pass_rate 1.0 on the 87K tests, AceCoder rows only).
"""
import ast
import glob
import gzip
import hashlib
import json
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, "/tmp/wt_lux/examples/instella_think")
from code_reward import extract_code, _answer_of  # noqa: E402
from prepare_code_data import parse_tests  # noqa: E402

V7 = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data"
WORK = "/tmp/v7fn"
DOLCI = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--allenai--Dolci-Think-RL-7B/snapshots/*/data/*.parquet")))


def load_rows():
    ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
    rows = []
    for i, l in enumerate(open(f"{V7}/input/code-train/dolci-code-train.jsonl")):
        r = json.loads(l)
        lab = json.loads(r["label"])
        if not (isinstance(lab, list) and lab and isinstance(lab[0], str)):
            continue
        assert ids[i]["row"] == i
        rows.append({"row": i, "custom_id": ids[i]["custom_id"], "src": ids[i]["src"],
                     "olmo_passrate": ids[i]["olmo_passrate"], "prompt": r["prompt"][0]["content"],
                     "label_raw": r["label"], "tests": lab, "metadata": r["metadata"]})
    return rows


def code_of(text):
    c = extract_code(_answer_of(text or ""))
    return c if c and ("def " in c or "class " in c or "lambda" in c) else None


def v6_codes(path):
    out = []
    for l in gzip.open(path, "rt"):
        d = json.loads(l)
        try:
            rw = float(d["reward"])
        except (TypeError, ValueError):
            continue
        if rw <= 0:
            continue
        resp = d.get("response") or ""
        if not (resp.lstrip().startswith("<think>") and "</think>" in resp):
            continue
        c = code_of(resp)
        if c:
            out.append((d["problem_id"], rw, c))
    return out


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    rows = load_rows()
    want = {r["custom_id"] for r in rows}
    print("assert rows", len(rows), flush=True)

    # Dolci parquet: full tests + OLMo outputs
    dol = {}
    for f in DOLCI:
        df = pd.read_parquet(f, columns=["custom_id", "ground_truth", "outputs", "passrate"])
        df = df[df.custom_id.isin(want)]
        for r in df.to_dict("records"):
            dol[r["custom_id"]] = r
    print("dolci matched", len(dol), flush=True)

    # v6 rollouts keyed by (prompt body, label)
    key2rows = {}
    for r in rows:
        key2rows.setdefault((r["prompt"], r["label_raw"]), []).append(r["row"])
    pid2rows = {}
    for l in gzip.open("/tmp/v6roll/problems.jsonl.gz", "rt"):
        p = json.loads(l)
        body = p["prompt"].split("<｜User｜>", 1)[-1].rsplit("<｜Assistant｜>", 1)[0]
        k = (body, p["label"])
        if k in key2rows:
            pid2rows[p["problem_id"]] = key2rows[k]
    print("v6 problems matched", len(pid2rows), flush=True)
    files = sorted(glob.glob("/tmp/v6roll/samples/step_*.jsonl.gz"))
    v6 = {}
    with ProcessPoolExecutor(24) as ex:
        for res in ex.map(v6_codes, files):
            for pid, rw, c in res:
                for ri in pid2rows.get(pid, ()):
                    v6.setdefault(ri, []).append((rw, c))
    v6_seen = {ri for ris in pid2rows.values() for ri in ris}

    # AceCode-87K stored inferences
    m87 = json.load(open("/tmp/datacand/function/work/dolci_acecoder_to_87k.json"))
    need = {m87[r["custom_id"]] for r in rows if r["custom_id"] in m87}
    import pyarrow.parquet as pq
    ace = {}
    for f in sorted(glob.glob("/tmp/datacand/function/raw/TIGER-Lab__AceCode-87K/data/*")):
        for r in pq.read_table(f, columns=["id", "inferences"]).to_pylist():
            if r["id"] in need:
                ace[r["id"]] = [(x["pass_rate"], x["completion"]) for x in r["inferences"] if x["pass_rate"] >= 1.0]
    print("ace87k ids", len(ace), flush=True)

    for r in rows:
        d = dol.get(r["custom_id"])
        full, kind = parse_tests(list(d["ground_truth"])) if d is not None else (None, None)
        r["full_tests"] = full if kind == "assert" else None
        refs, seen = [], set()

        def add(code, kind, score):
            if not code:
                return
            h = hashlib.sha1(code.strip().encode()).hexdigest()
            if h in seen:
                return
            seen.add(h)
            refs.append({"code": code, "kind": kind, "score": score})

        if d is not None:
            for o in list(d["outputs"]):
                add(code_of(o), "olmo", None)
        for rw, c in sorted(v6.get(r["row"], []), key=lambda x: -x[0]):
            add(c, "v6", rw)
        aid = m87.get(r["custom_id"])
        for pr, c in ace.get(aid, [])[:8]:
            add(code_of(c) or c, "ace87k", pr)
        r["refs"] = refs
        r["in_v6"] = r["row"] in v6_seen
        r["v6_rewards"] = [rw for rw, _ in v6.get(r["row"], [])]
    import collections
    c = collections.Counter()
    for r in rows:
        c[(r["src"], "full_tests")] += r["full_tests"] is not None
        c[(r["src"], "any_ref")] += bool(r["refs"])
        c[(r["src"], "in_v6")] += r["in_v6"]
        for x in r["refs"]:
            c[(r["src"], "ref_" + x["kind"])] += 1
    for k, v in sorted(c.items()):
        print(k, v)
    pickle.dump(rows, open(f"{WORK}/rows.pkl", "wb"))
