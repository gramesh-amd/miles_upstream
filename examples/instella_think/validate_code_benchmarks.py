#!/usr/bin/env python3
"""Checks the benchmark sets from prepare_code_benchmarks.py against the bench sandbox.

For every set it grades, through code_reward's benchmark path (extraction included):
    ref      reference answers, which should all pass: HE-Pro/MBPP-Pro canonical solutions,
             CRUXEval gold outputs/inputs, accepted Python Codeforces submissions (--cf-subs)
    trivial  an answer that must fail everywhere (`pass` body, empty program, wrong literal)
    dump     (--dump) a samples.jsonl.gz eval dump regraded against these rows; reward-1.0
             samples should still pass and the rest still fail

    CODE_BENCH_API_URL=http://host:18951/<token> python3 validate_code_benchmarks.py \
        --bench-dir $CODE_BENCH_DIR --sets lcb_v5 humanevalpro cruxeval_o \
        [--dump .../samples.jsonl.gz] [--cf-subs .../selected_accepted.parquet]
"""

import argparse
import collections
import gzip
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_reward as cr  # noqa: E402
import prepare_code_benchmarks as pcb  # noqa: E402


def rows_of(bench_dir, name):
    with open(os.path.join(bench_dir, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def fenced(code, olmes=False):
    head = "Here is the completed function:\n\n" if olmes else ""
    return f"<think>\nok\n</think>\n{head}```python\n{code}\n```\n"


def trivial_response(label):
    if label["extract"] == "crux":
        return "<think>x</think>\n[ANSWER]\nassert f(0) == '__never__'\n[/ANSWER]"
    if label["extract"] == "olmes":
        return fenced("def f():\n    pass", olmes=True)
    if label.get("route") == "call":
        return fenced(f"class Solution:\n    def {label['fn_name']}(self, *a):\n        return None")
    return fenced("print(0)")


def ref_responses(name, rows, raw_dir, cf_subs):
    """{task_id: [response, ...]} of reference answers for the sets that have them."""
    out = collections.defaultdict(list)
    if name in ("humanevalpro", "mbpppro"):
        import pandas as pd
        for d in pd.read_parquet(os.path.join(raw_dir, f"{name}.parquet")).to_dict("records"):
            out[f"{name}/{d['id']}"].append(fenced(pcb.pro_canonical(d), olmes=True))
    elif name.startswith("cruxeval_"):
        with open(os.path.join(raw_dir, "cruxeval.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                out[d["id"]].append(f"<think>trace</think>\n[ANSWER]\nassert f({d['input']}) == {d['output']}\n[/ANSWER]")
    elif name == "codeforces" and cf_subs:
        import pandas as pd
        import pyarrow.parquet as pq
        src = "og_source" if "og_source" in pq.ParquetFile(cf_subs).schema_arrow.names else "source"
        s = pd.read_parquet(cf_subs, columns=["problem_id", "programmingLanguage", "verdict", src])
        s = s[s["verdict"].eq("OK") & s["programmingLanguage"].str.contains("Py", case=False)]
        ids = {r["metadata"]["task_id"] for r in rows}
        for d in s[s["problem_id"].isin(ids)].to_dict("records"):
            out[d["problem_id"]].append(fenced(d[src]))
    return out


def grade(jobs, workers):
    """jobs: [(key, response, label)] -> {key: [reward, ...]}"""
    res = collections.defaultdict(list)
    with ThreadPoolExecutor(workers) as ex:
        for key, r in zip([j[0] for j in jobs], ex.map(lambda j: cr._score_bench(j[1], j[2]), jobs)):
            res[key].append(r)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--dump", default=None, help="eval dump samples.jsonl.gz to regrade")
    ap.add_argument("--dump-max", type=int, default=0, help="random sample per set, 0 = all samples")
    ap.add_argument("--dump-save", default=None, help="write per-sample [key, old, new] to <path>.<set>.json")
    ap.add_argument("--dump-keys", default=None, help="regrade only these [task_id, index] keys (JSON list)")
    ap.add_argument("--cf-subs", default=None, help="open-r1/codeforces-submissions parquet")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-trivial", action="store_true")
    args = ap.parse_args()
    raw_dir = args.raw_dir or os.path.join(args.bench_dir, "raw")
    if cr.health() is None and not os.environ.get("CODE_BENCH_API_URL"):
        raise SystemExit("set CODE_BENCH_API_URL (or CODE_EXEC_API_URL) to the bench sandbox")
    report = {}
    for name in args.sets:
        rows = rows_of(args.bench_dir, name)
        labels = {str(r["metadata"]["task_id"]): json.loads(r["label"]) for r in rows}
        rep = {"rows": len(rows)}
        refs = ref_responses(name, rows, raw_dir, args.cf_subs)
        if refs:
            g = grade([(t, resp, labels[t]) for t, rs in refs.items() if t in labels for resp in rs], args.workers)
            fails = sorted(t for t, v in g.items() if not any(v))
            rep["ref"] = {"problems": len(g), "pass_any": len(g) - len(fails),
                          "samples": sum(map(len, g.values())), "samples_pass": int(sum(sum(v) for v in g.values())),
                          "failing": fails[:40]}
        if not args.no_trivial:
            g = grade([(t, trivial_response(lab), lab) for t, lab in labels.items()], args.workers)
            rep["trivial"] = {"problems": len(g), "pass": int(sum(v[0] for v in g.values())),
                              "passing": sorted(t for t, v in g.items() if v[0])[:20]}
        if args.dump:
            jobs, want = [], {}
            for line in gzip.open(args.dump, "rt"):
                d = json.loads(line)
                tid = d["problem_id"].split("/", 1)[1]
                if tid in labels:
                    key = (tid, d["index"])
                    jobs.append((key, d["response"], labels[tid]))
                    want[key] = float(d["reward"])
            if args.dump_max and len(jobs) > args.dump_max:
                jobs = random.Random(0).sample(jobs, args.dump_max)
            if args.dump_keys:
                keep = {tuple(k) for k in json.load(open(args.dump_keys))}
                jobs = [j for j in jobs if j[0] in keep]
            g = grade(jobs, args.workers)
            if args.dump_save:
                with open(f"{args.dump_save}.{name}.json", "w") as f:
                    json.dump([[k, want[k], v[0]] for k, v in g.items()], f)
            conf = collections.Counter((want[k], v[0]) for k, v in g.items())
            rep["dump"] = {"samples": len(g), "problems": len({k[0] for k in g}),
                           "was1_now1": conf[(1.0, 1.0)], "was1_now0": conf[(1.0, 0.0)],
                           "was0_now1": conf[(0.0, 1.0)], "was0_now0": conf[(0.0, 0.0)],
                           "flips": sorted({k[0] for k, v in g.items() if want[k] != v[0]})[:20]}
        report[name] = rep
        print(json.dumps({name: rep}), flush=True)
    cr.STATS.dump()
    if cr.STATS.counts["api_error"]:
        print(f"WARNING api_error={cr.STATS.counts['api_error']}: the sandbox dropped requests")


if __name__ == "__main__":
    main()
