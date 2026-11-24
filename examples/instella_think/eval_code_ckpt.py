#!/usr/bin/env python3
"""Offline code benchmarks (prepare_code_benchmarks.py sets) for one served checkpoint.

    CODE_BENCH_API_URL=$(cluster/code_exec.sh url bench) python3 eval_code_ckpt.py \
        --model <HF dir> --url http://127.0.0.1:30000 --bench-dir $CODE_BENCH_DIR --out <dir>

Prompts are rendered with the model's own chat template (add_generation_prompt), as miles'
--apply-chat-template does in training, and graded through code_reward's benchmark path
(_bench_answer -> _bench_code -> bench_pass), the one the in-training benchmark evals take.
--api openai sends the same rendered text to /v1/completions (vLLM, or sglang's OpenAI API).

Writes to --out:
    gen.jsonl.gz     every generation, written before grading
    graded.jsonl.gz  one record per sample: correct, reason (no_code / api_error / gen_error), lengths
    summary.json     per set: pass@{1,2,n}, truncation, length stats, by_difficulty / by_route,
                     per_problem; lcb_v5v6 (pass@1 over the union of both sets' problems)

--regrade-from <gen.jsonl.gz> rescores saved generations without generating, e.g. after a
grader change or to check a grader against an earlier run's graded.jsonl.gz.
"""

import argparse
import gzip
import json
import math
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DATASETS = ["lcb_v3", "lcb_v5", "lcb_v6", "humanevalplus", "mbppplus"]


def load_rows(bench_dir: str, datasets: list, limit: int) -> dict:
    """{(set, problem index): row}, the first `limit` problems of each set (0 = all)."""
    rows = {}
    for ds in datasets:
        with open(os.path.join(bench_dir, f"{ds}.jsonl")) as f:
            for pi, line in enumerate(f):
                if limit and pi >= limit:
                    break
                rows[(ds, pi)] = json.loads(line)
    return rows


def preflight_sandbox(cr, url: str, rows: dict) -> None:
    """The sandbox pads tests past its --max-tests as failures, so a low cap fails correct answers."""
    try:
        cap = int(requests.get(f"{url}/health", timeout=10).json().get("max_tests", 0))
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"bench sandbox unreachable ({type(exc).__name__}); check --bench-api-url") from None
    need = max((int(lab.get("n_tests", 1)) for lab in (cr._parse_tests(r["label"]) for r in rows.values())
                if lab and lab.get("route") != "script"), default=1)
    if cap < need:
        raise SystemExit(f"bench sandbox max_tests={cap} < {need} tests in the selected sets; "
                         "restart it with a higher --max-tests (cluster/code_exec.sh start bench uses 128)")
    print(f"bench sandbox: max_tests={cap} (largest problem: {need} tests)", flush=True)


def make_generate(a, samp: dict):
    """job -> {response, ntok, truncated}; raises on a failed request."""
    if a.api == "sglang":
        requests.get(f"{a.url}/health", timeout=30).raise_for_status()

        def generate(job):
            r = requests.post(f"{a.url}/generate", json={
                "text": job["text"], "sampling_params": {**samp, "max_new_tokens": job["budget"]}}, timeout=14400)
            r.raise_for_status()
            j = r.json()
            meta = j.get("meta_info", {})
            return {"response": j.get("text", ""), "ntok": meta.get("completion_tokens"),
                    "truncated": meta.get("finish_reason", {}).get("type") == "length"}
        return generate

    models = requests.get(f"{a.url}/v1/models", timeout=30)
    models.raise_for_status()
    name = a.served_model_name or models.json()["data"][0]["id"]

    def generate(job):
        r = requests.post(f"{a.url}/v1/completions", json={
            "model": name, "prompt": job["text"], "max_tokens": job["budget"], **samp}, timeout=14400)
        r.raise_for_status()
        j = r.json()
        choice = j["choices"][0]
        return {"response": choice.get("text", ""), "ntok": (j.get("usage") or {}).get("completion_tokens"),
                "truncated": choice.get("finish_reason") == "length"}
    return generate


def grade(cr, rec: dict, label) -> dict:
    """Grades rec["response"] in place (and drops it): the benchmark path of code_reward._score_bench,
    plus 3 retries 5 s apart when the sandbox does not answer; a sample still unanswered is wrong."""
    resp = rec.get("response", "")
    label = cr._parse_tests(label)
    code = cr._bench_code(cr._bench_answer(resp), label.get("extract", "olmes"), label)
    ok, reason = False, None
    if not code.strip():
        reason = "no_code"
    else:
        for _ in range(3):
            ok = cr.bench_pass(code, label)
            if ok is not None:
                break
            time.sleep(5)
        if ok is None:
            ok, reason = False, "api_error"
    rec.update(correct=bool(ok), reason=reason, think_closed="</think>" in resp,
               multi_think_close=resp.count("</think>") > 1, code_chars=len(code))
    rec.pop("response", None)
    return rec


def sample_record(cr, ds: str, pi: int, k: int, row: dict) -> dict:
    m = row.get("metadata") or {}
    return {"ds": ds, "pi": pi, "k": k, "task_id": m.get("task_id"), "difficulty": m.get("difficulty"),
            "platform": m.get("platform"), "route": cr._parse_tests(row["label"]).get("route")}


def run_generation(a, cr, rows: dict, out: Path, samp: dict) -> list:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    generate = make_generate(a, samp)
    jobs = []
    for (ds, pi), row in rows.items():
        text = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        ntok = len(tok(text, add_special_tokens=False)["input_ids"])
        budget = min(a.max_new, a.ctx - ntok - 64)
        for k in range(a.n_samples):
            jobs.append({"ds": ds, "pi": pi, "k": k, "text": text, "budget": budget})
    random.Random(0).shuffle(jobs)
    print(f"{len(jobs)} generations over {','.join(a.datasets)}", flush=True)

    lock = threading.Lock()
    done = [0]
    t0 = time.time()
    records = []
    grader = ThreadPoolExecutor(a.grade_workers)
    graded = []

    def finish(rec, label):
        grade(cr, rec, label)
        with lock:
            records.append(rec)

    def run(job):
        row = rows[(job["ds"], job["pi"])]
        rec = sample_record(cr, job["ds"], job["pi"], job["k"], row)
        for _ in range(a.gen_retries):
            try:
                rec |= generate(job)
                rec.pop("error", None)
                break
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)[:200]
                time.sleep(10)
        with lock:
            gen_fh.write(json.dumps(rec) + "\n")
            done[0] += 1
            if done[0] % 250 == 0:
                print(f"  {done[0]}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        if "response" in rec:
            graded.append(grader.submit(finish, dict(rec), row["label"]))
        else:
            with lock:
                records.append(dict(rec, correct=False, reason="gen_error"))

    with gzip.open(out / "gen.jsonl.gz", "wt") as gen_fh, ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(run, jobs))
    for f in graded:
        f.result()
    grader.shutdown()
    return records


def run_regrade(a, cr, rows: dict) -> list:
    gens = []
    with gzip.open(a.regrade_from, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if (rec["ds"], rec["pi"]) in rows:
                gens.append(rec)
    print(f"regrading {len(gens)} generations from {a.regrade_from}", flush=True)

    def one(rec):
        if "response" not in rec:
            return dict(rec, correct=False, reason="gen_error")
        return grade(cr, rec, rows[(rec["ds"], rec["pi"])]["label"])

    with ThreadPoolExecutor(a.grade_workers) as pool:
        return list(pool.map(one, gens))


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def _mean(x):
    return round(sum(x) / len(x), 4) if x else None


def _q(x, q):
    if not x:
        return None
    x = sorted(x)
    return x[min(len(x) - 1, int(q * len(x)))]


def _block(rs: list, n: int) -> dict:
    """Metrics over a set of sample records (one benchmark, or one slice of it)."""
    per = defaultdict(list)
    for r in rs:
        per[r["pi"]].append(r)
    out = {"n_problems": len(per), "n_samples": len(rs)}
    for k in sorted({1, 2, n} & set(range(1, n + 1))):
        out[f"pass@{k}"] = _mean([pass_at_k(len(v), sum(x["correct"] for x in v), k) for v in per.values()])
    counts = [sum(x["correct"] for x in v) for v in per.values()]
    out["solved_all"] = _mean([c == len(v) for c, v in zip(counts, per.values(), strict=True)])
    out["solved_none"] = _mean([c == 0 for c in counts])
    out["truncated"] = _mean([bool(r.get("truncated")) for r in rs])
    for key in ("no_code", "api_error", "gen_error"):
        out[f"{key}_rate"] = _mean([r.get("reason") == key for r in rs])
    out["no_think_close_rate"] = _mean([not r.get("think_closed", False) for r in rs if "think_closed" in r])
    out["multi_think_close_rate"] = _mean([r.get("multi_think_close", False) for r in rs if "think_closed" in r])
    nt = [r["ntok"] for r in rs if r.get("ntok") is not None]
    out["len_mean"] = round(sum(nt) / len(nt)) if nt else None
    out["len_p50"], out["len_p90"] = _q(nt, 0.5), _q(nt, 0.9)
    ok = [r["ntok"] for r in rs if r.get("ntok") is not None and r["correct"]]
    bad = [r["ntok"] for r in rs if r.get("ntok") is not None and not r["correct"] and not r.get("truncated")]
    out["len_correct_mean"] = round(sum(ok) / len(ok)) if ok else None
    out["len_wrong_finished_mean"] = round(sum(bad) / len(bad)) if bad else None
    fin = [r for r in rs if not r.get("truncated") and "response" not in r and r.get("reason") != "gen_error"]
    out["acc_when_finished"] = _mean([r["correct"] for r in fin])
    return out


def summarize(records: list, n: int) -> dict:
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["ds"]].append(r)
    res = {}
    for ds, rs in by_ds.items():
        b = _block(rs, n)
        for field in ("difficulty", "route"):
            vals = sorted({r.get(field) for r in rs if r.get(field)})
            if len(vals) > 1:
                b[f"by_{field}"] = {v: _block([r for r in rs if r.get(field) == v], n) for v in vals}
        per = defaultdict(list)
        for r in rs:
            per[r["pi"]].append(r)
        b["per_problem"] = [{"pi": pi, "task_id": v[0].get("task_id"), "difficulty": v[0].get("difficulty"),
                             "route": v[0].get("route"), "n_correct": sum(x["correct"] for x in v), "n": len(v),
                             "n_truncated": sum(bool(x.get("truncated")) for x in v),
                             "len_mean": round(sum(x.get("ntok") or 0 for x in v) / len(v))}
                            for pi, v in sorted(per.items())]
        res[ds] = b
    res["gen_errors"] = sum(r.get("reason") == "gen_error" for r in records)
    res["api_errors"] = sum(r.get("reason") == "api_error" for r in records)
    if "lcb_v5" in by_ds and "lcb_v6" in by_ds:
        per = defaultdict(list)
        for r in by_ds["lcb_v5"] + by_ds["lcb_v6"]:
            per[(r["ds"], r["pi"])].append(r)
        res["lcb_v5v6"] = _mean([pass_at_k(len(v), sum(x["correct"] for x in v), 1) for v in per.values()])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="HF checkpoint dir (tokenizer and chat template); required to generate")
    ap.add_argument("--url", help="inference server, e.g. http://127.0.0.1:30000; required to generate")
    ap.add_argument("--api", choices=("sglang", "openai"), default="sglang",
                    help="sglang: native /generate; openai: /v1/completions")
    ap.add_argument("--served-model-name", default=None, help="--api openai model name (default: first of /v1/models)")
    ap.add_argument("--bench-dir", default=os.environ.get("CODE_BENCH_DIR"),
                    help="prepare_code_benchmarks.py output (default: $CODE_BENCH_DIR)")
    ap.add_argument("--bench-api-url", default=os.environ.get("CODE_BENCH_API_URL"),
                    help="bench sandbox URL with its token (default: $CODE_BENCH_API_URL)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="first N problems of each set (0 = all)")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--ctx", type=int, default=36864, help="server context length; caps each request's budget")
    ap.add_argument("--concurrency", type=int, default=512, help="generation requests in flight")
    ap.add_argument("--grade-workers", type=int, default=64, help="programs in flight to the sandbox")
    ap.add_argument("--gen-retries", type=int, default=3, help="attempts per generation request")
    ap.add_argument("--regrade-from", default=None, help="gen.jsonl.gz to rescore instead of generating")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.datasets = a.datasets.split(",")
    if not a.bench_dir:
        ap.error("--bench-dir (or CODE_BENCH_DIR) is required")
    if not a.bench_api_url:
        ap.error("--bench-api-url (or CODE_BENCH_API_URL) is required")
    if not a.regrade_from and not (a.model and a.url):
        ap.error("--model and --url are required unless --regrade-from is given")
    a.url = (a.url or "").rstrip("/")

    # code_reward reads its sandbox URL at import.
    os.environ["CODE_BENCH_API_URL"] = a.bench_api_url.strip().rstrip("/")
    import code_reward as cr

    rows = load_rows(a.bench_dir, a.datasets, a.limit)
    preflight_sandbox(cr, os.environ["CODE_BENCH_API_URL"], rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samp = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
    t0 = time.time()
    if a.regrade_from:
        # The generating run's settings, from its summary.json when it sits next to the generations.
        prev = Path(a.regrade_from).with_name("summary.json")
        head = json.loads(prev.read_text()) if prev.is_file() else {}
        records = run_regrade(a, cr, rows)
        summary = {"model": head.get("model", a.model), "n_samples": head.get("n_samples", a.n_samples),
                   "sampling": head.get("sampling", samp), "max_new": head.get("max_new", a.max_new),
                   "gen_seconds": head.get("gen_seconds")}
    else:
        records = run_generation(a, cr, rows, out, samp)
        summary = {"model": a.model, "n_samples": a.n_samples, "sampling": samp, "max_new": a.max_new,
                   "gen_seconds": round(time.time() - t0)}

    with gzip.open(out / "graded.jsonl.gz", "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary.update(summarize(records, summary["n_samples"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v.get("pass@1") for k, v in summary.items() if isinstance(v, dict) and "pass@1" in v}
                     | {k: summary[k] for k in ("lcb_v5v6", "gen_errors", "api_errors") if k in summary}),
          flush=True)


if __name__ == "__main__":
    main()
