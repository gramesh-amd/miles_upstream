#!/usr/bin/env python3
"""Run one cell of the long-context math benchmark: one model at one generation budget.

    python3 run_cell.py --model ours_r1c_399 --length 64k

Serves the model on every allocated node, generates every sample, writes every raw generation
before grading anything, runs the budget-forcing pass on truncated samples, grades both columns and
tears the servers down. Pass --urls to use servers that are already running instead.

Outputs, under BENCH_OUT_DIR:
    gen_<model>_<length>.jsonl.gz      one record per sample, written before grading
    force_<model>_<length>.jsonl.gz    forced answers for truncated samples
    summary_<model>_<length>.json      strict and forced scores, truncation, token percentiles

Serving goes over ssh to each node (inside BENCH_CONTAINER via `docker exec` when it is set) and
expects this package, the models and BENCH_OUT_DIR at the same paths on every node.
"""

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import detect_close, forcing_suffix, is_correct  # noqa: E402

from models import (  # noqa: E402
    BENCHES,
    DEFAULT_SAMPLING,
    LENGTHS,
    MODELS,
    OUT_DIR,
    PKG_DIR,
    model_path,
    rope_override,
)

CONTAINER = os.environ.get("BENCH_CONTAINER", "")
# Slack so a prompt whose length was measured with the tokenizer cannot overrun the window by a few tokens.
RESERVE = 256
# New tokens for the forced answer. At 512, 43% of forced completions were cut off mid write-up
# before reaching a box; 2048 took boxed 58.8% -> 84.0% on 400 at-cap samples.
FORCE_NEW = 2048

# BENCH_COLOC_MEMF set means the nodes are running a live training job: serve through
# serve_sglang_coloc.sh (no pkill, own process group, explicit memory fraction) and tear down by
# process group only. The default launcher pkills every sglang process on the node.
COLOC_MEMF = os.environ.get("BENCH_COLOC_MEMF", "")
COLOC_MAXRUN = os.environ.get("BENCH_COLOC_MAXRUN", "64")


def sh(cmd: str, timeout: int = 300) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def remote(host: str, cmd: str, detach: bool = False, timeout: int = 180) -> str:
    if CONTAINER:
        cmd = f"docker exec {'-d ' if detach else ''}{CONTAINER} bash -lc {json.dumps(cmd)}"
    elif detach:
        cmd = f"nohup bash -lc {json.dumps(cmd)} >/dev/null 2>&1 &"
    ssh = ["ssh", "-n", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", host, cmd]
    return subprocess.run(ssh, capture_output=True, text=True, timeout=timeout).stdout.strip()


def nodes() -> list[str]:
    """BENCH_NODES if set, else every host across all of this user's running slurm jobs, minus BENCH_EXCLUDE_NODES.

    All jobs, not just the first, because extra nodes often arrive as separate allocations.
    """
    pinned = [h for h in os.environ.get("BENCH_NODES", "").split(",") if h]
    if pinned:
        return pinned
    out: list[str] = []
    for nodelist in sh("squeue -u $USER -h -o '%N'").splitlines():
        if nodelist.strip():
            out += sh(f"scontrol show hostnames {nodelist.strip()}").split()
    skip = {h for h in os.environ.get("BENCH_EXCLUDE_NODES", "").split(",") if h}
    return [h for h in dict.fromkeys(out) if h not in skip]


def pgid_file(tag: str) -> str:
    return f"/tmp/bench_coloc_{tag}.pgid"


def serve(model_key: str, ctx: int, hosts: list[str], port: int, tag: str) -> list[str]:
    m = MODELS[model_key]
    ov = rope_override(m, ctx) or {}
    if ctx > m["advertised"]:
        ov["max_position_embeddings"] = max(ctx, m["native"])
    env = " ".join(f"{k}={v}" for k, v in m["env"].items())
    if COLOC_MEMF:
        launcher = f"BENCH_MEM_FRACTION={COLOC_MEMF} BENCH_MAX_RUNNING={COLOC_MAXRUN} BENCH_PGID_FILE={pgid_file(tag)} bash {PKG_DIR}/serve_sglang_coloc.sh"
    else:
        launcher = f"bash {PKG_DIR}/serve_sglang.sh"
    log = OUT_DIR / "logs" / f"serve_{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    urls = []
    for h in hosts:
        inner = (
            f"{env} BENCH_MODEL={model_path(m)} BENCH_TP={m['tp']} BENCH_DP={m['dp']} "
            f"BENCH_CTX={ctx} BENCH_PORT={port} BENCH_LOG={log} "
            f"BENCH_EXTRA={json.dumps(' '.join(m['extra']))} "
            f"BENCH_OVERRIDE={json.dumps(json.dumps(ov) if ov else '')} "
            f"{launcher}"
        )
        remote(h, inner, detach=True)
        urls.append(f"http://{h}:{port}")
    return urls


def _health(u: str) -> bool:
    try:
        return requests.get(f"{u}/health", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def wait_ready(urls: list[str], minutes: int = 25) -> list[str]:
    deadline = time.time() + minutes * 60
    up: list[str] = []
    while time.time() < deadline:
        up = [u for u in urls if _health(u)]
        print(f"  servers up {len(up)}/{len(urls)}", flush=True)
        if len(up) == len(urls):
            return up
        time.sleep(30)
    return up


def teardown(hosts: list[str], tag: str) -> None:
    if COLOC_MEMF:
        kill = f"f={pgid_file(tag)}; [ -s $f ] && kill -9 -- -$(cat $f) && rm -f $f"
    else:
        kill = "pkill -9 -f sglang.launch_server"
    for h in hosts:
        remote(h, kill, timeout=120)


def summarize(rows: list[dict], forced: dict[tuple, str] | None) -> tuple[dict, float, float]:
    """Per-benchmark strict and forced scores, truncation and realized token percentiles.

    forced=None means no forcing pass exists, and the forced column repeats strict.
    """
    per_bench = {}
    print(f"\n{'benchmark':16s}{'n':>6}{'err':>5}{'trunc':>8}{'STRICT':>9}{'FORCED':>9}{'delta':>8}{'med tok':>9}{'p90 tok':>9}")
    for bench, _ in BENCHES:
        rs = [r for r in rows if r["bench"] == bench and "error" not in r]
        er = sum(1 for r in rows if r["bench"] == bench and "error" in r)
        if not rs:
            continue

        def ok_forced(r):
            if not r["truncated"]:
                return is_correct(r["response"], r["label"])
            t = forced.get((r["bench"], r["pi"], r["k"]), "")
            return not t.startswith("<<ERROR") and is_correct(t, r["label"])

        st = sum(is_correct(r["response"], r["label"]) for r in rs) / len(rs)
        fo = st if forced is None else sum(map(ok_forced, rs)) / len(rs)
        tr = sum(bool(r["truncated"]) for r in rs) / len(rs)
        toks = sorted(r["ntok"] or 0 for r in rs)
        med, p90 = (toks[min(int(q * len(toks)), len(toks) - 1)] for q in (0.5, 0.9))
        per_bench[bench] = {
            "n": len(rs),
            "errors": er,
            "strict": round(st, 4),
            "forced": round(fo, 4),
            "trunc": round(tr, 4),
            "med_tok": med,
            "p90_tok": p90,
        }
        print(f"{bench:16s}{len(rs):6d}{er:5d}{tr:8.1%}{st:9.3f}{fo:9.3f}{fo - st:+8.3f}{med:9,d}{p90:9,d}")
    nb = max(len(per_bench), 1)
    cs = sum(v["strict"] for v in per_bench.values()) / nb
    cf = sum(v["forced"] for v in per_bench.values()) / nb
    return per_bench, cs, cf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--length", required=True, choices=list(LENGTHS))
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--urls", nargs="+", help="use these running servers instead of launching new ones")
    ap.add_argument("--keep-serving", action="store_true")
    a = ap.parse_args()

    ctx = LENGTHS[a.length]
    m = MODELS[a.model]
    tag = f"{a.model}_{a.length}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_path = OUT_DIR / f"gen_{tag}.jsonl.gz"

    pfile = OUT_DIR / f"prompts_{a.model}.json"
    if not pfile.exists():
        sys.exit(f"missing {pfile}; run prepare_prompts.py --model {a.model} first")

    # The served window is wider than the budget under test. A truncated sample fills the budget, so
    # appending the stop-thinking sentence and FORCE_NEW tokens would otherwise overrun the window and
    # every forcing call would fail with HTTP 400. Widening the window instead of shrinking the budget
    # keeps the generated-token budget identical across cells.
    serve_ctx = ctx + FORCE_NEW + RESERVE
    hosts: list[str] = []
    if a.urls:
        urls = [u for u in a.urls if _health(u)]
    else:
        hosts = nodes()
        if not hosts:
            sys.exit("no nodes: set BENCH_NODES or hold a slurm allocation")
        if not COLOC_MEMF:
            busy = [h for h in hosts if remote(h, "pgrep -f 'train_async.py|MegatronTrainRayActor'", timeout=60)]
            if busy:
                sys.exit(f"training is running on {busy}; the default launcher would kill its sglang servers. Set BENCH_COLOC_MEMF to co-locate instead.")
        ov = rope_override(m, serve_ctx)
        print(f"=== {m['note']} @ {a.length} on {len(hosts)} nodes: {' '.join(hosts)}" + (f"  [co-located, mem fraction {COLOC_MEMF}]" if COLOC_MEMF else ""))
        print(f"    budget {ctx:,} tok, served window {serve_ctx:,} tok, rope {'native' if ov is None else ov['rope_scaling']}")
        urls = wait_ready(serve(a.model, serve_ctx, hosts, a.port, tag))
    if not urls:
        sys.exit("no server came up")

    jobs, prompt_of = [], {}
    for p in json.loads(pfile.read_text()):
        prompt_of[(p["bench"], p["pi"])] = p["text"]
        for k in range(a.n_samples):
            jobs.append({"bench": p["bench"], "pi": p["pi"], "k": k, "label": p["label"], "text": p["text"], "budget": ctx - p["ntok"] - RESERVE})
    samp = m.get("sampling", DEFAULT_SAMPLING)
    print(f"    sampling {samp}, {len(jobs)} generations", flush=True)
    random.Random(0).shuffle(jobs)

    done, lock = [0], threading.Lock()
    fh = gzip.open(gen_path, "wt")

    def generate(ij):
        i, job = ij
        rec = {k: job[k] for k in ("bench", "pi", "k", "label")}
        try:
            r = requests.post(
                f"{urls[i % len(urls)]}/generate",
                json={"text": job["text"], "sampling_params": {**samp, "max_new_tokens": job["budget"]}},
                timeout=28800,
            )
            r.raise_for_status()
            j = r.json()
            meta = j.get("meta_info", {})
            rec |= {"response": j.get("text", ""), "ntok": meta.get("completion_tokens"), "truncated": meta.get("finish_reason", {}).get("type") == "length"}
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:160]
        with lock:
            fh.write(json.dumps(rec) + "\n")
            done[0] += 1
            if done[0] % 250 == 0:
                print(f"    {done[0]}/{len(jobs)}", flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        rows = list(ex.map(generate, enumerate(jobs)))
    fh.close()

    # Budget forcing, the second scoring rule: continue each truncated trace with the stop-thinking
    # sentence and the model's own end-of-thinking marker, then grade whatever it boxes.
    close = detect_close([r for r in rows if "error" not in r and not r["truncated"]])
    stop = forcing_suffix(close)
    todo = [r for r in rows if "error" not in r and r["truncated"]]
    print(f"\n    forcing {len(todo)} truncated samples (close marker {close!r})", flush=True)

    def force(ir):
        i, r = ir
        key = (r["bench"], r["pi"], r["k"])
        try:
            q = requests.post(
                f"{urls[i % len(urls)]}/generate",
                json={"text": prompt_of[(r["bench"], r["pi"])] + r["response"] + stop, "sampling_params": {**samp, "max_new_tokens": FORCE_NEW}},
                timeout=3600,
            )
            q.raise_for_status()
            return key, q.json().get("text", "")
        except Exception as exc:  # noqa: BLE001
            return key, f"<<ERROR {exc}>>"

    forced: dict[tuple, str] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=min(a.concurrency, 256)) as ex:
            forced = dict(ex.map(force, enumerate(todo)))
        with gzip.open(OUT_DIR / f"force_{tag}.jsonl.gz", "wt") as ffh:
            for (b, pi, k), txt in forced.items():
                ffh.write(json.dumps({"bench": b, "pi": pi, "k": k, "forced": txt}) + "\n")
    mins = (time.time() - t0) / 60

    per_bench, cs, cf = summarize(rows, forced)
    print(f"{'composite':16s}{'':19}{cs:9.3f}{cf:9.3f}{cf - cs:+8.3f}   [{mins:.0f} min]")
    summary = {
        "model": a.model,
        "note": m["note"],
        "length": a.length,
        "ctx": ctx,
        "n_samples": a.n_samples,
        "sampling": samp,
        "minutes": round(mins, 1),
        "close_marker": close,
        "per_bench": per_bench,
        "composite_strict": round(cs, 4),
        "composite_forced": round(cf, 4),
    }
    (OUT_DIR / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"\ngenerations -> {gen_path}\nsummary     -> {OUT_DIR}/summary_{tag}.json")

    if hosts and not a.keep_serving:
        teardown(hosts, tag)


if __name__ == "__main__":
    main()
