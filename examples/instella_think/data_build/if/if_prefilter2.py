#!/usr/bin/env python3
"""Second IF prefilter: 8 rollouts per prompt against our DPO, sampled as RL samples.

  python3 scripts/if_prefilter2.py

What the first prefilter (if_prefilter.py) got wrong, and this fixes:
  * it POSTed the raw prompt with no chat template, so the model had to invent
    the <｜Assistant｜> turn itself; this sends `rendered`, the exact templated
    text training uses;
  * T=1.0 against a T=0.8 training run; this samples at the training settings;
  * 8 sequential requests per thread at 320 threads (~320 sequences in flight on
    40 GPUs) -- ~9.5 h for 26k prompts; this sends n=8 in one request and keeps
    far more in flight.
Resumable: prompts already present in the shards are skipped.
"""
import argparse, gzip, json, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import bench_run  # noqa: E402
from bench_run import CTR, HOST_ROOT, addr, nodes, serve, wait_ready  # noqa: E402

OUT = HOST_ROOT / "out/if_prefilter2"
_lock = threading.Lock()

bench_run.MODELS["instella_dpo_pf"] = {
    **bench_run.MODELS["instella_dpo"],
    "extra": bench_run.MODELS["instella_dpo"]["extra"] + ["--max-running-requests", "2048"],
}


def load_done(shards):
    done = set()
    for p in shards:
        if not p.exists():
            continue
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line)["key"])
                    except Exception:
                        continue
        except Exception:
            pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/if/if_candidates2.jsonl")
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--port", type=int, default=30971)
    ap.add_argument("--ctx", type=int, default=36864)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=1600)
    ap.add_argument("--no-serve", action="store_true", help="servers already up")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (HOST_ROOT / a.candidates).open()]
    hosts = nodes()
    shards = [OUT / f"gen_{h.split('-')[-1]}.jsonl.gz" for h in hosts]
    done = load_done(shards)
    todo = [r for r in rows if r["metadata"]["key"] not in done]
    print(f"=== IF prefilter2: {len(rows):,} candidates, {len(done):,} done, {len(todo):,} to go "
          f"x {a.rollouts} on {len(hosts)} nodes, T={a.temperature} ===", flush=True)
    if not todo:
        return

    if a.no_serve:
        urls = [f"http://{addr(h)}:{a.port}" for h in hosts]
    else:
        urls = wait_ready(serve("instella_dpo_pf", a.ctx, hosts, a.port, "if_prefilter2"))
    if not urls:
        sys.exit("no server came up")
    print(f"    {len(urls)} servers ready", flush=True)

    fhs = [gzip.open(p, "at") for p in shards]
    stat = {"n": 0, "fail": 0, "tok": 0}
    t0 = time.time()

    def work(i_r):
        i, r = i_r
        u = urls[i % len(urls)]
        outs, ntok = [""] * a.rollouts, 0
        try:
            q = requests.post(f"{u}/generate", json={
                "text": r["rendered"],
                "sampling_params": {"temperature": a.temperature, "top_p": a.top_p,
                                    "max_new_tokens": a.max_new, "n": a.rollouts}}, timeout=7200)
            if q.status_code == 200:
                js = q.json()
                js = js if isinstance(js, list) else [js]
                outs = [j.get("text", "") for j in js][: a.rollouts]
                ntok = sum(j.get("meta_info", {}).get("completion_tokens", 0) for j in js)
        except Exception:
            pass
        bad = len([o for o in outs if o]) < a.rollouts
        rec = {"key": r["metadata"]["key"], "label": r["label"], "prompt": r["prompt"],
               "domain": r["metadata"]["domain"], "nc": r["metadata"]["n_constraints"],
               "responses": outs}
        with _lock:
            if bad:
                stat["fail"] += 1
            else:
                fhs[i % len(fhs)].write(json.dumps(rec) + "\n")
                fhs[i % len(fhs)].flush()
            stat["n"] += 1
            stat["tok"] += ntok
            n = stat["n"]
        if n % 500 == 0:
            el = (time.time() - t0) / 60
            rate = n / el if el else 0
            print(f"    {n:,}/{len(todo):,} prompts  {el:.0f} min  {rate:.0f}/min  "
                  f"mean_len {stat['tok'] / max(n * a.rollouts, 1):.0f}  fail {stat['fail']}  "
                  f"eta {(len(todo) - n) / max(rate, 1e-9):.0f} min", flush=True)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, enumerate(todo)))
    for fh in fhs:
        fh.close()
    print(f"\ngeneration done in {(time.time() - t0) / 60:.0f} min; failed {stat['fail']} "
          f"(not written, rerun to retry)", flush=True)


if __name__ == "__main__":
    main()
