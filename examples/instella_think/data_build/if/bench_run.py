#!/usr/bin/env python3
"""One cell of the long-context math benchmark: one model at one context length.

  python3 scripts/bench_run.py --model ours_r1c_399 --length 64k

Serves the model across the allocation, generates every sample, STORES EVERY
GENERATION, grades, and tears the servers down. Run from the host (it ssh's);
the generation half talks HTTP so it does not care where it runs.

EVERY GENERATION IS KEPT, gzipped, one JSON object per sample, before any
grading happens. Grading is a guess about string equality that will be wrong for
some answer formats, and a benchmark whose raw outputs were discarded can only
be re-graded by re-running it -- which at 128k is hours of GPU time per cell.
The summary is derived from the stored file, never the other way round.

LENGTHS ARE RUN, NOT EXTRAPOLATED. Each cell generates at its own context
ceiling. A 32k number taken by truncating 128k generations would measure
something else entirely: the model's token budget changes how it reasons, not
just where it stops -- this run's own traces grew from 11.7k to 35.2k tokens
while truncation went 0 -> 25.8%.

RESERVE leaves room for nothing in particular: it is slack so that a prompt
whose length we measured with the tokenizer cannot overrun the server's window
by a few tokens and turn the whole cell into HTTP 400s. That happened once
already, on the eval-forcing run, where at-cap traces filled the window exactly.
"""
import argparse
import gzip
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from bench_models import (DEFAULT_SAMPLING, LENGTHS, MODELS, SUITES,  # noqa: E402
                          rope_override, suite_suffix)

HOST_ROOT = Path("/it-share-prj2-1/data/goramesh-instella-rl")
WS = "/workspace"
CTR = "instella_rl"
RESERVE = 256
# Tokens allowed for the forced answer. 512 is NOT enough and the failure is
# silent: the model takes the stop-thinking sentence as "write the solution now"
# and produces a full formal write-up, so 43.4% of forced completions were cut
# off mid-derivation before reaching a box. Measured on 400 at-cap samples,
# raising this to 2048 took boxed 58.8% -> 84.0% and correct 16.0% -> 30.0%.
# Past that the curve flattens and the cost is real, since this fires on every
# truncated sample in the cell.
FORCE_NEW = 2048
BOXED = re.compile(r"\\boxed\s*\{")


def norm(s: str) -> str:
    s = s.strip().strip("$").strip().rstrip(".")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:text|mathrm|displaystyle)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return re.sub(r"[{}\s,$\\]", "", s).lower()


def brace_body(text: str, start: int) -> str | None:
    depth, i, out = 1, start, []
    while i < len(text):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return "".join(out)
        out.append(text[i])
        i += 1
    return None


def last_boxed(text: str) -> str | None:
    m = None
    for m in BOXED.finditer(text):
        pass
    return None if m is None else brace_body(text, m.end())


# Candidates in the order they are preferred when several appear. "</think>" is
# listed first because it is the de-facto convention; the rest exist because
# chat templates and emitted text disagree often enough that guessing is unsafe.
CLOSE_CANDIDATES = ["</think>", "</thought>", "</reasoning>", "<|end_of_thought|>"]


def _detect_close(finished: list[dict], sample: int = 400, floor: float = 0.5) -> str:
    """Which end-of-thinking marker does THIS model actually emit?

    Asked of the model's own finished generations rather than its chat template,
    because the two disagree: ours and Gemma's templates contain no think markers
    while ours closes with </think> in 100% of finished samples. The floor guards
    the other direction -- a model that genuinely has no thinking phase must get
    an empty string, not a marker that happened to appear in a few traces.
    """
    if not finished:
        return ""
    look = finished[:sample]
    best, rate = "", 0.0
    for cand in CLOSE_CANDIDATES:
        r = sum(cand in f["response"] for f in look) / len(look)
        if r > rate:
            best, rate = cand, r
    return best if rate >= floor else ""


def sh(cmd: str, timeout: int = 300) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


# Hostnames that do not resolve from THIS vm but are perfectly reachable by ip.
# tus1-p3-g10 and g44 have no DNS entry here, yet a compute node resolves them
# and ssh to the address works; slurm's own srun also cannot reach g10. Treating
# them as unusable cost us a node for a day, so the address is pinned instead.
KNOWN_IPS = {"tus1-p3-g10": "10.32.81.245", "tus1-p3-g44": "10.32.80.114"}


def addr(host: str) -> str:
    """Something ssh and curl can both use. Prefer DNS, fall back to the pin."""
    ip = sh(f"getent hosts {host} | awk '{{print $1}}'")
    return ip or KNOWN_IPS.get(host, host)


def nodes() -> list[str]:
    """Allocated hosts across ALL of our running jobs, minus BENCH_EXCLUDE_NODES.

    Across all jobs, not just the first: the fourth usable node arrived as a
    separate single-node allocation, and reading only one job silently halved
    the cluster.

    The exclusion exists because holding a node through slurm does not stop
    other people's containers appearing on it -- g32 picked up a
    rocm/jax-training container occupying all 8 GPUs, and sglang then fails to
    start with no useful error.

    BENCH_NODES, when set, is the exact host list and nothing is discovered.
    The serve script pkills every sglang process on each host it touches, so a
    host running training must never be picked up by discovery.
    """
    pinned = [h for h in os.environ.get("BENCH_NODES", "").split(",") if h]
    if pinned:
        return pinned
    out: list[str] = []
    for nodelist in sh("squeue -u $USER -h -o '%N'").splitlines():
        nodelist = nodelist.strip()
        if nodelist:
            out += sh(f"scontrol show hostnames {nodelist}").split()
    skip = {h for h in os.environ.get("BENCH_EXCLUDE_NODES", "").split(",") if h}
    return [h for h in dict.fromkeys(out) if h not in skip]


# CO-LOCATION. BENCH_COLOC_MEMF set means the hosts are running someone's live
# training job: serve through _bench_serve_coloc.sh (no pkill, own process
# group, explicit memory fraction) and tear down by process group only. The
# default launcher and teardown both pkill every sglang process on the host,
# which on a rollout node would kill the training run's own servers.
COLOC_MEMF = os.environ.get("BENCH_COLOC_MEMF", "")
COLOC_MAXRUN = os.environ.get("BENCH_COLOC_MAXRUN", "64")


def pgid_file(tag: str) -> str:
    return f"/tmp/bench_coloc_{tag}.pgid"


def serve(model_key: str, ctx: int, host_list: list[str], port: int, tag: str) -> list[str]:
    m = MODELS[model_key]
    ov = rope_override(m, ctx) or {}
    if ctx > m["advertised"]:
        ov["max_position_embeddings"] = max(ctx, m["native"])
    env = " ".join(f"{k}={v}" for k, v in m["env"].items())
    extra = " ".join(m["extra"])
    if COLOC_MEMF:
        launcher = (f"BENCH_MEM_FRACTION={COLOC_MEMF} BENCH_MAX_RUNNING={COLOC_MAXRUN} "
                    f"BENCH_PGID_FILE={pgid_file(tag)} bash {WS}/scripts/_bench_serve_coloc.sh")
    else:
        launcher = f"bash {WS}/scripts/_bench_serve_inner.sh"
    urls = []
    for h in host_list:
        log = f"{WS}/logs/bench_serve_{tag}.log"
        inner = (f"{env} BENCH_MODEL={m['path']} BENCH_TP={m['tp']} BENCH_DP={m['dp']} "
                 f"BENCH_CTX={ctx} BENCH_PORT={port} BENCH_LOG={log} "
                 f"BENCH_EXTRA={json.dumps(extra)} "
                 f"BENCH_OVERRIDE={json.dumps(json.dumps(ov) if ov else '')} "
                 f"{launcher}")
        subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", addr(h),
                        f"docker exec -d {CTR} bash -lc {json.dumps(inner)}"],
                       capture_output=True, timeout=180)
        urls.append(f"http://{addr(h)}:{port}")
    return urls


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


def _health(u: str) -> bool:
    try:
        return requests.get(f"{u}/health", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--length", required=True, choices=list(LENGTHS))
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--keep-serving", action="store_true")
    ap.add_argument("--suite", default="default", choices=list(SUITES))
    a = ap.parse_args()

    ctx = LENGTHS[a.length]
    m = MODELS[a.model]
    if m.get("engine") == "vllm":
        sys.exit(f"{a.model} needs vLLM; sglang has no {Path(m['path']).name} architecture")
    benches = SUITES[a.suite]
    sfx = suite_suffix(a.suite)
    tag = f"{a.model}_{a.length}{sfx}"
    outdir = HOST_ROOT / "out/bench"
    outdir.mkdir(parents=True, exist_ok=True)
    gen_path = outdir / f"gen_{tag}.jsonl.gz"

    hosts = nodes()
    if not COLOC_MEMF:
        busy = [h for h in hosts if sh(
            f"ssh -n -o ConnectTimeout=10 {addr(h)} \"docker exec {CTR} pgrep -f "
            f"'train_async.py|MegatronTrainRayActor'\"", timeout=60)]
        if busy:
            sys.exit(f"training is running on {busy}; the default launcher would pkill its "
                     f"sglang servers. Set BENCH_COLOC_MEMF to co-locate instead.")
    print(f"=== {m['note']} @ {a.length} ({ctx} tok) on {len(hosts)} nodes: {' '.join(hosts)}"
          + (f"  [co-located, mem fraction {COLOC_MEMF}]" if COLOC_MEMF else ""))
    serve_ctx_pre = ctx + FORCE_NEW + 256
    ov = rope_override(m, serve_ctx_pre)
    print(f"    budget {ctx:,} tok, served window {serve_ctx_pre:,} tok")
    print(f"    rope: {'native' if ov is None else ov['rope_scaling']}"
          f"{'  (+max_position_embeddings raised)' if serve_ctx_pre > m['advertised'] else ''}")

    # THE SERVER WINDOW IS BIGGER THAN THE BUDGET BEING BENCHMARKED, on purpose.
    # A truncated sample fills the window to within RESERVE, so appending the
    # stop-thinking sentence and asking for FORCE_NEW more tokens overruns it and
    # every forcing call returns HTTP 400 -- which is silent, because the error is
    # stored per-record and only shows up as a forced column that cannot beat
    # strict. That is exactly what happened to the first seven cells.
    #
    # Widening the window rather than shrinking the generation budget keeps the
    # quantity under test -- tokens the model may generate -- identical to the
    # cells already finished, so the four lengths stay comparable across the whole
    # sweep. rope is computed against the SERVED length for the same reason the
    # ceiling is raised at all.
    serve_ctx = ctx + FORCE_NEW + 256
    urls = wait_ready(serve(a.model, serve_ctx, hosts, a.port, tag))
    if not urls:
        sys.exit("no server came up")

    # Rendered by bench_prepare_prompts.py inside the container; the host has no
    # transformers, and the chat template is per-model.
    pfile = HOST_ROOT / f"out/bench/prompts_{a.model}{sfx}.json"
    if not pfile.exists():
        sys.exit(f"missing {pfile}; run bench_prepare_prompts.py --model {a.model} "
                 f"--suite {a.suite} in the container")
    jobs = []
    prompt_of = {}
    for p in json.loads(pfile.read_text()):
        budget = ctx - p["ntok"] - RESERVE
        prompt_of[(p["bench"], p["pi"])] = p["text"]
        for k in range(a.n_samples):
            jobs.append({"bench": p["bench"], "pi": p["pi"], "k": k,
                         "label": p["label"], "text": p["text"], "budget": budget})
    # Each model is sampled the way its own card specifies; see bench_models.py.
    samp = m.get("sampling", DEFAULT_SAMPLING)
    print(f"    sampling {samp}", flush=True)
    random.Random(0).shuffle(jobs)
    print(f"    {len(jobs)} generations", flush=True)

    done, lock = [0], threading.Lock()
    fh = gzip.open(gen_path, "wt")

    def run(ij):
        i, job = ij
        u = urls[i % len(urls)]
        rec = {k: job[k] for k in ("bench", "pi", "k", "label")}
        try:
            r = requests.post(f"{u}/generate", json={
                "text": job["text"],
                "sampling_params": {**samp, "max_new_tokens": job["budget"]}},
                timeout=28800)
            r.raise_for_status()
            j = r.json()
            meta = j.get("meta_info", {})
            rec |= {"response": j.get("text", ""),
                    "ntok": meta.get("completion_tokens"),
                    "truncated": meta.get("finish_reason", {}).get("type") == "length"}
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
        rows = list(ex.map(run, enumerate(jobs)))
    fh.close()

    # ---- budget forcing, the second scoring condition --------------------
    # A hard cap alone measures whose output-length habit fits the box, not who
    # can do the maths: this model was RL'd at 57,344 tokens and its p90 output
    # sits exactly on a 32k cap, where accuracy among samples that DO finish is
    # still 0.98. Qwen3 ships the remedy as the documented way to run its own
    # models under a budget (Qwen3 Technical Report, thinking_budget.md): halt
    # the thinking, insert a stop-thinking sentence, call again for the answer.
    # Evaluating a Qwen model at a cap WITHOUT this is evaluating it in the way
    # its authors say is wrong, so it is applied to every model identically.
    #
    # QWEN3'S PUBLISHED SENTENCE, VERBATIM, rather than wording of my own. An
    # intervention I tuned while reading this model's failure traces would be
    # fitted to it even if problem-agnostic; a third party's published string
    # cannot be.
    #
    # THE DELIMITER IS DETECTED, NOT HARDCODED. Chat templates disagree -- the
    # Qwen templates carry <think>/</think>, ours and Gemma's carry no markers
    # at all even though ours emits </think> in 100% of finished samples. So ask
    # the cell's own finished generations what this model actually closes with.
    forced: dict[tuple, str] = {}
    close = _detect_close([r for r in rows if "error" not in r and not r["truncated"]])
    stop = ("\n\nConsidering the limited time by the user, "
            "I have to give the solution based on the thinking directly now.\n"
            + (close + "\n\n" if close else "\n"))
    todo = [r for r in rows if "error" not in r and r["truncated"]]
    print(f"\n    forcing {len(todo)} truncated samples (close marker {close!r})", flush=True)

    def force(ir):
        i, r = ir
        u = urls[i % len(urls)]
        key = (r["bench"], r["pi"], r["k"])
        try:
            q = requests.post(f"{u}/generate", json={
                "text": prompt_of[(r["bench"], r["pi"])] + r["response"] + stop,
                "sampling_params": {**samp, "max_new_tokens": FORCE_NEW}},
                timeout=3600)
            q.raise_for_status()
            return key, q.json().get("text", "")
        except Exception as exc:  # noqa: BLE001
            return key, f"<<ERROR {exc}>>"

    if todo:
        with ThreadPoolExecutor(max_workers=min(a.concurrency, 256)) as ex:
            forced = dict(ex.map(force, enumerate(todo)))
        with gzip.open(str(gen_path).replace("gen_", "force_"), "wt") as ffh:
            for (b, pi, k), txt in forced.items():
                ffh.write(json.dumps({"bench": b, "pi": pi, "k": k, "forced": txt}) + "\n")
    mins = (time.time() - t0) / 60

    summary = {"model": a.model, "note": m["note"], "length": a.length, "ctx": ctx,
               "n_samples": a.n_samples, "minutes": round(mins, 1),
               "close_marker": close, "per_bench": {}}
    # Realized tokens are reported next to every score, not just the budget that
    # was set. Whether a model used its allowance is what says if a cap bound it,
    # and comparing accuracy across models with very different natural lengths is
    # meaningless without it.
    print(f"\n{'benchmark':16s}{'n':>6}{'err':>5}{'trunc':>8}{'STRICT':>9}{'FORCED':>9}"
          f"{'delta':>8}{'med tok':>9}{'p90 tok':>9}")
    for bench, _ in benches:
        rs = [r for r in rows if r["bench"] == bench and "error" not in r]
        er = sum(1 for r in rows if r["bench"] == bench and "error" in r)
        if not rs:
            continue

        def ok_strict(r):
            b = last_boxed(r["response"])
            return b is not None and norm(b) == norm(r["label"])

        def ok_forced(r):
            if not r["truncated"]:
                return ok_strict(r)
            t = forced.get((r["bench"], r["pi"], r["k"]), "")
            if t.startswith("<<ERROR"):
                return False
            b = last_boxed(t)
            return b is not None and norm(b) == norm(r["label"])

        st = sum(map(ok_strict, rs)) / len(rs)
        fo = sum(map(ok_forced, rs)) / len(rs)
        tr = sum(bool(r["truncated"]) for r in rs) / len(rs)
        toks = sorted(r["ntok"] or 0 for r in rs)
        p = lambda q: toks[min(int(q * len(toks)), len(toks) - 1)]  # noqa: E731
        summary["per_bench"][bench] = {"n": len(rs), "errors": er,
                                       "strict": round(st, 4), "forced": round(fo, 4),
                                       "trunc": round(tr, 4),
                                       "med_tok": p(0.5), "p90_tok": p(0.9)}
        print(f"{bench:16s}{len(rs):6d}{er:5d}{tr:8.1%}{st:9.3f}{fo:9.3f}"
              f"{fo-st:+8.3f}{p(0.5):9,d}{p(0.9):9,d}")
    nb = max(len(summary["per_bench"]), 1)
    cs = sum(v["strict"] for v in summary["per_bench"].values()) / nb
    cf = sum(v["forced"] for v in summary["per_bench"].values()) / nb
    summary["composite_strict"], summary["composite_forced"] = round(cs, 4), round(cf, 4)
    print(f"{'composite':16s}{'':6}{'':5}{'':8}{cs:9.3f}{cf:9.3f}{cf-cs:+8.3f}   [{mins:.0f} min]")

    (outdir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"\ngenerations -> {gen_path}\nsummary     -> {outdir}/summary_{tag}.json")

    if not a.keep_serving:
        if COLOC_MEMF:
            kill = f"f={pgid_file(tag)}; [ -s $f ] && kill -9 -- -$(cat $f) && rm -f $f"
        else:
            kill = "pkill -9 -f sglang.launch_server"
        for h in hosts:
            subprocess.run(["ssh", "-n", "-o", "StrictHostKeyChecking=accept-new", addr(h),
                            f"docker exec {CTR} bash -lc {json.dumps(kill)}"],
                           capture_output=True, timeout=120)


if __name__ == "__main__":
    main()
