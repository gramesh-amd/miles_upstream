"""OLMo 3 offline difficulty filtering for the math RLVR prompt set.

Implements Olmo 3 section "Step 2: offline difficulty filtering" verbatim:

    "we generate eight rollouts for each prompt from the initial checkpoint of the
     model we train (e.g., if starting from the DPO-trained model, we generate from
     the DPO checkpoint). We then remove all samples that the model easily solves
     (that is, those with a pass rate greater than 62.5%). We sample with a
     temperature of 1.0 and top-p of 1.0, matching how we sample during RL training."

So: 8 rollouts, T=1.0, top_p=1.0, DROP pass_rate > 0.625 (i.e. 6, 7 or 8 of 8
correct). Prompts the model never solves are KEPT -- the paper only filters the
easy side offline, and leans on in-training active sampling for the rest.

Grading uses ``olmo3_math_verify.verify``, which scores the WHOLE response via
``rfind("\\boxed")``. CAUTION: that is NOT the same as ``--rm-type deepscaler``,
which returns 0 unless the response contains one of ("<|open|>response<|sep|>",
"</think>", "###Response") and then grades only the text AFTER it. deepscaler is
therefore strictly stricter, and the two CAN disagree: a prompt this filter sees
as mid-difficulty can be all-zero under deepscaler if the model does not close
its reasoning. Keep the filter's verifier and the run's --rm-type consistent, or
measure the gap first (scripts/probe_reward_closer.py).

Needs an SGLang server up with the DPO checkpoint:

    MODEL=/workspace/models/Instella-MoE-16B-A3B-DPO \
        bash /workspace/scripts/serve_sglang_farskip_dp.sh

    python3 filter_rl_difficulty.py \
        --input  /workspace/data/math_prompts.jsonl \
        --output /workspace/data/math_prompts_filtered.jsonl

COST. 30180 prompts x 8 rollouts x <=8192 tokens is ~241k generations -- far more
than one node does in reasonable time. Use --limit for a bounded pilot and read
the pass-rate histogram before committing. Olmo 3 hit this same wall and skipped
offline filtering for their 32B, relying on active sampling instead; the
equivalent in miles is
``--dynamic-sampling-filter-path ...common_filters.apply_reward_nonzero_std_filter``
with ``--over-sampling-batch-size``, which is already wired into
run_rl_instella_math.sh.

Results are cached per prompt in ``<output>.cache.jsonl`` and the run is
resumable, which matters because these nodes get reclaimed without warning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

N_ROLLOUTS = 8
PASS_RATE_DROP_ABOVE = 0.625  # strictly greater is dropped: 6/8 = 0.75 goes
TEMPERATURE = 1.0
TOP_P = 1.0


def load_verifier():
    for p in ("/workspace/repos/Primus-Instella-Internal/rl/examples/if_rl",
              os.path.expanduser("~/Primus-Instella-Internal/rl/examples/if_rl")):
        if os.path.isdir(p):
            sys.path.insert(0, p)
            break
    import olmo3_math_verify  # noqa: E402

    return olmo3_math_verify


def rollouts_for(url: str, prompt: list[dict],
                 max_tokens: int) -> tuple[list[str], int]:
    """n=8 in one request so the server can batch them together.

    Also returns how many rollouts stopped because they hit the token cap. This
    matters: a prompt scoring 0/8 purely because its reasoning was truncated is
    NOT a hard prompt, but the paper keeps 0/8 prompts, so without this the
    filter would bank truncation artefacts as genuine difficulty.
    """
    r = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": prompt,
            "n": N_ROLLOUTS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": max_tokens,
        },
        timeout=3600,
    )
    r.raise_for_status()
    choices = r.json()["choices"]
    texts = [c["message"]["content"] or "" for c in choices]
    n_truncated = sum(1 for c in choices if c.get("finish_reason") == "length")
    return texts, n_truncated


def wait_for_server(url: str, timeout_s: int = 900) -> None:
    """Block until /health answers, then fail hard.

    Needed for unattended runs: the container is started with a docker restart
    policy so it resumes from the cache after a crash, but if it comes back while
    the engine is still loading, every request would fail and the whole prompt
    list would be burned through doing no work. Exiting non-zero instead lets the
    restart policy back off and try again.
    """
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=10).status_code == 200:
                # /health answers before the engine can actually generate, so
                # probe with a real (tiny) completion. Without this the first
                # wave of concurrent requests hits a half-initialised server.
                r = requests.post(
                    f"{url}/v1/chat/completions",
                    json={"model": "default",
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 1, "temperature": 0},
                    timeout=120,
                )
                if r.status_code == 200:
                    print(f"server ready (generation probe OK) at {url}", flush=True)
                    return
        except requests.RequestException:
            pass
        time.sleep(15)
    sys.exit(f"ERROR: {url} not ready after {timeout_s}s; exiting non-zero so the "
             f"restart policy retries rather than failing every prompt")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="match --rollout-max-response-len of the RL run (32768). "
                         "At 8192, 63/64 rollouts truncated and every pass rate "
                         "was 0, making the filter meaningless. The filter MUST "
                         "use the same budget as training or its pass rates do "
                         "not describe the model you will actually train.")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    args = ap.parse_args()

    verifier = load_verifier()
    wait_for_server(args.url)

    rows = [json.loads(l) for l in open(args.input)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"prompts: {len(rows)}  rollouts each: {N_ROLLOUTS}  "
          f"max_tokens: {args.max_tokens}", flush=True)

    cache_path = args.output + ".cache.jsonl"
    done: dict[int, int] = {}
    done_trunc: dict[int, int] = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            try:
                d = json.loads(line)
                done[d["i"]] = d["n_correct"]
                done_trunc[d["i"]] = d.get("n_truncated", 0)
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resuming: {len(done)} prompts already graded", flush=True)

    cache = open(cache_path, "a")

    # If the engine dies, every in-flight request fails. Previously that marked
    # all 30180 prompts failed and still exited 0, so the container restart
    # policy never fired and nothing got graded. Retry per prompt, waiting for
    # the engine to come back, and abort the process if failures are sustained.
    import threading
    fail_lock = threading.Lock()
    consecutive_failures = [0]
    MAX_CONSECUTIVE_FAILURES = 32

    def grade(i: int) -> tuple[int, int, int]:
        if i in done:
            return i, done[i], done_trunc.get(i, 0)
        rec = rows[i]
        for attempt in range(3):
            try:
                outs, n_trunc = rollouts_for(args.url, rec["prompt"], args.max_tokens)
                with fail_lock:
                    consecutive_failures[0] = 0
                n_correct = sum(1 for o in outs if verifier.verify(o, rec["label"]))
                return i, n_correct, n_trunc
            except Exception as e:
                if attempt == 0:
                    print(f"  prompt {i} attempt {attempt+1} failed: "
                          f"{type(e).__name__}; waiting for engine", flush=True)
                try:
                    wait_for_server(args.url, timeout_s=600)
                except SystemExit:
                    break
        with fail_lock:
            consecutive_failures[0] += 1
            n = consecutive_failures[0]
        if n >= MAX_CONSECUTIVE_FAILURES:
            print(f"ABORT: {n} consecutive prompt failures -- the engine is not "
                  f"serving. Exiting non-zero so the container restarts and "
                  f"resumes from the cache.", flush=True)
            os._exit(1)
        return i, -1, 0

    results: dict[int, int] = dict(done)
    truncs: dict[int, int] = dict(done_trunc)
    todo = [i for i in range(len(rows)) if i not in done]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for k, (i, n_correct, n_trunc) in enumerate(ex.map(grade, todo), 1):
            if n_correct >= 0:
                results[i] = n_correct
                truncs[i] = n_trunc
                cache.write(json.dumps(
                    {"i": i, "n_correct": n_correct, "n_truncated": n_trunc}) + "\n")
                cache.flush()
            if k % 50 == 0:
                print(f"  graded {k}/{len(todo)}", flush=True)
    cache.close()

    hist = [0] * (N_ROLLOUTS + 1)
    for n in results.values():
        hist[n] += 1
    print("\npass-rate histogram (correct out of 8):")
    for n, c in enumerate(hist):
        mark = "  <- DROPPED (too easy)" if n / N_ROLLOUTS > PASS_RATE_DROP_ABOVE else ""
        print(f"  {n}/8  {c:6d}{mark}")

    kept = [rows[i] for i, n in sorted(results.items())
            if n / N_ROLLOUTS <= PASS_RATE_DROP_ABOVE]
    n_unsolved = sum(1 for n in results.values() if n == 0)

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"\ngraded {len(results)} prompts")
    print(f"kept   {len(kept)} ({100*len(kept)/max(len(results),1):.1f}%)")
    print(f"dropped {len(results)-len(kept)} with pass rate > 62.5%")
    print(f"note: {n_unsolved} prompts were never solved (0/8) and are KEPT, "
          f"per the paper -- watch these, they may be unlearnable")

    # Truncation audit. If a large share of rollouts hit the cap, the pass rates
    # understate real ability and --max-tokens is too low: raise it and re-run
    # (the cache is keyed per prompt, so delete it first to regrade).
    tot_trunc = sum(truncs.values())
    heavy = sum(1 for i, t in truncs.items() if t >= N_ROLLOUTS // 2)
    zero_and_trunc = sum(1 for i, n in results.items()
                         if n == 0 and truncs.get(i, 0) > 0)
    print(f"\ntruncation audit (--max-tokens {args.max_tokens}):")
    print(f"  rollouts cut off at the cap : {tot_trunc} / {len(results)*N_ROLLOUTS}")
    print(f"  prompts with >=4/8 cut off  : {heavy}")
    print(f"  0/8 prompts WITH truncation : {zero_and_trunc}"
          f"  <- these may be truncation artefacts, not hard prompts")
    if tot_trunc > 0.2 * len(results) * N_ROLLOUTS:
        print("  WARNING: >20% of rollouts truncated. Raise --max-tokens and regrade;"
              " pass rates here understate the model.")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
