#!/usr/bin/env python3
"""Offline difficulty filter for RL prompt sets (OLMo 3, "Step 2: offline difficulty filtering").

Samples N rollouts per prompt from the checkpoint RL will start from, grades them with the
SAME reward the run will use, and drops prompts the model already solves too often:

    "we generate eight rollouts for each prompt from the initial checkpoint of the model we
     train ... We then remove all samples that the model easily solves (that is, those with a
     pass rate greater than 62.5%). We sample with a temperature of 1.0 and top-p of 1.0,
     matching how we sample during RL training."  (arXiv 2512.13961)

So the defaults are 8 rollouts at T=1.0 / top_p=1.0, dropping pass rate > 0.625 (6, 7 and 8 of
8). Prompts never solved (0/8) are kept, as in the paper; --drop-unsolved removes them too.

    python3 difficulty_filter.py --task code --input train.jsonl --output train.filtered.jsonl
    python3 difficulty_filter.py --task math --rm-type deepscaler --input ... --output ...

Normally driven by prefilter.sh, which starts the sglang server and fills in paths.

WHY IT MATCHES TRAINING EXACTLY. A pass rate measured under different conditions describes a
different model, so everything that decides a reward is taken from the training path rather
than restated here:
  * prompt: miles' chat_template_utils.apply_chat_template with the checkpoint's tokenizer
    (what --apply-chat-template produces), encoded without special tokens and sent as
    input_ids, exactly as sglang_rollout.py does (not through sglang's /v1/chat endpoint,
    which renders and tokenizes on its own);
  * sampling: sglang's native /generate with the rollout's sampling params (no_stop_trim,
    special tokens kept), so the response string is what the reward sees in training;
  * grading: --task code uses code_reward.train_all_pass (the reward's think gate, code
    extraction and sandbox; a solve means every test passes, since training's partial
    credit would count one passing test as solved); --task math uses miles' rm_hub with
    --rm-type, the same function --rm-type selects in training.
--max-new-tokens must equal the run's ROLLOUT_MAX_RESPONSE_LEN: at a shorter budget the
truncated rollouts score 0 and every pass rate is understated.

OUTPUT. Kept rows are written unchanged except for metadata.prefilter = {n, n_correct,
n_truncated}, so a later run can re-threshold without regenerating. <output>.stats.json holds
the pass-rate histogram and the truncation audit.

RESUMABLE. Results are cached per prompt (keyed by a hash of prompt + label) in
<output>.cache.jsonl, whose first line records the sampling and grading config. A cache written
under a different config is refused, because mixing the two would silently blend two
different filters; pass a new --output or --reset-cache.

FAILURE HANDLING. A prompt whose generation or grading fails (engine restart, sandbox
api_error) is retried after waiting for the server, never recorded as 0/N: a recorded 0/N is
KEPT, so an outage would quietly bank unlearnable "hard" prompts. After
--max-consecutive-failures prompts fail in a row the process exits non-zero, so a restart
policy (or a rerun) resumes from the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

_INSTELLA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _render(tokenizer, prompt, chat_template_kwargs: dict) -> str:
    from miles.utils.chat_template_utils import apply_chat_template

    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    return apply_chat_template(
        messages, tokenizer=tokenizer, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )


def _prompt_ids(tokenizer, prompt, chat_template_kwargs: dict) -> list[int]:
    # The rendered template already carries BOS, so encode without special tokens, exactly as
    # sglang_rollout.py does, and send ids rather than leave tokenization to the server.
    return tokenizer.encode(_render(tokenizer, prompt, chat_template_kwargs), add_special_tokens=False)


def _make_grader(args):
    """Returns grade(response, row) -> (correct: bool | None, reason: str | None)."""
    if args.task == "code":
        sys.path.insert(0, _INSTELLA_DIR)
        import code_reward

        if code_reward.health() is None:
            sys.exit(
                f"ERROR: code-exec service not reachable at CODE_EXEC_API_URL="
                f"{os.environ.get('CODE_EXEC_API_URL') or '(unset)'}; start it with cluster/code_exec.sh start"
            )

        def grade(response, row):
            return code_reward.train_all_pass(response, row.get(args.label_key))

        return grade

    import asyncio
    from types import SimpleNamespace

    from miles.rollout.rm_hub import async_rm
    from miles.utils.types import Sample

    rm_args = SimpleNamespace(custom_rm_path=None, rm_type=args.rm_type)

    def grade(response, row):
        meta = row.get(args.metadata_key) or {}
        sample = Sample(response=response, label=row.get(args.label_key), metadata=meta)
        reward = asyncio.run(async_rm(rm_args, sample))
        if isinstance(reward, dict):
            reward = reward.get("score", 0.0)
        return float(reward) > 0, None

    return grade


def _post_generate(url: str, ids: list[int], n: int, sampling: dict, timeout: float) -> list[dict]:
    r = requests.post(f"{url}/generate", json={"input_ids": [ids] * n, "sampling_params": sampling}, timeout=timeout)
    r.raise_for_status()
    outs = r.json()
    if not isinstance(outs, list) or len(outs) != n:
        raise RuntimeError(
            f"/generate returned {type(outs).__name__} of length {len(outs) if isinstance(outs, list) else '?'}"
        )
    return outs


def _finish_type(out: dict) -> str:
    fr = (out.get("meta_info") or {}).get("finish_reason")
    return fr.get("type", "") if isinstance(fr, dict) else str(fr or "")


def wait_for_server(url: str, timeout_s: int) -> None:
    """Block until the engine can actually generate; /health answers while it is still loading."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.post(
                f"{url}/generate",
                json={"text": "hi", "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
                timeout=120,
            )
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(15)
    sys.exit(f"ERROR: sglang at {url} cannot generate after {timeout_s}s")


def _key(row: dict, args) -> str:
    blob = json.dumps([row.get(args.input_key), row.get(args.label_key)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _config(args) -> dict:
    cfg = {
        "task": args.task,
        "tokenizer": os.path.realpath(args.tokenizer),
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "chat_template_kwargs": args.apply_chat_template_kwargs,
    }
    if args.task == "code":
        cfg["code_require_think"] = os.environ.get("CODE_REQUIRE_THINK", "1")
        cfg["code_max_tests"] = os.environ.get("CODE_MAX_TESTS", "32")
        cfg["code_exec_timeout"] = os.environ.get("CODE_EXEC_TIMEOUT", "5")
    else:
        cfg["rm_type"] = args.rm_type
    return cfg


def _open_cache(path: str, cfg: dict, reset: bool) -> tuple[dict, object]:
    done: dict[str, dict] = {}
    if reset and os.path.exists(path):
        os.remove(path)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            first = f.readline()
            header = json.loads(first).get("_config") if first.strip() else None
            if header != cfg:
                sys.exit(
                    f"ERROR: {path} was written under a different config.\n"
                    f"  cache: {header}\n  now:   {cfg}\n"
                    "Use a new --output, or --reset-cache to discard it."
                )
            for line in f:
                try:
                    d = json.loads(line)
                    done[d["key"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue
        fh = open(path, "a")
    else:
        fh = open(path, "w")
        fh.write(json.dumps({"_config": cfg}) + "\n")
        fh.flush()
    return done, fh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["code", "math"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tokenizer", default=os.environ.get("HF_CKPT"), help="HF checkpoint dir (default $HF_CKPT)")
    ap.add_argument("--rm-type", default="deepscaler", help="--task math: the run's --rm-type")
    ap.add_argument("--n", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--max-pass-rate", type=float, default=0.625, help="drop prompts whose pass rate is ABOVE this")
    ap.add_argument("--drop-unsolved", action="store_true", help="also drop 0/N prompts (OLMo 3 keeps them)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, required=True, help="the run's ROLLOUT_MAX_RESPONSE_LEN")
    ap.add_argument("--apply-chat-template-kwargs", type=json.loads, default={})
    ap.add_argument("--input-key", default="prompt")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--metadata-key", default="metadata")
    ap.add_argument("--concurrency", type=int, default=16, help="prompts in flight (x --n sequences)")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (pilot); 0 = all")
    ap.add_argument("--request-timeout", type=float, default=7200)
    ap.add_argument("--server-wait", type=int, default=1800, help="seconds to wait for the engine")
    ap.add_argument("--max-consecutive-failures", type=int, default=32)
    ap.add_argument("--retry-backoff", type=float, default=20.0, help="seconds x attempt before a retry")
    ap.add_argument("--reset-cache", action="store_true")
    args = ap.parse_args()
    if not args.tokenizer:
        ap.error("--tokenizer (or $HF_CKPT) is required")

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = _load_tokenizer(args.tokenizer)
    grade = _make_grader(args)
    cfg = _config(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done, cache = _open_cache(args.output + ".cache.jsonl", cfg, args.reset_cache)
    cache_lock = threading.Lock()

    keys = [_key(r, args) for r in rows]
    todo = [i for i, k in enumerate(keys) if k not in done]
    print(
        f"prompts {len(rows)} | cached {len(rows) - len(todo)} | to grade {len(todo)} | "
        f"{args.n} rollouts each, max_new_tokens {args.max_new_tokens}, T={args.temperature} top_p={args.top_p}",
        flush=True,
    )
    wait_for_server(args.url, args.server_wait)

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    fail_lock = threading.Lock()
    consecutive_failures = [0]
    counters = {"api_error_retries": 0, "no_tests": 0}

    def run_one(i: int) -> None:
        row = rows[i]
        ids = _prompt_ids(tokenizer, row.get(args.input_key), args.apply_chat_template_kwargs)
        for attempt in range(3):
            try:
                outs = _post_generate(args.url, ids, args.n, sampling, args.request_timeout)
                finish = [_finish_type(o) for o in outs]
                if "abort" in finish:
                    raise RuntimeError("engine aborted a request")
                verdicts = [grade(o.get("text", ""), row) for o in outs]
                if any(v is None for v, _ in verdicts):
                    with fail_lock:
                        counters["api_error_retries"] += 1
                    raise RuntimeError("grader unavailable (api_error)")
                rec = {
                    "key": keys[i],
                    "n": args.n,
                    "n_correct": sum(bool(v) for v, _ in verdicts),
                    "n_truncated": sum(f == "length" for f in finish),
                    "no_tests": any(r == "no_tests" for _, r in verdicts),
                }
                with cache_lock:
                    cache.write(json.dumps(rec) + "\n")
                    cache.flush()
                    done[keys[i]] = rec
                with fail_lock:
                    consecutive_failures[0] = 0
                return
            except Exception as e:  # noqa: BLE001 - any failure means "retry after the server is back"
                print(f"  prompt {i} attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(args.retry_backoff * (attempt + 1))
                wait_for_server(args.url, args.server_wait)
        with fail_lock:
            consecutive_failures[0] += 1
            n_fail = consecutive_failures[0]
        if n_fail >= args.max_consecutive_failures:
            print(f"ABORT: {n_fail} consecutive prompts failed; rerun to resume from the cache", flush=True)
            os._exit(1)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for k, _ in enumerate(ex.map(run_one, todo), 1):
            if k % 50 == 0 or k == len(todo):
                rate = k / max(time.time() - t0, 1e-9) * 3600
                print(f"  graded {k}/{len(todo)} ({rate:.0f} prompts/h)", flush=True)
    cache.close()

    missing = [i for i, k in enumerate(keys) if k not in done]
    if missing:
        sys.exit(f"ERROR: {len(missing)} prompts still ungraded (first: {missing[:5]}); rerun to resume")

    hist = [0] * (args.n + 1)
    kept, dropped_easy, dropped_unsolved, dropped_no_tests = [], 0, 0, 0
    tot_trunc = zero_with_trunc = 0
    for row, k in zip(rows, keys, strict=True):
        rec = done[k]
        c = rec["n_correct"]
        hist[c] += 1
        tot_trunc += rec["n_truncated"]
        zero_with_trunc += int(c == 0 and rec["n_truncated"] > 0)
        if rec.get("no_tests"):
            dropped_no_tests += 1
            continue
        if c / args.n > args.max_pass_rate:
            dropped_easy += 1
            continue
        if c == 0 and args.drop_unsolved:
            dropped_unsolved += 1
            continue
        out = dict(row)
        meta = dict(out.get(args.metadata_key) or {})
        meta["prefilter"] = {"n": rec["n"], "n_correct": c, "n_truncated": rec["n_truncated"]}
        out[args.metadata_key] = meta
        kept.append(out)

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_rows = len(rows)
    trunc_frac = tot_trunc / max(n_rows * args.n, 1)
    stats = {
        "config": cfg,
        "max_pass_rate": args.max_pass_rate,
        "drop_unsolved": args.drop_unsolved,
        "prompts": n_rows,
        "kept": len(kept),
        "dropped_easy": dropped_easy,
        "dropped_unsolved": dropped_unsolved,
        "dropped_no_tests": dropped_no_tests,
        "histogram": {f"{c}/{args.n}": hist[c] for c in range(args.n + 1)},
        "truncated_rollout_frac": trunc_frac,
        "unsolved_with_truncation": zero_with_trunc,
        "api_error_retries": counters["api_error_retries"],
    }
    with open(args.output + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\npass-rate histogram (correct out of {args.n}):")
    for c, cnt in enumerate(hist):
        mark = "  <- dropped (too easy)" if c / args.n > args.max_pass_rate else ""
        if c == 0:
            mark = "  <- dropped (--drop-unsolved)" if args.drop_unsolved else "  <- kept (never solved)"
        print(f"  {c}/{args.n}  {cnt:7d}{mark}")
    print(
        f"kept {len(kept)} / {n_rows} ({100 * len(kept) / max(n_rows, 1):.1f}%)  "
        f"easy {dropped_easy}  unsolved {dropped_unsolved}  no_tests {dropped_no_tests}"
    )
    print(
        f"truncation: {100 * trunc_frac:.1f}% of rollouts hit max_new_tokens; "
        f"{zero_with_trunc} 0/{args.n} prompts had truncated rollouts (may be budget, not difficulty)"
    )
    if trunc_frac > 0.2:
        print(
            "WARNING: >20% of rollouts truncated, so pass rates understate the model. "
            "Check --max-new-tokens matches the run's ROLLOUT_MAX_RESPONSE_LEN."
        )
    print(f"wrote {args.output} and {args.output}.stats.json")


if __name__ == "__main__":
    main()
