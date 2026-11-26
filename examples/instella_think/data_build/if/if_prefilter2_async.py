#!/usr/bin/env python3
"""IF prefilter, generation: 8 rollouts per candidate from the DPO model at the training
settings (T=0.8, top_p 1.0, 32K). Sends the exact chat-templated text training uses (`rendered`).

Needs sglang servers already up (../serve_grader.sh with PORT=30971 MAX_RUNNING=2048), one per node:

    python3 if_prefilter2_async.py --hosts g2:10.0.0.1,g34:10.0.0.2 \
        --candidates $ROOT/data/if/if_candidates2.jsonl --out $ROOT/out/if_prefilter2

Every rollout is its own request, so a finished sequence frees its slot at once (an n=8 request
holds the slot until the slowest of the 8 finishes). Resumable: prompts already in --out are skipped.
"""
import argparse, asyncio, glob, json, os, time

import aiohttp

OUT = None


def load_done():
    # Plain JSONL, not gzip: a killed gzip writer hides every later append from
    # stock readers. A torn final line here is just skipped.
    done = set()
    for p in glob.glob(f"{OUT}/*.jsonl"):
        with open(p) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    continue
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="/workspace/data/if/if_candidates2.jsonl")
    ap.add_argument("--out", default="/workspace/out/if_prefilter2")
    ap.add_argument("--hosts", required=True, help="tag:ip,tag:ip,... one sglang server per node")
    ap.add_argument("--port", type=int, default=30971)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--per-server", type=int, default=1600)
    a = ap.parse_args()
    global OUT
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)

    hosts = [h.split(":") for h in a.hosts.split(",")]
    shards = [f"{OUT}/gen_{tag}.jsonl" for tag, _ in hosts]
    urls = [f"http://{ip}:{a.port}/generate" for _, ip in hosts]
    rows = [json.loads(l) for l in open(a.candidates)]
    done = load_done()
    todo = [r for r in rows if r["metadata"]["key"] not in done]
    print(f"=== prefilter2 async: {len(rows):,} candidates, {len(done):,} done, {len(todo):,} to go, "
          f"{len(urls)} servers x {a.per_server} ===", flush=True)

    sems = [asyncio.Semaphore(a.per_server) for _ in urls]
    fhs = [open(p, "a") for p in shards]
    st = {"n": 0, "fail": 0, "tok": 0, "rr": 0}
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=7200)
    conn = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as sess:
        async def one(text):
            k = st["rr"] % len(urls)
            st["rr"] += 1
            async with sems[k]:
                for _ in range(2):
                    try:
                        async with sess.post(urls[k], json={"text": text, "sampling_params": {
                                "temperature": a.temperature, "top_p": a.top_p,
                                "max_new_tokens": a.max_new}}) as q:
                            if q.status == 200:
                                js = await q.json()
                                return js.get("text", ""), js.get("meta_info", {}).get("completion_tokens", 0)
                    except Exception as exc:
                        st["err"] = repr(exc)[:160]
                        await asyncio.sleep(2)
            return "", 0

        async def prompt(i, r):
            res = await asyncio.gather(*[one(r["rendered"]) for _ in range(a.rollouts)])
            outs = [t for t, _ in res]
            st["tok"] += sum(n for _, n in res)
            st["n"] += 1
            if all(outs):
                rec = {"key": r["metadata"]["key"], "label": r["label"], "prompt": r["prompt"],
                       "domain": r["metadata"]["domain"], "nc": r["metadata"]["n_constraints"],
                       "responses": outs}
                fh = fhs[i % len(fhs)]
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            else:
                st["fail"] += 1
            n = st["n"]
            if n % 500 == 0:
                el = (time.time() - t0) / 60
                rate = n / el if el else 0
                print(f"    {n:,}/{len(todo):,} prompts  {el:.0f} min  {rate:.0f}/min  "
                      f"mean_len {st['tok'] / max(n * a.rollouts, 1):.0f}  fail {st['fail']}  "
                      f"eta {(len(todo) - n) / max(rate, 1e-9):.0f} min  {st.get('err', '')}", flush=True)

        # Bound prompts in flight so memory stays flat; sequences are bounded by sems.
        window = asyncio.Semaphore(len(urls) * a.per_server // a.rollouts * 2)

        async def guarded(i, r):
            async with window:
                await prompt(i, r)

        await asyncio.gather(*[guarded(i, r) for i, r in enumerate(todo)])
    for fh in fhs:
        fh.close()
    print(f"\ngeneration done in {(time.time() - t0) / 60:.0f} min; failed {st['fail']} (rerun to retry)",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
