#!/usr/bin/env python3
"""Async driver for the second IF prefilter. Runs IN-CONTAINER against servers
already started by if_prefilter2.py (same shards, same resume rule).

Why not if_prefilter2.py's thread pool: an n=8 request holds its slot until the
slowest of the 8 finishes, and with this long-tailed length distribution ~78% of
in-flight sequences were finished siblings waiting -- GPUs ran ~70 sequences at
7% KV use. Here every rollout is its own request, so a finished sequence frees
its slot at once, and a per-server semaphore keeps each server at its
--max-running-requests.
"""
import argparse, asyncio, glob, json, time

import aiohttp

OUT = "/workspace/out/if_prefilter2"


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
    ap.add_argument("--hosts", default="g2:10.32.81.253,g34:10.32.80.106,g26:10.32.82.4,"
                                       "g52:10.32.80.122,g53:10.32.80.123")
    ap.add_argument("--port", type=int, default=30971)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--per-server", type=int, default=1600)
    a = ap.parse_args()

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
