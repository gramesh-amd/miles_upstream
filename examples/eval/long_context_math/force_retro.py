#!/usr/bin/env python3
"""Add the forced column to a cell whose generations are already on disk.

    python3 force_retro.py --model ours_r1c_399 --length 32k --urls http://node:30100

Each truncated sample costs one short call (prompt + stored trace + stop-thinking sentence), so the
model has to be served, but no reasoning is regenerated. Concurrency defaults low so it can share
servers with a live cell.
"""

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import detect_close, forcing_suffix  # noqa: E402
from run_cell import FORCE_NEW, _health, summarize  # noqa: E402

from models import DEFAULT_SAMPLING, MODELS, OUT_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--length", required=True)
    ap.add_argument("--urls", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--force-new", type=int, default=int(os.environ.get("FORCE_NEW_OVERRIDE", FORCE_NEW)))
    a = ap.parse_args()

    tag = f"{a.model}_{a.length}"
    samp = MODELS[a.model].get("sampling", DEFAULT_SAMPLING)
    rows = [json.loads(line) for line in gzip.open(OUT_DIR / f"gen_{tag}.jsonl.gz", "rt")]
    prompt_of = {(p["bench"], p["pi"]): p["text"] for p in json.loads((OUT_DIR / f"prompts_{a.model}.json").read_text())}

    ok = [r for r in rows if "error" not in r]
    close = detect_close([r for r in ok if not r["truncated"]])
    stop = forcing_suffix(close)
    todo = [r for r in ok if r["truncated"]]
    print(f"{tag}: {len(ok)} rows, {len(todo)} truncated, close marker {close!r}", flush=True)

    urls = [u for u in a.urls if _health(u)]
    if not urls:
        sys.exit("no reachable server")

    def force(ir):
        i, r = ir
        key = (r["bench"], r["pi"], r["k"])
        try:
            q = requests.post(
                f"{urls[i % len(urls)]}/generate",
                json={"text": prompt_of[(r["bench"], r["pi"])] + r["response"] + stop, "sampling_params": {**samp, "max_new_tokens": a.force_new}},
                timeout=3600,
            )
            q.raise_for_status()
            return key, q.json().get("text", "")
        except Exception as exc:  # noqa: BLE001
            return key, f"<<ERROR {exc}>>"

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        forced = dict(ex.map(force, enumerate(todo)))
    with gzip.open(OUT_DIR / f"force_{tag}.jsonl.gz", "wt") as fh:
        for (b, pi, k), t in forced.items():
            fh.write(json.dumps({"bench": b, "pi": pi, "k": k, "forced": t}) + "\n")

    per_bench, cs, cf = summarize(rows, forced)
    path = OUT_DIR / f"summary_{tag}.json"
    s = json.loads(path.read_text())
    s.pop("forced_unavailable", None)
    s |= {"close_marker": close, "per_bench": per_bench, "composite_strict": round(cs, 4), "composite_forced": round(cf, 4)}
    path.write_text(json.dumps(s, indent=1))
    print(f"composite strict {cs:.3f} forced {cf:.3f}  -> {path}")


if __name__ == "__main__":
    main()
