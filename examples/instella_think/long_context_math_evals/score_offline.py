#!/usr/bin/env python3
"""Write a strict-only summary for a cell from its stored generations, with no server.

    python3 score_offline.py ours_r1c_399 90k

For cells that finished generating but were interrupted during the forcing pass. Strict is exact from
the generation file; forced needs a live server to continue the truncated traces, so use
force_retro.py for that column.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cell import summarize  # noqa: E402

from models import LENGTHS, MODELS, OUT_DIR, num_problems  # noqa: E402


def main(model: str, length: str) -> None:
    tag = f"{model}_{length}"
    rows = [json.loads(line) for line in gzip.open(OUT_DIR / f"gen_{tag}.jsonl.gz", "rt")]
    per_bench, cs, _ = summarize(rows, forced=None)
    for v in per_bench.values():
        v.pop("forced")
    summary = {
        "model": model,
        "note": MODELS[model]["note"],
        "length": length,
        "ctx": LENGTHS[length],
        "n_samples": len(rows) // num_problems(),
        "sampling": MODELS[model]["sampling"],
        "minutes": None,
        "close_marker": None,
        "per_bench": per_bench,
        "composite_strict": round(cs, 4),
        "forced_unavailable": "scored offline from stored generations; run force_retro.py for the forced column",
    }
    out = OUT_DIR / f"summary_{tag}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\ncomposite strict {cs:.4f}  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
