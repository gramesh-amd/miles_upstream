#!/usr/bin/env python3
"""Check every cell under BENCH_OUT_DIR for the failure modes that produce a plausible but wrong number.

    python3 validate.py

  rows             the generation file must hold exactly problems x n_samples records.
  errors           thousands of errors means the server died, and the survivors are a biased subset
                   of short, easy samples that can score higher than any real cell.
  forced >= strict forcing only replaces zero-scoring truncated samples, so a lower forced score
                   means the forcing pass itself failed.
  forced errors    HTTP 400 from an over-full context window, or a dead server.
  truncation       a median near the budget with 0% truncation, or heavy truncation with a tiny
                   median, means the cell did not generate what it claims.

Exits non-zero if any cell has a finding.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import OUT_DIR, num_problems  # noqa: E402


def check(tag: str, problems: int) -> list[str]:
    bad: list[str] = []
    s = json.loads((OUT_DIR / f"summary_{tag}.json").read_text())
    n = s.get("n_samples")
    gen = OUT_DIR / f"gen_{tag}.jsonl.gz"
    if not gen.exists():
        return ["no generation file"]
    try:
        rows = [json.loads(line) for line in gzip.open(gen, "rt")]
    except Exception as exc:  # noqa: BLE001
        return [f"generation file unreadable: {str(exc)[:50]}"]

    if len(rows) != problems * n:
        bad.append(f"rows {len(rows)} != {problems * n} ({problems} x n={n})")
    errs = sum(1 for r in rows if "error" in r)
    if errs:
        bad.append(f"{errs} generation errors ({errs / max(len(rows), 1):.1%})")

    cs, cf = s.get("composite_strict"), s.get("composite_forced")
    if cf is not None and cf < cs - 1e-9:
        bad.append(f"forced {cf:.4f} < strict {cs:.4f} -- forcing pass failed")

    frc = OUT_DIR / f"force_{tag}.jsonl.gz"
    if frc.exists():
        fr = [json.loads(line) for line in gzip.open(frc, "rt")]
        fe = sum(1 for r in fr if r["forced"].startswith("<<ERROR"))
        if fe:
            bad.append(f"{fe}/{len(fr)} forcing calls errored ({fe / max(len(fr), 1):.1%})")

    ok = [r for r in rows if "error" not in r]
    if ok:
        tr = sum(bool(r.get("truncated")) for r in ok) / len(ok)
        med = sorted(r.get("ntok") or 0 for r in ok)[len(ok) // 2]
        budget = s.get("ctx", 0)
        if tr == 0 and med > 0.8 * budget:
            bad.append(f"0% truncation but median {med:,} near budget {budget:,}")
        if tr > 0.3 and med < 0.2 * budget:
            bad.append(f"{tr:.0%} truncation but median only {med:,} of {budget:,}")
    return bad


def main() -> int:
    problems = num_problems()
    tags = sorted(p.name[len("summary_") : -len(".json")] for p in OUT_DIR.glob("summary_*.json"))
    findings = 0
    print(f"{'cell':34s}{'n':>4}{'strict':>9}{'forced':>9}  verdict")
    for t in tags:
        s = json.loads((OUT_DIR / f"summary_{t}.json").read_text())
        bad = check(t, problems)
        findings += bool(bad)
        cf = s.get("composite_forced")
        print(f"{t:34s}{s.get('n_samples', 0):>4}{s.get('composite_strict', 0):>9.4f}{(f'{cf:>9.4f}' if cf is not None else '        -')}  {'OK' if not bad else '; '.join(bad)}")
    print(f"\n{len(tags)} cells checked, {findings} with findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
