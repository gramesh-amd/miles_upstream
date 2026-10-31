#!/usr/bin/env python3
"""Retain the newest AND the best-performing checkpoints; delete the rest.

    # one pass, show what would go
    python3 prune_checkpoints.py <SAVE_DIR> --dry-run --once

    # watch loop alongside training (what run_if.sh starts)
    python3 prune_checkpoints.py <SAVE_DIR> --keep-recent 3 --keep-best 5

WHY THIS EXISTS
This miles keeps every checkpoint forever: the only occurrence of save_retain_interval in
the tree is `args.save_retain_interval = None` (miles/utils/arguments.py:2997) and nothing
reads it. One Instella-MoE-16B checkpoint is ~207 GB, so at --save-interval 20 over 3000
rollouts that is ~31 TB on a /shared_nfs already at 98% and shared with the team.

WHY NOT JUST KEEP THE NEWEST N
Recency alone throws away the peak. RL runs here do not improve monotonically -- the
reference math run (wandb rytwnj5z) peaked around AIME 0.756 at step ~39 and had decayed to
0.635 by step ~179 -- so the checkpoint you actually want to keep is usually NOT among the
newest few. This keeps a union:

    * whatever latest_checkpointed_iteration.txt points at   (resume must never break)
    * the newest --keep-recent                               (resume / recent debugging)
    * the top --keep-best by eval score                      (the ones worth evaluating)

WHERE THE SCORES COME FROM
miles prints every eval to the training log as
    eval 39: {'eval/ifeval': 0.41, 'eval/ifbench': 0.12, ...}
(miles/ray/rollout/metrics.py:38,53). That id is the rollout_id, which is exactly the
checkpoint's iteration number, so `eval 39` pairs with iter_0000039 with no guesswork and
no wandb API, auth or network. Parsed scores are cached in <SAVE_DIR>/.ckpt_scores.json so
a log rotation or a restarted pruner does not lose the history it already saw.

SAFETY: if no scores can be found at all, this falls back to pure recency and says so. An
empty score set is never treated as "everything scored zero".
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# "eval 39: {'eval/ifeval': 0.41, ...}" -- the id is the rollout_id == checkpoint iteration.
EVAL_LINE = re.compile(r"eval (\d+): (\{.*\})\s*$")
ITER_DIR = re.compile(r"^iter_(\d+)$")
DEFAULT_METRICS = ("eval/ifeval", "eval/ifbench")


def parse_scores(log_paths: list[Path], metrics: tuple[str, ...]) -> dict[int, float]:
    """rollout_id -> mean of whichever requested metrics are present."""
    scores: dict[int, float] = {}
    for path in log_paths:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            m = EVAL_LINE.search(line)
            if not m:
                continue
            try:
                # The log holds a python repr (single quotes), not JSON.
                d = ast.literal_eval(m.group(2))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            vals = [float(d[k]) for k in metrics if isinstance(d.get(k), (int, float))]
            if vals:
                # A later eval of the same id wins (a resumed run re-evaluating that step).
                scores[int(m.group(1))] = sum(vals) / len(vals)
    return scores


def load_cache(path: Path) -> dict[int, float]:
    try:
        return {int(k): float(v) for k, v in json.loads(path.read_text()).items()}
    except Exception:
        return {}


def save_cache(path: Path, scores: dict[int, float]) -> None:
    try:
        path.write_text(json.dumps({str(k): v for k, v in sorted(scores.items())}, indent=1))
    except OSError as exc:
        print(f"  WARN could not write score cache {path}: {exc}")


def find_logs(save_dir: Path, explicit: list[str], tag: str | None) -> tuple[list[Path], bool]:
    """Return (logs, attributable). `attributable` is False when the logs cannot be tied to
    THIS save dir, in which case scores must not be trusted.

    This guard exists because of a real near-miss: eval lines are keyed only by rollout_id,
    so globbing every log in rl-logs and matching by id happily attributes ANOTHER run's
    eval score to this run's checkpoint. Run against several old save dirs at once it
    reported the identical score (0.4679) for iter_0000199 in four unrelated runs -- which,
    acted on, would have deleted the wrong 222 GB checkpoints for the "right" reason.
    Scores are only trusted when the caller names the log (--log) or a filename tag matches.
    """
    if explicit:
        return [Path(p) for p in explicit if Path(p).exists()], True

    out: list[Path] = []
    d = Path(os.environ["LOG_DIR"]) if os.environ.get("LOG_DIR") else None
    if d is not None and d.is_dir():
        out = list(d.glob("*.log")) + list(d.glob("*.out"))
    if tag:
        tagged = [p for p in out if tag in p.name]
        if tagged:
            return sorted(tagged, key=lambda p: p.stat().st_mtime), True
    # Nothing ties these logs to this directory.
    return [], False


def iters_on_disk(save_dir: Path) -> list[int]:
    found = []
    for child in save_dir.iterdir() if save_dir.is_dir() else []:
        m = ITER_DIR.match(child.name)
        if m and child.is_dir():
            found.append(int(m.group(1)))
    return sorted(found)  # numeric, not lexical


def latest_pointer(save_dir: Path) -> int | None:
    f = save_dir / "latest_checkpointed_iteration.txt"
    try:
        return int("".join(ch for ch in f.read_text() if ch.isdigit()))
    except Exception:
        return None


def protected_iters(save_dir: Path, explicit: list[int] | None) -> set[int]:
    """Iterations that must never be deleted: --protect plus <SAVE_DIR>/.protected."""
    out = set(explicit or ())
    f = save_dir / ".protected"
    try:
        for line in f.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                out.add(int(digits))
    except FileNotFoundError:
        pass
    except Exception as exc:
        # Unreadable pin list must not silently downgrade to "nothing is protected".
        print(f"  WARN could not read {f}: {exc}; treating as no pins")
    return out


def dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total / 1e9


def prune_once(a) -> None:
    save_dir = Path(a.save_dir)
    if not save_dir.is_dir():
        print(f"{time.strftime('%FT%TZ', time.gmtime())} waiting for {save_dir}")
        return

    iters = iters_on_disk(save_dir)
    if not iters:
        print(f"{time.strftime('%FT%TZ', time.gmtime())} no iter_* yet in {save_dir}")
        return

    cache_path = save_dir / ".ckpt_scores.json"
    scores = load_cache(cache_path)
    logs, attributable = find_logs(save_dir, a.log, a.tag)
    if attributable:
        fresh = parse_scores(logs, tuple(a.metric))
        scores.update(fresh)
        if fresh:
            save_cache(cache_path, scores)
    elif not scores:
        print(
            "  WARN cannot attribute any training log to this save dir (no --log, no --tag "
            "match). Eval lines are keyed only by rollout_id, so using unrelated logs would "
            "assign another run's scores to these checkpoints. Scoring disabled for this pass."
        )

    keep: dict[int, str] = {}

    # Pinned checkpoints, from --protect and from <SAVE_DIR>/.protected (one iteration per
    # line, "#" comments allowed). Score-based eviction is a ranking, not a judgement about
    # worth: iter_0000659 was deleted when iter_0001039 beat it by 0.0004, and a checkpoint
    # can matter for a reason this pruner cannot see -- a release candidate, or a peak on a
    # metric that is not in --metric. Pinning is read from the save dir so it survives
    # without every caller having to remember the flag.
    for it in protected_iters(save_dir, a.protect):
        if it in iters:
            keep[it] = "PROTECTED"
        else:
            print(f"  WARN protected iter_{it:07d} is not on disk")

    ptr = latest_pointer(save_dir)
    if ptr is not None and ptr in iters:
        keep[ptr] = "latest_checkpointed_iteration" if ptr not in keep else keep[ptr] + " + latest"

    for it in iters[-a.keep_recent :] if a.keep_recent > 0 else []:
        keep.setdefault(it, f"newest {a.keep_recent}")

    scored = {it: scores[it] for it in iters if it in scores}
    if scored:
        best = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[: a.keep_best]
        for it, sc in best:
            if it in keep:
                keep[it] += f" + top score {sc:.4f}"
            else:
                keep[it] = f"top {a.keep_best} score {sc:.4f}"
    else:
        # Never let a missing score set justify deleting history.
        extra = iters[-(a.keep_recent + a.keep_best) :]
        for it in extra:
            keep.setdefault(it, "no scores found -> recency fallback")
        print(
            "  WARN no eval scores parsed; best-performing retention is INACTIVE and this "
            "pass is keeping the newest "
            f"{a.keep_recent + a.keep_best} by recency instead. Pass --log <training log> "
            "if the log is somewhere unusual."
        )

    doomed = [it for it in iters if it not in keep]

    print(f"{time.strftime('%FT%TZ', time.gmtime())} {save_dir.name}: "
          f"{len(iters)} on disk, {len(scored)} scored, keeping {len(keep)}")
    for it in sorted(keep):
        sc = scores.get(it)
        print(f"    KEEP iter_{it:07d}  {('score=%.4f' % sc) if sc is not None else 'score=?':>14}  ({keep[it]})")
    if not doomed:
        return

    for it in doomed:
        d = save_dir / f"iter_{it:07d}"
        sc = scores.get(it)
        label = f"score={sc:.4f}" if sc is not None else "unscored"
        if a.dry_run:
            print(f"    would DELETE iter_{it:07d}  {label}  ({dir_size_gb(d):.0f} GB)")
            continue
        size = dir_size_gb(d)
        try:
            shutil.rmtree(d)
            print(f"    DELETED iter_{it:07d}  {label}  (freed {size:.0f} GB)")
        except OSError as exc:
            print(f"    WARN failed to delete {d}: {exc}")
            continue
        # The rollout dataset state for that step is dead weight once the ckpt is gone.
        (save_dir / "rollout" / f"global_dataset_state_dict_{it}.pt").unlink(missing_ok=True)

    usage = shutil.disk_usage(save_dir)
    print(f"    volume: {usage.free / 1e12:.1f} TB free of {usage.total / 1e12:.0f} TB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("save_dir")
    p.add_argument("--keep-recent", type=int, default=3, help="newest N always kept")
    p.add_argument("--keep-best", type=int, default=5, help="top N by eval score always kept")
    p.add_argument("--protect", type=int, action="append", metavar="ITER",
                   help="never delete this iteration (repeatable); also read from "
                        "<SAVE_DIR>/.protected")
    p.add_argument(
        "--metric",
        action="append",
        default=None,
        help=f"eval metric key(s); mean of those present. default: {' '.join(DEFAULT_METRICS)}",
    )
    p.add_argument("--log", action="append", default=[], help="training log(s) to parse for eval lines")
    p.add_argument("--tag", default=None, help="prefer logs whose filename contains this")
    p.add_argument("--interval", type=int, default=300, help="loop sleep seconds")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.metric:
        a.metric = list(DEFAULT_METRICS)
    if a.keep_recent < 1:
        print("FATAL: --keep-recent must be >= 1 so a resume is always possible", file=sys.stderr)
        return 1

    print(
        f"prune_checkpoints: dir={a.save_dir} keep_recent={a.keep_recent} "
        f"keep_best={a.keep_best} metric={'+'.join(a.metric)} dry_run={a.dry_run}"
    )
    if a.once:
        prune_once(a)
        return 0
    while True:
        try:
            prune_once(a)
        except Exception as exc:  # a pruner crash must never take the run with it
            print(f"  WARN prune pass failed: {type(exc).__name__}: {exc}")
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
