#!/usr/bin/env python3
"""W&B bookkeeping for eval-only jobs (eval_in_container.sh runs it after each eval).

    eval_series.py <SAVE_DIR> <WANDB_GROUP> <EVAL_GROUP> [<SERIES_ID>]

1. Upload check. Each eval job logs its scores to its own W&B run (group EVAL_GROUP). The W&B
   service sometimes loses its connection while the job shuts down ("ConnectionResetError:
   Connection lost"), and the scores never reach the server although they are in the run's local
   .wandb file. So when a run of EVAL_GROUP has no eval/lcb_v3 in its summary, the local run
   directory that holds it is uploaded with `wandb sync --append`.

2. Eval series. One W&B run per training run, SERIES_ID (default evals-<basename of SAVE_DIR>),
   in WANDB_GROUP, plots every checkpoint's scores against the training step (eval/step), with
   the base model at step 0 when REF_LOAD's .evals.txt has it. The per-eval runs cannot: each
   holds one point, logged at eval step 0. Points already in the series are not logged again, and
   the run is never deleted (W&B refuses to reuse a deleted run id).

Reads SAVE_DIR/.evals.txt ("iter <N> of <dir>: eval 0: {...}" lines, written by the eval jobs).
Needs W&B credentials (NETRC or WANDB_API_KEY) and network access to W&B.
"""

import ast
import glob
import os
import re
import subprocess
import sys

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "genaiamd")
PROJECT = os.environ.get("WANDB_PROJECT", "miles-MI355-code")
LOCAL_DIR = os.environ.get("WANDB_LOCAL_DIR", "")
KEEP = ("eval/humanevalplus", "eval/mbppplus", "eval/lcb_v3", "eval/lcb_v3/repetition_frac",
        "eval/lcb_v3/response_len/median", "eval/lcb_v3/truncated_ratio")


def read_evals(path):
    """{step: metrics} from an .evals.txt; 'release' (a weights-only base checkpoint) is step 0."""
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        m = re.match(r"iter (\S+) of .*?eval 0: (\{.*\})", line)
        if m:
            it, d = m.group(1), ast.literal_eval(m.group(2))
            rows[0 if it == "release" else int(it) + 1] = {k: d[k] for k in KEEP if k in d}
    return rows


def check_uploads(api, eval_group):
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": eval_group}):
        if "eval/lcb_v3" in run.summary:
            print(f"{run.id}: scores on W&B")
            continue
        dirs = [d for d in glob.glob(os.path.join(LOCAL_DIR, "wandb", f"run-*-{run.id}"))
                if any(b"eval/lcb_v3" in open(f, "rb").read() for f in glob.glob(os.path.join(d, "*.wandb")))]
        print(f"{run.id}: scores missing on W&B; uploading {dirs}")
        for d in dirs:
            subprocess.run(["wandb", "sync", "--append", d], check=False)


def update_series(api, save_dir, group, series_id):
    rows = read_evals(os.path.join(save_dir, ".evals.txt"))
    ref_load = os.environ.get("REF_LOAD", "")
    base = read_evals(os.path.join(ref_load, ".evals.txt")).get(0) if ref_load else None
    if base and 0 not in rows:
        rows[0] = base
    try:
        have = {int(r["eval/step"]) for r in api.run(f"{ENTITY}/{PROJECT}/{series_id}").scan_history(keys=["eval/step"])}
    except Exception:
        have = set()
    new = sorted(s for s in rows if s not in have)
    if not new:
        print(f"{series_id}: up to date ({sorted(have)})")
        return
    run = wandb.init(entity=ENTITY, project=PROJECT, id=series_id, name=f"{group}-EVALS", group=group,
                     job_type="eval-series", resume="allow")
    run.define_metric("eval/step")
    run.define_metric("eval/*", step_metric="eval/step")
    for step in new:
        run.log({"eval/step": step, **rows[step]})
    run.finish()
    print(f"{series_id}: logged steps {new}; {run.url}")


def main():
    save_dir, group, eval_group = sys.argv[1:4]
    series_id = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else f"evals-{os.path.basename(save_dir.rstrip('/'))}"
    api = wandb.Api()
    check_uploads(api, eval_group)
    update_series(api, save_dir, group, series_id)


if __name__ == "__main__":
    main()
