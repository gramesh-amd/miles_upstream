#!/usr/bin/env python3
"""Logs one eval_code_ckpt.py summary.json to W&B, as one point of a series keyed by training step.

    python3 eval_code_wandb.py --summary <out>/summary.json --step 400 \
        --project <project> --run-name <SAVE_TAG>-offline_eval [--group <training run's group>]

Credentials are W&B's own (WANDB_API_KEY or the netrc); WANDB_ENTITY / WANDB_BASE_URL apply
as usual. Every checkpoint of a run goes to the same run id (--run-name), resumed, rather
than into the live training run: two processes writing one run interleave their history and
W&B drops the out-of-order steps.
"""

import argparse
import json
from pathlib import Path

import wandb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--step", type=int, required=True, help="training step of the checkpoint")
    ap.add_argument("--project", required=True)
    ap.add_argument("--run-name", required=True, help="also the run id, so later checkpoints resume it")
    ap.add_argument("--group", default=None)
    a = ap.parse_args()

    s = json.loads(Path(a.summary).read_text())
    row = {"train/step": a.step}
    tables = {}

    def flat(prefix, block):
        for k, v in block.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[f"{prefix}/{k}"] = v

    for name, v in s.items():
        if not (isinstance(v, dict) and "pass@1" in v):
            continue
        row[f"offline_eval/{name}"] = v["pass@1"]
        flat(f"offline_eval_detail/{name}", v)
        for field in ("by_difficulty", "by_route"):
            for val, blk in (v.get(field) or {}).items():
                flat(f"offline_eval_{field}/{name}/{val}", blk)
        if v.get("per_problem"):
            cols = list(v["per_problem"][0])
            tables[f"offline_eval_problems/{name}"] = wandb.Table(
                columns=cols, data=[[p[c] for c in cols] for p in v["per_problem"]])
    for key in ("lcb_v5v6", "gen_errors", "api_errors", "grade_errors"):
        if s.get(key) is not None:
            row[f"offline_eval/{key}"] = s[key]

    run = wandb.init(project=a.project, group=a.group, id=a.run_name, name=a.run_name, resume="allow",
                     job_type="offline_eval",
                     config={"n_samples": s.get("n_samples"), "sampling": s.get("sampling"), "max_new": s.get("max_new")})
    run.define_metric("train/step")
    for prefix in ("offline_eval", "offline_eval_detail", "offline_eval_by_difficulty", "offline_eval_by_route"):
        run.define_metric(f"{prefix}/*", step_metric="train/step")
    run.log(row)
    for key, t in tables.items():
        run.log({f"{key}_step{a.step}": t})
    run.finish()
    print("logged", len(row), "scalars,", len(tables), "tables;",
          {k: round(v, 4) for k, v in row.items() if k.count("/") == 1 and isinstance(v, float)})


if __name__ == "__main__":
    main()
