#!/usr/bin/env python3
"""Create (or replace) a saved W&B view whose charts colour a training run by segment.

A training run resumed across Slurm jobs, or restarted inside a held job, logs to one W&B run;
each launch is a segment (<group>/segment, with <group>/slurm_job_id beside it, see
run_async.sh). Standard W&B line charts cannot mark where a segment starts, so this view draws
each metric with W&B's built-in line-chart preset (wandb/line/v0), coloured by segment.

    wandb_segment_view.py <run name regex> [--view-name NAME] [--metric KEY ...]

Needs the wandb-workspaces package (on Lux: PYTHONPATH=$LUX_ROOT/shared/pylib) and W&B
credentials. The view is separate from anyone's personal workspace; re-running replaces it.
"""
import argparse
import os

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

DEFAULT_METRICS = [
    "passrate/pass@1",
    "passrate/pass@4",
    "rollout/raw_reward",
    "rollout/response_len/mean",
    "rollout/truncated_ratio",
    "train/grad_norm",
]


def segment_chart(metric):
    # Each metric group is plotted against its own step and segment keys (train/* vs train/step).
    group = "train" if metric.startswith("train/") else "rollout"
    x, stroke = f"{group}/step", f"{group}/segment"
    return wr.CustomChart(
        query={"history": {"keys": [x, metric, stroke]}},
        chart_name="wandb/line/v0",
        chart_fields={"x": x, "y": metric, "stroke": stroke},
        chart_strings={"title": f"{metric} by segment"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_regex", help="regex on run names, e.g. 'lux-v6' or '^holdtest'")
    ap.add_argument("--view-name", default=None)
    ap.add_argument("--metric", action="append", default=None)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "genaiamd"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "miles-MI355-code"))
    args = ap.parse_args()
    metrics = args.metric or DEFAULT_METRICS
    workspace = ws.Workspace(
        entity=args.entity,
        project=args.project,
        name=args.view_name or f"Segments: {args.run_regex}",
        sections=[ws.Section(name="Metrics coloured by segment (one colour per launch)",
                             panels=[segment_chart(m) for m in metrics], is_open=True)],
        runset_settings=ws.RunsetSettings(query=args.run_regex, regex_query=True),
    )
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
