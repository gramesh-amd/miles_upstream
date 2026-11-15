---
title: Monitoring & Logging
description: wandb, structured logs, profiling, and what to look at when something looks off.
---
Miles emits per-rollout metrics to stdout and (optionally) Weights & Biases. SGLang and
Ray write their own logs to their default directories.

## What gets logged by default

Each rollout iteration emits a structured row to stdout (illustrative shape — exact
fields depend on backend and config):

```text
[trainer] iter 12/3000 | loss=0.412 reward=0.61 kl=0.018
                      | rollout=18.4s train=22.1s p2p=2.1s  (total 42.6s)
                      | grad_norm=0.93 lr=1.0e-06
```

When `--use-wandb` is set, metrics also go to wandb under the `train/`, `rollout/`,
and `perf/` namespaces (see `miles/utils/wandb_utils.py`).

## Enabling wandb

```bash
ray job submit --address=auto -- \
  python3 train.py ... \
    --use-wandb \
    --wandb-project miles \
    --wandb-group qwen3-30b-grpo
```

Available flags: `--use-wandb`, `--wandb-project`, `--wandb-group`. `WANDB_API_KEY`
should be supplied via Ray's `env_vars` rather than baked into the launch script.

## What to watch

| Signal | Healthy pattern | Red flag |
|---|---|---|
| `loss` | Slow decay over hundreds of iterations | Spike → crash within an iteration |
| `raw_reward` | Trending up, with healthy variance | Saturates near a single value (collapse) |
| `kl_loss` | Bounded, drifts up over time | Sudden jump (policy diverged from ref) — only logged when `--use-kl-loss` |
| `train_rollout_logprob_abs_diff` | Stable and small (≪ 1.0) | Climbing without bound → train/inference precision drift |
| `entropy_loss` | Slowly decreasing | Falls to ~0 too fast (mode collapse) |
| `grad_norm` | < `clip_grad` (1.0 by default) | Repeatedly hitting clip threshold |
| `rollout_time` / `train_time` | Roughly balanced | One ≫ other → resource imbalance |
| `train/pg_clipfrac` | < 0.2 | > 0.5 means policy is moving fast → drop LR |

Panel names follow what `loss.py` and the rollout logger emit; Miles's wandb metrics
live under `train/`, `rollout/`, `perf/`, `multi_turn/`, `passrate/` namespaces.

## Custom loggers

Replace the default rollout logger with your own to push to internal systems:

```python
def my_log(rollout_id, args, samples, extra, rollout_time) -> bool:
    statsd.gauge("miles.reward", mean([s.reward for s in samples]))
    return False   # also keep default logging
```

```bash
--custom-rollout-log-function-path my_pkg.logging.my_log
```

## Profiling

| Tool | When |
|---|---|
| `nvidia-smi dmon -s u` | Quick sanity check on GPU utilization |
| `nsys profile` | Deep CUDA-level profiling |
| `py-spy dump --pid <ray worker>` | Find Python-side stalls |
| `ray timeline` | Inspect Ray task scheduling |

### Built-in PyTorch profiler

The PyTorch profiler is wired into Miles via `miles/utils/profile_utils.py` and works with both
backends:

```bash
--use-pytorch-profiler
--profile-step-start 10           # training steps (train_actor calls), counted from 0 after init
--profile-step-end 12
--tensorboard-dir /data/tb-run-42 # required: traces are written here (<name>_rank_<r>.*.pt.trace.json.gz)
--profile-target train_overall    # FSDP also accepts train_actor / train_log_probs
```

On the Megatron backend only `train_overall` is implemented (the other targets raise
`NotImplementedError`); it records whole training steps on every rank, including the log-prob pass,
the weight update to the rollout engines and the wait for rollouts. Timed blocks
(`miles/utils/timer.py`) appear in the trace as `miles/<timer>` ranges (`miles/log_probs`,
`miles/actor_train`, `miles/update_weights`, ...), so kernels can be attributed per phase.
Environment variables (they must reach the Ray workers' runtime env):

| Variable | Effect |
|---|---|
| `MILES_PROFILE_RANKS=0,4` | profile only these global ranks (default: all) |
| `MILES_PROFILE_{STACK,SHAPES,MEMORY,FLOPS}=0` | drop the costly extras (default on; an MoE step with all of them is GBs per rank) |
| `MILES_PROFILE_INLINE_STEP=N` | profile the N-th `train_actor` call from inside it (Megatron), written to `--tensorboard-dir` as `inline_rank_<r>_rollout<id>.pt.trace.json.gz` |

`--profile` (Megatron's nsys hook) and `--profile-ranks` are accepted but have no effect: Miles runs
its own training loop.

Known issue (ROCm 7.2, torch 2.9.1, MI355X): in the Megatron actor the profiler started at init
records CPU ops but no GPU kernels; `MILES_PROFILE_INLINE_STEP` is the alternative being evaluated.

**sglang rollout engines** are profiled through their own HTTP API (`/start_profile`, `/stop_profile`)
on each engine's port (not the Miles router, which forwards to a single worker); see
`examples/instella_think/cluster/lux/sglang_profile.sh`.

Open the trace in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

## Where the log files live

| Source | Path |
|---|---|
| Trainer stdout | wherever you redirected `ray job submit` (or Ray dashboard) |
| Ray workers | `~/.ray/session_latest/logs/` |
| wandb local cache | `wandb/run-<id>/files/` |
| PyTorch profiler / memory snapshot | `--tensorboard-dir`, `--memory-snapshot-dir` |

## Router endpoints

The router exposes a small FastAPI surface used internally by Miles:

| Endpoint | Method | What |
|---|---|---|
| `/add_worker` | POST | Register an SGLang engine |
| `/list_workers` | GET | List registered workers |
| any other path | GET / POST / PUT / DELETE | Proxied to a selected SGLang worker — e.g. `/generate`, `/v1/chat/completions`, `/health`. |
