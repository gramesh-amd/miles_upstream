# Instella-MoE on latest miles (fully-async RL)

Self-contained port of Instella-MoE (16B-A3B, 27-layer gated-MLA DeepSeek-V3-style
FarSkip MoE, 64 experts / top-6 / 2 shared) training support onto latest miles
(v0.2.1). All Instella-specific launch/setup logic lives in this folder; the only
external touch is a small, idempotent patch to the installed `mbridge` (gate_proj),
applied by `patch_mbridge_gate_proj.sh`.

## Files

| File | Role |
|------|------|
| `paths.sh` | Per-user locations (checkpoints, data, NFS logs, Primus checkout, sandbox token), all overridable; per-user overrides go in `~/.instella-env.sh`. Sourced by every script below. |
| `cluster/` | Login-node tooling for Spur: `bringup.sh` (containers), `on_node.sh` (run a payload in a node's container), `env_setup.sh`, `ray_head.sh` / `ray_worker.sh`, `launch.sh`, `code_exec.sh` (sandbox). |
| `run_code.sh` | **Code RL arm.** Dolci-Think-RL `code_rlvr`, unit-test reward, held-out pass@1 eval. Wrapper over `run_async.sh`. |
| `code_reward.py` | The code reward (`--custom-rm-path code_reward.batched_reward`). Pass rate while training, all-or-nothing at eval. Self-tests with `python3 code_reward.py` (needs the sandbox). |
| `code_exec_server.py` | Stdlib-only code-execution sandbox for a CPU host (rlimits + audit-hook guard, no containers needed). Started by `cluster/code_exec.sh`. |
| `prepare_code_data.py` | Builds the code train JSONL and a 256-row held-out eval JSONL from Dolci. |
| `overlong_filter.py` | Optional drop-and-recentre overlong filter (`OVERLONG_FILTER=1`). |
| `drgrpo_reducer.py` | Optional Dr. GRPO constant-denominator policy loss (`DRGRPO=1`, needs `DRGRPO_DIVISOR`). |
| `model_args.sh` | `MODEL_ARGS=(...)` + `MOE_LAYER_FREQ` (layer 0 dense, 1–26 MoE). MLA + `--gated-attention` + `--use-simple-farskip-layer`. **No `--spec`** (this Megatron forbids it for MoE; FarSkip is config-driven). |
| `run_async.sh` | Main launcher. `PHASE=prepare` (HF→torch_dist) then `PHASE=train` (submit async DAPO-Math job to Ray). `SINGLE_NODE=1` for the 1-node smoke. |
| `investigate_megatron.sh` | Assemble a FarSkip-capable Megatron at `MEGATRON_PATH` (clone NVIDIA pin `847781764` + overlay patch tree + `pip install -e`). |
| `setup_farskip.sh` | Per-node: overlay the verified sglang515 FarSkip files onto the container sglang. |
| `patch_mbridge_gate_proj.sh` | Add the gated-attention `linear_gate_proj` mapping to the stock `DeepseekV3` mbridge (once per container). |
| `patch_megatron_compat.sh` | Stub newer-Megatron symbols latest miles imports at load (e.g. `optimizer.muon`) but this pin lacks. |
| `validate.sh` | 5-rung ladder: FarSkip+mbridge smoke → convert → sglang load → weight-sync parity → one train step. |
| `patch_megatron_optim_step.py` | Only for runs using `--optimizer-cpu-offload` (not these recipes). Backports upstream Megatron `f4502eb1c` so such checkpoints resume with optimizer state. |
| `fix_ckpt_common_optimizer_state.py` | Repairs checkpoints already written *without* that patch, so they can resume with optimizer state. |
| `run_if.sh` | **Instruction-following arm** at 32k. Same recipe as W&B run `rytwnj5z`, task swapped from math to verifiable IF. Wrapper over `run_async.sh`. |
| `if_reward.py` | The IF reward (`--custom-rm-path if_reward.batched_reward`). Partial credit while training, strict at eval. Self-tests with `python3 if_reward.py`. |
| `IFEvalG/` | Vendored IFEval constraint checkers (Apache-2.0, Google Research via AllenAI). ~53 constraint types, no network at runtime. |
| `prepare_if_data.py` | Builds the IF train/eval JSONLs. `inspect` dumps a source's real schema; `build` writes the files and refuses to write a badly-mapped one. |
| `setup_if_deps.sh` | Per-node: install the checker deps and seed nltk `punkt` onto NFS. `--verify` checks the whole reward stack. |
| `if_eval_log.py` | Logs IFEval's loose metric next to the strict one (`--custom-eval-rollout-log-function-path`). |
| `prune_checkpoints.py` | **Required at `--save-interval 20`.** Keeps the newest N *and the best-scoring N*; this miles implements no retention. |

## Code RL on a fresh allocation, step by step

Run everything from the login node unless a step says otherwise. Paths below are relative
to the miles checkout, with `I=examples/instella_think`. The checkout must live in your NFS
home: the GPU nodes and the sandbox host run these files from there.

### 0. Once per user

```bash
git clone -b instella-code-rl <this repo> ~/miles && cd ~/miles && I=examples/instella_think
# FarSkip Megatron + sglang overlay come from the internal Primus repo:
git clone git@github.com:AMD-AGI/Primus-Instella-Internal.git ~/Primus-Instella-Internal
```

Create `~/.instella-env.sh` (sourced by `paths.sh`; see that file for every knob):

```bash
NFS_ROOT=/shared_nfs/<you>          # logs, rollout dumps, wandb files
CODE_EXEC_HOST=<a CPU host>         # runs the sandbox; reachable by ssh from here and HTTP from GPU nodes
# The DPO base and its torch_dist conversion are world-readable; reuse them instead of
# converting your own (step 4). Checkpoints you train still go to your own CKPT_ROOT.
HF_CKPT=/shared_nfs/gramesh/checkpoints/Instella-MoE-16B-A3B-DPO
REF_LOAD=/shared_nfs/gramesh/checkpoints/Instella-MoE-16B-A3B-DPO_torch_dist
# The built code data is also readable; skip step 4's data build by pointing at it:
CODE_TRAIN=/shared_nfs/gramesh/data/code-train/dolci-code-train.jsonl
CODE_EVAL=/shared_nfs/gramesh/data/code-eval/dolci-code-heldout.jsonl
```

W&B credentials go in `~/.netrc` (`wandb login` writes it); the scripts pass it to the
workers by absolute path because the container runs as root with `HOME=/root`.

Start every shell below (login node or container) with:

```bash
cd ~/miles && I=examples/instella_think && source $I/paths.sh
```

### 1. Get a node (8x MI355X)

```bash
sbatch --parsable --nodes=1 --ntasks-per-node=1 --exclusive --cpus-per-task=236 --gres=gpu:8 \
  --time=24:00:00 --partition=amd-spur,amd-spur-v2 \
  --account=<amd-brain-models|amd-burst> --qos=<amd-brain-models-qos|amd-burst-qos> \
  --job-name=code-rl --wrap="sleep infinity"
squeue -u $USER                       # wait for RUNNING, note JOBID and the node name
```

List both partitions; one alone can queue for a day while the other has idle nodes. Burst
jobs are preemptible (cancelled, not requeued), so long runs belong on a team QOS.

### 2. Containers (5-20 min, mostly image pull)

```bash
$I/cluster/bringup.sh <JOBID>
ls ~/.miles/status-<JOBID>/           # one <node>.ready per node when done
```

`ssh -p 18822 $USER@<node>` then gets you a shell in the container (`source /opt/venv/bin/activate`).

### 3. Per-container setup (every new container)

```bash
NODE=<node>; J=<JOBID>
$I/cluster/on_node.sh $J $NODE cluster/env_setup.sh      # sglang FarSkip overlay, Megatron shims, mbridge patch
```

Expect `ALL OK`. Spur allows one interactive step per node, so run `on_node.sh` calls one
after another, never in parallel on the same node.

### 4. Once per user (skip what step 0 already points at)

Inside the container (`ssh -p 18822 $USER@$NODE`, `source /opt/venv/bin/activate`, then
the shell prelude above):

```bash
bash $I/investigate_megatron.sh                     # FarSkip Megatron into ~/Primus-Instella-Internal
PHASE=prepare bash $I/run_async.sh                  # HF -> torch_dist (8 GPU), only if REF_LOAD is unset
HF_HOME=$NFS_ROOT/hf_cache python3 $I/prepare_code_data.py build   # -> $DATA_ROOT, only if CODE_TRAIN is unset
```

### 5. Sandbox on the CPU host

```bash
$I/cluster/code_exec.sh start         # creates ~/.code-exec-token, starts code_exec_server.py
$I/cluster/code_exec.sh status        # /health must answer
```

### 6. Ray head, then launch

```bash
HEAD_IP=$(ssh -p 18822 $NODE "ip -o -4 route get 1" | grep -oP '(?<=src )\S+')   # routable IP, not docker0
$I/cluster/on_node.sh $J $NODE cluster/ray_head.sh $HEAD_IP
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP CHECK_ONLY=1    # preflights only
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP DRY_RUN=1       # print resolved argv, submit nothing
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP SINGLE_NODE=1 SAVE_TAG=code_1node_$USER
```

`launch.sh` takes `VAR=VALUE` overrides (srun does not forward your environment) and runs
`run_code.sh` by default (`RUN_SCRIPT=run_if.sh` for the IF arm). It prints the log path:
`$NFS_ROOT/rl-logs/launch-code<N>k-<node>-<ts>.out`.

`DRY_RUN=1` prints the runtime env and every `train_async.py` flag, then exits without touching
Ray or the GPUs. Some flags appear twice (e.g. from `model_args.sh` and an override); argparse
keeps the last one, so check the printed list rather than reading the scripts.

The reference run below is exactly:

```bash
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP SINGLE_NODE=1 SAVE_TAG=code_1node_$USER \
  ROLLOUT_MAX_RESPONSE_LEN=16384 EVAL_MAX_RESPONSE_LEN=16384 \
  SGLANG_MAX_RUNNING_REQUESTS=128 SAVE_INTERVAL=10 EVAL_INTERVAL=10
```

### Dr. GRPO loss (`DRGRPO=1`, opt-in)

By default the policy loss averages each sample over its own length, so a token in a long
sample gets less weight. `DRGRPO=1` sums over tokens and divides by a constant
(`drgrpo_reducer.py`), so every token counts the same. It needs `DRGRPO_DIVISOR`, the step-0
mean response length (10686 for the DPO base at a 16k cap; measure it again for another cap or
checkpoint), and should be paired with `OVERLONG_FILTER=1`:

```bash
... cluster/launch.sh $HEAD_IP SINGLE_NODE=1 SAVE_TAG=code_drgrpo_$USER \
  ROLLOUT_MAX_RESPONSE_LEN=16384 EVAL_MAX_RESPONSE_LEN=16384 \
  DRGRPO=1 DRGRPO_DIVISOR=10686 OVERLONG_FILTER=1
```

Measured on the reference run's rollouts (52 steps, 13,312 samples): 28% of samples truncate,
at 1.5x the mean length and mean reward 0.03. Under Dr. GRPO they would carry 41% of the
gradient (29% today), which is why the overlong filter drops them and recentres the rest of
the group; 71% of groups keep a reward spread after that. Only 0.8% of samples finish
without `</think>`, so the math recipe's extra unclosed-think filter is not needed here.

Check `train/grad_norm` in the first 3 steps: it should be the same order as the reference
run's ~0.04-0.06. `overlong_filter: ... usable=` in the log is the number of groups still
contributing; if it collapses, the filter is starving the update.

`SINGLE_NODE=1` splits the node into 4 trainer GPUs (EP4) and 4 rollout GPUs (one sglang
engine). For 3 nodes: bring up all three, run `env_setup.sh` on each, `ray_head.sh` on the
head, `cluster/ray_worker.sh $HEAD_IP` on the other two, then launch without `SINGLE_NODE`.

### 7. Watch it

```bash
L=$(ls -t $NFS_ROOT/rl-logs/launch-code*.out | head -1)
grep "CODE_REWARD scored" $L | tail -1     # api_error must stay 0
grep -o "'rollout/[a-z_]*': [0-9.e-]*" $L | tail -4
grep -o "'eval/code-heldout': [0-9.]*" $L
$I/cluster/code_exec.sh status
```

Stop a run from inside the container: `RAY_ADDRESS=http://$HEAD_IP:8265 ray job list`, then
`ray job stop <id>`. **Relaunching with the same `SAVE_TAG` resumes** from the newest
`iter_*` in `$CKPT_ROOT/Instella-MoE-16B-A3B-DPO_<SAVE_TAG>`; a new tag starts from the DPO base.

### What a healthy run looks like (1 node, 16k context, reference run `miles_code16k_1node`)

| | |
|---|---|
| step time | ~20 min (rollout-bound; the train step itself is ~2 min, weight sync ~3 s) |
| train reward | 0.45 at step 1, 0.5-0.6 by step 30 |
| `eval/code-heldout` | 0.309 @10, 0.328 @20, 0.324 @30, 0.355 @40 |
| eval cost | ~30 min per eval, during which training waits |
| checkpoint | ~207 GB, ~66 s to save |

Things learned on that run:

- **Truncation is the main loss.** At 16k, 25-40% of training responses and ~40% of eval
  responses hit the limit mid-thought. Those fail the think gate (`think_gate_fail` in the
  `CODE_REWARD` line) and score 0. The policy always opens `<think>`; the gate failures are
  truncation, not a template problem. Hence the 32k default.
- **Disk.** The checkpoint volume is shared and fills up from other users too. `run_code.sh`
  keeps the newest checkpoint plus the 2 best by `eval/code-heldout` (`KEEP_RECENT=1`,
  `KEEP_BEST=2`), i.e. at most ~620 GB. Restart the pruner with lower values if space runs out.
- **Occasional router `/generate` 500s** (`httpcore.ReadError` tracebacks from the
  RolloutManager) are retried automatically and are harmless unless the retry count climbs.
- The sandbox has the standard library plus PyYAML only; no sampled solution imported
  numpy or other third-party packages, so none are installed.

## How model/weights wire up (no `--spec`, no shadow ckpt)

- **FarSkip layers** are built by Megatron's `get_gpt_decoder_block_spec` from
  `--use-simple-farskip-layer` (a config flag). This Megatron **asserts `--spec is None`
  for MoE**, so we pass no `--spec`.
- **gate_proj weight mapping**: Instella reports `model_type: deepseek_v3`, so
  `AutoBridge` routes to the stock `DeepseekV3Bridge`. `patch_mbridge_gate_proj.sh`
  teaches that bridge the extra `self_attention.linear_gate_proj.weight ↔
  model.layers.{n}.self_attn.gate_proj.weight` key (idempotent, mirrors Primus's
  `rl/setup/mbridge_gate_proj_patch.sh`). Used for both HF→torch_dist convert and
  RL weight sync. No custom bridge / shadow checkpoint needed.
- sglang is unaffected (keys off `architectures=InstellaMoEForCausalLM` via the overlay).

## Single-node smoke test (1× 8-GPU node) — DO THIS FIRST

Fully-async can't colocate (`train_async.py` asserts `not args.colocate`), so a single
node is split **4 trainer (EP4) + 4 rollout (1 engine EP4/DP4)**. `SINGLE_NODE=1` starts a
local Ray head, cleans stale procs, and applies the reduced topology automatically.

**Which container?** Run everything below in the **miles training container** (the one
with Ray + miles installed, e.g. `gora-async` / `rocm/sgl-dev:miles-rocm720-mi35x`) — NOT
the sglang-only lm_eval container (`lmsysorg/sglang-rocm:*`), which has sglang/aiter but
**no Ray or miles**, so `run_async.sh` will die with `ray: command not found`. Verify with
`source /opt/venv/bin/activate && which ray`.

```bash
# In the miles container:  source /opt/venv/bin/activate; cd ~/miles
# Paths (MEGATRON_PATH, CKPT_ROOT, ...) come from paths.sh; wandb from ~/.netrc or WANDB_KEY.

# 1. FarSkip Megatron (once; needs git):        bash examples/instella_think/investigate_megatron.sh
# 1b. miles<->old-Megatron import shims:         bash examples/instella_think/patch_megatron_compat.sh
# 2. sglang FarSkip overlay:                     bash examples/instella_think/setup_farskip.sh
# 3. mbridge gate_proj patch (once/container):   bash examples/instella_think/patch_mbridge_gate_proj.sh
# 4. FarSkip + mbridge smoke (rung 1):           bash examples/instella_think/validate.sh
# 5. Convert HF -> torch_dist (uses all 8 GPU):
PHASE=prepare bash examples/instella_think/run_async.sh
#    (HF_CKPT defaults to $CKPT_ROOT/Instella-MoE-16B-A3B-DPO;
#     override it only to point at a checkpoint somewhere else)
# 6. Tiny end-to-end RL smoke (few steps, eval off):
SINGLE_NODE=1 NUM_ROLLOUT=8 EVAL_INTERVAL=100000 \
bash examples/instella_think/run_async.sh
```

## Full run (3 nodes / 24× MI355X, disaggregated: 1 trainer + 2 rollout)

```bash
# 0. Bring up 3 containers + Ray cluster (cluster/bringup.sh, cluster/env_setup.sh on EVERY
#    node, cluster/ray_head.sh + cluster/ray_worker.sh; see "Code RL ... step by step").

# 5. Train (fully async, staleness 4):
RAY_ADDRESS=http://<HEAD_IP>:8265 MASTER_ADDR=<HEAD_IP> \
WANDB_KEY=... bash examples/instella_think/run_async.sh
```

## Instruction-following run (32k, same hyperparams as `rytwnj5z`)

Task swap only. Every learning hyperparameter is inherited from `run_async.sh`, whose
defaults are exactly the logged config of
[`rytwnj5z`](https://wandb.ai/genaiamd/miles-mis/runs/rytwnj5z) — 32k lengths, GRPO with
`kl-coef 0` / `entropy-coef 0` / `eps-clip 0.2` / `eps-clip-high 0.28` / `--use-tis`,
`rollout-batch-size 32` × `n-samples-per-prompt 8`, `global-batch-size 256`, lr 1e-6
constant, adam 0.9/0.98, EP8, `max-tokens-per-gpu 8192`, fp32 LM head on both sides.

```bash
# 1. Per node: the usual env setup plus the 4 checker deps + nltk punkt on NFS:
#   examples/instella_think/cluster/on_node.sh <J> <node> cluster/env_setup.sh --if
# or in-container:
bash examples/instella_think/setup_if_deps.sh
bash examples/instella_think/setup_if_deps.sh --verify

# 2. Once, anywhere with network + pyarrow/datasets (i.e. in-container):
python3 examples/instella_think/prepare_if_data.py inspect --source dolci   # look first
python3 examples/instella_think/prepare_if_data.py build --out-dir "$DATA_ROOT"

# 3. Smoke, then train:
SINGLE_NODE=1 NUM_ROLLOUT=8 bash examples/instella_think/run_if.sh
bash examples/instella_think/run_if.sh
```

### Datasets (built, verified)

| Set | Rows | Source | Mapped |
|---|---|---|---|
| train | 29,813 | `allenai/Dolci-Think-RL-7B`, `IF_multi_constraints` subset | 29,813 / 29,813 |
| ifeval | 541 | `google/IFEval` | 541 / 541 |
| ifbench | 300 | `allenai/IFBench_test` | 300 / 300 |

Two traps were found here, both of which would have produced a flat-zero reward rather than
an error:

**Dolci nests the constraint spec two deep.** `ground_truth` is a *list holding one string*,
and that string is a python-literal list of dicts:
`["[{'instruction_id': ['detectable_format:json_format'], 'kwargs': [None]}]"]`. A string
element must therefore be parsed before it is judged; treating any colon-bearing string as
an instruction id swallows the whole repr as one bogus id. Dolci is also grouped by
`dataset_source`, so a head-of-dataset scan sees only math rows and finds zero IF prompts.

**IFBench needs its own registry.** All 300 IFBench prompts use a newer constraint family
(`sentence:keyword`, `words:consonants`, `ratio:stop_words`, `format:sub-bullets`, …) that
IFEvalG does not implement, so that eval set would have scored 0.0 everywhere and looked
like total model failure. `if_reward.py` now consults IFBench's `INSTRUCTION_DICT`
(83 types, cloned to `$NFS_ROOT/IFBench`, `IF_IFBENCH_DIR`) for ids IFEvalG
lacks; IFEvalG stays primary so training never depends on that clone. This is why
`emoji` and `syllapy` are installed.

### ⚠ Checkpoints at `--save-interval 20`: pruning is mandatory

One checkpoint is **~207 GB** (16 torch_dist shards, weights + optimizer), and this miles
has **no retention** — the only mention of `save_retain_interval` is
`args.save_retain_interval = None` (`arguments.py:2997`); nothing reads it. At save-every-20
over 3000 rollouts that is 150 checkpoints ≈ **31 TB**, on a `/shared_nfs` already at 98%
(~11 TB free) and shared with the team. Filling it makes writes fail cluster-wide.

`run_if.sh` therefore refuses to launch if fewer than 8 checkpoints fit, and starts
`prune_checkpoints.py` alongside training.

**Retention keeps the best, not just the newest.** Recency alone is the wrong policy here:
these runs do not improve monotonically — the reference math run peaked at AIME ≈0.756 by
step 39 and had decayed to 0.635 by step 179 — so the checkpoint you want is usually *not*
among the newest few. The pruner keeps the union of:

- whatever `latest_checkpointed_iteration.txt` points at (a resume must never break)
- the newest `KEEP_RECENT` (default 3)
- the top `KEEP_BEST` (default 5) by `PRUNE_METRIC` (default the mean of `eval/ifeval` and
  `eval/ifbench`)

Scores come from the training log, not the wandb API: miles prints
`eval 39: {'eval/ifeval': 0.41, ...}` (`miles/ray/rollout/metrics.py:38,53`), and that id is
the `rollout_id`, which *is* the checkpoint's iteration number — so `eval 39` pairs with
`iter_0000039` with no auth, no network and no guessing. Parsed scores are cached in
`<SAVE_DIR>/.ckpt_scores.json` so log rotation or a pruner restart doesn't lose history.

If no scores can be parsed at all, best-retention goes **inactive** and it falls back to
recency with a loud warning — an empty score set is never read as "everything scored zero".
Tune with `KEEP_RECENT` / `KEEP_BEST` / `PRUNE_METRIC`; `PRUNE_CHECKPOINTS=0` disables it;
`--dry-run --once` shows what it would remove.

### Container lifetime

`cluster/node_setup.sh` starts the container detached (`sleep infinity`) with a
`spur_job_id` label and returns; nothing holds it open. Spur's epilog removes labeled
containers when the job ends, so there is no watchdog and no manual cleanup. Patches applied
inside the container (`env_setup.sh`) are lost with it, so re-run that step on every new job.

### What was changed in `run_async.sh` (defaults preserve the math run byte-for-byte)

Four narrow env hooks, nothing else: `RM_TYPE`/`CUSTOM_RM_PATH` (reward routing),
`EVAL_PROMPT_DATA` (name/path pairs), `N_SAMPLES_PER_EVAL_PROMPT`, `WANDB_GROUP`, plus
`IF_*`/`NLTK_DATA` passthrough in the Ray `runtime_env`. Unset, every one of them
reproduces the previous hardcoded value.

The `IF_*` passthrough is not cosmetic: the reward executes in the Ray rollout workers,
whose environment is that `runtime_env` dict, so exporting `IF_REQUIRE_THINK=0` in the
launching shell alone would be silently ignored.

### Things the old `Instella-MoE/rl` IF example got wrong or can no longer do

Checked rather than copied, since that repo pins `radixark/miles@2b5b7fc` plus a 761-line
core patch that does not apply here.

- **`REWARD_SCALE=10.0` is not portable.** It only works with their
  `miles_core_changes.patch`, which redefines "perfect reward" as `_REWARD_PERFECT=10` in
  `miles/utils/metric_utils.py` and `log_utils.py`. Without those hunks a 10× reward makes
  miles' own pass-rate and eval-accuracy metrics read `7.28` for 72.8%, and multiplies every
  advantage by 10 against an unchanged `--eps-clip`. `if_reward.py` uses 1.0.
- **Its prompt-injection branch is dead code.** `reward_model.py:716-718` does
  `if "prompt" in checker.get_instruction_args(): build_description(prompt=...)`, but no
  checker has an arg named `prompt` — all five that need one use `prompt_to_repeat`. So
  `combination:repeat_prompt`, `copy:*` and `new:copy_span_idx` raised `ValueError` inside
  the bare `except: return False` and scored a silent 0. And that second
  `build_description` call would have discarded the first call's kwargs anyway.
  `if_reward.py` fills `prompt_to_repeat` from the prompt in one call.
- **`--balance-by-flops` does not exist in this miles** and the old launcher passes it.
  It is also already commented out in our `PERF_ARGS`.
- **Its dependency list is bigger than needed.** `emoji`, `syllapy` and `pylatexenc` are
  for IFBench's *extended* registry, which we do not vendor. `IFEvalG/` imports only
  `langdetect`, `absl`, `immutabledict`, `nltk`.
- **`grpo_iterations=2` in the reference config is a red herring** — it belongs to the
  Primus Megatron fork's own unused GRPO code (`training/arguments.py:1892`), not to miles,
  which has no such flag. Nothing to match.
- **`--rollout-task-type` is inert** in this miles: parsed at `arguments.py:2057` and read
  nowhere. Reward selection is `--custom-rm-path` / `--rm-type` / per-sample metadata only.
- **Its strict-vs-partial eval split could not have worked here.** It keys on
  `kwargs["evaluation"]`, but miles threads `evaluation` down to `generate_and_rm()` and
  then calls the reward without it (`sglang_rollout.py:354`, `:363`, `:396`), so that kwarg
  is *always* `False` for a custom RM — including on the eval path. `if_reward.py` therefore
  drives the split from `metadata.split`, which `prepare_if_data.py` stamps into the eval
  JSONLs and miles passes through untouched. The kwarg is still honoured if miles ever
  starts forwarding it.

### Why not the built-in `--rm-type ifbench`

miles ships one (`miles/rollout/rm_hub/ifbench.py`), but at **import** time it `git clone`s
`allenai/IFBench` into the repo's parent and `pip install`s into the live venv, on every
worker. That is a startup network dependency plus a mutating install inside a 24-GPU job.
It is also strict-binary only, so it cannot give the partial credit that keeps early
multi-constraint groups from being discarded as zero-variance by
`check_reward_nonzero_std`. Vendoring the checkers avoids both.

### Reading a flat reward

`if_reward.py` logs an `IF_REWARD` counter line every 200 samples that separates model
failure from wiring failure: `think_gate_fail` (policy not emitting `<think>`; try
`IF_REQUIRE_THINK=0`), `no_metadata` (JSONL lost its metadata column), `unknown_id`
(constraint the vendored checker lacks; rebuild with `--drop-unknown`), `checker_error`
(usually missing nltk `punkt`). Check that before touching hyperparameters.

## FarSkip contract (must match on both sides)
- **Training**: `--use-simple-farskip-layer` (reference connectivity; in `model_args.sh`).
- **Rollout**: `FARSKIP_REFERENCE_DECODER_LAYER=1` (in the Ray `runtime_env` of `run_async.sh`).
Keeping both on the *reference* FarSkip path is what makes train/infer numerics agree
(this is the train/infer-mismatch investigation this port targets). Switch both to the
overlapped path together once reference parity is confirmed.

## Core miles edits (no plugin seam exists for these)
The RL weight-sync converter (`miles/backends/megatron_utils/megatron_to_hf/`) is a
hand-rolled, hardcoded dispatcher with no registration hook, so Instella support is a
minimal core edit:
- `megatron_to_hf/__init__.py`: route `instella*` model names to the DeepSeek-V3 converter.
- `megatron_to_hf/deepseekv3.py`: add the gated-attention `self_attention.linear_gate_proj.weight`
  → `self_attn.gate_proj.weight` mapping (harmless to plain DeepSeek-V3, which lacks that weight).

No atomic-update-group change is needed: Instella has `q_lora_rank=None`, so
`get_atomic_update_groups` returns empty (full `q_proj`, no fused q-lora group).

## Notes on the miles core-fixes analysis (from the old Instella-MoE/rl pin)
Most old `miles_core_changes` hunks are unnecessary on latest miles:
- `HIP_VISIBLE_DEVICES` fallback already present (`miles/.../sglang_engine.py`).
- sglang arg-name shims not needed — latest miles uses `--sglang-dp-size` / `--sglang-ep-size`
  and auto-derives `--sglang-*` from `ServerArgs`.
- Vocab padding: instead of the old fork's miles-core `_target_vocab_for_hf` patch (which
  would wrongly export TP-padding rows for other models), we pass `--vocab-size 128896`.
  Instella's HF config vocab_size *is* 128896 (real weights, not padding), so stock
  `remove_padding(args.vocab_size)` keeps all rows and matches sglang. Fix stays local.
- Router-revive / `--balance-by-flops` left **optional** (flags commented in `run_async.sh`);
  enable only if the corresponding core change exists in this miles build.

## Known issue: aiter MoE GPU memory fault on gfx950 (MI355X)

**Symptom:** with `SGLANG_USE_AITER=1`, the rollout engine dies with
`Memory access fault by GPU node-X … on address 0x…… Reason: Unknown` (often all/most
rollout GPUs, at varying tiny addresses), during the **first large/uneven DP-attention
prefill**. Because `--skip-eval-before-train` puts that first big prefill *after* the
first weight sync, it looks sync-triggered — but the weight sync itself completes fine.

**This is NOT a bug in the RL/weight-sync/FarSkip code.** What was ruled out (all on
gfx950, in-engine):
- Not negative/invalid `topk_ids` (live: ids are global `0..num_experts-1`, none negative).
- Not bad MoE values/shapes/indices: the *exact* dumped faulting `fused_moe` inputs
  (M≈13k) replay **cleanly** in isolation at every batch size.
- Not a stale/rebound weight pointer: live `w13`/`w2` are contiguous with correct strides
  and full storage right before the fault.
- It is **intermittent** (varying near-null fault addresses; some runs pass the first big
  prefill and fault on a later one) → signature of a small **out-of-bounds read inside an
  aiter kernel** that only faults when the OOB page is unmapped. A fresh-allocator replay
  never hits it. Mirrors upstream [sglang #27987](https://github.com/sgl-project/sglang/issues/27987)
  (uninitialized MoE routing/permute padding slots) and the AMD+DP-attention+aiter fault
  family ([#7691](https://github.com/sgl-project/sglang/issues/7691),
  [#21942](https://github.com/sgl-project/sglang/issues/21942)) — none of which patches
  this EP4 standard-CK + DP-attention prefill path.

**Status: `SGLANG_USE_AITER=1` is now the default and is REQUIRED for correctness.**
`SGLANG_USE_AITER=0` is *not* a safe mitigation: the sglang515 overlay stores MoE weights in
aiter's shuffled layout, so the non-aiter path reads them unshuffled and **silently** emits
gibberish (Primus-Instella-Internal#2: "2+2=5", `arc_challenge` 0.195, below the
random-chance floor) at normal-looking throughput — there is no crash to warn you.

The empty-KV out-of-bounds fault that previously forced `ENABLE_DP_ATTENTION=0` is fixed by
the `k.shape[0]==0` guard in sglang's `triton_backend` (commit `b54ceda`), verified by two
clean `lm_eval` runs at EP8/DP8 and by the healthy 1-node smoke. So `aiter=1` +
dp-attention is now both correct and stable, and `ENABLE_DP_ATTENTION=1` is the default
(it also keeps sglang's chunked-prefill clamp at 4096).

## Known issue: `--optimizer-cpu-offload` breaks optimizer-state resume (Megatron patch required)

**Symptom:** resuming any mid-training checkpoint *with* optimizer state dies in
`megatron/core/dist_checkpointing/serialization.py:126`:

```
ValueError: Cannot merge two lists with different lengths (1 and 116, encountered at
level ('optimizer', 0, 'param_state', 0, (torch.bfloat16, torch.float32), 0))
```

where the second length varies per rank (47 / 116 / 123 / …). Weights-only resume
(`--no-load-optim`) is unaffected.

**Cause:** `--optimizer-cpu-offload` makes the CPU sub-optimizer `torch.optim.AdamW`, which
keeps a **per-parameter `step` tensor**. TE's `FusedAdam` does not — its step lives in
`param_groups` — which is why plain `--use-precision-aware-optimizer` never hits this. The
`dp_reshardable` save path did `if key == 'step': continue`, so `step` was never wrapped as a
`ShardedTensor` and stayed a plain tensor. Plain tensors aren't `ShardedBase`, so
`extract_sharded_base` routes them into the **common** state dict — and `common.pt` is written
by **global rank 0 only**. The saved list therefore carries rank 0's bucket-entry count (1)
while every other rank expects its own. `dp_reshardable` was the only sharding format leaking
`step`; `fully_reshardable` and `fs_model_space` already skip it on save *and* load.

**No optimizer data is ever lost by this bug.** `exp_avg` / `exp_avg_sq` / `param` are all in
the `.distcp` shards at 100% coverage, and the authoritative scalar step is in the sharded
`param_groups`. Only a redundant, wrongly-shaped residue reaches `common.pt`.

**Fix (both parts):**

```bash
# 1. Future saves — once per container, since Megatron-LM lives in the image, not on NFS:
MEGATRON_PATH=/root/Megatron-LM python3 examples/instella_think/patch_megatron_optim_step.py

# 2. Checkpoints already written without the patch (drops the residue; .distcp untouched):
python3 examples/instella_think/fix_ckpt_common_optimizer_state.py <ckpt>/iter_0000039 --apply
```

Part 1 is upstream `f4502eb1c`, "Handle `step` key correctly in checkpoint save with
`--optimizer-cpu-offload`" (PR #2874), which is **not** an ancestor of this fork's
`miles-main`. It should be cherry-picked properly — the combination
`--optimizer-cpu-offload` + `--use-precision-aware-optimizer` has **no coverage** in
`tests/unit_tests/dist_checkpointing/`, so nothing catches this.

**Do NOT try to fix it by padding or truncating lists in `dist_checkpointing`'s `merge()`.**
That was attempted: it merges and loads cleanly, then dies on the *first optimizer step* with
`RuntimeError: a Tensor with 32 elements cannot be converted to Scalar` in `_fused_adam`,
because padding entries build `step` from `torch.empty(gap)` — uninitialized garbage of the
wrong shape. The checkpoint's length-1 entry is *correct*; the model-side 116 is a different
axis, so no merge-level change can reconcile them.

Verified working end to end: iteration 39 resumed with optimizer state and trained on past the
step where the pre-fix attempt crashed, with `grad_norm` continuous across the restart
(0.1486–0.1597 vs 0.1314 before) rather than spiking as it does on an Adam reset.

## Risks to confirm in-container (see `validate.sh`)
1. mbridge mapping completeness — `gate_proj` / full `q_proj` (q_lora_rank=null) must map
   with no "Unsupported parameter name" during convert.
2. FarSkip Megatron availability — `investigate_megatron.sh` must produce importable
   `SimpleFarSkipTransformerLayer`.
3. Weight-sync broadcast under `in_place` pause — **confirmed working** (sync completes,
   values bit-exact; the post-sync crash is the aiter MoE fault above, not sync).
   `torch_memory_saver` and the old sglang weight-loader/aiter patches are OPT-IN
   (`BUILD_TMS=1`), only needed if the broadcast hangs or you hit
   `'Parameter' object has no attribute 'weight_loader'`.
