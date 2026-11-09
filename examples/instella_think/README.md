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
| `run_code.sh` | **Code RL arm.** Dolci-Think-RL `code_rlvr`, unit-test reward, HumanEval+/MBPP+/LiveCodeBench v3 evals. Wrapper over `run_async.sh`. |
| `code_reward.py` | The code reward (`--custom-rm-path code_reward.batched_reward`). Pass rate while training, all-or-nothing at eval. Self-tests with `python3 code_reward.py` (needs the sandbox). |
| `code_exec_server.py` | Stdlib-only code-execution sandbox for a CPU host (rlimits + audit-hook guard, no containers needed). Started by `cluster/code_exec.sh`. |
| `prepare_code_data.py` | Builds the code train JSONL from Dolci. |
| `prepare_code_benchmarks.py` | Builds HumanEval+, MBPP+ and LiveCodeBench v3 as eval sets with OLMo 3's oe-eval protocol. |
| `difficulty_filter.py` | OLMo 3 offline difficulty filter: 8 rollouts per prompt from the starting checkpoint, graded by the run's own reward, drops prompts solved more than 62.5% of the time. `--task code` or `--task math`. Resumable. |
| `prefilter.sh` | Runs the filter on a node: `serve` (sglang with the rollout engine's settings), `filter`, `status`, `stop`. See "Optional: difficulty-prefilter the training set". |
| `overlong_filter.py` | Optional drop-and-recentre overlong filter (`OVERLONG_FILTER=1`); `OVERLONG_FILTER_DROP_UNCLOSED=1` also drops samples that stopped without closing `</think>`. |
| `router_log_hook.py` | Optional trainer-side router telemetry (`ROUTER_LOG=1`): router weight and expert-bias drift from step 0, as JSONL. |
| `adaptive_difficulty_filter.py` | Optional online difficulty filter (`ADAPTIVE_FILTER=1`): drops groups the current policy has already mastered. See "Other launch knobs". |
| `per_domain_logger.py` | Per-source W&B metrics (`domain/<dataset_source>/*`), on by default (`PER_DOMAIN_METRICS=0` disables). |
| `drgrpo_reducer.py` | Optional Dr. GRPO constant-denominator policy loss (`DRGRPO=1`, needs `DRGRPO_DIVISOR`). |
| `model_args.sh` | `MODEL_ARGS=(...)` + `MOE_LAYER_FREQ` (layer 0 dense, 1–26 MoE). MLA + `--gated-attention` + `--use-simple-farskip-layer`. **No `--spec`** (this Megatron forbids it for MoE; FarSkip is config-driven). |
| `run_async.sh` | Main launcher. `PHASE=prepare` (HF→torch_dist) then `PHASE=train` (submit async DAPO-Math job to Ray). `SINGLE_NODE=1` for the 1-node smoke. |
| `investigate_megatron.sh` | Assemble a FarSkip-capable Megatron at `MEGATRON_PATH` (clone NVIDIA pin `847781764` + overlay patch tree + `pip install -e`). |
| `patch_megatron_r3.sh` | Add miles' routing-replay hooks to that Megatron, which lacks them. Needed for `USE_ROUTING_REPLAY=1`; `cluster/env_setup.sh` runs it, and `run_async.sh` refuses R3 without it. |
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
# The built code data is also readable; skip step 4's data builds by pointing at it:
CODE_TRAIN=/shared_nfs/gramesh/data/code-train/dolci-code-train.jsonl
CODE_BENCH_DIR=/shared_nfs/gramesh/data/code-bench
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
$I/cluster/on_node.sh $J $NODE cluster/env_setup.sh      # sglang FarSkip overlay, Megatron shims + R3 hooks, mbridge patch
```

Expect `ALL OK`. Spur allows one interactive step per node, so run `on_node.sh` calls one
after another, never in parallel on the same node.

### 4. Once per user (skip what step 0 already points at)

Inside the container (`ssh -p 18822 $USER@$NODE`, `source /opt/venv/bin/activate`, then
the shell prelude above):

```bash
bash $I/investigate_megatron.sh                     # FarSkip Megatron into ~/Primus-Instella-Internal
bash $I/patch_megatron_r3.sh                        # its routing-replay hooks (env_setup.sh also runs this)
PHASE=prepare bash $I/run_async.sh                  # HF -> torch_dist (8 GPU), only if REF_LOAD is unset
HF_HOME=$NFS_ROOT/hf_cache python3 $I/prepare_code_data.py build   # -> $DATA_ROOT, only if CODE_TRAIN is unset
python3 $I/prepare_code_benchmarks.py --out-dir $CODE_BENCH_DIR     # eval sets, skip if step 0 points CODE_BENCH_DIR at the shared copy
```

### 5. Sandboxes on the CPU host

```bash
$I/cluster/code_exec.sh start         # training reward; creates ~/.code-exec-token
$I/cluster/code_exec.sh start bench   # benchmark evals, port 18951; builds ~/.code-exec-venv (numpy) on first start
$I/cluster/code_exec.sh status; $I/cluster/code_exec.sh status bench   # /health must answer
```

### 5b. Optional: difficulty-prefilter the training set

OLMo 3's offline difficulty filter (arXiv 2512.13961, "Step 2"): sample 8 rollouts per prompt
from the checkpoint RL starts from, at the rollout's own sampling settings (T=1.0, top_p=1.0),
and drop every prompt solved more than 62.5% of the time (6, 7 or 8 of 8). Those prompts give
almost no gradient while still costing full generations. Never-solved prompts (0/8) are kept,
as in the paper. In-training, `check_reward_nonzero_std` still drops groups that come back
all-right or all-wrong.

`difficulty_filter.py` takes everything that decides a reward from the training path, so a
pass rate here describes the run you will launch:
- **Prompt:** rendered with miles' `apply_chat_template` and sent to sglang as `input_ids`,
  as `sglang_rollout.py` sends it.
- **Sampling:** the rollout's sampling params, with special tokens kept and no stop trimming.
- **Grading:** `code_reward.train_all_pass`, which applies the reward's think gate, code
  extraction and sandbox. A solve means every test passes. Training gives partial credit, and
  counting reward > 0 as a solve would treat one passing test as solved.

Run it on a node that is not training: the server takes all 8 GPUs. It needs step 3's
container setup and the training sandbox from step 5.

```bash
$I/cluster/on_node.sh $J $NODE prefilter.sh serve                   # sglang on HF_CKPT, detached
$I/cluster/on_node.sh $J $NODE prefilter.sh status                  # wait for "health OK" (a few minutes)
$I/cluster/on_node.sh $J $NODE prefilter.sh LIMIT=200 filter        # pilot: read the histogram first
$I/cluster/on_node.sh $J $NODE prefilter.sh filter                  # full set; resumes from the pilot's cache
$I/cluster/on_node.sh $J $NODE prefilter.sh status                  # graded N / total, log tails
$I/cluster/on_node.sh $J $NODE prefilter.sh stop
```

The output goes to `$DATA_ROOT/prefilter/<input name>.filtered.jsonl`. Train on it with
`CODE_TRAIN=$DATA_ROOT/prefilter/dolci-code-train.filtered.jsonl` (a `launch.sh` override, or
`~/.instella-env.sh`). Each kept row gains `metadata.prefilter = {n, n_correct, n_truncated}`,
so a different threshold can be applied later without regenerating. The histogram, drop
counts and truncation audit are in `<output>.stats.json` and at the end of the log
(`$NFS_ROOT/rl-logs/prefilter/`).

**Settings that must match the run.** `MAX_NEW_TOKENS` defaults to `ROLLOUT_MAX_RESPONSE_LEN`
(32768), and `HF_CKPT` to the same DPO checkpoint as `run_async.sh`. Change them together with
the run: at a shorter budget, truncated rollouts score 0 and every pass rate is understated. The
filter warns when more than 20% of rollouts truncate. `CODE_REQUIRE_THINK`, `CODE_MAX_TESTS` and
`CODE_EXEC_TIMEOUT` take `run_code.sh`'s defaults. All of these are recorded in the cache's
first line, and a rerun under different settings refuses to reuse the cache (use a new
`OUTPUT=` or `RESET_CACHE=1`).

**Failures are retried, never banked.** If generation fails or the sandbox returns
`api_error`, the prompt is retried after the server answers again. It is not recorded as 0/8,
because 0/8 prompts are kept and an outage would otherwise fill the set with prompts that only
looked hard. After 32 consecutive failed prompts the filter exits; `prefilter.sh filter` then
resumes from `<output>.cache.jsonl`.

**Cost.** This has not been run at full scale yet. 21k prompts x 8 rollouts is ~170k
generations at up to 32k tokens. At 11k tokens per response, the step-0 mean at a 16k cap
(responses at 32k run longer), that is at least 2B tokens: 1.5-2 days on one node if decode
runs at twice the 4-GPU rollout engine's ~6,900 tokens/s. The pilot's log prints prompts/hour, so measure it before committing
a node. `CONCURRENCY` (default 16 prompts, i.e. 128 sequences) is bounded by KV memory and by
sandbox capacity: each finished prompt sends 8 programs at up to 32 tests each to the sandbox.

Math sets: `TASK=math INPUT=<PROMPT_DATA> RM_TYPE=<the run's --rm-type>` grades with miles'
`rm_hub`, the same function `--rm-type` selects in training.

`FARSKIP_SETUP=<script>` applies the FarSkip sglang patches before serving. It is unset here
because step 3's `env_setup.sh` already ran `setup_farskip.sh` in this container; set it for
containers set up another way (e.g. the MI325X patch set).

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

`DRY_RUN=1` prints the runtime env (with `HF_TOKEN` and `WANDB_API_KEY` redacted) and every
`train_async.py` flag. It then runs miles' own `parse_args` on them and imports every custom
function path (reward, filter, loggers, rollout), and exits without touching Ray or the GPUs.
That parse is the only check that sees the `--sglang-*` flags, which come from the installed
sglang, so a bad flag fails here instead of after the cluster is up. It needs the training
container; `DRY_RUN_PARSE=0` skips it elsewhere. Some flags appear twice (e.g. from
`model_args.sh` and an override); argparse keeps the last one, and the parse prints the
resolved values of the important ones.

The reference run below is exactly:

```bash
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP SINGLE_NODE=1 SAVE_TAG=code_1node_$USER \
  ROLLOUT_MAX_RESPONSE_LEN=16384 EVAL_MAX_RESPONSE_LEN=16384 \
  SGLANG_MAX_RUNNING_REQUESTS=128 SAVE_INTERVAL=10 EVAL_INTERVAL=10
```

### Longer responses (`CONTEXT_LEN`)

`CONTEXT_LEN` (default 32768) is the prompt+response budget for both Megatron (`--seq-length`)
and sglang. The checkpoint's `config.json` says 32768, so above that `run_async.sh` also passes
`--sglang-context-length` and sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (without it
sglang refuses to start). For ~61k responses:

```bash
... CONTEXT_LEN=63488 ROLLOUT_MAX_RESPONSE_LEN=61440 EVAL_MAX_RESPONSE_LEN=61440
```

Not yet run on this setup: a 61k sample is one microbatch on one GPU (CP=1), with ~32 GB of
fp32 logits, so smoke-test it with `SINGLE_NODE=1 NUM_ROLLOUT=2` first.

### Dr. GRPO loss (`DRGRPO=1`, opt-in)

By default the policy loss averages each sample over its own length, so a token in a long
sample gets less weight. `DRGRPO=1` sums over tokens and divides by a constant
(`drgrpo_reducer.py`), so every token counts the same. It needs `DRGRPO_DIVISOR`, the step-0
mean response length (10686 for the DPO base at a 16k cap), and should be paired with
`OVERLONG_FILTER=1`. For any other cap or checkpoint, measure it: launch the same config with
`DRGRPO=0`, read `rollout/response_len/mean` for the first 2-3 rollouts from the log, stop the
job before its first checkpoint, then relaunch with `DRGRPO=1` and a fresh `SAVE_TAG`. It cannot
be extrapolated from a shorter cap, because truncated samples' true lengths are unknown.
`NUM_ROLLOUT=1` is not a shortcut: miles always checkpoints the final rollout (~207 GB).

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

### Other launch knobs

All are `VAR=VALUE` overrides to `cluster/launch.sh`; defaults leave the recipe unchanged.

| Variable | Default | Effect |
|---|---|---|
| `ROLLOUT_TEMPERATURE` | 1 | Training sampling temperature. On math, the DPO base fell from pass@1 0.771 at 0.8 to 0.454 at 1.1. |
| `OVER_SAMPLING_BATCH_SIZE` | 64 | Must be at least `ROLLOUT_BATCH_SIZE`; miles refuses to start otherwise. |
| `MAX_TOKENS_PER_GPU` | 8192 | Trainer dynamic-batching budget per microbatch. |
| `ADVANTAGE_ESTIMATOR` | grpo | `gspo` only differs when there is more than one optimizer step per rollout. |
| `ADAPTIVE_FILTER` | 0 | Online difficulty filter, see below. |
| `OBSERVE_ENTROPY` | 1 | Logs the policy entropy (`train/entropy_loss`). With `--entropy-coef 0`, miles otherwise logs it as 0. |
| `PER_DOMAIN_METRICS` | 1 | Per-source metrics under `domain/<dataset_source>/`, see "Watch it". |
| `DUMP_TRAIN_DATA` | 0 | Trainer-side per-rank tensors per step, next to the rollout dumps. One file per rank per step, so use it for a few steps only. |
| `OVERLONG_FILTER_DROP_UNCLOSED` | 0 | With `OVERLONG_FILTER=1`, also drop samples that finished under the cap without closing `</think>` (a stray low-probability EOS, always scored 0). See `overlong_filter._is_unclosed`. |
| `ROUTER_LOG` | 0 | Router telemetry to `ROUTER_LOG_PATH` (default `$LOG_DIR/router_log_<SAVE_TAG>.jsonl`) every `ROUTER_LOG_EVERY` (5) steps. |
| `WANDB_BASE_URL` | | Self-hosted W&B endpoint, e.g. `https://api.forge.coreweave.com`. |
| `EXTRA_ARGS` | | Any `train_async.py` flags, appended last so they win, e.g. `EXTRA_ARGS="--lr 2e-6 --save-interval 10"`. Values can't contain spaces. |

**W&B credentials** reach the workers as `WANDB_API_KEY` through the Ray runtime env (from
`WANDB_KEY`), or via the netrc. They are never passed as `--wandb-key`, because miles uploads
every parsed argument into the run's W&B config, where anyone with project access can read
it. Runs launched before this change have the key in their config, so rotate it.

**Online difficulty filter (`ADAPTIVE_FILTER=1`).** The default dynamic filter only drops
groups whose 8 rewards are identical, so a 7/8 group still trains at full generation cost.
`adaptive_difficulty_filter.py` also drops groups where more than `ADAPTIVE_FILTER_MAX_PASS`
of the samples are fully solved by the *current* policy, so prompts graduate out as training
masters them. The offline prefilter (5b) can't do this, because it only sees the starting
checkpoint; on math, 22% of groups had drifted back above its threshold by step ~339.
- *Solved* means the full reward (`SOLVED_REWARD`, which `run_code.sh` and `run_if.sh` set to
  their reward scale), i.e. every test or constraint passes. Partial credit doesn't count, or
  one passing test would make a group look mastered.
- The default threshold 0.75 drops 7/8 groups. 0.625 (the offline number) also drops 6/8, and
  on math that discarded 19-29% of groups every step, including the difficulty band AIME
  sits in. 0.875 drops nothing beyond the default filter at 8 samples.
- Cost: more groups are dropped, so each step pulls more prompts and takes longer. Watch
  `rollout/dynamic_filter/frac_kept`; if it collapses, the data has become too easy for the
  policy.

`SINGLE_NODE=1` splits the node into 4 trainer GPUs (EP4) and 4 rollout GPUs (one sglang
engine). For 3 nodes: bring up all three, run `env_setup.sh` on each, `ray_head.sh` on the
head, `cluster/ray_worker.sh $HEAD_IP` on the other two, then launch without `SINGLE_NODE`.

### Evals: OLMo 3's code benchmarks

Every eval runs HumanEval+, MBPP+ and LiveCodeBench v3, logged as `eval/humanevalplus`,
`eval/mbppplus` and `eval/lcb_v3`. Step 4 builds the data and step 5 starts their sandbox.
Prompts, answer extraction, program assembly, tests and timeouts are oe-eval's `::olmo3:adapt` tasks.
Sampling is OLMo 3 Think's: 10 samples, temperature 0.6, top_p 0.95, 32768 tokens. So each
value is their pass@1 and compares directly with the paper (arXiv 2512.13961, Table 16):

| | OLMo 3 Think 7B (SFT / DPO / final) | 32B (SFT / DPO / final) |
|---|---|---|
| HumanEval+ | 88.2 / 91.4 / 89.9 | 90.0 / 91.6 / 91.4 |
| MBPP+ | 63.2 / 63.0 / 64.7 | 66.7 / 67.2 / 68.0 |
| LiveCodeBench v3 | 67.8 / 75.1 / 75.2 | 75.8 / 81.9 / 83.5 |

The data needs no manual work. `prepare_code_benchmarks.py` downloads pinned Hugging Face
revisions (evalplus/humanevalplus, evalplus/mbppplus, livecodebench/code_generation_lite
`release_v3`), checks their sha256, and converts them. LiveCodeBench's ~5 GB of tests go
to `$CODE_BENCH_DIR/tests`, which the sandbox host must be able to read. Best-checkpoint
pruning ranks by the mean of the three.

The grading reproduces two oe-eval behaviours on purpose:
- A response with no `</think>` is still graded, because oe-eval searches the whole text.
  The training reward scores such a response 0.
- Model-written `if __name__ == "__main__":` blocks don't run for HumanEval+, MBPP+ or
  LiveCodeBench call-based problems.

Every canonical solution passes except three that fail under oe-eval too, so those are the
same ceilings as the paper's: HumanEval/32 (its own test fails the canonical solution), and
MBPP/4 and MBPP/6 (oe-eval's prompt cuts the type-annotated signature into a syntax error).

**Cost.** That's 1,154 problems x 10 samples = 11,540 generations at up to 32k tokens, and
training waits for the eval. The reference run's engine decodes ~6,900 tokens/s in total on
1 node (128 running requests). At an assumed 7-11k tokens per response (unmeasured on these
sets) that's 80-130M tokens, about **3-5 hours per eval on 1 node**, or 12-20 training steps.
`CODE_BENCH_N=1` costs about a tenth (roughly 40-60 min including the 32k-token tail) and is
the practical in-training setting. Keep `CODE_BENCH_N=10` for reported numbers, e.g. on
saved checkpoints. `CODE_BENCH_SETS` picks a subset. Watch `cluster/code_exec.sh status bench`: it
shares the CPU host with the training sandbox, and if `timeouts` climb, CPU contention is
failing correct LiveCodeBench solutions (6 s per test).

### 7. Watch it

```bash
L=$(ls -t $NFS_ROOT/rl-logs/launch-code*.out | head -1)
grep "CODE_REWARD scored" $L | tail -1     # api_error must stay 0
grep -o "'rollout/[a-z_]*': [0-9.e-]*" $L | tail -4
grep -o "'eval/\(humanevalplus\|mbppplus\|lcb_v3\)': [0-9.]*" $L
$I/cluster/code_exec.sh status; $I/cluster/code_exec.sh status bench
grep "domain metrics" $L | tail -1          # per-source solve rate, length, truncation
```

In W&B, besides miles' own `rollout/*`, `eval/*` and `train/*`:
- `domain/<dataset_source>/*`: one panel per training source. `solve_rate` counts full solves
  and `nonzero_rate` any partial credit; `resp_len_solved` and `resp_len_unsolved` show
  whether failures are long, and `truncation_rate` comes from the engine's finish status.
- `rollout/dynamic_filter/frac_<reason>`: the share of groups dropped by the dynamic filter
  for each reason (`zero_std_0.0` all wrong, `zero_std_1.0` all solved, `mastered_7of8` with
  `ADAPTIVE_FILTER=1`), plus `frac_kept`. These are comparable across steps; the raw counts
  in `rollout/async/*` are not.
- `train/entropy_loss`: the policy entropy on training tokens. A steady fall is the
  sharpening that preceded collapse in the math runs.

Stop a run from inside the container: `RAY_ADDRESS=http://$HEAD_IP:8265 ray job list`, then
`ray job stop <id>`. **Relaunching with the same `SAVE_TAG` resumes** from the newest
`iter_*` in `$CKPT_ROOT/Instella-MoE-16B-A3B-DPO_<SAVE_TAG>`; a new tag starts from the DPO base.

### What a healthy run looks like (1 node, 16k context, reference run `miles_code16k_1node`)

| | |
|---|---|
| step time | ~20 min (rollout-bound; the train step itself is ~2 min, weight sync ~3 s) |
| train reward | 0.45 at step 1, 0.5-0.6 by step 30 |
| eval cost | see "Evals" above; training waits while it runs |
| checkpoint | ~207 GB, ~66 s to save |

Things learned on that run:

- **Truncation is the main loss.** At 16k, 25-40% of training responses hit the limit
  mid-thought. Those fail the think gate (`think_gate_fail` in the
  `CODE_REWARD` line) and score 0. The policy always opens `<think>`; the gate failures are
  truncation, not a template problem. Hence the 32k default.
- **Disk.** The checkpoint volume is shared and fills up from other users too. `run_code.sh`
  keeps the newest checkpoint plus the 2 best by the benchmark mean (`KEEP_RECENT=1`,
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

### Math: the MI325X R1/R2 recipe

`run_async.sh` with no task overrides is the math run (DAPO-style prompts, `deepscaler`
grader, AIME eval). R1 is Dr. GRPO plus overlong filtering at a 56k cap with the router
frozen; R2 is R1 plus dropping unclosed-`</think>` samples. On this branch:

```bash
PROMPT_DATA=$DATA_ROOT/mix_v5_math.jsonl EVAL_CONFIG=<math eval yaml> SAVE_TAG=<fresh tag> \
DRGRPO=1 DRGRPO_DIVISOR=21680.0 PER_TOKEN_LOSS=0 OVERLONG_FILTER=1 OVERLONG_FILTER_DROP_UNCLOSED=1 \
ROLLOUT_MAX_RESPONSE_LEN=57344 CONTEXT_LEN=61440 ROLLOUT_TEMPERATURE=0.8 \
ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=16 GLOBAL_BATCH_SIZE=512 MAX_TOKENS_PER_GPU=16384 \
DISABLE_STD_NORM=1 USE_ROUTING_REPLAY=1 ROUTER_LOG=1 \
NUM_ROLLOUT=220 EVAL_INTERVAL=25 SAVE_INTERVAL=20 \
EXTRA_ARGS="--log-passrate" bash examples/instella_think/run_async.sh
```

- `DRGRPO_DIVISOR=21680` is the step-0 mean response length of the DPO-64k checkpoint at
  this cap. Measure it again for another checkpoint or cap (see "Dr. GRPO loss").
- `mix_v5_math.jsonl` and the `*-olmes.jsonl` eval sets aren't built by this repo; copy them
  from the MI325X share. R1/R2's eval YAML covered AIME 2024/2025, HMMT Feb 2025 and
  BeyondAIME at n=16, temperature 0.6, top_p 0.95, 65536 tokens.
- The async "retry" (`--async-unused-samples-handler retry` on MI325X's newer miles) is built
  in here: `examples/fully_async` always returns aborted and over-stale groups to the buffer.
  Don't pass that flag, or `--async-data-buffer-capacity-factor`; this miles has neither.
- R1 ran 40 rollout GPUs and R2 24; the 3-node layout here has 16. Add rollout nodes and
  `EXTRA_ARGS="--rollout-num-gpus <n>"` to match.
- Through `cluster/launch.sh`, pass the same variables as `VAR=VALUE` arguments plus
  `RUN_SCRIPT=run_async.sh` (its default is `run_code.sh`).
- Confirm with `DRY_RUN=1` first: it prints the resolved loss, filter and router settings.

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
