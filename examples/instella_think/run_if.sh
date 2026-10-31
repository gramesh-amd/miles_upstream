#!/bin/bash
# Instruction-following arm of the Instella-MoE-16B-A3B 32k RL runs.
#
# Same experiment as the math run logged at
#   https://wandb.ai/genaiamd/miles-mis/runs/rytwnj5z  (instella-moe-16b-async-s4-3node_FP32LMHEAD)
# with the task swapped from DAPO-Math/deepscaler to verifiable instruction following.
# Everything else is inherited from run_async.sh, whose defaults ARE that run's settings:
#
#   32768 seq-length / max-position-embeddings / rollout+eval max response len
#   grpo, kl-coef 0, entropy-coef 0, eps-clip 0.2, eps-clip-high 0.28, --use-tis
#   rollout-batch-size 32, n-samples-per-prompt 8, global-batch-size 256
#   lr 1e-6 constant, adam 0.9/0.98, weight-decay 0.1, clip-grad 1.0
#   TP1/PP1/CP1, EP8, max-tokens-per-gpu 8192, --use-dynamic-batch-size
#   1 trainer node (8 GPU) + 2 rollout nodes (16 GPU, 2 engines EP8/DP8), max-staleness 4
#   fp32 LM head on BOTH sides, dp-attention on, sglang mem-fraction 0.60
#   dynamic sampling filter check_reward_nonzero_std, over-sampling-batch-size 64
#
# I verified those against the run's own logged config rather than trusting the script,
# so this really is the same recipe. Two entries in that config are red herrings worth
# knowing about before you "match" them:
#   * grpo_iterations=2 is NOT a miles setting. It comes from the Primus Megatron fork's
#     own unused GRPO implementation (training/arguments.py:1892, default 2) and miles
#     never reads it. There is no --grpo-iterations in miles at all.
#   * disable_grpo_std_normalization is absent from that config but present in today's
#     miles parser, i.e. the flag was added after 20260820. The reference run therefore
#     ran WITH std normalization, which is what this script reproduces (DISABLE_STD_NORM
#     defaults to 0 in run_async.sh).
#
# DELIBERATE DIFFERENCES FROM THE MATH RUN, and why
#   1. Reward: --custom-rm-path if_reward.batched_reward instead of --rm-type deepscaler.
#      Partial credit while training, strict all-or-nothing at eval. See if_reward.py.
#   2. n-samples-per-eval-prompt 16 -> 1. AIME is 30 prompts/set; IFEval is 541. At 16
#      samples that is 8656 sequences per eval at a 32k cap, which would cost more than
#      the 20 training steps between evals. IFEval's own reported metric is single-sample
#      strict accuracy anyway, and Instella's eval_config_if_thinkrl.yaml also used 1.
#   3. save-interval 200 -> 50. Not a learning hyperparameter. The math run's own notes
#      say 200 was too coarse to locate a slow collapse (it left only iter_199/iter_399
#      across 410 steps), and 50 costs ~207 GB per extra checkpoint on the same disk.
#
# WHAT TO CHECK IN THE FIRST 20 MINUTES, in order. The IF reward has more ways to be
# silently zero than a math grader does, so do not read a flat reward as "no learning":
#   a. `grep IF_REWARD <log>` -- the reward module logs a counter line every 200 samples.
#      think_gate_fail high  => the policy is not emitting <think>...</think>; rerun with
#                               IF_REQUIRE_THINK=0 (and see the note in if_reward.py).
#      no_metadata high      => the JSONL lost its metadata column; rebuild the data.
#      unknown_id high       => constraint ids the vendored checker does not implement;
#                               rebuild with `prepare_if_data.py build --drop-unknown`.
#      checker_error high    => usually a missing nltk punkt; run setup_if_deps.sh --verify.
#   b. rollout/truncated_ratio. IF answers are short, so unlike math this should be near
#      zero at a 32k cap. If it is not, the model is padding or looping, not reasoning.
#   c. The fraction of groups the dynamic filter drops. Strict multi-constraint IF is hard,
#      so early groups can come back all-zero and get discarded as zero-variance, which
#      starves the update. Partial credit exists to prevent exactly that -- if the drop
#      rate is still very high, lower the constraint count in the training mix before
#      touching the loss.
#
# USAGE -- from inside the head node's container (the Ray head for this allocation).
#   One-time, once per node:  bash examples/instella_think/setup_if_deps.sh
#   One-time, any node:       python3 examples/instella_think/prepare_if_data.py build
#   Smoke on 1 node:          SINGLE_NODE=1 NUM_ROLLOUT=8 bash examples/instella_think/run_if.sh
#   3-node train:             bash examples/instella_think/run_if.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${SCRIPT_DIR}/run_async.sh"
[ -f "${TARGET}" ] || { echo "FATAL: ${TARGET} not found" >&2; exit 1; }
source "${SCRIPT_DIR}/paths.sh"

# --- Ray head for this allocation -------------------------------------------------
# Left unset so run_async.sh's own discovery runs; override when it cannot find the head.
export RAY_ADDRESS="${RAY_ADDRESS:-}"
export MASTER_ADDR="${MASTER_ADDR:-}"

# wandb credentials: WANDB_KEY if exported, else the wandb SDK's own netrc auth. An empty
# --wandb-key is not fatal; wandb then resolves WANDB_API_KEY or the netrc itself.
export WANDB_KEY="${WANDB_KEY:-}"
if [ -z "${WANDB_KEY}" ]; then
   # Do NOT test "${HOME}/.netrc": inside the training container everything runs as root,
   # so HOME is /root and the user's real credentials are invisible. That is how a run
   # loses all wandb logging while looking fine. Point NETRC at the absolute path instead
   # (honoured by requests/wandb regardless of HOME), and forward it to the workers via
   # run_async.sh's runtime_env.
   if [ -f "${NETRC_PATH}" ]; then
      export NETRC="${NETRC_PATH}"
      echo "NOTE: WANDB_KEY unset; wandb will authenticate via NETRC=${NETRC}."
   else
      echo "WARNING: no WANDB_KEY and no netrc at ${NETRC_PATH} -- metrics may not be logged." >&2
      echo "         Export WANDB_KEY=... if this run needs wandb." >&2
   fi
fi

# --- data -------------------------------------------------------------------------
IF_TRAIN="${IF_TRAIN:-${DATA_ROOT}/if-train/dolci-if-train.jsonl}"
IFEVAL="${IFEVAL:-${DATA_ROOT}/ifeval/ifeval-eval.jsonl}"
IFBENCH="${IFBENCH:-${DATA_ROOT}/ifbench/ifbench-eval.jsonl}"

missing=0
for f in "${IF_TRAIN}" "${IFEVAL}" "${IFBENCH}"; do
   [ -s "${f}" ] || { echo "MISSING: ${f}" >&2; missing=1; }
done
if [ "${missing}" = "1" ]; then
   cat >&2 <<EOF

FATAL: instruction-following data not built yet. Run (needs network, login node is fine):
   python3 ${SCRIPT_DIR}/prepare_if_data.py build --out-dir ${DATA_ROOT}
Inspect the source schema first if that reports a low yield:
   python3 ${SCRIPT_DIR}/prepare_if_data.py inspect --source dolci
EOF
   exit 1
fi

export PROMPT_DATA="${PROMPT_DATA:-${IF_TRAIN}}"
export EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-ifeval ${IFEVAL} ifbench ${IFBENCH}}"

# --- reward -----------------------------------------------------------------------
export CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-if_reward.batched_reward}"
# 1.0, not Instella's 10.0: that scale only works with their miles_core_changes.patch,
# which we do not carry. See the REWARD SCALE block in if_reward.py.
export IF_REWARD_SCALE="${IF_REWARD_SCALE:-1.0}"
export IF_PARTIAL_CREDIT="${IF_PARTIAL_CREDIT:-1}"
export IF_REQUIRE_THINK="${IF_REQUIRE_THINK:-1}"
export NLTK_DATA

# Fresh tag is mandatory: run_async.sh points --load at SAVE_DIR, so reusing an existing
# tag RESUMES that run's newest iter_* instead of starting from the DPO weights.
# --- run identity -----------------------------------------------------------------
# Names encode what distinguishes this run, because the three checkpoint dirs already on
# disk (miles_async_s4 / _olmo3 / _persample) are only tellable apart by their tag:
#   if32k  : instruction following at 32k context
#   nostd  : DISABLE_STD_NORM=1 (advantages centred, not std-normalised)
#   r3     : USE_ROUTING_REPLAY=1
# Its own wandb project too -- this run's eval axes are eval/ifeval and eval/ifbench, which
# have nothing to plot against the math project's eval/aime2024.
export SAVE_TAG="${SAVE_TAG:-miles_if32k_nostd_r3}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles-MI355-IF}"
# Deterministic run name: with the suffix on, this logs as "<group>_<random6>-RANK_0" and
# has to be renamed in the UI to say anything, which is how rytwnj5z ended up hand-labelled
# "_FP32LMHEAD" while its group string said nothing about the experiment.
export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"

# Core hyperparameters, named here so the run name below is BUILT FROM THE VALUES ACTUALLY
# PASSED rather than a hand-written label that can drift out of sync with them. Defaults are
# the reference run's (rytwnj5z) values.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export MAX_WEIGHT_STALENESS="${MAX_WEIGHT_STALENESS:-4}"
export LR="${LR:-1e-6}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
export FP32_TRAINER_READOUT="${FP32_TRAINER_READOUT:-1}"

# The run name itself is assembled AFTER the objective block below, because it encodes those
# toggles and they are not set yet here.
# Checkpoint on every eval, so any eval datapoint can be re-examined with the exact weights
# that produced it. READ THE DISK WARNING BELOW -- this cadence needs the pruner.
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
# Per-step rollout dump for debugging: reward/response_length/rollout_log_probs per sample,
# which is enough to recompute Cov(A,L) and Cov(log p, A) on the real training batches.
# ~150-200 MB/step, written off the critical path. See the DUMP_ROLLOUTS block in
# run_async.sh for why this is --save-debug-rollout-data and not --dump-details.
export DUMP_ROLLOUTS="${DUMP_ROLLOUTS:-1}"
export DUMP_DIR="${DUMP_DIR:-${DUMP_ROOT}/${SAVE_TAG}}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
# Log IFEval's LOOSE metric next to the strict one. eval/<name> stays STRICT (the reward),
# and eval/<name>-loose is the number comparable to published IFEval results and to the
# earlier MI300 runs -- measured 73.2% strict vs 79.7% loose on identical step-39
# responses, so reporting only strict understates the model by ~6 points.
export CUSTOM_EVAL_LOG_PATH="${CUSTOM_EVAL_LOG_PATH:-if_eval_log.log_eval_rollout_data}"

# --- objective: reference run, plus the two Olmo3 pieces that are actually supported ----
# The Olmo3 Think recipe is
# DISABLE_STD_NORM=1 + PER_TOKEN_LOSS=1 + USE_ROUTING_REPLAY=1. We take two of the three.
#
# DISABLE_STD_NORM=1 -- centre advantages without dividing by group std. Stronger reason
# here than on math: this reward is GRADED (mean 2.8 constraints/prompt, so partial credit
# lands on {0, 1/3, 2/3, 1}-style grids), and a group where all 8 samples score the same
# fraction has a tiny std. Dividing by it inflates that group's advantage arbitrarily, i.e.
# std normalisation amplifies checker noise exactly where the group is least informative.
# On math this cost was already measured as misallocation toward near-solved prompts.
#
# PER_TOKEN_LOSS stays 0 -- the one Olmo3 setting NOT to copy. The olmo3 math arm degraded
# ~2x faster than its per-sample control (AIME2024 0.756@39 -> 0.635@179, a level the
# baseline needed ~379 steps to reach), and the probed mechanism was token-level loss
# teaching the policy to answer before the cap: truncations converted into wrong finished
# answers ~one-for-one with NO gain in the correct rate. The length bias it carries is also
# actively wrong for this task -- token-level weighting is a verbosity gradient, and the
# constraint set includes both "at least N words" and "fewer than N" families, so upweighting
# long responses trades one constraint family against the other.
#
# USE_ROUTING_REPLAY=1 -- R3. Off in the reference run only because that run (20260820)
# predates the 20260824 fix that ported the two missing hooks into the Primus Megatron. It
# is worth ~4.3x on the train/inference logprob gap (0.0116 vs 0.0538 on the concurrent
# no-R3 control), which directly sharpens the --use-tis correction this recipe relies on.
export DISABLE_STD_NORM="${DISABLE_STD_NORM:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
export PER_TOKEN_LOSS="${PER_TOKEN_LOSS:-0}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-0}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
# Eval length stays at the FULL 32768. Shortening it would bias the measurement: a truncated
# response never closes </think>, so it scores 0 and reads as a compliance failure that the
# policy did not actually commit. The eval fan-out is made safe by capping concurrency
# instead (SGLANG_MAX_RUNNING_REQUESTS below), which changes throughput, not the metric.
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32768}"

# Eval is 541 IFEval + 300 IFBench = 841 prompts. At n=1 that is 841 concurrent-ish
# sequences at a 32k cap, and 960 such sequences is precisely what killed both earlier
# 3-node runs (rollout-engine HSA_STATUS_ERROR_OUT_OF_RESOURCES). Cap concurrency so peak
# memory is bounded while every response still runs to its natural length.
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"

# --- run name -----------------------------------------------------------------------
# Assembled from the values above rather than hand-written, so the name cannot disagree
# with the config it describes: change LR or turn off R3 and the name follows. Fields are
# dash-separated, values inside a field use underscores. run_async.sh appends the topology
# (-3node / -1node-smoke), so that is deliberately not included here.
#
# Example: instella16b-if32k-async_stale4-grpo-bs32x8-gbs256-lr1e-6-clip0.2_0.28-tis
#          -nostd-r3-persample-fp32head-partial-3node
_n=("instella16b" "if32k" "async_stale${MAX_WEIGHT_STALENESS}" "grpo")
_n+=("bs${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}" "gbs${GLOBAL_BATCH_SIZE}")
_n+=("lr${LR}" "clip${EPS_CLIP}_${EPS_CLIP_HIGH}" "tis")
[ "${DISABLE_STD_NORM}" = "1" ]     && _n+=("nostd")     || _n+=("std")
[ "${USE_ROUTING_REPLAY}" = "1" ]   && _n+=("r3")        || _n+=("nor3")
[ "${PER_TOKEN_LOSS}" = "1" ]       && _n+=("ptl")       || _n+=("persample")
[ "${OVERLONG_FILTER}" = "1" ]      && _n+=("overlong")
[ "${FP32_TRAINER_READOUT}" = "1" ] && _n+=("fp32head")
[ "${IF_PARTIAL_CREDIT}" = "1" ]    && _n+=("partial")   || _n+=("strict")
[ "${IF_REQUIRE_THINK}" = "1" ]     || _n+=("nothink")
[ "${EVAL_MAX_RESPONSE_LEN}" = "${ROLLOUT_MAX_RESPONSE_LEN}" ] || _n+=("evallen${EVAL_MAX_RESPONSE_LEN}")
_joined="$(IFS=-; printf '%s' "${_n[*]}")"
export WANDB_GROUP="${WANDB_GROUP:-${_joined}}"

# --- preflight: disk, because this cadence can fill a SHARED volume -----------------
# One checkpoint is ~207 GB and this miles implements NO retention (the only mention of
# save_retain_interval is `args.save_retain_interval = None`, arguments.py:2997 -- nothing
# reads it). At SAVE_INTERVAL=20 over NUM_ROLLOUT=3000 that is 150 checkpoints, ~31 TB, on
# a volume shared with the team. Rollout dumps add ~150-200 MB/step (~0.5 TB over 3000).
CKPT_GB=207
# Exported so run_async.sh writes to the same directory the pruner watches.
export SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/Instella-MoE-16B-A3B-DPO_${SAVE_TAG}}"
# Retention policy: newest few for resuming, plus the best few by eval score. Recency alone
# is wrong here -- the reference math run peaked at step ~39 and had decayed by ~179, so the
# checkpoint worth keeping is usually not among the newest.
KEEP_RECENT="${KEEP_RECENT:-2}"
KEEP_BEST="${KEEP_BEST:-3}"
PRUNE_METRIC="${PRUNE_METRIC:-eval/ifeval+eval/ifbench}"
AVAIL_GB="$(df -BG --output=avail "${CKPT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "${AVAIL_GB}" ]; then
   FITS=$(( AVAIL_GB / CKPT_GB ))
   echo "=== disk preflight ==="
   echo "  ${CKPT_ROOT} free: ${AVAIL_GB} GiB  =>  room for ~${FITS} checkpoints at ~${CKPT_GB} GiB each"
   if [ "${FITS}" -lt 8 ]; then
      echo "  REFUSING TO LAUNCH: fewer than 8 checkpoints fit. Free space or lower SAVE_INTERVAL." >&2
      echo "  Biggest consumers:" >&2
      du -sh "${CKPT_ROOT}"/* 2>/dev/null | sort -rh | head -5 | sed 's/^/    /' >&2
      exit 1
   fi
   if [ "${PRUNE_CHECKPOINTS:-1}" = "1" ]; then
      echo "  pruner keeps newest ${KEEP_RECENT} + best ${KEEP_BEST} by ${PRUNE_METRIC} in ${SAVE_DIR}"
      echo "  (PRUNE_CHECKPOINTS=0 disables; it never deletes the newest or the one in"
      echo "   latest_checkpointed_iteration.txt, and never prunes on best-score when no"
      echo "   eval scores are available)"
   else
      echo "  WARNING: pruner disabled; ${FITS} checkpoints = $(( FITS * SAVE_INTERVAL )) steps before the volume is full" >&2
   fi
fi

# --- preflight: the reward must import and score before we burn an allocation -------
if [ "${SKIP_REWARD_PREFLIGHT:-0}" != "1" ]; then
   echo "=== reward preflight ==="
   if ! NLTK_DATA="${NLTK_DATA}" python3 "${SCRIPT_DIR}/if_reward.py"; then
      cat >&2 <<EOF

FATAL: if_reward.py self-test failed. Fix this BEFORE submitting -- the reward runs on the
rollout workers, where a bad import degrades to reward 0.0 on every sample rather than
crashing, and that looks exactly like a model that cannot follow instructions.
   bash ${SCRIPT_DIR}/setup_if_deps.sh            # install on THIS node
   bash ${SCRIPT_DIR}/setup_if_deps.sh --verify   # check
Set SKIP_REWARD_PREFLIGHT=1 only if you know why it fails.
EOF
      exit 1
   fi
fi

# --- freeze and launch ------------------------------------------------------------
if [ "${CHECK_ONLY:-0}" = "1" ]; then
   echo
   echo "=== CHECK_ONLY=1: all preflights passed, NOT submitting ==="
   echo "  project    : ${WANDB_PROJECT}"
   echo "  run name   : ${WANDB_GROUP}-$([ "${SINGLE_NODE:-0}" = "1" ] && echo 1node-smoke || echo 3node)"
   echo "  train data : ${PROMPT_DATA} ($(wc -l < "${PROMPT_DATA}") prompts)"
   echo "  eval data  : ${EVAL_PROMPT_DATA}"
   echo "  save dir   : ${SAVE_DIR} (every ${SAVE_INTERVAL} steps)"
   echo "  retention  : newest ${KEEP_RECENT} + best ${KEEP_BEST} by ${PRUNE_METRIC}"
   echo "  rollout dmp: ${DUMP_DIR}/rollout_data/"
   echo "  drop CHECK_ONLY to launch."
   exit 0
fi

cd "${REPO_DIR}" || exit 1
mkdir -p "${LOG_DIR}" "${WANDB_LOCAL_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${LOG_DIR}/launch-if32k-$(hostname -s)-${TS}.out"

# Freeze run_async.sh, not this wrapper: bash reads a script incrementally, so editing the
# live file mid-run makes the open shell resume at a stale byte offset and execute a garbled
# fragment (observed 20260820). The snapshot must stay in this directory because these
# scripts derive SCRIPT_DIR from their own location and source model_args.sh from there.
FROZEN="${SCRIPT_DIR}/.run-frozen-${TS}.sh"
cp "${TARGET}" "${FROZEN}" || exit 1

echo "=== instruction-following 32k arm ==="
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_GROUP}-$([ "${SINGLE_NODE:-0}" = "1" ] && echo 1node-smoke || echo 3node)"
for V in SAVE_TAG WANDB_PROJECT WANDB_GROUP WANDB_RANDOM_SUFFIX CUSTOM_RM_PATH IF_REWARD_SCALE IF_PARTIAL_CREDIT \
         IF_REQUIRE_THINK SAVE_INTERVAL EVAL_INTERVAL N_SAMPLES_PER_EVAL_PROMPT \
         EVAL_TEMPERATURE ROLLOUT_MAX_RESPONSE_LEN EVAL_MAX_RESPONSE_LEN \
         SGLANG_MAX_RUNNING_REQUESTS PER_TOKEN_LOSS OVERLONG_FILTER \
         DISABLE_STD_NORM USE_ROUTING_REPLAY; do
   echo "  ${V}=${!V}"
done
echo "  PROMPT_DATA=${PROMPT_DATA} ($(wc -l < "${PROMPT_DATA}") prompts)"
echo "  EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA}"
echo "  SINGLE_NODE=${SINGLE_NODE:-0}  NUM_ROLLOUT=${NUM_ROLLOUT:-3000}"
echo

nohup bash "${FROZEN}" > "${OUT}" 2>&1 &
TRAIN_PID=$!
echo "LAUNCHED pid=${TRAIN_PID} frozen=${FROZEN} log=${OUT}"

# Pruner alongside the run, since miles will not do it (see prune_checkpoints.sh).
if [ "${PRUNE_CHECKPOINTS:-1}" = "1" ]; then
   PRUNE_LOG="${LOG_DIR}/prune-${SAVE_TAG}-${TS}.log"
   # --tag SAVE_TAG so it reads THIS run's logs; the metric keys are eval/<dataset name>,
   # i.e. the names given to --eval-prompt-data (miles/ray/rollout/metrics.py:38).
   PRUNE_METRIC_ARGS=()
   IFS='+' read -r -a _metrics <<< "${PRUNE_METRIC}"
   for m in "${_metrics[@]}"; do PRUNE_METRIC_ARGS+=(--metric "${m}"); done
   setsid nohup python3 -u "${SCRIPT_DIR}/prune_checkpoints.py" "${SAVE_DIR}" \
      --keep-recent "${KEEP_RECENT}" --keep-best "${KEEP_BEST}" \
      "${PRUNE_METRIC_ARGS[@]}" --tag "${SAVE_TAG}" --log "${OUT}" \
      > "${PRUNE_LOG}" 2>&1 &
   echo "PRUNER  pid=$! keep_recent=${KEEP_RECENT} keep_best=${KEEP_BEST} log=${PRUNE_LOG}"
fi

echo
echo "Watch:       tail -f ${OUT}"
echo "Reward:      grep IF_REWARD ${OUT}"
echo "Checkpoints: ls ${SAVE_DIR}"
echo "Rollouts:    ls ${DUMP_DIR}/rollout_data/"