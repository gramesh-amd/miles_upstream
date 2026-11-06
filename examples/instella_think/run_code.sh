#!/bin/bash
# Code RL arm of the Instella-MoE-16B-A3B 32k RL runs.
#
# Same recipe as run_if.sh (itself the rytwnj5z math recipe, see that file's header),
# with the task swapped to unit-test-verified code generation:
#   data   : Dolci-Think-RL code_rlvr (21k prompts, stdin/stdout + assert tests)
#   reward : code_reward.batched_reward -- pass rate while training, all-or-nothing at eval
#   eval   : HumanEval+, MBPP+ and LiveCodeBench v3 under OLMo 3's protocol (README)
# Objective defaults (nostd + R3, per-sample loss, fp32 head) are run_if.sh's, for the
# same reasons given there; this file only sets what differs.
#
# Programs never run on the GPU nodes. They go to the code-exec service
# (code_exec_server.py) on a CPU host, started with cluster/code_exec.sh.
#
# WHAT TO CHECK IN THE FIRST 20 MINUTES
#   a. `grep "CODE_REWARD scored" <log>`: api_error must be 0. Anything else means the
#      sandbox is unreachable or overloaded and rewards are measuring the network.
#      think_gate_fail counts responses without a closed <think>...</think>; nearly all of
#      them are responses truncated mid-thought, so it tracks rollout/truncated_ratio.
#   b. `cluster/code_exec.sh status`: timeouts climbing faster than tests means
#      CODE_EXEC_TIMEOUT is too tight or the host is saturated.
#   c. rollout/truncated_ratio: code answers with reasoning are long. At 16k about a third
#      of responses truncate (and score 0); 32k is the default for that reason.
#
# USAGE -- from inside the head node's container (see README "Code RL"):
#   One-time, any node:  python3 examples/instella_think/prepare_code_data.py build
#                        python3 examples/instella_think/prepare_code_benchmarks.py --out-dir "$CODE_BENCH_DIR"
#   Sandboxes:           examples/instella_think/cluster/code_exec.sh start         (login node)
#                        examples/instella_think/cluster/code_exec.sh start bench
#   Smoke on 1 node:     SINGLE_NODE=1 NUM_ROLLOUT=8 bash examples/instella_think/run_code.sh
#   Preflight only:      CHECK_ONLY=1 bash examples/instella_think/run_code.sh
#   3-node train:        bash examples/instella_think/run_code.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${SCRIPT_DIR}/run_async.sh"
[ -f "${TARGET}" ] || { echo "FATAL: ${TARGET} not found" >&2; exit 1; }
source "${SCRIPT_DIR}/paths.sh"

export RAY_ADDRESS="${RAY_ADDRESS:-}"
export MASTER_ADDR="${MASTER_ADDR:-}"

# wandb: same netrc fallback as run_if.sh (HOME is /root inside the container).
export WANDB_KEY="${WANDB_KEY:-}"
if [ -z "${WANDB_KEY}" ]; then
   if [ -f "${NETRC_PATH}" ]; then
      export NETRC="${NETRC_PATH}"
   else
      echo "WARNING: no WANDB_KEY and no netrc at ${NETRC_PATH} -- metrics may not be logged." >&2
   fi
fi

# --- data -------------------------------------------------------------------------
CODE_TRAIN="${CODE_TRAIN:-${DATA_ROOT}/code-train/dolci-code-train.jsonl}"
[ -s "${CODE_TRAIN}" ] || { echo "FATAL: missing ${CODE_TRAIN}; run: python3 ${SCRIPT_DIR}/prepare_code_data.py build" >&2; exit 1; }
export PROMPT_DATA="${PROMPT_DATA:-${CODE_TRAIN}}"

# Eval: OLMo 3's code benchmarks (prepare_code_benchmarks.py builds them; `cluster/code_exec.sh
# start bench` serves their tests). Sampling is OLMo 3 Think's (arXiv 2512.13961 Table 16):
# 10 samples, temperature 0.6, top_p 0.95, 32768 tokens, so eval/<set> is their pass@1.
CODE_BENCH_SETS="${CODE_BENCH_SETS:-humanevalplus mbppplus lcb_v3}"
CODE_BENCH_N="${CODE_BENCH_N:-10}"
CODE_BENCH_MAX_LEN="${CODE_BENCH_MAX_LEN:-32768}"
for s in ${CODE_BENCH_SETS}; do
   [ -s "${CODE_BENCH_DIR}/${s}.jsonl" ] || {
      echo "FATAL: missing ${CODE_BENCH_DIR}/${s}.jsonl; run: python3 ${SCRIPT_DIR}/prepare_code_benchmarks.py --out-dir ${CODE_BENCH_DIR}" >&2; exit 1; }
done
if [ -z "${CODE_BENCH_API_URL:-}" ]; then
   [ -n "${CODE_EXEC_HOST:-}" ] || { echo "FATAL: set CODE_EXEC_HOST or CODE_BENCH_API_URL" >&2; exit 1; }
   export CODE_BENCH_API_URL="http://${CODE_EXEC_HOST}:${CODE_BENCH_PORT}/$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}" 2>/dev/null)"
fi
# miles reads per-set sampling from an eval YAML; dry runs write theirs to /tmp.
if [ -z "${EVAL_CONFIG:-}" ]; then
   [ "${DRY_RUN:-0}" = "1" ] && _cfg_dir=/tmp || _cfg_dir="${LOG_DIR}/eval-configs"
   mkdir -p "${_cfg_dir}"
   # Hostname too: runs launched in the same second on several nodes share this directory.
   export EVAL_CONFIG="${_cfg_dir}/code-bench-$(hostname -s)-$(date +%Y%m%d_%H%M%S).yaml"
   {
      echo "eval:"
      echo "  datasets:"
      for s in ${CODE_BENCH_SETS}; do
         echo "    - {name: ${s}, path: ${CODE_BENCH_DIR}/${s}.jsonl, n_samples_per_eval_prompt: ${CODE_BENCH_N},"
         echo "       temperature: 0.6, top_p: 0.95, top_k: -1, max_response_len: ${CODE_BENCH_MAX_LEN}}"
      done
   } > "${EVAL_CONFIG}"
fi

# --- reward -----------------------------------------------------------------------
export CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-code_reward.batched_reward}"
if [ -z "${CODE_EXEC_API_URL:-}" ]; then
   _tok="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}" 2>/dev/null)"
   [ -n "${_tok}" ] || { echo "FATAL: no CODE_EXEC_API_URL and no ${CODE_EXEC_TOKEN_FILE}; run cluster/code_exec.sh start" >&2; exit 1; }
   [ -n "${CODE_EXEC_HOST:-}" ] || { echo "FATAL: set CODE_EXEC_HOST (the sandbox host) or CODE_EXEC_API_URL, e.g. in ${INSTELLA_ENV}" >&2; exit 1; }
   export CODE_EXEC_API_URL="http://${CODE_EXEC_HOST}:${CODE_EXEC_PORT}/${_tok}"
fi
export CODE_EXEC_TIMEOUT="${CODE_EXEC_TIMEOUT:-5}"
export CODE_MAX_TESTS="${CODE_MAX_TESTS:-32}"
export CODE_PARTIAL_CREDIT="${CODE_PARTIAL_CREDIT:-1}"
export CODE_REQUIRE_THINK="${CODE_REQUIRE_THINK:-1}"
export CODE_REWARD_SCALE="${CODE_REWARD_SCALE:-1.0}"

# --- run identity -----------------------------------------------------------------
# Fresh tag: run_async.sh points --load at SAVE_DIR, so reusing a tag RESUMES that run.
export SAVE_TAG="${SAVE_TAG:-miles_code32k_nostd_r3}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles-MI355-code}"
export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export MAX_WEIGHT_STALENESS="${MAX_WEIGHT_STALENESS:-4}"
export LR="${LR:-1e-6}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
export FP32_TRAINER_READOUT="${FP32_TRAINER_READOUT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
export DUMP_ROLLOUTS="${DUMP_ROLLOUTS:-1}"
export DUMP_DIR="${DUMP_DIR:-${DUMP_ROOT}/${SAVE_TAG}}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
export DISABLE_STD_NORM="${DISABLE_STD_NORM:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
export PER_TOKEN_LOSS="${PER_TOKEN_LOSS:-0}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-0}"
# DRGRPO=1: constant loss denominator (drgrpo_reducer.py). Needs DRGRPO_DIVISOR = the step-0 mean
# response length for this cap and checkpoint (10686 at a 16384 cap from the DPO base); use it
# with OVERLONG_FILTER=1, or truncated samples carry ~41% of the gradient.
export DRGRPO="${DRGRPO:-0}"
export DRGRPO_DIVISOR="${DRGRPO_DIVISOR:-}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32768}"
# Prompt+response budget; raise it above 32768 for longer caps (e.g. CONTEXT_LEN=63488 with
# ROLLOUT_MAX_RESPONSE_LEN=61440). sglang's auto-truncate gives each request context - prompt.
export CONTEXT_LEN="${CONTEXT_LEN:-32768}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"

_n=("instella16b" "code$(( ROLLOUT_MAX_RESPONSE_LEN / 1024 ))k" "async_stale${MAX_WEIGHT_STALENESS}" "grpo")
_n+=("bs${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}" "gbs${GLOBAL_BATCH_SIZE}")
_n+=("lr${LR}" "clip${EPS_CLIP}_${EPS_CLIP_HIGH}" "tis")
[ "${DISABLE_STD_NORM}" = "1" ]     && _n+=("nostd")     || _n+=("std")
[ "${USE_ROUTING_REPLAY}" = "1" ]   && _n+=("r3")        || _n+=("nor3")
if [ "${PER_TOKEN_LOSS}" = "1" ]; then _n+=("ptl")
elif [ "${DRGRPO}" = "1" ]; then _n+=("drgrpo${DRGRPO_DIVISOR%%.*}")
else _n+=("persample"); fi
[ "${OVERLONG_FILTER}" = "1" ]      && _n+=("overlong")
[ "${FP32_TRAINER_READOUT}" = "1" ] && _n+=("fp32head")
[ "${CODE_PARTIAL_CREDIT}" = "1" ]  && _n+=("passrate")  || _n+=("allpass")
[ "${CODE_REQUIRE_THINK}" = "1" ]   || _n+=("nothink")
[ "${EVAL_MAX_RESPONSE_LEN}" = "${ROLLOUT_MAX_RESPONSE_LEN}" ] || _n+=("evallen${EVAL_MAX_RESPONSE_LEN}")
[ "${CONTEXT_LEN}" = "32768" ] || _n+=("ctx${CONTEXT_LEN}")
_g="$(IFS=-; printf '%s' "${_n[*]}")"
# W&B rejects groups over 128 characters, and run_async.sh appends "-1node-smoke"/"-3node". A
# DRGRPO or OVERLONG_FILTER name gets past that, so keep the readable start and end with a hash.
[ "${#_g}" -le 116 ] || _g="${_g:0:107}-$(printf '%s' "${_g}" | md5sum | cut -c1-8)"
export WANDB_GROUP="${WANDB_GROUP:-${_g}}"

# --- preflight: loss config ---------------------------------------------------------
if [ "${DRGRPO}" = "1" ]; then
   [ "${PER_TOKEN_LOSS}" = "0" ] || { echo "FATAL: DRGRPO=1 requires PER_TOKEN_LOSS=0" >&2; exit 1; }
   [[ "${DRGRPO_DIVISOR}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
      echo "FATAL: DRGRPO=1 needs DRGRPO_DIVISOR (step-0 mean response length, e.g. 10686 at a 16k cap)" >&2; exit 1; }
   [ "${OVERLONG_FILTER}" = "1" ] || echo "WARNING: DRGRPO=1 without OVERLONG_FILTER=1: truncated samples get ~41% of the gradient" >&2
fi

# --- preflight: disk ----------------------------------------------------------------
CKPT_GB=207
export SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/Instella-MoE-16B-A3B-DPO_${SAVE_TAG}}"
# The volume is shared, so keep the footprint small: newest (for resume) + best by eval.
KEEP_RECENT="${KEEP_RECENT:-1}"
KEEP_BEST="${KEEP_BEST:-2}"
# Space-separated; the pruner ranks checkpoints by the mean of these.
PRUNE_METRIC="${PRUNE_METRIC:-$(for s in ${CODE_BENCH_SETS}; do printf 'eval/%s ' "$s"; done)}"
AVAIL_GB="$(df -BG --output=avail "${CKPT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')"
# Peak footprint is the retained set plus the one being written before the pruner runs.
NEED_CKPTS=$(( KEEP_RECENT + KEEP_BEST + 1 ))
if [ -n "${AVAIL_GB}" ]; then
   FITS=$(( AVAIL_GB / CKPT_GB ))
   echo "=== disk preflight: ${CKPT_ROOT} ${AVAIL_GB} GiB free, ~${FITS} checkpoints, need ${NEED_CKPTS} ==="
   if [ "${FITS}" -lt "${NEED_CKPTS}" ]; then
      echo "  REFUSING TO LAUNCH: fewer than ${NEED_CKPTS} checkpoints fit. Free space or lower KEEP_BEST." >&2
      [ "${DRY_RUN:-0}" = "1" ] || exit 1
   fi
fi

# --- preflight: sandbox + reward ----------------------------------------------------
# A dead sandbox does not crash anything: every sample silently scores 0, which reads as
# a model that cannot code. So it must answer, and score known cases, before launch.
if [ "${SKIP_REWARD_PREFLIGHT:-0}" != "1" ]; then
   echo "=== reward preflight (sandbox at ${CODE_EXEC_API_URL%/*}/<token>) ==="
   if ! python3 "${SCRIPT_DIR}/code_reward.py"; then
      echo "FATAL: code_reward.py self-test failed. Is the sandbox up? cluster/code_exec.sh status" >&2
      exit 1
   fi
   echo "=== benchmark preflight (sandbox at ${CODE_BENCH_API_URL%/*}/<token>) ==="
   # One canonical HumanEval+ solution must pass: proves numpy, the 20 s cap and the route.
   if ! python3 - "${SCRIPT_DIR}" "${CODE_BENCH_DIR}" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import code_reward as cr
label = json.loads(json.loads(open(f"{sys.argv[2]}/humanevalplus.jsonl").readline())["label"])
body = "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n"
ok = cr.bench_pass(label["prefix"] + body, label)
print(f"  HumanEval/0 reference solution: {ok}")
sys.exit(0 if ok else 1)
EOF
   then
      echo "FATAL: benchmark sandbox check failed. Is it up? cluster/code_exec.sh status bench" >&2
      exit 1
   fi
fi
if [ "${CODE_BENCH_MAX_LEN}" -gt "${CONTEXT_LEN}" ]; then
   echo "FATAL: CODE_BENCH_MAX_LEN=${CODE_BENCH_MAX_LEN} exceeds CONTEXT_LEN=${CONTEXT_LEN}" >&2; exit 1
fi

if [ "${CHECK_ONLY:-0}" = "1" ]; then
   echo
   echo "=== CHECK_ONLY=1: all preflights passed, NOT submitting ==="
   echo "  project    : ${WANDB_PROJECT}"
   echo "  run name   : ${WANDB_GROUP}-$([ "${SINGLE_NODE:-0}" = "1" ] && echo 1node-smoke || echo 3node)"
   echo "  train data : ${PROMPT_DATA} ($(wc -l < "${PROMPT_DATA}") prompts)"
   echo "  eval       : ${CODE_BENCH_SETS} (n=${CODE_BENCH_N}, ${CODE_BENCH_MAX_LEN} tokens) via ${EVAL_CONFIG}"
   echo "  save dir   : ${SAVE_DIR} (every ${SAVE_INTERVAL} steps)"
   exit 0
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
   cd "${REPO_DIR}" || exit 1
   exec bash "${TARGET}"
fi

# --- freeze and launch (same mechanics as run_if.sh) --------------------------------
cd "${REPO_DIR}" || exit 1
mkdir -p "${LOG_DIR}" "${WANDB_LOCAL_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${LOG_DIR}/launch-code$(( ROLLOUT_MAX_RESPONSE_LEN / 1024 ))k-$(hostname -s)-${TS}.out"
FROZEN="${SCRIPT_DIR}/.run-frozen-${TS}.sh"
cp "${TARGET}" "${FROZEN}" || exit 1

echo "=== code RL $(( ROLLOUT_MAX_RESPONSE_LEN / 1024 ))k arm ==="
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_GROUP}-$([ "${SINGLE_NODE:-0}" = "1" ] && echo 1node-smoke || echo 3node)"
for V in SAVE_TAG CUSTOM_RM_PATH CODE_EXEC_TIMEOUT CODE_MAX_TESTS CODE_PARTIAL_CREDIT CODE_REQUIRE_THINK \
         SAVE_INTERVAL EVAL_INTERVAL ROLLOUT_MAX_RESPONSE_LEN DISABLE_STD_NORM USE_ROUTING_REPLAY \
         CONTEXT_LEN PER_TOKEN_LOSS DRGRPO DRGRPO_DIVISOR OVERLONG_FILTER; do
   echo "  ${V}=${!V}"
done
echo "  eval: ${CODE_BENCH_SETS} (n=${CODE_BENCH_N}) config ${EVAL_CONFIG}"
echo "  PROMPT_DATA=${PROMPT_DATA} ($(wc -l < "${PROMPT_DATA}") prompts)"
echo "  SINGLE_NODE=${SINGLE_NODE:-0}  NUM_ROLLOUT=${NUM_ROLLOUT:-3000}"

nohup bash "${FROZEN}" > "${OUT}" 2>&1 &
echo "LAUNCHED pid=$! frozen=${FROZEN} log=${OUT}"

if [ "${PRUNE_CHECKPOINTS:-1}" = "1" ]; then
   PRUNE_LOG="${LOG_DIR}/prune-${SAVE_TAG}-${TS}.log"
   setsid nohup python3 -u "${SCRIPT_DIR}/prune_checkpoints.py" "${SAVE_DIR}" \
      --keep-recent "${KEEP_RECENT}" --keep-best "${KEEP_BEST}" \
      $(for m in ${PRUNE_METRIC}; do printf -- '--metric %s ' "$m"; done) --tag "${SAVE_TAG}" --log "${OUT}" \
      > "${PRUNE_LOG}" 2>&1 &
   echo "PRUNER  pid=$! log=${PRUNE_LOG}"
fi

echo
echo "Watch:   tail -f ${OUT}"
echo "Reward:  grep 'CODE_REWARD scored' ${OUT}"
echo "Sandbox: ${SCRIPT_DIR}/cluster/code_exec.sh status"
