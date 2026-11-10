#!/bin/bash
# Inside the container of one GPU node of an eval-only run.sbatch job (EVAL_ONLY=1; see
# eval_gpu_node.sh). Evaluates the training checkpoint EVAL_SOURCE_DIR/iter_<EVAL_STEP> (or the HF
# base model for EVAL_STEP=release) with every GPU node serving inference, then records the result.
#
#   rank 0     : converts the checkpoint to HF format once (cached in EVAL_SOURCE_DIR/hf/), starts
#                the Ray head, runs the spec with --debug-rollout-only so miles starts only sglang
#                engines (loading those HF weights) and evaluates them at step 0, then appends the
#                scores to EVAL_SOURCE_DIR/.evals.txt and does the W&B bookkeeping (eval_series.py)
#   ranks 1..N : join as Ray workers until the head stops
#
# The conversion is tools/convert_torch_dist_to_hf.py: it reads the checkpoint on CPU (skipping
# optimizer state) and maps names with megatron_to_hf.convert_to_hf, the converter training uses
# for its weight updates to sglang (--megatron-to-hf-mode raw, the default).
set -uo pipefail
source /opt/venv/bin/activate
cd "${USER_HOME}/miles" || exit 1
I=examples/instella_think
source "${I}/paths.sh"
HOST="$(hostname -s)"
say() { echo "[$(date '+%F %T')] [${HOST}] $*"; }
fail() { say "FATAL: $*"; echo "failed on ${HOST}: $*" > "${JOB_DIR}/failed"; exit 1; }
await() {  # <file> <timeout_s> <what>
   local waited=0
   until [ -e "$1" ]; do
      [ -e "${JOB_DIR}/failed" ] && fail "gave up waiting for $3: $(cat "${JOB_DIR}/failed")"
      [ "${waited}" -ge "$2" ] && fail "timed out after $2 s waiting for $3"
      sleep 5; waited=$((waited + 5))
   done
}

# --- 1. per-container setup, as run_in_container.sh ----------------------------------------
sleep $(( NODE_RANK * 20 ))
pip install -q -e "${MEGATRON_PATH}" --no-deps --no-build-isolation 2>&1 | grep -v "notice" || true
bash "${I}/cluster/env_setup.sh" || fail "env_setup.sh failed"
touch "${JOB_DIR}/setup.${HOST}"

# --- 2a. Ray workers -----------------------------------------------------------------------
if [ "${NODE_RANK}" != 0 ]; then
   await "${JOB_DIR}/head_ip" 3600 "the Ray head"
   say "joining Ray head $(cat "${JOB_DIR}/head_ip")"
   ray start --address="$(cat "${JOB_DIR}/head_ip"):6379" --num-gpus 8 --disable-usage-stats --block
   say "Ray worker exited"
   exit 0
fi

# --- 2b. head ------------------------------------------------------------------------------
source "${JOB_DIR}/spec.env"
: "${EVAL_STEP:?EVAL_ONLY needs EVAL_STEP}" "${EVAL_SOURCE_DIR:?EVAL_ONLY needs EVAL_SOURCE_DIR}"
BASE_HF="${HF_CKPT:-${CKPT_ROOT}/Instella-MoE-16B-A3B-DPO}"
if [ "${EVAL_STEP}" = release ]; then
   # The base model: its HF checkpoint is the weights.
   EVAL_HF="${BASE_HF}"
else
   SRC="${EVAL_SOURCE_DIR}/iter_$(printf %07d "${EVAL_STEP}")"
   [ -f "${SRC}/common.pt" ] || fail "no checkpoint iter ${EVAL_STEP} in ${EVAL_SOURCE_DIR}"
   EVAL_HF="${EVAL_SOURCE_DIR}/hf/iter_$(printf %07d "${EVAL_STEP}")"
   if [ -f "${EVAL_HF}/model.safetensors.index.json" ]; then
      say "using converted checkpoint ${EVAL_HF}"
   else
      say "converting ${SRC} to HF format in ${EVAL_HF}"
      TMP="${EVAL_HF}.tmp.${SLURM_JOB_ID}"
      rm -rf "${TMP}"; mkdir -p "$(dirname "${EVAL_HF}")"
      PYTHONPATH="${PWD}:${MEGATRON_PATH}" python3 tools/convert_torch_dist_to_hf.py \
         --input-dir "${SRC}" --output-dir "${TMP}" --origin-hf-dir "${BASE_HF}" \
         > "${JOB_DIR}/convert.log" 2>&1 || fail "checkpoint conversion failed (see ${JOB_DIR}/convert.log)"
      mv "${TMP}" "${EVAL_HF}" 2>/dev/null || { rm -rf "${TMP}"; say "another job converted it first"; }
      say "converted ($(grep -c . "${EVAL_HF}/model.safetensors.index.json") index lines)"
   fi
fi

await "${JOB_DIR}/sandbox.ready" 900 "the reward sandbox (see ${JOB_DIR}/sandbox.log)"
await "${JOB_DIR}/sandbox-bench.ready" 900 "the benchmark sandbox (see ${JOB_DIR}/sandbox*.log)"
TOK="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}")"
export CODE_EXEC_API_URL="http://$(cat "${JOB_DIR}/sandbox.ready"):${CODE_EXEC_PORT}/${TOK}"
export CODE_BENCH_API_URL="http://$(cat "${JOB_DIR}/sandbox-bench.ready"):${CODE_BENCH_PORT}/${TOK}"

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address="${NODE_IP}" --port=6379 --num-gpus 8 \
   --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats || fail "ray start --head failed"
echo "${NODE_IP}" > "${JOB_DIR}/head_ip"
waited=0
until [ "$(python3 -c 'import ray; ray.init(address="auto", logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0)))' 2>/dev/null)" = "$(( GPU_NODE_COUNT * 8 ))" ]; do
   [ -e "${JOB_DIR}/failed" ] && fail "a worker failed: $(cat "${JOB_DIR}/failed")"
   [ "${waited}" -ge 1800 ] && fail "Ray cluster never reached $(( GPU_NODE_COUNT * 8 )) GPUs"
   sleep 10; waited=$((waited + 10))
done
say "Ray cluster up: $(( GPU_NODE_COUNT * 8 )) GPUs"
export RAY_ADDRESS="http://${NODE_IP}:8265" MASTER_ADDR="${NODE_IP}"

# Every GPU node is an inference engine; --debug-rollout-only leaves miles without a trainer, so
# the engines keep the HF weights they load and miles evaluates them at step 0.
TRAIN_GROUP="${WANDB_GROUP:-eval}"
export HF_CKPT="${EVAL_HF}" ROLLOUT_NODES="${GPU_NODE_COUNT}" EXTRA_ARGS="${EXTRA_ARGS:-} --debug-rollout-only" \
   SAVE_TAG="${SAVE_TAG}_eval${EVAL_STEP}" SKIP_EVAL_BEFORE_TRAIN=0 EVAL_INTERVAL=100000 PRUNE_CHECKPOINTS=0 \
   WANDB_GROUP="${TRAIN_GROUP}-eval-iter${EVAL_STEP}"
say "launching ${RUN_SCRIPT:-run_code.sh} (eval of ${EVAL_HF})"
bash "${I}/${RUN_SCRIPT:-run_code.sh}" 2>&1 | tee "${JOB_DIR}/launch.txt"
[ "${PIPESTATUS[0]}" = 0 ] || fail "${RUN_SCRIPT:-run_code.sh} failed before launching (preflight?)"
pid="$(grep -o 'LAUNCHED pid=[0-9]*' "${JOB_DIR}/launch.txt" | cut -d= -f2)"
log="$(grep -o 'log=[^ ]*' "${JOB_DIR}/launch.txt" | head -1 | cut -d= -f2)"
[ -n "${pid}" ] || fail "no LAUNCHED pid in launch output"
ln -sf "${log}" "${JOB_DIR}/run.log"
say "run pid ${pid}, log ${log}"
until grep -q -E "^.*eval 0: \{" "${log}" 2>/dev/null; do
   kill -0 "${pid}" 2>/dev/null || fail "run ended before logging the eval (see ${log})"
   sleep 30
done
grep -o -E "eval 0: \{.*" "${log}" | head -1 > "${JOB_DIR}/eval_result.txt"
echo "iter ${EVAL_STEP} of ${EVAL_SOURCE_DIR}: $(cat "${JOB_DIR}/eval_result.txt")" >> "${EVAL_SOURCE_DIR}/.evals.txt"
echo "eval done" > "${JOB_DIR}/run.status"
say "eval of iter ${EVAL_STEP} logged; stopping"
# Give W&B a moment to send the scores before the run is torn down.
sleep 60
for id in $(ray job list 2>/dev/null | grep -o "raysubmit_[A-Za-z0-9]*" | sort -u); do ray job stop "${id}" >/dev/null 2>&1; done
ray stop --force >/dev/null 2>&1 || true

# W&B: re-upload the scores if the shutdown lost them, and add the point to the training run's
# eval series. Best effort: the result is already in .evals.txt.
if [ "${WANDB_MODE:-online}" != offline ]; then
   [ -n "${EVAL_SERIES_ID:-}" ] || EVAL_SERIES_ID="evals-$(basename "${EVAL_SOURCE_DIR}")"
   NETRC="${NETRC_PATH}" HTTPS_PROXY="${LUX_PROXY}" HTTP_PROXY="${LUX_PROXY}" WANDB_LOCAL_DIR="${WANDB_LOCAL_DIR}" \
      WANDB_PROJECT="${WANDB_PROJECT:-miles-MI355-code}" REF_LOAD="${REF_LOAD:-${CKPT_ROOT}/Instella-MoE-16B-A3B-DPO_torch_dist}" \
      timeout 900 python3 "${I}/cluster/lux/eval_series.py" "${EVAL_SOURCE_DIR}" "${TRAIN_GROUP}" \
      "${TRAIN_GROUP}-eval-iter${EVAL_STEP}" "${EVAL_SERIES_ID}" > "${JOB_DIR}/wandb.log" 2>&1 \
      && say "W&B: $(grep -E '^[a-z0-9-]+: ' "${JOB_DIR}/wandb.log" | tr '\n' ' ')" \
      || say "W&B bookkeeping failed (see ${JOB_DIR}/wandb.log; scores are in ${EVAL_SOURCE_DIR}/.evals.txt)"
fi
exit 0
