#!/bin/bash
# Inside the container of one GPU node of a run.sbatch job (see run_gpu_node.sh): set the
# container up, then take this node's role.
#   1 GPU node : run the spec with SINGLE_NODE=1 (run_async.sh starts its own Ray head)
#   3 GPU nodes: rank 0 starts the Ray head, waits for the workers and runs the spec;
#                ranks 1-2 join as Ray workers until the head stops
# Coordination is through files in $JOB_DIR: setup.<host>, head_ip, sandbox.ready, failed, run.status.
set -uo pipefail
source /opt/venv/bin/activate
cd "${USER_HOME}/miles" || exit 1
I=examples/instella_think
source "${I}/paths.sh"
HOST="$(hostname -s)"
say() { echo "[$(date '+%F %T')] [${HOST}] $*"; }
fail() { say "FATAL: $*"; echo "failed on ${HOST}: $*" > "${JOB_DIR}/failed"; exit 1; }
# Waits until a file exists, or fails on timeout or once another node has failed.
await() {  # <file> <timeout_s> <what>
   local waited=0
   until [ -e "$1" ]; do
      [ -e "${JOB_DIR}/failed" ] && fail "gave up waiting for $3: $(cat "${JOB_DIR}/failed")"
      [ "${waited}" -ge "$2" ] && fail "timed out after $2 s waiting for $3"
      sleep 5; waited=$((waited + 5))
   done
}

# --- 1. per-container setup (lost with the container, so every job) ------------------------
# The editable install writes into the shared Megatron checkout on NFS; stagger the nodes.
sleep $(( NODE_RANK * 20 ))
pip install -q -e "${MEGATRON_PATH}" --no-deps --no-build-isolation 2>&1 | grep -v "notice" || true
python3 -c "import megatron.core, os, sys; sys.exit(0 if os.path.realpath(megatron.core.__file__).startswith(os.path.realpath('${MEGATRON_PATH}')) else 1)" \
   || fail "megatron.core does not resolve to ${MEGATRON_PATH}"
bash "${I}/cluster/env_setup.sh" || fail "env_setup.sh failed"
touch "${JOB_DIR}/setup.${HOST}"

# --- 2a. Ray workers -----------------------------------------------------------------------
if [ "${NODE_RANK}" != 0 ]; then
   await "${JOB_DIR}/head_ip" 1800 "the Ray head"
   HEAD_IP="$(cat "${JOB_DIR}/head_ip")"
   say "joining Ray head ${HEAD_IP}"
   # --block keeps Ray inside this session; it returns when the head goes away.
   ray start --address="${HEAD_IP}:6379" --num-gpus 8 --disable-usage-stats --block
   say "Ray worker exited"
   exit 0
fi

# --- 2b. head: the sandbox, the Ray cluster, then the run -----------------------------------
await "${JOB_DIR}/sandbox.ready" 900 "the sandbox (see ${JOB_DIR}/sandbox.log)"
TOK="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}")"
export CODE_EXEC_API_URL="http://${SANDBOX_HOST}:${CODE_EXEC_PORT}/${TOK}"
export CODE_BENCH_API_URL="http://${SANDBOX_HOST}:${CODE_BENCH_PORT}/${TOK}"

if [ "${GPU_NODE_COUNT}" = 1 ]; then
   export SINGLE_NODE=1
else
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
fi

# The spec last, so it can override anything above.
source "${JOB_DIR}/spec.env"
say "launching ${RUN_SCRIPT:-run_code.sh} with: $(sed 's/^export //' "${JOB_DIR}/spec.env" | tr '\n' ' ')"
bash "${I}/${RUN_SCRIPT:-run_code.sh}" 2>&1 | tee "${JOB_DIR}/launch.txt"
[ "${PIPESTATUS[0]}" = 0 ] || fail "${RUN_SCRIPT:-run_code.sh} failed before launching (preflight?)"
# run_code.sh backgrounds the run and prints its pid; it has to finish inside this session.
pid="$(grep -o 'LAUNCHED pid=[0-9]*' "${JOB_DIR}/launch.txt" | cut -d= -f2)"
log="$(grep -o 'log=[^ ]*' "${JOB_DIR}/launch.txt" | head -1 | cut -d= -f2)"
[ -n "${pid}" ] || fail "no LAUNCHED pid in launch output"
ln -sf "${log}" "${JOB_DIR}/run.log"
say "run pid ${pid}, log ${log}"
while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
status="$(grep -q -E "Job 'raysubmit_[A-Za-z0-9]+' succeeded" "${log}" && echo succeeded || echo "ended (no success line; see ${log})")"
echo "${status}" > "${JOB_DIR}/run.status"
say "run ${status}"
ray stop --force >/dev/null 2>&1 || true
