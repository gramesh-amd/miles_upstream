#!/bin/bash
# Inside the container of one GPU node of a run.sbatch job (see run_gpu_node.sh): set the
# container up, then take this node's role.
#   1 GPU node : run the spec with SINGLE_NODE=1 (run_async.sh starts its own Ray head)
#   N GPU nodes: rank 0 starts the Ray head, waits for the workers and runs the spec;
#                ranks 1..N-1 join as Ray workers until the head stops
# Coordination is through files in $JOB_DIR: setup.<host>, head_ip, sandbox[-bench].ready, failed,
# run.status, released.
#
# HOLD=1 in the spec keeps the job (and its nodes, sandbox and set-up containers) after the run ends
# or is stopped, and runs the next spec it is given; see job_ctl.sh (stop / run / release / status).
# Every run starts from a fresh Ray cluster and in a subshell, so nothing leaks between runs.
set -uo pipefail
# No core dumps: a killed sglang dumps ~2.4 GB per GPU process into the working directory (the NFS
# home), which filled the home quota on 2026-11-09 (see eval_in_container.sh).
ulimit -c 0
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
   # One Ray cluster per run: the head removes head_ip before stopping Ray and writes it again for
   # the next run, and touches `released` when the job is done.
   while true; do
      until [ -e "${JOB_DIR}/head_ip" ] || [ -e "${JOB_DIR}/released" ]; do
         [ -e "${JOB_DIR}/failed" ] && fail "head failed: $(cat "${JOB_DIR}/failed")"
         sleep 5
      done
      [ -e "${JOB_DIR}/released" ] && { say "job released"; exit 0; }
      HEAD_IP="$(cat "${JOB_DIR}/head_ip")"
      say "joining Ray head ${HEAD_IP}"
      # --block keeps Ray inside this session; it returns when the head's Ray stops.
      ray start --address="${HEAD_IP}:6379" --num-gpus 8 --disable-usage-stats --block
      ray stop --force >/dev/null 2>&1 || true
      say "Ray worker exited"
      sleep 10
   done
fi

# --- 2b. head: the sandbox, the Ray cluster, then the run -----------------------------------
await "${JOB_DIR}/sandbox.ready" 900 "the reward sandbox (see ${JOB_DIR}/sandbox.log)"
await "${JOB_DIR}/sandbox-bench.ready" 900 "the benchmark sandbox (see ${JOB_DIR}/sandbox*.log)"
TOK="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}")"
export CODE_EXEC_API_URL="http://$(cat "${JOB_DIR}/sandbox.ready"):${CODE_EXEC_PORT}/${TOK}"
export CODE_BENCH_API_URL="http://$(cat "${JOB_DIR}/sandbox-bench.ready"):${CODE_BENCH_PORT}/${TOK}"

start_ray() {
   if [ "${GPU_NODE_COUNT}" = 1 ]; then
      export SINGLE_NODE=1
      return
   fi
   ray stop --force >/dev/null 2>&1 || true
   ray start --head --node-ip-address="${NODE_IP}" --port=6379 --num-gpus 8 \
      --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats || fail "ray start --head failed"
   echo "${NODE_IP}" > "${JOB_DIR}/head_ip"
   local waited=0
   until [ "$(python3 -c 'import ray; ray.init(address="auto", logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0)))' 2>/dev/null)" = "$(( GPU_NODE_COUNT * 8 ))" ]; do
      [ -e "${JOB_DIR}/failed" ] && fail "a worker failed: $(cat "${JOB_DIR}/failed")"
      [ "${waited}" -ge 1800 ] && fail "Ray cluster never reached $(( GPU_NODE_COUNT * 8 )) GPUs"
      sleep 10; waited=$((waited + 10))
   done
   say "Ray cluster up: $(( GPU_NODE_COUNT * 8 )) GPUs"
   export RAY_ADDRESS="http://${NODE_IP}:8265" MASTER_ADDR="${NODE_IP}"
}

stop_ray() {  # stops the run's Ray jobs and the cluster (workers wait for the next head_ip)
   for id in $(ray job list 2>/dev/null | grep -o "raysubmit_[A-Za-z0-9]*" | sort -u); do ray job stop "${id}" >/dev/null 2>&1; done
   rm -f "${JOB_DIR}/head_ip"
   ray stop --force >/dev/null 2>&1 || true
   sleep 15
}

release() {
   touch "${JOB_DIR}/released"
   stop_ray
   say "job released"
   exit 0
}

# One run per spec: spec.env first, then (HOLD=1) each next.<n>.env that job_ctl.sh writes.
SPEC="${JOB_DIR}/spec.env"
n=0
while true; do
   start_ray
   HOLD="$(source "${SPEC}"; echo "${HOLD:-0}")"
   EVAL_ONLY="$(source "${SPEC}"; echo "${EVAL_ONLY:-0}")"
   rm -f "${JOB_DIR}/stop.request"
   echo "running ${SPEC}" > "${JOB_DIR}/run.status"
   say "launching ${RUN_SCRIPT:-run_code.sh} with: $(sed 's/^export //' "${SPEC}" | tr '\n' ' ')"
   # A subshell, so this spec's settings do not outlive its run.
   (
      # The spec last, so it can override anything above.
      source "${SPEC}"
      # EVAL_ONLY=1 (legacy path; run.sbatch now sends eval-only jobs to eval_gpu_node.sh): evaluate
      # EVAL_SOURCE_DIR/iter_<EVAL_STEP>, loaded the way a fresh run loads its base model.
      if [ "${EVAL_ONLY:-0}" = 1 ]; then
         : "${EVAL_STEP:?EVAL_ONLY needs EVAL_STEP}" "${EVAL_SOURCE_DIR:?EVAL_ONLY needs EVAL_SOURCE_DIR}"
         if [ "${EVAL_STEP}" = release ]; then
            export EXTRA_ARGS="${EXTRA_ARGS:-} --ref-load ${EVAL_SOURCE_DIR} --no-load-optim --no-load-rng"
         else
            export EXTRA_ARGS="${EXTRA_ARGS:-} --ref-load ${EVAL_SOURCE_DIR} --ckpt-step ${EVAL_STEP} --ref-ckpt-step ${EVAL_STEP} --no-load-optim --no-load-rng"
         fi
         export SAVE_TAG="${SAVE_TAG}_eval${EVAL_STEP}" SKIP_EVAL_BEFORE_TRAIN=0 EVAL_INTERVAL=100000 \
            PRUNE_CHECKPOINTS=0 WANDB_GROUP="${WANDB_GROUP:-eval}-eval-iter${EVAL_STEP}"
      fi
      bash "${I}/${RUN_SCRIPT:-run_code.sh}"
   ) 2>&1 | tee "${JOB_DIR}/launch.txt"
   if [ "${PIPESTATUS[0]}" != 0 ]; then
      [ "${HOLD}" = 1 ] || fail "${RUN_SCRIPT:-run_code.sh} failed before launching (preflight?)"
      echo "failed before launching (preflight?); holding" > "${JOB_DIR}/run.status"
      say "launch failed; holding for the next spec"
   else
      # run_code.sh backgrounds the run and prints its pid; it has to finish inside this session.
      pid="$(grep -o 'LAUNCHED pid=[0-9]*' "${JOB_DIR}/launch.txt" | cut -d= -f2)"
      log="$(grep -o 'log=[^ ]*' "${JOB_DIR}/launch.txt" | head -1 | cut -d= -f2)"
      [ -n "${pid}" ] || fail "no LAUNCHED pid in launch output"
      ln -sf "${log}" "${JOB_DIR}/run.log"
      say "run pid ${pid}, log ${log}"
      if [ "${EVAL_ONLY}" = 1 ]; then
         # The eval is logged as "eval 0: {...}"; once it is there, the training that follows is not wanted.
         until grep -q -E "^.*eval 0: \{" "${log}" 2>/dev/null; do
            kill -0 "${pid}" 2>/dev/null || fail "run ended before logging the eval (see ${log})"
            sleep 30
         done
         src="$(source "${SPEC}"; echo "iter ${EVAL_STEP} of ${EVAL_SOURCE_DIR}")"
         grep -o -E "eval 0: \{.*" "${log}" | head -1 > "${JOB_DIR}/eval_result.txt"
         echo "${src}: $(cat "${JOB_DIR}/eval_result.txt")" >> "$(source "${SPEC}"; echo "${EVAL_SOURCE_DIR}")/.evals.txt"
         echo "eval done" > "${JOB_DIR}/run.status"
         say "eval logged; stopping"
         release
      fi
      stopped=0
      while kill -0 "${pid}" 2>/dev/null; do
         if [ -e "${JOB_DIR}/stop.request" ]; then
            say "stop requested; stopping the run"
            stopped=1
            stop_ray
            for _ in $(seq 1 60); do kill -0 "${pid}" 2>/dev/null || break; sleep 5; done
            kill "${pid}" 2>/dev/null
            break
         fi
         sleep 30
      done
      if [ "${stopped}" = 1 ]; then
         status="stopped on request (see ${log})"
      else
         status="$(grep -q -E "Job 'raysubmit_[A-Za-z0-9]+' succeeded" "${log}" && echo succeeded || echo "ended (no success line; see ${log})")"
      fi
      echo "${status}" > "${JOB_DIR}/run.status"
      say "run ${status}"
   fi
   stop_ray
   [ "${HOLD}" = 1 ] || release
   # Holding: wait for the next spec (job_ctl.sh run) or a release (job_ctl.sh release / time limit).
   n=$((n + 1))
   echo "holding: $(cat "${JOB_DIR}/run.status")" > "${JOB_DIR}/run.status"
   say "holding the nodes; waiting for ${JOB_DIR}/next.${n}.env or release"
   until [ -e "${JOB_DIR}/next.${n}.env" ] || [ -e "${JOB_DIR}/release.request" ]; do sleep 10; done
   [ -e "${JOB_DIR}/release.request" ] && release
   SPEC="${JOB_DIR}/next.${n}.env"
done
