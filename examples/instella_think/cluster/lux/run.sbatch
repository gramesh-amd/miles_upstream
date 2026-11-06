#!/bin/bash
#SBATCH -J instella-run
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH -p batch
# One complete code RL run in one Slurm job. The first SANDBOX_NODES nodes (default 1) run the
# code-exec sandboxes; the rest are GPU nodes: 1 => the SINGLE_NODE layout, N >= 2 => 1 trainer node
# + N-1 rollout nodes (one engine each; 3 is the 3-node recipe). With SANDBOX_NODES=2, node 0 serves
# training rewards and node 1 the benchmark evals. The arguments are the run spec, the same
# VAR=VALUE overrides cluster/launch.sh takes on Crusoe. From the miles checkout:
#
#   sbatch -A ven136 -N 4 -t 24:00:00 -o ~/logs/%x-%j.out examples/instella_think/cluster/lux/run.sbatch \
#     CONTEXT_LEN=32768 DRGRPO=1 DRGRPO_DIVISOR=17400 OVERLONG_FILTER=1 SAVE_TAG=code32k_drgrpo
#
# The job ends when the run does (or fails). Logs: this job's -o file, $LUX_ROOT/shared/jobs/<jobid>/
# (per-node setup logs, the run's launch output) and the run log under $LOG_DIR. Metrics go to W&B
# live; add WANDB_MODE=offline to the spec to log locally instead.
set -uo pipefail
LUX_DIR="${LUX_DIR:-${SLURM_SUBMIT_DIR}/examples/instella_think/cluster/lux}"
[ -f "${LUX_DIR}/lux_env.sh" ] || { echo "FATAL: submit from the miles checkout (or set LUX_DIR)"; exit 1; }
source "${LUX_DIR}/lux_env.sh"

SANDBOX_NODES=1
for kv in "$@"; do case "${kv}" in SANDBOX_NODES=*) SANDBOX_NODES="${kv#*=}" ;; esac; done
[ "${SANDBOX_NODES}" = 1 ] || [ "${SANDBOX_NODES}" = 2 ] || { echo "FATAL: SANDBOX_NODES must be 1 or 2"; exit 1; }
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SANDBOXES=("${NODES[@]:0:${SANDBOX_NODES}}")
SANDBOX_HOST="${SANDBOXES[0]}"
GPU_NODES=("${NODES[@]:${SANDBOX_NODES}}")
G=${#GPU_NODES[@]}
[ "${G}" -ge 1 ] || { echo "FATAL: -N ${#NODES[@]} leaves no GPU nodes after ${SANDBOX_NODES} sandbox node(s)"; exit 1; }

JOB_DIR="${LUX_ROOT}/shared/jobs/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"
# The run spec, exported in the head node's container before run_code.sh.
: > "${JOB_DIR}/spec.env"
for kv in "$@"; do
   [[ "${kv}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "FATAL: not VAR=VALUE: ${kv}"; exit 1; }
   printf 'export %q\n' "${kv}" >> "${JOB_DIR}/spec.env"
done
# The rollout node count follows from the allocation.
[ "${G}" -ge 2 ] && echo "export ROLLOUT_NODES=$(( G - 1 ))" >> "${JOB_DIR}/spec.env"
echo "job ${SLURM_JOB_ID}: sandbox ${SANDBOXES[*]}, GPU nodes ${GPU_NODES[*]} (head ${GPU_NODES[0]})"
echo "spec: $*"
echo "job dir: ${JOB_DIR}"

# --cpu-bind=none: the default binding gives each task a single core on Lux.
# sandbox.ready names the training-reward host, sandbox-bench.ready the benchmark host.
SANDBOX_PIDS=()
if [ "${SANDBOX_NODES}" = 1 ]; then
   srun -N 1 -n 1 -w "${SANDBOXES[0]}" --overlap --cpu-bind=none \
      --export=ALL,READY_FILE="${JOB_DIR}/sandbox.ready",BENCH_READY_FILE="${JOB_DIR}/sandbox-bench.ready" \
      bash "${LUX_DIR}/run_sandbox_node.sh" > "${JOB_DIR}/sandbox.log" 2>&1 &
   SANDBOX_PIDS+=($!)
else
   srun -N 1 -n 1 -w "${SANDBOXES[0]}" --overlap --cpu-bind=none \
      --export=ALL,READY_FILE="${JOB_DIR}/sandbox.ready",CODE_BENCH_WORKERS=8 \
      bash "${LUX_DIR}/run_sandbox_node.sh" > "${JOB_DIR}/sandbox.log" 2>&1 &
   SANDBOX_PIDS+=($!)
   srun -N 1 -n 1 -w "${SANDBOXES[1]}" --overlap --cpu-bind=none \
      --export=ALL,READY_FILE="${JOB_DIR}/sandbox-bench.ready",CODE_BENCH_WORKERS=112,CODE_EXEC_WORKERS=8 \
      bash "${LUX_DIR}/run_sandbox_node.sh" > "${JOB_DIR}/sandbox-bench.log" 2>&1 &
   SANDBOX_PIDS+=($!)
fi
srun -N "${G}" -n "${G}" --ntasks-per-node=1 -w "$(IFS=,; echo "${GPU_NODES[*]}")" --overlap --cpu-bind=none \
   --export=ALL,LUX_DIR="${LUX_DIR}",JOB_DIR="${JOB_DIR}",SANDBOX_HOST="${SANDBOX_HOST}",HEAD_HOST="${GPU_NODES[0]}",GPU_NODE_COUNT="${G}",ALL_HOSTS="$(IFS=,; echo "${NODES[*]}")" \
   bash "${LUX_DIR}/run_gpu_node.sh" &
GPU_PID=$!

wait "${GPU_PID}"
rc=$?
echo "GPU nodes finished (rc=${rc}); stopping the sandbox"
kill "${SANDBOX_PIDS[@]}" 2>/dev/null
wait "${SANDBOX_PIDS[@]}" 2>/dev/null
[ -f "${JOB_DIR}/run.status" ] && echo "run status: $(cat "${JOB_DIR}/run.status")"
exit "${rc}"
