#!/bin/bash
#SBATCH -J instella-run
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH -p batch
# One complete code RL run in one Slurm job. Node 0 runs the code-exec sandboxes; the other
# nodes are GPU nodes: 1 => the SINGLE_NODE layout, 3 => the 3-node recipe. The arguments are the
# run spec, the same VAR=VALUE overrides cluster/launch.sh takes on Crusoe. From the miles checkout:
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

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SANDBOX_HOST="${NODES[0]}"
GPU_NODES=("${NODES[@]:1}")
G=${#GPU_NODES[@]}
if [ "${G}" != 1 ] && [ "${G}" != 3 ]; then
   echo "FATAL: need -N 2 (sandbox + 1 GPU node) or -N 4 (sandbox + 3), got -N ${#NODES[@]}"; exit 1
fi

JOB_DIR="${LUX_ROOT}/shared/jobs/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"
# The run spec, exported in the head node's container before run_code.sh.
: > "${JOB_DIR}/spec.env"
for kv in "$@"; do
   [[ "${kv}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "FATAL: not VAR=VALUE: ${kv}"; exit 1; }
   printf 'export %q\n' "${kv}" >> "${JOB_DIR}/spec.env"
done
echo "job ${SLURM_JOB_ID}: sandbox ${SANDBOX_HOST}, GPU nodes ${GPU_NODES[*]} (head ${GPU_NODES[0]})"
echo "spec: $*"
echo "job dir: ${JOB_DIR}"

# --cpu-bind=none: the default binding gives each task a single core on Lux.
srun -N 1 -n 1 -w "${SANDBOX_HOST}" --overlap --cpu-bind=none --export=ALL,READY_FILE="${JOB_DIR}/sandbox.ready" \
   bash "${LUX_DIR}/run_sandbox_node.sh" > "${JOB_DIR}/sandbox.log" 2>&1 &
SANDBOX_PID=$!
srun -N "${G}" -n "${G}" --ntasks-per-node=1 -w "$(IFS=,; echo "${GPU_NODES[*]}")" --overlap --cpu-bind=none \
   --export=ALL,LUX_DIR="${LUX_DIR}",JOB_DIR="${JOB_DIR}",SANDBOX_HOST="${SANDBOX_HOST}",HEAD_HOST="${GPU_NODES[0]}",GPU_NODE_COUNT="${G}",ALL_HOSTS="$(IFS=,; echo "${NODES[*]}")" \
   bash "${LUX_DIR}/run_gpu_node.sh" &
GPU_PID=$!

wait "${GPU_PID}"
rc=$?
echo "GPU nodes finished (rc=${rc}); stopping the sandbox"
kill "${SANDBOX_PID}" 2>/dev/null
wait "${SANDBOX_PID}" 2>/dev/null
[ -f "${JOB_DIR}/run.status" ] && echo "run status: $(cat "${JOB_DIR}/run.status")"
exit "${rc}"
