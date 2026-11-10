#!/bin/bash
#SBATCH -J instella-prefilter
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH -p batch
# Offline difficulty prefilter (../../prefilter.sh, difficulty_filter.py; README "5b") across many
# Lux nodes. The prompt set is split into shards beforehand (prefilter_shards.py split); each GPU
# node serves the base checkpoint with sglang and filters one shard, and the first SANDBOX_NODES
# nodes (default 2) grade the programs. Results are cached per prompt, so a job that ends early
# (time limit, outage) loses only in-flight prompts: resubmit the same shards to finish them. From
# the miles checkout:
#
#   python3 examples/instella_think/cluster/lux/prefilter_shards.py split 14        # once
#   sbatch -A ven136 -N 16 -t 9:00:00 -o ~/logs/%x-%j.out examples/instella_think/cluster/lux/prefilter.sbatch
#   python3 examples/instella_think/cluster/lux/prefilter_shards.py status
#   python3 examples/instella_think/cluster/lux/prefilter_shards.py merge           # when all are done
#
# GPU node k (0-based) filters shard SHARD_BASE+k. Settings pass through the environment
# (prefilter.sh's CONCURRENCY, SGLANG_MAX_RUNNING_REQUESTS, N, MAX_PASS_RATE, ...).
set -uo pipefail
LUX_DIR="${LUX_DIR:-${SLURM_SUBMIT_DIR}/examples/instella_think/cluster/lux}"
[ -f "${LUX_DIR}/lux_env.sh" ] || { echo "FATAL: submit from the miles checkout (or set LUX_DIR)"; exit 1; }
source "${LUX_DIR}/lux_env.sh"

SANDBOX_NODES="${SANDBOX_NODES:-2}"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SANDBOXES=("${NODES[@]:0:${SANDBOX_NODES}}")
GPU_NODES=("${NODES[@]:${SANDBOX_NODES}}")
G=${#GPU_NODES[@]}
[ "${G}" -ge 1 ] || { echo "FATAL: no GPU nodes after ${SANDBOX_NODES} sandbox node(s)"; exit 1; }
JOB_DIR="${LUX_ROOT}/shared/jobs/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"
echo "job ${SLURM_JOB_ID}: sandboxes ${SANDBOXES[*]}, ${G} GPU nodes ${GPU_NODES[*]}, shards ${SHARD_BASE:-0}..$(( ${SHARD_BASE:-0} + G - 1 ))"

PIDS=()
for i in "${!SANDBOXES[@]}"; do
   srun -N 1 -n 1 -w "${SANDBOXES[$i]}" --overlap --cpu-bind=none \
      --export=ALL,READY_FILE="${JOB_DIR}/sandbox${i}.ready",CODE_EXEC_WORKERS="${CODE_EXEC_WORKERS:-112}",CODE_EXEC_PROCESSES="${CODE_EXEC_PROCESSES:-8}",CODE_BENCH_WORKERS=4 \
      bash "${LUX_DIR}/run_sandbox_node.sh" > "${JOB_DIR}/sandbox${i}.log" 2>&1 &
   PIDS+=($!)
done
# --kill-on-bad-exit=0: one node giving up must not kill the others' shards (Lux kills the whole
# step on a failed task by default; that ended job 407229 1.5 h early).
srun -N "${G}" -n "${G}" --ntasks-per-node=1 -w "$(IFS=,; echo "${GPU_NODES[*]}")" --overlap --cpu-bind=none --kill-on-bad-exit=0 \
   --export=ALL,LUX_DIR="${LUX_DIR}",JOB_DIR="${JOB_DIR}",SANDBOX_NODES="${SANDBOX_NODES}",SHARD_BASE="${SHARD_BASE:-0}" \
   bash "${LUX_DIR}/prefilter_node.sh" &
GPU_PID=$!
wait "${GPU_PID}"
rc=$?
echo "GPU nodes finished (rc=${rc}); stopping the sandboxes"
kill "${PIDS[@]}" 2>/dev/null
wait "${PIDS[@]}" 2>/dev/null
exit "${rc}"
