#!/bin/bash
# Join this node's container to an existing Ray head as a worker, and return.
#
#   cluster/on_node.sh <JOBID> <NODE> cluster/ray_worker.sh <HEAD_IP>
set -uo pipefail
HEAD_IP="${1:?usage: ray_worker.sh <HEAD_IP> [PORT]}"
PORT="${2:-6379}"
NPROC="${NPROC:-8}"
HOST="$(hostname -s)"

# shellcheck disable=SC1091
source /opt/venv/bin/activate 2>/dev/null

echo "[${HOST}] joining ray head ${HEAD_IP}:${PORT}"
ray stop --force >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
   (exec 3<>"/dev/tcp/${HEAD_IP}/${PORT}") 2>/dev/null && { exec 3>&- 3<&-; break; }
   sleep 2
done
# --block would hold the step open forever; it must return so the caller continues.
ray start --address="${HEAD_IP}:${PORT}" --num-gpus="${NPROC}" --disable-usage-stats 2>&1 | tail -4
echo "[${HOST}] raylet procs: $(pgrep -c raylet || echo 0)"
