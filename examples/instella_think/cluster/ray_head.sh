#!/bin/bash
# Start the Ray HEAD in this node's container and return.
#
#   cluster/on_node.sh <JOBID> <HEAD_NODE> cluster/ray_head.sh <HEAD_IP>
#
# Head and workers are separate payloads because a combined head+worker srun step often
# never returns on this cluster even though `ray start --head` succeeded, and re-running
# it would `ray stop --force` the healthy head.
set -uo pipefail
HEAD_IP="${1:?usage: ray_head.sh <HEAD_IP> [PORT] [DASHBOARD_PORT]}"
PORT="${2:-6379}"
DASH="${3:-8265}"
NPROC="${NPROC:-8}"
HOST="$(hostname -s)"

# shellcheck disable=SC1091
source /opt/venv/bin/activate 2>/dev/null

echo "[${HOST}] starting ray head on ${HEAD_IP}:${PORT} (dashboard ${DASH})"
ray stop --force >/dev/null 2>&1 || true
sleep 2
ray start --head --node-ip-address="${HEAD_IP}" --port="${PORT}" \
   --num-gpus="${NPROC}" --dashboard-host=0.0.0.0 --dashboard-port="${DASH}" \
   --disable-usage-stats 2>&1 | tail -4
echo "[${HOST}] raylet procs: $(pgrep -c raylet || echo 0)"
