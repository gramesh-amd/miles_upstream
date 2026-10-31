#!/usr/bin/env bash
# Bring the miles training container up on every node of a job you already hold.
# Run on the login node.
#
#   cluster/bringup.sh <JOBID>
#   IMAGE=... PORT=... cluster/bringup.sh <JOBID>
#
# The remote step returns once the container is up. Spur allows a single interactive step
# per job per node, so a step left running blocks every later srun on that node with "an
# interactive session for this job is already active on this node". The container is owned
# by dockerd and outlives the step; Spur's epilog removes it by its spur_job_id label.
set -uo pipefail
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817,http://crs-m2m-cpu-spur-006.crusoe.amd.com:6817,http://crs-m2m-cpu-spur-007.crusoe.amd.com:6817}"

CLUSTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
JOBID="${1:?usage: bringup.sh <JOBID>}"
IMAGE="${IMAGE:-rocm/sgl-dev:miles-rocm720-mi35x-20260727}"
PORT="${PORT:-18822}"
USER_NAME="$(id -un)"
HOME_DIR="${HOME}"
NAME="${NAME:-${USER_NAME}-miles-dev}"
SETUP="${CLUSTER_DIR}/node_setup.sh"
LOGDIR="${HOME_DIR}/logs"
STATUS_DIR="${HOME_DIR}/.miles/status-${JOBID}"

squeue -j "$JOBID" -h -o "%T" 2>/dev/null | grep -q RUNNING \
    || { echo "ERROR: job $JOBID is not RUNNING" >&2; exit 1; }

mkdir -p "$LOGDIR"
rm -rf "$STATUS_DIR"; mkdir -p "$STATUS_DIR"

NODES=$("${CLUSTER_DIR}/nodes.sh" "$JOBID")
[ -n "$NODES" ] || { echo "ERROR: could not resolve nodes for job $JOBID" >&2; exit 1; }
echo "job $JOBID -> nodes: $(echo $NODES | tr '\n' ' ')"
echo "image: $IMAGE"

for node in $NODES; do
    log="${LOGDIR}/bringup-${JOBID}-${node}.log"
    # base64 keeps the nested quoting intact through srun -> bash -c -> payload.
    payload=$(printf 'export IMAGE=%q PORT=%q USER_NAME=%q HOME_DIR=%q NAME=%q SLURM_JOB_ID=%q\nbash %q\n' \
        "$IMAGE" "$PORT" "$USER_NAME" "$HOME_DIR" "$NAME" "$JOBID" "$SETUP" | base64 -w0)
    # srun on Spur wants a TTY, hence `script`.
    nohup script -qec \
        "srun --jobid=$JOBID --overlap -w $node bash -lc 'echo $payload | base64 -d | bash'" \
        /dev/null > "$log" 2>&1 &
    echo "  $node -> $log (pid $!)"
    disown 2>/dev/null || true
done

echo
echo "Bring-up running in the background (image pull can take 10-20 min if uncached)."
echo "Ready when every node has a file in: ${STATUS_DIR}/"
echo "Watch a node:  tail -f ${LOGDIR}/bringup-${JOBID}-<node>.log"
