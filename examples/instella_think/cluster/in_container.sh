#!/bin/bash
# Run a payload script inside this node's container for a job. Runs ON THE COMPUTE NODE
# (outside the container); use on_node.sh to drive it from the login node.
#
#   bash in_container.sh <JOBID> <payload.sh> [VAR=VALUE ...] [args...]
#
# The container is resolved here, on the node, rather than in the srun command string:
# the login shell would expand a `$(docker ps ...)` locally, where there are no containers.
# srun does not carry the caller's environment, so leading VAR=VALUE args are set as env
# for the payload (e.g. `run_async.sh PHASE=prepare`); the rest are passed as args.
set -uo pipefail

JOBID="${1:?usage: in_container.sh <JOBID> <payload.sh> [VAR=VALUE ...] [args...]}"
PAYLOAD="${2:?usage: in_container.sh <JOBID> <payload.sh> [VAR=VALUE ...] [args...]}"
shift 2
HOST="$(hostname -s)"

ENV_ARGS=()
while [ $# -gt 0 ] && [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
   ENV_ARGS+=(-e "$1")
   shift
done

[ -f "${PAYLOAD}" ] || { echo "[${HOST}] FATAL: payload missing: ${PAYLOAD}"; exit 1; }

CN="$(docker ps --format '{{.Names}}' | grep -E -- "-${JOBID}-" | grep -v -- '-watchdog' | head -1)"
if [ -z "${CN}" ]; then
   echo "[${HOST}] FATAL: no container for job ${JOBID}. Containers present:"
   docker ps --format '  {{.Names}}  {{.Status}}'
   exit 1
fi

echo "[${HOST}] container=${CN}"
docker exec "${ENV_ARGS[@]}" "${CN}" bash "${PAYLOAD}" "$@"
rc=$?
echo "[${HOST}] docker exec rc=${rc}"
exit "${rc}"
