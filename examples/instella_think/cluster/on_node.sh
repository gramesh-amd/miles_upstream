#!/usr/bin/env bash
# From the login node, run a payload inside the container on one node of a job.
#
#   cluster/on_node.sh <JOBID> <NODE> <payload.sh> [VAR=VALUE ...] [args...]
#   cluster/on_node.sh <JOBID> <NODE> run_async.sh PHASE=prepare
#   cluster/on_node.sh <JOBID> <NODE> cluster/env_setup.sh
#
# Payload paths are resolved relative to this repo, so "cluster/ray_head.sh" works.
# The step returns when the payload does: Spur allows only one interactive step per job
# per node, so payloads must not block (see ray_head.sh / launch.sh).
set -uo pipefail
# Site-specific: Spur's srun/squeue find the controllers through SPUR_CONTROLLER_ADDR
# (comma-separated http://host:port list); export it in your login shell.
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:?set SPUR_CONTROLLER_ADDR (http://host:port,... of the Spur controllers)}"

CLUSTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INSTELLA_DIR="$(dirname "${CLUSTER_DIR}")"
JOBID="${1:?usage: on_node.sh <JOBID> <NODE> <payload.sh> [args...]}"
NODE="${2:?usage: on_node.sh <JOBID> <NODE> <payload.sh> [args...]}"
PAYLOAD="${3:?usage: on_node.sh <JOBID> <NODE> <payload.sh> [args...]}"
shift 3

case "${PAYLOAD}" in
   /*) ;;
   *) PAYLOAD="${INSTELLA_DIR}/${PAYLOAD}" ;;
esac
[ -f "${PAYLOAD}" ] || { echo "FATAL: payload missing: ${PAYLOAD}" >&2; exit 1; }

args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
# srun on Spur wants a TTY, hence `script`.
exec script -qec "srun --jobid=${JOBID} --overlap -w ${NODE} bash ${CLUSTER_DIR}/in_container.sh ${JOBID} ${PAYLOAD}${args}" /dev/null
