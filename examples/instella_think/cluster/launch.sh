#!/bin/bash
# Launch a recipe from inside the HEAD container and return (the run itself is nohup'd).
#
#   cluster/on_node.sh <JOBID> <HEAD_NODE> cluster/launch.sh <HEAD_IP> [VAR=VALUE ...]
#   cluster/on_node.sh ... cluster/launch.sh <HEAD_IP> SINGLE_NODE=1 NUM_ROLLOUT=8
#   cluster/on_node.sh ... cluster/launch.sh <HEAD_IP> RUN_SCRIPT=run_if.sh
#
# Overrides are passed as VAR=VALUE args rather than env because srun does not forward
# the caller's environment to the node.
set -uo pipefail
# shellcheck disable=SC1091
source /opt/venv/bin/activate 2>/dev/null

INSTELLA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${INSTELLA_DIR}/../.." &>/dev/null && pwd)"
HEAD_IP="${1:?usage: launch.sh <HEAD_IP> [VAR=VALUE ...]}"
shift

export RAY_ADDRESS="http://${HEAD_IP}:8265"
export MASTER_ADDR="${HEAD_IP}"

for kv in "$@"; do
   case "${kv}" in
      *=*) export "${kv?}" ;;
      *) echo "ignoring non VAR=VALUE arg: ${kv}" >&2 ;;
   esac
done

RUN_SCRIPT="${RUN_SCRIPT:-run_code.sh}"
cd "${REPO_DIR}" || exit 1
echo "[$(hostname -s)] RAY_ADDRESS=${RAY_ADDRESS} RUN_SCRIPT=${RUN_SCRIPT} overrides: $*"
bash "${INSTELLA_DIR}/${RUN_SCRIPT}"
rc=$?
echo "[$(hostname -s)] ${RUN_SCRIPT} rc=${rc}"
exit "${rc}"
