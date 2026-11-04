#!/usr/bin/env bash
# From the login node, run a command inside the miles container on one node of a node.sbatch
# job (the Lux counterpart of cluster/on_node.sh). The command runs as root in the container,
# from the user's home, via node_agent.sh; its background processes live until the job ends.
#
#   cluster/lux/on_node.sh [-b] [-p] [-t SECS] <JOBID> <NODE|-> '<command>'
#     -b  return at once and print the log path (for long or never-ending commands)
#     -p  set the ORNL proxy for this command (downloads, pip, git)
#     -t  wait at most SECS (default 1800), then print the log tail; the command keeps running
#   NODE '-' means the job's first node.
#
# e.g. cluster/lux/on_node.sh 386700 - 'cd ~/miles && bash examples/instella_think/cluster/env_setup.sh'
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/lux_env.sh"

BG=0; PROXY=0; TIMEOUT=1800
while getopts "bpt:" o; do
   case "$o" in b) BG=1 ;; p) PROXY=1 ;; t) TIMEOUT="$OPTARG" ;; *) sed -n '2,12p' "$0"; exit 2 ;; esac
done
shift $((OPTIND - 1))
JOBID="${1:?usage: on_node.sh [-b] [-p] [-t SECS] <JOBID> <NODE|-> '<command>'}"
NODE="${2:?node}"; CMD="${3:?command}"
[ "${NODE}" = "-" ] && NODE="$(squeue -j "${JOBID}" -h -o %N | sed -E 's/\[([0-9]+).*/\1/; s/,.*//')"

MB="${MAILBOX_ROOT}/${JOBID}/${NODE}"
[ -e "${MB}/ready" ] || { echo "FATAL: no ready agent at ${MB} (job not running or container still starting)" >&2; exit 2; }

ID="$(date +%Y%m%d_%H%M%S)-$$"
{
   echo "set -o pipefail"
   [ "${PROXY}" = 1 ] && echo "export http_proxy=\$LUX_PROXY https_proxy=\$LUX_PROXY HTTP_PROXY=\$LUX_PROXY HTTPS_PROXY=\$LUX_PROXY no_proxy=localhost,127.0.0.1"
   printf '%s\n' "${CMD}"
} > "${MB}/${ID}.tmp"
mv "${MB}/${ID}.tmp" "${MB}/${ID}.cmd"
LOG="${MB}/${ID}.log"

if [ "${BG}" = 1 ]; then
   echo "submitted ${ID} on ${NODE}; log ${LOG}"
   exit 0
fi
waited=0
while [ ! -e "${MB}/${ID}.rc" ]; do
   if [ "${waited}" -ge "${TIMEOUT}" ]; then
      tail -n 40 "${LOG}" 2>/dev/null
      echo "on_node: still running after ${TIMEOUT}s; log ${LOG}" >&2
      exit 124
   fi
   sleep 2; waited=$((waited + 2))
done
cat "${LOG}"
exit "$(cat "${MB}/${ID}.rc")"
