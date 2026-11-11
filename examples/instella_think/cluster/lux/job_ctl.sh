#!/bin/bash
# Control a running run.sbatch job that holds its nodes (HOLD=1 in its spec). From the login node:
#
#   job_ctl.sh <jobid> status                 # what the job is doing (running / holding / released)
#   job_ctl.sh <jobid> stop                   # stop the current run; the job keeps its nodes
#   job_ctl.sh <jobid> run VAR=VALUE ...      # start the next run (same form as run.sbatch's spec)
#   job_ctl.sh <jobid> release                # end the job and give the nodes back
#
# `run` only queues the spec: the job starts it once the current run has ended or been stopped. The
# node layout comes from the allocation (ROLLOUT_NODES is copied from the job's first spec), and the
# new spec should keep HOLD=1 to go on holding afterwards. The job's Slurm time limit still applies.
set -uo pipefail
LUX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${LUX_DIR}/lux_env.sh"
JOB="${1:?usage: job_ctl.sh <jobid> status|stop|run VAR=VALUE...|release}"
CMD="${2:-status}"
shift 2 2>/dev/null || shift $#
JOB_DIR="${LUX_ROOT}/shared/jobs/${JOB}"
[ -d "${JOB_DIR}" ] || { echo "no job dir ${JOB_DIR}"; exit 1; }

case "${CMD}" in
status)
   squeue -j "${JOB}" -h -o "%i %j %T %M / %l" 2>/dev/null || true
   echo "run: $(cat "${JOB_DIR}/run.status" 2>/dev/null || echo unknown)"
   [ -e "${JOB_DIR}/released" ] && echo "released"
   [ -e "${JOB_DIR}/stop.request" ] && echo "stop requested"
   ls "${JOB_DIR}" | grep -E '^next\.[0-9]+\.env$' | sort -t. -k2 -n | tail -1 | sed 's/^/latest queued spec: /'
   ;;
stop)
   touch "${JOB_DIR}/stop.request"
   echo "stop requested; the run stops within ~1 min and the job holds its nodes if HOLD=1"
   ;;
run)
   [ $# -gt 0 ] || { echo "run needs VAR=VALUE arguments"; exit 1; }
   n=1
   while [ -e "${JOB_DIR}/next.${n}.env" ]; do n=$((n + 1)); done
   tmp="${JOB_DIR}/.next.${n}.env.tmp"
   : > "${tmp}"
   for kv in "$@"; do
      [[ "${kv}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "not VAR=VALUE: ${kv}"; rm -f "${tmp}"; exit 1; }
      printf 'export %q\n' "${kv}" >> "${tmp}"
   done
   grep '^export ROLLOUT_NODES=' "${JOB_DIR}/spec.env" >> "${tmp}"
   mv "${tmp}" "${JOB_DIR}/next.${n}.env"
   echo "queued as run ${n} ($(wc -l < "${JOB_DIR}/next.${n}.env") settings); it starts when the current run has ended"
   ;;
release)
   touch "${JOB_DIR}/release.request"
   [ -e "${JOB_DIR}/stop.request" ] || touch "${JOB_DIR}/stop.request"
   echo "release requested; the job stops its run (if any) and ends"
   ;;
*)
   echo "unknown command ${CMD}"; exit 1 ;;
esac
