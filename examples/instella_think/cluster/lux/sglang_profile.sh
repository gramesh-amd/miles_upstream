#!/bin/bash
# Torch-profile the sglang rollout engines of a running run.sbatch job, from the login node.
#
#   sglang_profile.sh <jobid> [LABEL] [ENGINES=1] [STEPS=10] [BY_STAGE=1]
#
# Takes the engine URLs the job registered with its router (add_worker lines in run.log) and POSTs
# /start_profile to the first ENGINES of them: STEPS prefill and STEPS decode forward steps each
# (profile_by_stage), no stacks or shapes. Every DP rank of an engine writes its own
# <LABEL>-...-TP-<n>-DP-<n>-EP-<n>-EXTEND|DECODE.trace.json.gz into PROFILE_DIR/sglang (PROFILE_DIR
# from the job's spec, as run_async.sh sets it). Call it once rollouts are steady (after the request
# ramp). Analyse with profile_breakdown.py.
#
# BY_STAGE=1 (sglang's non-V2 profiler) profiles whichever stage comes first and, in steady state with
# requests decoding, finishes after the decode steps without ever profiling a prefill. BY_STAGE=0
# records the next STEPS forward passes of both kinds in one trace (<LABEL>-...-TP-n-DP-n-EP-n.trace...);
# profile_breakdown.py then tells prefill from decode only by kernel names.
set -uo pipefail
LUX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${LUX_DIR}/lux_env.sh"
JOB="${1:?usage: sglang_profile.sh <jobid> [LABEL] [ENGINES] [STEPS]}"
LABEL="${2:-sglang}"
ENGINES="${3:-1}"
STEPS="${4:-10}"
BY_STAGE="$([ "${5:-1}" = 1 ] && echo true || echo false)"
JOB_DIR="${LUX_ROOT}/shared/jobs/${JOB}"
[ -e "${JOB_DIR}/run.log" ] || { echo "no run.log in ${JOB_DIR}"; exit 1; }
SAVE_TAG="$(source "${JOB_DIR}/spec.env"; echo "${SAVE_TAG:?spec has no SAVE_TAG}")"
OUT="$(source "${JOB_DIR}/spec.env"; echo "${PROFILE_DIR:-${LUX_ROOT}/shared/rollout-dumps/${SAVE_TAG}/profile}")/sglang"
mkdir -p "${OUT}"
mapfile -t URLS < <(tr '\r' '\n' < "${JOB_DIR}/run.log" | grep -o 'add_worker?url=http://[0-9.:]*' | cut -d= -f2 | sort -u | head -n "${ENGINES}")
[ "${#URLS[@]}" -gt 0 ] || { echo "no engine URLs in ${JOB_DIR}/run.log yet"; exit 1; }
body="{\"output_dir\": \"${OUT}\", \"num_steps\": ${STEPS}, \"profile_by_stage\": ${BY_STAGE}, \"with_stack\": false, \"record_shapes\": false, \"profile_prefix\": \"${LABEL}\"}"
for u in "${URLS[@]}"; do
   echo "$(date '+%F %T') ${u}/start_profile ${body}"
   ( curl -s -m 1800 -H 'Content-Type: application/json' -d "${body}" "${u}/start_profile"; echo " <- ${u} done $(date '+%T')" ) &
done
wait
ls -la "${OUT}" | grep -c "${LABEL}" | sed "s/^/trace files with prefix ${LABEL}: /"
