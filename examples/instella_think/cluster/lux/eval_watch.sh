#!/bin/bash
# Submit one eval-only run.sbatch job for every checkpoint of a training run that has none yet,
# so benchmark evals run on their own nodes (and their own grading node) while training goes on.
# One pass, idempotent: submitted checkpoints are marked in SAVE_DIR/.eval_submitted/. Run it from
# the miles checkout on the login node, by hand or on a schedule:
#
#   examples/instella_think/cluster/lux/eval_watch.sh <SAVE_DIR> [VAR=VALUE ...]
#
# The VAR=VALUE args are the eval job's run spec: the training spec, so the eval uses the same
# model and lengths (sampling and benchmark settings come from run_code.sh). EVAL_NODES (default 4:
# grading node + 3 inference engines; eval_in_container.sh converts the checkpoint to HF format and
# runs no trainer) and EVAL_HOURS (default 6) size the jobs. Each job also uploads its scores to W&B
# and adds them to the training run's eval-series run (EVAL_SERIES_ID in the spec, default
# evals-<SAVE_DIR name>). The training job's in-flight checkpoint is skipped until
# latest_checkpointed_iteration.txt reaches it.
set -uo pipefail
LUX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${LUX_DIR}/lux_env.sh"
SAVE_DIR="${1:?usage: eval_watch.sh <SAVE_DIR> [VAR=VALUE ...]}"
shift
EVAL_NODES="${EVAL_NODES:-4}"
EVAL_HOURS="${EVAL_HOURS:-6}"
MARKS="${SAVE_DIR}/.eval_submitted"
mkdir -p "${MARKS}"

latest="$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)"
[[ "${latest}" =~ ^[0-9]+$ ]] || { echo "no complete checkpoint in ${SAVE_DIR} yet"; exit 0; }
for d in "${SAVE_DIR}"/iter_*; do
   [ -d "${d}" ] || continue
   step=$((10#${d##*iter_}))
   [ "${step}" -le "${latest}" ] || continue        # still being written
   [ -e "${MARKS}/${step}" ] && continue
   # Engine cap: 256 per engine (32/GPU) unless the spec sets one. At 64/GPU, sustained full load
   # crashed every prefilter server within ~35 min (HSA_STATUS_ERROR_EXCEPTION); 32/GPU plus the
   # eval request ramp (EVAL_RAMP_S) scored the same as 16/GPU (v5 iter 599, 2026-11-11).
   cap=SGLANG_MAX_RUNNING_REQUESTS=256
   for kv in "$@"; do case "${kv}" in SGLANG_MAX_RUNNING_REQUESTS=*) cap="${kv}" ;; esac; done
   # The training spec's sandbox sizing favours rewards; an eval job grades benchmarks.
   job="$(cd "${LUX_DIR}/../../../.." && sbatch --parsable -A "${LUX_ACCOUNT}" -N "${EVAL_NODES}" -t "${EVAL_HOURS}:00:00" \
      -J instella-eval -o "${LUX_HOME}/logs/%x-%j.out" "${LUX_DIR}/run.sbatch" "$@" "${cap}" \
      CODE_EXEC_WORKERS=8 CODE_BENCH_WORKERS=112 EVAL_ONLY=1 EVAL_STEP="${step}" EVAL_SOURCE_DIR="${SAVE_DIR}")" \
      || { echo "sbatch failed for iter ${step}"; continue; }
   echo "${job}" > "${MARKS}/${step}"
   echo "iter ${step}: submitted eval job ${job}"
done
echo "results so far: ${SAVE_DIR}/.evals.txt"
