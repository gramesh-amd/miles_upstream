#!/bin/bash
# One GPU node of an eval-only run.sbatch job (EVAL_ONLY=1 in the spec): run_gpu_node.sh's
# container start, running eval_in_container.sh instead of run_in_container.sh. Every GPU node of
# an eval job serves inference; none is a trainer.
#
# A separate file rather than a switch in run_gpu_node.sh: a training job reads run_gpu_node.sh
# and run_in_container.sh for its whole life, and replacing them under it (git pull) breaks it
# with "Stale file handle". TODO: fold back into run_gpu_node.sh at a moment no job is running.
set -uo pipefail
source "${LUX_DIR:?}/lux_env.sh"
: "${JOB_DIR:?}" "${SANDBOX_HOST:?}" "${HEAD_HOST:?}" "${GPU_NODE_COUNT:?}" "${ALL_HOSTS:?}"

HOST="$(hostname -s)"
NL="/mnt/bb/${SLURM_JOB_ID}"
INST="run-${SLURM_JOB_ID}"
LOG="${JOB_DIR}/node-${HOST}.log"
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] [${HOST}] $*"; }
mkdir -p "${NL}"/{overlay,tmp,triton,cache}

NODE_IP="$(ip -o -4 route get 1 | grep -oP '(?<=src )\S+')"
WANDB_MODE="$(source "${JOB_DIR}/spec.env"; echo "${WANDB_MODE:-online}")"
WANDB_ENTITY="$(source "${JOB_DIR}/spec.env"; echo "${WANDB_ENTITY:-${LUX_WANDB_ENTITY}}")"
# Requests ramp up (miles MILES_ROLLOUT_RAMP_S): thousands of requests at once onto idle engines
# produced degenerate HumanEval+ generations.
RAMP_S="$(source "${JOB_DIR}/spec.env"; echo "${EVAL_RAMP_S:-900}")"

cat > "${NL}/container.env" <<EOF
HOME=/root
USER_HOME=${LUX_HOME}
INSTELLA_USER=${LUX_USER}
HF_HOME=${LUX_ROOT}/hf
TRITON_CACHE_DIR=${NL}/triton
XDG_CACHE_HOME=${NL}/cache
TMPDIR=${NL}/tmp
WANDB_MODE=${WANDB_MODE}
WANDB_ENTITY=${WANDB_ENTITY}
MILES_ROLLOUT_RAMP_S=${RAMP_S}
WANDB_HTTPS_PROXY=${LUX_PROXY}
WANDB_HTTP_PROXY=${LUX_PROXY}
LUX_PROXY=${LUX_PROXY}
SLURM_JOB_ID=${SLURM_JOB_ID}
JOB_DIR=${JOB_DIR}
SANDBOX_HOST=${SANDBOX_HOST}
HEAD_HOST=${HEAD_HOST}
GPU_NODE_COUNT=${GPU_NODE_COUNT}
NODE_RANK=${SLURM_PROCID}
NODE_IP=${NODE_IP}
EOF

export APPTAINER_TMPDIR="${NL}/tmp"
say "starting container (rank ${SLURM_PROCID}, ip ${NODE_IP})"
apptainer instance start --cleanenv --fakeroot --overlay "${NL}/overlay" --no-home \
   --bind /lustre/orion --bind "${LUX_HOME}" --bind "${NL}" --env-file "${NL}/container.env" \
   "${SIF}" "${INST}" || { say "FATAL: instance start failed"; touch "${JOB_DIR}/failed"; exit 1; }
apptainer exec --cleanenv --env-file "${NL}/container.env" "instance://${INST}" \
   bash "${LUX_DIR}/eval_in_container.sh"
rc=$?
say "container session ended (rc=${rc})"
apptainer instance stop "${INST}" >/dev/null 2>&1
exit "${rc}"
