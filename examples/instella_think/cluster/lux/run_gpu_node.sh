#!/bin/bash
# One GPU node of a run.sbatch job (one task per node). Starts the miles container (fakeroot, with
# a writable overlay on node-local NVMe, as node_agent.sh does) and runs run_in_container.sh in
# it in the foreground until the run ends. Everything the node starts (Ray, the run) lives in that
# one `apptainer exec`: under fakeroot, a process outliving the exec that started it hangs.
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
# Ray workers take their environment from the container, not from the shell that launches the
# run, so spec settings they need must be applied here.
WANDB_MODE="$(source "${JOB_DIR}/spec.env"; echo "${WANDB_MODE:-online}")"
# miles passes no W&B entity, so without this runs land in the key owner's default team.
WANDB_ENTITY="$(source "${JOB_DIR}/spec.env"; echo "${WANDB_ENTITY:-${LUX_WANDB_ENTITY}}")"
# Eval-only jobs ramp their requests up (miles MILES_ROLLOUT_RAMP_S): thousands of requests at
# once onto idle engines produced degenerate HumanEval+ generations.
RAMP_S="$(source "${JOB_DIR}/spec.env"; [ "${EVAL_ONLY:-0}" = 1 ] && echo "${EVAL_RAMP_S:-900}" || echo 0)"

# --cleanenv drops Slurm's GPU masks (all 8 GPUs stay visible) and the ORNL proxy, so traffic
# between nodes (sandbox, Ray, router, RCCL) goes direct; W&B alone gets the proxy, through its
# own settings. paths.sh takes the user from the checkout's owner, which reads as root under
# fakeroot, hence USER_HOME/INSTELLA_USER.
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
   bash "${LUX_DIR}/run_in_container.sh"
rc=$?
say "container session ended (rc=${rc})"
apptainer instance stop "${INST}" >/dev/null 2>&1
exit "${rc}"
