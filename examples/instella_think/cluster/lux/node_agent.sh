#!/bin/bash
# Per-node container + command agent for Lux; the Lux stand-in for Spur's long-lived Docker
# container. node.sbatch runs one copy per node in a single srun step that lasts the job.
#
# Starts an Apptainer instance of the miles SIF (fakeroot, so it runs as root like the Docker
# container, with a writable overlay on node-local NVMe), then executes the command files
# on_node.sh drops into $MAILBOX_ROOT/<jobid>/<node>/. Each command runs inside the instance
# as a child of this step, so what it leaves in the background (Ray, a launched run) keeps
# running after on_node.sh returns, until the job ends. An srun --overlap step could not do
# that: Slurm kills a step's processes when the step exits.
set -uo pipefail
source "${LUX_DIR:?}/lux_env.sh"

HOST="$(hostname -s)"
NL="/mnt/bb/${SLURM_JOB_ID}"
MB="${MAILBOX_ROOT}/${SLURM_JOB_ID}/${HOST}"
INST="miles-${SLURM_JOB_ID}"
say() { echo "[$(date '+%F %T')] [${HOST}] $*"; }

mkdir -p "${MB}" "${NL}"/{overlay,tmp,triton,cache}
# Instance environment. --cleanenv drops Slurm's GPU masks (all 8 GPUs stay visible) and the
# ORNL proxy (so HTTP to the sandbox and Ray goes direct). paths.sh takes the user from the
# checkout's owner, which reads as root under fakeroot, hence USER_HOME/INSTELLA_USER.
cat > "${NL}/container.env" <<EOF
# HOME stays the image's /root (writable through the overlay), as in the Docker container;
# --no-home keeps the image's /root contents (e.g. /root/Megatron-LM) visible.
HOME=/root
USER_HOME=${LUX_HOME}
INSTELLA_USER=${LUX_USER}
LUX_PROXY=${LUX_PROXY}
HF_HOME=${LUX_ROOT}/hf
TRITON_CACHE_DIR=${NL}/triton
XDG_CACHE_HOME=${NL}/cache
TMPDIR=${NL}/tmp
WANDB_MODE=${WANDB_MODE:-offline}
SLURM_JOB_ID=${SLURM_JOB_ID}
EOF

export APPTAINER_TMPDIR="${NL}/tmp"
say "starting instance ${INST} from ${SIF}"
apptainer instance start --cleanenv --fakeroot \
   --overlay "${NL}/overlay" --no-home \
   --bind /lustre/orion --bind "${LUX_HOME}" --bind "${NL}" \
   --env-file "${NL}/container.env" \
   "${SIF}" "${INST}" 2>&1 || { say "FATAL: instance start failed"; exit 1; }
echo "${HOST}" > "${MB}/ready"
say "ready; mailbox ${MB}"

trap 'say "stopping"; apptainer instance stop "${INST}" >/dev/null 2>&1; exit 0' TERM INT
while :; do
   for f in "${MB}"/*.cmd; do
      [ -e "${f}" ] || continue
      base="${f%.cmd}"
      mv "${f}" "${base}.run" || continue
      say "run $(basename "${base}")"
      (
         apptainer exec --cleanenv --env-file "${NL}/container.env" --pwd "${LUX_HOME}" \
            "instance://${INST}" bash "${base}.run" > "${base}.log" 2>&1 < /dev/null
         echo $? > "${base}.rc.tmp" && mv "${base}.rc.tmp" "${base}.rc"
      ) &
   done
   sleep 2
done
