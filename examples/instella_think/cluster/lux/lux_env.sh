# Lux (OLCF) settings for the cluster/lux/ tools. Sourced on the login node and by the batch
# scripts; every value can be overridden from the environment.
#
# Lux has no Docker and no Spur: the miles image runs under Apptainer from a SIF on Lustre,
# with a writable overlay on the job's node-local NVMe (/mnt/bb/<jobid>, gone at job end).

# Site-specific: your OLCF project id (sbatch -A). Inside a job Slurm provides it.
LUX_ACCOUNT="${LUX_ACCOUNT:-${SLURM_JOB_ACCOUNT:-}}"
LUX_USER="${LUX_USER:-$(id -un)}"
LUX_HOME="${LUX_HOME:-$(getent passwd "${LUX_USER}" | cut -d: -f6)}"
# Big files (SIF, checkpoints, data, logs) live on Orion Lustre, not in the 50 GB NFS home.
# LUX_FS (site-specific) is that filesystem; every container binds it.
LUX_FS="${LUX_FS:-/lustre/orion}"
LUX_ROOT="${LUX_ROOT:-${LUX_FS}/${LUX_ACCOUNT:?set LUX_ACCOUNT (your OLCF project id) or LUX_ROOT}/scratch/${LUX_USER}/instella}"
IMAGE="${IMAGE:-rocm/sgl-dev:miles-rocm720-mi35x-20260727}"
SIF="${SIF:-${LUX_ROOT}/images/${IMAGE##*:}.sif}"
# Site-specific: compute nodes reach the internet only through this proxy, and Slurm jobs
# inherit it. The containers start without it (--cleanenv), so HTTP between nodes (sandbox,
# Ray) goes direct; commands that download pass --proxy to on_node.sh.
LUX_PROXY="${LUX_PROXY:-http://proxy.ccs.ornl.gov:3128}"
# Login node <-> node_agent.sh command drop, one directory per job and node.
MAILBOX_ROOT="${MAILBOX_ROOT:-${LUX_HOME}/.miles/lux-mailbox}"
# Host of the code-exec sandbox job (sandbox.sbatch writes it; ~/.instella-env.sh reads it).
CODE_EXEC_HOST_FILE="${CODE_EXEC_HOST_FILE:-${LUX_HOME}/.miles/code-exec-host}"

# W&B entity that run.sbatch runs log to (WANDB_ENTITY in a run spec overrides it); unset logs to
# the W&B user's default entity.
LUX_WANDB_ENTITY="${LUX_WANDB_ENTITY:-}"

export LUX_WANDB_ENTITY LUX_ACCOUNT LUX_USER LUX_HOME LUX_FS LUX_ROOT IMAGE SIF LUX_PROXY MAILBOX_ROOT CODE_EXEC_HOST_FILE
