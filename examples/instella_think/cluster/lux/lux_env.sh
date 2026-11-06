# Lux (OLCF) settings for the cluster/lux/ tools. Sourced on the login node and by the batch
# scripts; every value can be overridden from the environment.
#
# Lux has no Docker and no Spur: the miles image runs under Apptainer from a SIF on Lustre,
# with a writable overlay on the job's node-local NVMe (/mnt/bb/<jobid>, gone at job end).

LUX_ACCOUNT="${LUX_ACCOUNT:-ven136}"
LUX_USER="${LUX_USER:-$(id -un)}"
LUX_HOME="${LUX_HOME:-$(getent passwd "${LUX_USER}" | cut -d: -f6)}"
# Big files (SIF, checkpoints, data, logs) live on Orion Lustre, not in the 50 GB NFS home.
LUX_ROOT="${LUX_ROOT:-/lustre/orion/${LUX_ACCOUNT}/scratch/${LUX_USER}/instella}"
IMAGE="${IMAGE:-rocm/sgl-dev:miles-rocm720-mi35x-20260727}"
SIF="${SIF:-${LUX_ROOT}/images/${IMAGE##*:}.sif}"
# Compute nodes reach the internet only through this proxy, and Slurm jobs inherit it. The
# containers start without it (--cleanenv), so HTTP between nodes (sandbox, Ray) goes direct;
# commands that download pass --proxy to on_node.sh.
LUX_PROXY="${LUX_PROXY:-http://proxy.ccs.ornl.gov:3128}"
# Login node <-> node_agent.sh command drop, one directory per job and node.
MAILBOX_ROOT="${MAILBOX_ROOT:-${LUX_HOME}/.miles/lux-mailbox}"
# Host of the code-exec sandbox job (sandbox.sbatch writes it; ~/.instella-env.sh reads it).
CODE_EXEC_HOST_FILE="${CODE_EXEC_HOST_FILE:-${LUX_HOME}/.miles/code-exec-host}"

# W&B team that run.sbatch runs log to (WANDB_ENTITY in a run spec overrides it).
LUX_WANDB_ENTITY="${LUX_WANDB_ENTITY:-genaiamd}"

export LUX_WANDB_ENTITY LUX_ACCOUNT LUX_USER LUX_HOME LUX_ROOT IMAGE SIF LUX_PROXY MAILBOX_ROOT CODE_EXEC_HOST_FILE
