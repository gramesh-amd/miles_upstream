#!/bin/bash
#SBATCH -J instella-gpu
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH -p batch
# Hold N GPU nodes with the miles container up on each (node_agent.sh), the Lux equivalent of
# `sbatch ... --wrap "sleep infinity"` + cluster/bringup.sh. Submit from the miles checkout:
#
#   sbatch -A ven136 -N 1 -t 04:00:00 -o ~/logs/%x-%j.out \
#     --export=ALL,LUX_DIR=$PWD/examples/instella_think/cluster/lux examples/instella_think/cluster/lux/node.sbatch
#
# Then drive the nodes with cluster/lux/on_node.sh. Cancel with scancel when done: the nodes
# are charged for the whole time limit whether busy or not.
set -uo pipefail
: "${LUX_DIR:?submit with --export=ALL,LUX_DIR=<this dir>}"
echo "job ${SLURM_JOB_ID}: nodes ${SLURM_JOB_NODELIST}"
# --cpu-bind=none: the default binding gives each task a single core on Lux.
srun --ntasks-per-node=1 --cpu-bind=none --export=ALL bash "${LUX_DIR}/node_agent.sh"
