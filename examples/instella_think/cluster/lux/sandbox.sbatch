#!/bin/bash
#SBATCH -J instella-sandbox
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p batch
# The code-exec sandboxes on a node of their own (run_sandbox_node.sh), for runs driven by hand
# with node.sbatch + on_node.sh; run.sbatch starts its own. The Lux counterpart of
# `cluster/code_exec.sh start` + `start bench`, which ssh to a fixed CPU host. Writes this node's
# name to $CODE_EXEC_HOST_FILE, which ~/.instella-env.sh reads. Submit from the miles checkout:
#
#   sbatch -A ven136 -t 04:00:00 -o ~/logs/%x-%j.out \
#     --export=ALL,LUX_DIR=$PWD/examples/instella_think/cluster/lux examples/instella_think/cluster/lux/sandbox.sbatch
exec bash "${LUX_DIR:?submit with --export=ALL,LUX_DIR=<this dir>}/run_sandbox_node.sh"
