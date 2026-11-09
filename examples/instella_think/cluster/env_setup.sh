#!/bin/bash
# In-container environment setup for the Instella recipes. Run once per node per container
# (containers are recreated per job), via on_node.sh:
#
#   cluster/on_node.sh <JOBID> <NODE> cluster/env_setup.sh          # code RL / math
#   cluster/on_node.sh <JOBID> <NODE> cluster/env_setup.sh --if     # also the IF reward deps
#
# Idempotent: every step it calls is idempotent, so re-running is safe.
set -uo pipefail

INSTELLA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${INSTELLA_DIR}/../.." &>/dev/null && pwd)"
HOST="$(hostname -s)"
WITH_IF=0
[ "${1:-}" = "--if" ] && WITH_IF=1
rc=0

say() { echo "[${HOST}] $*"; }
step() {
   local name="$1"; shift
   say "=== ${name} ==="
   if "$@"; then
      say "--- ${name}: OK"
   else
      say "--- ${name}: FAILED (rc=$?)"
      rc=1
   fi
}

# shellcheck disable=SC1091
source /opt/venv/bin/activate 2>/dev/null || say "WARN: /opt/venv/bin/activate not found"
cd "${REPO_DIR}" || { say "FATAL: ${REPO_DIR} missing"; exit 1; }

say "python=$(python3 -V 2>&1)  torch=$(python3 -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"

# 1. sglang FarSkip overlay (sglang in the container).
step "setup_farskip.sh" bash "${INSTELLA_DIR}/setup_farskip.sh"
# 2. Megatron import shims (writes into MEGATRON_PATH on NFS).
step "patch_megatron_compat.sh" bash "${INSTELLA_DIR}/patch_megatron_compat.sh"
# 2b. Routing-replay hooks (USE_ROUTING_REPLAY=1, the run_code.sh / run_if.sh default).
step "patch_megatron_r3.sh" bash "${INSTELLA_DIR}/patch_megatron_r3.sh"
# 2c. Resume from our own checkpoints (their .metadata has no mcore_data).
step "patch_megatron_mcore_data.sh" bash "${INSTELLA_DIR}/patch_megatron_mcore_data.sh"
# 2d. ...including the optimizer state, when the parallel layout is unchanged.
step "patch_megatron_nd_same_layout.sh" bash "${INSTELLA_DIR}/patch_megatron_nd_same_layout.sh"
# 3. mbridge gate_proj mapping (patches the container's installed mbridge).
step "patch_mbridge_gate_proj.sh" bash "${INSTELLA_DIR}/patch_mbridge_gate_proj.sh"
# 4. IF checker deps + nltk corpus (runs its own --verify at the end).
[ "${WITH_IF}" = "1" ] && step "setup_if_deps.sh" bash "${INSTELLA_DIR}/setup_if_deps.sh"

say "=== summary for ${HOST}: $([ "${rc}" -eq 0 ] && echo ALL OK || echo SOME STEPS FAILED) ==="
exit "${rc}"
