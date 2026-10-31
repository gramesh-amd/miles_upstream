#!/bin/bash
# Assemble a FarSkip-capable Megatron, faithfully reproducing the CURRENT known-good
# recipe: Primus-Instella-Internal/rl/setup/primus_instella_patch.sh (2026-07-23).
#
#   1. git submodule update --init --force training/third_party/Megatron-LM
#      -> checks out the SUBMODULE PIN (847781764...), the commit the FarSkip patch
#         tree was authored against. Used AS-IS, no external checkout.
#   2. overlay the FarSkip patch tree  (training/patches/megatron-lm/.)
#   3. pip install -e <submodule>      (makes it THE container megatron-core)
#   4. smoke-import SimpleFarSkipTransformerLayer
#
# NOTE: an older script (rl/patches/setup_megatron_farskip_sglang59.sh) checked out
# 43a09cbfe instead; that was superseded by the pin-as-is approach above. Do NOT use it.
#
# Run INSIDE the miles container (needs git + build tools for pip install -e).
set -uo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/paths.sh"
PRIMUS_REPO="${PRIMUS_REPO:-${PRIMUS_ROOT}}"
PRIMUS_TRAINING="${PRIMUS_REPO}/training"
MEGATRON_SUBMODULE="${MEGATRON_SUBMODULE:-${PRIMUS_TRAINING}/third_party/Megatron-LM}"
MEGATRON_PATCHES="${MEGATRON_PATCHES:-${PRIMUS_TRAINING}/patches/megatron-lm}"
INSTALL="${INSTALL:-1}"          # pip install -e <submodule> (1=yes)
# Fallback when Primus is a file-sync deploy (no .git) and the submodule is empty:
# clone the base directly at the recorded submodule pin (from .gitmodules gitlink).
MEGATRON_URL="${MEGATRON_URL:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_PIN="${MEGATRON_PIN:-847781764fe468c90caec16309deded245c1022c}"

[ -d "${MEGATRON_PATCHES}/megatron" ] || { echo "ERROR: FarSkip patch tree missing: ${MEGATRON_PATCHES}" >&2; exit 1; }

echo "=== FarSkip Megatron assembly (pin-as-is) ==="
echo "  submodule: ${MEGATRON_SUBMODULE}"
echo "  overlay:   ${MEGATRON_PATCHES}"
echo

# --- Step 1: get the Megatron base at its recorded pin (847781764...) ---
if [ -f "${MEGATRON_SUBMODULE}/megatron/core/__init__.py" ]; then
   echo "[1/4] Megatron base already present at ${MEGATRON_SUBMODULE}"
   git -C "${MEGATRON_SUBMODULE}" rev-parse --short HEAD 2>/dev/null | sed 's/^/      HEAD /' || true
elif git -C "${PRIMUS_REPO}" rev-parse --git-dir >/dev/null 2>&1; then
   echo "[1/4] git submodule update --init --force ${MEGATRON_SUBMODULE}"
   git -C "${PRIMUS_REPO}" submodule update --init --force "${MEGATRON_SUBMODULE}" \
      || { echo "ERROR: submodule init failed (offline?)." >&2; exit 1; }
   echo "      submodule at pin $(git -C "${MEGATRON_SUBMODULE}" rev-parse --short HEAD)"
else
   # File-sync deploy (no .git) + empty submodule: clone the base directly at the pin.
   echo "[1/4] ${PRIMUS_REPO} has no .git; cloning ${MEGATRON_URL} @ ${MEGATRON_PIN:0:12}"
   rm -rf "${MEGATRON_SUBMODULE}"
   mkdir -p "$(dirname "${MEGATRON_SUBMODULE}")"
   git clone "${MEGATRON_URL}" "${MEGATRON_SUBMODULE}" \
      || { echo "ERROR: clone failed (offline? no access to NVIDIA/Megatron-LM?)." >&2; exit 1; }
   git -C "${MEGATRON_SUBMODULE}" checkout "${MEGATRON_PIN}" \
      || { echo "ERROR: pin ${MEGATRON_PIN} not found in ${MEGATRON_URL}." >&2; exit 1; }
   echo "      cloned base at $(git -C "${MEGATRON_SUBMODULE}" rev-parse --short HEAD)"
fi

# --- Step 2: overlay FarSkip patch tree ---
echo "[2/4] overlaying FarSkip patches (${MEGATRON_PATCHES}/. -> ${MEGATRON_SUBMODULE}/)"
cp -a "${MEGATRON_PATCHES}/." "${MEGATRON_SUBMODULE}/"
grep -q "class SimpleFarSkipTransformerLayer" \
   "${MEGATRON_SUBMODULE}/megatron/core/transformer/transformer_layer.py" \
   || { echo "ERROR: overlay did not land (no SimpleFarSkipTransformerLayer)." >&2; exit 1; }
echo "      overlay OK (SimpleFarSkipTransformerLayer present)"

# --- Step 3: pip install -e (make it the active megatron-core) ---
if [ "${INSTALL}" = "1" ]; then
   echo "[3/4] pip install -e ${MEGATRON_SUBMODULE}"
   pip install -e "${MEGATRON_SUBMODULE}" 2>&1 | tail -3
   echo "      megatron-core: $(pip show megatron-core 2>/dev/null | grep -i version)"
else
   echo "[3/4] SKIP pip install -e (INSTALL=0). Ensure PYTHONPATH puts ${MEGATRON_SUBMODULE} FIRST."
fi

# --- Step 4: smoke-check FarSkip layers import ---
echo "[4/4] smoke-check FarSkip layers"
MEGATRON_SUBMODULE="${MEGATRON_SUBMODULE}" python3 - <<'PY'
import os, sys
sub = os.environ["MEGATRON_SUBMODULE"]
sys.path.insert(0, sub)   # force submodule ahead of any other installed megatron
import importlib
m = importlib.import_module("megatron.core.transformer.transformer_layer")
print(f"  megatron.core loaded from: {m.__file__}")
for cls in ("SimpleFarSkipTransformerLayer", "OverlappedFarSkipTransformerLayer"):
    assert hasattr(m, cls), f"missing {cls} in {m.__file__}"
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
get_gpt_layer_local_spec(num_experts=64, multi_latent_attention=True, use_simple_farskip_layer=True)
print("OK: FarSkip Megatron imports and builds a simple FarSkip MLA MoE spec")
PY

echo
echo "Set MEGATRON_PATH=${MEGATRON_SUBMODULE} when running examples/instella_think/run_async.sh"
