#!/bin/bash
# Apply the verified sglang515 FarSkip overlay onto this container's sglang install.
# Run this ON EVERY RAY NODE (inside the container) before starting the async job,
# and re-run after any sglang reinstall. Idempotent.
#
# Overlays the 16 FarSkip files (instella_moe.py + patched deepseek_v2.py etc.) from
# Primus-Instella-Internal/inference/sglang515/python onto $SGLANG_PYTHON, then
# verifies. Same overlay you verified working with lm_eval.
set -eo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/paths.sh"
SGLANG_PYTHON="${SGLANG_PYTHON:-/sgl-workspace/sglang/python}"

OVERLAY_SRC="${PRIMUS_ROOT}/inference/sglang515/python"
COPIER="${PRIMUS_ROOT}/inference/scripts/update_sglang_workspace_files.sh"
VERIFIER="${PRIMUS_ROOT}/inference/scripts/verify_sglang_overlay_515.py"

[ -d "${OVERLAY_SRC}" ] || { echo "ERROR: overlay src missing: ${OVERLAY_SRC}" >&2; exit 1; }
[ -d "${SGLANG_PYTHON}" ] || { echo "ERROR: SGLANG_PYTHON missing: ${SGLANG_PYTHON}" >&2; exit 1; }

echo "Overlaying FarSkip sglang515 files: ${OVERLAY_SRC} -> ${SGLANG_PYTHON}"
bash "${COPIER}" "${OVERLAY_SRC}" "${SGLANG_PYTHON}"

rm -rf ~/.triton/cache /tmp/triton_cache 2>/dev/null || true

if [ -f "${VERIFIER}" ]; then
   echo "Verifying overlay..."
   # NB: don't let `set -e` abort on a non-zero verifier — we want to inspect which
   # checks failed below (the aiter unshuffle round-trip is allowed to fail).
   VERIFY_OUT="$(python3 "${VERIFIER}" 2>&1)" && VRC=0 || VRC=$?
   printf '%s\n' "${VERIFY_OUT}"
   if [ "${VRC}" -ne 0 ]; then
      # The aiter MoE preshuffle round-trip (unshuffle_weight) is ONLY exercised on
      # the SGLANG_USE_AITER=1 weight-sync path. We default to the triton MoE runner
      # (SGLANG_USE_AITER=0, see run_async.sh), so unshuffle_weight is never called
      # and its check is irrelevant. Treat *only* that one failing as acceptable;
      # any other FAIL still gates. Set ALLOW_NO_AITER_UNSHUFFLE=0 to make it strict.
      FAILS="$(printf '%s\n' "${VERIFY_OUT}" | grep '^FAIL ' || true)"
      UNEXPECTED="$(printf '%s\n' "${FAILS}" | grep -v 'unshuffle(shuffle(x))==x' | grep -v '^[[:space:]]*$' || true)"
      if [ "${ALLOW_NO_AITER_UNSHUFFLE:-1}" = "1" ] && [ -z "${UNEXPECTED}" ]; then
         echo "WARN: only the aiter unshuffle round-trip check failed -> OK for the" >&2
         echo "      triton MoE runner (SGLANG_USE_AITER=0). It is not used unless you" >&2
         echo "      explicitly run SGLANG_USE_AITER=1 (aiter preshuffle weight sync)." >&2
      else
         echo "ERROR: sglang overlay verify failed" >&2; exit 1
      fi
   fi
fi
echo "FarSkip sglang overlay applied OK."

# ---------------------------------------------------------------------------
# torch_memory_saver (ROCm/HIP build) — OPT-IN (BUILD_TMS=1), default OFF.
# The old recipe built this to keep the in_place weight broadcast from hanging,
# but colocate was verified working WITHOUT it in MI35x, so we don't build it by default.
# Enable only if you actually observe the broadcast hanging at step 0.
# ---------------------------------------------------------------------------
if [ "${BUILD_TMS:-0}" = "1" ]; then
   TMS_DIR="${TMS_DIR:-/tmp/torch_memory_saver}"
   TMS_SO="${TMS_DIR}/torch_memory_saver_hook_mode_preload.abi3.so"
   if [ -f "${TMS_SO}" ]; then
      echo "torch_memory_saver preload .so already built: ${TMS_SO}"
   else
      echo "Building torch_memory_saver (rocm branch)..."
      rm -rf "${TMS_DIR}"
      git clone https://github.com/zyzshishui/torch_memory_saver.git "${TMS_DIR}" 2>&1 | tail -1
      ( cd "${TMS_DIR}" \
        && git checkout -b rocm origin/rocm 2>&1 | tail -1 \
        && pip install -e . 2>&1 | tail -3 \
        && python3 setup.py build_ext --inplace 2>&1 | tail -3 )
      [ -f "${TMS_SO}" ] && echo "torch_memory_saver built: ${TMS_SO}" \
         || echo "WARN: torch_memory_saver build did not produce ${TMS_SO}."
   fi
fi

echo
echo "NOTE: the old recipe also patched two sglang model-loader spots for RL weight"
echo "      sync (deepseek_weight_loader.py 'weight_loader safety' + unquant.py aiter"
echo "      MoE shuffle round-trip). Likely NOT needed here (the sglang515 overlay may"
echo "      already cover them). Only revisit if the smoke run dies with"
echo "      \"'Parameter' object has no attribute 'weight_loader'\" or shows corrupted"
echo "      experts after a weight update. (BUILD_TMS=1 to also build torch_memory_saver.)"
