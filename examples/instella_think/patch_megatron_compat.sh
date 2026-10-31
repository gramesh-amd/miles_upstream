#!/bin/bash
# Bridge the gap between latest miles and this (pre-mcore-0.14) FarSkip Megatron pin.
# Latest miles imports a few newer-Megatron symbols at module load that this pin
# lacks. These are only *used* on code paths we don't take (e.g. Muon optimizer),
# so a stub that satisfies the import is enough. Run INSIDE the container (writes
# into MEGATRON_PATH; needs the same write perms as the pip install -e). Idempotent.
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/paths.sh"
OPT_DIR="${MEGATRON_PATH}/megatron/core/optimizer"
[ -d "${OPT_DIR}" ] || { echo "ERROR: ${OPT_DIR} not found (is MEGATRON_PATH correct?)" >&2; exit 1; }

# --- shim: megatron.core.optimizer.muon.get_megatron_muon_optimizer -----------
MUON="${OPT_DIR}/muon.py"
if [ -f "${MUON}" ] && grep -q "get_megatron_muon_optimizer" "${MUON}"; then
    echo "  v muon shim already present: ${MUON}"
else
    cat > "${MUON}" <<'PY'
# Compatibility shim for latest miles on this (pre-mcore-0.14) FarSkip Megatron.
# miles/backends/megatron_utils/model.py imports get_megatron_muon_optimizer at
# module load; this Megatron pin predates the Muon optimizer. Muon is only invoked
# when --optimizer contains "muon" (we use adam), so failing loudly if actually
# called is enough while keeping the import resolvable.


def get_megatron_muon_optimizer(*args, **kwargs):
    raise NotImplementedError(
        "get_megatron_muon_optimizer is unavailable in this Megatron pin "
        "(FarSkip base, pre-mcore-0.14). Use --optimizer adam, or upgrade Megatron."
    )
PY
    python3 -c "import ast,sys; ast.parse(open('${MUON}').read())"
    echo "  v wrote muon shim: ${MUON}"
fi

echo "Megatron compat shims applied OK."
