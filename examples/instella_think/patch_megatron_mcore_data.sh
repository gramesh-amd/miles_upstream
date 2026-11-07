#!/bin/bash
# Make Megatron's checkpoint load tolerate a .metadata without `mcore_data`.
# Idempotent. Run in EVERY node's container, like the other megatron patches.
#
# WHY. Resuming from any checkpoint this run produced dies with:
#   AttributeError: 'Metadata' object has no attribute 'mcore_data'
#   .../dist_checkpointing/strategies/torch.py:874 in get_reformulation_metadata
#
# Every iter_* we saved (199, 339, 359 -- including ones that never left local
# disk) has a 4,371,823-byte .metadata holding a plain torch DCP `Metadata`,
# which carries planner_data/state_dict_metadata/storage_data but no
# `mcore_data`. So this is not corruption and not the bucket round-trip; the
# save simply never wrote Megatron's extended field. The run itself never hit
# this because it started from REF_LOAD (a weights-only `release` checkpoint)
# and ran straight through -- resume was never exercised.
#
# The fix is to treat a MISSING attribute the same as a missing KEY, which the
# function already handles: for 1-D flattened tensors it warns "Legacy
# checkpoint format detected" and skips reformulation. Only genuinely N-D
# flattened tensors still raise, and they now raise the intended
# CheckpointingException instead of an AttributeError. The same file already
# reads this field defensively at line ~980 (`getattr(metadata, 'mcore_data',
# {})`), so this only makes the two load paths agree.
#
# If this still fails with CheckpointingException for an N-D flattened tensor,
# the fallback is to resume with --no-load-optim (losing Adam state).
set -euo pipefail
[ -n "${MEGATRON_PATH:-}" ] || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/paths.sh"

MG="${MEGATRON_PATH:-/root/megatron_farskip}"
F="${MG}/megatron/core/dist_checkpointing/strategies/torch.py"

[ -f "$F" ] || { echo "ERROR: $F not found" >&2; exit 1; }

if grep -q "getattr(ckpt_metadata, 'mcore_data', {})" "$F"; then
    echo "already patched: $F"
    exit 0
fi

[ -f "${F}.mcore_bak" ] || cp "$F" "${F}.mcore_bak"

python3 - "$F" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

# The lookup that raises AttributeError when the field is absent entirely.
old_lookup = """            ckpt_global_shape = ckpt_metadata.mcore_data[sh_ten.key][
                'nd_reformulated_orig_global_shape'
            ]"""
new_lookup = """            ckpt_global_shape = getattr(ckpt_metadata, 'mcore_data', {})[sh_ten.key][
                'nd_reformulated_orig_global_shape'
            ]"""

# The diagnostic also dereferences the field, so it would raise while building
# the message for the N-D case.
old_msg = """                f'in checkpoint metadata: {ckpt_metadata.mcore_data}'"""
new_msg = """                f'in checkpoint metadata: {getattr(ckpt_metadata, "mcore_data", None)}'"""

for old, new, label in ((old_lookup, new_lookup, "lookup"), (old_msg, new_msg, "message")):
    if old not in src:
        sys.exit(f"ERROR: {label} anchor not found; refusing to patch blindly")
    src = src.replace(old, new, 1)

open(path, "w").write(src)
print("patched lookup + message")
PYEOF

python3 -c "import ast,sys;ast.parse(open('$F').read())" || {
    echo "ERROR: patch produced invalid python; restoring" >&2
    cp "${F}.mcore_bak" "$F"; exit 1; }

echo "OK: $F"
