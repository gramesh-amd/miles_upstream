#!/bin/bash
# Let a resume load OPTIMIZER state from our checkpoints when the parallel layout is unchanged.
# Idempotent. Run in EVERY node's container, after patch_megatron_mcore_data.sh.
#
# WHY. Our checkpoints' .metadata has no `mcore_data` (see patch_megatron_mcore_data.sh), so
# get_reformulation_metadata cannot read the original N-D shape of the distributed optimizer's
# flattened N-D tensors and resuming with optimizer state dies with
#   CheckpointingException: Cannot find global shape metadata for N-D flattened tensor
#   ShardedTensor(key='optimizer.state.exp_avg.output_layer.weight', ...)
# The only workaround so far was --no-load-optim, which restarts Adam from zero moments.
#
# THE FIX. That metadata is only needed to RESHARD (a different TP/EP/DP layout than the save).
# When the tensor the checkpoint stored has exactly the reformulated shape this run would write
# (nd_flattened_tensor_reformulated_global_shape), the layouts agree, and apply_nd_flattened_
# tensors_reformulation already loads such a tensor as-is ("Same shape - no need to reshard").
# So in that case the missing entry is filled with this run's own global shape; any other shape
# still raises, so a genuine reshard can never load misplaced moments.
set -euo pipefail
[ -n "${MEGATRON_PATH:-}" ] || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/paths.sh"

F="${MEGATRON_PATH}/megatron/core/dist_checkpointing/strategies/torch.py"
[ -f "$F" ] || { echo "ERROR: $F not found" >&2; exit 1; }
MARK="instella: same-layout N-D resume"
if grep -q "${MARK}" "$F"; then
   echo "already patched: $F"
   exit 0
fi

# Several nodes run this at once on shared storage: edit a private copy, then rename it in.
TMP="${F}.nd.$$"
python3 - "$F" "$TMP" "${MARK}" <<'PYEOF'
import sys

src, dst, mark = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()
old = """                continue
            raise CheckpointingException(
                f'Cannot find global shape metadata for N-D flattened tensor {sh_ten} '"""
new = f"""                continue
            # {mark}: without mcore_data, accept a tensor saved in exactly the layout this run
            # loads (no resharding needed); anything else still raises below.
            saved = ckpt_metadata.state_dict_metadata.get(sh_ten.key)
            if saved is not None and tuple(saved.size) == tuple(
                nd_flattened_tensor_reformulated_global_shape(sh_ten)
            ):
                reformulation_metadata[sh_ten.key] = TensorReformulationMetadata(
                    tuple(sh_ten.global_shape), saved.size
                )
                continue
            raise CheckpointingException(
                f'Cannot find global shape metadata for N-D flattened tensor {{sh_ten}} '"""
if text.count(old) != 1:
    sys.exit("ERROR: anchor not found exactly once (run patch_megatron_mcore_data.sh first?); refusing to patch blindly")
open(dst, "w").write(text.replace(old, new, 1))
PYEOF
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$TMP" || { echo "ERROR: patch produced invalid python" >&2; rm -f "$TMP"; exit 1; }
grep -q "${MARK}" "$F" && { rm -f "$TMP"; echo "already patched: $F"; exit 0; }
mv -f "$TMP" "$F"
echo "OK: $F"
