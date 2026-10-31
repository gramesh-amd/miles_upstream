#!/usr/bin/env python3
"""Backport upstream Megatron f4502eb1c so optimizer state survives a checkpoint round-trip.

Run INSIDE the container (writes into MEGATRON_PATH), once per container:

    MEGATRON_PATH=/root/Megatron-LM python3 examples/instella_think/patch_megatron_optim_step.py

Needed only by runs that use --optimizer-cpu-offload together with
--use-precision-aware-optimizer (the Instella recipes here use neither). Without this patch, resuming from a
mid-training checkpoint with optimizer state dies in
megatron/core/dist_checkpointing/serialization.py:126 with

  ValueError: Cannot merge two lists with different lengths (1 and 116, encountered at
  level ('optimizer', 0, 'param_state', 0, (torch.bfloat16, torch.float32), 0))

Cause: --optimizer-cpu-offload makes the CPU sub-optimizer torch.optim.AdamW, which keeps a
per-parameter `step` tensor. (TE's FusedAdam does not -- its step lives in param_groups,
which is why plain --use-precision-aware-optimizer is unaffected.) The dp_reshardable save
path did `if key == 'step': continue`, so `step` was never wrapped as a ShardedTensor and
stayed a plain tensor. Plain tensors are not ShardedBase, so extract_sharded_base routes them
into the COMMON state dict -- and common.pt is written by global rank 0 ONLY. The saved list
therefore has rank 0's bucket-entry count (1) while every other rank expects its own
(48/116/125/123/113/47). dp_reshardable was the only sharding format leaking `step`;
fully_reshardable and fs_model_space already skip it on both save and load.

Upstream fixed this in f4502eb1c, "Handle `step` key correctly in checkpoint save with
--optimizer-cpu-offload" (PR #2874), which is NOT an ancestor of this fork's miles-main HEAD.
Wrapping `step` in LocalNonpersistentObject makes it rank-local, so it is dropped at save and
param_state disappears from common.pt entirely. The authoritative scalar step is restored from
param_groups (distrib_optimizer.py:864-873).

Do NOT instead "fix" this by padding/truncating lists in dist_checkpointing merge(). That was
tried and it loads cleanly but then dies on the first optimizer step with "RuntimeError: a
Tensor with 32 elements cannot be converted to Scalar" in _fused_adam, because padding entries
build `step` via torch.empty(gap), i.e. uninitialized garbage of the wrong shape.

This fixes FUTURE saves. Checkpoints already written without it also need
fix_ckpt_common_optimizer_state.py. Idempotent; keeps a .bak-step copy. Megatron-LM lives in
the container image rather than on shared storage, so re-apply after any container recreate.
"""
import os
import pathlib
import shutil
import sys

MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
TARGET = pathlib.Path(MEGATRON_PATH) / "megatron/core/optimizer/distrib_optimizer.py"
MARKER = "handled separately via param_groups"

OLD = """                            if key == 'step':
                                continue
"""

NEW = """                            if key == 'step':
                                # The optimizer state of STEP is a 0-dim tensor and is
                                # handled separately via param_groups, not as part of the
                                # gradient buffer.
                                tensors[key] = LocalNonpersistentObject(tensors[key])
                                continue
"""


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (is MEGATRON_PATH correct?)", file=sys.stderr)
        return 1

    src = TARGET.read_text()
    if MARKER in src:
        print(f"  v optimizer step patch already present: {TARGET}")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"ERROR: expected exactly 1 patch anchor, found {n}", file=sys.stderr)
        return 2
    if "LocalNonpersistentObject" not in src:
        print("ERROR: LocalNonpersistentObject not imported in target", file=sys.stderr)
        return 3

    shutil.copy2(TARGET, TARGET.with_suffix(TARGET.suffix + ".bak-step"))
    TARGET.write_text(src.replace(OLD, NEW, 1))

    import ast

    ast.parse(TARGET.read_text())
    print(f"  v wrote optimizer step patch: {TARGET}")
    print("Megatron optimizer-step checkpoint patch applied OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
