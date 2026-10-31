#!/usr/bin/env python3
"""Repair a checkpoint written before patch_megatron_optim_step.py, so it can resume.

Run INSIDE the container with the venv active:

    python3 examples/instella_think/fix_ckpt_common_optimizer_state.py <iter_dir> [--apply]

Without --apply it only reports what it would change. Example:

    python3 examples/instella_think/fix_ckpt_common_optimizer_state.py \\
        /home/goramesh/checkpoints/Qwen3.6-35B-A3B_miles_async_s4/iter_0000039 --apply

Checkpoints saved with --optimizer-cpu-offload before the patch carry a bogus 'param_state'
subtree in common.pt: the CPU sub-optimizer (torch.optim.AdamW) keeps a per-parameter `step`,
the dp_reshardable save path left it as a plain tensor, and plain tensors land in the COMMON
state dict -- which only global rank 0 writes. The result is a list sized for rank 0 (1 entry)
where every other rank expects its own bucket-entry count, so resuming with optimizer state
dies in megatron/core/dist_checkpointing/serialization.py:126 with
"Cannot merge two lists with different lengths (1 and 116, ...)".

Dropping the whole param_state subtree reproduces exactly what a patched save would have
written: merge() then finds no 'param_state' key in the common dict and simply inserts the
model-side subtree, so no length comparison ever happens.

Nothing of value is lost. exp_avg / exp_avg_sq / param live in the .distcp shards (verified at
100% coverage of every global tensor), and the authoritative scalar step lives in the sharded
param_groups. Only the redundant rank-0-shaped residue is removed.

Two caveats worth knowing:
  - Deleting only the inner 'step' keys is NOT enough; the lists would still be length 1 vs
    116. The whole param_state subtree has to go.
  - .distcp files and .metadata are never touched, so this is cheap regardless of checkpoint
    size, and the original common.pt is kept as common.pt.bak-preoptimfix.
"""
import argparse
import pathlib
import shutil
import sys

import torch


def describe(param_state, indent="      "):
    """Summarize a param_state subtree shallowly enough to eyeball its shape."""
    out = []
    if isinstance(param_state, dict):
        for k, v in param_state.items():
            if isinstance(v, dict):
                for dk, dv in v.items():
                    n = len(dv) if hasattr(dv, "__len__") else "?"
                    out.append(f"{indent}[{k!r}][{dk!r}] -> {type(dv).__name__} len={n}")
            else:
                n = len(v) if hasattr(v, "__len__") else "?"
                out.append(f"{indent}[{k!r}] -> {type(v).__name__} len={n}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("iter_dir", help="checkpoint iteration dir, e.g. .../iter_0000039")
    ap.add_argument("--apply", action="store_true", help="actually rewrite common.pt")
    args = ap.parse_args()

    common = pathlib.Path(args.iter_dir) / "common.pt"
    if not common.exists():
        print(f"ERROR: no common.pt at {common}", file=sys.stderr)
        return 1

    sd = torch.load(common, map_location="cpu", weights_only=False)
    opt = sd.get("optimizer")
    if not isinstance(opt, dict):
        print(f"ERROR: unexpected optimizer type {type(opt).__name__}", file=sys.stderr)
        return 2

    found = []
    for k, v in opt.items():
        if isinstance(v, dict) and "param_state" in v:
            found.append(k)
            print(f"  optimizer[{k!r}] keys={list(v.keys())}")
            for line in describe(v["param_state"]):
                print(line)

    if not found:
        print("  v already clean: no param_state in common.pt")
        return 0

    if not args.apply:
        print(f"DRY RUN: would drop param_state from optimizer{found}; re-run with --apply")
        return 0

    backup = common.with_suffix(".pt.bak-preoptimfix")
    if backup.exists():
        print(f"  backup already exists, keeping it -> {backup}")
    else:
        shutil.copy2(common, backup)
        print(f"  backup -> {backup}")

    for k in found:
        opt[k].pop("param_state", None)
    torch.save(sd, common)

    # Re-read to prove the rewrite is loadable and actually clean.
    sd2 = torch.load(common, map_location="cpu", weights_only=False)
    left = [k for k, v in sd2["optimizer"].items() if isinstance(v, dict) and "param_state" in v]
    if left:
        print(f"ERROR: param_state still present in {left}", file=sys.stderr)
        return 3
    kept = {k: list(v.keys()) for k, v in sd2["optimizer"].items() if isinstance(v, dict)}
    print(f"  after: optimizer keys per entry = {kept}")
    print(f"  iteration still = {sd2.get('iteration')}")
    print("Checkpoint common.pt repaired OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
