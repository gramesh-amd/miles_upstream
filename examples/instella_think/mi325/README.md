# Instella FarSkip on MI325X (sglang 0.5.21)

Node bringup kit for running `examples/instella_think` on MI325X (gfx942) nodes with the
`rocm/sgl-dev:miles-rocm10-mi30x-20260930` image. The MI355 equivalents live in
`../cluster/`; this kit is separate because the image, the sglang version (0.5.21 instead
of 0.5.15) and the patch set all differ.

## What a node needs

The image ships stock sglang and a Megatron without FarSkip. Five container-local steps
turn it into a node that can train and serve Instella-MoE. `bringup_node.sh` runs them in
this order; all are idempotent.

| Script | Side | What it does |
|---|---|---|
| `apply_farskip_training.sh` | train | Builds `/root/megatron_farskip`: NVIDIA Megatron-LM @ `8477817` from Primus-Instella-Internal plus its FarSkip overlay, backfills modules and symbols miles imports but the old pin lacks, fixes `np.product` for numpy 2, then `pip install --no-deps -e`. Skipped when the tree already exists and resolves, because it `rm -rf`s and would discard the next two patches. |
| `sglang_port/apply_farskip.sh` | rollout | Applies the 17 FarSkip diffs in `sglang_port/farskip_delta/` to `/sgl-workspace/sglang`. Diffs, not whole-file copies, so the image's sglang stays intact everywhere else. `--check` reports what would apply. |
| `patch_triton_mla_lds.sh` | rollout | Adds Instella's `Lq = 512 + 32 = 544` to the triton extend-attention tile whitelist. Without it the first absorbed prefill pads to 1024 and needs 128 KiB of LDS against gfx942's 64 KiB (`OutOfResources: Required: 131072`). |
| `patch_megatron_r3.sh` | train | Ports miles' routing-replay hooks (`router.py`, `moe_utils.py`) into the FarSkip Megatron. Without them `USE_ROUTING_REPLAY=1` dies on an empty `replay_list` (`IndexError`). |
| `patch_megatron_mcore_data.sh` | train | Lets checkpoint load tolerate a DCP `.metadata` without `mcore_data`. Without it, resuming from this run's own checkpoints dies with `AttributeError: 'Metadata' object has no attribute 'mcore_data'`. |

`bringup_verify.sh` then checks every post-condition independently (greps plus runtime
imports, never the scripts' own exit codes), and `bringup_ray_check.py` confirms the node
actually registered with the intended Ray head with the expected IP and GPU count.

The sglang diffs are built against the image's tree (sglang `0.5.21.dev65`, commit
`57e7d0e848` on `sglang-miles`). On a different sglang some hunks will not apply: against
the 0.5.21 PyPI release, 12 of 17 apply and 5 need review. Run
`sglang_port/apply_farskip.sh --check` first on any other image.
`apply_farskip.sh` finishes by running `sglang_port/verify_sglang_overlay_515.py`, AMD's
invariant checks for the FarSkip overlay.

## Bringing up a node

Run from a host that can ssh to the node (the containers have no ssh). The miles checkout
must sit on a share every node mounts, because the container bind-mounts it at
`/root/miles` and runs this kit from there.

```bash
export SHARE_HOST=/path/to/shared/workdir     # -> /workspace in the container
export RAY_HEAD_IP=<head node IP>
# MILES_HOST defaults to $SHARE_HOST/miles; set it if the checkout lives elsewhere.
# PII_ROOT defaults to /workspace/repos/Primus-Instella-Internal (in-container path).

AS_HEAD=1 RAY_HEAD_IP=<head IP> bash bringup_node.sh <head-host>   # first node
bash bringup_node.sh <worker-host>                                  # each worker
bash bringup_node.sh <host> --check-only                            # read-only audit
bash bringup_node.sh <host> --teardown [--remove]                   # ray stop, then stop
```

`bash bringup_node.sh --help` lists every knob (`CTR`, `IMAGE`, `NUM_GPUS`,
`EXPECT_NET_IF`, `ALLOW_IMAGE_MISMATCH`, busy thresholds). The script refuses nodes whose
GPUs are in use, whose image content differs from the head's, or that already belong to
another Ray cluster, and it exits non-zero at the first failed check rather than joining a
half-provisioned node.

## Launching runs on MI325X

After bringup, launch `run_async.sh`, `run_code.sh` or `run_if.sh` from inside the head
container as usual, with:

```bash
export MEGATRON_PATH=/root/megatron_farskip    # paths.sh defaults to the Primus submodule
export PRIMUS_ROOT=/workspace/repos/Primus-Instella-Internal
```

`MEGATRON_PATH` must point at the tree `apply_farskip_training.sh` built: the Primus
submodule has FarSkip but not the R3 or mcore-data patches, and it is first on the Ray
workers' `PYTHONPATH`.

`prefilter.sh` works unchanged. Leave `FARSKIP_SETUP` unset: the sglang patches are
already in the container.
