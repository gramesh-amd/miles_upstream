# Instella-MoE on latest miles (fully-async RL)

Self-contained port of Instella-MoE (16B-A3B, 27-layer gated-MLA DeepSeek-V3-style
FarSkip MoE, 64 experts / top-6 / 2 shared) training support onto latest miles
(v0.2.1). All Instella-specific launch/setup logic lives in this folder; the only
external touch is a small, idempotent patch to the installed `mbridge` (gate_proj),
applied by `patch_mbridge_gate_proj.sh`.

## Files

| File | Role |
|------|------|
| `model_args.sh` | `MODEL_ARGS=(...)` + `MOE_LAYER_FREQ` (layer 0 dense, 1–26 MoE). MLA + `--gated-attention` + `--use-simple-farskip-layer`. **No `--spec`** (this Megatron forbids it for MoE; FarSkip is config-driven). |
| `run_async.sh` | Main launcher. `PHASE=prepare` (HF→torch_dist) then `PHASE=train` (submit async DAPO-Math job to Ray). `SINGLE_NODE=1` for the 1-node smoke. |
| `investigate_megatron.sh` | Assemble a FarSkip-capable Megatron at `MEGATRON_PATH` (clone NVIDIA pin `847781764` + overlay patch tree + `pip install -e`). |
| `setup_farskip.sh` | Per-node: overlay the verified sglang515 FarSkip files onto the container sglang. |
| `patch_mbridge_gate_proj.sh` | Add the gated-attention `linear_gate_proj` mapping to the stock `DeepseekV3` mbridge (once per container). |
| `patch_megatron_compat.sh` | Stub newer-Megatron symbols latest miles imports at load (e.g. `optimizer.muon`) but this pin lacks. |
| `validate.sh` | 5-rung ladder: FarSkip+mbridge smoke → convert → sglang load → weight-sync parity → one train step. |
| `run-qwen3.6-35b-a3b-async-stale4-3node.sh` | Qwen3.6-35B-A3B fully-async DAPO launcher (3 nodes: 1 trainer EP8/CP4 + 2 rollout engines TP8/EP8, staleness 4). |
| `patch_megatron_optim_step.py` | **Required for optimizer-state resume.** Backports upstream Megatron `f4502eb1c` so `--optimizer-cpu-offload` checkpoints round-trip. Once per container. |
| `fix_ckpt_common_optimizer_state.py` | Repairs checkpoints already written *without* that patch, so they can resume with optimizer state. |

## How model/weights wire up (no `--spec`, no shadow ckpt)

- **FarSkip layers** are built by Megatron's `get_gpt_decoder_block_spec` from
  `--use-simple-farskip-layer` (a config flag). This Megatron **asserts `--spec is None`
  for MoE**, so we pass no `--spec`.
- **gate_proj weight mapping**: Instella reports `model_type: deepseek_v3`, so
  `AutoBridge` routes to the stock `DeepseekV3Bridge`. `patch_mbridge_gate_proj.sh`
  teaches that bridge the extra `self_attention.linear_gate_proj.weight ↔
  model.layers.{n}.self_attn.gate_proj.weight` key (idempotent, mirrors Primus's
  `rl/setup/mbridge_gate_proj_patch.sh`). Used for both HF→torch_dist convert and
  RL weight sync. No custom bridge / shadow checkpoint needed.
- sglang is unaffected (keys off `architectures=InstellaMoEForCausalLM` via the overlay).

## Single-node smoke test (1× 8-GPU node, e.g. 070) — DO THIS FIRST

Fully-async can't colocate (`train_async.py` asserts `not args.colocate`), so a single
node is split **4 trainer (EP4) + 4 rollout (1 engine EP4/DP4)**. `SINGLE_NODE=1` starts a
local Ray head, cleans stale procs, and applies the reduced topology automatically.

**Which container?** Run everything below in the **miles training container** (the one
with Ray + miles installed, e.g. `gora-async` / `rocm/sgl-dev:miles-rocm720-mi35x`) — NOT
the sglang-only lm_eval container (`lmsysorg/sglang-rocm:*`), which has sglang/aiter but
**no Ray or miles**, so `run_async.sh` will die with `ray: command not found`. Verify with
`source /opt/venv/bin/activate && which ray`.

```bash
# In the miles container on node 070:  source /opt/venv/bin/activate; cd /home/goramesh/miles-dev
export MEGATRON_PATH=/home/goramesh/Primus-Instella-Internal/training/third_party/Megatron-LM
export WANDB_KEY=...

# 1. FarSkip Megatron (once; needs git):        bash examples/instella_think/investigate_megatron.sh
# 1b. miles<->old-Megatron import shims:         bash examples/instella_think/patch_megatron_compat.sh
# 2. sglang FarSkip overlay:                     bash examples/instella_think/setup_farskip.sh
# 3. mbridge gate_proj patch (once/container):   bash examples/instella_think/patch_mbridge_gate_proj.sh
# 4. FarSkip + mbridge smoke (rung 1):           bash examples/instella_think/validate.sh
# 5. Convert HF -> torch_dist (uses all 8 GPU):
PHASE=prepare bash examples/instella_think/run_async.sh
#    (HF_CKPT defaults to $ASSET_ROOT/checkpoints/Instella-MoE-16B-A3B-DPO;
#     override it only to point at a checkpoint somewhere else)
# 6. Tiny end-to-end RL smoke (few steps, eval off):
SINGLE_NODE=1 NUM_ROLLOUT=8 EVAL_INTERVAL=100000 \
bash examples/instella_think/run_async.sh
```

## Full run (3 nodes / 24× MI355X, disaggregated: 1 trainer + 2 rollout)

```bash
# 0. Bring up 3 containers + Ray cluster.  Steps 1-4 same as above on the HEAD node,
#    and run setup_farskip.sh on EVERY node.

# 5. Train (fully async, staleness 4):
RAY_ADDRESS=http://<HEAD_IP>:8265 MASTER_ADDR=<HEAD_IP> \
WANDB_KEY=... bash examples/instella_think/run_async.sh
```

## FarSkip contract (must match on both sides)
- **Training**: `--use-simple-farskip-layer` (reference connectivity; in `model_args.sh`).
- **Rollout**: `FARSKIP_REFERENCE_DECODER_LAYER=1` (in the Ray `runtime_env` of `run_async.sh`).
Keeping both on the *reference* FarSkip path is what makes train/infer numerics agree
(this is the train/infer-mismatch investigation this port targets). Switch both to the
overlapped path together once reference parity is confirmed.

## Core miles edits (no plugin seam exists for these)
The RL weight-sync converter (`miles/backends/megatron_utils/megatron_to_hf/`) is a
hand-rolled, hardcoded dispatcher with no registration hook, so Instella support is a
minimal core edit:
- `megatron_to_hf/__init__.py`: route `instella*` model names to the DeepSeek-V3 converter.
- `megatron_to_hf/deepseekv3.py`: add the gated-attention `self_attention.linear_gate_proj.weight`
  → `self_attn.gate_proj.weight` mapping (harmless to plain DeepSeek-V3, which lacks that weight).

No atomic-update-group change is needed: Instella has `q_lora_rank=None`, so
`get_atomic_update_groups` returns empty (full `q_proj`, no fused q-lora group).

## Notes on the miles core-fixes analysis (from the old Instella-MoE/rl pin)
Most old `miles_core_changes` hunks are unnecessary on latest miles:
- `HIP_VISIBLE_DEVICES` fallback already present (`miles/.../sglang_engine.py`).
- sglang arg-name shims not needed — latest miles uses `--sglang-dp-size` / `--sglang-ep-size`
  and auto-derives `--sglang-*` from `ServerArgs`.
- Vocab padding: instead of the old fork's miles-core `_target_vocab_for_hf` patch (which
  would wrongly export TP-padding rows for other models), we pass `--vocab-size 128896`.
  Instella's HF config vocab_size *is* 128896 (real weights, not padding), so stock
  `remove_padding(args.vocab_size)` keeps all rows and matches sglang. Fix stays local.
- Router-revive / `--balance-by-flops` left **optional** (flags commented in `run_async.sh`);
  enable only if the corresponding core change exists in this miles build.

## Known issue: aiter MoE GPU memory fault on gfx950 (MI355X)

**Symptom:** with `SGLANG_USE_AITER=1`, the rollout engine dies with
`Memory access fault by GPU node-X … on address 0x…… Reason: Unknown` (often all/most
rollout GPUs, at varying tiny addresses), during the **first large/uneven DP-attention
prefill**. Because `--skip-eval-before-train` puts that first big prefill *after* the
first weight sync, it looks sync-triggered — but the weight sync itself completes fine.

**This is NOT a bug in the RL/weight-sync/FarSkip code.** What was ruled out (all on
gfx950, in-engine):
- Not negative/invalid `topk_ids` (live: ids are global `0..num_experts-1`, none negative).
- Not bad MoE values/shapes/indices: the *exact* dumped faulting `fused_moe` inputs
  (M≈13k) replay **cleanly** in isolation at every batch size.
- Not a stale/rebound weight pointer: live `w13`/`w2` are contiguous with correct strides
  and full storage right before the fault.
- It is **intermittent** (varying near-null fault addresses; some runs pass the first big
  prefill and fault on a later one) → signature of a small **out-of-bounds read inside an
  aiter kernel** that only faults when the OOB page is unmapped. A fresh-allocator replay
  never hits it. Mirrors upstream [sglang #27987](https://github.com/sgl-project/sglang/issues/27987)
  (uninitialized MoE routing/permute padding slots) and the AMD+DP-attention+aiter fault
  family ([#7691](https://github.com/sgl-project/sglang/issues/7691),
  [#21942](https://github.com/sgl-project/sglang/issues/21942)) — none of which patches
  this EP4 standard-CK + DP-attention prefill path.

**Status: `SGLANG_USE_AITER=1` is now the default and is REQUIRED for correctness.**
`SGLANG_USE_AITER=0` is *not* a safe mitigation: the sglang515 overlay stores MoE weights in
aiter's shuffled layout, so the non-aiter path reads them unshuffled and **silently** emits
gibberish (Primus-Instella-Internal#2: "2+2=5", `arc_challenge` 0.195, below the
random-chance floor) at normal-looking throughput — there is no crash to warn you.

The empty-KV out-of-bounds fault that previously forced `ENABLE_DP_ATTENTION=0` is fixed by
the `k.shape[0]==0` guard in sglang's `triton_backend` (commit `b54ceda`), verified by two
clean `lm_eval` runs at EP8/DP8 and by the healthy 1-node smoke. So `aiter=1` +
dp-attention is now both correct and stable, and `ENABLE_DP_ATTENTION=1` is the default
(it also keeps sglang's chunked-prefill clamp at 4096).

## Known issue: `--optimizer-cpu-offload` breaks optimizer-state resume (Megatron patch required)

**Symptom:** resuming any mid-training checkpoint *with* optimizer state dies in
`megatron/core/dist_checkpointing/serialization.py:126`:

```
ValueError: Cannot merge two lists with different lengths (1 and 116, encountered at
level ('optimizer', 0, 'param_state', 0, (torch.bfloat16, torch.float32), 0))
```

where the second length varies per rank (47 / 116 / 123 / …). Weights-only resume
(`--no-load-optim`) is unaffected.

**Cause:** `--optimizer-cpu-offload` makes the CPU sub-optimizer `torch.optim.AdamW`, which
keeps a **per-parameter `step` tensor**. TE's `FusedAdam` does not — its step lives in
`param_groups` — which is why plain `--use-precision-aware-optimizer` never hits this. The
`dp_reshardable` save path did `if key == 'step': continue`, so `step` was never wrapped as a
`ShardedTensor` and stayed a plain tensor. Plain tensors aren't `ShardedBase`, so
`extract_sharded_base` routes them into the **common** state dict — and `common.pt` is written
by **global rank 0 only**. The saved list therefore carries rank 0's bucket-entry count (1)
while every other rank expects its own. `dp_reshardable` was the only sharding format leaking
`step`; `fully_reshardable` and `fs_model_space` already skip it on save *and* load.

**No optimizer data is ever lost by this bug.** `exp_avg` / `exp_avg_sq` / `param` are all in
the `.distcp` shards at 100% coverage, and the authoritative scalar step is in the sharded
`param_groups`. Only a redundant, wrongly-shaped residue reaches `common.pt`.

**Fix (both parts):**

```bash
# 1. Future saves — once per container, since Megatron-LM lives in the image, not on NFS:
MEGATRON_PATH=/root/Megatron-LM python3 examples/instella_think/patch_megatron_optim_step.py

# 2. Checkpoints already written without the patch (drops the residue; .distcp untouched):
python3 examples/instella_think/fix_ckpt_common_optimizer_state.py <ckpt>/iter_0000039 --apply
```

Part 1 is upstream `f4502eb1c`, "Handle `step` key correctly in checkpoint save with
`--optimizer-cpu-offload`" (PR #2874), which is **not** an ancestor of this fork's
`miles-main`. It should be cherry-picked properly — the combination
`--optimizer-cpu-offload` + `--use-precision-aware-optimizer` has **no coverage** in
`tests/unit_tests/dist_checkpointing/`, so nothing catches this.

**Do NOT try to fix it by padding or truncating lists in `dist_checkpointing`'s `merge()`.**
That was attempted: it merges and loads cleanly, then dies on the *first optimizer step* with
`RuntimeError: a Tensor with 32 elements cannot be converted to Scalar` in `_fused_adam`,
because padding entries build `step` from `torch.empty(gap)` — uninitialized garbage of the
wrong shape. The checkpoint's length-1 entry is *correct*; the model-side 116 is a different
axis, so no merge-level change can reconcile them.

Verified working end to end: iteration 39 resumed with optimizer state and trained on past the
step where the pre-fix attempt crashed, with `grad_norm` continuous across the restart
(0.1486–0.1597 vs 0.1314 before) rather than spiking as it does on an Adam reset.

## Risks to confirm in-container (see `validate.sh`)
1. mbridge mapping completeness — `gate_proj` / full `q_proj` (q_lora_rank=null) must map
   with no "Unsupported parameter name" during convert.
2. FarSkip Megatron availability — `investigate_megatron.sh` must produce importable
   `SimpleFarSkipTransformerLayer`.
3. Weight-sync broadcast under `in_place` pause — **confirmed working** (sync completes,
   values bit-exact; the post-sync crash is the aiter MoE fault above, not sync).
   `torch_memory_saver` and the old sglang weight-loader/aiter patches are OPT-IN
   (`BUILD_TMS=1`), only needed if the broadcast hangs or you hit
   `'Parameter' object has no attribute 'weight_loader'`.
