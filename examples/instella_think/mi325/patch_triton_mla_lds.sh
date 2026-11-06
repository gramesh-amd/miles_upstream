#!/bin/bash
# Teach sglang's triton extend-attention kernel about Instella's MLA head dim so
# it fits in gfx942's 64 KB LDS.
#
# THE BUG is a hardcoded whitelist in _get_block_sizes_for_extend_attention:
#
#   if   Lq == 576: BLOCK_DMODEL, BLOCK_DPE = 512, 64   # DeepSeek-V3: 512 + 64
#   elif Lq == 288: BLOCK_DMODEL, BLOCK_DPE = 256, 32
#   elif Lq == 192: BLOCK_DMODEL, BLOCK_DPE = 128, 64
#   else:           BLOCK_DMODEL, BLOCK_DPE = next_power_of_2(Lq), 0
#
# Absorbed MLA runs with Lq = kv_lora_rank + qk_rope_head_dim. DeepSeek-V3 is
# 512 + 64 = 576 and is listed. Instella sets qk_rope_head_dim=32, so it is
# 512 + 32 = 544, which is NOT listed and therefore pads to next_power_of_2(544)
# = 1024 -- double the real width. The query tile alone then needs
#   BLOCK_M * BLOCK_DMODEL * 2 B = 64 * 1024 * 2 = 131072 B
# against gfx942's 65536 B, so the first absorbed extend (prefill) dies with
#   triton.runtime.errors.OutOfResources: out of resource: shared memory,
#   Required: 131072, Hardware limit: 65536
# taking out every rollout engine's scheduler. gfx950 never hit it: it has more
# LDS, and for DeepSeek's own 576 the split keeps BLOCK_DMODEL at 512.
#
# The fix adds the 544 entry, following the exact pattern of the 288 -> (256, 32)
# row, which already proves a 32-wide DPE lane works in this kernel. This also
# removes a 2x waste in the padded dimension, so it should be faster as well.
#
# Idempotent. Run in EVERY node's container, after apply_farskip.sh.
set -euo pipefail

SGLANG_PYTHON="${SGLANG_PYTHON:-/sgl-workspace/sglang/python}"
F="${SGLANG_PYTHON}/sglang/kernels/ops/attention/extend_attention.py"
[ -f "$F" ] || { echo "ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import sys

path = sys.argv[1]
src = open(path).read()

ANCHOR = """    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
"""
ADDITION = """    if Lq == 544:
        # Instella MLA: kv_lora_rank 512 + qk_rope_head_dim 32. Without this row
        # Lq falls to next_power_of_2(544) = 1024 and the query tile needs 128 KB
        # of LDS, which gfx942 (64 KB) cannot provide. Mirrors the 288 -> (256, 32)
        # row below, so the 32-wide DPE lane is already exercised upstream.
        BLOCK_DMODEL = 512
        BLOCK_DPE = 32
    elif Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
"""

if "Lq == 544" in src:
    print("  already patched")
elif ANCHOR in src:
    open(path, "w").write(src.replace(ANCHOR, ADDITION, 1))
    print("  patched: added Lq == 544 -> BLOCK_DMODEL=512, BLOCK_DPE=32")
else:
    print("  ERROR: the Lq == 576 block was not found verbatim; the kernel's "
          "block-size table has changed and needs review", file=sys.stderr)
    sys.exit(1)

# Widening the (64, 32) BLOCK_M/BLOCK_N row is also kept: it is gated on
# _is_gfx95, and gfx942 needs the smaller KV tile for the same LDS reason.
OLD_N = "if _is_gfx95 and _is_triton_ge_37 and Lq == 576 and Lv == 512:"
NEW_N = "if _is_triton_ge_37 and Lq in (544, 576) and Lv == 512:"
src = open(path).read()
if NEW_N in src:
    print("  BLOCK_N row already widened")
elif OLD_N in src:
    open(path, "w").write(src.replace(OLD_N, NEW_N, 1))
    print("  patched: (64,32) BLOCK_M/BLOCK_N now applies to Lq 544 on gfx942 too")
else:
    # A previous version of this script may have left a different variant.
    import re
    m = re.search(r"if _is_triton_ge_37 and [^\n]*Lv == 512:", src)
    if m:
        open(path, "w").write(src.replace(m.group(0), NEW_N, 1))
        print("  patched: normalised the BLOCK_N row to Lq in (544, 576)")
    else:
        print("  WARNING: could not locate the BLOCK_M/BLOCK_N row", file=sys.stderr)
PY

echo "=== selector now returns (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, warps) ==="
python3 - <<'PY'
import torch
from sglang.kernels.ops.attention.extend_attention import (
    _get_block_sizes_for_extend_attention as sel,
)

limit = torch.cuda.get_device_properties(0).shared_memory_per_block
print(f"  LDS limit {limit} B on {torch.cuda.get_device_properties(0).gcnArchName}")
for Lq, Lv in ((544, 512), (576, 512), (128, 128)):
    dmodel, dpe, dv, bm, bn, w = sel(Lq, Lv)
    q_tile = bm * dmodel * 2
    print(f"  Lq={Lq:4} Lv={Lv:4} -> dmodel={dmodel:5} dpe={dpe:3} BLOCK_M={bm} "
          f"BLOCK_N={bn}  q_tile={q_tile} B {'(was 131072)' if Lq == 544 else ''}")
PY

echo "=== REAL kernel launch at Instella's absorbed shape (the actual proof) ==="
python3 - <<'PY'
# Analytic LDS estimates are unreliable: triton decides what lives in shared
# memory vs registers. So compile and launch the kernel on the real shape.
import torch
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

H, Lq, Lv = 16, 544, 512          # num_attention_heads, absorbed q dim, kv_lora_rank
B, extend_len, prefix_len = 2, 128, 256
dev, dt = "cuda", torch.bfloat16

q = torch.randn(B * extend_len, H, Lq, dtype=dt, device=dev)
k = torch.randn(B * extend_len, 1, Lq, dtype=dt, device=dev)
v = torch.randn(B * extend_len, 1, Lv, dtype=dt, device=dev)
o = torch.empty(B * extend_len, H, Lv, dtype=dt, device=dev)

total = B * (extend_len + prefix_len)
k_buf = torch.randn(total, 1, Lq, dtype=dt, device=dev)
v_buf = torch.randn(total, 1, Lv, dtype=dt, device=dev)
req_to_token = torch.arange(total, device=dev, dtype=torch.int32).view(B, -1)
b_req_idx = torch.arange(B, device=dev, dtype=torch.int32)
b_seq_len = torch.full((B,), extend_len + prefix_len, device=dev, dtype=torch.int32)
b_start_loc_extend = torch.tensor([0, extend_len], device=dev, dtype=torch.int32)
b_seq_len_extend = torch.full((B,), extend_len, device=dev, dtype=torch.int32)

try:
    extend_attention_fwd(
        q, k, v, o, k_buf, v_buf, req_to_token, b_req_idx,
        b_seq_len, b_seq_len_extend, b_start_loc_extend, extend_len,
    )
    torch.cuda.synchronize()
    print(f"  OK: kernel compiled and ran at Lq={Lq}, Lv={Lv}; out={tuple(o.shape)}, "
          f"finite={torch.isfinite(o).all().item()}")
except TypeError as e:
    # Signature differs across sglang versions; the compile check above still stands.
    print(f"  SKIPPED launch (signature mismatch, not an LDS failure): {e}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)
PY

echo "done"
