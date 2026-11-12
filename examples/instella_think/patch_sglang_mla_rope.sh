#!/bin/bash
# Let sglang's absorbed-MLA attention skip aiter's fused rope + concat + KV-cache-write kernel
# (fused_qk_rope_cat_and_cache_mla) when MLA_UNFUSED_ROPE=1, applying rope with sglang's own
# rotary_emb and concatenating q/k in torch instead. Without the variable the code path is
# unchanged. Patches the INSTALLED forward_mla.py, after setup_farskip.sh has overlaid it.
# Idempotent; run inside the container (env_setup.sh step 1b).
#
# Why: on gfx950 with SGLANG_USE_AITER=1, a prefill (extend) batch that runs absorbed MLA comes
# out wrong for Instella-MoE. With the triton attention backend, sglang uses absorbed MLA for any
# extend batch containing a chunked-prefill continuation (and for every batch under
# --enable-deterministic-inference); otherwise it prefills with MHA, which is fine, and decode is
# fine. So several requests arriving together (a burst after idle, or one long prompt split across
# 2048-token chunks) get gibberish: on a fresh prefilter-config server, 96/256 burst requests
# started with <think>, and at --chunked-prefill-size 256 only 8/64 single requests did. With
# MLA_UNFUSED_ROPE=1 every probe came back 100% correct (2026-11-12, jobs 409249/409283/409301,
# scripts in $LUX_ROOT/shared/diag/prefill_*). The triton extend kernel itself matches a torch
# reference for Lq=544 (kernel_test.py), so the fault is in the fused aiter path.
set -euo pipefail

python3 - <<'PY'
import os

import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla as m

path = m.__file__
src = open(path).read()
if "MLA_UNFUSED_ROPE" in src:
    print(f"already patched: {path}")
    raise SystemExit(0)
ENV = "__import__('os').environ.get('MLA_UNFUSED_ROPE')"
edits = [
    # prepare: apply rope here (it is skipped on gfx95 + aiter so the fused kernel can do it)
    ("            and (not _use_aiter or not _is_gfx95_supported or self.use_dsa)\n        ):\n"
     "            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)",
     f"            and (not _use_aiter or not _is_gfx95_supported or self.use_dsa or {ENV} == '1')\n        ):\n"
     "            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)"),
    ("    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:\n",
     "    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:\n"
     f"        if {ENV} == '1':\n            return False\n"),
    # core: take the torch.cat branch instead of fused_qk_rope_cat_and_cache_mla
    ("            if _use_aiter_gfx95:\n                cos = self.rotary_emb.cos_cache\n"
     "                sin = self.rotary_emb.sin_cache\n\n                kv_cache_dtype",
     f"            if _use_aiter_gfx95 and {ENV} != '1':\n                cos = self.rotary_emb.cos_cache\n"
     "                sin = self.rotary_emb.sin_cache\n\n                kv_cache_dtype"),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        # Fatal only when the workaround is requested; otherwise setup must not fail over it.
        msg = f"expected 1 match in {path}, found {n}: {old[:70]!r}"
        if os.environ.get("MLA_UNFUSED_ROPE") == "1":
            raise SystemExit(f"ERROR: {msg}")
        print(f"WARNING: not patched ({msg}); MLA_UNFUSED_ROPE=1 would have no effect")
        raise SystemExit(0)
    src = src.replace(old, new)
compile(src, path, "exec")
open(path, "w").write(src)
print(f"patched: {path} (absorbed MLA honours MLA_UNFUSED_ROPE=1)")
PY
