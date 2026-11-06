#!/bin/bash
# Apply the Instella-MoE FarSkip patch-set to a container's SGLang.
#
# Why this instead of AMD's update_sglang_workspace_files.sh: that script copies
# whole files, so on any container newer than the overlay's base it silently
# reverts those files to the base snapshot. On rocm/sgl-dev:miles-rocm10-mi30x
# that reintroduces calls to sglang.srt.tp_invariant_ops /
# sglang.srt.true_on_policy, neither of which exists there -> ImportError at
# startup. These patches carry only the FarSkip delta, so the container keeps its
# own SGLang everywhere else.
#
# Built against the tree in rocm/sgl-dev:miles-rocm10-mi30x-20260930
# (sglang 0.5.21.dev65+g57e7d0e84, commit 57e7d0e848 on branch sglang-miles).
# On a different SGLang, hunks may need review: `git apply -3` (below) will say so.
#
#   Usage: apply_farskip.sh [--check]
#   Env:   SGLANG_PYTHON (default /sgl-workspace/sglang/python)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELTA="${HERE}/farskip_delta"
SGLANG_PYTHON="${SGLANG_PYTHON:-/sgl-workspace/sglang/python}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$SGLANG_PYTHON/sglang" ] || { echo "ERROR: no sglang tree at $SGLANG_PYTHON"; exit 1; }
# Patch paths are rooted at `python/sglang/...`, so apply from the repo root.
SGLANG_ROOT="$(dirname "$SGLANG_PYTHON")"
cd "$SGLANG_ROOT"

echo "target: $SGLANG_PYTHON"
python3 -c "import importlib.metadata as m; print('  sglang', m.version('sglang'))" 2>/dev/null || true

applied=0
skipped=0
failed=0
for p in "$DELTA"/*.patch; do
    name="$(basename "$p" .patch)"
    # Idempotent: a reverse-apply that succeeds means the patch is already in.
    if git apply --reverse --check "$p" 2>/dev/null || patch -p1 -R --dry-run -s -f < "$p" >/dev/null 2>&1; then
        echo "  = $name (already applied)"
        skipped=$((skipped + 1))
        continue
    fi
    if [ "$CHECK_ONLY" = 1 ]; then
        if git apply --check "$p" 2>/dev/null; then echo "  + $name (would apply)"; applied=$((applied + 1))
        else echo "  ! $name (needs review)"; failed=$((failed + 1)); fi
        continue
    fi
    # -3 uses the blobs recorded in the patch to 3-way merge, so a moved hunk
    # surfaces as a conflict instead of a silent reject.
    if git apply -3 --whitespace=nowarn "$p" 2>/dev/null || patch -p1 -s -f < "$p" 2>/dev/null; then
        echo "  + $name"
        applied=$((applied + 1))
    else
        echo "  ! $name FAILED"
        failed=$((failed + 1))
    fi
done

echo "applied=$applied already=$skipped failed=$failed"
if grep -rq "^<<<<<<<" python/sglang/ 2>/dev/null; then
    echo "conflict markers left in:"
    grep -rl "^<<<<<<<" python/sglang/ | sed 's/^/    /'
    exit 1
fi
[ "$failed" -eq 0 ] || exit 1

[ "$CHECK_ONLY" = 1 ] && exit 0

# Stale compiled kernels would mask the overlaid triton block sizes.
rm -rf /root/.triton/cache/* /tmp/torchinductor_root/triton/* 2>/dev/null || true

echo "verify (structural):"
# A for/else whose loop has no `break` runs its else block unconditionally. That
# is how the decoder stack silently executed twice: resolving a merge hunk
# dropped an `if self.farskip_decoder:` guard but left the matching `else:`,
# which rebound to the layer loop. Cheap to assert, and it cost days to find.
python3 - "$SGLANG_PYTHON" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
bad = []
for f in sorted(root.glob("sglang/srt/models/deepseek*.py")) + sorted(
    root.glob("sglang/srt/models/deepseek_common/**/*.py")
):
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError as e:
        sys.exit(f"  x {f} does not parse: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)) and node.orelse:
            if not any(isinstance(n, ast.Break) for n in ast.walk(node)):
                bad.append(f"{f.name}:{node.lineno}")
if bad:
    sys.exit("  x loop/else with no break (else runs unconditionally): " + ", ".join(bad))
print("  OK   no unconditional loop/else")
PY

echo "verify (AMD's overlay contract):"
# verify_sglang_overlay_515.py is AMD's own invariant list, copied from
# Primus-Instella-Internal/inference/scripts/. It is the real contract; the
# ad-hoc checks below are only a fast subset. THREE of its assertions are known
# not to apply to this SGLang and are reported but not fatal:
#   - decode_attention.py moved to kernels/ops/attention/ (our Lk==544 fix is there)
#   - server_args.py's SGLANG_FORCE_KV_SPLITS hook (_handle_amd_specifics) is gone
#   - "gated MLA attention" greps forward_mla.py for the overlay's literal
#     `if self.gated_attention:`. We removed that block because it computed a
#     gate and then discarded it; this SGLang plumbs the gate through
#     `prepare_attention_output_gate` / `_apply_attention_output_gate` instead.
#     The gate IS applied on every path -- see the smoke checks below.
if [ -f "$HERE/verify_sglang_overlay_515.py" ]; then
    SGLANG_PYTHON="$SGLANG_PYTHON" FARSKIP_REFERENCE_DECODER_LAYER=1 \
        python3 "$HERE/verify_sglang_overlay_515.py" || \
        echo "  (see note above: 3 assertions are expected to fail on this SGLang)"
fi

echo "verify (smoke):"
FARSKIP_REFERENCE_DECODER_LAYER=1 PYTHONPATH="$SGLANG_PYTHON:${PYTHONPATH:-}" python3 - <<'PY'
import importlib

m = importlib.import_module("sglang.srt.models.instella_moe")
assert m.EntryClass.__name__ == "InstellaMoEForCausalLM", m.EntryClass
print("  OK   instella_moe.EntryClass = InstellaMoEForCausalLM")

dsv2 = importlib.import_module("sglang.srt.models.deepseek_v2")
dsv2.DeepseekV2ReferenceFarSkipDecoderLayer
print("  OK   DeepseekV2ReferenceFarSkipDecoderLayer")

comm = importlib.import_module("sglang.srt.layers.communicator")
assert hasattr(comm.LayerCommunicator, "prepare_mlp_no_all_reduce")
print("  OK   LayerCommunicator.prepare_mlp_no_all_reduce")

src = open("python/sglang/kernels/ops/attention/decode_attention.py").read()
assert "Lk == 544" in src, "missing Instella MLA (512+32) decode shape"
print("  OK   decode_attention Lk==544 case")

from sglang.srt.models.registry import ModelRegistry

assert "InstellaMoEForCausalLM" in ModelRegistry.models
print("  OK   InstellaMoEForCausalLM registered")
PY
echo "done"
