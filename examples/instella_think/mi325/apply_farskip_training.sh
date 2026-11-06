#!/bin/bash
# Training-side FarSkip setup: put the FarSkip layers into Megatron so the miles
# ACTOR can train the Instella MoE. Run INSIDE the container (it pip-installs).
#
# Counterpart to sglang_port/apply_farskip.sh, which handles the ROLLOUT side.
#
# What this does, mirroring Primus-Instella-Internal/rl/setup/primus_instella_patch.sh:
#   1. overlay AMD's 11-file FarSkip patch tree onto their pinned Megatron
#      (NVIDIA/Megatron-LM @ 8477817, submodule already checked out)
#   2. pip install -e that Megatron, so `import megatron.core` resolves to it
#
# KNOWN RISK, read before running:
#   This image ships its OWN Megatron at /root/Megatron-LM (radixark/Megatron-LM,
#   branch miles-main) which the image's miles was built against. AMD's script was
#   written for the rlsys/miles image, where miles and Megatron were matched.
#   Installing AMD's older pinned Megatron REPLACES the image's, so miles may
#   break on version skew -- the same class of problem the SGLang port hit.
#   This script therefore records what it displaced and verifies BOTH sides
#   afterwards: FarSkip layers importable AND miles still imports.
#
# Usage (inside container):
#   bash /root/miles/examples/instella_think/mi325/apply_farskip_training.sh [--check]
# Env:
#   PII_ROOT    Primus-Instella-Internal mount (default /pii)
#   MILES_ROOT  miles checkout whose megatron imports get backfilled (default /root/miles)
set -euo pipefail

PII_ROOT="${PII_ROOT:-/pii}"
MILES_ROOT="${MILES_ROOT:-/root/miles}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

MEGATRON_SRC="${PII_ROOT}/training/third_party/Megatron-LM"
PATCHES="${PII_ROOT}/training/patches/megatron-lm"
PRIMUS_ROOT="${PII_ROOT}/training"

for p in "$MEGATRON_SRC/megatron/core/__init__.py" \
         "$PATCHES/megatron/core/transformer/transformer_layer.py"; do
    [ -f "$p" ] || { echo "ERROR: missing $p (is $PII_ROOT mounted, submodule initialised?)"; exit 1; }
done

echo "=== before ==="
python3 -c "import megatron.core as m; print('  megatron.core ->', m.__file__)" 2>/dev/null || echo "  megatron.core not importable"
python3 -c "import miles; print('  miles         ->', miles.__file__)" 2>/dev/null || echo "  miles not importable"

if [ "$CHECK_ONLY" = 1 ]; then
    echo "--check: stopping before any modification"
    exit 0
fi

# Work on a writable copy: PII_ROOT is mounted read-only and /home is shared.
WORK=/root/megatron_farskip
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$MEGATRON_SRC/." "$WORK/"
echo "=== overlaying FarSkip patches ($(find "$PATCHES" -type f | wc -l) files) ==="
cp -r "$PATCHES/." "$WORK/"

# AMD's pinned Megatron predates modules that current miles imports, so the
# backend dies at `parse_args()` with e.g.
#   ModuleNotFoundError: No module named 'megatron.core.tokenizers.utils'
# Backfill any module miles imports that this Megatron lacks from the image's
# newer Megatron. Measured: 4 of the 51 megatron modules miles imports are
# missing (3 exotic optimizers we do not use, plus tokenizers.utils).
#
# This is deliberately chosen over porting AMD's 11 FarSkip files onto the
# image's Megatron: the FarSkip delta is ~1650 lines but the two trees have
# drifted ~17000 lines, so a forward-port is far larger than this backfill.
# Caveat: presence is not API compatibility -- a signature change inside a
# backfilled module would still surface at runtime, so the verify below now
# exercises miles' real Megatron backend rather than just `import miles`.
IMAGE_MEGATRON="${IMAGE_MEGATRON:-/root/Megatron-LM}"
if [ -d "$IMAGE_MEGATRON" ]; then
    echo "=== backfilling modules missing from AMD's pin, from $IMAGE_MEGATRON ==="
    grep -rhoE "from megatron[a-zA-Z0-9_.]* import|import megatron[a-zA-Z0-9_.]*" \
        "$MILES_ROOT/miles" "$MILES_ROOT/train_async.py" 2>/dev/null \
        | sed -E 's/^from //; s/^import //; s/ import$//' | sort -u \
    | while read -r mod; do
        rel="${mod//.//}"
        [ -e "$WORK/$rel.py" ] || [ -d "$WORK/$rel" ] && continue
        if [ -d "$IMAGE_MEGATRON/$rel" ]; then
            mkdir -p "$(dirname "$WORK/$rel")" && cp -r "$IMAGE_MEGATRON/$rel" "$WORK/$rel"
            echo "  + $mod (package)"
        elif [ -f "$IMAGE_MEGATRON/$rel.py" ]; then
            mkdir -p "$(dirname "$WORK/$rel.py")" && cp "$IMAGE_MEGATRON/$rel.py" "$WORK/$rel.py"
            echo "  + $mod"
        fi
    done
fi

# Symbol-level backfill. The module-level backfill above only helps when a whole module
# is absent; miles also imports individual NAMES that AMD's older pin predates. Concrete
# case: miles_plugins/models/kimi_k3/layers.py does
#   from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
# and model plugins are auto-loaded, so an Instella run dies at import with
#   ImportError: cannot import name 'ensure_metadata_has_dp_cp_group'
# even though Kimi-K3 is irrelevant here. Append the definition from the image's newer
# Megatron; it only needs parallel_state and Optional, both already imported there.
if [ -d "$IMAGE_MEGATRON" ]; then
    echo "=== backfilling missing SYMBOLS from $IMAGE_MEGATRON ==="
    python3 - "$WORK" "$IMAGE_MEGATRON" <<'PY'
import re, sys
work, image = sys.argv[1], sys.argv[2]
# (relative module path, symbol)
WANTED = [("megatron/core/transformer/utils.py", "ensure_metadata_has_dp_cp_group")]
for rel, sym in WANTED:
    dst, src = f"{work}/{rel}", f"{image}/{rel}"
    try:
        d = open(dst).read()
    except OSError:
        print(f"  skip {sym}: {dst} absent"); continue
    if re.search(rf"^def {sym}\b", d, re.M):
        print(f"  ok   {sym} already present"); continue
    try:
        s = open(src).read()
    except OSError:
        print(f"  WARN {sym}: no source at {src}"); continue
    m = re.search(rf"^def {sym}\b.*?(?=^def |\Z)", s, re.M | re.S)
    if not m:
        print(f"  WARN {sym} not found in {src}"); continue
    with open(dst, "a") as f:
        f.write("\n\n# --- backfilled from the image's newer Megatron (see "
                "apply_farskip_training.sh) ---\n" + m.group(0).rstrip() + "\n")
    print(f"  +    {sym} appended to {rel}")
PY
fi

# NumPy 2 compatibility. np.product was deprecated in numpy 1.25 and REMOVED in 2.0;
# this image ships numpy 2.3.5. AMD's pinned Megatron still calls it in three
# dist_checkpointing files, so the conversion dies at the save step with
#   AttributeError: module 'numpy' has no attribute 'product'
# after successfully loading the model onto all 8 GPUs. np.prod is the documented
# replacement with identical semantics for these shape products.
echo "=== numpy 2 compat: np.product -> np.prod ==="
_np_hits=$(grep -rl "np\.product\|numpy\.product" "$WORK" --include=*.py 2>/dev/null || true)
if [ -n "$_np_hits" ]; then
    echo "$_np_hits" | while read -r f; do
        sed -i 's/\bnp\.product(/np.prod(/g; s/\bnumpy\.product(/numpy.prod(/g' "$f"
        echo "  fixed $(basename "$f")"
    done
else
    echo "  none found (already patched or newer Megatron)"
fi

echo "=== pip install -e --no-deps (replaces the image's megatron-core) ==="
# --no-deps is REQUIRED. Without it pip resolves Megatron's declared
# requirements and perturbs the image's pinned environment -- observed: it
# broke `import miles` with "module 'numpy' has no attribute 'long'", even
# though neither Megatron nor the FarSkip patch tree references np.long
# (grep confirms zero hits). The image already satisfies Megatron's deps.
pip install -q --no-deps -e "$WORK" 2>&1 | tail -3 || { echo "ERROR: pip install failed"; exit 1; }

# Primus provides the FarSkip layer specs the Instella model args reference.
export PYTHONPATH="${PRIMUS_ROOT}:${PYTHONPATH:-}"

echo "=== verify ==="
PYTHONPATH="${WORK}:${PRIMUS_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
import sys

ok = True

def check(label, fn):
    global ok
    try:
        fn()
        print("  OK   " + label)
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}")
        ok = False

def farskip_layers():
    from megatron.core.transformer.transformer_layer import (  # noqa: F401
        OverlappedFarSkipTransformerLayer,
        SimpleFarSkipTransformerLayer,
    )

def layer_spec():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    import inspect
    assert "use_simple_farskip_layer" in inspect.signature(get_gpt_layer_local_spec).parameters

def megatron_loc():
    import megatron.core as m
    print("       megatron.core ->", m.__file__)

# The whole point of the risk note: miles must still work afterwards.
def miles_still_imports():
    import miles  # noqa: F401
    import miles.utils.arguments  # noqa: F401


# `import miles.utils.arguments` is NOT sufficient -- it does not touch the
# Megatron backend, which is imported lazily inside parse_args(). That is where
# the version skew actually bites, so exercise it directly.
def miles_megatron_backend():
    import miles.backends.megatron_utils.arguments  # noqa: F401

check("megatron location", megatron_loc)
check("FarSkip transformer layers", farskip_layers)
check("get_gpt_layer_local_spec(use_simple_farskip_layer=...)", layer_spec)
check("miles still imports against this megatron", miles_still_imports)
check("miles MEGATRON BACKEND imports (where skew bites)", miles_megatron_backend)

sys.exit(0 if ok else 1)
PY

echo "done. Add to your run env:  export PYTHONPATH=${PRIMUS_ROOT}:\$PYTHONPATH"
