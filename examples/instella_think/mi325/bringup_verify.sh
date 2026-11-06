#!/bin/bash
# Independent post-condition verification for a provisioned RL node.
# Runs INSIDE the container. Reports on every check, exits non-zero if any fail.
#
# This deliberately does NOT trust the patch scripts' own output. Each patch is
# verified by grepping the target file for the text the patch is supposed to have
# left behind, plus a runtime import that exercises the result. A patch script can
# print "already patched" because its own idempotence guard matched something
# unrelated, or exit 0 after a `git apply` that produced a half-merged file; the
# checks below would still catch both.
#
#   Usage: bringup_verify.sh            # all checks
# Env:
#   MEGATRON_PATH   default /root/megatron_farskip
#   SGLANG_PYTHON   default /sgl-workspace/sglang/python
#   PII_ROOT        default /workspace/repos/Primus-Instella-Internal
set -uo pipefail

MEGATRON_PATH="${MEGATRON_PATH:-/root/megatron_farskip}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/sgl-workspace/sglang/python}"
PII_ROOT="${PII_ROOT:-/workspace/repos/Primus-Instella-Internal}"

FAILED=0
pass() { printf '  \033[32mOK  \033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAILED=$((FAILED + 1)); }

# grep_check <label> <file> <fixed-string>
grep_check() {
    local label="$1" file="$2" needle="$3"
    if [ ! -f "$file" ]; then fail "$label -- $file does not exist"; return; fi
    if grep -qF -- "$needle" "$file"; then pass "$label"; else
        fail "$label -- '$needle' not found in $file"
    fi
}

echo "=== node $(hostname) / container $(cat /etc/hostname) ==="

# ---------------------------------------------------------------- 0. mount topology
# /workspace and /root/miles come from the share and are therefore identical on
# every node for free. /sgl-workspace/sglang and /root/megatron_farskip live in the
# container's own overlay and must be patched per node. If that ever stops being
# true the whole per-node patching model is wrong, so assert it rather than assume.
echo "--- mount topology"
for p in /workspace /root/miles; do
    src="$(findmnt -no SOURCE --target "$p" 2>/dev/null)"
    case "$(findmnt -no FSTYPE --target "$p" 2>/dev/null)" in
        overlay|"") fail "$p should be a SHARED bind mount but resolves to the container overlay" ;;
        *)          pass "$p is shared (${src})" ;;
    esac
done
for p in "$SGLANG_PYTHON" "$MEGATRON_PATH"; do
    if [ "$(findmnt -no FSTYPE --target "$p" 2>/dev/null)" = "overlay" ]; then
        pass "$p is container-local (overlay) -- must be patched per node"
    else
        fail "$p is NOT on the container overlay; per-node patching assumption broken"
    fi
done

# ----------------------------------------------- 1. megatron_farskip tree exists
# NOT created by any of the three "patch" scripts -- it is built by
# apply_farskip_training.sh, and it is not in the image. patch_megatron_r3.sh and
# patch_megatron_mcore_data.sh both hard-fail without it.
echo "--- megatron_farskip tree (built by apply_farskip_training.sh)"
if [ -d "$MEGATRON_PATH/megatron/core" ]; then
    pass "$MEGATRON_PATH exists"
else
    fail "$MEGATRON_PATH missing -- apply_farskip_training.sh has not run"
fi
grep_check "FarSkip transformer layers overlaid" \
    "$MEGATRON_PATH/megatron/core/transformer/transformer_layer.py" "FarSkip"
# np.product was removed in numpy 2.0 and this image ships numpy 2.3.
if [ -d "$MEGATRON_PATH" ] && grep -rqF "np.product(" "$MEGATRON_PATH" --include=*.py 2>/dev/null; then
    fail "np.product( still present (removed in numpy 2.0) -- numpy-2 fixup did not run"
else
    pass "no np.product( left in $MEGATRON_PATH"
fi

# ------------------------------------------------------- 2. sglang FarSkip patch
# Rollout side. Without it sglang has no InstellaMoEForCausalLM, silently falls
# back to the remote-code transformers backend and dies in CUDA-graph capture.
echo "--- sglang FarSkip patch ($SGLANG_PYTHON)"
grep_check "instella_moe.py EntryClass" \
    "$SGLANG_PYTHON/sglang/srt/models/instella_moe.py" "InstellaMoEForCausalLM"
grep_check "deepseek_v2 reference FarSkip decoder layer" \
    "$SGLANG_PYTHON/sglang/srt/models/deepseek_v2.py" "DeepseekV2ReferenceFarSkipDecoderLayer"
grep_check "communicator prepare_mlp_no_all_reduce" \
    "$SGLANG_PYTHON/sglang/srt/layers/communicator.py" "prepare_mlp_no_all_reduce"
grep_check "decode_attention Instella MLA shape" \
    "$SGLANG_PYTHON/sglang/kernels/ops/attention/decode_attention.py" "Lk == 544"
# Applied by patch_triton_mla_lds.sh, NOT by apply_farskip.sh. Omitting it does
# not degrade quality -- it crashes the engine with OutOfResources (131072 vs
# gfx942's 65536 LDS) at the first absorbed prefill, so this must be checked here
# rather than discovered an hour into a run.
grep_check "extend_attention Instella MLA block size (gfx942 LDS)" \
    "$SGLANG_PYTHON/sglang/kernels/ops/attention/extend_attention.py" "Lq == 544"
if grep -rqE "^<<<<<<< |^>>>>>>> " "$SGLANG_PYTHON/sglang/" 2>/dev/null; then
    fail "merge conflict markers left in the sglang tree:"
    grep -rlE "^<<<<<<< " "$SGLANG_PYTHON/sglang/" 2>/dev/null | sed 's/^/        /'
else
    pass "no merge conflict markers in the sglang tree"
fi
# A for/else whose body has no `break` runs the else unconditionally -- this is how
# the decoder stack once silently executed twice. Re-checked here rather than
# relying on apply_farskip.sh having run its own copy of this check.
if python3 - "$SGLANG_PYTHON" <<'PY'
import ast, pathlib, sys
root = pathlib.Path(sys.argv[1])
bad = []
for f in sorted(root.glob("sglang/srt/models/deepseek*.py")) + sorted(
        root.glob("sglang/srt/models/deepseek_common/**/*.py")):
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError as e:
        print(f"{f}: {e}"); sys.exit(1)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)) and node.orelse:
            if not any(isinstance(n, ast.Break) for n in ast.walk(node)):
                bad.append(f"{f.name}:{node.lineno}")
if bad:
    print(", ".join(bad)); sys.exit(1)
PY
then pass "no unconditional loop/else in the deepseek model files"
else fail "loop/else with no break (else runs unconditionally) -- see above"
fi
# The decisive runtime check: the registry is what sglang actually consults.
if python3 -c "
import sys
from sglang.srt.models.registry import ModelRegistry
sys.exit(0 if 'InstellaMoEForCausalLM' in ModelRegistry.models else 1)
" 2>/dev/null
then pass "InstellaMoEForCausalLM resolves in sglang's live ModelRegistry"
else fail "InstellaMoEForCausalLM NOT in sglang's ModelRegistry -- rollout would silently fall back"
fi

# --------------------------------------------------------- 3. R3 routing-replay
# Without both hooks replay_list stays empty and the trainer dies in
# register_replay_list_moe with IndexError before step 1.
echo "--- R3 routing replay hooks"
grep_check "router.py registers a Replay per TopKRouter" \
    "$MEGATRON_PATH/megatron/core/transformer/moe/router.py" \
    "routing_replay_manager.register_to_module"
grep_check "moe_utils.py routes compute_topk through the manager" \
    "$MEGATRON_PATH/megatron/core/transformer/moe/moe_utils.py" \
    "get_topk_fn(compute_topk"
# The fused path returns before compute_topk is reached, so the wrap must come
# before the score_function dispatch or R3 is silently bypassed.
if python3 - "$MEGATRON_PATH" <<'PY'
import sys
p = sys.argv[1] + "/megatron/core/transformer/moe/moe_utils.py"
try:
    s = open(p).read()
except OSError:
    sys.exit(1)
try:
    sys.exit(0 if s.index("get_topk_fn(compute_topk") < s.index('if score_function == "softmax":') else 1)
except ValueError:
    sys.exit(1)
PY
then pass "compute_topk is wrapped BEFORE the score_function dispatch"
else fail "compute_topk wrap is missing or lands after the dispatch -- R3 would be bypassed"
fi

# ------------------------------------------------- 4. mcore_data resume tolerance
echo "--- dist-checkpoint mcore_data tolerance"
grep_check "torch.py reads mcore_data defensively" \
    "$MEGATRON_PATH/megatron/core/dist_checkpointing/strategies/torch.py" \
    "getattr(ckpt_metadata, 'mcore_data', {})"

# ------------------------------------------- 5. the tree is the one actually used
# All the greps above are worthless if `import megatron.core` resolves somewhere
# else -- the image ships its own /root/Megatron-LM, and without the editable
# install that is what wins.
#
# Two separate checks, deliberately. 5a runs WITHOUT $MEGATRON_PATH on PYTHONPATH,
# so it can only pass if `pip install -e` actually took; putting the tree on
# PYTHONPATH would make this pass whether or not the install happened. 5b then adds
# it, because that is the environment the run really uses (run_async.sh:820,
# "MEGATRON_PATH first so the FarSkip Megatron wins") and because the editable
# install exposes only megatron.core -- megatron.training is reachable solely via
# PYTHONPATH, and miles' Megatron backend imports it.
echo "--- 5a. editable install took (no MEGATRON_PATH on PYTHONPATH)"
if PYTHONPATH="${PII_ROOT}/training" python3 -c "
import sys, megatron.core as m
sys.exit(0 if m.__file__.startswith(sys.argv[1]) else 2)" "$MEGATRON_PATH" 2>/dev/null
then pass "megatron.core resolves to $MEGATRON_PATH via the editable install"
else fail "megatron.core does NOT resolve to $MEGATRON_PATH -- 'pip install -e' did not take, so the image's /root/Megatron-LM (no FarSkip) would win"
fi

echo "--- 5b. the real runtime environment imports (MEGATRON_PATH first, as the run sets it)"
PYTHONPATH="${MEGATRON_PATH}:${PII_ROOT}/training:${PYTHONPATH:-}" python3 <<'PY' >/tmp/bringup_rt.log 2>&1
import inspect
from megatron.core.transformer.transformer_layer import (  # noqa: F401
    OverlappedFarSkipTransformerLayer, SimpleFarSkipTransformerLayer)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
assert "use_simple_farskip_layer" in inspect.signature(get_gpt_layer_local_spec).parameters, \
    "get_gpt_layer_local_spec has no use_simple_farskip_layer parameter"
import miles  # noqa: F401
# NOT just `import miles`: the Megatron backend is imported lazily inside
# parse_args(), and that is where version skew between the image's miles and
# AMD's older pinned Megatron actually bites.
import miles.backends.megatron_utils.arguments  # noqa: F401
print("RUNTIME_OK")
PY
if grep -q '^RUNTIME_OK$' /tmp/bringup_rt.log 2>/dev/null; then
    pass "FarSkip layers + layer spec + miles Megatron backend all import"
else
    fail "runtime import failed; last lines of the traceback:"
    tail -6 /tmp/bringup_rt.log | sed 's/^/        /'
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo "VERIFY: all checks passed on $(hostname)"
    exit 0
fi
echo "VERIFY: ${FAILED} check(s) FAILED on $(hostname)"
exit 1
