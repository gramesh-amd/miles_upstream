#!/bin/bash
# Add the FarSkip gated-attention weight mapping to the INSTALLED mbridge's
# DeepseekV3 bridge, so Megatron's self_attention.linear_gate_proj.weight
# round-trips to/from the HF name model.layers.{n}.self_attn.gate_proj.weight.
#
# Instella-MoE reports model_type=deepseek_v3, so AutoBridge routes to
# DeepseekV3Bridge; this patch teaches that stock bridge the extra gate_proj
# key. No --spec / subclass / shadow-checkpoint needed. Idempotent.
#
# Mirrors Primus-Instella-Internal/rl/setup/mbridge_gate_proj_patch.sh (proven).
# Run INSIDE the container (needs mbridge importable). Re-run after any mbridge
# reinstall.
set -euo pipefail

MBRIDGE_FILE="$(python3 -c "import mbridge, pathlib; print(pathlib.Path(mbridge.__file__).parent/'models'/'deepseek_v3.py')" 2>/dev/null || true)"
if [ -z "${MBRIDGE_FILE}" ] || [ ! -f "${MBRIDGE_FILE}" ]; then
    echo "ERROR: mbridge deepseek_v3.py not found (is mbridge installed?)" >&2
    exit 1
fi
echo "Patching ${MBRIDGE_FILE}"

python3 - "${MBRIDGE_FILE}" <<'PY'
import ast
import sys
path = sys.argv[1]
src = open(path).read()

# Idempotent, but only if the existing patch is also syntactically valid.
if "linear_gate_proj" in src:
    try:
        ast.parse(src)
    except SyntaxError as e:
        sys.exit(f"  x mbridge already patched but file won't parse ({e}); "
                 f"reinstall mbridge (pip install --force-reinstall --no-deps mbridge) and rerun")
    print("  v mbridge gate_proj mapping already present")
    sys.exit(0)

lines = src.splitlines(keepends=True)

# gate_proj is a sibling dict key: anchor on q_a_layernorm.weight, then insert
# AFTER the "]," that closes its entry.
anchor = next((i for i, l in enumerate(lines) if "q_a_layernorm.weight" in l), None)
if anchor is None:
    sys.exit("  x q_a_layernorm.weight anchor not found in mbridge")
close = next((j for j in range(anchor, len(lines)) if lines[j].strip() == "],"), None)
if close is None:
    sys.exit("  x could not find end of q_a_layernorm entry")

indent = lines[close][: len(lines[close]) - len(lines[close].lstrip())]
lines[close + 1:close + 1] = [
    indent + "# FarSkip gated attention support\n",
    indent + '"self_attention.linear_gate_proj.weight": [\n',
    indent + '    "model.layers.{layer_number}.self_attn.gate_proj.weight"\n',
    indent + "],\n",
]
new = "".join(lines)
ast.parse(new)  # fail loud rather than write a file that breaks `import mbridge`
open(path, "w").write(new)
print("  v mbridge gate_proj mapping added + verified (syntax OK)")
PY
