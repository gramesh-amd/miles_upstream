#!/bin/bash
# Validation ladder for the Instella-MoE miles port. Run INSIDE the container.
# Each rung gates the next.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/paths.sh"
export PYTHONPATH="${REPO_DIR}:${MEGATRON_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "### Rung 1: FarSkip Megatron builds + mbridge gate_proj mapping present ###"
python3 - <<'PY'
import pathlib
# FarSkip layers build from config (no --spec) in this Megatron.
from megatron.core.transformer.transformer_layer import SimpleFarSkipTransformerLayer  # noqa: F401
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
get_gpt_layer_local_spec(num_experts=64, multi_latent_attention=True, use_simple_farskip_layer=True)
# gate_proj mapping patched into the stock DeepseekV3 bridge.
import mbridge
dv3 = pathlib.Path(mbridge.__file__).parent / "models" / "deepseek_v3.py"
assert "linear_gate_proj" in dv3.read_text(), (
    f"gate_proj mapping missing in {dv3}. Run: bash examples/instella_think/patch_mbridge_gate_proj.sh"
)
print("OK: FarSkip MLA MoE spec builds; mbridge deepseek_v3 has gate_proj mapping")
PY
[ $? -eq 0 ] || { echo "Rung 1 FAILED"; exit 1; }

echo "### Rung 2: HF -> torch_dist conversion (PHASE=prepare) ###"
echo "Run: MEGATRON_PATH=${MEGATRON_PATH} PHASE=prepare bash ${SCRIPT_DIR}/run_async.sh"
echo "Expect: 'Converted -> ...torch_dist'. Watch for any 'Unsupported parameter name'"
echo "        (mbridge mapping gap, e.g. gate_proj or full q_proj)."

echo "### Rung 3: sglang loads InstellaMoEForCausalLM with FarSkip ###"
echo "Reuse the verified path: FARSKIP_REFERENCE_DECODER_LAYER=1 + the sglang515 overlay"
echo "(setup_farskip.sh must have been run on the node)."

echo "### Rung 4: weight-sync parity (torch_dist -> hf -> sglang vs HF baseline) ###"
echo "Sanity a few prompts; answers should match the HF/sglang baseline you evaluated."

echo "### Rung 5: one async train step ###"
echo "Train with small --num-rollout; confirm loss finite + weight broadcast completes."
echo "If broadcast hangs, preload torch_memory_saver (Instella-MoE/rl/setup/torch_memory_saver.sh)."
