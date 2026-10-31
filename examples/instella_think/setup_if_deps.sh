#!/bin/bash
# Install what the vendored IFEvalG checkers need, and pre-seed the nltk corpus.
# Run ONCE PER NODE, inside that node's training container (same place as setup_farskip.sh):
#
#   bash examples/instella_think/setup_if_deps.sh
#
# Then verify from the head node only:
#
#   bash examples/instella_think/setup_if_deps.sh --verify
#
# The dependency set is deliberately smaller than Instella-MoE/rl/setup/install_python_deps.sh,
# which also installs emoji, syllapy and pylatexenc. Those are needed by IFBench's EXTENDED
# registry, which we do not vendor -- ./IFEvalG imports exactly langdetect, absl, immutabledict
# and nltk (verified by grepping its imports). Installing the rest would be cargo cult.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/paths.sh"

verify() {
   python3 - "$SCRIPT_DIR" "$NLTK_DATA" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("NLTK_DATA", sys.argv[2])
rc = 0
for mod in ("langdetect", "absl", "immutabledict", "nltk", "emoji", "syllapy"):
    try:
        __import__(mod)
        print(f"  ok   {mod}")
    except ImportError as e:
        print(f"  MISS {mod}: {e}")
        rc = 1
try:
    from IFEvalG import instructions_registry as r
    n = len(r.INSTRUCTION_DICT)
    print(f"  ok   IFEvalG registry: {n} constraint types")
    if n < 40:
        print(f"  WARN registry unexpectedly small ({n})")
        rc = 1
except Exception as e:
    print(f"  FAIL IFEvalG import: {type(e).__name__}: {e}")
    rc = 1
# The tokenizer is the one lazy dependency that can fail at REWARD time rather than
# import time, which would show up as mass checker_error instead of a clean crash.
try:
    import nltk
    nltk.word_tokenize("a quick test sentence.")
    print("  ok   nltk.word_tokenize (punkt present)")
except Exception as e:
    print(f"  FAIL nltk tokenizer: {type(e).__name__}: {e}")
    rc = 1
# End to end, including the think gate and strict-vs-partial split.
try:
    import if_reward
    print(f"  ok   if_reward imported (REWARD_SCALE={if_reward.REWARD_SCALE})")
    if if_reward.REWARD_SCALE != 1.0:
        print("  WARN REWARD_SCALE != 1.0; miles' pass-rate metrics assume 1.0")
    # Without this the IFBench eval set scores a flat 0.0 on all 300 prompts.
    if if_reward.ifbench_registry is not None:
        n = len(if_reward.ifbench_registry.INSTRUCTION_DICT)
        print(f"  ok   IFBench extended registry: {n} constraint types")
    else:
        print("  FAIL IFBench registry not loaded -> IFBench eval would score 0 everywhere")
        rc = 1
except Exception as e:
    print(f"  FAIL if_reward import: {type(e).__name__}: {e}")
    rc = 1
print("VERIFY " + ("PASS" if rc == 0 else "FAIL"))
sys.exit(rc)
PY
}

if [ "${1:-}" = "--verify" ]; then
   echo "=== verifying IF reward stack on $(hostname -s) ==="
   verify
   exit $?
fi

echo "=== installing IFEvalG deps on $(hostname -s) ==="
# langdetect/absl-py/immutabledict/nltk are for the vendored IFEvalG. emoji and syllapy are
# for IFBench's EXTENDED registry, which the IFBench eval set needs: 300/300 of its prompts
# use constraints IFEvalG does not implement (words:consonants, ratio:stop_words, ...).
# spacy is in IFBench's requirements.txt but its instructions.py never imports it, so it is
# deliberately skipped.
python3 -m pip install --quiet langdetect absl-py immutabledict nltk emoji syllapy || {
   echo "FATAL: pip install failed" >&2; exit 1; }

echo "=== seeding nltk punkt into ${NLTK_DATA} ==="
mkdir -p "${NLTK_DATA}"
# punkt_tab is what instructions_util.py looks for first; punkt is the older fallback it
# tries next, so fetch both and let the import find whichever this nltk version wants.
NLTK_DATA="${NLTK_DATA}" python3 -c "
import nltk
for pkg in ('punkt_tab', 'punkt'):
    ok = nltk.download(pkg, download_dir='${NLTK_DATA}', quiet=True)
    print(f'  {pkg}: {\"ok\" if ok else \"FAILED\"}')
"

echo
verify