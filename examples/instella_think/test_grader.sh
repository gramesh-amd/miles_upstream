#!/usr/bin/env bash
# Runs every grader test against a private code_exec_server.py on 127.0.0.1 (own token, scratch dir
# and free port, all removed on exit):
#   test_code_exec_hardening.py, test_code_exec_adversarial.py, test_code_hack_monitor.py in both
#   reward modes, the code_reward.py self-test, the Python >= 3.13 refusal, and (when that copy
#   is reachable) that the lux verification's vendored code_exec_server.py matches this one.
#
#   bash examples/instella_think/test_grader.sh
#   GRADER_PYTHON=/path/to/venv/bin/python bash ...   # interpreter for programs (numpy cases need numpy)
#   GRADER_PROCESSES=4 GRADER_WORKERS=16 bash ...       # server with --processes 4 (the Lux sandbox layout)
#
# Clients need python3 with `requests`. Timing cases need --max-timeout >= 5 and an idle host.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${GRADER_PYTHON:-python3}"
WORKERS="${GRADER_WORKERS:-8}"
PROCESSES="${GRADER_PROCESSES:-1}"
LUX_COPY="${LUX_SERVER_COPY-/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/lux/code_exec_server.py}"
D="$(mktemp -d)"
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null; wait 2>/dev/null; rm -rf "$D"' EXIT
fails=0
step() { echo; echo "=== $1"; shift; "$@" || { echo "FAILED: $*"; fails=$((fails + 1)); }; }

lux_copy() {
   [ -f "$LUX_COPY" ] || { echo "skip: no $LUX_COPY"; return 0; }
   cmp -s "$HERE/code_exec_server.py" "$LUX_COPY" && { echo "lux copy matches"; return 0; }
   echo "lux copy differs: cp $HERE/code_exec_server.py $LUX_COPY, then update SERVER_SHA256 in verify_checkers.py" \
        "(now $(sha256sum "$HERE/code_exec_server.py" | cut -c1-64))"
   return 1
}
refuses_313() {
   printf '#!/bin/sh\necho 3 13\n' > "$D/py313"; chmod +x "$D/py313"
   out="$(timeout 30 python3 "$HERE/code_exec_server.py" --host 127.0.0.1 --port 0 --python "$D/py313" \
          --token-file "$D/tok313" --scratch "$D/s313" 2>&1)"
   [ $? -ne 0 ] && grep -q "must be <= 3.12" <<< "$out" && echo "refused: $out"
}
step "lux server copy" lux_copy
step "refuses Python 3.13" refuses_313

PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
python3 "$HERE/code_exec_server.py" --host 127.0.0.1 --port "$PORT" --workers "$WORKERS" --max-timeout 10 \
   --processes "$PROCESSES" --python "$PY" --token-file "$D/token" --scratch "$D/scratch" > "$D/server.log" 2>&1 &
SRV=$!
export CODE_EXEC_TEST_URL="http://127.0.0.1:${PORT}/$(for _ in $(seq 50); do [ -s "$D/token" ] && break; sleep 0.2; done; tr -d '[:space:]' < "$D/token")"
for _ in $(seq 30); do python3 -c 'import os, urllib.request; urllib.request.urlopen(os.environ["CODE_EXEC_TEST_URL"] + "/health", timeout=2)' 2>/dev/null && break; sleep 1; done

step "hardening" python3 "$HERE/test_code_exec_hardening.py"
step "adversarial" python3 "$HERE/test_code_exec_adversarial.py"
step "hack monitor (binary)" env CODE_REWARD_MODE=binary python3 "$HERE/test_code_hack_monitor.py"
step "hack monitor (partial)" env CODE_REWARD_MODE=partial python3 "$HERE/test_code_hack_monitor.py"
step "code_reward self-test" env CODE_EXEC_API_URL="$CODE_EXEC_TEST_URL" python3 "$HERE/code_reward.py"
echo; tail -2 "$D/server.log"
echo; [ "$fails" -eq 0 ] && echo "ALL GRADER TESTS PASSED" || echo "$fails STEP(S) FAILED"
exit $(( fails > 0 ))
