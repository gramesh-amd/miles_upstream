#!/usr/bin/env bash
# Run the code RL sandbox (../code_exec_server.py) on a CPU host. Run on the login node.
#
#   cluster/code_exec.sh start     # start it on $CODE_EXEC_HOST (idempotent)
#   cluster/code_exec.sh status    # process + /health from this node
#   cluster/code_exec.sh stop
#   cluster/code_exec.sh url       # prints CODE_EXEC_API_URL for the training run
#
# The host must be reachable by ssh from here and by HTTP from the GPU nodes, and must
# see this checkout (shared home). The URL embeds the secret in $CODE_EXEC_TOKEN_FILE
# (created mode 600 on first start); the server rejects any path without it.
set -euo pipefail

INSTELLA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${INSTELLA_DIR}/paths.sh"

HOST="${CODE_EXEC_HOST:?set CODE_EXEC_HOST (a CPU host), e.g. in ~/.instella-env.sh}"
PORT="${CODE_EXEC_PORT}"
# Leave headroom on shared CPU hosts: default to all but 4 cores, at least 4 workers.
WORKERS="${CODE_EXEC_WORKERS:-}"
SERVER="${INSTELLA_DIR}/code_exec_server.py"
TOKEN_FILE="${CODE_EXEC_TOKEN_FILE}"
LOG="${USER_HOME}/logs/code-exec-${HOST%%.*}.log"

if [ ! -s "$TOKEN_FILE" ]; then
  ( umask 077; python3 -c 'import secrets; print(secrets.token_hex(16))' > "$TOKEN_FILE" )
fi
URL="http://${HOST}:${PORT}/$(tr -d '[:space:]' < "$TOKEN_FILE")"

remote() { timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "$@"; }

pid_on_host() {
  remote "pgrep -u \$USER -f 'code_exec_server.py --port ${PORT}'" 2>/dev/null | head -1
}

health() {
  python3 - "$URL" <<'EOF'
import json, sys, urllib.request
try:
    print(json.dumps(json.loads(urllib.request.urlopen(sys.argv[1] + "/health", timeout=5).read())))
except Exception as e:
    print(f"unreachable: {e}"); sys.exit(1)
EOF
}

case "${1:-status}" in
  start)
    if [ -n "$(pid_on_host)" ]; then
      echo "already running on $HOST (pid $(pid_on_host))"
    else
      mkdir -p "$(dirname "$LOG")"
      [ -n "$WORKERS" ] || WORKERS="$(remote nproc | awk '{n=$1-4; print (n<4?4:n)}')"
      remote "setsid nohup python3 -u $SERVER --port $PORT --workers $WORKERS --token-file $TOKEN_FILE \
              >> $LOG 2>&1 < /dev/null & disown"
      for _ in 1 2 3 4 5 6 7 8 9 10; do health >/dev/null 2>&1 && break; sleep 1; done
      echo "started on $HOST:$PORT (pid $(pid_on_host), $WORKERS workers), log $LOG"
    fi
    health
    ;;
  stop)
    pid="$(pid_on_host)"
    if [ -n "$pid" ]; then remote "kill $pid" && echo "stopped pid $pid on $HOST"; else echo "not running on $HOST"; fi
    ;;
  status)
    pid="$(pid_on_host)"
    echo "host: $HOST:$PORT  pid: ${pid:-none}"
    health || true
    tail -3 "$LOG" 2>/dev/null || true
    ;;
  url)
    echo "$URL"
    ;;
  *)
    sed -n '2,11p' "$0"; exit 1 ;;
esac
