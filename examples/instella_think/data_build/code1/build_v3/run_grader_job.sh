#!/bin/bash
# Start a private code_exec_server on this node, run one client command against it, then stop it.
#   bash run_grader_job.sh <tag> <port> <workers> <cmd...>
# The client reads the server URL from V7_GRADER_URL. The token stays in a 0700 temp dir and is never printed.
TAG=$1; PORT=$2; W=$3; shift 3
ROOT="${INSTELLA_RL_ROOT:-/it-share-prj2-1/data/goramesh-instella-rl}"
PY="${GRADER_PYTHON:-$ROOT/out/v7_grader/venv/bin/python}"          # python the graded programs run under
SRV="${GRADER_SERVER:-$(dirname "$0")/../../../code_exec_server.py}"
LOG="${CODE1_WORK:-$ROOT/out/v7_improve/data_audit}/server_$TAG.log"
S=$(mktemp -d); chmod 700 "$S"
python3 "$SRV" --host 127.0.0.1 --port "$PORT" --workers "$W" --max-tests 64 --max-timeout 20 --python "$PY" \
  --token-file "$S/token" --scratch "$S/scratch" > "$LOG" 2>&1 &
SP=$!; trap 'kill $SP; sleep 2; rm -rf "$S"' EXIT
for i in $(seq 100); do [ -s "$S/token" ] && break; sleep 0.2; done; sleep 2
export V7_GRADER_URL="http://127.0.0.1:$PORT/$(tr -d '[:space:]' < "$S/token")"
"$@"
