#!/bin/bash
# sglang launcher for serving next to a live training job on the same node.
#
# Unlike serve_sglang.sh it never pkills anything, since the training job's own rollout servers share
# the node. The server runs in its own session and its process-group id is written to BENCH_PGID_FILE
# so teardown can kill exactly this server and its children. BENCH_MEM_FRACTION is required because
# it decides how much GPU memory is left to the training job.
set -uo pipefail

MODEL="${BENCH_MODEL:?}"
TP="${BENCH_TP:-1}"
DP="${BENCH_DP:-8}"
CTX="${BENCH_CTX:-65536}"
PORT="${BENCH_PORT:?}"
LOG="${BENCH_LOG:?}"
MEMF="${BENCH_MEM_FRACTION:?}"
MAXRUN="${BENCH_MAX_RUNNING:-64}"
PGID_FILE="${BENCH_PGID_FILE:?}"
EXTRA="${BENCH_EXTRA:-}"
OVERRIDE="${BENCH_OVERRIDE:-}"

if [ -s "$PGID_FILE" ] && kill -0 -- "-$(cat "$PGID_FILE")" 2>/dev/null; then
    echo "a co-located server from $PGID_FILE is still alive; refusing to start a second" >&2
    exit 1
fi

ARGS=(--model-path "$MODEL" --trust-remote-code
      --tp-size "$TP" --dp-size "$DP"
      --host 0.0.0.0 --port "$PORT"
      --context-length "$CTX"
      --mem-fraction-static "$MEMF"
      --max-running-requests "$MAXRUN"
      --disable-radix-cache)
[ -n "$OVERRIDE" ] && ARGS+=(--json-model-override-args "$OVERRIDE")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

setsid nohup python3 -m sglang.launch_server "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
pid=$!
sleep 1
ps -o pgid= -p "$pid" | tr -d ' ' > "$PGID_FILE"
echo "launched pid $pid pgid $(cat "$PGID_FILE") port $PORT memf $MEMF -> $LOG"
