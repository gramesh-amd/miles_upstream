#!/bin/bash
# sglang launcher for one benchmark node, driven entirely by environment so the caller never nests quotes:
#
#   BENCH_MODEL=<path> BENCH_TP=1 BENCH_DP=8 BENCH_CTX=131072 BENCH_PORT=30100 BENCH_LOG=<path> \
#   BENCH_EXTRA='--attention-backend triton' BENCH_OVERRIDE='{"rope_scaling":...}' bash serve_sglang.sh
#
# Kills every sglang process on the node first. Never point it at a node that is running training;
# use serve_sglang_coloc.sh for that.
set -uo pipefail

MODEL="${BENCH_MODEL:?}"
TP="${BENCH_TP:-1}"
DP="${BENCH_DP:-8}"
CTX="${BENCH_CTX:-65536}"
PORT="${BENCH_PORT:-30100}"
LOG="${BENCH_LOG:?}"
EXTRA="${BENCH_EXTRA:-}"
OVERRIDE="${BENCH_OVERRIDE:-}"

# Scheduler subprocesses re-exec under different command lines and keep holding the
# torch.distributed port, so kill the whole family and wait for it to exit.
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang::" 2>/dev/null
pkill -9 -f "sglang/srt" 2>/dev/null
for _ in $(seq 1 15); do
    pgrep -f "sglang" >/dev/null 2>&1 || break
    sleep 2
done
sleep 6

ARGS=(--model-path "$MODEL" --trust-remote-code
      --tp-size "$TP" --dp-size "$DP"
      --host 0.0.0.0 --port "$PORT"
      --context-length "$CTX"
      --mem-fraction-static 0.78
      --max-running-requests 256
      --disable-radix-cache)

# Shell callers should pass the override as a file: ssh -> docker exec -> bash -lc strips one layer of
# quoting each, which turns the JSON's double quotes into invalid JSON.
if [ -n "${BENCH_OVERRIDE_FILE:-}" ] && [ -s "${BENCH_OVERRIDE_FILE}" ]; then
    OVERRIDE="$(cat "${BENCH_OVERRIDE_FILE}")"
fi
[ -n "$OVERRIDE" ] && ARGS+=(--json-model-override-args "$OVERRIDE")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

nohup python3 -m sglang.launch_server "${ARGS[@]}" >> "$LOG" 2>&1 &
echo "launched pid $! port $PORT ctx $CTX tp $TP dp $DP -> $LOG"
