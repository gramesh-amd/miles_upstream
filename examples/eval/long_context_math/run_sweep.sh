#!/bin/bash
# Run a queue of cells across a fixed set of serving slots, one cell per slot at a time.
#
#   BENCH_SLOTS="node-a:30600 node-b,node-c:30700" bash run_sweep.sh queue.txt
#
# queue.txt holds one "<model> <length>" per line (# comments allowed). A slot is a comma-separated
# host list plus a port. Cells with a summary are skipped, so the sweep can be restarted at any time.
# Models that never serve, or whose cell stops making progress, are struck off in
# $BENCH_OUT_DIR/.dead_models so they do not burn a slot on every remaining budget.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${BENCH_OUT_DIR:-out/long_context_math}"
mkdir -p "$OUT/logs"
export BENCH_OUT_DIR="$(cd "$OUT" && pwd)"
OUT="$BENCH_OUT_DIR"
QUEUE_FILE="${1:?usage: run_sweep.sh queue.txt}"
read -r -a SLOTS <<< "${BENCH_SLOTS:?set BENCH_SLOTS to space-separated host[,host]:port entries}"
N_SAMPLES="${BENCH_N_SAMPLES:-32}"
CONCURRENCY="${BENCH_CONCURRENCY:-192}"
# Generous on purpose: slow models legitimately go >2h between progress lines, and an early kill
# throws away a multi-hour cell while a late kill only idles one slot.
STALL_MIN="${BENCH_STALL_MIN:-330}"
DEADFILE="$OUT/.dead_models"
say () { echo "$(date '+%F %T') $*"; }

mapfile -t QUEUE < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$QUEUE_FILE")
is_dead () { [ -f "$DEADFILE" ] && grep -qx "$1" "$DEADFILE"; }
cell_log () { echo "$OUT/logs/cell_$1_$2.log"; }

# A cell whose log has been idle for STALL_MIN is killed and its model struck. A cell in its forcing
# pass prints nothing for up to an hour, so those are left alone.
reap_stalled () {
    local pid cmd model len log age
    while read -r pid cmd; do
        model=$(sed -n 's/.*--model \([a-z0-9_]*\).*/\1/p' <<<"$cmd")
        len=$(sed -n 's/.*--length \([0-9]*k\).*/\1/p' <<<"$cmd")
        [ -n "$model" ] && [ -n "$len" ] || continue
        log=$(cell_log "$model" "$len")
        [ -f "$log" ] || continue
        grep -q "forcing .* truncated samples" "$log" 2>/dev/null && continue
        age=$(( ($(date +%s) - $(stat -c %Y "$log")) / 60 ))
        if [ "$age" -ge "$STALL_MIN" ]; then
            say "STALLED $model@$len: log idle ${age}m, killing and striking $model"
            kill "$pid" 2>/dev/null
            echo "$model" >> "$DEADFILE"
        fi
    done < <(pgrep -af '[r]un_cell.py')
}

# Two strikes, not one: a single serve failure can be a transient port still held by the previous cell.
strike_unservable () {
    local model=$1 len=$2 log n
    log=$(cell_log "$model" "$len")
    [ -f "$OUT/summary_${model}_${len}.json" ] && return
    n=$(grep -c "no server came up" "$log" 2>/dev/null || echo 0)
    [ "$n" -ge 1 ] || return
    [ "$n" -ge 2 ] || { say "$model@$len did not serve (attempt $n); will retry"; return; }
    is_dead "$model" || { say "$model never served ($len), $n attempts -- striking it"; echo "$model" >> "$DEADFILE"; }
}

i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
    reap_stalled
    for slot in "${SLOTS[@]}"; do
        [ "$i" -lt "${#QUEUE[@]}" ] || break
        hosts=${slot%:*}; port=${slot#*:}
        pgrep -af '[r]un_cell.py' | grep -q -- "--port $port" && continue
        read -r model len <<< "${QUEUE[$i]}"
        # Match in-flight cells on model+length, not port, so one cell never runs on two slots.
        if pgrep -af '[r]un_cell.py' | grep -q -- "--model $model --length $len "; then
            say "IN FLIGHT $model@$len"; i=$((i+1)); continue
        fi
        if [ -f "$OUT/summary_${model}_${len}.json" ]; then
            say "SKIP $model@$len (done)"; i=$((i+1)); continue
        fi
        if is_dead "$model"; then
            say "SKIP $model@$len ($model struck off)"; i=$((i+1)); continue
        fi
        say "launch $model @ $len on $hosts (port $port)"
        ( env BENCH_NODES="$hosts" python3 "$HERE/run_cell.py" \
              --model "$model" --length "$len" --n-samples "$N_SAMPLES" --concurrency "$CONCURRENCY" \
              --port "$port" >> "$(cell_log "$model" "$len")" 2>&1
          strike_unservable "$model" "$len" ) &
        i=$((i+1)); sleep 20
    done
    sleep 300
done
say "ALL QUEUED -- waiting for completion"
while pgrep -f '[r]un_cell.py' >/dev/null 2>&1; do reap_stalled; sleep 300; done
say "SWEEP COMPLETE"; python3 "$HERE/validate.py" 2>&1 | tail -3
