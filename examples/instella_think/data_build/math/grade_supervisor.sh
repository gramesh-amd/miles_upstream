#!/bin/bash
# Autonomous overnight grading supervisor.
#
# WHAT IT DOES, every PERIOD seconds:
#   1. Recomputes which prompts are still ungraded, across every chunk ever run.
#   2. Leaves any node with a LIVE grader strictly alone.
#   3. Gives idle nodes (ours, no grader) the next unclaimed chunk.
#   4. Picks up pool nodes that have become free, provisions and serves them.
#   5. Restarts a server that died under a node whose grader is gone.
#
# DESIGN CONSTRAINTS LEARNED THE HARD WAY TODAY -- do not "simplify" these away:
#
#  * NEVER reassign part of a shard a running grader owns. filter_rl_difficulty
#    reads its whole input file at startup and keys its cache by ROW INDEX IN
#    THAT FILE. Handing the tail of a live shard to another node means the
#    original still grades it (that is how g33 ended up with 602 duplicate cache
#    records), and re-sharding to prevent it silently remaps every cache key.
#    So: one chunk file per assignment, its own output/cache path, and a node
#    with a live grader is never touched.
#
#  * The cache stores only {i, n_correct, n_truncated} -- no prompt identity. A
#    chunk file is therefore the ONLY thing that maps row i back to a prompt.
#    Chunk files are never deleted or rewritten once assigned.
#
#  * pkill -f <pattern> MATCHES THE SHELL RUNNING IT when the pattern appears in
#    its own command line. That has cost this project three separate incidents
#    (a self-killed shell, servers that produced no log, a wait-on-itself loop).
#    Every kill here excludes $$ and $PPID explicitly, and servers are launched
#    from a generated script file so the pattern never appears in an exec cmdline.
#
#  * bringup_node.sh starts a RAY HEAD, which grading does not use. It is called
#    only when a node has no container at all (it is the one thing that applies
#    the five sglang/megatron patches); a ray failure there is tolerated.
#
#  * Only nodes in POOL are ever touched, and only if they are idle and carry no
#    foreign container. The wider fleet belongs to other people.
set -uo pipefail

ROOT=/it-share-prj2-1/data/goramesh-instella-rl
WORK=$ROOT/out/analysis_1012/work
POOL="${POOL:-g15 g32 g33 g34 g35 g14 g2 g25 g26 g27 g29 g24}"
PERIOD="${PERIOD:-600}"
CHUNK="${CHUNK:-600}"
CONC="${CONC:-48}"
CTX=36864
MAXTOK=32768
LOG=$WORK/supervisor.log
mkdir -p "$WORK/chunks" "$WORK/claims"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# ---- sources to grade: (name, jsonl) -------------------------------------
# Every prompt is identified by "<source>:<line-number>" so grades from
# different chunk files can be merged without colliding.
SRC_DAPO=$ROOT/out/analysis_1012/dapo/dapo_17k_en.jsonl
SRC_DOLCI=$ROOT/out/analysis_1012/dolci_filter/pool_30k.jsonl

have_container() { ssh -n -o ConnectTimeout=10 "tus1-p3-$1" "docker exec instella_rl true" >/dev/null 2>&1; }
grader_live()   { [ "$(ssh -n -o ConnectTimeout=10 "tus1-p3-$1" "docker exec instella_rl bash -lc 'ps -eo args | grep -c \"[f]ilter_rl_difficulty\"'" 2>/dev/null | tail -1)" -gt 0 ] 2>/dev/null; }
server_live()   { ssh -n -o ConnectTimeout=10 "tus1-p3-$1" "docker exec instella_rl bash -lc 'curl -sf -m 8 http://127.0.0.1:30000/health_generate >/dev/null && echo up'" 2>/dev/null | grep -q up; }

node_is_ours_and_idle() {
    local n=$1
    local ctr
    ctr=$(ssh -n -o ConnectTimeout=10 "tus1-p3-$n" "docker ps --format '{{.Names}}' | grep -vE '^(device-metrics-exporter|fleet-)' | tr '\n' ' '" 2>/dev/null)
    case "$ctr" in
        *instella_rl*) return 0 ;;                 # ours
        "" ) return 0 ;;                            # empty, claimable
        * ) return 1 ;;                             # someone else is here
    esac
}

start_server() {
    local n=$1
    ssh -n "tus1-p3-$n" "docker exec instella_rl bash -lc 'cat > /workspace/out/analysis_1012/_serve_$n.sh <<\"EOS\"
#!/bin/bash
cd /workspace
export MODEL=/workspace/models/Instella-MoE-16B-A3B-DPO
export CONTEXT_LENGTH=$CTX MEM_FRACTION=0.90 PORT=30000
export ALLOW_AUTO_TRUNCATE=1 MAX_RUNNING=256 DECODE_STEPS=2 GRAPH_MAX_BS=128
export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=$MAXTOK
exec bash /workspace/scripts/serve_sglang_farskip_dp.sh
EOS
chmod +x /workspace/out/analysis_1012/_serve_$n.sh'" >/dev/null 2>&1
    ssh -n "tus1-p3-$n" "docker exec -d instella_rl bash -c '/workspace/out/analysis_1012/_serve_$n.sh > /workspace/out/analysis_1012/serve_$n.log 2>&1'" >/dev/null 2>&1
    say "[$n] server starting (ctx $CTX)"
    for _ in $(seq 1 20); do
        sleep 30
        server_live "$n" && { say "[$n] server ready"; return 0; }
    done
    say "[$n] server NEVER became ready"
    return 1
}

assign_chunk() {
    local n=$1 cid=$2
    ssh -n "tus1-p3-$n" "docker exec -d instella_rl bash -c 'cd /workspace && python3 scripts/filter_rl_difficulty.py \
        --input  /workspace/out/analysis_1012/work/chunks/$cid.jsonl \
        --output /workspace/out/analysis_1012/work/chunks/graded_$cid.jsonl \
        --url http://127.0.0.1:30000 --max-tokens $MAXTOK --concurrency $CONC \
        > /workspace/out/analysis_1012/work/chunks/log_$cid.txt 2>&1'" >/dev/null 2>&1
    echo "$n" > "$WORK/claims/$cid"
    say "[$n] assigned $cid ($(wc -l < "$WORK/chunks/$cid.jsonl") prompts) at concurrency $CONC"
}

say "=== supervisor start: pool='$POOL' period=${PERIOD}s chunk=$CHUNK conc=$CONC ==="

while true; do
    # 1. rebuild the ungraded set from every chunk + the pre-existing shards
    LIVE=""
    for n in $POOL; do
        grader_live "$n" && LIVE="$LIVE$n,"
    done
    LIVE="${LIVE%,}"
    say "live graders: ${LIVE:-none}"
    python3 "$ROOT/scripts/grade_remaining.py" --work "$WORK" --chunk "$CHUNK" --live "$LIVE" >>"$LOG" 2>&1

    # 2. find unclaimed chunks
    mapfile -t PENDING < <(cd "$WORK/chunks" 2>/dev/null && ls -1 chunk_*.jsonl 2>/dev/null \
        | sed 's/\.jsonl$//' | while read -r c; do [ -f "$WORK/claims/$c" ] || echo "$c"; done)
    say "pending unclaimed chunks: ${#PENDING[@]}"

    pi=0
    for n in $POOL; do
        node_is_ours_and_idle "$n" || { say "[$n] skip: foreign container"; continue; }
        if grader_live "$n"; then continue; fi          # busy: never disturb
        [ "$pi" -ge "${#PENDING[@]}" ] && continue       # nothing left to hand out

        if ! have_container "$n"; then
            ip=$(ssh -n -o ConnectTimeout=10 "tus1-p3-$n" "ip -4 route get 1.1.1.1 | sed -n 's/.*src \([0-9.]*\).*/\1/p'" 2>/dev/null)
            [ -z "$ip" ] && { say "[$n] unreachable"; continue; }
            say "[$n] provisioning"
            AS_HEAD=1 RAY_HEAD_IP="$ip" bash "$ROOT/scripts/bringup_node.sh" "tus1-p3-$n" >"/tmp/sup_bu_$n.log" 2>&1
            have_container "$n" || { say "[$n] provision failed"; continue; }
        fi
        server_live "$n" || start_server "$n" || continue
        assign_chunk "$n" "${PENDING[$pi]}"
        pi=$((pi + 1))
    done

    sleep "$PERIOD"
done
