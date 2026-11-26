#!/bin/bash
# Grade a prompt set with the DPO model on several nodes in parallel: 8 rollouts per prompt.
#
#   INPUT=<prompts.jsonl> OUTDIR=<dir under $ROOT> NODES="g2 g15 ..." bash mixv4_grade_parallel.sh
#
# Each node must already run the `instella_rl` container with this miles checkout at /root/miles and the
# share at /workspace (the MI325 bringup, mi325/bringup_node.sh, sets this up). The script shards the input, starts
# ../serve_grader.sh on each node, waits for it, then runs filter_rl_difficulty.py on that node's shard.
#
# CONTEXT 36864, RESPONSE 32768: grade at the budget training uses (4K prompt + 32K response).
# Grading at a smaller context truncates long answers and makes hard prompts look harder.
#
# Resumable: filter_rl_difficulty.py caches per prompt in <output>.cache.jsonl, and shards are
# deterministic, so re-running picks up where it stopped.
set -uo pipefail

ROOT="${INSTELLA_RL_ROOT:-/it-share-prj2-1/data/goramesh-instella-rl}"   # share path on the host
CROOT="${CROOT:-/workspace}"                                             # same share inside the container
CKIT="${CKIT:-/root/miles/examples/instella_think}"                      # examples/instella_think inside the container
CONTAINER="${CONTAINER:-instella_rl}"
HOST_PREFIX="${HOST_PREFIX:-tus1-p3-}"
INPUT="${INPUT:-$ROOT/out/analysis_1012/dapo/dapo_17k_en.jsonl}"
OUTDIR="${OUTDIR:-$ROOT/out/analysis_1012/dapo/grades}"
NODES="${NODES:?set NODES, e.g. NODES=\"g2 g15 g32\"}"
CONTEXT="${CONTEXT:-36864}"
MAXTOK="${MAXTOK:-32768}"
CONC="${CONC:-32}"
MODEL="${MODEL:-$CROOT/models/Instella-MoE-16B-A3B-DPO}"

case "$OUTDIR" in
    "$ROOT"/*) COUT="$CROOT/${OUTDIR#"$ROOT"/}" ;;
    *) echo "OUTDIR must live under $ROOT (got $OUTDIR)"; exit 1 ;;
esac
mkdir -p "$OUTDIR"
set -- $NODES
N=$#
echo "input=$INPUT total=$(wc -l < "$INPUT") nodes=$N context=$CONTEXT max_tokens=$MAXTOK"

i=0
for n in $NODES; do
    awk -v i="$i" -v k="$N" 'NR % k == i' "$INPUT" > "$OUTDIR/shard_$n.jsonl"
    echo "  $n: $(wc -l < "$OUTDIR/shard_$n.jsonl") prompts"
    i=$((i + 1))
done

for n in $NODES; do
(
    HOST="$HOST_PREFIX$n"
    dx() { ssh -n -o ConnectTimeout=10 "$HOST" "docker exec $CONTAINER bash -lc '$1'" 2>/dev/null; }
    dx true || { echo "  [$n] container $CONTAINER not running on $HOST"; exit 1; }

    dx "pkill -f sglang.launch_server; sleep 5; cd $CROOT && \
        MODEL=$MODEL CONTEXT_LENGTH=$CONTEXT MEM_FRACTION=0.85 PORT=30000 \
        nohup bash $CKIT/data_build/serve_grader.sh > $COUT/serve_$n.log 2>&1 &"
    for _ in $(seq 1 60); do
        dx "grep -qa \"server is fired up\" $COUT/serve_$n.log" && { ok=1; break; }
        sleep 20
    done
    [ "${ok:-0}" = 1 ] || { echo "  [$n] server never became ready (see $OUTDIR/serve_$n.log)"; exit 1; }

    ssh -n "$HOST" "docker exec -d $CONTAINER bash -lc 'cd $CKIT/data_build/math && \
        nohup python3 filter_rl_difficulty.py --input $COUT/shard_$n.jsonl --output $COUT/graded_$n.jsonl \
          --url http://127.0.0.1:30000 --max-tokens $MAXTOK --concurrency $CONC > $COUT/grade_$n.log 2>&1'"
    echo "  [$n] grading started"
) &
done
wait
echo "progress: wc -l $OUTDIR/graded_*.jsonl.cache.jsonl"
