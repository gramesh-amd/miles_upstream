#!/bin/bash
# Grade a prompt set across every node we have, in parallel.
#
# WHY A SCRIPT. Grading is ~300 prompts/hour/node, so 14k prompts is 46 node-hours;
# only parallelism makes it tractable. Launching six nodes by hand is how earlier
# work in this project ended up with inconsistent settings between arms, so every
# shared setting is written exactly once here.
#
# CONTEXT 36864, RESPONSE 32768, deliberately. Training can afford ~36k total
# (4k prompt + 32k response). Grading at 65536 would measure difficulty under a
# budget we cannot train at, and grading at 32768 repeats the original filter's
# mistake: with allow_auto_truncate the response budget there was 32768 minus the
# prompt, and the recovered grades showed pass rate tracking length so tightly
# that 49.2% of 0/8 completions were simply cut off at the ceiling. Grading at the
# budget we will actually use is the only way the numbers transfer.
#
# RESUMABLE. filter_rl_difficulty.py caches per prompt in <output>.cache.jsonl, so
# a reclaimed node costs only its in-flight work. Re-running this script picks up
# where it left off.
set -uo pipefail

ROOT=/it-share-prj2-1/data/goramesh-instella-rl
INPUT="${INPUT:-$ROOT/out/analysis_1012/dapo/dapo_17k_en.jsonl}"
OUTDIR="${OUTDIR:-$ROOT/out/analysis_1012/dapo/grades}"
NODES="${NODES:-g2 g15 g32 g33 g34 g35}"
CONTEXT="${CONTEXT:-36864}"
MAXTOK="${MAXTOK:-32768}"
CONC="${CONC:-32}"
MODEL="${MODEL:-/workspace/models/Instella-MoE-16B-A3B-DPO}"

mkdir -p "$OUTDIR"
# The nodes see $ROOT as /workspace, so every in-container path has to be the
# OUTDIR rewritten against that mount -- not a literal. This was hardcoded to
# dapo/grades, which meant pointing OUTDIR anywhere else silently graded the
# DAPO shards again and overwrote their output.
case "$OUTDIR" in
    "$ROOT"/*) COUT="/workspace/${OUTDIR#"$ROOT"/}" ;;
    *) echo "OUTDIR must live under $ROOT (got $OUTDIR)"; exit 1 ;;
esac
TOTAL=$(wc -l < "$INPUT")
set -- $NODES
N=$#
echo "input=$INPUT total=$TOTAL nodes=$N context=$CONTEXT max_tokens=$MAXTOK"

# ---- shard the input ------------------------------------------------------
# Deterministic contiguous shards so a re-run maps a node to the same slice and
# its cache stays valid.
i=0
for n in $NODES; do
    awk -v i="$i" -v k="$N" 'NR % k == i' "$INPUT" > "$OUTDIR/shard_$n.jsonl"
    echo "  $n: $(wc -l < "$OUTDIR/shard_$n.jsonl") prompts"
    i=$((i + 1))
done

# ---- bring up a server on each node, then grade its shard ------------------
for n in $NODES; do
(
    HOST=tus1-p3-$n
    # Container must exist and be patched. bringup is the only thing that
    # guarantees the five sglang/megatron patches are applied.
    if ! ssh -n -o ConnectTimeout=10 "$HOST" "docker exec instella_rl true" 2>/dev/null; then
        echo "  [$n] no container, provisioning..."
        ip=$(ssh -n -o ConnectTimeout=10 "$HOST" "ip -4 route get 1.1.1.1 | sed -n 's/.*src \([0-9.]*\).*/\1/p'")
        AS_HEAD=1 RAY_HEAD_IP="$ip" bash "$ROOT/scripts/bringup_node.sh" "$HOST" >"/tmp/grade_bringup_$n.log" 2>&1 \
            || { echo "  [$n] PROVISION FAILED (see /tmp/grade_bringup_$n.log)"; exit 1; }
    fi

    # Restart the server only if it is not already serving at the right context.
    cur=$(ssh -n "$HOST" "docker exec instella_rl bash -lc 'grep -aoE \"context_length.: ?[0-9]+\" /workspace/out/analysis_1012/serve_$n.log 2>/dev/null | tail -1'" 2>/dev/null)
    up=$(ssh -n "$HOST" "docker exec instella_rl bash -lc 'pgrep -fc sglang.launch_server || echo 0'" 2>/dev/null | tail -1)
    if [ "${up:-0}" -eq 0 ] || [ "${cur#*: }" != "$CONTEXT" ]; then
        ssh -n "$HOST" "docker exec instella_rl bash -lc '
            pkill -f sglang.launch_server 2>/dev/null; sleep 5
            cd /workspace
            MODEL=$MODEL CONTEXT_LENGTH=$CONTEXT MEM_FRACTION=0.85 PORT=30000 \
            nohup bash /workspace/scripts/serve_sglang_farskip_dp.sh \
              > /workspace/out/analysis_1012/serve_$n.log 2>&1 &' " 2>/dev/null
        echo "  [$n] server starting (context $CONTEXT)"
    else
        echo "  [$n] server already up at context $CONTEXT"
    fi

    # Wait for readiness; the model load is minutes, not seconds.
    for _ in $(seq 1 60); do
        if ssh -n "$HOST" "docker exec instella_rl bash -lc 'grep -qa \"server is fired up\" /workspace/out/analysis_1012/serve_$n.log'" 2>/dev/null; then
            ok=1; break
        fi
        sleep 20
    done
    [ "${ok:-0}" = 1 ] || { echo "  [$n] SERVER NEVER READY"; exit 1; }

    ssh -n "$HOST" "docker exec -d instella_rl bash -lc '
        cd /workspace && nohup python3 scripts/filter_rl_difficulty.py \
          --input  $COUT/shard_$n.jsonl \
          --output $COUT/graded_$n.jsonl \
          --url http://127.0.0.1:30000 --max-tokens $MAXTOK --concurrency $CONC \
          > $COUT/grade_$n.log 2>&1 &'" 2>/dev/null
    echo "  [$n] grading started"
) &
done
wait
echo "all nodes dispatched; progress: wc -l $OUTDIR/graded_*.jsonl.cache.jsonl"
