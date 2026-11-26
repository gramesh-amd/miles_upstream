#!/bin/bash
# Wait until no grader is running on the pool, then dispatch the next grading
# job. Generalised from polaris_chain.sh so queued jobs do not need a new script
# each time.
#
#   INPUT=<prompts.jsonl> OUTDIR=<dir> [NODES=...] bash scripts/grade_chain.sh
#
# WHY CHAIN RATHER THAN RUN CONCURRENTLY. filter_rl_difficulty already saturates
# a node's server at concurrency 48. A second grader on the same node does not
# add throughput, it splits it, and it makes both ETAs unreadable.
#
# TWO TRAPS THIS FILE EXISTS TO AVOID, both of which have already cost time here:
#
#  * `pgrep -f filter_rl_difficulty` MATCHES THE SHELL RUNNING IT, because the
#    pattern appears in that shell's own command line. The count then never
#    falls below 1 and the wait never ends -- this stalled the Polaris launch
#    for three hours. Written as "[f]ilter_rl_difficulty" the cmdline contains a
#    literal bracket while the regex still matches the real grader.
#  * NO pkill ANYWHERE. Same self-match, but destructive. This script only polls.
#
# Re-running is safe: grading caches per prompt in <output>.cache.jsonl, so a
# second invocation resumes rather than regrades.
set -uo pipefail

ROOT=/it-share-prj2-1/data/goramesh-instella-rl
NODES="${NODES:-g2 g15 g33 g34 g35}"
INPUT="${INPUT:?set INPUT to the prompts jsonl}"
OUTDIR="${OUTDIR:?set OUTDIR}"
CONC="${CONC:-48}"
MAXWAIT="${MAXWAIT:-1440}"   # polls at 60s => 24h ceiling
LOG="${LOG:-$OUTDIR/chain.log}"
mkdir -p "$OUTDIR"

say() { echo "$(date -u +'%F %T') $*" | tee -a "$LOG"; }

say "queued: $(wc -l < "$INPUT") prompts -> $OUTDIR"
say "waiting for in-flight graders on: $NODES"
busy=1
for attempt in $(seq 1 "$MAXWAIT"); do
    busy=0
    for n in $NODES; do
        c=$(timeout 20 ssh -n -o ConnectTimeout=10 "tus1-p3-$n" \
              "docker exec instella_rl bash -lc 'pgrep -fc \"[f]ilter_rl_difficulty\" || echo 0'" \
              2>/dev/null | tail -1)
        c=${c:-0}
        [[ "$c" =~ ^[0-9]+$ ]] || c=0
        busy=$((busy + c))
    done
    if [ "$busy" -eq 0 ]; then
        say "graders idle after ${attempt} poll(s)"
        break
    fi
    [ $((attempt % 15)) -eq 1 ] && say "  $busy grader(s) still running"
    sleep 60
done

if [ "$busy" -ne 0 ]; then
    say "ABORT: still busy at the ${MAXWAIT}-poll ceiling; not launching"
    exit 1
fi

say "dispatching"
INPUT="$INPUT" OUTDIR="$OUTDIR" NODES="$NODES" CONC="$CONC" \
    bash "$ROOT/scripts/mixv4_grade_parallel.sh" >>"$LOG" 2>&1
say "dispatched; progress: wc -l $OUTDIR/graded_*.jsonl.cache.jsonl"
