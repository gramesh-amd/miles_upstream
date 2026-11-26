#!/bin/bash
# Wait for the Skywork math re-grade to finish, then start the Polaris band 1-5
# grading on the same nodes.
#
# WHY CHAIN RATHER THAN JUST LAUNCH. filter_rl_difficulty saturates a node's
# server at concurrency 48. Starting a second grader alongside the first would
# not add throughput, it would split it and make both jobs' ETAs unreadable --
# and the Skywork result is the last input to the 7B-vs-32B label question.
#
# NO pkill ANYWHERE IN THIS FILE. `pkill -f <pattern>` matches the shell running
# it whenever the pattern appears in its own command line, which has cost this
# project three separate incidents. This script only ever POLLS.
#
# Safe to re-run: grading is resumable per prompt via <output>.cache.jsonl, so a
# second invocation picks up where the first stopped rather than redoing work.
set -uo pipefail

ROOT=/it-share-prj2-1/data/goramesh-instella-rl
NODES="${NODES:-g2 g15 g33 g34 g35}"
POOL="$ROOT/out/analysis_1012/polaris/pool_band1_5.jsonl"
OUTDIR="$ROOT/out/analysis_1012/polaris/grades"
LOG="$ROOT/out/analysis_1012/polaris/chain.log"
mkdir -p "$(dirname "$LOG")"

say() { echo "$(date -u +'%F %T') $*" | tee -a "$LOG"; }

say "waiting for in-flight graders to finish on: $NODES"
for attempt in $(seq 1 240); do   # 240 x 60s = 4h ceiling
    busy=0
    for n in $NODES; do
        # Count graders INSIDE the container. `|| echo 0` keeps a node that is
        # unreachable or has lost its container from reading as "busy forever".
        # THE BRACKET IS LOAD-BEARING. `pgrep -f filter_rl_difficulty` matches
        # the bash -lc process running the pgrep, because that string is in its
        # own command line -- so the count never falls below 1 and the wait
        # never ends. This cost a 3-hour stall on the first attempt. Writing it
        # as [f]ilter... makes the cmdline literal "[f]ilter..." while the regex
        # still matches the real grader.
        c=$(timeout 20 ssh -n -o ConnectTimeout=10 "tus1-p3-$n" \
              "docker exec instella_rl bash -lc 'pgrep -fc \"[f]ilter_rl_difficulty\" || echo 0'" \
              2>/dev/null | tail -1)
        c=${c:-0}
        [[ "$c" =~ ^[0-9]+$ ]] || c=0
        busy=$((busy + c))
    done
    if [ "$busy" -eq 0 ]; then
        say "all graders idle after ${attempt} poll(s)"
        break
    fi
    say "  $busy grader process(es) still running; sleeping 60s"
    sleep 60
done

if [ "$busy" -ne 0 ]; then
    say "ABORT: graders still busy after the 4h ceiling; not launching"
    exit 1
fi

# Record the Skywork result before overwriting the shared server state, so the
# label-transfer question is answered even if the Polaris run later fails.
say "recomputing label calibration (Polaris + Skywork)"
(cd "$ROOT/scripts" && timeout 600 python3 mixv4_label_calib.py) >>"$LOG" 2>&1

say "launching Polaris band 1-5 grading: $(wc -l < "$POOL") prompts over $NODES"
INPUT="$POOL" OUTDIR="$OUTDIR" NODES="$NODES" CONC=48 \
    bash "$ROOT/scripts/mixv4_grade_parallel.sh" >>"$LOG" 2>&1
say "dispatched; progress: wc -l $OUTDIR/graded_*.jsonl.cache.jsonl"
