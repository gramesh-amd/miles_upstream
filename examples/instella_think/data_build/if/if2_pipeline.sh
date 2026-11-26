#!/bin/bash
# Unattended: finish prefilter2 -> score -> assemble 20k -> launch IF3 under the watchdog.
# Run on the login host:  nohup bash scripts/if2_pipeline.sh >/dev/null 2>&1 &
# Progress: out/if2_pipeline.log
set -uo pipefail
cd /it-share-prj2-1/data/goramesh-instella-rl
PLOG=out/if2_pipeline.log
GEN=out/if_prefilter2
SCORES=out/if_prefilter2_scores
N_CAND=40000
IPS="g2:10.32.81.253 g34:10.32.80.106 g26:10.32.82.4 g52:10.32.80.122 g53:10.32.80.123"
HEAD="ssh -n -o ConnectTimeout=20 tus1-p3-g2"
say() { echo "[$(date '+%F %T')] $*" >> "$PLOG"; }

n_done() { cat $GEN/*.jsonl 2>/dev/null | wc -l; }
driver_up() { $HEAD "docker exec instella_rl pgrep -f if_prefilter2_async" >/dev/null 2>&1; }
healthy_hosts() {
    local out=""
    for hp in $IPS; do
        ip=${hp#*:}
        [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$ip:30971/health)" = 200 ] && out="$out,$hp"
    done
    echo "${out#,}"
}
start_driver() {
    local hosts; hosts=$(healthy_hosts)
    if [ -z "$hosts" ]; then
        say "no healthy servers; restarting all"
        python3 -u scripts/if_prefilter2.py --candidates data/if/if_candidates2.jsonl >> out/if_prefilter2_serve.log 2>&1 &
        local p=$!
        for _ in $(seq 1 40); do sleep 30; hosts=$(healthy_hosts); [ "$(echo "$hosts" | tr ',' '\n' | wc -l)" -ge 4 ] && break; done
        kill $p 2>/dev/null; pkill -f if_prefilter2.py
        hosts=$(healthy_hosts)
        [ -n "$hosts" ] || { say "servers will not come up; giving up"; exit 1; }
    fi
    say "starting driver on hosts: $hosts"
    $HEAD "docker exec -d instella_rl bash -c 'ulimit -n 524288 && nohup python3 -u /workspace/scripts/if_prefilter2_async.py --hosts $hosts >> /workspace/out/if_prefilter2_async.log 2>&1'"
}

# ---------------- phase 1: generation ----------------
say "pipeline start; prefilter done=$(n_done)/$N_CAND"
last_n=$(n_done); last_change=$(date +%s); retried_fail=0
while true; do
    sleep 300
    n=$(n_done)
    [ "$n" -gt "$last_n" ] && { last_n=$n; last_change=$(date +%s); }
    idle=$(( $(date +%s) - last_change ))
    say "prefilter done=$n/$N_CAND driver_up=$(driver_up && echo 1 || echo 0) idle=${idle}s"
    if [ "$n" -ge $((N_CAND * 98 / 100)) ] && { ! driver_up || [ "$idle" -gt 900 ]; }; then
        say "generation complete enough ($n)"; break
    fi
    if ! driver_up; then
        if grep -q "generation done" out/if_prefilter2_async.log && [ "$retried_fail" = 1 ]; then
            say "driver finished twice; proceeding with $n"; break
        fi
        grep -q "generation done" out/if_prefilter2_async.log && retried_fail=1
        starts=$(( ${starts:-0} + 1 ))
        if [ "$starts" -gt 6 ]; then say "driver restarted 6 times; proceeding with $n"; break; fi
        say "driver not running; (re)starting to cover the remaining $((N_CAND - n))"
        start_driver; last_change=$(date +%s)
    elif [ "$idle" -gt 1500 ]; then
        say "no progress for ${idle}s; restarting driver"
        $HEAD "docker exec instella_rl pkill -f if_prefilter2_async" ; sleep 10
        start_driver; last_change=$(date +%s)
    fi
done
$HEAD "docker exec instella_rl pkill -f if_prefilter2_async" >/dev/null 2>&1
for h in tus1-p3-g2 tus1-p3-g34 tus1-p3-g26 tus1-p3-g52 tus1-p3-g53; do
    timeout 60 ssh -n $h "docker exec instella_rl pkill -9 -f sglang.launch_server" >/dev/null 2>&1
done
say "servers torn down"

# ---------------- phase 2: score ----------------
mkdir -p $SCORES
$HEAD "docker exec instella_rl bash -c 'GEN_GLOB=\"/workspace/$GEN/*.jsonl\" OUT=/workspace/$SCORES/partial_scores_mat.jsonl NPROC=96 python3 /workspace/scripts/if_partial_scores.py'" >> "$PLOG" 2>&1
say "scored: $(wc -l < $SCORES/partial_scores_mat.jsonl) prompts"

# ---------------- phase 3: assemble ----------------
(cd scripts && python3 if_assemble_hybrid.py --mat ../$SCORES/partial_scores_mat.jsonl \
    --candidates data/if/if_candidates2.jsonl --conflicts '' --target 20000 \
    --out data/if/if_train_hybrid20k.jsonl) >> "$PLOG" 2>&1
N_TRAIN=$(wc -l < data/if/if_train_hybrid20k.jsonl)
say "assembled $N_TRAIN training prompts"
[ "$N_TRAIN" -ge 5000 ] || { say "too few prompts ($N_TRAIN); not launching -- needs a human"; exit 1; }

# ---------------- phase 4: train ----------------
sleep 60
GPUS=$($HEAD "docker exec instella_rl ray status" 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1)
say "ray GPUs: $GPUS; launching IF3"
RUN=IF3 SAVE_TAG=IF3_hybrid20k_32k LAUNCHER=launch_IF3.sh FRESH=1 \
    nohup bash scripts/if_watchdog.sh >/dev/null 2>&1 < /dev/null &
disown
say "watchdog started (out/IF3_watchdog.log); pipeline done"
