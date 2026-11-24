#!/bin/bash
# Build the v8 code RL mix end to end, or replay its last three steps from verified intermediates.
#
#   SANDBOX_URLS=http://h1:18950,http://h2:18950 WORK=/data/v8work OUT=/data/v8 bash build_v8.sh \
#       --union union.csv.gz --clusters clusters.parquet --carry carry_over.jsonl [--status problem_status.parquet] \
#       --eval-items eval_items.parquet --eval-hashes eval_hashes.jsonl \
#       --leetcode-pool leetcode_call.jsonl.gz --leetcode-refs leetcode_refs.jsonl.gz --codewars-pool codewars_fn.jsonl.gz
#
#   WORK=... OUT=... bash build_v8.sh --from assemble --legacy-dedup new_vs_carry_dups.csv \
#       --carry carry_over.jsonl --status problem_status.parquet --eval-items ... --eval-hashes ...   # exact replay of WORK/build
#
# Steps (--from STEP resumes; every step is resumable on its own): fetch refs cands verify fn assemble decontam finalize.
# SANDBOX_URLS: code_exec_server.py base URLs without the token; the token comes from CODE_EXEC_TOKEN or the file
# CODE_EXEC_TOKEN_FILE (default ~/.code-exec-token) and is never printed. verify runs one shard per URL, then a second
# pass that retries the rows a sandbox error dropped; the build stops if any row is still dropped that way, or if
# any step or shard fails. Each sandbox's /health is checked first. WORKERS: request threads per shard (default 32).
# --from verify and --from fn need neither --union nor --clusters.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FROM=fetch STATUS="" DEDUP="" UNION="" CLUSTERS="" CARRY="" EVAL_ITEMS="" EVAL_HASHES="" LC_POOL="" LC_REFS="" CW_POOL=""
while [ $# -gt 0 ]; do
  case $1 in
    --from) FROM=$2 ;;
    --union) UNION=$2 ;;
    --clusters) CLUSTERS=$2 ;;
    --carry) CARRY=$2 ;;
    --status) STATUS=$2 ;;
    --eval-items) EVAL_ITEMS=$2 ;;
    --eval-hashes) EVAL_HASHES=$2 ;;
    --leetcode-pool) LC_POOL=$2 ;;
    --leetcode-refs) LC_REFS=$2 ;;
    --codewars-pool) CW_POOL=$2 ;;
    --legacy-dedup) DEDUP=$2 ;;
    -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift 2
done
: "${WORK:?set WORK (intermediates)}" "${OUT:?set OUT (final files)}"
WORKERS=${WORKERS:-32}
STEPS=(fetch refs cands verify fn assemble decontam finalize)
start=-1
for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$FROM" ] && start=$i; done
[ $start -ge 0 ] || { echo "--from must be one of: ${STEPS[*]}" >&2; exit 2; }
need() { local v n; for v in "$@"; do n=${v%%:*}; [ -n "${!n}" ] || { echo "missing --${v#*:}" >&2; exit 2; }; done; }
runs() { local i; for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && [ $i -ge $start ] && return 0; done; return 1; }
[ $start -le 2 ] && need UNION:union CLUSTERS:clusters
[ $start -le 4 ] && need LC_POOL:leetcode-pool LC_REFS:leetcode-refs CW_POOL:codewars-pool
need CARRY:carry EVAL_ITEMS:eval-items EVAL_HASHES:eval-hashes
# Paths are relative to the caller's directory; the steps run from this one.
for v in WORK OUT UNION CLUSTERS CARRY STATUS EVAL_ITEMS EVAL_HASHES LC_POOL LC_REFS CW_POOL DEDUP; do
  if [ -n "${!v}" ]; then printf -v "$v" '%s' "$(realpath -m -- "${!v}")"; fi
done
B=$WORK/build
mkdir -p "$B" "$OUT"
cd "$HERE"

# Waits for every pid in PIDS; names the failed ones and stops the build if any failed.
wait_all() {
  local what=$1 p failed=0
  for p in "${PIDS[@]}"; do
    wait "$p" || { echo "FAILED: $what (${LABEL[$p]})" >&2; failed=1; }
  done
  [ $failed -eq 0 ] || exit 1
}

sandbox_urls() {
  : "${SANDBOX_URLS:?set SANDBOX_URLS (comma-separated code_exec_server.py URLs)}"
  local tok=${CODE_EXEC_TOKEN:-}
  [ -n "$tok" ] || tok=$(cat "${CODE_EXEC_TOKEN_FILE:-$HOME/.code-exec-token}")
  URLS=()
  local u
  IFS=, read -ra base <<< "$SANDBOX_URLS"
  for u in "${base[@]}"; do URLS+=("${u%/}/$tok"); done
  local k
  for k in "${!URLS[@]}"; do
    # The URL goes through the environment, so the token is in no command line or message.
    U=${URLS[$k]} python3 -c 'import os, urllib.request; urllib.request.urlopen(os.environ["U"] + "/health", timeout=10)' \
      2> /dev/null || { echo "sandbox ${base[$k]%/} failed its /health check (down, or a wrong token)" >&2; exit 1; }
  done
}

if runs fetch; then
  for s in openr1_verifiable openr1_tests hardtests_problems hardtests_tests ccplus_1x cf_subs deepcoder_taco deepcoder_pi; do
    python3 fetch.py --source $s --union "$UNION" --clusters "$CLUSTERS" --work "$WORK" --workers 4
  done
fi
if runs refs; then
  python3 build_refs.py --union "$UNION" --clusters "$CLUSTERS" --codewars-pool "$CW_POOL" --work "$WORK"
fi
declare -A LABEL
if runs cands; then
  PIDS=()
  for s in openr1 hardtests ccplus taco; do
    python3 stdio_cands.py $s --union "$UNION" --clusters "$CLUSTERS" --extract "$WORK/extract" \
      --taco-verified "$WORK/refs/taco_verified.jsonl.gz" --out-dir "$B" &
    PIDS+=($!); LABEL[$!]="stdio_cands.py $s"
  done
  wait_all cands
fi
if runs verify; then
  sandbox_urls
  n=${#URLS[@]}
  for pass in 1 2; do
    for s in openr1 hardtests ccplus taco; do
      PIDS=()
      for k in "${!URLS[@]}"; do
        log=$B/verify_$s.s${k}of$n.pass$pass.log
        CODE_EXEC_API_URL=${URLS[$k]} python3 -W ignore stdio_verify.py "$B/cands_$s.jsonl.gz" --out-dir "$B" \
          --workers "$WORKERS" --shard "$k/$n" > "$log" 2>&1 &
        PIDS+=($!); LABEL[$!]="stdio_verify.py $s shard $k/$n pass $pass, see $log"
      done
      wait_all verify
    done
  done
  for s in openr1 hardtests ccplus taco; do
    python3 stdio_verify.py "$B/cands_$s.jsonl.gz" --out-dir "$B" --check \
      || { echo "FAILED: rows of $s are still dropped by sandbox errors after 2 passes; rerun --from verify" >&2; exit 1; }
  done
fi
if runs fn; then
  sandbox_urls
  CODE_EXEC_API_URL=${URLS[0]} python3 -W ignore fn_build.py --leetcode-pool "$LC_POOL" --leetcode-refs "$LC_REFS" \
    --codewars-pool "$CW_POOL" --codewars-refs "$WORK/refs/codewars_refs.json" --out-dir "$B" --workers "$WORKERS"
  if n_err=$(grep -c ',sandbox_error' "$B/fn_dropped.csv"); then
    echo "FAILED: $n_err function rows still dropped by sandbox errors; rerun --from fn" >&2; exit 1
  fi
fi
if runs assemble; then
  args=(--build-dir "$B" --carry "$CARRY" --out-dir "$OUT")
  [ -n "$STATUS" ] && args+=(--status "$STATUS")
  if [ -n "$DEDUP" ]; then
    args+=(--hash legacy --order legacy --duplicates last --dedup-list "$DEDUP")
  fi
  python3 assemble.py "${args[@]}"
fi
if runs decontam; then
  python3 decontam.py "$OUT/train_pre_decontam.jsonl" --eval-items "$EVAL_ITEMS" --eval-hashes "$EVAL_HASHES" --out-dir "$OUT"
fi
if runs finalize; then
  python3 finalize.py --pre "$OUT/train_pre_decontam.jsonl" --drop "$OUT/decontam_drop.txt" --out "$OUT/train.jsonl"
fi
