#!/bin/bash
# Offline difficulty prefilter (difficulty_filter.py) for an RL prompt set. Runs INSIDE a node's
# container, on a node that is not training (it takes all 8 GPUs for one sglang server).
#
#   prefilter.sh serve     # start sglang on the checkpoint RL starts from (detached)
#   prefilter.sh filter    # start the filter against it (detached, resumable)
#   prefilter.sh status    # server health, filter progress, log tails
#   prefilter.sh stop      # stop both
#
# From the login node every subcommand goes through on_node.sh, which takes VAR=VALUE overrides:
#   cluster/on_node.sh <JOBID> <NODE> prefilter.sh serve
#   cluster/on_node.sh <JOBID> <NODE> prefilter.sh LIMIT=200 filter      # pilot first
#   cluster/on_node.sh <JOBID> <NODE> prefilter.sh filter                # then the full set
# Both start detached and return, because Spur allows one interactive step per node at a time.
#
# Settings (environment):
#   TASK              code (default) | math
#   INPUT             code: $CODE_TRAIN (default $DATA_ROOT/code-train/dolci-code-train.jsonl)
#                     math: required, the run's PROMPT_DATA
#   OUTPUT            default $DATA_ROOT/prefilter/<input name>.filtered.jsonl; train on it
#                     with PROMPT_DATA=<OUTPUT> (run_code.sh: CODE_TRAIN=<OUTPUT>)
#   HF_CKPT           checkpoint RL starts from (default $CKPT_ROOT/Instella-MoE-16B-A3B-DPO,
#                     same as run_async.sh). Filtering against any other checkpoint measures
#                     the wrong model.
#   MAX_NEW_TOKENS    default $ROLLOUT_MAX_RESPONSE_LEN or 32768. MUST equal the run's value.
#   CONTEXT_LEN       default 32768, as in run_async.sh
#   N / MAX_PASS_RATE 8 / 0.625 (OLMo 3). DROP_UNSOLVED=1 also drops 0/N prompts.
#   TEMPERATURE       default 1.0 (run_async.sh's --rollout-temperature)
#   RM_TYPE           math only, default deepscaler (the run's --rm-type)
#   CONCURRENCY       prompts in flight, default 16 (x N = 128 sequences)
#   LIMIT             first N prompts only (pilot)
#   GPUS / PORT       default 8 / 30000
#   FARSKIP_SETUP     script that applies the FarSkip sglang patches before serving. Empty by
#                     default, because cluster/env_setup.sh already ran setup_farskip.sh in
#                     this container; set it for a container that has not been set up.
#   CODE_EXEC_API_URL code only; resolved from CODE_EXEC_HOST + the token file like run_code.sh.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/paths.sh"
# shellcheck disable=SC1091
source /opt/venv/bin/activate 2>/dev/null || true

CMD="${1:-status}"
HOST="$(hostname -s)"
TASK="${TASK:-code}"
PORT="${PORT:-30000}"
GPUS="${GPUS:-8}"
URL="http://127.0.0.1:${PORT}"
HF_CKPT="${HF_CKPT:-${CKPT_ROOT}/Instella-MoE-16B-A3B-DPO}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${ROLLOUT_MAX_RESPONSE_LEN:-32768}}"
STATE_DIR="${LOG_DIR}/prefilter"
SERVER_PID="${STATE_DIR}/sglang-${HOST}-${PORT}.pid"
FILTER_PID="${STATE_DIR}/filter-${HOST}.pid"
mkdir -p "${STATE_DIR}"

case "${TASK}" in
   code) INPUT="${INPUT:-${CODE_TRAIN:-${DATA_ROOT}/code-train/dolci-code-train.jsonl}}" ;;
   math) INPUT="${INPUT:-${PROMPT_DATA:-}}" ;;
   *) echo "FATAL: TASK must be code or math, got '${TASK}'" >&2; exit 1 ;;
esac
_base="$(basename "${INPUT%.jsonl}")"
OUTPUT="${OUTPUT:-${DATA_ROOT}/prefilter/${_base}.filtered.jsonl}"

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
healthy() { curl -sf -m 5 "${URL}/health" >/dev/null 2>&1; }

case "${CMD}" in
serve)
   if alive "${SERVER_PID}"; then echo "sglang already running (pid $(cat "${SERVER_PID}"))"; exit 0; fi
   [ -f "${HF_CKPT}/config.json" ] || { echo "FATAL: no HF checkpoint at ${HF_CKPT}" >&2; exit 1; }
   if [ -n "${FARSKIP_SETUP:-}" ]; then
      bash "${FARSKIP_SETUP}" || { echo "FATAL: ${FARSKIP_SETUP} failed" >&2; exit 1; }
   fi
   # Same engine settings as run_async.sh's rollout engines (SGLANG_ARGS and RUNTIME_ENV_JSON),
   # so the filter samples the model the run will sample. aiter=1 is required for correct
   # output with the FarSkip overlay (see README "Known issue: aiter").
   export FARSKIP_REFERENCE_DECODER_LAYER=1
   export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
   export SGLANG_ROCM_FUSED_DECODE_MLA=0
   export GPU_MAX_HW_QUEUES=8
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="${CONTEXT_LEN}"
   EXTRA=()
   if [ "${CONTEXT_LEN}" -gt 32768 ]; then
      export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
      EXTRA+=(--context-length "${CONTEXT_LEN}")
   fi
   LOG="${STATE_DIR}/sglang-${HOST}-$(date +%Y%m%d_%H%M%S).log"
   setsid nohup python3 -m sglang.launch_server \
      --model-path "${HF_CKPT}" --trust-remote-code \
      --tp-size "${GPUS}" --ep-size "${GPUS}" --dp-size "${GPUS}" --enable-dp-attention \
      --attention-backend "${SGLANG_ATTENTION_BACKEND:-triton}" \
      --disable-radix-cache --disable-shared-experts-fusion \
      --enable-fp32-lm-head --allow-auto-truncate \
      --mem-fraction-static "${SGLANG_MEM_FRACTION:-0.60}" \
      --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-128}" \
      --cuda-graph-max-bs-decode "${SGLANG_CUDA_GRAPH_MAX_BS:-64}" \
      "${EXTRA[@]}" --host 127.0.0.1 --port "${PORT}" \
      > "${LOG}" 2>&1 < /dev/null &
   echo $! > "${SERVER_PID}"
   echo "sglang starting: pid $(cat "${SERVER_PID}"), ${HF_CKPT} on ${GPUS} GPUs, port ${PORT}"
   echo "  log ${LOG}"
   echo "  ready when 'prefilter.sh status' shows health OK (model load takes a few minutes)"
   ;;

filter)
   if alive "${FILTER_PID}"; then echo "filter already running (pid $(cat "${FILTER_PID}"))"; exit 0; fi
   [ -s "${INPUT}" ] || { echo "FATAL: INPUT '${INPUT}' missing or empty" >&2; exit 1; }
   FARGS=(--task "${TASK}" --input "${INPUT}" --output "${OUTPUT}" --url "${URL}"
          --tokenizer "${HF_CKPT}" --max-new-tokens "${MAX_NEW_TOKENS}"
          --n "${N:-8}" --max-pass-rate "${MAX_PASS_RATE:-0.625}"
          --temperature "${TEMPERATURE:-1.0}" --concurrency "${CONCURRENCY:-16}"
          --limit "${LIMIT:-0}" --server-wait "${SERVER_WAIT:-1800}")
   [ "${DROP_UNSOLVED:-0}" = "1" ] && FARGS+=(--drop-unsolved)
   [ "${RESET_CACHE:-0}" = "1" ] && FARGS+=(--reset-cache)
   if [ "${TASK}" = "code" ]; then
      if [ -z "${CODE_EXEC_API_URL:-}" ]; then
         _tok="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}" 2>/dev/null)"
         [ -n "${_tok}" ] && [ -n "${CODE_EXEC_HOST:-}" ] || {
            echo "FATAL: set CODE_EXEC_API_URL, or CODE_EXEC_HOST plus ${CODE_EXEC_TOKEN_FILE} (cluster/code_exec.sh start)" >&2; exit 1; }
         export CODE_EXEC_API_URL="http://${CODE_EXEC_HOST}:${CODE_EXEC_PORT}/${_tok}"
      fi
      # Same reward gates as run_code.sh, which the filter's grading reads from the env.
      export CODE_REQUIRE_THINK="${CODE_REQUIRE_THINK:-1}" CODE_MAX_TESTS="${CODE_MAX_TESTS:-32}" \
             CODE_EXEC_TIMEOUT="${CODE_EXEC_TIMEOUT:-5}"
   else
      FARGS+=(--rm-type "${RM_TYPE:-deepscaler}")
   fi
   export PYTHONPATH="${REPO_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"
   mkdir -p "$(dirname "${OUTPUT}")"
   LOG="${STATE_DIR}/filter-${TASK}-${_base}-$(date +%Y%m%d_%H%M%S).log"
   if [ "${FOREGROUND:-0}" = "1" ]; then
      exec python3 -u "${SCRIPT_DIR}/difficulty_filter.py" "${FARGS[@]}"
   fi
   setsid nohup python3 -u "${SCRIPT_DIR}/difficulty_filter.py" "${FARGS[@]}" > "${LOG}" 2>&1 < /dev/null &
   echo $! > "${FILTER_PID}"
   echo "filter started: pid $(cat "${FILTER_PID}")"
   echo "  ${TASK}: ${INPUT} -> ${OUTPUT}"
   echo "  max_new_tokens ${MAX_NEW_TOKENS}, N ${N:-8}, drop pass rate > ${MAX_PASS_RATE:-0.625}, limit ${LIMIT:-0}"
   echo "  log ${LOG}"
   ;;

status)
   if alive "${SERVER_PID}"; then
      echo "sglang: pid $(cat "${SERVER_PID}"), health $(healthy && echo OK || echo 'not ready')"
   else
      echo "sglang: not running"
   fi
   if alive "${FILTER_PID}"; then echo "filter: pid $(cat "${FILTER_PID}")"; else echo "filter: not running"; fi
   CACHE="${OUTPUT}.cache.jsonl"
   if [ -f "${CACHE}" ] && [ -f "${INPUT}" ]; then
      echo "graded: $(( $(wc -l < "${CACHE}") - 1 )) / $(wc -l < "${INPUT}") prompts (${CACHE})"
   fi
   [ -f "${OUTPUT}.stats.json" ] && echo "done: ${OUTPUT}.stats.json"
   for pat in "sglang-${HOST}-*.log" "filter-${TASK}-${_base}-*.log"; do
      # shellcheck disable=SC2012
      f="$(ls -t "${STATE_DIR}"/${pat} 2>/dev/null | head -1)"
      [ -n "${f}" ] && { echo "--- ${f}"; tail -n 4 "${f}"; }
   done
   ;;

stop)
   for p in "${FILTER_PID}" "${SERVER_PID}"; do
      if alive "${p}"; then
         kill -- "-$(cat "${p}")" 2>/dev/null || kill "$(cat "${p}")"
         echo "stopped $(basename "${p}" .pid) (pid $(cat "${p}"))"
      fi
      rm -f "${p}"
   done
   ;;

*)
   sed -n '2,35p' "$0"; exit 1 ;;
esac
