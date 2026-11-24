#!/bin/bash
# Offline code benchmarks (eval_code_ckpt.py) for one checkpoint. Runs INSIDE a node's container,
# on a node that is not training (the server takes all its GPUs).
#
#   eval_code_ckpt.sh export   # torch_dist CKPT -> HF safetensors in HF (CPU only)
#   eval_code_ckpt.sh serve    # sglang on HF (detached)
#   eval_code_ckpt.sh status   # server health, eval progress, log tails
#   eval_code_ckpt.sh eval     # eval_code_ckpt.py against it (detached; FOREGROUND=1 to wait)
#   eval_code_ckpt.sh stop     # stop both
#
# From the login node, after `cluster/code_exec.sh start bench`:
#   cluster/on_node.sh <JOBID> <NODE> eval_code_ckpt.sh CKPT=<run dir>/iter_0000400 export
#   cluster/on_node.sh <JOBID> <NODE> eval_code_ckpt.sh HF=<run dir>/iter_0000400_hf serve
#   cluster/on_node.sh <JOBID> <NODE> eval_code_ckpt.sh HF=<run dir>/iter_0000400_hf eval
#
# Settings (environment):
#   CKPT          torch_dist iteration dir (export)
#   HF            HF checkpoint to serve and evaluate (default ${CKPT}_hf, else the DPO base)
#   BASE_HF       HF dir the export copies config, tokenizer and modeling code from
#                 (default $CKPT_ROOT/Instella-MoE-16B-A3B-DPO)
#   VOCAB_SIZE    lm_head / embedding rows kept by the export (default: BASE_HF's vocab_size)
#   FORCE=1       let the export replace an existing HF dir
#   TP / DP       default 1 / 8: one replica per GPU
#   PORT / CTX    default 30000 / 36864 (32768 new tokens plus the longest prompt)
#   OUT           default $LOG_DIR/code-evals/<basename of HF>
#   DATASETS      default eval_code_ckpt.py's (lcb_v3,lcb_v5,lcb_v6,humanevalplus,mbppplus)
#   N_SAMPLES     default 4
#   EVAL_ARGS     more eval_code_ckpt.py flags, e.g. "--limit 20" for a smoke test
#   CODE_BENCH_API_URL  resolved from CODE_EXEC_HOST + the token file like cluster/code_exec.sh
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/paths.sh"
# shellcheck disable=SC1091
source /opt/venv/bin/activate 2>/dev/null || true

CMD="${1:-status}"
HOST="$(hostname -s)"
CKPT="${CKPT:-}"
BASE_HF="${BASE_HF:-${CKPT_ROOT}/Instella-MoE-16B-A3B-DPO}"
HF="${HF:-${CKPT:+${CKPT%/}_hf}}"
HF="${HF:-${BASE_HF}}"
TP="${TP:-1}"
DP="${DP:-8}"
PORT="${PORT:-30000}"
CTX="${CTX:-36864}"
URL="http://127.0.0.1:${PORT}"
OUT="${OUT:-${LOG_DIR}/code-evals/$(basename "${HF%/}")}"
STATE_DIR="${LOG_DIR}/code-evals"
SERVER_PID="${STATE_DIR}/sglang-${HOST}-${PORT}.pid"
EVAL_PID="${STATE_DIR}/eval-${HOST}.pid"
mkdir -p "${STATE_DIR}"

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
healthy() { curl -sf -m 5 "${URL}/health" >/dev/null 2>&1; }

case "${CMD}" in
export)
   [ -f "${CKPT}/common.pt" ] || { echo "FATAL: CKPT='${CKPT}' is not a torch_dist iteration dir" >&2; exit 1; }
   [ -f "${BASE_HF}/config.json" ] || { echo "FATAL: no BASE_HF at ${BASE_HF}" >&2; exit 1; }
   [ "$(realpath -m "${HF}")" != "$(realpath -m "${BASE_HF}")" ] || { echo "FATAL: HF must not be BASE_HF" >&2; exit 1; }
   if [ -e "${HF}" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "FATAL: ${HF} exists (FORCE=1 replaces it)" >&2; exit 1
   fi
   VOCAB_SIZE="${VOCAB_SIZE:-$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["vocab_size"])' \
                               "${BASE_HF}/config.json")}"
   # --vocab-size: without it the converter trims to the checkpoint's own --vocab-size, which
   # model_args.sh sets to PADDED_VOCAB_SIZE; rows past config.json's vocab_size would be ids
   # sglang can sample but the tokenizer cannot decode.
   # --model-name: Instella is converted as DeepSeek-V3 (gated attention included).
   echo "export ${CKPT} -> ${HF} (vocab ${VOCAB_SIZE}, assets from ${BASE_HF})"
   rm -rf "${HF}.tmp"
   PYTHONPATH="${REPO_DIR}:${MEGATRON_PATH}:${PYTHONPATH:-}" python3 "${REPO_DIR}/tools/convert_torch_dist_to_hf.py" \
      --model-name deepseekv3 --input-dir "${CKPT}" --output-dir "${HF}.tmp" \
      --origin-hf-dir "${BASE_HF}" --vocab-size "${VOCAB_SIZE}" --force || exit 1
   rm -rf "${HF}" && mv "${HF}.tmp" "${HF}"
   du -sh "${HF}"
   ;;

serve)
   if alive "${SERVER_PID}"; then echo "sglang already running (pid $(cat "${SERVER_PID}"))"; exit 0; fi
   [ -f "${HF}/model.safetensors.index.json" ] || { echo "FATAL: no HF checkpoint at ${HF}" >&2; exit 1; }
   # The rollout engine's FarSkip / ROCm environment (run_async.sh RUNTIME_ENV_JSON).
   export FARSKIP_REFERENCE_DECODER_LAYER=1
   export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
   export SGLANG_ROCM_FUSED_DECODE_MLA=0
   export MLA_UNFUSED_ROPE="${MLA_UNFUSED_ROPE:-0}"
   export GPU_MAX_HW_QUEUES=8
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   LOG="${STATE_DIR}/sglang-${HOST}-$(date +%Y%m%d_%H%M%S).log"
   # The rollout's numerics (triton attention, no shared-expert fusion, fp32 lm_head). config.json
   # advertises 32768 positions; the rope already covers more, so raising max_position_embeddings
   # is all a 36864 context needs.
   setsid nohup python3 -m sglang.launch_server \
      --model-path "${HF}" --trust-remote-code \
      --tp-size "${TP}" --dp-size "${DP}" \
      --host 127.0.0.1 --port "${PORT}" \
      --context-length "${CTX}" \
      --json-model-override-args "{\"max_position_embeddings\": ${CTX}}" \
      --mem-fraction-static 0.78 \
      --max-running-requests 256 \
      --disable-radix-cache \
      --attention-backend triton \
      --disable-shared-experts-fusion \
      --enable-fp32-lm-head \
      > "${LOG}" 2>&1 < /dev/null &
   echo $! > "${SERVER_PID}"
   echo "sglang starting: pid $(cat "${SERVER_PID}"), ${HF}, tp ${TP} dp ${DP}, ctx ${CTX}, port ${PORT}"
   echo "  log ${LOG}"
   echo "  ready when 'eval_code_ckpt.sh status' shows health OK"
   ;;

eval)
   if alive "${EVAL_PID}"; then echo "eval already running (pid $(cat "${EVAL_PID}"))"; exit 0; fi
   healthy || { echo "FATAL: no healthy server at ${URL} (eval_code_ckpt.sh serve, then status)" >&2; exit 1; }
   if [ -z "${CODE_BENCH_API_URL:-}" ]; then
      _tok="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}" 2>/dev/null)"
      [ -n "${_tok}" ] && [ -n "${CODE_EXEC_HOST:-}" ] || {
         echo "FATAL: set CODE_BENCH_API_URL, or CODE_EXEC_HOST plus ${CODE_EXEC_TOKEN_FILE} (cluster/code_exec.sh start bench)" >&2; exit 1; }
      export CODE_BENCH_API_URL="http://${CODE_EXEC_HOST}:${CODE_BENCH_PORT}/${_tok}"
   fi
   ARGS=(--model "${HF}" --url "${URL}" --ctx "${CTX}" --bench-dir "${CODE_BENCH_DIR}"
         --n-samples "${N_SAMPLES:-4}" --out "${OUT}")
   [ -n "${DATASETS:-}" ] && ARGS+=(--datasets "${DATASETS}")
   # shellcheck disable=SC2206
   [ -n "${EVAL_ARGS:-}" ] && ARGS+=(${EVAL_ARGS})
   mkdir -p "${OUT}"
   if [ "${FOREGROUND:-0}" = "1" ]; then
      exec python3 -u "${SCRIPT_DIR}/eval_code_ckpt.py" "${ARGS[@]}"
   fi
   setsid nohup python3 -u "${SCRIPT_DIR}/eval_code_ckpt.py" "${ARGS[@]}" > "${OUT}/eval.log" 2>&1 < /dev/null &
   echo $! > "${EVAL_PID}"
   echo "eval started: pid $(cat "${EVAL_PID}"), ${HF} -> ${OUT}"
   echo "  log ${OUT}/eval.log"
   ;;

status)
   if alive "${SERVER_PID}"; then
      echo "sglang: pid $(cat "${SERVER_PID}"), health $(healthy && echo OK || echo 'not ready')"
   else
      echo "sglang: not running"
   fi
   if alive "${EVAL_PID}"; then echo "eval: pid $(cat "${EVAL_PID}")"; else echo "eval: not running"; fi
   [ -f "${OUT}/summary.json" ] && echo "done: ${OUT}/summary.json"
   # shellcheck disable=SC2012
   f="$(ls -t "${STATE_DIR}"/sglang-"${HOST}"-*.log 2>/dev/null | head -1)"
   [ -n "${f}" ] && { echo "--- ${f}"; tail -n 4 "${f}"; }
   [ -f "${OUT}/eval.log" ] && { echo "--- ${OUT}/eval.log"; tail -n 4 "${OUT}/eval.log"; }
   ;;

stop)
   for p in "${EVAL_PID}" "${SERVER_PID}"; do
      if alive "${p}"; then
         kill -- "-$(cat "${p}")" 2>/dev/null || kill "$(cat "${p}")"
         echo "stopped $(basename "${p}" .pid) (pid $(cat "${p}"))"
      fi
      rm -f "${p}"
   done
   ;;

*)
   sed -n '2,29p' "$0"; exit 1 ;;
esac
