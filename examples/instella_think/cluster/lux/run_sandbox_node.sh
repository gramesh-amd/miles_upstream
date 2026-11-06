#!/bin/bash
# The code-exec sandboxes (../../code_exec_server.py) on this node, in the foreground: the training
# reward instance on $CODE_EXEC_PORT and the benchmark instance on $CODE_BENCH_PORT. Runs inside a
# Slurm step (sandbox.sbatch, run.sbatch); once both answer /health, writes this node's name to
# $READY_FILE (default $CODE_EXEC_HOST_FILE, which ~/.instella-env.sh reads).
#
# Both servers run in the miles SIF for its Python 3.10 and numpy (the EvalPlus tests need it);
# the host's Python is 3.9. Health: curl --noproxy "*" http://<host>:<port>/<token>/health
set -uo pipefail
LUX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${LUX_DIR}/lux_env.sh"
INSTELLA_DIR="$(cd -- "${LUX_DIR}/../.." && pwd)"
export USER_HOME="${LUX_HOME}" INSTELLA_USER="${LUX_USER}"
source "${INSTELLA_DIR}/paths.sh"
READY_FILE="${READY_FILE:-${CODE_EXEC_HOST_FILE}}"
HOST="$(hostname -s)"
say() { echo "[$(date '+%F %T')] [${HOST}] $*"; }

NCPU="$(nproc)"
BENCH_WORKERS="${CODE_BENCH_WORKERS:-64}"
TRAIN_WORKERS="${CODE_EXEC_WORKERS:-$(( NCPU - BENCH_WORKERS - 8 ))}"
if [ ! -s "${CODE_EXEC_TOKEN_FILE}" ]; then
   ( umask 077; python3 -c 'import secrets; print(secrets.token_hex(16))' > "${CODE_EXEC_TOKEN_FILE}" )
fi
mkdir -p "${USER_HOME}/logs"

srv() {  # <port> <workers> <log> [extra server args...]
   local port="$1" workers="$2" log="$3"; shift 3
   apptainer exec --cleanenv --bind /lustre/orion --bind "${USER_HOME}" "${SIF}" \
      /opt/venv/bin/python -u "${INSTELLA_DIR}/code_exec_server.py" --port "${port}" --workers "${workers}" \
      --token-file "${CODE_EXEC_TOKEN_FILE}" "$@" >> "${log}" 2>&1 < /dev/null &
}
say "nproc=${NCPU}: train ${TRAIN_WORKERS} workers on :${CODE_EXEC_PORT}, bench ${BENCH_WORKERS} on :${CODE_BENCH_PORT}"
srv "${CODE_EXEC_PORT}" "${TRAIN_WORKERS}" "${USER_HOME}/logs/code-exec-${HOST}.log"
srv "${CODE_BENCH_PORT}" "${BENCH_WORKERS}" "${USER_HOME}/logs/code-exec-bench-${HOST}.log" \
   --python /opt/venv/bin/python --max-tests 128 --max-timeout 20 --mem-mb 4096 \
   --tests-dir "${CODE_BENCH_DIR}/tests" --scratch "/tmp/code-exec-bench-$(id -u)"

TOK="$(tr -d '[:space:]' < "${CODE_EXEC_TOKEN_FILE}")"
for port in "${CODE_EXEC_PORT}" "${CODE_BENCH_PORT}"; do
   for _ in $(seq 1 150); do
      curl -sf --noproxy '*' "http://${HOST}:${port}/${TOK}/health" >/dev/null && break
      sleep 2
   done
   h="$(curl -sf --noproxy '*' "http://${HOST}:${port}/${TOK}/health")" \
      || { say "FATAL: :${port} never answered /health; see ${USER_HOME}/logs/code-exec*-${HOST}.log"; exit 1; }
   say ":${port} health: ${h}"
done
echo "${HOST}" > "${READY_FILE}"
[ -z "${BENCH_READY_FILE:-}" ] || echo "${HOST}" > "${BENCH_READY_FILE}"
say "sandbox host ${HOST} written to ${READY_FILE} ${BENCH_READY_FILE:-}"
wait
