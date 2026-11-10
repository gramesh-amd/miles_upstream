#!/bin/bash
# One GPU node of a prefilter.sbatch job: starts the miles container (as run_gpu_node.sh does) and,
# inside it, sets the container up, serves the base checkpoint with sglang (prefilter.sh serve) and
# filters this node's shard in the foreground (prefilter.sh filter), retrying until it completes.
# Everything stays inside this one `apptainer exec`: under fakeroot a process outliving it hangs.
set -uo pipefail
source "${LUX_DIR:?}/lux_env.sh"
: "${JOB_DIR:?}" "${SANDBOX_NODES:?}"
HOST="$(hostname -s)"
NL="/mnt/bb/${SLURM_JOB_ID}"
INST="prefilter-${SLURM_JOB_ID}"
SHARD="$(printf %02d $(( ${SHARD_BASE:-0} + SLURM_PROCID )))"
exec > >(tee -a "${JOB_DIR}/node-${HOST}.log") 2>&1
say() { echo "[$(date '+%F %T')] [${HOST}] $*"; }
mkdir -p "${NL}"/{overlay,tmp,triton,cache}

cat > "${NL}/container.env" <<EOF
HOME=/root
USER_HOME=${LUX_HOME}
INSTELLA_USER=${LUX_USER}
HF_HOME=${LUX_ROOT}/hf
TRITON_CACHE_DIR=${NL}/triton
XDG_CACHE_HOME=${NL}/cache
TMPDIR=${NL}/tmp
JOB_DIR=${JOB_DIR}
SHARD=${SHARD}
SANDBOX_INDEX=$(( SLURM_PROCID % SANDBOX_NODES ))
CONCURRENCY=${CONCURRENCY:-64}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-512}
EOF
export APPTAINER_TMPDIR="${NL}/tmp"
say "starting container for shard ${SHARD}"
apptainer instance start --cleanenv --fakeroot --overlay "${NL}/overlay" --no-home \
   --bind /lustre/orion --bind "${LUX_HOME}" --bind "${NL}" --env-file "${NL}/container.env" \
   "${SIF}" "${INST}" || { say "FATAL: instance start failed"; exit 1; }
apptainer exec --cleanenv --env-file "${NL}/container.env" "instance://${INST}" bash -c '
   ulimit -c 0   # see eval_in_container.sh: a killed sglang dumps ~2.4 GB per GPU process into the home
   source /opt/venv/bin/activate
   cd "${USER_HOME}/miles" || exit 1
   I=examples/instella_think
   source "${I}/paths.sh"
   say() { echo "[$(date "+%F %T")] [$(hostname -s)] $*"; }
   bash "${I}/cluster/env_setup.sh" || { say "FATAL: env_setup.sh failed"; exit 1; }
   until [ -s "${JOB_DIR}/sandbox${SANDBOX_INDEX}.ready" ]; do sleep 5; done
   TOK="$(tr -d "[:space:]" < "${CODE_EXEC_TOKEN_FILE}")"
   export CODE_EXEC_API_URL="http://$(cat "${JOB_DIR}/sandbox${SANDBOX_INDEX}.ready"):${CODE_EXEC_PORT}/${TOK}"
   export INPUT="${DATA_ROOT}/prefilter/shards/shard_${SHARD}.jsonl" OUTPUT="${DATA_ROOT}/prefilter/shards/shard_${SHARD}.filtered.jsonl"
   [ -s "${INPUT}" ] || { say "FATAL: no ${INPUT}; run prefilter_shards.py split first"; exit 1; }
   [ -f "${OUTPUT}.stats.json" ] && { say "shard ${SHARD} already done"; exit 0; }
   # (Re)starts sglang when it does not answer. At long contexts and high load the server can die
   # with HSA_STATUS_ERROR_EXCEPTION (a GPU memory fault): all 14 nodes at 512 running requests,
   # 2 of 14 after 2.5 h at 256 (2026-11-09/10); a dead server is restarted, not waited on.
   ensure_server() {
      curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null && return 0
      say "starting sglang"
      bash "${I}/prefilter.sh" stop >/dev/null 2>&1
      bash "${I}/prefilter.sh" serve
      for _ in $(seq 1 180); do curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null && return 0; sleep 10; done
      return 1
   }
   for attempt in 1 2 3 4 5 6 7 8; do
      ensure_server || { say "sglang never became healthy (attempt ${attempt})"; continue; }
      say "sglang healthy; filtering shard ${SHARD} (concurrency ${CONCURRENCY}, sandbox ${CODE_EXEC_API_URL%/*})"
      SERVER_WAIT=300 FOREGROUND=1 bash "${I}/prefilter.sh" filter && break
      say "filter exited non-zero (attempt ${attempt}); resuming from the cache"
      sleep 30
   done
   bash "${I}/prefilter.sh" stop
   [ -f "${OUTPUT}.stats.json" ] && say "shard ${SHARD} done" || { say "shard ${SHARD} NOT finished"; exit 1; }
'
rc=$?
say "container session ended (rc=${rc})"
apptainer instance stop "${INST}" >/dev/null 2>&1
exit "${rc}"
