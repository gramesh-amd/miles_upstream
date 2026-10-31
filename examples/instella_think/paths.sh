# Per-user locations for the Instella recipes. Sourced by run_async.sh, run_code.sh,
# run_if.sh and the cluster/ scripts; every value can be overridden from the environment.
#
# These scripts usually run as root inside the training container, where HOME is /root.
# The user is therefore taken from the owner of this checkout (node_setup.sh creates that
# user in the container with the host uid), not from $HOME or $USER.
#
# Per-user overrides go in ~/.instella-env.sh (not in git), e.g.
#   NFS_ROOT=/shared_nfs/<your nfs dir>
#   CODE_EXEC_HOST=<cpu host running code_exec_server.py>

_INSTELLA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INSTELLA_USER="${INSTELLA_USER:-$(stat -c %U "${_INSTELLA_DIR}")}"
USER_HOME="${USER_HOME:-$(getent passwd "${INSTELLA_USER}" | cut -d: -f6)}"
USER_HOME="${USER_HOME:-${HOME}}"

INSTELLA_ENV="${INSTELLA_ENV:-${USER_HOME}/.instella-env.sh}"
# shellcheck disable=SC1090
[ -f "${INSTELLA_ENV}" ] && source "${INSTELLA_ENV}"

# Shared NFS area for logs, rollout dumps, wandb files and small shared assets.
NFS_ROOT="${NFS_ROOT:-/shared_nfs/${INSTELLA_USER}}"
# Holds checkpoints/ (HF model, torch_dist conversion, run outputs) and data/ (JSONLs).
ASSET_ROOT="${ASSET_ROOT:-${USER_HOME}}"
DATA_ROOT="${DATA_ROOT:-${ASSET_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-${ASSET_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${NFS_ROOT}/rl-logs}"
DUMP_ROOT="${DUMP_ROOT:-${NFS_ROOT}/rollout-dumps}"
WANDB_LOCAL_DIR="${WANDB_LOCAL_DIR:-${NFS_ROOT}/wandb-async}"

# Primus-Instella-Internal checkout: FarSkip Megatron (investigate_megatron.sh) and the
# sglang overlay (setup_farskip.sh).
PRIMUS_ROOT="${PRIMUS_ROOT:-${USER_HOME}/Primus-Instella-Internal}"
MEGATRON_PATH="${MEGATRON_PATH:-${PRIMUS_ROOT}/training/third_party/Megatron-LM}"

# wandb credentials: an absolute netrc path, because HOME is /root in the container.
NETRC_PATH="${NETRC:-${USER_HOME}/.netrc}"

# IF reward assets (setup_if_deps.sh seeds them).
NLTK_DATA="${NLTK_DATA:-${NFS_ROOT}/nltk_data}"
IF_IFBENCH_DIR="${IF_IFBENCH_DIR:-${NFS_ROOT}/IFBench}"

# Code RL sandbox (cluster/code_exec.sh).
CODE_EXEC_TOKEN_FILE="${CODE_EXEC_TOKEN_FILE:-${USER_HOME}/.code-exec-token}"
CODE_EXEC_PORT="${CODE_EXEC_PORT:-18950}"
# Benchmark evals (prepare_code_benchmarks.py output; the sandbox host must read it) and
# their sandbox instance (`cluster/code_exec.sh start bench`).
CODE_BENCH_DIR="${CODE_BENCH_DIR:-${DATA_ROOT}/code-bench}"
CODE_BENCH_PORT="${CODE_BENCH_PORT:-18951}"
CODE_BENCH_VENV="${CODE_BENCH_VENV:-${USER_HOME}/.code-exec-venv}"

export INSTELLA_USER USER_HOME NFS_ROOT ASSET_ROOT DATA_ROOT CKPT_ROOT LOG_DIR DUMP_ROOT \
       WANDB_LOCAL_DIR PRIMUS_ROOT MEGATRON_PATH NETRC_PATH NLTK_DATA IF_IFBENCH_DIR \
       CODE_EXEC_TOKEN_FILE CODE_EXEC_PORT CODE_BENCH_DIR CODE_BENCH_PORT CODE_BENCH_VENV
