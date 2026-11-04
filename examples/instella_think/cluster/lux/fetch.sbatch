#!/bin/bash
#SBATCH -J instella-fetch
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH -p batch
# One-off staging on a compute node (heavy network and Lustre writes stay off the login node):
# the SIF, the DPO checkpoint, the code train data and the benchmark sets. Idempotent: each step
# is skipped when its output exists. Submit from the login node:
#
#   sbatch -A ven136 -o ~/logs/%x-%j.out --export=ALL,LUX_DIR=$PWD/examples/instella_think/cluster/lux \
#     examples/instella_think/cluster/lux/fetch.sbatch
set -uo pipefail
source "${LUX_DIR:?submit with --export=ALL,LUX_DIR=<this dir>}/lux_env.sh"
INSTELLA_DIR="$(cd -- "${LUX_DIR}/../.." && pwd)"
NL="/mnt/bb/${SLURM_JOB_ID}"
say() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$(dirname "${SIF}")" "${LUX_ROOT}"/{checkpoints,data,hf}
# Jobs inherit the submitting shell's environment, and the login node sets no proxy.
export http_proxy="${LUX_PROXY}" https_proxy="${LUX_PROXY}" HTTP_PROXY="${LUX_PROXY}" HTTPS_PROXY="${LUX_PROXY}"
export APPTAINER_CACHEDIR="${NL}/apptainer/cache" APPTAINER_TMPDIR="${NL}/apptainer/tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# 1. SIF. Written to a temp name so an interrupted pull never looks complete.
if [ -s "${SIF}" ]; then
   say "SIF present: ${SIF}"
else
   say "pulling docker://${IMAGE} -> ${SIF}"
   apptainer pull --disable-cache "${SIF}.partial" "docker://${IMAGE}" && mv "${SIF}.partial" "${SIF}" \
      || { say "FATAL: pull failed"; exit 1; }
   say "SIF done: $(du -h "${SIF}" | cut -f1)"
fi

ctr() {
   apptainer exec --cleanenv --bind /lustre/orion --bind "${LUX_HOME}" \
      --env "http_proxy=${LUX_PROXY},https_proxy=${LUX_PROXY},HTTP_PROXY=${LUX_PROXY},HTTPS_PROXY=${LUX_PROXY}" \
      --env "HF_HOME=${LUX_ROOT}/hf,USER_HOME=${LUX_HOME},INSTELLA_USER=${LUX_USER}" \
      "${SIF}" bash -c "source /opt/venv/bin/activate 2>/dev/null; $1"
}

# 2. DPO checkpoint (public, ungated).
CK="${LUX_ROOT}/checkpoints/Instella-MoE-16B-A3B-DPO"
if [ -s "${CK}/model.safetensors.index.json" ] && [ -s "${CK}/model-00005-of-00006.safetensors" ]; then
   say "checkpoint present: ${CK}"
else
   say "downloading amd/Instella-MoE-16B-A3B-DPO"
   ctr "python3 -c \"from huggingface_hub import snapshot_download as s; print(s('amd/Instella-MoE-16B-A3B-DPO', local_dir='${CK}', max_workers=8))\"" \
      || { say "FATAL: checkpoint download failed"; exit 1; }
fi

# 3. Code train data and the OLMo 3 benchmark sets.
if [ -s "${LUX_ROOT}/data/code-train/dolci-code-train.jsonl" ]; then
   say "train data present"
else
   say "building train data"
   ctr "python3 ${INSTELLA_DIR}/prepare_code_data.py build --out-dir ${LUX_ROOT}/data" || say "ERROR: prepare_code_data failed"
fi
if [ -s "${LUX_ROOT}/data/code-bench/lcb_v3.jsonl" ]; then
   say "benchmarks present"
else
   say "building benchmarks"
   ctr "python3 ${INSTELLA_DIR}/prepare_code_benchmarks.py --out-dir ${LUX_ROOT}/data/code-bench" || say "ERROR: prepare_code_benchmarks failed"
fi

# 4. Probes for node_agent.sh: can a fakeroot container with a directory overlay write into the
# image (the sglang/mbridge/Megatron patches need it) and still see the GPUs?
say "probe: fakeroot + overlay"
mkdir -p "${NL}/probe-overlay"
apptainer exec --cleanenv --fakeroot --overlay "${NL}/probe-overlay" --bind /lustre/orion "${SIF}" bash -c '
   source /opt/venv/bin/activate 2>/dev/null
   echo "id: $(id)"; python3 -V
   echo "ls -ld /opt/venv /sgl-workspace /root/Megatron-LM:"; ls -ld /opt/venv /sgl-workspace /root/Megatron-LM 2>&1
   touch /opt/venv/.probe && echo "write /opt/venv: OK" || echo "write /opt/venv: FAILED"
   SG=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))" 2>&1); echo "sglang: $SG"
   touch "$SG/.probe" && echo "write sglang: OK" || echo "write sglang: FAILED"
   MB=$(python3 -c "import mbridge,os;print(os.path.dirname(mbridge.__file__))" 2>&1); echo "mbridge: $MB"
   touch "$MB/.probe" && echo "write mbridge: OK" || echo "write mbridge: FAILED"
   ls -l /dev/kfd; python3 -c "import torch; print(\"torch\", torch.__version__, \"gpus\", torch.cuda.device_count()); print(torch.ones(1, device=\"cuda\") + 1)"
   which ray sglang 2>&1; pip list 2>/dev/null | grep -i -E "^(ray|sglang|megatron|mbridge|torch|transformers|wandb) "
' 2>&1
say "fetch job done"
