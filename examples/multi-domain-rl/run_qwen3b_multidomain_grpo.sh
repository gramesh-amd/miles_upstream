#!/bin/bash
# =============================================================================
# Multi-Domain RL: Qwen2.5-3B-Instruct
#
# Multi-domain GRPO training across Math, Code, and Instruction Following.
# Domains: Math (MATH L1-3) + Code (KodCode 10K) + IF (IFBench 1-2 constraints)
#
# Key design choices:
#   1. Mean-only advantage normalization (no std division — natural curriculum)
#   2. Domain-balanced batching (round-robin interleaved data, shuffle OFF)
#   3. 24 prompts/rollout × 16 samples = 384 samples, 500 rollouts
#   4. beta=0.0 (no KL penalty)
#   5. LR=1e-6 constant, weight-decay=0.1
#   6. response_len=2048 (sufficient for instruct math/code/IF)
#   7. Token-level loss (--calculate-per-token-loss) — DAPO-style, each token weighted equally
#   8. Active sampling with prompt retirement (>93.75% pass rate)
#   9. Eval at step 0 (baseline), then every 10 steps
#   10. KodCode (10K diverse, 7.7 tests/problem → rich partial credit)
#   11. entropy-coef=0 (standard GRPO, all Miles examples use 0)
#   12. eps-clip-high=0.28 (DAPO asymmetric clip for exploration)
#   13. Constant LR 1e-6 (matches all RL literature: MGS, DAPO, Omni-Thinker)
#
# GPU layout: 2 GPUs for Megatron training, 6 GPUs for SGLang rollout
# =============================================================================

set -ex

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

export PYTHONUNBUFFERED=1

# =================================================================
# GPU and environment (MI325X)
# =================================================================
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_TIMEOUT=3600
export RCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_ENABLE_MONITORING=0

export RAY_grpc_client_keepalive_time_ms=86400000
export RAY_grpc_client_keepalive_timeout_ms=86400000
export RAY_grpc_keepalive_timeout_ms=86400000
export RAY_gcs_rpc_server_reconnect_timeout_s=86400

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=2048
export SGLANG_REQ_RUNNING_TIMEOUT=600

unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset SGLANG_MOE_PADDING 2>/dev/null || true
export SGLANG_USE_AITER=1
unset SGLANG_USE_ROCM700A 2>/dev/null || true
unset SGLANG_ROCM_FUSED_DECODE_MLA 2>/dev/null || true

export CODE_EXEC_TIMEOUT=10

# =================================================================
# Paths
# =================================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT="${SCRIPT_DIR}/../.."

MEGATRON_PATH="${MEGATRON_PATH:-/home/goramesh/Primus-Instella/third_party/Megatron-LM}"
export PYTHONPATH="${MEGATRON_PATH}:${MILES_ROOT}:${PYTHONPATH:-}"

HF_CHECKPOINT="/wekafs/gramesh/checkpoint/Qwen2.5-3B-Instruct"
HF_SGLANG="${HF_CHECKPOINT}"
MEGATRON_CKPT="/wekafs/gramesh/checkpoint/Qwen2.5-3B-Instruct_torch_dist"

DATA_DIR="/wekafs/gramesh/olmo3_rl_research/data"
OUTPUT_DIR="/wekafs/gramesh/olmo3_rl_research/output/multidomain_qwen25_3b"

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"

# =================================================================
# Prepare balanced training data (round-robin interleaved)
# =================================================================
TRAIN_DATA="${DATA_DIR}/gdpo_balanced_kodcode_4k_train.jsonl"
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "[ERROR] Balanced training data not found at ${TRAIN_DATA}"
    echo "Run the data prep script first."
    exit 1
fi

# =================================================================
# torch_memory_saver LD_PRELOAD
# =================================================================
TMS_SO=$(find /tmp/torch_memory_saver -maxdepth 1 -name '*preload*.so' 2>/dev/null | head -1)
if [ -z "${TMS_SO}" ]; then
    TMS_SO=$(python3 -c "from torch_memory_saver.utils import get_binary_path_from_package; print(get_binary_path_from_package('torch_memory_saver_hook_mode_preload'))" 2>/dev/null || true)
fi
if [ -n "${TMS_SO}" ]; then
    echo "Setting LD_PRELOAD=${TMS_SO}"
    export LD_PRELOAD="${TMS_SO}${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "WARNING: torch_memory_saver preload .so not found!"
fi

# =================================================================
# Verify checkpoints exist
# =================================================================
if [ ! -d "${HF_SGLANG}" ]; then
    echo "[ERROR] HF checkpoint not found at ${HF_SGLANG}"
    exit 1
fi
if [ ! -f "${MEGATRON_CKPT}/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] Megatron checkpoint not found at ${MEGATRON_CKPT}"
    exit 1
fi
echo "=== Checkpoints ready ==="

# =================================================================
# Launch Ray
# =================================================================
echo "=== Starting Ray ==="
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats
sleep 5
ray status

# =================================================================
# Model architecture — Qwen2.5-3B
# =================================================================
MODEL_ARGS=(
    --swiglu
    --num-layers 36
    --hidden-size 2048
    --ffn-hidden-size 11008
    --num-attention-heads 16
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --normalization "RMSNorm"
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --group-query-attention
    --num-query-groups 2
    --vocab-size 151936
    --max-position-embeddings 32768
)

# =================================================================
# Training configuration — Multi-Domain RL
# =================================================================

CKPT_ARGS=(
    --hf-checkpoint ${HF_SGLANG}
    --ref-load ${MEGATRON_CKPT}
    --load ${OUTPUT_DIR}/checkpoints
    --save ${OUTPUT_DIR}/checkpoints
    --save-interval 50
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${HF_CHECKPOINT}
)

ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template

    --custom-rm-path examples.multi-domain-rl.reward_model.batched_reward

    --num-rollout 500
    --rollout-batch-size 24
    --n-samples-per-prompt 16
    --rollout-max-response-len 2048
    --rollout-temperature 1.0

    --global-batch-size 64
    --balance-data
)

ACTIVE_SAMPLING_ARGS=(
    --dynamic-sampling-filter-path
        examples.multi-domain-rl.dynamic_sampling_filter.check_reward_nonzero_std_and_retirement
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --disable-grpo-std-normalization
    --kl-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0
    --use-tis
    --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 3

    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --seq-length 4096
    --max-position-embeddings 32768
    --bf16

    --distributed-timeout-minutes 480
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.80
    --sglang-chunked-prefill-size 4096
    --sglang-max-running-requests 128
    --sglang-disable-custom-all-reduce
    --sglang-disable-radix-cache
    --sglang-attention-backend triton
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-config ${SCRIPT_DIR}/eval_config.yaml
    --eval-function-path examples.multi-domain-rl.eval_with_flush.generate_rollout
    --log-passrate
)

LOGGING_ARGS=(
    --custom-rollout-log-function-path
        examples.multi-domain-rl.per_domain_logger.log_per_domain_metrics
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project olmo-rlvr-poc
    --wandb-group multidomain-qwen25-3b
    --wandb-mode ${WANDB_MODE:-online}
    --wandb-dir ${OUTPUT_DIR}/wandb
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --no-gradient-accumulation-fusion
    --no-check-for-nan-in-loss-and-grad
    --update-weight-buffer-size $((4 * 1024 * 1024 * 1024))
    --use-fault-tolerance
    --rollout-health-check-timeout 600
    --dump-details ${OUTPUT_DIR}/dump_details
    --make-vocab-size-divisible-by 1
)

# =================================================================
# Launch training
# =================================================================
cd "${MILES_ROOT}"
echo "=== Launching Multi-Domain RL: Qwen2.5-3B-Instruct ==="
echo "  Model:             Qwen2.5-3B-Instruct"
echo "  HF checkpoint:     ${HF_CHECKPOINT}"
echo "  Megatron ckpt:     ${MEGATRON_CKPT}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Dataset:           ${TRAIN_DATA}"
echo "  Domains:           Math (MATH L1-3) + Code (KodCode 10K) + IF (IFBench)"
echo "  Algorithm:         GRPO mean-only (no std norm) + token-level loss"
echo "  LR:                1e-6 (constant)"
echo "  Weight decay:      0.1"
echo "  KL (beta):         0.0"
echo "  Samples/prompt:    16"
echo "  Batch size:        24 prompts x 16 = 384 samples (6 train steps/rollout)"
echo "  Loss reducer:      calculate-per-token-loss (DAPO-style token-mean)"
echo "  Response len:      2048"
echo "  Temperature:       1.0"
echo "  Active sampling:   ON (retire >93.75% pass rate)"
echo "  Eval:              step 0 + every 10 steps (4 samples, temp=0.6)"
echo "  GPU layout:        2 train (TP=1, DP=2) + 6 rollout (6 engines, TP=1)"

python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-num-gpus 6 \
    --num-gpus-per-node 8 \
    --update-weights-interval 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ACTIVE_SAMPLING_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MISC_ARGS[@]}"
