#!/bin/bash
# Serve Instella-MoE with sglang + FarSkip for offline difficulty grading (math and IF prefilters).
# Runs inside the training container on one 8-GPU node.
#
#   MODEL=/workspace/models/Instella-MoE-16B-A3B-DPO CONTEXT_LENGTH=36864 PORT=30000 bash serve_grader.sh
#
# Same layout AMD validated for FarSkip inference: tp = ep = dp = 8 with dp-attention. With tp=1 the
# FarSkip layer takes a code path that gives wrong outputs.
set -x

MODEL=${MODEL:-/workspace/models/Instella-MoE-16B-A3B-DPO}
PORT=${PORT:-30000}
SIZE=${SIZE:-8}
FP32_LM_HEAD=${FP32_LM_HEAD-1}   # `-` not `:-`: an explicitly empty value serves a bf16 head
# FarSkip sglang overlay. Idempotent; MI325 bringup and setup_farskip.sh (MI355) already apply it, so an empty
# FARSKIP_APPLY skips this step. Default: the MI325 kit's copy if present.
_KIT_APPLY=/root/miles/examples/instella_think/mi325/sglang_port/apply_farskip.sh
FARSKIP_APPLY=${FARSKIP_APPLY-$([ -f "$_KIT_APPLY" ] && echo "$_KIT_APPLY")}

# MATCH_TRAINING=1 copies the rollout engines' scheduler settings (admission, retraction, auto-truncate).
# These change how often requests are retracted and re-prefilled, not the numerics.
if [ "${MATCH_TRAINING:-0}" = "1" ]; then
    export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="${SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION:-32768}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
    MEM_FRACTION="${MEM_FRACTION:-0.6}"
    MAX_RUNNING="${MAX_RUNNING:-1024}"
    DECODE_STEPS="${DECODE_STEPS:-2}"
    GRAPH_MAX_BS="${GRAPH_MAX_BS:-128}"
    ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-1}"
fi
MEM_FRACTION="${MEM_FRACTION:-0.8}"

[ -z "$FARSKIP_APPLY" ] || bash "$FARSKIP_APPLY" || exit 1

export FARSKIP_REFERENCE_DECODER_LAYER=1
unset FARSKIP_OVERLAPPED_DECODER_LAYER
export SGLANG_USE_AITER=1          # routed scaling is applied inside the aiter MoE kernel; required
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export GPU_MAX_HW_QUEUES=8

# --enable-fp32-lm-head: training rollouts use it; a bf16 head makes sharpened post-RL models loop at T=0.6.
# --context-length 36864: room for a 32K response plus the prompt (YaRN covers positions past 32768).
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp-size "$SIZE" \
    --ep-size "$SIZE" \
    --dp-size "$SIZE" \
    --enable-dp-attention \
    --dtype bfloat16 \
    --attention-backend triton \
    --disable-radix-cache \
    ${DISABLE_CUDA_GRAPH:+--disable-cuda-graph} \
    --disable-shared-experts-fusion \
    ${FP32_LM_HEAD:+--enable-fp32-lm-head} \
    ${ALLOW_AUTO_TRUNCATE:+--allow-auto-truncate} \
    ${MAX_RUNNING:+--max-running-requests "$MAX_RUNNING"} \
    ${DECODE_STEPS:+--num-continuous-decode-steps "$DECODE_STEPS"} \
    ${GRAPH_MAX_BS:+--cuda-graph-max-bs-decode "$GRAPH_MAX_BS"} \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "${CONTEXT_LENGTH:-36864}" \
    --host 0.0.0.0 \
    --port "$PORT"
