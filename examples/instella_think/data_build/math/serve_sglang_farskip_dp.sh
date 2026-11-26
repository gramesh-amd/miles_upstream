#!/bin/bash
# C1 (see DIAGNOSIS.md): serve in AMD's *validated* FarSkip configuration.
#
# Mirrors inference/instella_moe_scripts/instella_moe_sglang_lm_eval.sh:
# tp=ep=dp=8 with dp-attention. This matters because the reference FarSkip layer
# branches on it —
#
#   farskip_already_scattered = self.reference_farskip and dp_attn_active
#   if not should_allreduce_fusion and not farskip_already_scattered:
#       hidden_states, residual = layer_communicator.postprocess_layer(...)
#
# With dp-attention ON, postprocess_layer is skipped. Our tp=1 run calls it, on a
# (hidden_states, residual) pair FarSkip has repurposed as (full, no_routed) —
# a path AMD likely never exercised. If output is correct here, the port is fine
# and the earlier tp=1 wrongness is a launch-config issue.
set -x

MODEL=${MODEL:-/workspace/models/Instella-MoE-16B-A3B-Think}
PORT=${PORT:-30000}
SIZE=${SIZE:-8}
FP32_LM_HEAD=${FP32_LM_HEAD-1}   # `-` not `:-`: an explicitly empty value disables it

# MATCH_TRAINING=1 reproduces the admission/retraction regime the miles rollout
# engines ran under. These are NOT numerics flags -- a full diff of 506 resolved
# ServerArgs found no numeric difference between the two configs -- but they
# govern how aggressively the scheduler admits work and therefore how often it
# RETRACTS. With --disable-radix-cache every retracted request is re-prefilled
# mid-generation, so an over-admitting scheduler can plausibly produce the
# longer, loopier offline output we measured (32.0% truncation and 12.5%
# repetition offline vs 18.1% and 3.1% in training on identical weights).
# The estimator is the big one: the container default of 4096 badly
# under-estimates the KV a 32768-token reasoning trace needs.
if [ "${MATCH_TRAINING:-0}" = "1" ]; then
    export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="${SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION:-32768}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
    MEM_FRACTION="${MEM_FRACTION:-0.6}"
    MAX_RUNNING="${MAX_RUNNING:-1024}"
    DECODE_STEPS="${DECODE_STEPS:-2}"
    GRAPH_MAX_BS="${GRAPH_MAX_BS:-128}"
    # Load-bearing with CONTEXT_LENGTH=32768. miles ran allow_auto_truncate=True,
    # so a request asking for 32768 completion tokens against a 32768 context was
    # CLAMPED to (32768 - prompt_len) -- those are the 30k+ "Truncating
    # max_new_tokens" lines in the training log. Without this flag sglang instead
    # REJECTS the request ("Requested token count exceeds ... 32917 total"), which
    # fails every sample. Note the earlier offline runs used context 36864, where
    # 32768 + prompt still fits, so they got a slightly LARGER budget than
    # training ever did.
    ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-1}"
fi
MEM_FRACTION="${MEM_FRACTION:-0.8}"

bash /workspace/sglang_port/apply_farskip.sh || exit 1

export FARSKIP_REFERENCE_DECODER_LAYER=1
unset FARSKIP_OVERLAPPED_DECODER_LAYER

# Required: the routed scaling factor is applied inside the aiter MoE kernel
# (DIAGNOSIS.md R1 — disabling aiter breaks the expert path).
export SGLANG_USE_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export GPU_MAX_HW_QUEUES=8

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
# FP32_LM_HEAD=1 (default below) adds --enable-fp32-lm-head; set FP32_LM_HEAD=
# empty to serve with a bf16 head instead, which is what an A/B of the two
# precisions needs.
# --enable-fp32-lm-head: miles launches its rollout engines with this on
# (sglang_enable_fp32_lm_head=True in the run log). Without it the logits come
# out in bf16, whose spacing near the top of a post-RL (sharpened) distribution
# is coarse enough to lock the sampler into a repeating cycle at T=0.6 -- an
# offline-only degeneration that does not exist in the in-training eval.
# --context-length: without it sglang falls back to the config's
# max_position_embeddings (32768), which cannot hold a 32768-token response plus
# a prompt. 36864 matches run_rl_instella_math.sh so the difficulty filter samples
# under the same budget the RL run will. Exceeding max_position_embeddings is
# intended: YaRN factor 40 over original_max_position_embeddings 4096 builds a
# 163840-position rope table, so nothing wraps.
