# Instella-MoE 16B FarSkip (gated-MLA MoE, DeepSeek-V3-style) model args for miles.
#
# Defines MOE_LAYER_FREQ and MODEL_ARGS=(...), sourced by run_async.sh.
#
# FarSkip deltas vs vanilla DeepSeek-V3 MoE:
#   --gated-attention           per-layer self_attn gate projection (linear_gate_proj)
#   --use-simple-farskip-layer  FarSkip layer spec (reference connectivity, no overlap)
#
# REQUIRES a FarSkip-capable Megatron on MEGATRON_PATH (see investigate_megatron.sh):
# --use-simple-farskip-layer and --gated-attention only exist in FarSkip Megatron.
# The HF DPO checkpoint has num_nextn_predict_layers=0, so MTP is OFF here.

# MoE layer frequency: layer 0 dense, layers 1..26 MoE (first_k_dense_replace=1).
NLAYERS="${MODEL_ARGS_NUM_LAYERS:-27}"
FIRST_K_DENSE_REPLACE=1
arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# Instella's HF config.json literally sets vocab_size=128896: the extra rows above
# the tokenizer's 128000 are REAL trained weights, not TP padding. So 128896 is the
# true vocab, and we set --vocab-size to it (default would be tokenizer.vocab_size
# =128000, per actor.py). This keeps the fix local to Instella: RL weight-sync's
# remove_padding(args.vocab_size) then keeps all 128896 rows (matching sglang's
# org_vocab_size=128896), without a miles-core change that would wrongly export
# garbage padding rows for models whose padded>vocab really is padding.
PADDED_VOCAB_SIZE="${PADDED_VOCAB_SIZE:-128896}"

MODEL_ARGS=(
    # NOTE: no --spec. This Megatron forbids --spec for MoE ("Model Spec must be
    # None when using MoEs"); FarSkip layers are built from --use-simple-farskip-layer
    # inside get_gpt_decoder_block_spec. gate_proj weight mapping is handled by
    # patch_mbridge_gate_proj.sh (patches the stock DeepseekV3 bridge), so model_type
    # stays deepseek_v3 and no custom --spec / shadow checkpoint is needed.
    --disable-bias-linear
    --num-layers "$NLAYERS"
    --hidden-size 2048
    --ffn-hidden-size 10944
    --num-attention-heads 16
    --kv-channels 128
    --normalization RMSNorm
    --position-embedding-type rope
    --no-position-embedding
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size "$PADDED_VOCAB_SIZE"
    --padded-vocab-size "$PADDED_VOCAB_SIZE"

    # --- MLA (multi-latent attention) + FarSkip gated attention ---
    # q_lora_rank is null in the HF config => full linear_q_proj (no q compression).
    --multi-latent-attention
    --gated-attention
    --kv-lora-rank 512
    --qk-head-dim 96
    --qk-pos-emb-head-dim 32
    --v-head-dim 128
    --qk-layernorm
    --rotary-scaling-factor 40
    --rotary-base 8000000
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion

    --use-simple-farskip-layer

    # --- MoE ---
    --num-experts 64
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size 1408
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 2816
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-token-dispatcher-type alltoall
    # Router load balancing is OFF for RL: no aux loss and a frozen expert bias. Both are
    # non-task signals that move the router every step, perturbing the DPO checkpoint's expert
    # assignment and drifting training routing away from what the rollout engine used. With them
    # on, math arm I and code v3 both collapsed (length down, evals down); math R1, identical
    # except for these flags, trained 400 steps without decline.
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    # Keep: the expert bias is part of the checkpoint and the rollout engine routes with it.
    # Dropping this flag would discard it; bias-update-rate 0 above only freezes it.
    --moe-router-enable-expert-bias

    # --- runtime / dtype ---
    --seq-length "${CONTEXT_LEN:-32768}"
    --max-position-embeddings "${CONTEXT_LEN:-32768}"
    --bf16
)
