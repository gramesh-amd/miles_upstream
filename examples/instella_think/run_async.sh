#!/bin/bash
# Instella-MoE-16B-A3B, fully-async RL on DAPO-Math-17k with AIME eval, max staleness 4.
# Model/parallelism/sglang specifics are the Instella FarSkip known-good values ported
# from Instella-MoE/rl. run_code.sh and run_if.sh are task wrappers over this file.
#
# TWO TOPOLOGIES (fully-async => always disaggregated; colocate is unsupported in train_async):
#   * SINGLE_NODE=1  smoke test on ONE 8-GPU node: 4 trainer (EP4) + 4 rollout (1 engine EP4/DP4).
#                    Starts a local Ray head automatically. Run this before any 3-node attempt.
#   * default        3 nodes / 24x MI355X: 1 trainer node (8 GPU, EP8) + 2 rollout nodes
#                    (16 GPU, 2 engines x EP8/DP8). Connects to an external Ray cluster.
#
# FarSkip contract (BOTH sides): training uses --use-simple-farskip-layer (model_args.sh);
# sglang rollout uses FARSKIP_REFERENCE_DECODER_LAYER=1 (Ray runtime_env below).
#
# PREREQUISITES (in order):
#   1. Bring up container(s) + (for 3-node) the Ray cluster.
#   2. On EVERY node (in-container): bash examples/instella_think/setup_farskip.sh
#   3. Point MEGATRON_PATH at a FarSkip Megatron (see investigate_megatron.sh).
#   4. One-time convert:   PHASE=prepare bash <this script>   (any 8-GPU container)
#   5a. SINGLE_NODE smoke: SINGLE_NODE=1 NUM_ROLLOUT=8 bash <this script>
#   5b. 3-node train:      RAY_ADDRESS=http://<HEAD_IP>:8265 MASTER_ADDR=<HEAD_IP> bash <this script>

set -eo pipefail

PHASE="${PHASE:-train}"
SINGLE_NODE="${SINGLE_NODE:-0}"     # 1 => single 8-GPU node smoke (4 trainer + 4 rollout)

# Response-length budget. Instella's context (max-position-embeddings) is 32768; miles sends
# max_new_tokens unclamped, so SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=32768 in the runtime env
# below clamps sglang's estimation (same as Primus-Instella-Internal/rl/examples/{if_rl,mopd})
# rather than us reserving prompt headroom here.
#
# TRAIN length is 32768, matching eval and the model's full context. This is deliberately
# ABOVE Instella's own if_rl/mopd recipes (16384): the first 3-node run at 16384 truncated
# 43.4% of responses at the cap (rollout/truncated_ratio 0.434, response_len median 15190),
# and a truncated solution has no boxed answer, so it scores wrong -- we were paying to
# generate reasoning and then throwing away its reward signal on nearly half the batch.
#
# 32768 response + any non-empty prompt overflows the 32768 context, and sglang rejects that
# outright with HTTP 400 ("You requested a total of 32851 tokens: 83 tokens from the input
# messages and 32768 tokens for the completion"). --sglang-allow-auto-truncate is the fix and
# is why no prompt headroom is reserved here: sglang clamps max_new_tokens to
# context_len - input_token_num per request (tokenizer_manager._validate_one_request), so each
# prompt gets the largest completion that fits. The separate input-truncation branch in that
# same function only fires when the PROMPT alone exceeds context; ours are 130 tokens median
# and 1453 max across dapo + both aime sets, so no prompt is ever silently chopped.
#
# Cost of 32768, all bounded and checked:
#   * fp32 logits per microbatch ([tokens, 128896]) go 8.4 GB -> 16.9 GB, because dynamic
#     batching gives an over-budget sample its own microbatch and never splits it (first-fit
#     in miles/utils/data.py:get_minimum_num_micro_batch_size) and CP=1 keeps it on one GPU.
#     Affordable against ~237 GB free, and --log-probs-chunk-size 2048 plus
#     --recompute-loss-function keep the cross-entropy workspace off the peak.
#   * Generation gets slower, so expect perf/wait_time_ratio (already 0.74-0.80 at 16384,
#     i.e. trainer idle waiting on rollouts) to rise. That is throughput, not stability.
# The two 20260819 smoke OOMs were NOT length-driven -- they were the allocator config (see
# PYTORCH_ALLOC_CONF below), which is why 30720 and 16384 both died.
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32768}"

export PYTHONUNBUFFERED=1
export PYTHONBUFFERED=16
ulimit -n "$(ulimit -Hn)" || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"   # examples/instella_think
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"                    # miles root
FULLY_ASYNC_DIR="${REPO_DIR}/examples/fully_async"
INSTELLA_DIR="${SCRIPT_DIR}"   # kept on PYTHONPATH for convenience (no modules required here anymore)

# Per-user paths (USER_HOME, NFS_ROOT, CKPT_ROOT, MEGATRON_PATH, ...); see paths.sh.
# MEGATRON_PATH is the FarSkip-capable Megatron assembled by investigate_megatron.sh.
source "${SCRIPT_DIR}/paths.sh"

# ---- model definition (MLA + gated attention + --use-simple-farskip-layer) --
source "${SCRIPT_DIR}/model_args.sh"

# ---- assets on shared NFS (visible from every node/container) ---------------
echo "Using CKPT_ROOT=${CKPT_ROOT}"

MODEL_NAME="Instella-MoE-16B-A3B-DPO"
HF_CKPT="${HF_CKPT:-${CKPT_ROOT}/${MODEL_NAME}}"         # real HF ckpt (model_type=deepseek_v3)
REF_LOAD="${REF_LOAD:-${CKPT_ROOT}/${MODEL_NAME}_torch_dist}"
# SAVE_TAG must change for any run that is not meant to continue an existing one: --load
# points at SAVE_DIR, so reusing a tag silently RESUMES from the newest iter_* in it rather
# than starting from the DPO weights.
SAVE_TAG="${SAVE_TAG:-miles_async_s4}"
SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/${MODEL_NAME}_${SAVE_TAG}}"

# gate_proj weight mapping is added to the stock DeepseekV3 bridge by
# patch_mbridge_gate_proj.sh (run once per container). model_type stays deepseek_v3,
# so we point --hf-checkpoint straight at the real HF checkpoint (no shadow dir).

# =====================================================================
# PHASE: prepare  (convert HF weights -> torch_dist, 8 GPU)
# =====================================================================
if [ "${PHASE}" = "prepare" ]; then
   set -x
   mkdir -p "${CKPT_ROOT}"
   [ -e "${HF_CKPT}" ] || { echo "ERROR: HF checkpoint missing: ${HF_CKPT}" >&2; exit 1; }
   if [ "$(cat "${REF_LOAD}/latest_checkpointed_iteration.txt" 2>/dev/null)" = "release" ]; then
      echo "Already converted: ${REF_LOAD}"; exit 0
   fi
   python3 -c "import mbridge,pathlib;s=open(pathlib.Path(mbridge.__file__).parent/'models'/'deepseek_v3.py').read();exit(0 if 'linear_gate_proj' in s else 1)" \
      || { echo "ERROR: mbridge missing gate_proj mapping. Run: bash ${INSTELLA_DIR}/patch_mbridge_gate_proj.sh" >&2; exit 1; }
   DEPRECATED_MEGATRON_COMPATIBLE=1 \
   PYTHONPATH="${REPO_DIR}:${MEGATRON_PATH}:${INSTELLA_DIR}:${PYTHONPATH:-}" \
   torchrun --nproc-per-node 8 \
      "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${HF_CKPT}" \
      --save "${REF_LOAD}"
   echo "Converted -> ${REF_LOAD}"
   exit 0
fi

# =====================================================================
# PHASE: train  (submit the async job to the running Ray cluster)
# =====================================================================
[ -e "${REF_LOAD}/latest_checkpointed_iteration.txt" ] || {
   echo "ERROR: ${REF_LOAD} missing. Run 'PHASE=prepare bash $0' first." >&2; exit 1; }

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   # --ref-load is NOT a "use a reference model" switch and must stay even though we run with
   # no KL anchor (see GRPO_ARGS): it is where the INITIAL weights come from. On a fresh run
   # --load points at a SAVE_DIR that does not exist yet, so miles falls back to
   # `args.load = args.ref_load` (utils/arguments.py:2607) and finetune-loads from here.
   --ref-load "${REF_LOAD}"
   --load "${SAVE_DIR}"
   --save "${SAVE_DIR}"
   # 200 was far too coarse to debug a slow collapse: the 410-step run left only iter_199 and
   # iter_399, so there is no checkpoint anywhere near where repetition crossed from 3% to 9%.
   --save-interval "${SAVE_INTERVAL:-200}"
)

# BOS/EOS: this checkpoint ships INCONSISTENT token ids, verified 20260819. tokenizer.json is
# Think-style -- id 0 = <|begin of sentence|>, id 1 = <|end of sentence|>, id 2 = <|pad|> -- but
# config.json and generation_config.json declare bos=1 / eos=2, i.e. they name EOS as BOS and PAD
# as EOS. The MODEL agrees with the tokenizer, not the configs: fed a completed answer it emits
# id 1 with p=1.0000.
#
# Nothing here is broken, but only because sglang checks three stop sources in
# schedule_batch._check_token_based_finish and the THIRD one saves us:
#   sampling_params.stop_token_ids (unset) | eos_token_ids (={2} from the configs, inert) |
#   token_id == tokenizer.eos_token_id (=1, resolved from the eos_token STRING) <-- this one.
# Confirmed live: finish_reason {"type":"stop","matched":1}. Training stays consistent because
# no_stop_trim=True keeps that id 1 in the response, and padding uses id 0 masked out of the loss
# (training_utils/data.py:166). If sglang ever drops the tokenizer check, pin it explicitly with
# --rollout-stop-token-ids 1. Do NOT "fix" this by trusting config.eos_token_id -- id 2 is pad and
# is essentially never emitted, so generation would run to the cap.
# Reward routing. Default is the math rule-based grader; run_if.sh sets CUSTOM_RM_PATH to
# the instruction-following verifier instead. These are mutually exclusive, not additive:
# miles short-circuits on custom_rm_path before it ever looks at rm_type
# (miles/rollout/rm_hub/__init__.py:46), so passing both would silently ignore --rm-type.
if [ -n "${CUSTOM_RM_PATH:-}" ]; then
   RM_ARGS=(--custom-rm-path "${CUSTOM_RM_PATH}")
else
   RM_ARGS=(--rm-type "${RM_TYPE:-deepscaler}")
fi

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   # PROMPT_DATA lets an offline difficulty-filtered subset be swapped in. Unfiltered
   # dapo-math wastes most of the rollout budget: at step 410, 60.6% of groups were dropped as
   # all-correct and 8.0% as all-wrong, so only 31.4% of generated sequences produced any
   # gradient at all.
   --prompt-data "${PROMPT_DATA:-${ASSET_ROOT}/data/dapo-math-17k/dapo-math-17k.jsonl}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   "${RM_ARGS[@]}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --over-sampling-batch-size 64
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
   --async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT:-512}"
   --max-weight-staleness "${MAX_WEIGHT_STALENESS:-4}"
   --num-rollout "${NUM_ROLLOUT:-3000}"
)

# Eval sets. EVAL_PROMPT_DATA overrides with space-separated "name path name path" pairs
# (the CLI form miles wants, arguments.py:995-1003); safe to word-split because none of
# these paths contain spaces. run_if.sh points this at the IFEval/IFBench sets.
if [ -n "${EVAL_PROMPT_DATA:-}" ]; then
   # shellcheck disable=SC2206  # deliberate word-splitting into name/path pairs
   EVAL_DATA_ARGS=(--eval-prompt-data ${EVAL_PROMPT_DATA})
else
   EVAL_DATA_ARGS=(--eval-prompt-data aime2024 "${ASSET_ROOT}/data/aime-2024/aime-2024-boxed.jsonl" \
                                      aime2025 "${ASSET_ROOT}/data/aime-2025/aime-2025-boxed.jsonl")
fi

EVAL_ARGS=(
   --eval-function-path miles.rollout.sglang_rollout.generate_rollout
   --eval-interval "${EVAL_INTERVAL:-20}"
   "${EVAL_DATA_ARGS[@]}"
   # 16 suits AIME's 30 prompts (30x16=480/set). An eval set with hundreds of prompts needs
   # far fewer samples or the eval alone costs more than the training steps between evals --
   # IFEval is 541 prompts, so run_if.sh drops this to 1.
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-16}"
   # Eval temperature is the single most misleading knob in this recipe. Training samples at
   # 1.0, where repetition is ~0.4%, but eval samples at 0.6, where the SAME step-401 weights
   # loop on 9.2% of AIME responses: pass@1 measured 0.6083 at 0.6 versus 0.7000 at 1.0. Loop
   # escape length is L/(1-p^L), so it crosses the 32k budget at p ~ 0.99997 and loops flip
   # from self-healing to absorbing. Keep 0.6 for continuity with Instella's published recipe
   # and prior runs, and track the 1.0 number separately (study_tim/probe_collapse.py) so a
   # temperature artefact is never mistaken for a capability loss again.
   --eval-temperature "${EVAL_TEMPERATURE:-0.6}"
   --eval-top-p 0.95
   # Instella's own recipe (HF model card + Instella-MoE/rl eval_config_mopd.yaml) uses
   # temp 0.6 / top_p 0.95 with NO top_k. top_k=20 was a Qwen3 carry-over; -1 disables it.
   --eval-top-k -1
   # Matches Instella's eval_config_{if_thinkrl,mopd}.yaml (max_response_len 32768).
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
)
# Optional extra eval metrics. The hook returns False, so miles' default eval logging
# still runs and this only ADDS keys (run_if.sh uses it for IFEval's loose metric).
if [ -n "${CUSTOM_EVAL_LOG_PATH:-}" ]; then
   EVAL_ARGS+=(--custom-eval-rollout-log-function-path "${CUSTOM_EVAL_LOG_PATH}")
fi

# train_async.py:59 runs a baseline eval at rollout_id==0 unless this is set. That eval fans
# out 960 sequences (2 datasets x 30 prompts x 16 samples) at EVAL_MAX_RESPONSE_LEN on the
# rollout engines, and it is exactly where BOTH prior 3-node runs died: rollout-engine
# HSA_STATUS_ERROR_OUT_OF_RESOURCES with "Available Free mem : 14 MB" at "Eval aime2024 47%,
# #running-req: 233" (20260813_203611 and 20260814_011033). It also costs ~45 min before the
# first training step. --eval-interval still gives the first datapoint at step 20.
# Set SKIP_EVAL_BEFORE_TRAIN=0 to restore the baseline eval.
if [ "${SKIP_EVAL_BEFORE_TRAIN:-1}" = "1" ]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

ASYNC_ARGS=(
   --pause-generation-mode in_place
   --update-weight-transfer-mode broadcast
)

if [ "${SINGLE_NODE}" = "1" ]; then
   # One 8-GPU node: 4 trainer (EP4) + 4 rollout (1 engine EP4/DP4). 64 experts % 4 == 0.
   TOPOLOGY_ARGS=(
      --actor-num-nodes 1
      --actor-num-gpus-per-node 4
      --num-gpus-per-node 8
      --rollout-num-gpus 4
   )
   EP_SIZE=4
   ROLLOUT_ENGINE_GPUS=4
else
   # 3 nodes: trainer node (8 GPU, EP8) + 2 rollout nodes (16 GPU, 2 engines EP8/DP8).
   TOPOLOGY_ARGS=(
      --actor-num-nodes 1
      --actor-num-gpus-per-node 8
      --num-gpus-per-node 8
      --rollout-num-gpus 16
   )
   EP_SIZE=8
   ROLLOUT_ENGINE_GPUS=8
fi

# --- trainer parallelism (Instella known-good from Instella-MoE/rl if_rl) -----
PERF_ARGS=(
   --tensor-model-parallel-size 1
   # NO-OP here: at TP=1 Megatron warns "Disabling sequence parallelism because tensor model
   # parallelism is disabled" and forces sequence_parallel=False to keep numerics identical
   # (training/arguments.py, "disable sequence parallelism when tp=1"). Kept only to mirror the
   # reference recipe; it does nothing unless TP is raised above 1.
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size 1
   --moe-token-dispatcher-type alltoall
   --use-distributed-optimizer
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   # Peak VRAM in the log-prob pass is the fp32 logits tensor [tokens, padded_vocab/TP]:
   # 16384 x 128896 x 4B = 8.4 GB at TP1. The pass itself is @torch.no_grad (no activations
   # retained), so these two flags target the remaining transients: chunk bounds the
   # cross-entropy/entropy workspace over the token dim (exact, per-token independent), and
   # recompute-loss-function rebuilds the loss in backward instead of holding log-softmax.
   --log-probs-chunk-size 2048
   --recompute-loss-function
   # --balance-by-flops   # enable if the balance-by-flops core fix is applied
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # No KL anchor (matches Instella-MoE/rl if_rl + mopd, which pass --kl-coef 0 and never
   # --use-kl-loss). Both must stay off: miles allocates a reference model whenever
   # `kl_coef != 0 or use_kl_loss` (miles/ray/placement_group.py:157), which costs ~10 GB of
   # pinned HOST ram per rank (variants live on CPU, see miles/utils/tensor_backper.py) and
   # a full extra long-sequence log-prob forward per step, for a term worth exactly zero.
   # It does NOT cut peak VRAM -- only one weight set is ever GPU-resident.
   --kl-coef 0
   --entropy-coef 0.00
   # R3 routing replay (arxiv 2510.11370), verified present in this miles at
   # arguments.py:1298 -- sglang then returns its expert choices per token
   # (enable_return_routed_experts is auto-set) and the training forward replays them instead
   # of re-routing. This attacks the train/inference mismatch FLOOR: at step 1 the trainer and
   # the engine hold identical weights, yet train_rollout_logprob_abs_diff was already 0.122
   # against a mean rollout logprob of -0.859 (14.2% relative). Token ids are NOT the cause --
   # miles reuses the engine's own output ids verbatim (sglang_rollout.py:260, and prompt
   # encoding passes add_special_tokens=False, so exactly one BOS). For a top-6-of-64 MoE a
   # single differing expert moves a token's logprob a lot, so routing is the leading suspect,
   # alongside FarSkip running two implementations (reference decoder layer in sglang vs
   # --use-simple-farskip-layer in Megatron) and aiter's shuffled-layout MoE GEMMs.
   --eps-clip "${EPS_CLIP:-0.2}"
   # NOTE: pg_clipfrac has been exactly 0.0 at every logged step of the 20260819_054510 run, so
   # clip-higher is never actually binding and is doing nothing to hold entropy up here. Do not
   # count it as the entropy-collapse safeguard: entropy_loss still fell 0.0694 -> 0.0110 (6.3x)
   # over 115 steps with --entropy-coef 0 and no KL anchor.
   --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
   --use-tis
)

# The crash that forced this OFF is FIXED as of 20260824; the remaining caveat is that our own
# measurement says it will not buy what it was reached for.
#   1. Cause of the crash, now repaired. The Megatron on PYTHONPATH is the Primus fork, which
#      carries FarSkip but never had the R3 hooks, so no Replay was ever registered:
#      replay_list stayed EMPTY and register_replay_list_moe died on replay_list[0] with
#      "IndexError: list index out of range" (20260820_054007). It was never a FarSkip layer
#      layout problem -- register_replay_list_moe indexes replay_data by GLOBAL layer id, and
#      sglang does emit num_layers=27 streams (sglang_rollout.py:278 reshapes to
#      [tokens-1, num_layers, topk] and that reshape succeeded), so layer 0 being dense is
#      handled correctly. The newer /root/Megatron-LM has the two hooks but no FarSkip, so we
#      ported them into the Primus fork instead of switching trees: TopKRouter.__init__ now
#      registers one Replay per router, and topk_routing_with_score_function routes its topk
#      through the manager. Verified by study_tim/test_r3_port.py, which records a routing,
#      perturbs the logits by -50 so re-derivation shares zero of 72 picks, and confirms the
#      replayed routing still follows the recording on every token.
#   2. It is worth ~4.3x on the logprob gap -- KEEP IT ON, but read the metric carefully.
#      train_rollout_logprob_abs_diff is reduced by sum_of_sample_mean, and that reducer is
#      itself switched by --calculate-per-token-loss (cp_utils.py:107 gives equal weight per
#      SAMPLE, :115 equal weight per TOKEN), so this run's number is NOT directly comparable
#      to 20260820_231843's 0.0433: R3 and the loss normalisation were turned on together.
#      The clean control is the concurrent 20260824_001341, which has the same model, data and
#      fp32 readout with R3 off and the old reducer: it reads 0.0538, i.e. no better than the
#      run it replaced, so nothing improved on its own.
#      That the underlying divergence really shrank, rather than just being re-weighted, is
#      settled by three functionals of the SAME per-token divergence moving by three different
#      powers, which is what must happen if the distribution's scale shrinks ~4.3x and cannot
#      happen under a convex re-weighting of fixed per-token values:
#        mean |dlogprob| (tis_abs)  0.0500 -> 0.0116   4.3x    (linear in scale)
#        KL (train_rollout_kl)      0.00627 -> 0.00041 15.3x   (quadratic, 4.3^2 = 18)
#        tail frac (tis_clipfrac)   7.9e-4 -> 1.15e-6  ~685x   (tail probability)
#      So ~77% of the post-fp32 gap was MoE routing. The residual 0.0116 is the non-routing
#      remainder (kernels, precision), below Qwen3.6-35B-A3B's 0.0150.
#      Caveat on semantics: R3 does not make the trainer more accurate in isolation, it makes
#      the trainer reuse the rollout's expert choices, so what is now reported is a
#      routing-CONDITIONED logprob gap. That is the right quantity for the PPO ratio, which
#      needs the density that actually generated the tokens, but the metric no longer contains
#      routing divergence at all and should not be read as full trainer/sampler agreement.
#      This overturns the earlier conclusion recorded here, which said routing was a co-symptom
#      because cross-tabulating 2048 token positions showed a 0.110 nat gap for positions with
#      identical routing in all 27 layers versus 0.104 for positions with a flipped layer. That
#      measurement was taken BEFORE the fp32 LM-head fix, when bf16 readout error (0.5 nat ULP
#      at Instella's logit scale) swamped everything and conditioning on routing could not
#      resolve a difference. Once fp32 removed the dominant term, routing became the leading
#      remaining one. The lesson is that conditioning on a suspect while a larger error source
#      is still in the signal cannot exonerate the suspect.
if [ "${USE_ROUTING_REPLAY:-0}" = "1" ]; then
   GRPO_ARGS+=(--use-rollout-routing-replay)
fi

# Advantage normalisation. Default (std on) divides each group's centred rewards by their own
# std, which hands the largest per-sample advantage to the LEAST informative groups: a 7/8
# group gets ~2.65x the push of a balanced 4/4 group, versus ~1.75x when only centring. With
# dapo-math unfiltered, 52.5% of groups came back all-correct and the surviving mass sits at
# 7/8 and 1/8, so std normalisation spent most of the update on near-solved prompts. That is
# the pressure that drove mean token probability 0.448 -> 0.743 over 410 steps with nothing
# opposing it. Olmo3 centres without dividing.
if [ "${DISABLE_STD_NORM:-0}" = "1" ]; then
   GRPO_ARGS+=(--disable-grpo-std-normalization)
fi

# Loss normalisation. Without this miles normalises per SAMPLE, which gives a 32k truncated
# failure the same total weight as a 500-token success and so dilutes exactly the overlong
# negatives we need the gradient to see. --calculate-per-token-loss divides by total batch
# tokens instead (DAPO/Olmo3 token-level loss).
#
# MEASURED (scalar level only -- see caveat below): token-level multiplies a sample's influence
# by L_i/mean(L), and truncated samples sit at the 32768 cap, so their exact upweighting is
# cap/mean_length = 1.9-2.3x here. That put 37.6% of the batch's gradient mass on reward-0
# truncated traces at step 0 versus 19.5% under per-sample. Rollout truncation then fell
# 0.195 -> 0.070 while the per-sample run stayed flat at 0.21-0.23, and pass@1/(1-truncated)
# fell 0.92 -> 0.80 by step 139 where the per-sample run needed until step ~285.
#
# Do NOT read a mechanism into those ratios. Per-response probe data (study_tim/results/collapse,
# 240 base + 120 step-401 responses, same matcher on both) refutes the obvious story:
#   - "the policy commits earlier": FALSE. Among responses that FINISH and are WRONG, median
#     length went UP, 16871 -> 20750 tokens at T=0.6.
#   - "truncated always scores 0": FALSE. 3.8% of base and 13.3% of step-401 truncated responses
#     carried a scoring answer, so pass@1/(1-truncated_ratio) is biased, not an identity.
#   - what actually changes is the KIND of truncation, not the rate: repetition among truncated
#     responses went 13.5% (7/52) -> 73.3% (11/15) while total truncation FELL 21.7% -> 12.5%.
#     Every repetition case in both policies was truncated; 0 of 188 and 0 of 105 finished
#     responses repeated. Base truncation is "still reasoning at the cap", degraded-policy
#     truncation is an inescapable loop. An aggregate truncated_ratio cannot tell these apart.
# Token sharpness over the same responses, at matched T=1.0: fraction of sampled tokens with
# p>0.99 went 0.379 -> 0.623 (0.598 -> 0.771 at T=0.6), so the sharpening itself is real.
#
# CONFIRMED by probing this run's own engines at step ~170 (study_tim/results/olmo3_probe,
# 120 responses per temperature). What token-level loss + a punitive truncation reward teaches
# is to GUESS BEFORE THE CAP. At T=1.0, outcome composition moved as:
#     correct          75.4% -> 75.0%   (no gain at all)
#     finished WRONG    1.2% -> 17.5%
#     truncated        23.3% ->  7.5%
# i.e. truncations converted into wrong finished answers ~one-for-one, emitted at a median of
# 25687 tokens against the 32768 cap (base: 20590). The model did not learn to reason more
# efficiently; it learned to stop and answer before the ceiling, which is the cheapest way to
# dodge the penalty. The cost is the long correct traces, which is where the hard problems are:
# correct traces needing >=20k tokens 17.7% -> 10.0%, >=24k tokens 11.0% -> 3.3%, while
# corr(pass_rate, correct-trace length) is -0.63 on the base, i.e. hard problems need length.
# The Cov(A,L) imbalance is NOT yet self-limiting here: -19.9% (strict grader) to -27.2%
# (lenient) at step ~170, still negative in 10/10 groups, versus -4.8% for the per-sample run
# at step 401.
#
# THEORY, verified against the source (Ding et al., "On the Impossibility of Unbiased and
# Length-Invariant Policy Optimization with Outcome Rewards", arxiv 2607.23364 -- NOT Shao et al.,
# an attribution error made earlier here). Corollaries 8-9: the family f_alpha(L) = L^(alpha-1)
# has alpha=0 = GRPO (per-sample), which satisfies length invariance but NOT gradient
# unbiasedness, and alpha=1 = Dr. GRPO (token-level), which satisfies gradient unbiasedness but
# NOT length invariance. So token-level loss genuinely does carry a length bias; this is a
# property of the estimator, not of miles. At length ratio r the longer trajectory captures
# r/(1+r) of the gradient signal, so the signed imbalance is (r-1)/(r+1). Their extreme example:
# correct at 10 tokens vs incorrect at 10,000 gives the wrong trajectory 99.9% of the signal.
#
# CRITICAL CAVEAT, in the paper's own words (section 6): "The practical importance of the tradeoff
# depends on the variance of response lengths. When all responses to a given prompt have similar
# lengths [...] the difference between alpha=0 and alpha=1 is negligible." Measured here on REAL
# dapo-math batches (study_tim/online_cov.py on the step-0 dump), the within-prompt wrong/correct
# length ratio is only 1.11x, giving an imbalance of -11.3% unfiltered and +4.8% WITH overlong
# filtering. The 1.65-1.76x ratios behind the -19.9%/-24.7% figures above came from AIME probes,
# which are NOT representative of the training distribution. By the paper's own criterion this
# choice is therefore second-order for us, and the normalisation story is too small on real data
# to explain the ~2x faster degradation on its own. Do not treat it as the established cause.
#
# NOTE ON DAPO: overlong filtering and soft overlong punishment are DAPO's (arxiv 2503.14476),
# NOT Olmo3's -- the Olmo3 Think recipe is zero-gradient filtering, active sampling, token-level
# loss, no KL, clip-higher, TIS, and no std-dev normalisation, with no overlong filtering in it.
# Per-component point values from DAPO's ablation were quoted here previously; they are UNVERIFIED
# against the paper and must not be relied on until checked.
if [ "${PER_TOKEN_LOSS:-0}" = "1" ]; then
   GRPO_ARGS+=(--calculate-per-token-loss)
fi

# OVERLONG_FILTER=1: drop truncated samples from the loss AND recentre the survivors' advantages.
# Implemented in examples/instella_think/overlong_filter.py (on the workers' PYTHONPATH, see the
# runtime env below) via --custom-reward-post-process-path, because that hook is the only place
# that can do both halves: train_data_conversion.py reads sample.remove_sample AFTER the reward
# post-process returns, so one function can zero the mask and recentre in a single pass.
#
# NOT --rollout-sample-filter-path, which is the flag that looks right for this. Its own help
# states the catch -- "does not determine whether the sample participates in advantage
# normalization" -- i.e. it is the MASK form: zero the loss, leave the sample in the group mean.
# Truncated samples carry negative advantage, so that deletes negative push and leaves the
# survivors net POSITIVE: measured +19.4% token-level / +25.6% per-sample at step ~170, and
# +83.5%/+91.6% on the DPO base's 23% truncation rate. A uniform positive push on sampled tokens
# is a sharpening pressure, which is this model's actual failure mode, so the stock flag would
# accelerate the collapse. See study_tim/overlong_filter_test.py for the 2x2.
#
# RISK, and why the filter logs every call. Truncated samples are where nearly all the NEGATIVE
# examples live: at T=1.0 the base is 75.4% correct / 1.2% finished-wrong / 23.3% truncated, so
# dropping truncated removes ~95% of the contrast and leaves all-correct groups, which have zero
# advantage spread and contribute nothing. On AIME this took the base from 10 usable groups to 1
# (step ~170, less truncated, went 10 -> 7). dapo-math is easier so the real rate should differ,
# but if "usable=" in the overlong_filter log line collapses, this is starving the update and the
# right answer is a soft length penalty (DAPO's soft overlong punishment) rather than a hard drop.
if [ "${OVERLONG_FILTER:-0}" = "1" ]; then
   GRPO_ARGS+=(--custom-reward-post-process-path overlong_filter.overlong_filter_post_process)
fi

# DUMP_ROLLOUTS=1: per-step rollout dump, so the two covariances that drive this whole argument
# can be measured on the ACTUAL training batches instead of on AIME probes standing in for them.
# One .pt per step holding every sample's reward, response_length and rollout_log_probs is
# sufficient for both, because GRPO advantages are a deterministic function of the group's
# rewards (that is exactly what study_tim/mass_balance.py recomputes):
#     Cov(A, L)      -> the token-level gradient imbalance, i.e. the -19.9% number
#     Cov(log p, A)  -> the per-step entropy drift (arxiv 2505.22617); positive => collapse
#
# Deliberately NOT --dump-details, which would additionally set save_debug_train_data and write
# one .pt PER RANK (24 here, arguments.py:2722) containing that rank's full training tensors.
# That is ~24x the volume for data we do not need, since rewards+lengths+rollout logprobs already
# determine both covariances. The lean flag still gets a dashboard_columns parquet mirror for
# free (debug_data.py:113) because save_debug_rollout_data writes it unconditionally.
#
# Cost at this config: 256 sequences/step (--rollout-batch-size 32 x --n-samples-per-prompt 8) at
# ~15k tokens each is roughly 150-200 MB/step against ~8.5 min/step, so the NFS write is not on
# the critical path. Evals dump too (as eval_<id>.pt) at 960 sequences.
if [ "${DUMP_ROLLOUTS:-0}" = "1" ]; then
   DUMP_DIR="${DUMP_DIR:-${DUMP_ROOT}/${SAVE_TAG}}"
   mkdir -p "${DUMP_DIR}/rollout_data"
   GRPO_ARGS+=(--save-debug-rollout-data "${DUMP_DIR}/rollout_data/{rollout_id}.pt")
   echo "DUMP_ROLLOUTS=1 -> ${DUMP_DIR}/rollout_data/"
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

RUN_TAG="$([ "${SINGLE_NODE}" = "1" ] && echo 1node-smoke || echo 3node)"
WANDB_ARGS=(
   --use-wandb
   # Separate projects keep unrelated tasks' eval axes apart: the math runs log
   # eval/aime2024 while an IF run logs eval/ifeval, and mixing them in one project makes
   # every chart half-empty. Default is unchanged for the math recipes.
   --wandb-project "${WANDB_PROJECT:-miles-mis}"
   --wandb-group "${WANDB_GROUP:-instella-moe-16b-async-s4}-${RUN_TAG}"
   --wandb-key "${WANDB_KEY:-}"
   --wandb-dir "${WANDB_LOCAL_DIR}"
)
# miles appends a random id AND "-RANK_0" to the run name when the suffix is on
# (utils/tracking_utils/wandb_utils.py:51-57), which is why rytwnj5z had to be renamed by
# hand in the UI to say what it was. WANDB_RANDOM_SUFFIX=0 makes group == name == exactly
# the string set above, so a run is identifiable in the sidebar without renaming it.
if [ "${WANDB_RANDOM_SUFFIX:-1}" = "0" ]; then
   WANDB_ARGS+=(--disable-wandb-random-suffix)
fi

# Resolved once so the value used for the flag and the value printed in the
# ENABLE_DP_ATTENTION=0 message below cannot drift apart.
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.60}"

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_ENGINE_GPUS}"   # single-node:4 (1 engine); 3-node:8 (2 engines)
   # 0.60 (was 0.70): the static pool at 0.70 leaves too little headroom for the transient
   # non-KV spikes (fp32 logits [batch, 128896], sampling workspaces, kernel scratch) and the
   # 3-node runs aborted with the ROCm runtime unable to allocate at all ("Available Free
   # mem : 14 MB") while the KV pool itself was only at 54% usage. KV is heavily
   # over-provisioned for this workload anyway -- the healthy 1-node smoke runs at
   # "token usage: 0.10" -- so giving ~29 GB back to scratch costs no concurrency.
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
   --sglang-ep-size "${ROLLOUT_ENGINE_GPUS}"
   --use-miles-router
   # Router health-check: the miles router pings each engine /health on a hardcoded 5s
   # timeout (router/router.py:79) every --rollout-health-check-interval (default 30s) and
   # after --miles-router-health-check-failure-threshold (default 3) consecutive misses adds
   # it to dead_workers *permanently* -- reconnect is an open TODO (router.py:112). A busy
   # engine that can't answer /health in 5s during a heavy eval therefore gets evicted for
   # good; lose both of the 2 engines and every /generate 500s and the trainer hangs forever.
   # Make eviction effectively impossible: engines recover on their own once the batch drains.
   --rollout-health-check-interval 3600
   --miles-router-health-check-failure-threshold 1000000
   # Instella-MoE is DeepSeek-V3-style MLA (NOT mamba/hybrid), so no mamba flags.
   #
   # dp-attention is ON by default (ENABLE_DP_ATTENTION=1, applied below) and should stay
   # that way. Three independent reasons:
   #   1. CORRECT + STABLE. Verified for FarSkip at both ep4/dp4 (1-node smoke, and
   #      instella_moe_sglang_lm_eval_515.sh at EP_SIZE=DP_SIZE=4) and ep8/dp8 (3-node).
   #      The empty-KV out-of-bounds fault that once argued against it is fixed by the
   #      k.shape[0]==0 guard in sglang's triton_backend; instella_rl_preflight.py asserts
   #      that guard is present (dp_attention_fix), so check preflight before doubting this.
   #   2. FASTER. Work-matched lm_eval A/B on 3 nodes (2491 docs, cap 2048, bs 512):
   #      5.67 GPU-min with dp-attention vs 7.96 for the tp1/ep1/dp8 replica layout = 1.40x,
   #      at ~equal generated tokens and with scores inside one stderr. Sharded experts
   #      amortise their weight reads better than 8 full replicas do.
   #   3. It shrinks the prefill chunk 8x. With dp-attention on, sglang divides chunked-prefill
   #      by dp_size -- logging "DP attention is enabled. The chunked prefill size is adjusted
   #      to N to avoid MoE kernel issues" -- so we get 16384//8 = 2048 (confirmed in this run's
   #      server_args) instead of the 288 GB-tier default of 16384. Turning dp-attention off
   #      hands the MoE an 8x bigger prefill chunk, which is how both 3-node runs OOM'd.
   # ENABLE_DP_ATTENTION=0 is a debug-only escape hatch; see the else-branch below for the
   # extra flags it then needs just to avoid reproducing that OOM.
   # Required at ROLLOUT/EVAL_MAX_RESPONSE_LEN 32768 (see the length block at the top): without
   # it sglang 400s every request whose prompt + max_new_tokens exceeds the 32768 context.
   --sglang-allow-auto-truncate
   --sglang-disable-radix-cache
   --sglang-disable-shared-experts-fusion
   # The rollout runs triton attention while Megatron trains with --attention-backend flash
   # (below), so every logprob comparison crosses two different attention implementations. The
   # Qwen3.6-35B-A3B baseline left this at sglang's default and reached
   # train_rollout_logprob_abs_diff 0.0150 where we start at 0.122, so this is on the suspect list.
   # HAZARD before you change it: the triton backend is load-bearing here. The empty-KV
   # out-of-bounds fault under dp-attention is fixed by the k.shape[0]==0 guard *in
   # sglang's triton_backend* (reason 1 above). Switching backends with ENABLE_DP_ATTENTION=1
   # gives up that guard, so only A/B this together with a preflight re-check, or at dp1.
   --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-triton}"
   # HISTORICAL FAULT, now fixed (see RESOLVED below) -- kept because the ruling-out was
   # expensive and would otherwise be re-derived on the next scare. With SGLANG_USE_AITER=1 on
   # gfx950 (MI355X) the engine used to hit intermittent GPU "Memory access fault ... Reason:
   # Unknown" at varying near-null addresses during large/uneven dp-attention prefills (post
   # weight-sync, since --skip-eval-before-train makes the first big prefill land after the
   # first sync). What was ruled out:
   #   * NOT negative topk_ids: live instrumentation shows ids are global 0..63, neg=0.
   #   * NOT bad MoE values/shapes/indices: the EXACT faulting fused_moe inputs
   #     (dumped from the live engine, M=12998) replay CLEAN in isolation at all M.
   #   * NOT a stale/rebound weight pointer: live w13/w2 are contiguous with correct
   #     strides and full storage right before the fault.
   #   * Intermittent + varying addresses => suspected small OOB read inside an aiter
   #     kernel that only faults when the OOB page is unmapped (fresh-allocator replay
   #     never hits it). Precise attribution needs rocgdb/ROCm ATT.
   # NOTE (Primus-Instella-Internal#2): SGLANG_USE_AITER=0 is NOT a safe mitigation --
   # it stops the crash but SILENTLY corrupts output (the sglang515 overlay stores MoE
   # weights in aiter's shuffled layout; the non-aiter path reads them unshuffled =>
   # gibberish, e.g. "2+2=5", arc_challenge 0.195 which is below the random-chance floor).
   # RESOLVED: the current default (aiter=1 + dp-attention) is now BOTH correct and stable,
   # thanks to the k.shape[0]==0 triton_backend guard. Kept for history, the three configs:
   #   * aiter=0             -> stable but GIBBERISH (silent; throughput looks normal)
   #   * aiter=1 + dp-attn   -> WAS the OOB "Memory access fault" under load; now fixed,
   #                            and this is the default (see the dp-attention block above)
   #   * aiter=1, dp-attn=0  -> correct, but drops the chunked-prefill clamp and OOMs
   #                            without it (HSA_STATUS_ERROR_OUT_OF_RESOURCES, run 023 dpoff)
   # Residual cleanup (not a blocker) = tune aiter for this model's MLA GEMM shapes (N=544,
   # K=2048, large prefill M) so the untuned fallback ("not found tuned") is never taken.
   #
   # 64 is copied from the verified lm_eval config (which also runs SGLANG_USE_AITER=1). It is a
   # CAP: sglang only replays a decode graph when the live batch is <= this. Unlike lm_eval,
   # which saturates at #running-req 90-98 and logs "cuda graph: False", RL rollout decodes at
   # #running-req ~13-33, so it sits UNDER the cap and does run on graphs ("cuda graph: True" in
   # the 32768 run). The cap is therefore load-bearing here, not inert -- if concurrency ever
   # climbs past 64 the engines silently fall off graphs and decode gets slower.
   --sglang-cuda-graph-max-bs-decode 64
)
# Concurrency cap for the rollout engines. Unset by default (sglang decides), which is what
# the math runs used -- and what made the eval fan-out fatal: with no cap the engines took
# #running-req 233 during eval and died with HSA_STATUS_ERROR_OUT_OF_RESOURCES at
# "Available Free mem : 14 MB" (both 20260813/20260814 3-node runs). Capping bounds peak KV
# and scratch WITHOUT shortening responses, so the eval measurement is untouched -- a
# shorter --eval-max-response-len would instead truncate answers, and a truncated response
# never closes </think>, which scores 0 and biases the metric down. 64 also matches
# --sglang-cuda-graph-max-bs-decode above, so decode stays on cuda graphs.
if [ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]; then
   SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [ "${ENABLE_DP_ATTENTION:-1}" = "1" ]; then
   SGLANG_ARGS+=(--sglang-enable-dp-attention --sglang-dp-size "${ROLLOUT_ENGINE_GPUS}")
else
   # Without dp-attention sglang skips its "// dp_size ... to avoid MoE kernel issues" division,
   # so chunked-prefill stays at the 288 GB-tier default of 16384 -- 8x the 2048 the dp-attention
   # path runs at. Long-context prefills then OOM: run 023 dpoff died with
   # HSA_STATUS_ERROR_OUT_OF_RESOURCES / "Available Free mem : 14 MB" during its first eval
   # (960 sequences at --eval-max-response-len). So set 2048 explicitly to match what the
   # default path actually gets. SKIP_EVAL_BEFORE_TRAIN=1 only moves that first big fan-out to
   # step 20 rather than removing it, so this still matters. If it OOMs anyway, lower
   # SGLANG_MEM_FRACTION and/or the eval fan-out -- tuning knobs, not the aiter fault above.
   echo "ENABLE_DP_ATTENTION=0 -> rollout WITHOUT dp-attention (pure TP/EP); pinning chunked-prefill=${SGLANG_CHUNKED_PREFILL:-2048} (sglang would otherwise use 16384), mem-fraction=${SGLANG_MEM_FRACTION}"
   SGLANG_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL:-2048}")
fi
# USE_TRITON_MOE=1 is an opt-in surgical path: keep aiter on but force the MoE runner to
# triton. Default off, and leave it off -- with SGLANG_USE_AITER=1 it crashed the sglang
# scheduler at init (native fault inside launch_dp_attention_schedulers), and reaching it by
# turning aiter off instead lands on the unshuffled-weight gibberish described above.
if [ "${USE_TRITON_MOE:-0}" = "1" ]; then
   SGLANG_ARGS+=(--sglang-moe-runner-backend triton)
fi

# fp32 logit readout. Both halves of train_rollout_logprob_abs_diff currently round
# logits to a bf16 grid, and at Instella's logit magnitudes ([32,128)) one bf16 step is
# 0.5 nats, so that rounding is a first-order term in the metric rather than noise.
# The two sides are toggled separately because they carry different risk.
#
# Rollout side (safe, on by default): sglang's lm_head runs in bf16 unless
# --enable-fp32-lm-head is set. Measured on a frozen corpus, engine against itself:
# mean 0.198 -> 0.034 nats, >0.5-nat tail 14.06% -> 0.39%, i.e. down to Qwen's level.
# Cross-checked against the production engine at 0.080 nats with nothing above 1 nat, so
# it is the same function more accurately read, not a different one. Costs ~6% output
# tok/s. This is also the side --use-tis compares against, so improving it directly
# sharpens the TIS correction.
if [ "${FP32_ROLLOUT_READOUT:-1}" = "1" ]; then
   SGLANG_ARGS+=(--sglang-enable-fp32-lm-head)
fi

# Trainer side (ON by default; this is the dominant term, measured).
# The trainer head multiplies bf16 hidden states by a bf16 output_layer.weight and
# Float16Module upcasts the RESULT to fp32, which recovers nothing. Measured on the DPO
# checkpoint over 3974 real rollout tokens, reading the head out in fp32 from the identical
# hidden state moves log-probs by 0.0893 nats on average -- against a reported
# train_rollout_logprob_abs_diff of 0.0989. The same measurement gives 0.0135 on
# Qwen3.6-35B-A3B, because Instella's per-token max |logit| has median 118 (bf16 step 0.5
# nats) where Qwen's is 24 (step 0.125). So this is the whole gap, and it comes from the
# checkpoint's logit scale rather than from the inference stack.
#
# --fp32-lm-head is now applied to BOTH the log-prob pass and the gradient pass. It has to
# be: the PPO ratio is exp(new - old) with old from the log-prob pass and new from the
# training pass, and the readout error is spiky (median 0.026, P(>0.2)=17%, P(>0.28)=9.9%),
# so promoting one side alone would push a sixth of all tokens outside eps_clip every step.
# The gradient pass cannot just run Megatron's linear in fp32 (its fused wgrad kernels take
# half-precision operands), so fp32_lm_head.py implements the matmul as an autograd Function
# that accumulates dW into weight.main_grad following Megatron's own contract; see the unit
# test at study_tim/test_fp32_logits.py. It bypasses tensor-parallel plumbing and therefore
# asserts tp=1 for the head, which holds here (--tensor-model-parallel-size 1).
if [ "${FP32_TRAINER_READOUT:-1}" = "1" ]; then
   GRPO_ARGS+=(--fp32-lm-head)
fi

MISC_ARGS=(
   # miles defaults this to 10 (miles/utils/arguments.py:163), which is also the c10d store
   # timeout for a process group's FIRST NCCL bootstrap. During the first update_weights the
   # dense weights broadcast to the engines before the expert buckets are gathered, so ranks
   # that reach the EP all_gather early sit idle waiting on rank 0 for that whole window and
   # die with "retrieving ncclUniqueId from [0] ... wait timeout after 600000ms". The 1-node
   # smoke never hits this (8 GPUs, no cross-node hop); 24 GPUs can.
   --distributed-timeout-minutes 120
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-passrate
)

# --- logging to shared NFS ---------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${WANDB_LOCAL_DIR}"
LOG_FILE="${LOG_DIR}/instella-moe-16b-async-s4-${RUN_TAG}-$(hostname -s)-${TS}.log"
echo "Logging everything to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

HAS_NVLINK=0
NVLINK_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
   NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -c 'NV[0-9]' || true)
   [ "${NVLINK_COUNT:-0}" -gt 0 ] && HAS_NVLINK=1
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# --- Ray: local head for single-node smoke, else connect to external cluster --
if [ "${SINGLE_NODE}" = "1" ]; then
   # Clean any wedged processes from a previous smoke run, then start a local head.
   pkill -9 sglang 2>/dev/null || true
   ray stop --force 2>/dev/null || true
   pkill -9 ray 2>/dev/null || true
   pkill -9 -f train_async.py 2>/dev/null || true
   sleep 3
   export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
   export RAY_ADDRESS="http://${MASTER_ADDR}:8265"
else
   if [ -z "${RAY_ADDRESS:-}" ]; then
      ip=$(ps aux | grep -m1 'ray.*dashboard' | grep -oP '(?<=--node-ip-address=)[0-9\.]+' | head -1)
      port=$(ps aux | grep -m1 'ray.*dashboard' | grep -oP '(?<=dashboard-port=)\d+' | head -1)
      [ -n "$ip" ] && export RAY_ADDRESS="http://${ip}:${port:-8265}"
   fi
   [ -n "${RAY_ADDRESS:-}" ] || { echo "ERROR: RAY_ADDRESS unset and no local Ray head found." >&2; exit 1; }
   export MASTER_ADDR="${MASTER_ADDR:-$(echo "${RAY_ADDRESS}" | sed -E 's#https?://([^:/]+).*#\1#')}"
fi
echo "RAY_ADDRESS=${RAY_ADDRESS}  MASTER_ADDR=${MASTER_ADDR}  SINGLE_NODE=${SINGLE_NODE}"

# --- Debug toggles (override at launch) ---
# CORRECTNESS: aiter=1 is REQUIRED. The sglang515 overlay stores MoE weights in aiter's
# shuffled layout, so the non-aiter path reads them unshuffled and SILENTLY produces
# gibberish (Primus-Instella-Internal#2: "2+2=5", arc_challenge 0.195, below chance).
# Throughput looks normal, so there is no crash to alert you -- hence the default is 1.
# aiter=1 + dp-attention is now the correct AND stable combo: the empty-KV out-of-bounds
# fault that used to force ENABLE_DP_ATTENTION=0 is fixed by the k.shape[0]==0 guard in
# sglang's triton_backend (commit b54ceda), verified by clean lm_eval runs at EP8/DP8
# (gsm8k flexible-extract 0.848-0.892) and by 20 min of healthy generation in the 1-node smoke.
#
# ONE open caveat, to watch rather than act on: in a 3-node lm_eval sweep at a 32768 cap with
# GREEDY decoding, 112 of 2638 dp-attention docs degenerated into BOS-token spam (~650k chars)
# and ran to the cap, which the tp1/ep1/dp8 replica layout did not do. Rollouts differ on both
# axes that plausibly matter -- temperature 1 rather than greedy, and a 16384 cap -- and the
# 1-node smoke logged rollout/repetition_frac 0.0 on every cycle, so it has not appeared in
# training. repetition_frac is the canary: if it lifts off zero, come back to this.
SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
# Kernel serialization: DEBUG_KERNEL_SERIALIZE=1 pins the true faulting kernel for
# ROCm async memory faults (adds HIP_LAUNCH_BLOCKING + AMD_SERIALIZE_*). Slow; smoke only.
if [ "${DEBUG_KERNEL_SERIALIZE:-0}" = "1" ]; then
  SERIALIZE_ENV=",
    \"HIP_LAUNCH_BLOCKING\": \"1\",
    \"AMD_SERIALIZE_KERNEL\": \"3\",
    \"AMD_SERIALIZE_COPY\": \"3\""
else
  SERIALIZE_ENV=""
fi

# Pin NCCL/Gloo to the routable data interface. Left to auto-select, NCCL binds every
# interface it can see -- and because the containers run with --network host that includes the
# docker bridges present on these nodes (docker0 172.17.0.1/16, br-* 172.18.0.1/16) alongside
# the routable ens3. Cross-node collectives then stall: on the Qwen 3-node runs a 1 GiB
# broadcast that takes 0.13s pinned did not complete in 90s unpinned. This matters ONLY on
# multi-node, so the 1-node smoke cannot catch it.
NET_IF="${NET_IF:-$(ip -o -4 route get 1 2>/dev/null | grep -oP '(?<=dev )\S+' | head -1)}"
NET_IF="${NET_IF:-ens3}"
echo "NCCL_SOCKET_IFNAME=${NET_IF}"

# MEGATRON_PATH first so the FarSkip Megatron wins; FULLY_ASYNC_DIR for the rollout
# module. gate_proj bridge mapping comes from the patched installed mbridge (no module
# needed here). INSTELLA_DIR kept for convenience.
#
# The IF_* vars configure if_reward.py and are inert for math runs (nothing imports it);
# the CODE_* vars do the same for code_reward.py (run_code.sh).
# They have to be listed HERE rather than merely exported before launch: the reward runs
# inside the Ray rollout workers, whose environment is exactly this runtime_env, so an
# `export IF_REQUIRE_THINK=0` in the launching shell would otherwise be silently dropped
# and the worker would fall back to the module default.
#
# NLTK_DATA points at shared NFS so the IFEval word tokenizer finds punkt_tab without
# each of the 24 workers racing to download it at import (IFEvalG/instructions_util.py:25
# calls nltk.download on a miss). Seed it once with setup_if_deps.sh.
#
# PYTORCH_ALLOC_CONF matches Instella-MoE/rl if_rl exactly. Do NOT reintroduce
# max_split_size_mb: dynamic batching hands the allocator a DIFFERENTLY-sized fp32 logits
# tensor ([tokens, 128896], multi-GB) every microbatch, and capping the split size makes each
# distinct large size a non-reusable cached block, so the pool grows monotonically until the
# ROCm runtime can no longer allocate kernel scratch. That surfaces as an
# HSA_STATUS_ERROR_OUT_OF_RESOURCES abort at an NCCL dispatch with ~0 MB free and NO torch
# OOM -- exactly how the 20260819_0122 (30720 len) and _0233 (16384 len) smokes died, which
# is why halving the response length did not help. HSA_NO_SCRATCH_RECLAIM is deliberately
# unset (neither reference recipe sets it); pinning scratch compounds the same failure.
# Canonical name is PYTORCH_ALLOC_CONF; PYTORCH_HIP_ALLOC_CONF is deprecated and warns.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${FULLY_ASYNC_DIR}:${INSTELLA_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"NCCL_TIMEOUT_MS\": \"36000000\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",
    \"NVTE_CK_USES_BWD_V3\": \"1\",
    \"DEPRECATED_MEGATRON_COMPATIBLE\": \"1\",
    \"FARSKIP_REFERENCE_DECODER_LAYER\": \"1\",
    \"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION\": \"32768\",
    \"SGLANG_USE_AITER\": \"${SGLANG_USE_AITER}\",
    \"SGLANG_ROCM_FUSED_DECODE_MLA\": \"0\",
    \"GPU_MAX_HW_QUEUES\": \"8\",
    \"IF_REWARD_SCALE\": \"${IF_REWARD_SCALE:-1.0}\",
    \"IF_PARTIAL_CREDIT\": \"${IF_PARTIAL_CREDIT:-1}\",
    \"IF_REQUIRE_THINK\": \"${IF_REQUIRE_THINK:-1}\",
    \"IF_REWARD_CONCURRENCY\": \"${IF_REWARD_CONCURRENCY:-128}\",
    \"IF_LOG_EVERY\": \"${IF_LOG_EVERY:-200}\",
    \"IF_IFBENCH_DIR\": \"${IF_IFBENCH_DIR}\",
    \"NETRC\": \"${NETRC:-}\",
    \"NLTK_DATA\": \"${NLTK_DATA}\",
    \"CODE_EXEC_API_URL\": \"${CODE_EXEC_API_URL:-}\",
    \"CODE_EXEC_TIMEOUT\": \"${CODE_EXEC_TIMEOUT:-5}\",
    \"CODE_MAX_TESTS\": \"${CODE_MAX_TESTS:-32}\",
    \"CODE_PARTIAL_CREDIT\": \"${CODE_PARTIAL_CREDIT:-1}\",
    \"CODE_REQUIRE_THINK\": \"${CODE_REQUIRE_THINK:-1}\",
    \"CODE_REWARD_SCALE\": \"${CODE_REWARD_SCALE:-1.0}\",
    \"CODE_REWARD_CONCURRENCY\": \"${CODE_REWARD_CONCURRENCY:-64}\",
    \"CODE_LOG_EVERY\": \"${CODE_LOG_EVERY:-200}\",
    \"NCCL_SOCKET_IFNAME\": \"${NET_IF}\",
    \"GLOO_SOCKET_IFNAME\": \"${NET_IF}\"${SERIALIZE_ENV}
  }
}"

set -x
ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${REPO_DIR}/train_async.py" \
   "${MODEL_ARGS[@]}" \
   "${TOPOLOGY_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${ASYNC_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
