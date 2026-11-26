# IF: `if_train_hybrid20k.jsonl` (20,000 prompts)

Prompts from AI2's IF_multi_constraints_upto5 pool (IFEval + IFBench-train constraints), prefiltered with our DPO
model at the training settings and balanced by domain and constraint count. Trained with the hybrid reward
`0.5 × fraction of constraints met + 0.5 × all met` (`../../if_reward.py`, `IF_STRICT_WEIGHT=0.5`).

Paths below are relative to `$INSTELLA_RL_ROOT`.

## Steps

| # | Script | Does | Writes |
|---|---|---|---|
| 1 | `if_build_mix2.py` (container) | Draw 40,000 candidates, stratified by domain and constraint count, rarest verifier first. Skip constraint pairs IFEvalG marks as conflicting. Store the exact chat-templated prompt (`rendered`). | `data/if/if_candidates2.jsonl` |
| 2 | `if_prefilter2_async.py` | 8 rollouts per candidate from DPO at the training settings (T=0.8, top_p 1.0, 32K) against running `../serve_grader.sh` servers. Resumable. | `out/if_prefilter2/*.jsonl` (39,830 prompts) |
| 3 | `if_partial_scores.py` (container) | Score every rollout per constraint with `if_reward.py`. | `out/if_prefilter2_scores/partial_scores_mat.jsonl` |
| 4 | `if_assemble_hybrid.py` | Keep prompts whose 8 hybrid rewards differ and whose strict pass rate is ≤ 0.625; re-stratify to 20,000. | `data/if/if_train_hybrid20k.jsonl` |
| 5 | `if_make_eval.py` | IFEval (541) and IFBench test (300) in the eval format. | `data/if/eval_ifeval.jsonl`, `eval_ifbench.jsonl` |

Commands:

```bash
# 1. in the container
python3 if_build_mix2.py --out /workspace/data/if/if_candidates2.jsonl

# 2. one grader per node, then the driver
MODEL=/workspace/models/Instella-MoE-16B-A3B-DPO CONTEXT_LENGTH=36864 PORT=30971 MAX_RUNNING=2048 \
    bash ../serve_grader.sh                                   # on each node
python3 if_prefilter2_async.py --hosts g2:<ip>,g34:<ip>,... \
    --candidates /workspace/data/if/if_candidates2.jsonl --out /workspace/out/if_prefilter2

# 3. in the container
GEN_GLOB='/workspace/out/if_prefilter2/*.jsonl' \
OUT=/workspace/out/if_prefilter2_scores/partial_scores_mat.jsonl NPROC=96 python3 if_partial_scores.py

# 4 and 5. on the host
python3 if_assemble_hybrid.py --out data/if/if_train_hybrid20k.jsonl
python3 if_make_eval.py
```

## What the filter did (IF3)

- 39,830 scored prompts. 11,307 dropped for no reward variance, 4,169 dropped as too easy (strict pass > 0.625).
- 24,354 kept, of which 9,911 passed strictly at least once. Re-stratified to 20,000 (3,350 filled from other
  cells because some cells ran short).
- Result: mean strict pass 0.137, mean hybrid reward 0.352. 59.8% of prompts are 0/8 strict but still have hybrid
  reward variance, so they give a learning signal.

Rebuild check: step 4 on the saved scores reproduces the shipped file byte for byte.
