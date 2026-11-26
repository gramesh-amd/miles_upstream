# Data build scripts for Instella-MoE RL

These are the scripts that built the training sets used by the math, IF and code runs. They are copied as they
were run, so paths point at the MI325 share (`/it-share-prj2-1/data/goramesh-instella-rl`, called `$ROOT` below)
or at `/workspace` (the same share mounted inside the `instella_rl` container). Change those paths before reusing.

The built datasets are in the bucket `s3://ocieobject1/goramesh/instella-rl-mi325x/data/`:

| Run | Dataset | Rows | Bucket folder | Scripts |
|---|---|---:|---|---|
| Math R1 | `mix_v5_math.jsonl` | 25,162 | `lux_math_r1/` | `math/` |
| IF3 | `if_train_hybrid20k.jsonl` | 20,000 | `lux_if_if3/` | `if/` |
| CODE1 | `final_v3_candidate/train.jsonl` (`code1_train.jsonl`) | 13,286 | `lux_code/` | `code1/` |
| CODE2 | `v8_data/train.jsonl` (`code2_train.jsonl`) | 24,572 | `lux_code/` | `../data_v8/` |

All difficulty grading uses our DPO checkpoint (`Instella-MoE-16B-A3B-DPO`) with 8 rollouts per prompt.

## Math: `mix_v5_math.jsonl`

Built around Polaris prompts and graded with our own DPO model. Keep everything from 0/8 to 7/8 and drop only 8/8.

1. `prepare_rl_dolci_math.py`: export the math subset of Dolci-Think-RL (OLMo 3) as miles JSONL.
2. `mixv4_recover_grades.py` (uses `oci_ls.py` and `~/.oci_s3_creds.env`): recover the per-prompt pass rates
   already measured on DPO over the 30,180-prompt Dolci pool. Writes `out/analysis_1012/dolci_filter/pool_30k.jsonl`
   and `grades_30k.jsonl`.
3. `mixv4_fetch_dapo.py`: download DAPO-Math-17k-Processed. Writes `out/analysis_1012/dapo/dapo_17k_en.jsonl`.
4. `mixv4_build_polaris.py`: take Polaris difficulty bands 1–5 and dedup against the prompts above.
   Writes `out/analysis_1012/polaris/pool_band1_5.jsonl`.
5. Grade DAPO and Polaris with DPO: `mixv4_grade_parallel.sh` runs `filter_rl_difficulty.py` (8 rollouts,
   T=1.0, top_p 1.0, 32K response) on every node, serving with `serve_sglang_farskip_dp.sh`.
   `grade_chain.sh` / `polaris_chain.sh` queue grading jobs; `grade_supervisor.sh` + `grade_remaining.py` restart
   lost nodes and re-cut ungraded chunks. Grades land in `out/analysis_1012/{dapo,polaris}/grades/`.
6. `build_mix_v4.py`: join the three graded sources, keep 0/8–7/8, drop OMEGA and ZebraLogic, dedup across
   sources, decontaminate against AIME. Writes `data/mix_v4_math.jsonl` (25,162).
7. `build_mix_v5.py`: append the eval's `\boxed{}` instruction (byte-identical to `aime-2024-olmes.jsonl`) to every
   prompt. Writes `data/mix_v5_math.jsonl`.

`mix_v2_math.README.md` is the build record of the earlier mix_v2 (too easy; kept for history).

## IF: `if_train_hybrid20k.jsonl`

`if2_pipeline.sh` runs the whole chain unattended (generate, score, assemble, then launch IF3).

1. `if_build_mix2.py` (in container): draw 40,000 candidates from `IF_multi_constraints_upto5`, stratified by
   domain and constraint count, keeping only constraint sets IFEvalG allows together. Each row stores the exact
   chat-templated prompt. Writes `data/if/if_candidates2.jsonl`.
2. `if_prefilter2.py` starts the sglang servers (via `bench_run.py`); `if_prefilter2_async.py` sends 8 rollouts per
   prompt at the training settings (T=0.8). Writes `out/if_prefilter2/*.jsonl`.
3. `if_partial_scores.py` (in container): score each rollout with the IF reward. Writes
   `out/if_prefilter2_scores/partial_scores_mat.jsonl`.
4. `if_assemble_hybrid.py` (imports `if_assemble.py`): keep prompts whose 8 hybrid rewards differ, whose strict
   pass rate is ≤ 0.625, and that are satisfiable; re-stratify by domain and constraint count to 20,000.
5. `if_make_eval.py`: write `eval_ifeval.jsonl` (541) and `eval_ifbench.jsonl` (300).

## Code: CODE1 (`final_v3_candidate`)

CODE1 is `final_v2` (13,399 rows) cleaned and re-mixed by `build_v3/build_v3.py --mix B --write` (seed 0,
deterministic). `final_v3_REPORT.md` lists every step and count; `data_audit.md` has the evidence for each change.

- `build_v3/`: the final step. Dedup (same Codeforces id, shared tests, MinHash), restore grader-verified rows
  that `final_v2` dropped, add Open-R1 checker rows, keep 1/3 of the easy AceCoder rows, and add a second copy of
  each stdio row. `sanity_v3.py` and `final_stats.py` check the result.
- The stages before it cleaned the 21,385-row Dolci code set (the v5 data, from `prepare_code_data.py`) into
  `final_v2`:
  - `function/` and `stdio/`: verify tests and reference solutions per route, fix or drop broken rows.
  - `decontam/`: n-gram and execution checks against LCB, HumanEval+, MBPP+ and Codeforces evals.
  - `new_sources/`: survey of extra stdio sources (code_contests, TACO, INTELLECT-3) and their overlap with ours.
  - `final_v1/`: first assembly.
  - `final_v2/`: restore dropped Codeforces rows with Open-R1 statements and checkers, verify the checkers on Lux,
    and assemble.
- `v7_data_BRIEF.md` and `MERGE_NOTES.md` are the original task brief and merge rules for these stages.

CODE2 (`v8_data`) is built by `../data_v8/build_v8.sh`, already in this repo.
