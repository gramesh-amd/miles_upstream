# v7 data cleaning: shared brief

Goal: a clean code-RL training set built from the current Dolci code_rlvr data, with every data error fixed
or the row dropped. Each fix is verified by execution where possible. Each fix must be standard, legitimate
practice: no eval data, and nothing tuned to an eval benchmark.

## Input (local)
- `input/code-train/dolci-code-train.jsonl`: 21,385 rows, built with miles' prepare_code_data.py from
  allenai/Dolci-Think-RL-7B code_rlvr. This is the exact format training consumes:
  - `prompt`: [{role: user, content}].
  - `label`: a JSON list of tests. Either stdio `{input, output}` or assert strings. Capped at 32 tests.
  - `metadata`: rm_type code, split, n_tests, dataset_source.
- `input/row_ids.jsonl`: one entry per row (same order): row, custom_id (Dolci id), src
  (acecoder 10,107 / klear 6,272 / synthetic2 3,000 / nemotron 2,006), olmo_passrate.
  195 rows are exact prompt+label duplicates.
- Full uncapped tests and OLMo's own sampled solutions are in the Dolci parquet cache:
  `~/.cache/huggingface/hub/datasets--allenai--Dolci-Think-RL-7B/snapshots/*/data/*.parquet`.
  - Columns: ground_truth (all tests), outputs (OLMo generations, which act as candidate reference solutions),
    passrate, custom_id, prompt.
  - Join on custom_id. Use `examples/instella_think/prepare_code_data.py` parse_tests to decode ground_truth.
- More candidate solutions from our own runs: v6 training rollouts in `/tmp/v6roll/samples/step_*.jsonl.gz`
  (reward 1.0 = all tests pass). Join to prompts via `/tmp/v6roll/problems.jsonl.gz` (prompt + label).

## Grader (what "correct" means at train time)
- `/tmp/wt_lux/examples/instella_think/code_reward.py` and `code_exec_server.py` (branch instella-code-rl-lux).
- Routes:
  - stdio: exact match after strip/rstrip per line. Read the code to confirm normalisation.
  - script: assert strings appended to the program and run as `__main__`.
  - call: the LeetCode-style harness.
- CODE_MAX_TESTS=32, timeout 5 s per test. Partial credit = fraction of tests passed.
- Known grader holes: no float tolerance; no case folding for YES/NO; `sys.exit(0)` inside
  `if __name__=="__main__"` skips the appended asserts.
- Do NOT edit the grader. Propose grader changes in your report; one person will implement them.

## Prior findings to reuse (re-verify; do not trust blindly)
- `../v5_assessment/litreview/prompt_audit/`: REPORT.md, row_flags.csv.gz, row_issues.csv.gz,
  stdio_format.csv.gz, multi_answer_flags.csv.gz, s2_never_solved_diagnosis.csv. These are keyed by custom_id.
- `../v5_assessment/datamix/function/`: earlier fixes for AceCoder and Nemotron, with scripts.
- `../v6/rollout_analysis/function_sources/REPORT.md` and `stdio_sources/REPORT.md` (+ v7_row_actions.csv): defect
  counts. Examples:
  - 771 prompts never name the tested function.
  - Stub floor of 50% or more on 20% of AceCoder rows; 232 rows a constant fully passes.
  - 276 rows never solved, with one blocking test.
  - 703 Synthetic-2 rows whose tests are not stdin text.
  - Multi-answer rows without a checker.
  - Exact float compares.
  - Labels contradicting the statement's own sample (114 Synthetic-2, 98 Klear).
  - 56% of Nemotron rows reuse prompt examples as tests.
- Mapping caveat: earlier v6 analyses keyed rows by a 400-character prompt prefix, which misassigned about 1,100 rows.
  Key by custom_id or by exact prompt+label from now on.

## Eval policy (user decision, Nov 14)
- Headline benchmarks: LCB v5/v6, Codeforces, HE-Pro/MBPP-Pro, CRUXEval. LCB v3 is the dev set, used for
  checkpoint selection.
- Training-data date cutoff: before 2024-08-01.
- Overlap with headline sets: drop. Overlap with LCB v3 (2023-05..2024-07): keep and flag `metadata.lcb_v3_overlap`.
- Confirmed HumanEval+/MBPP+ paraphrases: drop.

## Rules
- Fix only correctness: tests match the statement; the interface the tests need is stated; tests are
  non-trivial; the statement is complete.
  - Do NOT add eval-style instructions or answer-format suffixes, and do not change the reasoning register.
    That is a separate decision.
  - Do NOT use any eval benchmark data (LiveCodeBench, HumanEval(+), MBPP(+), CRUXEval, Codeforces eval sets)
    to fix or select rows.
- Each fix must be verifiable:
  - Re-run the fixed tests against an independent reference solution (OLMo outputs or our solved rollouts).
  - Keep a row only if at least one reference passes all of its tests. Otherwise flag it "unverified", or
    drop it with a reason.
- Never hand-edit an expected output to match a model's answer unless the statement plus a brute-force
  check proves the label is wrong. In that case, record the proof.
- Execute untrusted code only in a sandbox: subprocess, timeout, temp dir, no network, resource limits. Never
  exec it in your own process.
- Write to your own subfolder under `/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/<topic>/`:
  - `fixed.jsonl`: training schema, plus metadata.custom_id, metadata.fixes [list of fix codes] and
    metadata.verified true/false.
  - `dropped.csv`: custom_id, reason.
  - `REPORT.md`: counts per defect, before/after, verification rates, and 10+ worked examples.
  - Scripts.
- Do not launch cluster jobs. Do not print credentials. Keep REPORT.md concise and quantitative.
