# v7 new stdio sources: scoping (2026-11-18)

Scope: research plus light counting. No solution code was executed, no cluster jobs were run, and nothing was pushed. Every dataset checked is public and ungated on HF; none needed a workaround. `likesaurus/TACO-verified` does not exist; the real repo is `likaixin/TACO-verified`.

## Bottom line
- **About 14.1K new unique stdio problems** survive all of the following: the cutoff (released before 2024-08-01), decontamination, dedup against v7, ≥5 tests, not interactive, and a usable checker when the problem is multi-answer. Today's v7 stdio set is 3,707 rows (3,265 verified).
- **Union composition:**

  | source | problems |
  |---|---|
  | open-r1/codeforces `verifiable` | 6,271 |
  | HardTests (AtCoder 2,204, CodeChef 2,546, HackerEarth 776, Aizu 478, CF 342, other 63) | 6,409 |
  | CC+ 1x "verified" (TPR and TNR ≥ 0.9), net of open-r1 | 890 |
  | TACO-verified plus DeepCoder, net of the above | 520 |

  File: `recommended_union.csv.gz`.
- **Projection.**
  - Sandbox verification should keep roughly 70–85%. In the v5 audit, references passed 84–88% of these sources' rows, and only 71–81% of accepted CF Python submissions run within 5 s under CPython. That leaves **about 10–12K verified**.
  - A DPO-base difficulty band would then leave about 6–8K. The teacher proxy, known for 82% of the union, puts 4,864 problems at a solve rate above 0.75 and 3,387 at 0.
- **Everything else is subsumed.** TACO-verified, both DeepCoder subsets, INTELLECT-3-RL, Klear, Skywork, Eurus-2, SYNTHETIC-2 and Nemotron-RL add only 5–634 unique clusters each once open-r1, CC+ and HardTests are in (`union_cumulative.csv`). Nemotron-RL is also contaminated with Codeforces-eval problems.
- **Two decontamination findings matter for any source:**
  - **Text matching misses a cross-lingual reuse.** CF 646B (2016) was reused in 2024 as eval problems 2010/C1 and C2. Its copies in 10 sources (open-r1, TACO-verified, both DeepCoder subsets, Klear ×2, Skywork, SYNTHETIC-2, Eurus-2, plus the cluster's HardTests copy) share 49 of the eval's hidden tests. Only the test-overlap detector catches it, and the cluster is now dropped.
  - **Nemotron-RL contains 184 Codeforces-eval rows (by text) plus 42 (by tests).** These rows are undated, so the date cutoff alone would not remove them.

## Per-source table (stdio rows; counts are rows, the last column gives unique clusters)
"after decontam + dedup" = stdio, released before 2024-08-01, no eval hit in the row's cluster, and no v7 row in the cluster. "Checker coverage" = the share of multi-answer rows with a checker our grader can run (a Python checker, or the grader's `case_insensitive` / `float_tol` options).

| source (HF id) | license | raw | pre-2024-08 | after decontam + dedup vs ours | ≥5 tests | checker coverage | final (not interactive, checker if multi) / unique | used by | recommendation |
|---|---|---|---|---|---|---|---|---|---|
| open-r1/codeforces `verifiable` (train 9,556; verifiable 8,338) | CC-BY-4.0 | 8,338 | 8,338 (all ≤ 2024-07-30) | 6,810 | 6,654 | **86.5%** (1,643 R1-generated Python checkers) | 6,414 / 6,271 | OlympicCoder (SFT); Nemotron-RL / Cascade proxy; OCR-2 | **Use (1st).** Strongest tests (WA-submission FPR 1.6% in v5), dates and ratings |
| ByteDance-Seed/Code-Contests-Plus 1x | CC-BY-4.0 | 11,690 | 11,688 | 8,893 | 8,841 | 6.5% (91% of multi-answer rows have only C++ testlib checkers) | 7,112 / 6,589; TPR/TNR ≥ 0.9: 3,209 | CC+ paper RL ablation (Seed) | **Use the verified subset (+890 net).** Up to 1.9K more (before dedup) if the grader gains testlib checkers |
| sigcp/hardtests_problems + hardtests_tests | MIT (card) | 47,136 | 29,890 (17,200 undated: Luogu 10.1K, SPOJ 3.5K, AtCoder 1.8K, UVa 1.4K) | 26,031 | 13,559 (tests exist for only 57%) | 2.7% | 12,391 / 11,619 | HardTests paper RL ablation | **Use (+5.2–6.4K net).** The only big AtCoder pool. Tests must be fetched and verified |
| likaixin/TACO-verified | MIT | 10,261 | 10,217 (TACO ≤ 2023-03) | 7,911 | 4,292 | 0% | 3,904 / 3,693 | DeepCoder, Code-R1, Skywork (TACO) | Subsumed (+59). 33% const-FP in v5 |
| agentica-org/DeepCoder-Preview `taco` | MIT | 6,387 | 6,385 | 4,265 | 4,261 | 0% | 3,874 / 3,666 | DeepCoder | Subsumed (+44) |
| DeepCoder `primeintellect` | MIT | 14,998 | 14,909 | 10,414 | 10,409 | 0% | 9,080 / 4,515 | DeepCoder; Klear (rllm-derived); INTELLECT-3 | Subsumed (+105). `lcbv5` split = LCB 2023-05..2024-07: drop wholesale |
| deepmind/code_contests | CC-BY-4.0 | 13,328 | 13,320 | 10,361 | n/a (meta only) | none | — | Seed-Coder, AceReason, Nemotron-RL | Superseded by CC+ (same problems, better tests) |
| PrimeIntellect/SYNTHETIC-2-RL (code) | none stated | 12,576 | 12,215 | 9,319 | 7,121 | 0% | 6,214 / 4,725 | OLMo 3 (Dolci) | Already our source; +417. Reserve |
| PRIME-RL/Eurus-2-RL-Data (code) | MIT | 22,618 | 22,108 | 17,365 | 13,482 | 0% | 11,639 / 8,209 | PRIME/Eurus | +371. Skip |
| Kwai-Klear/KlearReasoner-CodeSub-15K | Apache-2.0 | 13,575 | 13,540 | 8,487 | 8,458 | 0% | 7,854 / 3,341 | Klear; OLMo 3 (Dolci) | Already used; +5. It contains 81 LCB rows (text) plus 73 rows with LCB AtCoder tests |
| Skywork/Skywork-OR1-RL-Data (code) | none stated | 12,017 | 11,979 | 9,377 | 4,243 | 0% | 3,832 / 3,660 | Skywork-OR1 | +44. Skip. 432 LCB LeetCode hits, all windows |
| nvidia/Nemotron-RL-coding-competitive_coding | CC-BY-SA-4.0 | 16,083 | 15,723 | 12,714 | 10,055 | 0% | 8,444 / 6,767 | Nemotron-Cascade (released proxy) | **Avoid.** CF-eval and open-r1 test-split leakage; +94; share-alike |
| PrimeIntellect/INTELLECT-3-RL `code` (new) | none stated | 8,579 | 8,514 | 6,024 | 6,024 | 0% | 5,077 / 3,071 | INTELLECT-3 | DeepCoder repackaged: PI 5,720, TACO 2,836, lcbv5 23. +634 TACO problems whose other copies lack tests. Reserve |
| microsoft/rStar-Coder `seed_testcase` (1 of 30 shards) / `synthetic_rl` (398K) | CC-BY-4.0 | 979 / 398,107 | 978 / 0.9% dated | 795 / — | 671 / outputs not local | 0% | 536 / 533; synthetic: — | rStar-Coder (SFT) | Seed: +16 per shard. Synthetic: undated, QwQ mutual-verified outputs. Reserve only |
| BAAI/TACO | Apache-2.0 | 25,443 | 22,700 | 19,366 | n/a (meta only) | none | — | Skywork, Code-R1 (via verified) | Weak tests; use TACO-verified/HardTests copies |
| nvidia/OpenCodeReasoning(-2) | CC-BY-4.0 | 34,125 questions | n/a | n/a | no tests (blank questions) | — | — | OCR SFT | Labels only: R1 solve rate as a difficulty proxy |
| PrimeIntellect/verifiable-coding-problems | none stated | 144,169 | n/a | n/a | LLM-written tests | — | — | DeepCoder (16K verified subset) | Superset of DeepCoder-PI; skip |
| KodCode V1 / Light-RL-10K | **CC-BY-NC-4.0** | 447K / 10K | — | — | pytest, function-style | — | — | KodCode; rejected by DeepCoder | Not stdio, NC licence: skip |
| AceReason RL code data | — | not released | — | — | — | — | — | AceReason 1.0/1.1 | AceReason-1.1-SFT (CC-BY-4.0, 93 GB) has no tests |
| caijanfeng/CodeContests-O (new, 2026-01) | Apache-2.0 | 11,682 | ≤ 2021 (CodeContests) | not counted (325 GB) | ~40 per problem via generators + checkers | testlib | — | CodeContests-O paper (SFT) | Possible upgraded test source for CC problems. Needs generator execution |
| nvidia/Nemotron-Competitive-Programming-v1 | CC-BY-4.0 | 190 GB SFT | — | — | no tests | — | — | Nemotron SFT | Skip |

Other columns are in `funnel.csv`:
- Images (no filter applied): CC+ 23.5%, INTELLECT-3 21.7%, code_contests 21.3%, DeepCoder-PI 13.3%, Nemotron 12.2%, HardTests 7.8%, open-r1 0.8%. The recommended union contains 667 such rows; apply the MERGE_NOTES decorative-image rule.
- Interactive or file I/O: 0–1.5% per source.
- Multi-answer: 5–23% per source.

## Cross-source overlap (`overlap_matrix.csv`: share of A's surviving clusters that also contain a B row)
- HardTests and BAAI/TACO overlap almost every other source, at 96–99% and 74–99%. Both are supersets of the CC/TACO/CF pools.
- Klear, TACO-verified, DeepCoder-taco and Skywork overlap each other at 77–98%.
- open-r1's survivors appear in Nemotron-RL (91%), HardTests (97%) and CC+ (68%).
- Dedup against our current rows: 15–37% of each source's decontaminated rows are already in v7 (`dup_v7`). Another 1,148 problems in the union are Dolci problems that v7 dropped; they come back with new tests.

## Method (reused from `decontam/REPORT.md`; scripts in `scripts/`)
1. **Corpus (`build_corpus.py`).**
   - Statements and flags come from the v5 stdio audit (`/tmp/datacand/norm`, `v5_assessment/datamix/stdio/audit/all_rows.parquet`).
   - Added sources: code_contests, BAAI/TACO, INTELLECT-3-RL (96 MB downloaded to `raw/`), and a 30K sample of rStar synthetic (counts are reweighted).
   - Our rows: v7 stdio (3,707), v7 function (10,153) and all 21,385 Dolci rows.
2. **Dates.**
   - Own date first: CF `contest_start`, or Codeforces `contest.list`, cached at `raw/cf_contest_list.json`. Then the v5 cluster date, then the source's vintage bound (TACO ≤ 2023-03, CodeContests ≤ 2021). Then propagation through the new clusters, taking the latest dated member.
   - Cutoff is `date < 2024-08-01`. Undated rows are excluded.
   - 1,104 union rows fall inside the LCB v3 window (2023-05..2024-07). They are not LCB problems; flag them as `lcb_v3_window` per BRIEF.
3. **Decontamination (`overlap.py`, `testio.py`).** The eval corpus is v7 `eval.pkl`, plus the open-r1 CF test split (468, a superset of the CF eval's 422) and DeepCoder's CF test set (408, reported only, not dropped).
   - The v7 `eval.pkl` contains: LCB, all 1,055 problems; the Codeforces eval, 422 problems from contests 1993–2067 dated 2024-08-04..2025-02-11 (`codeforces_eval_ids.csv`); HE+/MBPP+; HE-Pro/MBPP-Pro raw and new; CRUXEval-I/O.
   - Detectors:
     - 8-gram coverage ≥ 0.5 (Tulu 3 / OLMo 3).
     - 13-gram ≥ 0.2, the v7 recall flag. With no hand check yet, these rows are dropped too.
     - Boilerplate n-grams are removed: those in ≥ 5 eval items or ≥ 100 corpus rows.
     - CF and AtCoder id equality.
     - Test I/O hash overlap with LCB / CF-eval hidden tests (≥ 4 shared non-trivial pairs; v7 false positives share 1–3). It runs on open-r1 `official_tests` and on the inline tests of the other sources.
   - A hit anywhere in a cluster drops the whole cluster. Hits by source: `contam_by_source.csv`; pairs: `text_overlap_hits.csv.gz`, `test_overlap_hits.csv`.
   - **LCB problems from before the cutoff (the LCB v1–v4 window) are dropped as well, per this task's "all versions" instruction.** This is stricter than BRIEF's "flag v3"; under the flag policy only Skywork's 273 rows change.
   - Spot checks, 9 pairs: all were the same problem, an easy/hard sibling, or a sibling in the same AtCoder contest with a shared statement.
   - HE+/MBPP+/Pro: 2 GfG/TACO rows (MBPP/300). CRUXEval: 0.
4. **Dedup.**
   - 8-gram hashes subsampled 1/4. Containment |A∩B| / min(|A|, |B|) ≥ 0.5, with ≥ 8 sampled grams on both sides. Plus CF/AtCoder id edges. Union-find gives 90K clusters; the largest has 62 rows.
   - Hand check of 5 borderline edges (containment 0.5–0.6, 1.4% of v7 edges): 3 true duplicates and 2 variants (bounded vs 0/1 knapsack; remainder vs quotient). The dedup is slightly conservative.
5. **Filters.**
   - ≥5 tests uses the known test count; HardTests tests exist for only 57% of rows.
   - Not interactive and not file I/O.
   - Multi-answer means a statement regex or a source checker flag. Such a row needs an open-r1 Python checker, a HardTests `judge_fn`, or a CC+ `yesno_ci`/`float` checker (grader options). CC+ `custom` checkers are C++ testlib, which our `/test_program_checker` (Python-only) cannot run.

## Limitations
- Test counts for CC+ are mostly estimated from metadata.
- HardTests test quality was measured on only 412 problems in v5: reference pass 0.84, WA-submission FPR 4.8%.
- The test-overlap detector saw only official or inline tests, not the full generated suites.
- The text detectors are ASCII-word based, so Chinese Luogu statements are not covered. They are excluded as undated anyway.

## Next steps (sandbox agent; nothing below has been run)
1. **Materialise tests, on `/tmp` local disk (594 GB free); the share has only 433 GB free.**
   - open-r1: inline pools already exist (`/tmp/datacand/pools_full/openr1_cf_verifiable.jsonl.gz`, 5,862 problems, FETCH_DONE). Fetch the missing ~400 from `verifiable_tests`.
   - CC+: fetch 1x tests for the 890 (pools_full holds 978 earlier picks).
   - HardTests: stream only the shards that contain the 6,409 pids (`/tmp/datacand/meta/idx_hardtests.parquet` maps pid → shard; the full set is ~378 GB), deleting each shard after extraction.
2. **Verify in the sandbox with the exact v7 grader:** CPython, 5 s per test, 32 tests (8 largest + 24 random), and the checker route for open-r1 rows.
   - References: accepted Python submissions (`open-r1/codeforces-submissions`), CC+ correct submissions, HardTests oracle solutions (94% have Python), TACO solutions.
   - Keep a row only if a reference passes all tests.
   - Run WA submissions for FPR, and a constant-output probe.
   - Validate each Python checker: it must accept the reference output and reject a perturbed one.
3. **Difficulty band:** run DPO-base 8× rollouts at T=1.0. Drop 8/8, and drop 0/8 where the teacher solve rate is also 0.
4. **Decontam follow-up.**
   - Rerun v7 `detect.py` (TF-IDF, numeric-example, full test I/O on the materialised suites) on the final union, then hand-check the 13-gram-only flags (41 pairs).
   - Apply the image rule.
   - Hand-review the 1,104 `lcb_v3_window` rows.
5. **Grader proposal (for the grader owner).** Compile and run CC+ testlib checkers. That would unlock up to 1.9K CC+ multi-answer problems (before dedup) and HardTests/CodeContests-O checkers.

## Files
- `funnel.csv`, `contam_by_source.csv`, `overlap_matrix.csv`, `union_cumulative.csv`.
- `recommended_union.csv.gz` with `recommended_union_summary.json`.
- `codeforces_eval_ids.csv`, `text_overlap_hits.csv.gz`, `test_overlap_hits.csv`.
- `cards/` (dataset cards), `raw/` (CF contest list, INTELLECT-3 code), `work/` (corpus, overlap, per-row funnel), `scripts/`.
