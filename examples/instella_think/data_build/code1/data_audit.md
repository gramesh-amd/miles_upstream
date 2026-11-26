# Code-RL training data audit: final_v2 → final_v3 recipe

**Bottom line.**
- **final_v2 has two structural problems:**
  - Stdio is only 23.7% of the rows. v5 and v6 had about 43% of draws.
  - About 33% of rows are probably Instella-easy (≥6/8), mostly AceCoder.
- **It also carries about 450 cross-copies of the same problem.** The same Codeforces problem appears in Klear and SYNTHETIC-2 formatting, or as a Nemotron function row; the earlier MinHash missed these.
- **Restores add about 420 new learnable stdio problems.** That is 170 verified Open-R1 checker rows, 139 image-placeholder rows and 111 multi-answer rows. Before dedup there were 716 candidate rows; most were copies of each other.
- **Recommended mix B, built in `v7_data/final_v3_candidate/`:**
  - 10,555 unique problems and 13,286 draws;
  - 45% of draws are stdio, 13.6% are 0/8 rows, and an estimated 23% are base-easy;
  - all new rows are grader-verified, and the sanity check is 50/50.

Scripts and intermediates are in `v7_improve/data_audit/`. All untrusted code ran only inside `code_exec_server.py` (same sha256 as the training grader) on the two hold allocations (≤ 88 workers each). No sbatch or salloc was used, and nothing was uploaded.

## 1. Drop accounting, refined (v6 stdio rows)

v6 trained on 12,173 distinct prompts; 5,233 of them are stdio. Joining v6 `problem_id` to Dolci `custom_id` and following every v1/v2 duplicate link shows that these 5,233 rows are only about 3,541 distinct problems.
- 2,034 of those problems are still in final_v2 (1,853 rows directly).
- 1,817 dropped rows are duplicate copies whose problem is still trained: S7 1,324, P7 MinHash 261, v2 duplicate/replaced 232. These are not lost.
- The loss is **1,507 distinct problems (1,563 rows)**.

| category of lost problem (v6 stdio rows) | rows | v6 solved ≥ once | v6 mean solve | decision |
|---|---|---|---|---|
| no reference passes (P5 346 + v2 unverified 33) | 379 | 0.3% | 0.000 | keep dropped (broken) |
| multi-answer: deferred to the checker job | 310 | 70% | 0.22 | **checker run here: 170 of 240 problems accepted** |
| multi-answer: Open-R1 has no checker (Open-R1 grades them by token compare) | 165 | 84% | 0.32 | **restore with exact compare when there is evidence** |
| multi-answer: no CF mapping (S3) / v2 multi_answer_no_checker | 43 / 17 | 58% / 82% | 0.25 / 0.36 | **restore with exact compare when there is evidence** |
| image placeholder (S4 tag 139, phrase 18, Open-R1 statement still refers to a figure 87) | 244 | 82–89% | 0.23–0.32 | **restore if grader-verified** |
| < 5 tests (S lt3 130, P6 107, v2 11) | 248 | 87–92% | 0.32 | keep dropped: verified copies have only 1–4 tests (GPU test generation, R2) |
| label differs from the statement sample | 26 | 46% | 0.13 | keep dropped |
| a constant passes every test | 21 | 81% | 0.35 | keep dropped (reward hackable) |
| ambiguous Open-R1 mapping / empty / interactive | 14 / 14 / 6 | – | – | keep dropped |
| non-English | 10 | 100% | 0.39 | keep dropped (17 rows pool-wide, all verify; optional) |

**Caveat on v6 solve rates.**
- v6 used a pass-rate reward with the nonzero-std group filter, so groups with all 8 solved were never logged. The maximum per-problem solve is therefore 0.875, and "mean solve" is biased low for easy rows.
- "Solved at least once" is unaffected.
- The parent's earlier figures reproduce: multi-answer 458 rows (74% solved at least once), < 5 tests 237 (90%).
- The image count is 697 v6 rows (80%, mean 0.31). It differs from 556 because it includes the copies that v2 later recovered or deduplicated.

### Per-category evidence (pool-wide, lost problems only; `restore_cands.jsonl.gz`, `restore_verify.jsonl`)

Verification method:
- Up to 8 references per row: v6 solved rollouts first, then OLMo samples, then v6 near-misses.
- Tests: the Dolci label plus the uncapped Dolci ground truth, ≤ 32, each test ≤ 1 MB.
- Grader: `/test_program_stdio` with the row's `float_tol` / `case_insensitive` options, 5 s per test.

| group | candidates | a reference passes all tests | + ≥ 5 tests, not constant | + exact-compare evidence | new after dedup |
|---|---|---|---|---|---|
| image placeholder | 369 | 319 | 310 | – | **139** |
| multi-answer, Open-R1 token compare | 233 | 208 | 202 | 180 | **82** |
| multi-answer, no checker anywhere | 107 | 75 | 61 | 56 | **29** |
| < 5 tests | 583 | 358 (84 with 1 test, 138 with 2, 105 with 3, 31 with 4) | 0 | – | 0 |
| non-English | 17 | 17 | 17 | – | not added |

Exact-compare evidence means v6 solved the row at least once under exact compare, or a second independent reference prints the label exactly. Across the multi-answer rows, 618 of 1,040 extra reference runs matched the label exactly. 27 rows where 0 of N independent references match the label were kept dropped, because those are real alternative-answer problems.

**Image placeholders: is the image needed?** I hand-read 20 sampled statements.
- About 4 need the image content: a 7-segment numbering "as in the picture", a phone keypad layout, and two letter-conversion tables that exist only in a figure.
- About 16 are illustrations of rules the text already states: grid or town maps, a billiard table, bus routes, card layouts, domino or tic-tac-toe pictures, "see the figure for the delimiter definition".
- Restoring only grader-verified rows filters toward the second kind. A model (v6 policy or OLMo) solved each restored row from the text alone. The residual risk is a well-known problem being solved from memory (e.g. CF 1340B's 7-segment digits).
- Restored rows carry `flags.image_placeholder` so they can be filtered.

## 2. Multi-answer rows: Open-R1 checker verification (lux job 0, run on CPU)

`verify_checkers.py` needs:
- Python ≥ 3.8, stdlib only;
- `code_exec_server.py` beside it (it checks the sha256);
- `checker_candidates.jsonl.gz`;
- a program interpreter (`--python`).

It starts its own server on 127.0.0.1 with a fresh token and a scratch dir, and removes both at the end.

**Bug found.** The shipped script crashed after about 20 minutes. `perturb()` calls `int()` on a 5,000-digit answer, which hits Python 3.10's int-string limit. The script writes nothing until it finishes, so that run lost everything. The shipped file was not edited. A patched copy in `data_audit/lux_patched/` sets `sys.set_int_max_str_digits(0)`, catches per-problem exceptions (0 occurred), and adds `--shard k/n`. The rerun used the exact commands below; wall time was 34 + 44 min.

```bash
P=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit
PY=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_grader/venv/bin/python
srun --jobid 3828 --overlap -N1 -n1 python3 $P/lux_patched/verify_checkers.py --shard 0/2 --workers 88 --concurrency 64 \
  --port 18995 --python $PY --scratch /tmp/v7audit_chk_s0 --out-dir $P/checkers_s0
srun --jobid 3829 --overlap -N1 -n1 python3 $P/lux_patched/verify_checkers.py --shard 1/2 --workers 88 --concurrency 64 \
  --port 18996 --python $PY --scratch /tmp/v7audit_chk_s1 --out-dir $P/checkers_s1
```

**Result: 170 of 240 CF problems are accepted (they cover 389 rows).**
- 16 have no passing reference.
- 53 checkers are rejected. The failed gates overlap: R3 accepts a one-token change in 33, R2 accepts the constant output in 20, A2 rejects a trusted reference output in 12.
- 1 is a final_v1 duplicate.

The accepted checkers work as intended:
- They accept 4,603 of 4,796 (96%) alternative outputs, i.e. outputs that differ from the label.
- They reject 401 of 417 (96%) wrong-answer CF submissions.

All 170 survive dedup and replace 10 Nemotron function copies that have fewer tests. Next step: run R1 (GPU-generated checkers) on the 53 rejected and the 233 problems with no checker, using the same acceptance script.

## 3. Other issues in final_v2

**Contamination and dates.**
- Decontamination (v7_data/decontam) covered all 21,385 Dolci rows against LCB v3/v5/v6, HumanEval(+/Pro), MBPP(+/Pro), CRUXEval and the CF eval. The 20 confirmed rows are already out.
- Dated rows (CF/LeetCode, 6,033 mapped) run up to 2024-07-07, so nothing is after the 2024-08-01 cutoff.
- 4 Nemotron rows fall inside the LCB v3 window (2023-05 to 2024-07). None is an LCB problem, but **v3 drops them** because the cost is negligible.
- Undated rows:
  - AceCoder 8,322: synthetic, no date.
  - Nemotron 1,357.
  - Klear and S2 stdio 1,244. Of these, 351 are AtCoder-style and 193 AIZU/ICPC-style.
  - The residual risk is AtCoder ABC rounds from 2023-05 onward, which LCB uses. The n-gram, test-overlap and numeric-example detectors found 0 LCB hits, and Klear/S2's upstream pools (CodeContests, TACO) predate 2023, so the risk is low.
  - A cheap extra guard would be to map the AtCoder-style rows to task ids by title and date them.
- Later benchmarks (LCB v5 from 2024-08, v6 to 2025-04) are clean by date.

**Prompt format by source.**
- Klear stdio: 73% say "read stdin, print to stdout" (the Klear suffix).
- SYNTHETIC-2 stdio: only 10% mention stdin. Their suffix is "Write Python code … Present the code in ```python```".
  - This is not a problem in practice. In v6 rollouts, 99.6% of S2 stdio responses that contain code read stdin (2,560/2,569; Klear 7,014/7,023).
  - No prompt change is recommended; it would also move away from v5's prompts.
- Function rows name the tested function in ≥ 97% of rows. The misses are test-helper names like `_v7_close`.
  - A call with arguments or a `def` signature appears in 81% of AceCoder, 97% of Nemotron and 71% of S2-function prompts.
  - The v1 `Implement \`def f(...)\`` line already covers the rest.
- Only Klear has the "Solve the following coding problem…" header. S2 uses a "Title: … Time Limit: None seconds" header.

**Huge tests.**
- 33 final_v2 rows have labels over 1 MB; 13 of them are over 5 MB, and the largest is 17 MB.
- 16 stdio rows send more than 2 MB in the graded first 32 tests, and 13 rows contain a single test over 1 MB.
- v3 trims tests over 1 MB: 5 rows and 8 tests trimmed, 2 rows dropped for having < 5 tests left.

**Test time limits.** From the v7 stdio timings (2,856 of 3,177 rows timed), the fastest verified reference needs:
- more than 1 s on some test in 113 rows;
- more than 2 s in 47;
- more than 3 s in 26;
- more than 4 s in 18.

At a 5 s limit on a loaded grader, the >4 s rows give flaky rewards, so v3 drops those 18.

**Near-identical problems still in final_v2.** 316 CF ids have more than one train row (679 rows). Examples: Klear "Solve the following…" against S2 "Title: … Time Limit"; `10^9` rendered as `109` in one copy; a Nemotron function copy of a Klear stdio row.

Test overlap finds 376 pairs: an io pair or assert shared in ≥ 50% of the smaller test set, and ≥ min(5, n) shared. Many of these are pairs v1 had "kept apart" because their digits differed, yet they share 30+ of 32 test inputs. The real differences are formatting, e.g. `10^9` turned into `109`.

The v3 dedup uses the union of three rules:
- same CF id;
- test overlap (stdio ≥ 50%, function ≥ 80% and ≥ 5 shared asserts);
- MinHash ≥ 0.7 with equal digits or ≥ 50% shared inputs.

It drops 446 final_v2 rows:

| dropped copy | kept copy | rows |
|---|---|---|
| S2 stdio | Klear stdio | 125 |
| Klear stdio | Klear stdio | 103 |
| Nemotron function | Klear stdio (more tests) | 71 |
| Klear stdio | S2 stdio | 39 |
| Nemotron function | Nemotron function | 36 |
| S2 stdio | S2 stdio | 30 |
| Nemotron function | S2 stdio | 14 |
| AceCoder | AceCoder | 19 |
| other | | 9 |

Kept copies are chosen by more tests first.

AceCoder text near-duplicates are rare: 0 pairs at Jaccard ≥ 0.7. Its 65 test-overlap pairs are mostly paraphrased variants such as factorial with different error rules, so the function threshold is stricter.

**AceCoder easy share.**
- The v6 prefilter is known only for v6 rows. Of the 8,332 AceCoder rows, 4,392 were in v6 (prefilter 0–5/8). The other 3,940 are about 82% Instella ≥6/8, estimated from the v6 report: 3,962 of the 4,857 not-in-v6 AceCoder Dolci rows were prefiltered out.
- Overall, about 33% of final_v2 rows are likely base-easy.
- In v7 training (cached W&B aggregates), AceCoder groups are all-solved 51% of the time in the first 100 steps and 56% in the last 100. S2-function groups are all-solved 81%.
- Counter-evidence: the v6 report attributes part of v6 < v5 to removing all easy rows, which train the "near-solved → reliable" regime.
- **Recommendation: downsample, don't remove.** OLMo pass rate barely separates easy from hard among v6 rows, so the downsample is random.

## 4. final_v3 mixes (all after the shared cleaning: dedup, restores, checker rows, trims, LCB-window drop)

| | A: clean + restore | **B: recommended (built)** | C: aggressive |
|---|---|---|---|
| likely-easy AceCoder + S2-function (not in v6) | keep all (4,176) | keep a random 1/3 (1,392) | drop all |
| stdio oversampling | none | ×2 for stdio rows with prefilter ≥ 1/8 or not in v6 | same as B |
| unique rows | 13,339 | 10,555 | 9,163 |
| draws per epoch | 13,339 | 13,286 | 11,894 |
| AceCoder / Nemotron / S2-F / Klear-F | 8,313 / 1,499 / 245 / 6 | 5,684 / 1,499 / 90 / 6 | 4,382 / 1,499 / 0 / 6 |
| Klear stdio / S2 stdio (unique; draws in B/C) | 2,476 / 800 | 2,476 / 800 (4,601 / 1,406) | same as B |
| stdio share of draws | 24.6% | **45.2%** | 50.5% |
| 0/8 share of draws (cap 15%) | 13.5% | 13.6% | 15.1% |
| estimated base-easy (≥6/8) share | 32.8% | 22.8% | 15.9% |

Why B:
- It restores the v5/v6 stdio share. v5 and v6 got their 43% partly from duplicate copies: 5,233 rows for about 3,541 problems, an implicit 1.5× oversample.
- It keeps a reliability-training slice of easy rows, which the v6 report says matters.
- It cuts about 2,630 rows that are mostly 8/8 groups, i.e. wasted rollouts.

Variants:
- C is the "drop 8/8-likely AceCoder" variant the literature favours. Its 0/8 share sits right at the 15% cap.
- A is the minimal-risk change from final_v2.

`build_v3.py --mix A|B|C --write --out DIR` builds any of them.

## 5. Not done / next

1. **GPU test generation (R2) for the 358 verified few-test stdio rows** (1–4 tests, v6 solved about 90% at least once). This is the largest remaining learnable pool. Also R1 generated checkers for the 53 rejected and 233 no-checker multi-answer problems.
2. **A real difficulty label for the 3,940 not-in-v6 AceCoder rows.** An Instella n=8 prefilter would replace the random 1/3 downsample with "drop 8/8, keep 6–7/8". Nothing else in this recipe needs a GPU.
3. **new_sources (~14.1K stdio candidates) are still unverified** and not in v3. They would go through the same reference verification, dedup (CF id and test overlap against v3) and date or contamination checks.
4. **Training compatibility.**
   - Checker rows need code_reward.py ≥ 5f4a09c1c0.
   - Oversample copies have distinct `custom_id`s (`__os1`), so per-problem bookkeeping keyed on custom_id does not collide; use `metadata.oversample_of` to group copies.

## Files (`v7_improve/data_audit/`)

- `master.csv` / `master2.csv`: every Dolci row with v6 stats (prefilter, solve), v1/v2 reasons and duplicate cluster.
- `v6_solve.json`: per-problem v6 rollout aggregates.
- `restore_cands.jsonl.gz`, `restore_verify.jsonl`, `restore_eval.csv`: restore candidates and their grader results.
- `checkers_s0/`, `checkers_s1/`: checker verification results, accepted rows and reports.
- `prompt_format.json`, `v2_rows.csv`, `neardup_pairs.jsonl`, `testdup_pairs.json`.
- Scripts:
  - `build_v3.py`: the builder;
  - `sanity_v3.py`: the grader sanity check;
  - `verify_restore.py`, `run_grader_job.sh`: server and client on one node, token never printed;
  - `lux_patched/`;
  - `neardup.py`, `testdup.py`, `prompt_fmt.py`, `stdin_use.py`.
