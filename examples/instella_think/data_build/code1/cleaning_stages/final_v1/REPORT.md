# final_v1: v7 code-RL training set (repair over drop, CPU only, binary reward)

**Result: 13,241 rows (function 10,240 / stdio 3,001; 233,381 tests).** The P1–P7 proposal had 11,680 rows
(9,018 / 2,662), and DATA_PRACTICES projected ~13,020 after the Tier-A CPU repairs (~9,950 / ~3,070).

Every row has a reference solution that passes every test through the sandboxed grader server, with the training
options (strict asserts, `name=tmp_sol`, per-row `float_tol` / `case_insensitive`, 5 s per test). Prompts are the v6
prompts. The only prompt edit is the existing F1 kind: an `Implement \`def f(...)\`` signature line, on 1,369 rows
already in v7 plus 190 new ones (restored function rows and S2 conversions).

**Sanity check** (`work/sanity.json`):
- 30 random rows with their reference: 30/30 score 1 (all tests, and the first 32 that training grades).
- 30 other random rows with trivial stubs: 0/30 give any stub a binary pass. Stubs were None / 0 / '' / [] /
  False / True / echo / empty program, and for stdio, printing the most common output. The best stub's partial
  credit averages 0.25 (max 0.87).

## 1. Counts: v6 pool → v7 → final_v1, by source and route

| source | route | v6 pool | v7 fixed (13,854) | v7 proposal P1–P7 | **final_v1** | vs proposal |
|---|---|---|---|---|---|---|
| AceCoder | function | 10,107 | 8,388 | 7,424 | **8,332** | +908 |
| Nemotron | function | 2,006 | 1,759 | 1,431 | **1,657** | +226 |
| Klear | function | 8 | 7 | 5 | **6** | +1 |
| Klear | stdio | 6,264 | 2,408 | 1,918 | **2,176** | +258 |
| S2 | stdio | 3,000 | 1,106 | 744 | **825** | +81 |
| S2 | function (converted) | 0 | 186 | 158 | **245** | +87 |
| **total** | | **21,385** | 13,854 | **11,680** | **13,241** | **+1,561** |
| function / stdio | | 12,121 / 9,264 | 10,340 / 3,514 | 9,018 / 2,662 | **10,240 / 3,001** | +1,222 / +339 |

Against the DATA_PRACTICES Tier-A projection (~9,950 / ~3,070):
- Function is +290. The floor rule here is "a constant passes 100%" (as instructed); DATA_PRACTICES counted
  floor ≤ 0.8.
- Stdio is −69. Restored rows collide with MinHash duplicates more than the 10% haircut assumed.

11,758 final rows were in v7 `fixed.jsonl`; 1,483 are rows v7 dropped. Every one of the 21,385 v6 rows is in
`train.jsonl` or `dropped.csv` exactly once.

## 2. Changes applied

**Repairs (rows in train.jsonl carrying the change, `metadata.v7_changes`):**

| change | rows | by source / route | what |
|---|---|---|---|
| `restored_binary_reward` | 1,231 | Ace-F 835, Nem-F 150, Klear-S 176, S2-S 59, S2-F 10, Klear-F 1 | rows v7 dropped only for the partial-credit floors (F2 stub floor, F2/F4 "< 3 meaningful", S5 constant floor), now kept because no constant passes every test |
| `restored_pruned_tests` | 419 | Ace-F 319, Nem-F 100 | the stub-passable tests F2 pruned are back (all 487 pruned rows were tried; see `kept_v7_pruned_version` for the 11 where no reference passes the full set) |
| `restored_thinned_tests` | 263 | Klear-S 168, S2-S 83, S2-F 12 | S5-thinned dominant-output tests are back |
| `merged_duplicate_tests` | 577 | Klear-S 522, S2-S 55 | tests of exact/near-duplicate copies (S7 clusters) merged into the kept copy: 3,157 tests from 644 copies (stage 1, before the 1 MB cap) |
| `image_decorative_kept` | 176 | Klear-S 148, S2-S 28 | decorative-image rows that a reference verifies (338 of 398 verified; the rest became duplicates, MinHash copies or < 5 tests) |
| `s2_nonstdin_converted` | 76 | S2-F 76 | S2 rows with Python-literal tests converted to asserts plus a signature line from the references (127 converted and verified; 51 then < 5 tests or duplicates) |
| `minhash_merged_tests` | 118 | Klear-S 116, S2-S 2 | tests of MinHash-duplicate copies merged into the kept copy (1,381 tests) |
| `added_uncapped_dolci_tests` | 21 | Klear-S 21 | rows under 5 tests topped up from uncapped Dolci ground truth (411 tests, each passed by the verifying reference) |
| `kept_v7_pruned_version` / `kept_v7_prompt` | 2 / 1 | Ace/Nem-F | restoring tests left no reference passing every test (11 rows; 9 of them then have < 5 tests and drop at P6), or changed the signature-line argument names (1 row); the v7 version or prompt is kept |

- **Restored-floor rows.** 1,059 were expected per DATA_PRACTICES; 1,231 are kept. The difference is that
  DATA_PRACTICES also required floor ≤ 0.8.
- **Weak-but-valid rows.** 182 function and 148 stdio rows have a constant that passes > 80% of the first 32
  tests. They are kept per instruction, since the reward is binary. `metadata.const_floor` lets you filter them.
- **Merged tests.** No merged test set has an input mapping to two outputs (`groups_dup_conflict` = 0,
  MinHash conflict = 0). 17 duplicate copies and 5 MinHash copies were not merged because the winner's reference
  fails them. 104 merged tests over 1 MB were left out, because a request over 64 MB fails on the server.

**Standard drops (sequential; each row is counted once, at its first rule):**

| rule | rows | by source | note |
|---|---|---|---|
| P1 confirmed contamination | 17 | Ace 17 | 14 from v7 fixed + 3 restored rows |
| P2 non-generic paraphrase | 76 | Ace 73, Nem 2, Klear 1 | 50 from v7 fixed + 26 restored |
| P4 solution-defined `__eq__` | 1 + 14 | Ace 14, Nem 1 | Ace 165-54, plus 14 rows that fail only under strict grading because the tests compare against objects the solution defines (`== Tweet(...)`, `== Person(...)`, `pathlib.Path`) |
| P5 unverified | 1,645 | Ace 884, Nem 260, Klear 260, S2 241 | 1,523 had no passing reference in the v7 runs; 37 decorative-image rows that no reference passes; 85 fail under the training grader (75 fail non-strict too, 8 flaky, 2 other). 1,417 of them are v7-fixed rows |
| date ≥ 2024-08-01 | 0 | – | latest mapped date is 2024-07-07 |
| P7 MinHash 5-gram ≥ 0.7 | 390 | Klear 333, S2 55, Nem 1, Ace 1 | after LaTeX normalisation, on the surviving rows; 552 pairs, 394 with equal digit sequences dedup'd; 158 pairs whose numbers differ are kept apart (easy/hard variants: `work/minhash_pairs.csv`) |
| P6 < 5 tests | 437 | Ace 182, S2 233, Nem 22 | after merges and uncapped top-up; v7 projected 482 |
| M0 Klear rows in both files | 7 | Klear | the stdio-file copy removed, the function copy kept (identical labels) |

The flags are kept as is:
- 4 `lcb_v3_flag` rows (Nemotron CF 1849F, 1981D, 1982E, 1983F).
- 306 verified multi-answer function rows (`flags.multi_answer_statement`; 299 in v7 plus 7 restored).

**Upstream v7 drops that stay** (see `dropped.csv`):

| category | rows |
|---|---|
| S7 duplicate statements (tests merged when consistent) | 2,479 |
| S4 non-decorative images | 1,067 |
| S3 multi-answer without checker | 620 |
| F3/S5 constant passes all | 513 / 33 |
| S6 < 3 tests | 349 |
| S1 non-stdin not convertible | 221 |
| S2 label vs sample | 82 |
| F multi-answer unverified | 48 |
| F hidden state | 36 |
| F hand-checked underspecified | 30 |
| F no tests left | 22 |
| F interface unrecoverable | 15 |
| statement-level: non-English / empty / interactive | 20 / 16 / 6 |
| F near-dup prompt | 3 |
| slow / conflicting outputs | 3 / 1 |

## 3. Tests per row

| | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| function | 13.9 | 14 | 7 | 20 | 5 | 75 |
| stdio | 30.4 | 32 | 18 | 35 | 5 | 75 |

Histogram (all rows):

| tests | rows |
|---|---|
| 5 | 373 |
| 6–10 | 2,858 |
| 11–15 | 2,839 |
| 16–20 | 4,247 |
| 21–31 | 601 |
| 32 | 1,650 |
| >32 | 673 |

**Caveat: training grades only the first 32 tests** (`CODE_MAX_TESTS=32` in code_reward.py).
- Merged and top-up tests are appended after the kept copy's own tests, so only 476 of the 4,944 added tests
  (34 rows) are inside the graded prefix. The rest only matter if `CODE_MAX_TESTS` is raised; the training
  server caps at 64 per request.
- For the 673 rows with more than 32 tests, no single constant passes all of the first 32, so no reordering was
  needed.

## 4. One example per repair

- **Binary-reward restore (function).** Ace 201-55 `isPalindrome`, 16 tests. v7 dropped it at a stub floor of
  0.81, because most tests expect `True`. A constant fails ≥ 1 test, so it now scores 0 under binary reward, and
  the row is kept (`const_floor` 0.81).
- **Binary-reward restore (stdio).** Klear 22-67, 32 tests. v7 dropped it at a constant floor of 0.97. The
  reference passes all 32; the one non-dominant test makes a constant fail.
- **Pruned tests restored.** Ace 171-72 `generate_passphrase`: v7 kept 3 tests after pruning, final_v1 has 11.
  - Counter-example: Ace 197-33 `minimumHammingDistance` (DATA_CHANGES F2 example). No reference passes its
    restored 13 tests, so it falls back to the verified v7 3-test version, which then drops at P6.
  - It is in `gpu_candidates/testgen_function.jsonl`.
- **Thinned tests restored.** Klear 116-41 (First/Second): v7 thinned it to 21 tests; it has 32 again.
- **Duplicate-test merge.** Klear 22-98 gains 1 test from its exact-duplicate copy. The test's input is new, and
  the winner's reference passes it.
- **Decorative image.** Klear 99-36: "Note: Dish from the first sample looks like this: `<image>`", on its own
  line in the Note section. An OLMo reference passes all tests, and the row also takes 1 test from a duplicate.
- **S2 non-stdin → function.** S2 38-60: the tests were `{'input': "[['American Redstart', ...]]", 'output': ...}`.
  The references agree on `generate_bird_codes(names)`. The row becomes `assert generate_bird_codes([...]) ==
  ['AMRE', ...]`, with "Implement `def generate_bird_codes(names)`." appended to the prompt, and is verified.
- **Uncapped top-up.** Klear 13-77 had 4 tests; 13 uncapped Dolci tests that its reference passes bring it to
  17.
- **MinHash merge.** Klear 126-22 absorbs 2 tests from its MinHash duplicate, a copy with and without `$…$`
  LaTeX.
- **P4 under strict grading.** Ace 12-82 `deserialize_tweet(...) == Tweet(...)`. It passes as `__main__` and
  non-strict, but fails strict canonicalisation, so the row is unpassable in training and dropped.
- **P7 kept apart.** Klear 85-37 vs S2 38-63 (Jaccard 0.89): the same statement except `10^{18}` vs `10^9`, an
  easy/hard pair, so both are kept.

## 5. How it was built (scripts/)

All execution went through a local `code_exec_server.py` (16 workers; ~316k sandboxed test runs). No code was run
outside it.

1. `fn_redecide.py`: re-runs `function/scripts/decide.py` + `finalize.py` on the v7 audit vectors, with step 4
   (stub-floor pruning) and step 6 (< 3 meaningful tests) removed. No local execution: the probe results and
   arity checks come from the v7 `decide.pkl`. It reproduces 9,666 v7 rows byte-for-byte.
2. `stdio_redecide.py`: re-runs `stdio/scripts/decide.py`. The constant-floor drop is only at 100%, nothing is
   thinned, and the dedup winner is chosen verified-first. Decorative-image rows go through normal cleaning
   (the heuristic is in `stdio_common.decorative_image`, the same regexes as DATA_PRACTICES). It saves every
   row before dedup.
3. `phase1.py`: grader runs for rows v7 never executed.
   - Decorative images: up to 8 references per row.
   - S2 Python-literal rows: the `analyze.to_asserts` converter with the name ≥2 references (and ≥50%) define;
     the signature from those references; verification; and the 7 constant stubs.
4. `phase2.py`: picks the record per row. Builds the S7 clusters (winner verified-first, then most tests),
   merge groups with the same-input/different-output check, and uncapped top-up for rows under 5 tests. Then
   re-verifies every row under the training grader (up to 4 references).
5. `classify_grader_fails.py`: splits training-grader failures into strict-only (P4) and other (P5).
6. `assemble.py stage1/stage2`: P1, P2, P4, P5 and date; MinHash, with grader-verified merges; P6; flags; and
   all outputs.
7. `sanity.py`, then `gpu_candidates.py`.

## 6. Could not do / judgement calls

- **GPU repairs not run** (by instruction): checkers, test generation, and teacher solve. See `GPU_REPAIRS.md`
  and `gpu_candidates/` (3,720 candidate rows, 3,047 priority).
- **Non-decorative images (1,067) not restored.** That needs the open-r1 text statements, and only the dataset
  card is cached. 822 of them map to a CF id. The same join could give checkers for 577 of the 670 multi-answer
  candidates.
- **The decorative-image heuristic was not hand-checked** beyond the examples. Each restored row is
  reference-verified, but a reference passing does not prove the image was unneeded.
- **MinHash digit guard.** 158 pairs above 0.7 were kept apart because their digit sequences differ.
  - Most differ in constraints or sample examples, which is the intent.
  - A few look like formatting-only differences, e.g. Klear 44-85 / 6-63 at Jaccard 0.95. Auditing
    `work/minhash_pairs.csv` (`kept_apart_numbers_differ`) could remove a few dozen more.
- **85 rows the v7 sandbox verified fail under the training grader** and are dropped:
  - 75 fail non-strict too (random-output or time-sensitive tasks, or references near the 5 s limit on a
    loaded host);
  - 8 pass on a re-run (flaky);
  - 14 strict-only failures are counted under P4 instead.
- **Prompt format (P8) is unchanged**, as instructed. The 190 new signature lines are the F1/S1 interface fix,
  not a format change.
- **S2 conversion name rule.** The name must be defined by ≥2 references (≥50%) at top level. 96 rows fail it
  ("references disagree"); a model rewrite à la OLMo A.7.3 could recover some.
- **P11 difficulty band** is not done. It needs an Instella n=8 GPU pass, and the restored rows carry no solve
  rate.
- **Two command deviations.** The server ran with `--host 127.0.0.1 --workers 16` added to the given command, to
  keep it local and within the 16-concurrency limit. It is stopped, and its scratch dir is removed.

## Files

- `train.jsonl`: v6 schema (`prompt`, `label`, `metadata`). `metadata` keeps `dataset_source` and the v7 fields
  (`custom_id`, `fixes`, `verified`, `flags`, `float_tol`, `case_insensitive`, …), and adds `src`, `route`
  (function | stdio), `v7_changes`, `lcb_v3_flag` and `const_floor`.
- `dropped.csv`: custom_id, src, reason, for all 8,144 v6 rows not in train.
- `restored.csv`: custom_id, src, route, kind (row | tests), reason. It has 1,483 rows v7 dropped (1,231
  floor-restore, 176 image, 76 S2 conversion) and 682 rows with tests restored (419 pruned, 263 thinned).
- `stats.json`: every count above, plus merge, MinHash and reason-by-source breakdowns.
- `GPU_REPAIRS.md`, `gpu_candidates/*.jsonl` and `gpu_candidates/summary.json`.
- `work/`:
  - `sanity.json`
  - `verifying_refs.json.gz` (the reference program per train row, for re-checks)
  - `minhash_pairs.csv`
  - `grader_fail_class.json`
  - `phase1_results.json`
- `scripts/`: the pipeline above. Intermediates are in `/tmp/v7final/` (local disk).
