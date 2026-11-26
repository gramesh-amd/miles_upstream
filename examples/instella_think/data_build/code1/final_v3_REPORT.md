# final_v3_candidate: final_v2, deduplicated, with restores, checker rows and mix B

**Result: 13,286 training rows. These are 10,555 unique problems plus 2,731 stdio oversample copies; 45.2% of draws are stdio (final_v2: 23.7%).** Not uploaded anywhere.

The full audit, with evidence and the alternative mixes A and C, is in
`/it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit.md`. The scripts are in `.../v7_improve/data_audit/`
(`build_v3.py --mix B --write` rebuilds this directory deterministically, seed 0).

## Counts

| source \| route | final_v2 | v3 unique | v3 draws (with oversample copies) |
|---|---|---|---|
| AceCoder \| function | 8,332 | 5,684 | 5,684 |
| Nemotron \| function | 1,639 | 1,499 | 1,499 |
| Klear \| function | 6 | 6 | 6 |
| SYNTHETIC-2 \| function | 245 | 90 | 90 |
| Klear \| stdio | 2,337 | 2,476 | 4,601 |
| SYNTHETIC-2 \| stdio | 840 | 800 | 1,406 |
| **total** | **13,399** | **10,555** | **13,286** |

Of the draws:
- Instella prefilter 0/8 (v6 label): 13.6%, which is under the 15% cap.
- Estimated base-easy (≥6/8): 22.8%, vs 32.8% without the downsampling.

## Changes vs final_v2 (each row counted once, at its first rule; `dropped_v3.csv` lists every dropped row)

| step | rows | notes |
|---|---|---|
| drop: inside the LCB v3 date window (2023-05-01 to 2024-07-31) | −4 | Nemotron CF 1849F/1981D/1982E/1983F (the existing `lcb_v3_flag` rows) |
| trim tests > 1 MB | 5 rows, 8 tests | `v3_trimmed_huge_tests`; 2 more rows dropped (< 5 tests left) |
| drop: fastest verified reference > 4 s on some test (limit 5 s) | −18 | flaky under load |
| add: Open-R1 checker rows (lux job 0, run here on CPU) | +170 | `openr1_checker`, `metadata.checker`; 10 Nemotron function rows replaced by checker rows with more tests |
| add: image-placeholder rows, grader-verified | +310 candidates → 139 after dedup | `v3_restored_image_placeholder`, `flags.image_placeholder` |
| add: multi-answer rows that Open-R1 grades by token compare | +180 → 82 after dedup | `v3_restored_multi_answer_openr1_token_compare` |
| add: multi-answer rows with no CF mapping or checker | +56 → 29 after dedup | `v3_restored_multi_answer_exact_compare` |
| dedup over the union (same CF id; test overlap; MinHash) | −446 final_v2 rows, −296 restored candidates | keeps the copy with more tests, then the final_v2 copy |
| mix B: keep a random 1/3 of the not-in-v6 AceCoder and SYNTHETIC-2 function rows | −2,629 AceCoder, −155 S2-F | about 82% of these are Instella ≥6/8 (estimated) |
| mix B: second copy of each stdio row with prefilter ≥ 1/8 or not in v6 | +2,731 draws | copy: `custom_id = <id>__os1`, `metadata.oversample_of`, `v3_oversample_copy` |

## Verification and sanity

- **Restored rows.** A reference (v6 solved rollout or OLMo sample) passes every test on the sandboxed grader:
  - training options (exact compare, plus the row's `float_tol` / `case_insensitive`), 5 s per test;
  - every row has ≥ 5 tests, and no constant output passes all graded tests;
  - multi-answer restores also need exact-compare evidence: v6 solved the row at least once under exact compare, or a second independent reference matches the label exactly.
- **Checker rows** passed `verify_checkers.py`'s acceptance rules A1/A2/R1–R3. A patched copy was used: the shipped script crashes on a 5,000-digit answer; see `data_audit.md`.
- **Sanity** (`sanity.json`, `sanity_v3.py` through the grader): 30 random new rows (restored or checker) and 20 old rows all score 1 with their reference (50/50). Stubs (an empty program for every row; for stdio, also printing the most common output) pass 0/50.
- **Schema** is the same as final_v2. New metadata keys:
  - `v6_prefilter_n_correct` (int, or null if the row was not in v6);
  - `oversample_of` (copies only);
  - `v3_restore_reason`;
  - `flags.image_placeholder`.

  `n_tests` equals `len(label)` on all rows. Checker rows need `code_reward.py` at code-rl-v7-fixes ≥ 5f4a09c1c0, which routes them through `/test_program_checker`.
- **Dates.** No row is on or after 2024-08-01. The 4 rows inside the LCB v3 window are removed.

## Files

- `train.jsonl`: rows in shuffled order.
- `dropped_v3.csv`: custom_id, src, route, reason.
- `verifying_refs.json.gz`: the reference per unique row.
- `stats.json`: step counts, A/B/C mix stats and final stats.
- `sanity.json`.
