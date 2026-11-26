# final_v2: final_v1 + Open-R1 statement-swap rows (checker rows deferred to lux)

**Result: 13,399 rows: 10,222 function and 3,177 stdio. There are 0 checker rows; those are pending the lux job.**
- 219 verified rows come back from the non-decorative image drops. Each one's image-dependent statement is
  replaced by Open-R1's text statement.
- 61 final_v1 rows were replaced by a recovered copy of the same CF problem that has more tests (43 stdio, 18 Nemotron
  function rows). Net change vs final_v1 (13,241): **+158**.

Every recovered row has a reference that passes all of its tests through the sandboxed grader:
- stdio route, training options (`float_tol` / `case_insensitive` from the row, 5 s per test);
- up to 32 tests per row (mean 31.0, min 7);
- references: v6 142, OLMo 59, accepted CF submission 18.

Scope change (user, mid-run): checker verification does not run on the login node. The 389 multi-answer and
checker rows (240 CF problems) are exported for lux. See [../final_v1/LUX_PLAN.md](../final_v1/LUX_PLAN.md) and
`lux/`. The code_reward.py checker routing is committed (code-rl-v7-fixes `5f4a09c1c0`), so the merged rows will
train correctly.

**Sanity check** (`work/sanity.json`, through `code_reward._score_one` in training mode):
- 30 random recovered rows with their reference: 30/30 get reward 1.
- The same rows with stubs (empty program, `print(0)`, printing the most common output): 0/30 get any reward.
- api_error = 0.

## 1. Counts

| source \| route | final_v1 | final_v2 | change |
|---|---|---|---|
| acecoder \| function | 8,332 | 8,332 | 0 |
| nemotron \| function | 1,657 | 1,639 | −18 (none recovered; see the note below) |
| klear \| function | 6 | 6 | 0 |
| synthetic2 \| function | 245 | 245 | 0 |
| klear \| stdio | 2,176 | 2,337 | +161 |
| synthetic2 \| stdio | 825 | 840 | +15 |
| **total** | **13,241** | **13,399** | **+158** |

Note on the function rows. The Nemotron −18 is the 18 replaced function rows: CF problems whose final_v1 copy was
a Nemotron function row. The recovered stdio copy has more tests, so under "keep the copy with more tests" it wins.
If you would rather never swap a verified final_v1 row, `dropped_v2.csv` lists them
(`replaced_by_recovered_duplicate_with_more_tests`), and dropping the 48 replacing rows from train restores them.

Recovered rows: Klear 187, S2 32. `metadata.v7_changes`:

| change | rows |
|---|---|
| `openr1_statement` | 219 |
| `openr1_official_tests` | 2 (50 Open-R1 official tests) |
| `merged_duplicate_tests` | 3 |

Options: 9 rows have `case_insensitive` and 1 has `float_tol`. `lcb_v3_flag` = 0. New metadata keys:
`openr1_id`, `cf_date`, `cf_rating`. Checker rows (lux) will carry `metadata.checker` and `checker_source`.

## 2. Funnel (rows; each row counted once, at its first failing rule)

Statement task (S4 non-decorative image rows; expected ~822):

| step | dropped | left |
|---|---|---|
| mapped to a CF id (decontam/date_check.csv) | | 822 (394 unique CF problems) |
| ambiguous mapping: our tests or samples give a different output than Open-R1's (same input) | 14 | 808 |
| Open-R1 statement still refers to a figure in description/input/output ("see the picture", "picture below", …) | 124 | 684 |
| multi-answer statement and no Open-R1 checker | 32 | 652 |
| < 5 tests | 2 | 650 |
| decontam (LCB all versions, CF eval incl. test overlap, CRUXEval, HE/MBPP(+Pro), open-r1 CF test split) | 0 | 650 |
| Open-R1 problem has a generated_checker: checker route, deferred to lux | 62 | 588 |
| duplicate of a final_v1 row with at least as many tests (CF id 113 groups, MinHash 75, S7 cluster 5; overlapping) | 95 | 493 |
| no reference passes all tests (≤ 10 references incl. accepted CF submissions) | 38 | 455 |
| duplicate copy of another recovered row (same CF problem; its tests merged when consistent) | 236 | **219** |

Checker task (S3 multi-answer + label≠sample rows; expected ~577):

| step | dropped | left |
|---|---|---|
| mapped to a CF id | | 577 (323 unique CF problems) |
| ambiguous mapping (constraint numbers differ) | 2 | 575 |
| interactive / file I/O | 2 | 573 |
| **Open-R1 has no generated_checker** | 233 | 340 |
| < 5 tests | 13 | 327 |
| decontam | 0 | 327 |
| deferred to lux checker verification | 327 | 0 |

Why far fewer than 822 / 577:
- The expected numbers count rows, but those rows are mostly copies: 822 image rows are 394 CF problems, and 577
  multi-answer rows are 323.
- Only 343 of the 577 multi-answer rows (219 problems) have an Open-R1 checker at all, not 577.
- 16% of the Open-R1 statements still point at a missing figure. Open-R1 converted formula images to LaTeX but
  dropped picture images silently.

Decontamination found 0 hits. Each detector was confirmed on positive controls:
- an LCB statement fires ng8/ng13 at 1.00;
- CF 2063/A fires the id check;
- the 646B cluster fires through `cluster(openr1_cf/646/B): tests49 codeforces|2010/C1`;
- 2010/C1's official tests fire the test overlap at 49.

Shared test pairs: 2 recovered problems share 1 pair with an LCB problem (reported only; v7 false positives share
1–3 pairs). The date check: all mapped problems are before 2024-08-01, the latest being 2024-07.

## 3. Examples

**Statement swap** (S2 row `synthetic2-…-90-71`, CF 601/D). The formula images were real
content.
- Before: `Let's denote the number of distinct strings which can be read this way as <image>. … We are interested
  in vertices with the maximum value of <image>.`
- After: `… which can be read this way as $\operatorname{dif}(v)$. … the maximum value of $\mathrm{dif}(v) + c_v$.`

The wrapper, the layout and our Examples block stay as they were. Only the description, input, output and note
text are swapped. Open-R1's `$$$…$$$` and `$$…$$` become `$…$`, the style of the Klear rows.

**Ambiguous mapping dropped** (774/C). 13 of our tests give a different output from Open-R1's official test for
the same input, so the mapping is rejected.

**Checker accepting an alternative answer and rejecting wrong ones.** This is from the login-node pilot, CF 114/B
"PFAST Inc.", before the scope change; the production run is on lux.
- Two accepted CF submissions print other valid teams than the label on 15 and 20 of 32 tests. The Open-R1 checker
  accepts all 32 of their outputs.
- It rejects the empty output (0/32 accepted) and the label with one integer changed (0/32 accepted).
- It rejects all three wrong CF submissions (they pass 6, 21 and 5 of 32 tests).

## 4. Code (code-rl-v7-fixes, local commit 5f4a09c1c0, not pushed)

`code_reward.py`: a row with `metadata.checker` (Python source), or a self-contained label
`{"tests": [...], "checker": "..."}`, is graded through `/test_program_checker`. Everything else is shared with
stdio rows: binary/partial reward, think gate, `CODE_MAX_TESTS`, `CODE_EXEC_TIMEOUT` (plus the server's 10 s
checker time in the read timeout), and the hack monitor. A crashing checker, or one printing anything but 1/100,
fails the test. An unusable checker counts as `no_tests` and gives reward 0. Other rows behave exactly as before.

Tests:
- self-test 41/41: alternative answer accepted, plain compare rejects it, wrong answer, crash, non-1 output,
  eval, think gate, infinite loop, unusable checker;
- `test_code_hack_monitor.py` 103/103 in binary and partial modes (75/75 offline), including 6 checker
  end-to-end cases;
- `test_code_exec_hardening.py` 57/57;
- `test_code_exec_adversarial.py` 111/138 (27 known residual/policy, 0 others).

## 5. How it was built (`scripts/`, intermediates in /tmp/v7v2)

1. `prepare.py`: join to Open-R1 (local parquet of the default config), mapping check, statement swap, test
   cleaning (python-literal tests, repeats, same-input conflicts, > 1 MB), Open-R1 official tests appended up to
   32, references (+ accepted / wrong CF Python submissions from `cf_py_subs`).
2. `decontam.py`: the detectors above.
3. `groups.py`: one group per CF problem, test merge, static dedup against final_v1.
4. `verify.py` with `VERIFY_ROUTES=stdio`: grader runs, 16 concurrent requests, 16 workers.
5. `finalize.py`: decisions, outputs.
6. `sanity_v2.py`.
7. `export_lux.py`: the lux candidate file.

## 6. Judgement calls and open items

- **Replacement of final_v1 duplicates** (61 rows, 18 of them function rows): this follows "keep the copy with
  more tests" literally. The reversal is described in section 1.
- **The figure-reference rule is conservative** (124 rows dropped). Some of those figures are decorative.
- **Open-R1 statements lost superscripts in places** ("1 ≤ ai ≤ 109"). This is the same style as many of our
  Codeforces-sourced rows, and was left as is.
- **Multi-answer rows without an Open-R1 checker** (233) could be graded by token compare (Open-R1 does this), but
  were not recovered here. They go to job 1 (generated checkers) in LUX_PLAN.md.
- **A2 acceptance rule bug fixed before shipping** the lux script. It originally also required our OLMo references'
  outputs to be accepted, but those can be wrong. A smoke test with a toy checker caught it. The gate is now:
  accepted CF submissions, and references that match the label on every test.

## Files

- `train.jsonl`: same schema as final_v1.
- `recovered.csv`: row, source, CF id, tests, verifier, `replaces_final_v1`.
- `dropped_v2.csv`: every candidate not recovered, plus the replaced final_v1 rows, with reasons.
- `stats.json`.
- `work/verifying_refs.json.gz`, `work/sanity.json`, `work/checker_acceptance.json` (empty until lux).
- `lux/`: `checker_candidates.jsonl.gz`, `verify_checkers.py`, `merge_checker_rows.py`, `code_exec_server.py`.
- `scripts/`.
