# v7 stdio sources: Klear + Synthetic-2 cleaning

Scope is 9,272 rows: Klear 6,272 (6,264 stdio, 8 assert) and Synthetic-2 (S2) 3,000. AceCoder and Nemotron have 0 stdio rows, so nothing was added from them.

## Headline
| | in | kept | verified (exact grader) | verified incl. stated float tol | dropped |
|---|---|---|---|---|---|
| Klear | 6,272 | 2,415 | 2,180 (90.3%) | 2,180 | 3,857 |
| S2 | 3,000 | 1,292 | 1,085 (84.0%) | 1,109 (85.8%) | 1,708 |
| total | 9,272 | 3,707 | 3,265 (88.1%) | 3,289 | 5,565 |

- **Before cleaning:** among executed rows, a reference passed the original tests on 88% of Klear rows (4,777 of 5,420) and 61% of S2 rows (1,604 of 2,633).
- **Verifier origin (kept rows):** a v6 solved rollout verifies 1,179 Klear and 538 S2 rows. The rest are verified only by OLMo outputs.
- **Unverified rows:** 442 kept rows carry `metadata.verified=false`. They have no proven defect; no reference solved them.

## Method
- **Sandbox:** the grader's own `code_exec_server.Runner`, reused as-is. Each test gets a fresh `python3 -I`, an audit-hook guard, rlimits, a temp dir and no network. Runs fan out over a 24-process `multiprocessing` pool.
- **Judging:** every reference output is judged under the exact grader rule. It is also judged with case-folding, float tolerance (1e-6 relative/absolute), and whitespace-token rules, so each failure can be classified offline.
- **References:** Dolci OLMo outputs (up to 8 per row) and v6 rollouts with reward 1.0. Rollouts with reward ≥0.5 were also run, for diagnosis only.
- **Run order:** round 1 tried up to 4 references per row and stopped at the first pass. Round 2 tried all remaining references on still-unverified rows.
- **Not executed:** rows that a statement-level check drops anyway (image, multi-answer, interactive and similar; 2,138 rows) were not run.
- **Timeouts:** the wall cap was 10 s on a shared host (load 40–75 on 32 cores). Anything over 5 s counts as slow.
- **Join keys:** custom_id, plus exact prompt and label for the v6 join. 5,239 in-scope rows matched v6 rollouts.

## Counts per defect (rows unless stated)
1. **Tests not in stdin form (S2: 691 rows reach this check).** The v5 classifier found fn_args_wrapped_out 359, pylist_of_lines 131, fn_args_list_repr 83, gfg_literal 55, fn_args_json 49, stdin_but_fn_statement 13 and empty_input 1. Outcomes:
   - **Fixed and verified:** 218 converted to asserts (`assert f(args) == v`, the grader's script route), 115 converted to stdin (lines joined), and 11 kept as-is because a reference already passes them.
   - **Dropped (347):**
     - 305 have no function named in the statement, or need a GfG driver print format. Most are Codewars katas whose starter code was stripped.
     - 42 have conversions that no reference passes.
   - **Survive later steps:** 186 assert conversions and 1 stdin conversion. Most stdin conversions have only 1–2 tests, so the <3-tests or constant-floor rules drop them.
   - **Mixed-in literals:** 4 more rows (3 Klear, 1 S2) had Python-literal tests mixed into a stdin test set. Those tests were dropped.
2. **Label vs the statement's own sample.** 75 rows not already flagged multi-answer have a test whose input equals a statement sample but whose output differs; all 75 are dropped:
   - 45 (26 Klear, 19 S2): references that pass every other test reproduce the label, not the statement. These are unflagged multi-answer problems, e.g. `2 3 4 1` vs `1 3 4 2`, or a statement typo.
   - 30 (4 Klear, 26 S2): unresolved.
   - In 30 more rows the difference is formatting only (case, float or whitespace); those rows are kept.
   - **Same input, conflicting outputs within one test set:** 11 rows. 10 were resolved by the statement sample or references, and the conflicting test was dropped. 1 is unresolved and dropped.
   - **Brute force** (`bruteforce/proofs.py`, exhaustive implementations of the statements) on 9 disputed labels from `blocking_candidates.csv`:
     - 4 labels are correct, although 2–12 references agreed against them.
     - 2 labels are proven wrong. One was relabelled (S2 99-78); the other row has only 2 tests and is dropped.
     - 3 tests have inputs that violate the statement and were dropped.
   - Majority agreement among references was never used to relabel.
3. **Multi-answer, no checker:** 594 dropped (Klear 360, S2 234), listed in `multi_answer.csv` for a future checker. With the 45 hidden cases from item 2, about 640 rows need a checker.
4. **Float tolerance stated:** 11 Klear and 54 S2 kept rows are flagged `metadata.float_tol`. Most values are 1e-6; when the statement gives no number, the flag uses 1e-6 plus `float_tol_unspecified_value`.
   - Reference runs that fail only on float precision: Klear 18, S2 219.
   - 24 S2 rows are verified only under tolerance (`verified_relaxed`).
5. **Case-insensitive YES/NO:** 52 Klear and 13 S2 kept rows are flagged `metadata.case_insensitive`. 11 rows (6 Klear, 5 S2) have a reference that fails only on case (13 and 32 runs); 6 of those 11 statements say case does not matter.
6. **Statement-level drops:**
   - images: 1,502 (`<image>`/`[Image]`/`![]` tags 1,033; "see the picture" phrases 469)
   - non-English: 20
   - empty or sample-only statements: 16
   - interactive: 6
   - requests for another programming language: 0
7. **Constant-output floor of 50% or more:** 1,111 rows reach this check (782 Klear, 329 S2).
   - 586 were thinned: tests with the dominant output were removed until one constant passes under 50%, keeping at least 3 tests.
   - 525 were dropped, mostly single-verdict problems (YES/NO, First/Second), where one answer always covers 50% or more.
8. **Fewer than 3 tests, empties, whitespace:**
   - **Fewer than 3 tests:** 356 rows (5 Klear, 351 S2). 21 were augmented with uncapped Dolci tests or statement samples (verified by references); 349 were dropped. One test means hard-coding the sample wins.
   - **Repeated identical tests:** collapsed in 1,026 rows.
   - **Empty-output tests:** 25 rows; these are legitimate "print nothing" cases and were kept.
   - **Trailing spaces:** 224 rows, but the grader strips them, so they are harmless.
   - **Doubled spaces in non-numeric lines:** 20 rows; the grader compares these literally. 2 rows were normalised where references failed only on that.
   - **CRLF inputs:** harmless, because Python's text-mode stdin translates them.
9. **Huge/slow tests:** 134 tests in 26 rows had no reference finish within 5 s. 110 of those tests (20 rows) were dropped where some reference passes every other test. 3 rows were dropped because fewer than 3 tests would remain.
10. **Duplicates:**
    - **Klear:** 3,800 exact-distinct statements after normalisation (lowercase, punctuation, the two Klear footers; the brief counted 4,887 with stricter text). Near-duplicates (5-gram Jaccard ≥0.85 and identical numbers, so easy/hard versions stay apart) bring this to 3,730 clusters.
    - **S2:** 2,985 clusters. 607 clusters span Klear and S2.
    - **Result:** 2,127 rows dropped (1,861 Klear, 266 S2), keeping the copy with the most tests, then verified, then the most verifying references.
11. **Never solved in v6 and 0/8 at prefilter:** 877 rows (526 Klear, 351 S2). 370 are kept (93 verified); the rest were dropped by the rules above.
    - Hand diagnosis of 40 (`never_solved_hand_labels.csv`, seeded 20+20):

      | | genuinely hard | broken | unclear |
      |---|---|---|---|
      | Klear | 15 | 1 | 4 |
      | S2 | 7 | 11 | 2 |

    - The broken S2 rows are: multi-answer ×5, empty statement ×2, inconsistent or malformed generated tests ×2, float under exact compare ×1, wrong label ×1, deliberately destroyed statement ×1.
    - The pipeline already drops or flags 11 of these 12. It misses S2 8-33, whose inputs are malformed (n=2 followed by 3 numbers).
    - The 6 unclear rows are cases where several references agree against 1–2 labels. The joystick case shows such agreement can be a shared bug.

## Worked examples
1. **S2 8-73 / 69-44, list of lines to stdin.** `"['3','aba','abac','abcd']"` becomes `"3\naba\nabac\nabcd\n"`; an OLMo reference passes. 8-73 is kept; 69-44 was later dropped because it has only 1 test.
2. **S2 103-42, function arguments to asserts.** `[{'input': "['yo', 3]", 'output': "['\"yo\" repeated 3 times is: \"yoyoyo\"']"}]` becomes `assert repeater('yo', 3) == '"yo" repeated 3 times is: "yoyoyo"'`. The name comes from the statement and is also defined by references; verified.
3. **S2 94-72 (candles), dropped.** The tests call a function that the statement never names (references invent `check_candle_fall`).
4. **S2 2-19 (strStr).** The only test expects -1 for haystack "hello" and needle "ll", where the statement's own example gives 2. No reference passes the converted test, so the row was dropped (conversion unverified).
5. **Klear 41-15 (mouse).** The same input appears twice, labelled `1` and `0`. Statement sample 3 says 1 and 4 references agree, so the `0` test was dropped and the row verified.
6. **S2 99-78 (co-prime cells).** The answer is π(n+m); exhaustive search confirms this for all n, m ≤ 5. π(2·10⁶) = 148,933, so the label 148,934 was relabelled and the row verified.
7. **Klear 27-83 / 5-30 (joystick 1 1).** Twelve references print 1, but the exhaustive game tree gives 0. The label was kept, which shows majority agreement is not proof.
8. **S2 62-48 (treasure hunt, n=1).** References print `Draw`; enumerating every single change gives beauties [9, 10, 9], so the `Shiro` label is correct.
9. **S2 63-92 (4-cycles).** The test has 2 duplicate roads, which the statement forbids. The test was dropped and the row verified.
10. **S2 77-60.** The statement allows 1e-4, but labels and references differ at about 1e-12 relative. The row is flagged `float_tol=1e-4`; under the current exact Decimal compare, correct code scores 0.84.
11. **S2 81-85.** A permutation problem where the label `2 3 4 1` differs from the statement's `1 3 4 2` and references copy the label. This is an unflagged multi-answer problem; dropped.
12. **Klear 116-41 (First/Second, 32 tests).** One constant output passed 50% or more of the tests. Thinned to 21 tests; verified.
13. **S2 98-74.** The expected output contains `Test case :  1` with doubled spaces, and references fail only on that. Normalised to single spaces; verified.
14. **Klear 112-74.** One of 32 tests was `"['123']" -> "['123']"` (a Python literal); dropped, and the row is verified on the remaining 31.

## Grader changes to propose (not implemented)
- **Float tolerance:** when `metadata.float_tol` is set, compare numeric tokens with `|a-b| ≤ tol·max(1,|b|)`. This would recover 24 S2 rows now verified only under tolerance, plus the partial credit lost on 65 flagged rows. The tolerance has to reach `pass_rate`, e.g. via the label or the `/test_program_stdio` body.
- **Case folding:** when `metadata.case_insensitive` is set, fold case before comparing (65 rows).
- **Whitespace:** compare `line.split()` token lists instead of exact stripped lines. This fixes doubled spaces inside non-numeric lines (20 rows) without changing numeric behaviour.
- **Unchanged and still open:** no checker for multi-answer (about 640 rows held out); `sys.exit` inside `__main__` skipping asserts affects the 193 kept assert rows.

## Judgement calls
- **Images:** all 1,502 image rows were dropped per the brief, although many Codeforces images are decorative and 1,239 are Klear. Reinstating the verified ones is one filter away; reasons are in `dropped.csv`.
- **Constant floor:** thinning tests is my reading of "drop the constant tests". The rule removes nearly every single-verdict YES/NO problem.
- **Slow tests:** dropping tests that no reference finishes makes those 20 rows easier, since slow algorithms can now pass. Timings come from a loaded host.
- **Missing function names:** the 305 Codewars-style rows could be rescued by adding a signature line to the prompt. I did not, because that is a prompt edit beyond "the statement names a function".
- **Hidden multi-answer drops:** the 45 rows where references follow the label could include a few statement typos.
- **Function-name choice:** for assert conversion the name is picked from names in the statement that some reference also defines. The reference only breaks ties; it never supplies the name.

## Files
- `fixed.jsonl`: training schema plus `metadata.custom_id`, `fixes`, `verified`, `verifier_origins`, and the optional `float_tol`, `case_insensitive` and `verified_relaxed`.
- `dropped.csv`, `multi_answer.csv`, `blocking_candidates.csv`, `never_solved_population.csv`, `never_solved_hand_labels.csv`, `stats.json` (every counter).
- `bruteforce/proofs.py` and `proofs.json`.
- `scripts/`: `build_workspace.py` (joins), `analyze.py` (statement flags, sample parser, conversions), `sandbox.py`, `run_refs.py`, `run_conv.py`, `dedup.py`, `decide.py`, `blocking_tests.py`, `never_solved_sample.py`.
- `work/`: `rows.jsonl.gz` (rows plus references, 648 MB), `results.pkl`, `pool.pkl`, `analysis.pkl`, `decisions.pkl`, and `never_solved_dump.txt`.
