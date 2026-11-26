# v7 decontamination: Dolci code_rlvr (21,385 rows) against all current and planned eval sets

## Bottom line
- **20 training rows are confirmed contamination and go on the drop list** (`drop_list.txt`, `dropped.csv`). They hit 11 HumanEval+ problems (17 rows, all AceCoder) and 2 Codeforces-eval problems (3 Klear rows).
- **LiveCodeBench v3/v5/v6, MBPP+, CRUXEval-O/I: 0 confirmed contamination.** HumanEval-Pro and MBPP-Pro are clean on the tested (new) function. HE-Pro embeds HumanEval/i verbatim as its first function, so 11 HE-Pro items inherit a contaminated sub-problem through the same 17 rows.
- No training prompt reaches any standard n-gram threshold for any eval set: 0 pairs at 13-gram ≥ 0.2, and 0 at 8-gram ≥ 0.5 (Tulu 3 / OLMo 3), with or without boilerplate filtering. The two contamination modes found are invisible to n-grams:
  - **Rewritten HumanEval tasks in AceCoder.** These were caught by function-name and test overlap, then confirmed by execution.
  - **A Russian original of a Codeforces problem.** Caught by 49 shared hidden tests.
- **Paraphrases (same idea, different statement and tests; reported, not dropped):** 72 rows on 29 eval items. Generic textbook collisions: 119 rows. Counts per set are below.
- **Date check:** of 11.3k competitive-programming rows, 6,033 map to a dated Codeforces or LeetCode problem. Dates run from 2010-02 to 2024-07. Only 4 Nemotron rows fall inside the LCB v3 window (CF 1849F, 1981D, 1982E, 1983F). None are LCB or Codeforces-eval problems, and none fall in the v5 or v6 windows.

## Per eval set (items = eval problems; rows = training rows)
| eval set | items | pairs hand-checked | contaminated items / rows | paraphrase items / rows | generic-collision items / rows | false-positive items |
|---|---|---|---|---|---|---|
| LCB v3 | 612 | 90 | 0 / 0 | 0 / 0 | 0 | 15 |
| LCB v5 | 279 | 20 | 0 / 0 | 1 / 2 (3659) | 0 | 8 |
| LCB v6 | 175 | 31 | 0 / 0 | 0 / 0 | 0 | 8 |
| HumanEval+ | 164 | 163 | **11 / 17** | 23 / 65 | 16 / 40 | 20 |
| MBPP+ | 378 | 104 | 0 / 0 | 3 / 3 | 44 / 73 | 24 |
| HumanEval-Pro | 164 | 139 | 0 new-func (11 raw sub-problem) | 2 / 2 | 0 | 23 |
| MBPP-Pro | 378 | 133 | 0 | 0 | 12 / 14 | 31 |
| CRUXEval-O / -I | 800 / 800 | 1 | 0 | 0 | 0 | 1 |
| Codeforces (eval) | 422 | 22 | **2 / 3** (2010/C1, 2010/C2) | 0 | 0 | 10 |

The contaminated HumanEval+ problems are HE/4, 6, 10, 18, 26, 63, 64, 71, 95, 148 and 160. Each training row was checked against every eval item; `contaminated.csv` holds the 698 adjudicated pairs (413 rows) with detector, score and verdict.

## Detectors and thresholds
Tokens are lowercase `[a-z0-9]+`, with LaTeX commands stripped. Eval text is the problem statement only, without prompt templates. For HE-Pro and MBPP-Pro, the raw and new parts are matched separately.

| detector | definition | flag for hand check | max observed |
|---|---|---|---|
| exact | normalised eval statement equal to, or a substring of, a train prompt | any | 0 hits |
| 13-gram | \|shared\| / \|eval 13-grams\|, excluding boilerplate (n-grams in ≥5 eval items or ≥100 train rows); also reported per train row | ≥ 0.20 | 0.18 (MBPP-Pro), LCB 0.13, CF 0.12 |
| 8-gram | same, n=8 (Tulu 3 / OLMo 3 standard is 0.5) | ≥ 0.50; also a raw (no boilerplate filter) pass at ≥ 0.20 | 0.43 (0.44 raw), MBPP-Pro/72 "product of list" |
| Jaccard | exact Jaccard of word 5-gram shingles (what MinHash estimates; exact is cheap at 21k × 3.9k) | ≥ 0.30 | 0.19 |
| TF-IDF | word 1+2-gram cosine; paraphrase candidates; plus top-10 neighbours of every HE/MBPP/Pro item were executed | ≥ 0.50 | 0.58 (HE) |
| function | tested-function name (and parameter names or arity) equals an HE/MBPP/Pro/LeetCode entry point; generic names excluded | same name and arity | 846 name pairs |
| test overlap | identical normalised stdin/stdout pair (input ≥ 10 chars, pair in < 3 eval items); identical (args, expected) call tuple (≥ 12 chars); identical assert string; checked against the capped labels and the full Dolci ground truth | ≥ 1 io, or a same-name call, or ≥ 3 calls, or ≥ 1 assert | 49 of 64 tests (CF 2010/C1) |
| numeric examples | a shared run of 8 numbers (≥ 4 distinct values) in the statements; catches re-hosted problems with the same examples | ≥ 1 and TF-IDF ≥ 0.3 | 1 real hit (LCB 3659) |
| ID / URL | CF contest and problem ids, AtCoder task ids, LeetCode URLs, eval titles (≥ 3 words) in the prompt | direct id; title only with TF-IDF ≥ 0.25 | 0 eval ids. Title hits are generic phrases ("a b c" ×758) |
| execution | A = share of the row's tests passed by the eval canonical solution. B = OLMo Dolci samples that solve the row and also pass the full eval test suite (HE+/MBPP+ harness, Pro asserts). Run in a sandbox: `unshare -rn`, temp dir, rlimits, timeout | A ≥ 0.85 | — |

**Why these thresholds.**
- The text thresholds sit below the published ones: 13-gram 0.2 against GPT-3's "any 13-gram"; 8-gram 0.5 as in Tulu 3/OLMo 3; decon's default score is 0.8. Recall matters more than precision here, because every flag was hand-checked.
- "Any shared 13-gram" (GPT-3 style) fires on 2,306 training rows. Almost all of those share boilerplate such as "the first line contains a single integer t". It is not usable without a frequency filter, and GPT-3 also dropped frequent n-grams.
- The test thresholds are deliberately loose. Of the 139 LCB/CF pairs flagged by stdio overlap, 133 share only one to three edge-case tests (`1\n1000000000 → 1`, `5\n1 2 3 4 5 → 5`) and are false positives. The 6 real ones (CF 2010/C1, C2) share 24–49 tests.

**Verdict rule for function-style sets.** The rule below gives the default verdict. 95 pairs carry a hand verdict (overrides plus the CF/LCB calls), listed in `scripts/verdicts.py:HAND` and grouped by reason.
- *contaminated*: A ≥ 0.85, same function name as the eval entry point, and either B > 0 or no reference was available. The row poses the eval problem itself.
- *paraphrase*: A ≥ 0.85 without that interface or reference evidence; or a hand-identified evolved variant (HE text plus an added twist).
- *paraphrase_generic*: the eval item is a textbook one-liner. This covers all of MBPP and MBPP-Pro, and HE/8, 9, 12, 13, 23, 28–31, 34, 35, 42, 45, 47, 48, 52, 53, 55, 58, 60, 61, 77, 90, 96 and 126.
- Everything else is *false_positive*.

## Hand-checked examples (15 of the 698 pairs)
1. **CF 2010/C1 and C2, contaminated.** Klear rows 1-56, 139-95 and 97-100 are the Russian statement of CF 646B ("Ошибка передачи сообщения", Технокубок 2016). Codeforces re-used that problem in Testing Round 19 (2024-08-28) as 2010/C1 and C2, and the Codeforces-eval set includes it. The rows share 49 of 64 hidden tests (C1) and 24 (C2), e.g. `abrakadabrabrakadabra → YES abrakadabra`. Text similarity is 0 because the statements are in different languages.
2. **HE/148 `bf`, contaminated (120-6).** The row restates the planets task with the same name and signature. It shares 7 HE+ test calls; the eval solution passes all 13 of the row's tests, and all 3 OLMo references pass HE+.
3. **HE/63 `fibfib`, contaminated (119-41, 189-54).** 189-54 names the function `fib` but defines "fibfib(0)=0, fibfib(1)=0, fibfib(2)=1, …"; A=13/13, and a reference passes HE+.
4. **HE/18 `how_many_times`, contaminated (91-37, 167-28, 202-72).** All three keep the HE name and signature with overlapping counts; A ≥ 0.95, and references pass HE+.
5. **HE/71 `triangle_area`, contaminated (213-16).** A near-verbatim restatement ("rounded to 2 decimal points … return -1 … sum of any two sides"); A=18/18, TF-IDF 0.58.
6. **HE/10 `make_palindrome`.** 163-13 is contaminated: plain shortest palindrome by appending, A=17/17, B=3/3, 8 shared HE+ calls. 238-11, 241-60 and 187-17 are paraphrases. They copy HE's algorithm text ("find the longest postfix … append the reverse of the prefix") but add "ignore case and non-alphanumeric". The eval solution passes 238-11's tests (A=11/11), but its OLMo reference fails HE+ (B=0/1), so the trained behaviour differs.
7. **HE/95 `check_dict_case`.** 83-6 and 66-100 are contaminated (same semantics, A=1). 219-100, 140-46 and 246-30 are paraphrases: they return 'Lower'/'Upper'/'Number', an evolved variant.
8. **HE/115 `max_fill`, paraphrase (160-16).** The same task renamed `well_emptying`, with a new statement and tests (A=11/11).
9. **LCB v5 3659 (count paths with XOR = k, 2024-12-21), paraphrase.** Klear 135-25 and 94-70 are CF 1006F (2018) with identical worked examples (grid `[[2,1,5],[7,10,0],[12,6,4]]`, k=11 → 3). The constraints differ: LCB has 300×300 cells with values < 16, solved by DP over XOR; CF has 20×20 cells with values ≤ 1e18, solved by meet-in-the-middle. Found by the numeric-example detector at TF-IDF 0.42.
10. **MBPP/610 `remove_kth_element`, paraphrase (69-42).** The row copies MBPP's wording ("returns a list with the same elements, but the k'th element removed") but changes the semantics (0-indexed, reverses when k ≥ len, …). A=1/19.
11. **MBPP/93 `power`, generic (246-50, 120-94, 125-61).** These rows contain `assert power(3, 4) == 81` and `power(2, 3) == 8`, which are 2 of MBPP's 3 asserts. The task and values are textbook; not dropped.
12. **MBPP/139 `circle_circumference`, generic (421-76).** It shares 8 MBPP+ call tuples, because 2πr on round radii is generic.
13. **LCB abc336_d / abc351_c, false positive (27 rows each).** The only shared test is `1\n1000000000 → 1`.
14. **HE/116 `sort_array`, false positive.** Nemotron 4-97 is "sort_array using median-of-three quicksort": same name, different task.
15. **CRUXEval sample_363, false positive.** AceCoder 65-13 names its function `f` and shares the assert `f([1]) == [1]`. All 800 CRUXEval functions have 13-gram coverage 0.

## MBPP/HumanEval-style trivial tasks: how others handle them
- **Tulu 3** (§3.2) checks prompts only, with 8-gram matching. A test instance counts as contaminated when more than 50% of its tokens fall in an 8-gram shared with a single training prompt. A training *dataset* counts as contaminated when it overlaps more than 2% of an eval. The authors tried embedding matching and dropped it because "it is difficult to distinguish mere distributional similarity from actual paraphrasing".
- **OLMo 3** applies the Tulu 3 procedure to all post-training data (§4.2.1, Step 4: "8-grams and an overlap threshold of 0.5"). It adds heuristics that ignore task-irrelevant generic phrases and math-symbol n-grams. Its midtraining uses `decon`: 5-gram sampling, cluster expansion, an IDF-weighted score with threshold 0.8, and a length penalty under which short texts need near-perfect matches. The penalty exists precisely because short, generic items such as MBPP collide.
- **Consequence.** Under the OLMo/Tulu criterion, none of our 698 pairs counts as contaminated: max 8-gram coverage is 0.44. A rewritten `is_prime` or `power` would never be flagged.
- **Our approach goes further.** We drop rows that are the eval problem under a new wording (same entry point and behaviour, confirmed by execution). We keep generic one-liners as `paraphrase_generic`, because the same task appears thousands of times in pretraining code and dropping it would not remove the exposure. `paraphrase_list.csv` lets a stricter policy drop them as well: 72 paraphrase rows plus 119 generic rows.

## Training prompts that contain their own hidden tests (allowed; quantified)
A stdio test counts as "in prompt" when both its whitespace-normalised input (≥ 3 chars) and its output occur in the prompt. An assert counts when its call and its expected value both occur in the prompt.

| source | rows with ≥1 label test in prompt | rows where all label tests are in prompt | label tests in prompt |
|---|---|---|---|
| AceCoder (10,107) | 191 (1.9%) | 34 | 0.7% |
| Klear (6,272) | 4,457 (71.1%); 94.6% against full GT | 0 | 5.0% |
| Synthetic-2 (3,000) | 1,549 (51.6%) | 170 (5.7%) | 5.8% |
| Nemotron (2,006) | 1,046 (52.1%) | 0 | 13.9% |
| **All** | **7,243 (33.9%)** | **204 (1.0%)** | **16,495 / 416,086 (4.0%)** |

The 204 "all tests are prompt examples" rows reward reproducing the samples. For Klear, 71% of rows reuse a sample within the 32-test cap and 95% somewhere in the full ground truth, so in 24% of rows the reused sample lies beyond the cap.

## Provenance: is OLMo's decontamination of Dolci documented?
Partly, and not specifically for the RL code data.
- The OLMo 3 paper (arXiv 2512.13961, §4.2.1 Step 4) says it filtered "all post-training data (all three stages)" against its eval suite with Tulu 3 8-gram/0.5 matching. The suite includes HumanEval+, MBPP+ and LiveCodeBench v3.
- The Dolci-Think-RL section (§4.4.2: sourcing, offline difficulty filtering, mixing) does not list a decontamination step, and the Dolci-Think-RL-7B dataset card mentions only "strict deduplication" and execution filtering.
- AceCoder and Nemotron prompts were rewritten by an LLM (§A.7.3). If n-gram decontamination ran before or independently of that rewrite, it could not catch rewritten HumanEval tasks. That is exactly what we find in AceCoder.
- Nothing covers LCB v5/v6, CRUXEval, HE-Pro/MBPP-Pro or the Codeforces eval set; those are our additions.
- RL-Zero data was additionally decontaminated against pretraining data. That step does not apply to this mix.

## Date check (LCB windows: v3 2023-05..2024-07, v5 2024-08..2025-02, v6 2025-02..2025-05)
- **Explicit ids.** Prompts carry almost no explicit ids. Of 730 id/URL regex hits, none points to an LCB or Codeforces-eval problem. The hits are `abc123` placeholders, "AtCoder Beginner Contest 100", two LeetCode URLs (two-sum, longest-palindromic-substring) and Wikipedia or Codewars links.
- **Statement matching.** Rows were matched to dated references: open-r1/codeforces (10,024 problems with contest dates) and LeetCodeDataset (2,869 problems with estimated dates), at 13-gram ≥ 0.2 or (TF-IDF ≥ 0.6 and 8-gram ≥ 0.3). 6,033 rows map (Klear 4,381, Synthetic-2 1,282, Nemotron 359, AceCoder 11).
  - By year: 2010–2021 holds 5,980 rows; 2022 has 46, 2023 has 4 and 2024 has 3. The latest is 2024-07-07.
  - Inside the v3 window: 4 Nemotron rows (CF 1849F 2023-07-27, 1981D 2024-05-31, 1982E 2024-06-25, 1983F 2024-07-07). None is in LCB, which holds only 9 Codeforces problems, all different, or in the Codeforces eval set (contests 1993–2067).
  - v5 / v6 windows: 0 rows.
- **AtCoder.** There is no dated AtCoder reference. AtCoder-style rows (Klear, Synthetic-2, from APPS/TACO) were matched against all 602 AtCoder problems in LCB, with no hit.
- **Cross-lingual exception.** The CF 646B → 2010/C case: a 2016 problem re-used in a 2024 contest, so a date filter alone would miss it.
- Details are in `date_check.csv`.

## Limitations
- Text detectors are ASCII-word based, so the 20 Russian-language prompts (Klear 16, Synthetic-2 4) are visible only to the test and numeric detectors. Three of those 20 were contaminated.
- Format-changed re-hostings with fresh examples and fresh tests (e.g., a Codeforces problem rewritten as a LeetCode function) are caught only by the TF-IDF top-k hand check and the numeric-example detector. I hand-checked the nearest training row for the 8 LCB and 8 Codeforces items with the highest TF-IDF (max 0.42); all except 3659 are topical look-alikes.
- B (the reference passes eval tests) needs an OLMo sample that solves the row; 0 of 8 samples did for some rows (B = 0/0), and those were judged from the statement.
- LCB has no canonical solutions, so LCB pairs were not execution-checked. None had any text or test signal beyond edge-case collisions.

## Files
- `contaminated.csv`: custom_id, src, eval_set, eval_id, part, detector, score, verdict, plus per-detector scores, exec_A and exec_B. LCB problems in several versions get one line per set.
- `drop_list.txt` / `dropped.csv`: the 20 confirmed rows, with reasons. `paraphrase_list.csv`: rows for an optional stricter policy.
- `per_eval_max_overlap.csv`: max 13-gram, 8-gram, Jaccard and TF-IDF per eval item, with the best train row. `per_train_max_overlap.csv.gz`: the same per train row (eval-side and train-side 13-gram ratio).
- `per_eval_counts.csv`, `date_check.csv`, `prompt_tests_summary.csv`.
- `scripts/`, in pipeline order:
  1. `build_corpus.py`: normalised corpora.
  2. `detect.py`: text, test, function and id detectors, plus dated-reference matching.
  3. `numeric.py`, `raw_ngram_check.py`.
  4. `candidates.py`: merge and side-by-side dumps.
  5. `exec_verify.py`: sandboxed execution.
  6. `verdicts.py`: rule plus hand overrides, then deliverables.
  7. `prompt_tests.py`, `report_stats.py`.
- `work/` holds the intermediate parquet files and the `review_*.md` / `cv_*.txt` side-by-side views used for the hand check.
