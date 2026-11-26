# v7 function-style (assert) rows: cleaning report

Scope: all 12,121 assert-label rows of `dolci-code-train.jsonl` (AceCoder 10,107, Nemotron 2,006, Klear 8; no
Synthetic-2 row has assert tests). Rows are keyed by custom_id; v6 rollouts are joined by exact prompt+label.

## Headline

| source | rows before → after | tests before → after | verified before → after (share) | stub ≥50% before → after | never-solved (v6 rollouts, executed) before → after |
|---|---|---|---|---|---|
| AceCoder | 10,107 → 8,388 | 148,952 → 118,835 | 9,047 (89.5%) → 7,705 (91.9%) | 1,910 (18.9%) → 0 | 862/5,250 (16.4%) → 655/4,330 (15.1%) |
| Nemotron | 2,006 → 1,759 | 20,400 → 16,870 | 1,632 (81.4%) → 1,518 (86.3%) | 313 (15.6%) → 0 | 401/1,684 (23.8%) → 280/1,472 (19.0%) |
| Klear | 8 → 6 | 200 → 149 | 7 → 5 | 1 → 0 | 1/6 → 1/5 |
| **total** | **12,121 → 10,153** | 169,552 → 135,854 | 10,686 (88.2%) → 9,228 (90.9%) | 2,224 → 0 | |

- "Verified" = at least one independent reference passes every final test. References: OLMo outputs from the Dolci
  parquet (8/row), v6 rollouts with reward > 0 (dedup by code), and AceCode-87K stored inferences with pass rate 1.0.
- All 9,228 verified rows were re-run through the grader's own per-test path (`code_exec_server.Runner.assert_tests`,
  program + test as `__main__`, 5 s). All 9,228 pass there (9,185 with the first reference tried, 43 with another);
  37 rows could not be reproduced and were demoted to unverified (`nondeterministic_tests`, all random-output tasks).
- Bulk execution used a fork-per-test sandbox (`scripts/sandbox.py`). It agreed with the grader route on 5,768/5,777
  test verdicts; all 9 disagreements were random-output tasks.
- Stub floor = best of None/0/''/[]/False/True/echo on the final tests. On a 600-row grader-route sample of the output,
  the mean floor is 0.089 and no row reaches 50%.
- Executed never-solved = no v6 rollout code passes every test, before (original tests) and after (final tests). The
  logged v6 rewards give 766/5,250 and 393/1,684 before.

## Counts per defect (rows unless marked)

| # | defect | AceCoder | Nemotron | Klear | handling |
|---|---|---|---|---|---|
| 1 | tested name missing from prompt | 1,626 | 71 | 3 | signature line added on 1,424 / 14 / 2 kept rows (1,383 / 2 / 2 verified); 4 dropped (no consistent arity) |
| 2 | stub ≥ 50% | 1,910 | 313 | 1 | 3,073 / 779 stub-passable tests pruned (370 / 117 rows); 1,017 / 153 / 1 dropped (<3 left) |
| 2 | a constant passes 100% | 506 | 20 | 0 | 492 / 21 dropped (after vacuous-test removal) |
| 3 | <3 meaningful tests after cleaning | 116 + 21 empty | 33 + 1 empty | 0 | dropped |
| 4 | duplicate tests (exact or AST-equal) | 1,178 rows / 5,215 tests | 34 / 59 | 0 | deduplicated |
| 4 | cross-test name dependency | 14 | 67 | 0 | setup tests merged into the dependent test (a single standalone test) |
| 4 | dependency on another test's state | 9 rows / 7 tests | 0 | 0 | test dropped (no reference passes it in isolation; the best passes it in sequence) |
| 4 | name defined only by a later test | 4 | 0 | 0 | test dropped |
| 5 | blocking test (≥2 distinct refs fail only it, none pass it) | 115 | 52 | 0 | 50 hand-checked: 11 wrong (test dropped), 9 right (kept), 30 underspecified (row dropped) |
| 5 | unverified after cleaning | 682 | 242 | 1 | kept, `verified=false` (patterns below) |
| 5 | unverified + "predefined/simulated database" wording | 31 | 0 | 0 | dropped (hand check: these encode hidden state) |
| 6 | exact float compare | 844 rows / 4,392 tests | 9 / 27 | 0 | rewritten to `_v7_close` (isclose, rel/abs 1e-9, same container types); a reference needed it in 41 / 1 rows |
| 6 | statement says the returned list is in any order, exact list compare | 27 | 4 | 0 | rewritten to `sorted(..., key=repr)`; needed by a reference in 21 / 1. Phrases about input order or order inside a pair/tuple are excluded (13 rows not rewritten) |
| 6 | multi-answer statement, no checker | 74 | 272 | 0 | 7 / 39 unverified dropped; 66 / 233 verified kept and flagged |
| 7 | prompt example reused as a test (kept rows) | 658 rows / 901 tests | 961 / 2,433 (55% of kept rows) | 4 / 5 | kept, flagged; held out: 117,934 / 14,437 / 144 tests remain; 27 / 138 rows would drop below 3 meaningful tests |
| 8 | test uses a stdlib name it never imports (`datetime`, `Decimal`, `math`, ...) | 134 | 27 | 0 | aliased import added inside the test (`from datetime import datetime as _v7_datetime`) |
| 8 | vacuous test (every constant passes: `f(x)==f(x)`, `!= literal`) | 106 rows / 329 tests (kept rows) | 50 / 101 | 0 | dropped |
| 8 | test with no assert (pure setup or "no crash") | 25 rows / 164 tests (input) | 50 / 145 | 0 | merged as setup or dropped |
| 8 | try/except-pass test, exit inside test | 1 / 1 | 10 / 0 | 0 | reported; caught by the vacuous/stub rules where trivially passable |
| 8 | program-side `sys.exit(0)` passes every assert | 600/600 sampled rows | | | grader change needed (below); 32 reference programs in our pool "pass" this way and were excluded as references |
| 9 | exact duplicate prompts (whitespace/case normalised) | 0 | 0 | 0 | none among assert rows (the 195 Dolci duplicates are stdio) |
| 9 | near-duplicate prompts (same tested names, 5-gram Jaccard ≥ .85) | 1 | 0 | 1 | dropped the lower-ranked copy |

Dropped total 1,968 (`dropped.csv`): stub floor unprunable 1,171; constant passes all 513; <3 meaningful 149;
multi-answer unverified 46; unverified hidden state 31; underspecified (hand) 30; no tests left 22; interface
unrecoverable 4; near-duplicate 2.

## Broken tests (defect 5)

- Rule: ≥2 distinct references fail exactly the same single test, no reference passes it, and the near-miss
  references are probed for their output on that test (`broken_test_evidence` in metadata). 101 candidates had
  references agreeing on one output.
- Hand check of 50 (`handcheck_blocking.csv`): **11 wrong** (the statement proves the test wrong), **9 right**
  (the models were wrong; 2 of them are the prompt's own example), **30 underspecified** (hidden lists, case/dup/order
  conventions). Consensus is therefore not evidence of a broken test (22% precision). Only the 11 hand-proved tests
  were dropped; the 30 underspecified rows were dropped; the rest are kept unverified.
- Unverified rows (925 = 683 AceCoder + 241 Nemotron + 1 Klear): 8/3 have no runnable reference; 37 nondeterministic; for the
  rest the best reference fails 1 test in 145/100 rows, 2–3 tests in 146/79, ≥4 in 347/59 (AceCoder/Nemotron). 190 AceCoder rows have a best reference
  under 25%. These are mostly tasks where every reference uses a different interface or convention.
  Per-row patterns: `unverified.csv`.

## Worked examples

1. Signature: `rlvr_acecoder_..-237-26` tests call `gcd_lcm`, prompt never names it → appended "Implement `def gcd_lcm(num1, num2)`."
   (args from the passing references; a stub with that signature binds on every test).
2. Rename-verified signature: `..-177-62` — all references named it `character_frequency`; `char_frequency_sorted = character_frequency`
   passes all tests → "Implement `def char_frequency_sorted(s)`." (134 AceCoder rows verified this way).
3. Stdlib alias: `..-193-86` `assert parse_rfc3339(..) == datetime(..., tzinfo=timezone.utc)` failed for every program that
   did `import datetime`; now `from datetime import datetime as _v7_datetime` + `_v7_timezone` inside the test.
4. Cross-test dependency: `nemotron-7-1-91` tests `sol = Solution(2, 2)`, `res1 = sol.flip()`, `assert ... res1 ...` → one
   test containing the setup chain; the setup-only tests are removed.
5. Float: `..-265-40` `calculate_total_cost(5, 0.05, 0.1) == 5.75` (refs return 5.750000000000001) → `_v7_close(...)`.
6. Any order: `..-99-41` ("You can return the answer in any order"; its near-duplicate `..-275-4` was dropped) `two_sum([2,7,11,15], 9) == [0, 1]` → `sorted(..., key=repr)` on both sides.
   Not rewritten: `..-126-90`, where the statement only frees the order inside each pair.
7. Vacuous: `..-262-47` `assert hello_world() != 'HELLO WORLD'` (every constant passes) → dropped.
8. Stub prune: `..-197-33` `minimumHammingDistance`: `0` stub scored 69% (9 of 13 tests expect 0) → 10 stub-passable
   tests removed, 3 remain, reference passes.
9. Constant drop: `..-528-11` `is_hour_in_range`: every test expects `True` → row dropped.
10. Broken test (W): `nemotron-8-3-41` `bipartite_queries(4, 4-cycle)` expects the last answer `'NO'`; an even cycle is
    bipartite. 4 distinct v6 references return `'YES'` → test dropped, row verified.
11. Broken test (W): `..-138-63` `largestIsland(...) == 5`: flipping (1,2) joins islands of size 2 and 3 → 6 (21 refs agree).
12. Right test, wrong models (M): `nemotron-7-3-54` test equals the statement's own example (16); refs return 12 → kept, unverified.
13. Underspecified (U): `..-394-52` `user_registration` against "a predefined list" → row dropped.
14. Multi-answer: `nemotron-6-14-98` ("print any") with exact compares and no passing reference → dropped.
15. Nondeterministic: `..-130-34` `create_array()` of random decimals; reference fails a different test on each run → unverified.

## Proposed grader changes (not implemented)

1. Assert route: run the program with `__name__ != "__main__"` (oe-eval does this), or append `print("__V7_OK__")` after
   the assert and require it. Today a program that only calls `sys.exit(0)` scores 1.0 on every sampled row.
2. Partial credit: binary, or `(pass - stub_floor) / (1 - stub_floor)`. Pruning removed the floor here, but the rule is
   cheap to compute per row from `metadata.flags.stub_floor`.
3. Prepend `from typing import *` (LCB does) so `List[int]` annotations do not fail at definition (88 reference programs).
4. A checker hook for multi-answer rows (299 kept rows carry "any valid answer" wording and are verified only against one answer).
5. Seed or exclude random-output tasks (245 kept rows mention random; 37 are flaky).

## Unsure whether legitimate

- Pruning stub-passable tests also removes real edge cases (`f([]) == []`) when the floor is ≥50%.
- Dropping vacuous / no-assert tests even when the floor is under 50%.
- Rename verification: the reference passes under another function name; only the name in the prompt changes.
- Tolerant float compare is applied to every float-literal `==` (4,419 tests), not only where a reference needed it (42 rows).
- The "hidden state" regex drop applies only to unverified rows; 209 verified AceCoder rows with the same wording are kept.
- 147 AceCoder rows test HumanEval entry points (v5 audit). Not acted on: the brief forbids using eval data for selection.

## Files

`fixed.jsonl` (training schema + metadata.custom_id/src/fixes/verified/flags), `dropped.csv`, `handcheck_blocking.csv`,
`unverified.csv`, `rewrites.csv` (every float/order rewrite), `stats.json`, `verify_grader.json`, `never_solved_v6_exec.json`.
Scripts (run in order): `build_inputs.py` → `audit.py` → `recheck_timeouts.py` → `reaudit.py` → `rename_verify.py` →
`decide.py` → `finalize.py` → `verify_grader.py` → `recheck_grader_failures.py` → `finalize.py` → `stats.py`;
`fidelity.py` checks the sandbox against the grader route.
