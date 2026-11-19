# Code RL v7: grader, reward and data handoff

Branch `code-rl-v7-fixes`, base `f610d0d915` (code-evals-v2). The branch closes reward hacks in the code grader,
makes the training reward all-or-nothing by default, adds reward-hack monitoring, and grades checker rows.

**After pulling, restart both sandboxes** (`cluster/code_exec.sh stop`, then `start`; the same with `bench`). The
server and the reward client change together.

Eval semantics change in exactly three places, all of which make grading stricter or fix a cap:
1. On the assert and call routes, an exit before the tests finish fails the test.
2. Tests and expected output are deleted before the program runs. On the call route, they are parsed only by the
   guard.
3. In `pass_rate`, pytest-style labels with more than `CODE_MAX_TESTS` test functions are now divided by the
   capped count, so they can reach 1.0.

Benchmark rows (`bench`) use the unchanged stdio and checker rules. Non-bench eval rows with `metadata.checker` now
use the checker route; no current eval set has any.

## What changed

| commit | change | why |
|---|---|---|
| `6d27091862` | early-exit hole closed; per-row stdio `float_tol` / YES-NO folding (training only) | `sys.exit(0)` before the appended asserts passed every test |
| `813242ea9c` | guard-held done token, strict canonical asserts for training, call JSON round trip, hardening tests | forged done files, rigged `__eq__` |
| `161a0a2b4b`, `c70316cb2e` | strict mode covers the v7 data's `_v7_close`; exact eval stdio rule; spoof-proof canon | `_v7_close` operands bypassed strict; type spoofs |
| `16b8ac68c1` | guard file named by content | servers of different versions share a scratch dir |
| `3c7a7132f2` | strict canon for every assert shape, fail-closed early exit, hidden tests | `and`/`in`/`<`/bare-call asserts, test peeking |
| `471182d70a` | server outcome `flags`, static scan, `rollout/code_hack/*` | see hacks the grader cannot stop |
| `1943f6c0a3` | `CODE_REWARD_MODE=binary` default, `rollout/code/*` | all-or-nothing reward; pass rate still logged |
| `5f4a09c1c0` | checker rows through `/test_program_checker` | multi-answer rows (lux) |
| `478bf18eb8` | call route parsed by a captured C JSON scanner; 400/500 errors instead of dropped connections; backlog 1024; refuse Python >= 3.13; cached strict rewrite | review S1, S2, S3, B3, N1 |
| `641450c421` | `helper_tamper` scan key; one bounded metrics window; client/server compare parity test | review B1, S5, S6 |
| `2565483787` | `CODE_STRICT_GRADING`, `CODE_EXEC_API_RETRIES`, `CODE_HACK_WINDOW_MAX` reach the Ray workers | review S4 (they were silently dropped) |
| `03c1161742` | `test_grader.sh`, numpy skip, call-route tamper cases, early-exit checks | review S8, S9 |

The evidence is in `/tmp/v7reg/{REVIEW,BEFORE_AFTER,FIXES,SURVEY}.md` on the login node. BEFORE_AFTER.md counts the
hacks closed, the false negatives fixed and the labelled regressions. Those files are in `/tmp`, so copy them next
to the data if they should outlive the node.

## How to run

```bash
I=examples/instella_think
$I/cluster/code_exec.sh start && $I/cluster/code_exec.sh start bench    # status / stop / url as before
# check the training grader's interpreter: <= 3.12 (it refuses 3.13+), numpy if rows need it
ssh $CODE_EXEC_HOST 'python3 -c "import sys; print(sys.version)"; python3 -c "import numpy"'
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP CHECK_ONLY=1 \
  CODE_TRAIN=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/train.jsonl
# expect "reward : CODE_REWARD_MODE=binary threshold=1.0 CODE_STRICT_GRADING=1", then launch without CHECK_ONLY
```

`CODE_TRAIN` still defaults to the old Dolci file, so v7 runs **must** set it. `cluster/code_exec.sh` starts the
training grader with the host's `python3` and no numpy, so numpy-using training rows score 0 there. To
reproduce an old pass-rate run, set `CODE_REWARD_MODE=partial`.

## Settings

Environment (reward client; all forwarded to the Ray workers by `run_async.sh`):

| variable | default | meaning |
|---|---|---|
| `CODE_REWARD_MODE` | `binary` | `binary`: 1 iff pass rate >= threshold; `partial`: the pass rate. Eval is always pass@1 |
| `CODE_REWARD_PASS_THRESHOLD` | `1.0` | binary threshold, in (0, 1] |
| `CODE_PARTIAL_CREDIT` | unset | legacy: 1 = partial, 0 = binary; contradicting `CODE_REWARD_MODE` is fatal |
| `CODE_STRICT_GRADING` | `1` | strict asserts + `name=tmp_sol` for training rows |
| `CODE_HACK_WINDOW_MAX` | `100000` | training samples kept for the next `rollout/code*` log |
| `CODE_EXEC_API_URL` / `CODE_BENCH_API_URL` | from `cluster/code_exec.sh` | `http://host:port/<token>`; `local` runs programs in-process (debug only) |
| `CODE_EXEC_TIMEOUT` | `5` | seconds per test |
| `CODE_MAX_TESTS` | `32` | tests per sample (the server caps at `--max-tests`, 64) |
| `CODE_REQUIRE_THINK` | `1` | training needs a clean `<think>...</think>` |
| `CODE_REWARD_SCALE` | `1.0` | reward multiplier |
| `CODE_REWARD_CONCURRENCY` | `64` | concurrent samples per batch |
| `CODE_LOG_EVERY` | `200` | samples between `CODE_REWARD` log lines |
| `CODE_EXEC_API_RETRIES` | `2` | HTTP retries per request |
| `CUSTOM_ROLLOUT_LOG_PATH` | `code_reward.log_rollout_data` (run_code.sh) | `""` disables the W&B hook |
| `CODE_EXEC_TESTS_CACHE` | `256` | server: cached `tests_ref` files |

Server flags (`code_exec_server.py`, unchanged on this branch apart from the 3.13 refusal): `--host 0.0.0.0`
(use `127.0.0.1` anywhere but the sandbox host), `--port 18950`, `--token-file ~/.code-exec-token`,
`--workers cpus-4`, `--mem-mb 2048`, `--max-tests 64`, `--max-timeout 10`, `--python <its own>` (<= 3.12),
`--tests-dir ""`, `--scratch /tmp/code-exec-<uid>`, `--stats-every 60`. Request fields and the response `flags`
are documented in the module docstring.

## How to verify

```bash
GRADER_PYTHON=<venv with numpy>/bin/python bash examples/instella_think/test_grader.sh
python3 examples/instella_think/test_code_hack_monitor.py     # offline part only, no server
```

Expected results:
- hardening: 57/57;
- adversarial: 116/143. The 27 failures are known (24 residual, 3 policy): 0 attacks succeed and there are
  0 false negatives;
- hack monitor: 200/200 in both reward modes;
- self-test: all ok, `api_error=0`;
- the 3.13 refusal and the lux copy check both pass.

Without numpy in `GRADER_PYTHON` the numpy cases print `SKIP`. On the first training step, check that the
`CODE_REWARD` line has `api_error=0` and the server log has `errors=0`.

## W&B metrics and alarms

| metric | watch for |
|---|---|
| `rollout/code/all_pass_rate` (= binary train reward), `pass_rate_mean` | the learning signal; `trained/*` is after the dynamic filter |
| `eval/humanevalplus`, `eval/mbppplus`, `eval/lcb_v3` | `all_pass_rate` rising with no eval gain means hacking |
| `rollout/code_hack/solved_any_rate` | **alarm** if it trends up over 20+ steps or exceeds ~2% |
| `solved_helper_tamper_rate`, `solved_touch_frames_rate`, `solved_touch_builtins_rate`, `solved_sys_modules_rate`, `solved_exit_call_rate` | any sustained non-zero value: read the dumped samples with `metadata.code_hack.<k>` set |
| `early_exit_rate`, `strict_reject_rate` | failed attempts; rising early in the run is the policy probing |
| `timeout_rate`, server `timeouts=` / `errors=` | sandbox saturation; `errors` must stay 0 |

The ~2% threshold is a starting point, not a measured baseline: 0.6% of 4,000 v6 policy samples carry some flag
(`exit_call`, `touch_frames`, `signal`, `ctypes`; mostly harmless), and 0% carry `helper_tamper`.

## Known residuals and policy decisions

These are listed in the adversarial suite's `RESIDUAL` and `POLICY`, and are left open on purpose:
- Test-side helpers (`__v7canon__`, `_v7_close`, builtins, `math.isclose`, `unittest`) rebound or patched by the
  solution in the middle of a test. Closing this needs the tests to run out of process. `helper_tamper` makes
  these attempts visible.
- A fake `sys.modules['gc']` disables canon's referent check.
- Type spoofs (`__class__`, `__len__`, `__float__`, user methods) that return the expected builtin value. These
  are no better than returning that value.
- `float_tol` applies to an integer answer printed as a float (`1000001.0` matches `1000000` at 1e-6). This
  follows testlib; only 29 training rows have `float_tol`.
- Policy: strict mode never calls a pure-Python `__eq__`, so dataclass, `Fraction` and user-`__eq__` answers
  score 0. Filtering such rows from the data is an open option.
- Frame introspection can read the guard's frame; the verdict does not depend on anything it can change there,
  up to Python 3.12. From 3.13 the guard's locals become writable, so the server refuses to start.
- The client resubmits a request after a read timeout (pre-existing), which adds load when the server is
  overloaded.

## Lux plan, in order

Full details: `final_v1/LUX_PLAN.md` and `final_v1/GPU_REPAIRS.md`, both under
`/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/`.

1. **Checker verification (no GPU)**, about 15–30 min at 100 workers. It covers 240 CF problems (389 rows):

   ```bash
   L=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/lux
   python3 $L/verify_checkers.py --workers 100 --python <python <= 3.12> --out-dir /tmp/v7_checkers
   python3 $L/merge_checker_rows.py --accepted /tmp/v7_checkers/accepted_rows.jsonl --dry-run
   python3 $L/merge_checker_rows.py --accepted /tmp/v7_checkers/accepted_rows.jsonl
   ```

   `$L/code_exec_server.py` must equal this branch's server. `verify_checkers.py` checks it against `SERVER_SHA256`
   and `test_grader.sh` against the repo, so after any server change re-copy it and update the constant. Then
   add the lux report numbers to `final_v2/REPORT.md` and run `final_v2/scripts/sanity_v2.py`.
2. **GPU checker / test generation**: R1 generated checkers for the multi-answer rows job 0 did not accept,
   verified with the same `verify_checkers.py`; then R2 test generation (`final_v1/gpu_candidates/testgen_*.jsonl`).
3. **Teacher solve** (R3, optional, 20–40 node-h) for the unverified rows.
4. **New stdio sources**: `new_sources/REPORT.md`. About 14.1K new unique problems (open-r1 `verifiable`,
   HardTests, CC+, TACO/DeepCoder), projected 10–12K after verification and 6–8K after a difficulty band. Follow
   its "Next steps": materialise tests on local disk, verify with this grader (checker route for open-r1),
   difficulty band, decontam follow-up. Merge with the final_v1 row rules: MinHash against train, >= 5 tests, no
   constant passes, decontam.

## Data

| file | rows | note |
|---|---|---|
| `v7_data/final_v2/train.jsonl` | 13,399 (10,222 function, 3,177 stdio; 0 checker) | train on this; see `final_v2/REPORT.md` |
| `v7_data/final_v2/{stats.json, dropped_v2.csv, recovered.csv, scripts/, lux/}` | | build provenance and the lux job |
| `v7_data/final_v1/train.jsonl` | 13,241 | `final_v1/REPORT.md`, `GPU_REPAIRS.md`, `LUX_PLAN.md`, `gpu_candidates/` |
| `v7_data/final_v2/lux/checker_candidates.jsonl.gz` | 240 problems / 389 rows | lux job 1 input |
| `v7_data/new_sources/` | about 14.1K candidate problems | `REPORT.md`, `recommended_union.csv.gz` |

## Open decisions

- **`CODE_MAX_TESTS` 32 vs 64.** Training grades the first 32 tests, and 673 rows have more. Most merged and
  top-up tests sit beyond 32: only 476 of 4,944 are inside the graded prefix (`final_v1/REPORT.md`). Going to 64
  doubles sandbox load per sample, and the server caps at 64.
- **Difficulty prefilter (P11).** Needs an Instella n=8 GPU pass on the final tests. Then drop 8/8, keep 20–25%
  at 6–7/8, and cap 0/8 at about 15%. 22% of the v6 solve labels are stale.
- **The 18 swapped Nemotron rows.** final_v2 replaced 18 verified Nemotron function rows with stdio copies of the
  same CF problem that have more tests. `final_v2/REPORT.md` section 1 explains how to revert.
- **Prompt-format alignment (P8).** The training prompts have no answer-format instruction, while the eval
  prompts do. Adding one per-route instruction changes scores, so freeze it before evals and disclose it.
- Also: drop the legacy `CODE_PARTIAL_CREDIT` alias, and decide whether to filter pure-Python-`__eq__` rows.
