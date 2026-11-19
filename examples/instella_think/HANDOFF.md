# Code RL v7: grader, reward and data handoff

Branch `instella-code-rl-lux-v7`: `instella-code-rl-lux` (the branch Lux runs use) with `code-rl-v7-fixes` merged
in (merge commit, both histories intact, a fast-forward of the lux branch). `code-rl-v7-fixes` (base `f610d0d915`,
code-evals-v2) closes reward hacks in the code grader, makes the training reward all-or-nothing by default, adds
reward-hack monitoring and grades checker rows; the lux side brings `code_exec_server --processes N`, the reward's
own thread pool, any number of rollout nodes / an optional second sandbox node, the offline and online difficulty
filters, `run.sbatch`, the sglang MLA rope patches and profiling.

**Lux runs launched from `instella-code-rl-lux` before this merge: read "Changed defaults" below before resuming
one from this branch.** The default code reward switches from partial (pass rate) to binary (all tests pass).

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
| merge into lux | `instella-code-rl-lux` + the above; conflicts in README, server `main`, `code_reward` (imports, gating, self-test), `run_code.sh` echo list | one branch for Lux runs |
| lux-v7 follow-ups | `/health` and the log sum all `--processes`; port-in-use refusal; extra processes exit with the first | per-process STATS hid most of the server's traffic |
| | `rollout_log_chain.py` | `per_domain_logger` silently replaced `code_reward.log_rollout_data` (miles keeps the last `--custom-rollout-log-function-path`) |
| | `train_all_pass` grades like `_score_one` (checker, strict, stdio options); solve = full reward; cache header records them; Lux prefilter containers get N / MAX_PASS_RATE / `CODE_*` (`--cleanenv` dropped them); `PREFILTER_SHARDS` | the prefilter graded v7 rows more leniently than training |
| | `run.sbatch` passes `CODE_EXEC_PROCESSES` / `CODE_BENCH_PROCESSES` to the sandbox node | spec sizing ignored |

The evidence is in `/tmp/v7reg/{REVIEW,BEFORE_AFTER,FIXES,SURVEY}.md` on the login node. BEFORE_AFTER.md counts the
hacks closed, the false negatives fixed and the labelled regressions. Those files are in `/tmp`, so copy them next
to the data if they should outlive the node.

## Changed defaults for runs from `instella-code-rl-lux`

| what | before (lux) | now | to keep the old behaviour |
|---|---|---|---|
| training reward | pass rate (`CODE_PARTIAL_CREDIT=1` default) | `CODE_REWARD_MODE=binary`, threshold 1.0 | `CODE_REWARD_MODE=partial` (or `CODE_PARTIAL_CREDIT=1`) in the spec |
| assert grading | plain asserts, program run as `__main__` | strict canon, module `tmp_sol` (`CODE_STRICT_GRADING=1`) | `CODE_STRICT_GRADING=0` |
| early exit | `sys.exit(0)` before the asserts passed | fails the test (train and eval) | none (it was a reward hack) |
| checker rows | plain stdout compare | `/test_program_checker` | none |
| W&B group name | `...-passrate-...` | `...-allpass-...` (`passrate` under partial) | set `WANDB_GROUP` |
| rollout log hooks | `per_domain_logger` only | `per_domain_logger` + `code_reward.log_rollout_data` via `rollout_log_chain.py` | `CUSTOM_ROLLOUT_LOG_PATH=""` |
| prefilter cache header | 11 keys | + `code_solve_pass_rate`, `code_strict_grading` | an old cache is refused; use a new `OUTPUT=` or `RESET_CACHE=1` |

A run resumed with the same `SAVE_TAG` keeps its W&B run (the id comes from `SAVE_TAG`), so a mid-run switch to the
binary reward would show up as a step change in `rollout/raw_reward`; pin `CODE_REWARD_MODE=partial` in the spec of
any resumed partial-credit run. Under the binary reward the dynamic filter drops every all-pass and all-fail group
as `zero_std`, so `frac_kept` falls and steps pull more prompts. Restart the sandboxes (or resubmit `run.sbatch`)
so server and client match.

## How to run

### Lux (`run.sbatch`, one Slurm job per run)

Node 0 runs the sandboxes (`SANDBOX_NODES=2`: node 0 rewards, node 1 benchmarks), the rest are GPU nodes (1 trainer +
N-1 rollout nodes). The arguments are the run spec; `run_in_container.sh` exports it on the head before
`run_code.sh`, and `run_async.sh` hands every `CODE_*` variable below to the Ray workers in the runtime env (checked
with `DRY_RUN=1`: `CODE_REWARD_MODE`, `CODE_REWARD_PASS_THRESHOLD`, `CODE_STRICT_GRADING`, `CODE_EXEC_API_RETRIES`,
`CODE_HACK_WINDOW_MAX`, `ROLLOUT_LOG_HOOKS` all arrive; `CODE_TRAIN` becomes `--prompt-data`). From `~/miles`:

```bash
# v7 data on Lux: copy final_v2/train.jsonl over first, e.g. to $LUX_ROOT/data/code-train/v7_final_v2.jsonl
sbatch -A ven136 -N 4 -t 24:00:00 -o ~/logs/%x-%j.out examples/instella_think/cluster/lux/run.sbatch \
  SAVE_TAG=code32k_v7_binary CODE_TRAIN=/lustre/orion/ven136/scratch/$USER/instella/data/code-train/v7_final_v2.jsonl
# add CHECK_ONLY=1 first: expect "reward : CODE_REWARD_MODE=binary threshold=1.0 CODE_STRICT_GRADING=1"
```

Sandbox sizing goes in the spec too: `CODE_EXEC_WORKERS`, `CODE_BENCH_WORKERS`, `CODE_EXEC_PROCESSES` (default 4),
`CODE_BENCH_PROCESSES` (default 16 when `CODE_BENCH_WORKERS` >= 16). The Lux sandboxes run the SIF's
`/opt/venv/bin/python` (3.10, numpy), so numpy rows grade there.

**`--processes N`.** N server processes share the port (SO_REUSEPORT), each with `workers / N` workers, its own
1024 listen backlog, scratch subdir `p<i>` (own content-named guard file), `tests_ref` cache and `strict_assert`
cache. `/health` and the periodic log line sum the counters of all N (`per_process_requests` shows the split,
`process` is the process that answered); per-request `flags` come from the process that ran the request. A server
with N > 1 refuses a port that is already in use (a stale server would otherwise share the traffic), needs a fixed
`--port`, and its extra processes exit when the first one does. One request's tests fan out over only `workers / N`
workers, so a single many-test request is slower than on a one-process server with the same `--workers` (LCB call
route: ~1.0 s vs ~0.4 s per answer at 12 workers, 2 vs 1 processes); throughput under load is higher.

### Crusoe / Spur

```bash
I=examples/instella_think
$I/cluster/code_exec.sh start && $I/cluster/code_exec.sh start bench    # status / stop / url as before
# check the training grader's interpreter: <= 3.12 (it refuses 3.13+), numpy if rows need it
ssh $CODE_EXEC_HOST 'python3 -c "import sys; print(sys.version)"; python3 -c "import numpy"'
$I/cluster/on_node.sh $J $NODE cluster/launch.sh $HEAD_IP CHECK_ONLY=1 \
  CODE_TRAIN=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/train.jsonl
# expect "reward : CODE_REWARD_MODE=binary threshold=1.0 CODE_STRICT_GRADING=1", then launch without CHECK_ONLY
```

`CODE_TRAIN` still defaults to the old Dolci file, so v7 runs **must** set it (in the spec on Lux,
`CODE_TRAIN=...` on `launch.sh` on Crusoe; `PROMPT_DATA` overrides it). `cluster/code_exec.sh` starts the
training grader with the host's `python3` and no numpy, so numpy-using training rows score 0 there. To
reproduce an old pass-rate run, set `CODE_REWARD_MODE=partial`.

### Difficulty prefilter (offline, on the lux branch)

`difficulty_filter.py` (OLMo 3 step 2: 8 rollouts per prompt from the starting checkpoint at T=1.0, drop prompts
solved > 62.5% of the time, keep 0/8) driven by `prefilter.sh` on one node or `cluster/lux/prefilter.sbatch` across
many (README "5b", `cluster/lux/README.md`). Code rows are graded by `code_reward.train_all_pass`, which grades
exactly as `_score_one` does: think gate, `metadata.checker` rows, strict asserts (`CODE_STRICT_GRADING`), per-row
`float_tol` / `case_insensitive`. A solve is a full training reward (pass rate >= `CODE_REWARD_PASS_THRESHOLD`
under binary, every test under partial), so it never reads `code_pass_rate` and works with either mode. Set the
grading knobs as the run will; they are recorded in the cache header.

It works on the v7 data as is (`prompt` message list, `label`, `metadata`; checked on 3 final_v2 rows with a stub
engine: rigged `__eq__` answers count 0/4 with strict grading and 2/4 without). On Lux, for final_v2:

```bash
L=examples/instella_think/cluster/lux
export PREFILTER_SHARDS=$LUX_ROOT/data/prefilter/v7_shards     # beside the Dolci shards
python3 $L/prefilter_shards.py split 14 $LUX_ROOT/data/code-train/v7_final_v2.jsonl
sbatch -A ven136 -N 16 -t 9:00:00 -o ~/logs/%x-%j.out --export=ALL,PREFILTER_SHARDS=$PREFILTER_SHARDS \
  $L/prefilter.sbatch                  # prefilter_node.sh passes N, MAX_PASS_RATE, CODE_* and PREFILTER_SHARDS on
python3 $L/prefilter_shards.py status
python3 $L/prefilter_shards.py merge $LUX_ROOT/data/prefilter/v7_final_v2.filtered.jsonl
# then train with CODE_TRAIN=$LUX_ROOT/data/prefilter/v7_final_v2.filtered.jsonl
```

Each kept row gets `metadata.prefilter = {n, n_correct, n_truncated}`, so another band (the "drop 8/8, keep 20–25%
at 6–7/8, cap 0/8 at ~15%" idea below) can be cut from the output without regenerating.

### Online difficulty filter

`ADAPTIVE_FILTER=1` (`adaptive_difficulty_filter.py`) drops groups whose share of solved samples exceeds
`ADAPTIVE_FILTER_MAX_PASS` (0.75). Solved = reward >= `SOLVED_REWARD`, which `run_code.sh` sets to
`CODE_REWARD_SCALE`: under binary that is pass rate >= threshold, under partial every test. `per_domain_logger`'s
`solve_rate` uses the same rule. Neither reads `code_pass_rate`.

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
| `CUSTOM_ROLLOUT_LOG_PATH` | `code_reward.log_rollout_data` (run_code.sh) | `""` disables the W&B hook; runs beside `per_domain_logger` (`PER_DOMAIN_METRICS=1`) through `rollout_log_chain.py` |
| `CODE_EXEC_TESTS_CACHE` | `256` | server: cached `tests_ref` files, per server process |
| `SOLVED_REWARD` | `CODE_REWARD_SCALE` (run_code.sh) | full reward for the online filter and `domain/*/solve_rate` |

Server flags (`code_exec_server.py`): `--host 0.0.0.0` (use `127.0.0.1` anywhere but the sandbox host),
`--port 18950`, `--token-file ~/.code-exec-token`, `--workers cpus-4` (total over processes), `--processes 1`,
`--mem-mb 2048`, `--max-tests 64`, `--max-timeout 10`, `--python <its own>` (<= 3.12), `--tests-dir ""`,
`--scratch /tmp/code-exec-<uid>`, `--stats-every 60`. Request fields and the response `flags` are documented in the
module docstring.

The reward client grades on its own thread pool (`CODE_REWARD_CONCURRENCY` threads, lux `6183014a31`). `STATS`,
`WINDOW` (the bounded metrics window) and the HTTP session are lock-protected, and each request's `flags` go into
that sample's own dict, so concurrent `_score_one` calls are safe (`test_code_hack_monitor.py` checks 500 calls
on 64 threads; the smoke below checks 256 through a real server). Both sandbox URLs (`CODE_EXEC_API_URL` for
training and checker rows, `CODE_BENCH_API_URL` for benchmark rows, possibly on the second sandbox node) take the
same routes, retries and flags.

## How to verify

```bash
GRADER_PYTHON=<venv with numpy>/bin/python bash examples/instella_think/test_grader.sh
GRADER_PROCESSES=4 GRADER_WORKERS=16 GRADER_PYTHON=... bash examples/instella_think/test_grader.sh  # Lux layout
python3 examples/instella_think/test_code_hack_monitor.py     # offline part only, no server
```

Expected results, with one server process and with `--processes 4` alike (each run ~50 s on the login node):
- hardening: 57/57;
- adversarial: 116/143. The 27 failures are known (24 residual, 3 policy): 0 attacks succeed and there are
  0 false negatives;
- hack monitor: 216/216 in both reward modes (188 without a server);
- self-test: all ok, `api_error=0`;
- the 3.13 refusal and the lux copy check both pass.

Merge verification (2026-11-18, `instella-code-rl-lux-v7`):
- concurrency, `--processes 4` on a CPU node: 128 connections released together x 4 rounds = 512 requests in 7.1 s,
  0 transport errors, 0 wrong results or flags, `/health` deltas exact (requests, early_exits, strict_rejects,
  timeouts; errors +0), split 126/136/127/123; then 256 `batched_reward` samples at `CODE_REWARD_CONCURRENCY=64`:
  per-sample rates, rewards and flags right, `STATS` and `WINDOW` sums exact;
- eval parity, function benchmarks (5,420 answers, `--processes 4`, 469 s): 0 differences from
  `code-rl-v7-fixes` on the same node; the 20 flips against `/tmp/v7reg/old_fn.json` (mbppplus/255,
  HumanEval/39) are the v7 branch's own, pass sequentially on both branches, and were timeouts under load in the old run;
- eval parity, LCB v3 (363 call-route + 100 stdio answers, login node, 258 s): stdio 0 flips; call 7 flips
  False -> True against `old_lcb.json`, and those 7 (plus 2 timing flips of the v7 run) pass sequentially on
  `code-rl-v7-fixes` too, so the merge changes nothing there.

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
- **Difficulty prefilter (P11).** The tool exists (lux branch, "Difficulty prefilter" above) and grades like
  training; what is open is the n=8 GPU pass on final_v2 and the band. Then drop 8/8, keep 20–25% at 6–7/8, and
  cap 0/8 at about 15%. 22% of the v6 solve labels are stale.
- **The 18 swapped Nemotron rows.** final_v2 replaced 18 verified Nemotron function rows with stdio copies of the
  same CF problem that have more tests. `final_v2/REPORT.md` section 1 explains how to revert.
- **Prompt-format alignment (P8).** The training prompts have no answer-format instruction, while the eval
  prompts do. Adding one per-route instruction changes scores, so freeze it before evals and disclose it.
- Also: drop the legacy `CODE_PARTIAL_CREDIT` alias, and decide whether to filter pure-Python-`__eq__` rows.
