# Code: CODE1 (`final_v3_candidate/train.jsonl`, 13,286 rows)

CODE1 trains on the Dolci code set (21,385 rows, the data of the earlier v5 runs) after cleaning and re-mixing:
10,555 unique problems plus 2,731 second copies of stdio problems, so 45% of draws are stdio.

Two parts:

- [`cleaning_stages/`](cleaning_stages/): the v7 cleaning that turned the 21,385 Dolci rows into `final_v2`
  (13,399 rows). Every kept row has a reference solution that passes all its tests in the sandboxed grader.
- [`build_v3/`](build_v3/): the last step, `final_v2` to `final_v3_candidate`: dedup, restores, checker rows and
  mix B.

`final_v3_REPORT.md` lists every `build_v3` step with its row counts. `data_audit.md` has the evidence for each
change (why stdio-heavy, why dedup, which drops were wrong).

## build_v3 steps

Run from `build_v3/`. `INSTELLA_RL_ROOT` is the data root. `CODE1_WORK` holds the intermediate files (default
`$INSTELLA_RL_ROOT/out/v7_improve/data_audit`). Steps marked "grader" run reference solutions through
`../../../code_exec_server.py` via `run_grader_job.sh`.

| # | Script | Does | Writes (in `CODE1_WORK`) |
|---|---|---|---|
| 1 | `v6_solve.py` | Per-problem solve stats from the v6 training rollouts. | `v6_solve.json` |
| 2 | `master.py` | One row per Dolci row: v6 stats, final_v1/final_v2 drop reasons, route. | `master.csv` |
| 3 | `clusters.py` | Group duplicate rows into clusters; label each row's drop category. | `master2.csv` |
| 4 | `build_cands.py` | Collect stdio rows that cleaning dropped but may be fine (decorative images, multi-answer, few tests). | `restore_cands.jsonl.gz` |
| 5 | `verify_restore.py` (grader) | Restore a row only if a reference solution passes every test. | `restore_verify.jsonl` |
| 6 | `../cleaning_stages/final_v2/verify_checkers.py` (grader) | Verify Open-R1 checkers for multi-answer rows. | `checkers_s{0,1}/accepted_rows.jsonl` |
| 7 | `build_v3.py --mix B --write` | Trim huge tests, drop slow references, add restores and checker rows, dedup (`neardup.py`, `testdup.py`), apply mix B (keep 1/3 of easy AceCoder rows, second copy of stdio rows). Seed 0. | `--out` dir: `train.jsonl`, `dropped_v3.csv`, `stats.json` |
| 8 | `sanity_v3.py` (grader), `final_stats.py` | References must pass and stubs must fail; counts and schema checks. | `sanity.json`, `stats.json` |

```bash
python3 build_v3.py --mix B --out /tmp/final_v3 --write
bash run_grader_job.sh sanity 18991 32 python3 sanity_v3.py /tmp/final_v3
python3 final_stats.py /tmp/final_v3
```

Rebuild check: step 7 on the saved intermediate files reproduces the shipped `train.jsonl` and `dropped_v3.csv`
byte for byte (sha256 of `train.jsonl` starts `6e733cd6`, the same as `code1_train.jsonl` in the bucket).
