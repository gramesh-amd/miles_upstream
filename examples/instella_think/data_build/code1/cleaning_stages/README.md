# v7 cleaning stages: Dolci code (21,385 rows) to `final_v2` (13,399 rows)

These scripts cleaned the Dolci code_rlvr set (`prepare_code_data.py` output) row by row. The rule: fix a row
when the fix can be verified by execution, otherwise drop it. Every kept row has a reference solution that passes
all of its tests through the sandboxed training grader (`code_exec_server.py`, strict asserts, 5 s per test).
Untrusted code (references, stubs) runs only inside that grader.

They are kept for reference: they were run once, by stage, and read scratch paths (`/tmp/v7*`, the share's
`out/v7_data/<stage>/work/`) for intermediate files. Each stage's `REPORT.md` gives its inputs, decisions and
counts. `BRIEF.md` is the original task brief (inputs, grader routes, rules); `MERGE_NOTES.md` holds the
corrections applied when merging the stages.

| Stage | Does | Entry points |
|---|---|---|
| `function/` | Assert-style rows: clean tests statically, run every reference and stub, fix or drop, prompt dedup. | `build_inputs.py` → `audit.py` → `decide.py` → `finalize.py`; `verify_grader.py` |
| `stdio/` | Stdin/stdout rows: statement analysis, run references on each row's test pool, fix wrong labels, dedup, decide. | `build_workspace.py` → `analyze.py` → `run_refs.py` → `dedup.py` → `decide.py` |
| `decontam/` | Text (n-gram, Jaccard, numeric runs) and execution checks of train rows against LCB, HumanEval+/MBPP+ (and Pro), Codeforces evals, CRUXEval. | `build_corpus.py` → `detect.py` → `candidates.py` → `exec_verify.py` → `verdicts.py` |
| `new_sources/` | Survey of extra stdio sources (code_contests, TACO, INTELLECT-3): decontam, dedup against our rows, funnel. | `build_corpus.py` → `overlap.py` → `funnel.py` |
| `final_v1/` | First assembly (13,241 rows): binary-reward re-decide of both routes, merges, MinHash dedup, ≥ 5-test floor. | `phase1.py` → `phase2.py` → `assemble.py` → `sanity.py` |
| `final_v2/` | Bring back dropped Codeforces rows using Open-R1 statements and checkers, decontam, verify, assemble (13,399 rows). Checker verification ran on Lux. | `prepare.py` → `decontam.py` → `groups.py` → `verify.py` → `finalize.py`; `verify_checkers.py`, `merge_checker_rows.py` |

`final_v1/gclient.py` and `final_v2/gclient2.py` are the grader clients that `build_v3/` also uses
(`V7_GRADER_URL`).
