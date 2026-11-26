# Data build for Instella-MoE RL

Scripts that built the training sets for the math, IF and code RL runs. Each folder has its own README with the
steps in order.

| Run | Training file | Rows | Folder | Rebuild check |
|---|---|---:|---|---|
| Math R1 | `mix_v5_math.jsonl` | 25,162 | [`math/`](math/) | identical (see `math/README.md` for one caveat) |
| IF3 | `if_train_hybrid20k.jsonl` | 20,000 | [`if/`](if/) | identical |
| CODE1 | `final_v3_candidate/train.jsonl` | 13,286 | [`code1/`](code1/) | identical |
| CODE2 | `v8_data/train.jsonl` | 24,572 | [`../data_v8/`](../data_v8/) | built by `build_v8.sh` |

"Identical" means re-running the final step on the saved intermediate files reproduces the shipped file byte for byte.

The shipped datasets, with their eval sets and checksums, are in the OCI bucket
`s3://ocieobject1/goramesh/instella-rl-mi325x/data/` (`lux_math_r1/`, `lux_if_if3/`, `lux_code/`).

## Paths

The scripts read and write under one data root, set with `INSTELLA_RL_ROOT`:

- on the MI325 hosts the default is `/it-share-prj2-1/data/goramesh-instella-rl`;
- inside the `instella_rl` container the same share is mounted at `/workspace`, so scripts that run in the
  container default to `/workspace/...`.

All intermediate files (grades, prefilter rollouts, cleaning-stage outputs) are still on that share.

## Difficulty grading

Math and IF prompts are graded with our own DPO checkpoint (`Instella-MoE-16B-A3B-DPO`), 8 rollouts per prompt,
at a 32K response budget. Labels from other models do not transfer: sets picked with other models' difficulty
labels were far too easy for Instella.

`serve_grader.sh` starts the sglang + FarSkip server used for grading (tp = ep = dp = 8 with dp-attention, fp32 LM
head, context 36,864). Run it inside the training container on each 8-GPU node. `FARSKIP_APPLY` points at the
FarSkip overlay script for the node's sglang version.

CODE1 needs no model grading: every row is checked by running reference solutions in the sandboxed code grader
(`../code_exec_server.py`).
