# Data build for Instella-MoE RL

Scripts that built the math and IF training sets, which were built and trained on this MI325 cluster. Each folder
has its own README with the steps in order.

| Run | Training file | Rows | Folder | Rebuild check |
|---|---|---:|---|---|
| Math R1 | `mix_v5_math.jsonl` | 25,162 | [`math/`](math/) | identical (see `math/README.md` for one caveat) |
| IF3 | `if_train_hybrid20k.jsonl` | 20,000 | [`if/`](if/) | identical |

"Identical" means re-running the final step on the saved intermediate files reproduces the shipped file byte for byte.

The code sets (CODE1 `final_v3`, CODE2 `v8_data`) are built on the
[`instella-code-rl-lux-v8`](https://github.com/gramesh-amd/miles_upstream/tree/instella-code-rl-lux-v8/examples/instella_think/data_build)
branch (`data_build/code1/` and `data_v8/`). They need that branch's code grader (`code_exec_server.py` /
`code_reward.py`, with the checker route and strict grading), which is newer than the one here.

The shipped datasets, with their eval sets and checksums, are in the OCI bucket
`s3://ocieobject1/goramesh/instella-rl-mi325x/data/` (`lux_math_r1/`, `lux_if_if3/`, `lux_code/`).

## Paths

The scripts read and write under one data root, set with `INSTELLA_RL_ROOT`:

- on the MI325 hosts the default is `/it-share-prj2-1/data/goramesh-instella-rl`;
- inside the `instella_rl` container (`mi325/bringup_node.sh`) the same share is mounted at `/workspace` and this
  checkout at `/root/miles`, so scripts that run in the container default to those.

All intermediate files (grades, prefilter rollouts) are still on that share.

## Difficulty grading

Math and IF prompts are graded with our own DPO checkpoint (`Instella-MoE-16B-A3B-DPO`), 8 rollouts per prompt,
at a 32K response budget. Labels from other models do not transfer: sets picked with other models' difficulty
labels were far too easy for Instella.

`serve_grader.sh` starts the sglang + FarSkip server used for grading (tp = ep = dp = 8 with dp-attention, fp32 LM
head, context 36,864). Run it inside the training container on each 8-GPU node. It re-applies the FarSkip overlay
from `mi325/sglang_port/apply_farskip.sh` (idempotent); set `FARSKIP_APPLY=` to skip that.
