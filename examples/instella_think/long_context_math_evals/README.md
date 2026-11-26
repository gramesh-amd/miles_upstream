# Long-context math evals

Competition math across four generation budgets (32k / 64k / 90k / 128k tokens), several models,
two sampling arms and two scoring rules. `models.py` is the executable record of every serving and
sampling decision; this file explains them.

## Problem set and prompt

AIME 2024 (30), AIME 2025 (30), HMMT Feb 2025 (30) and BeyondAIME (100): 190 problems in `data/`,
32 samples each, 6,080 generations per cell. Prompts are zero-shot in OLMES wording, ending
`Present the answer in LaTex format: \boxed{Your answer}`. The question text is identical across
models; only the chat template differs, and per-model template switches (`chat_kwargs`, for example
Gemma's `enable_thinking`) are set in `models.py`.

## Budgets

Each budget is generated at its own length and never extrapolated by truncating a longer run, since
the budget changes how a model reasons, not only where it stops. 90k is Polaris's evaluation budget;
128k matches OLMES's `max_gen_toks` of 131,072. The served context window is the budget plus room for
the forced answer, so forcing calls never overrun it.

## Sampling arms

Reported side by side, because they answer different questions.

- **OLMES standard** (`<model>_olmes`): `temperature=0.6, top_p=0.95`, no `top_k`, no penalties,
  from OLMES `aime:2024::olmo3:adapt`. Applied identically to every model, including those whose
  cards require a penalty. Our model's and SmolLM3's own sampling is byte-identical to this, so they
  have one arm.
- **Per-card recommended** (`<model>`): each model at its card's best-practice settings. Qwen3.5-4B
  is the only model whose card asks for a penalty (`presence_penalty=1.5`).
- **Sampling sweep for our model** (`ours_r1c_399_topk`, `_pen`, `_t08`): each arm changes one
  setting from the previous one. These are tuned beyond our recommended config, so keep them out of
  cross-model comparisons.

## Scoring rules

Both come from the same generations.

- **strict**: a sample that hits the budget scores zero.
- **forced**: a truncated sample is continued with Qwen3's published stop-thinking sentence plus the
  model's own end-of-thinking marker, and whatever it then boxes is graded. Samples that finished on
  their own are identical in both columns.

Strict alone penalises whoever truncates most; forced alone credits an intervention the model did
not perform itself. The end-of-thinking marker is detected from each cell's finished generations
rather than read from the chat template.

## Running

The scripts assume a shared filesystem: this directory, `BENCH_MODELS_DIR` and `BENCH_OUT_DIR` must
resolve to the same paths on the host and on every serving node. Serving goes over ssh, inside
`BENCH_CONTAINER` via `docker exec` when that is set.

```bash
export BENCH_MODELS_DIR=/shared/models          # model directories named as in models.py
export BENCH_OUT_DIR=/shared/out/long_context_math_evals
export BENCH_CONTAINER=my_sglang_container       # optional

# 1. Render prompts, wherever transformers and the tokenizer are available.
python3 prepare_prompts.py --model qwen3_5_4b

# 2a. One cell. Nodes come from BENCH_NODES, or every node in your running slurm jobs.
BENCH_NODES=node-a,node-b python3 run_cell.py --model qwen3_5_4b --length 64k

# 2b. Or against servers that are already up.
python3 run_cell.py --model qwen3_5_4b --length 64k --urls http://node-a:30000

# 2c. Or a whole queue across fixed slots (restartable; finished cells are skipped).
BENCH_SLOTS="node-a:30600 node-b:30700" bash run_sweep.sh queue.example.txt

# 3. Check every cell, then compare two of them.
python3 validate.py
python3 paired.py qwen3_30b_a3b_thinking_64k ours_r1c_399_64k
```

`run_cell.py` refuses to start on nodes running miles training, because the default launcher kills
every sglang process on the node. Set `BENCH_COLOC_MEMF` (and optionally `BENCH_COLOC_MAXRUN`) to
serve next to the training job through `serve_sglang_coloc.sh` instead.

Recovery tools:

- `score_offline.py <model> <length>` writes a strict-only summary from stored generations when a
  cell died during its forcing pass.
- `force_retro.py --model <model> --length <length> --urls ...` adds the forced column to a cell
  whose generations are already on disk.

## Reporting rules

1. Compare models with paired per-problem differences (`paired.py`), bootstrapped over problems,
   never two independent means.
2. State n and k: 190 problems, 32 samples.
3. Label macro vs micro. The mean of four benchmark means differs materially from the mean over
   190 problems, because BeyondAIME is 100 of them.
4. Report the decoding config with every number; it is stored in each summary.
5. Report tokens per correct answer beside accuracy, since efficiency gaps can be much larger than
   accuracy gaps.
6. Keep every generation. Raw traces are written before grading, so scoring can change without
   regenerating anything.

## Pitfalls these scripts guard against

- **Template switches that default off.** Gemma 4 defaults `enable_thinking=False` and GPT-OSS
  defaults to `reasoning_effort=medium`; both silently benchmark a weaker model. The tell is a low
  median token count with zero truncation.
- **Missing card penalties.** Qwen3.5-4B without `presence_penalty` loops on most truncated traces.
- **Over-parallelised serving.** Models that fit one GPU should run tp=1 with dp=8; tp=8 measured
  5.6x slower for the 16B MoE.
- **Forcing calls overrunning the context.** Handled by serving a window wider than the budget.
- **Survivorship from a dead server.** A cell where most generations errored can report a higher
  score than any real cell; `validate.py` flags it.

## Known limitations

- **Instella-MoE at 128k has no forced column.** Forcing boxes 88% of completions at 90k and almost
  none at 128k, even with a larger answer budget: past ~92k the model stops honouring the
  stop-thinking instruction. 131k is deep in YaRN-extrapolated territory for a config advertising
  65,536 positions.
- **Olmo-3-7B-Think could not be served** through sglang's transformers fallback (its
  sliding-window attention crashes or hangs on every backend tried), so it is not in the table.
