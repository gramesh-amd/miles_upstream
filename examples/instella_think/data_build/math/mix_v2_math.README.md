# mix_v2_math.jsonl

Built by `instella-farskip/scripts/mix_build_v2.py`. Inputs, all already on
disk (nothing downloaded; works under `HF_HUB_OFFLINE=1`):

- `data/dolci_math_filtered_deduped_nomrt.jsonl` (17,249) -- the pool the
  366-step run used
- `data/math_prompts.jsonl` (30,180 rows / 24,188 unique) -- the full math
  slice of `allenai/Dolci-Think-RL-7B`
- `hf_home/hub/datasets--allenai--Dolci-Think-RL-7B` -- parquet, for
  `original_dataset` provenance and AI2's `passrate`

## Schema

Identical in shape to the file the run already loads, so
`--input-key prompt --label-key label` needs no change:

```json
{"prompt": [{"role": "user", "content": "..."}], "label": "...",
 "metadata": {"dataset_source": "hamishivi/..._filtered"}}
```

`metadata.rm_type` is deliberately absent: it outranks `--rm-type` in miles,
which is why `scripts/strip_rm_type.py` exists.

## Transformations, in order

1. **Widen the candidate set.** 17,249 -> 24113 unique prompts, by adding
   the 6889 math prompts that are in the offline Dolci slice but were
   removed by the inherited dedup/difficulty pass.
2. **Gate labels on the real grader.** For each label we generate the
   renderings a competent model plausibly emits (whitespace, `\dfrac` vs
   `\frac`, `$...$`, `\text{}`, thousands separators, `^\circ`, `x = v` ->
   `v`, fraction vs decimal) and run every one through
   `get_deepscaler_rule_based_reward`. A row is dropped if any rendering fails,
   if the label is prose rather than a value, if it is empty, or if it exceeds
   40 characters.
3. **Drop near-duplicates of held-out problems.** Any prompt whose 14-gram
   Jaccard against an AIME-2024/2025 problem is >= 0.25 is removed. Exactly
   one row qualifies, at 0.355: a perturbed copy of an AIME-2024 problem
   (`n^4+1` divisible by `p^2` becomes `n^4+(-1)` divisible by `p^4`) carrying
   a different answer. Shared boilerplate tops out around 0.13, so the
   threshold separates cleanly. 0 exact and 0 canonical matches exist.
4. **Cap template families at 1%.** The family key is the first 8 words
   lowercased with digit runs collapsed to `#`, so the `<n>-gon` generator
   counts once instead of once per `n`.
5. **Cap OMEGA at 30%** (was 48.0%).
6. **Cap AI2-unsolved rows at 10%.** Rows Olmo-3-7B never solved in 8
   tries are the likeliest to sit at 0/8 for us too: zero GRPO gradient at full
   generation cost. Capped rather than removed, because the cross-model
   transfer is imperfect.
7. **Match the eval prompt format.** Every prompt without an existing
   answer-format instruction gets the exact AIME suffix appended:
   `\n\nPresent the answer in LaTex format: \boxed{Your answer}`.

Within any contested cap, rows are ranked by AI2 `passrate` descending with a
sha1 tiebreak. That is deliberate: every row in the offline slice sits at
`passrate <= 0.625`, while AIME-2024 under our own DPO init has a median
per-problem pass rate of 0.969, so the top of the available band is the end
closest to the benchmark.

## Rows dropped

| reason | rows | examples |
|---|---|---|
| `prose_label` | 116 | `a_0 = 0`, `[4; 2017]`, `a_n = n` |
| `label_too_long` | 53 | `\left(\frac{1+2^{1005}}{2^{1007}}\right)^{2}`, `(x^2 + y^2z^2)(y^2 + x^2z^2)(z^2 + x^2y^2) \geq 8xy^2z^3`, `\frac{1}{4}((\sqrt{2}+1)^{2023}-(\sqrt{2}-1)^{2023})^{2}` |
| `grader_fragile` | 3 | `x=y=z=1`, `x = y = z = 1`, `x = y = z = \sqrt{2}` |
| `aime_near_duplicate` | 1 | `Let p be the least prime number for which there exists a positive integer n such that n^4+` |

## Before / after

```

=== SOURCE MIX  before -> after
source                         before   share    after   share
omega-combined-no-boxed          8275   48.0%     5411   30.0%
AceReason-Math                   4276   24.8%     6108   33.9%
DAPO-Math-17k-Processed          1925   11.2%     2370   13.1%
MathSub-30K                      1463    8.5%     2022   11.2%
rlvr_orz_math_57k_collected      1310    7.6%     2125   11.8%
TOTAL                           17249  100.0%    18036  100.0%

=== TEMPLATE CONCENTRATION (digit-normalised first 8 words)
  before: 9727 families / 17249 rows  ratio 0.564  top family 9.72%
  after : 13075 families / 18036 rows  ratio 0.725  top family 1.00%
  biggest families before:
      1676  9.72%  what is the probability of such event happening:
       888  5.15%  suppose you have a #-gon, with vertices numbered
       310  1.80%  there are # people standing in a line
       224  1.30%  supposing you start in the bottom left (#,
       126  0.73%  calculate the area enclosed between the functions f(x)
  biggest families after:
       180  1.00%  what is the probability of such event happening:
       180  1.00%  there are # people standing in a line
       180  1.00%  suppose you have a #-gon, with vertices numbered
       180  1.00%  supposing you start in the bottom left (#,
       151  0.84%  define m to be the number of ordered

=== ANSWER MAGNITUDE
  before  int labels 13388/17249 (77.6%)  median   13  <=20  56.8%  >=100  23.7%
  after   int labels 13809/18036 (76.6%)  median   21  <=20  48.7%  >=100  32.2%
  AIME-24 reference: 100.0% int, median 207.5, <=20 3.3%

=== ANSWER-FORMAT INSTRUCTION
  before 341/17249 (2.0%)   after 18036/18036 (100.0%)

=== DIFFICULTY (AI2 passrate, Olmo-3-7B not us)
  before  n=17249 mean 0.3210 median 0.286  ==0  10.8%  >=0.5  32.4%
  after   n=18036 mean 0.4024 median 0.464  ==0  10.0%  >=0.5  50.0%

```

## Caveats

- AI2's `passrate` was measured against **Olmo-3-7B**, not against
  Instella-MoE-16B-A3B. It is used only as a ranking prior, never as a hard
  difficulty claim.
- Capping OMEGA at 30% rather than lower is a judgement call: non-OMEGA supply
  is the binding constraint, and cutting OMEGA further would shrink the pool
  below the current 17,249 for no diversity gain.
- Appending the eval suffix changes the prompt distribution the policy sees.
  It makes train and eval consistent, which is the point, but it also means
  reward numbers are not directly comparable to the 366-step run.
- Answer magnitude is still far from AIME's. Reweighting sources moves the
  median but cannot manufacture large-integer answers that are not in the pool.
