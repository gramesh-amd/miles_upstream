# Math: `mix_v5_math.jsonl` (25,162 prompts)

Polaris-led mix, graded with our DPO model. Keep 0/8 to 7/8 and drop only 8/8: the Polaris ablation found that
removing only 8/8 beats both no filtering and a harsher ≤4/8 cut, and OLMo 3 also filters only the easy side.
mix_v5 then adds the eval's `\boxed{}` answer instruction to every prompt.

Paths below are relative to `$INSTELLA_RL_ROOT`; `A` = `out/analysis_1012`.

## Steps

| # | Script | Does | Writes |
|---|---|---|---|
| 1 | `mixv4_recover_grades.py` | Download the 30,180-prompt Dolci math pool and the DPO grades already measured on it (8 rollouts, T=1.0) from the backup bucket. Uses `oci_ls.py` and `~/.oci_s3_creds.env`. | `A/dolci_filter/pool_30k.jsonl`, `grades_30k.jsonl` |
| 2 | `mixv4_fetch_dapo.py` | Download DAPO-Math-17k-Processed (English) and convert to our schema. | `A/dapo/dapo_17k_en.jsonl` (13,983) |
| 3 | `mixv4_build_polaris.py` | Take Polaris difficulty bands 1–5; dedup against every prompt we already hold and against the AIME eval sets. | `A/polaris/pool_band1_5.jsonl` (12,511) |
| 4 | `mixv4_grade_parallel.sh` | Grade DAPO and Polaris with DPO on several nodes: starts `../serve_grader.sh`, runs `filter_rl_difficulty.py` (8 rollouts, T=1.0, top_p 1.0, 32K) per shard. Resumable. | `A/{dapo,polaris}/grades/` |
| 5 | `build_mix_v4.py` | Join the three graded sources; keep 0/8–7/8; drop OMEGA (13,076) and ZebraLogic puzzles (614); dedup across sources. | `data/mix_v4_math.jsonl` |
| 6 | `build_mix_v5.py` | Append the eval's answer instruction, byte-identical to `aime-2024-olmes.jsonl`. | `data/mix_v5_math.jsonl` |

Grade step 4 once per source:

```bash
INPUT=$INSTELLA_RL_ROOT/out/analysis_1012/dapo/dapo_17k_en.jsonl \
OUTDIR=$INSTELLA_RL_ROOT/out/analysis_1012/dapo/grades NODES="g2 g15 g32" bash mixv4_grade_parallel.sh

INPUT=$INSTELLA_RL_ROOT/out/analysis_1012/polaris/pool_band1_5.jsonl \
OUTDIR=$INSTELLA_RL_ROOT/out/analysis_1012/polaris/grades NODES="g2 g15 g32" bash mixv4_grade_parallel.sh
```

`build_mix_v4.py` reads grades from the node names it was run with (`NODES` near the top of the file). Edit it if
you grade on other nodes.

## Notes

- `mixv4_build_polaris.py` dedups against the earlier `data/mix_v2_math.jsonl` and `data/mix_v3_aime.jsonl` too.
  These are on the share.
- The recovered Dolci grades use the OLMo 3 math verifier, which is more lenient than the training reward
  (`deepscaler` needs a closed `</think>`), so those pass rates are upper bounds.
- Rebuild check: `build_mix_v5.py` on the shipped `mix_v4` gives the shipped `mix_v5` byte for byte. Re-running
  `build_mix_v4.py` on the share today gives 25,169 rows: the shipped 25,162 in the same order, plus 7 Polaris
  prompts whose grades were written after the build.
