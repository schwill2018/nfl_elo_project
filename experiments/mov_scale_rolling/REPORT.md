# Five-season versus expanding MOV normalization

The five-season version is a reasonable candidate to keep testing, but the
overall improvement is very small. This experiment does not justify changing
production by itself. `nfl_elo.py` and every production CSV remained unchanged;
their SHA-256 hashes were checked before and after the run.

## What the rating history means

The production function does two things with the 2003–2007 games:

1. It chooses the settings for the first evaluation season, 2008.
2. It replays those games with the selected settings to establish the team
   ratings used at the start of 2008.

The second step is in the bootstrap loop at `nfl_elo.py:299`. Later seasons
inherit the preceding final ratings, apply offseason regression, and continue
with their own selected settings. The ratings are not reset to 1000 each year.

Concrete values from the saved production history:

- NYG begins its first 2008 game at **1068.7602**, after the warm-up replay.
- NYG ends 2008 at **1114.6467**.
- With 2009 carryover of 0.5, NYG starts 2009 at
  `0.5 × 1114.6467 + 0.5 × 1000 = 1057.3234`.
- BUF ends 2025 at **1150.5134** and starts 2026 at
  `0.575 × 1150.5134 + 0.425 × 1000 = 1086.5452`.

This shows ratings carrying between seasons separately from parameter selection.
It does not mean 2003 games belong to the 2026 five-season tuning window, or
that their remaining direct effect on current ratings is substantial.

## Comparison design

- Keep the same five-season parameter-tuning window, parameter grid, starting
  year, probability formula, overtime handling, game ordering, and offseason
  regression as production.
- Expanding MOV scale: average `log(abs(home_score-away_score)+1)` over all
  available completed games from 2003 through the prior season.
- Rolling MOV scale: the same calculation over the latest five prior seasons.
- Estimate the scale before the evaluated season; hold it fixed for that season.
- Run both a controlled change with saved K/HFA/carryover and a full comparison
  with the settings retuned separately for each MOV-scale policy every year.
- In the retuned comparison, each policy has its own continuous rating path and
  its own snapshots for initializing the five-season tuning simulations.
- Score non-ties with binary home-win log loss, matching the existing code.

The evaluated games are the existing `nfl_elo_history.csv` snapshot, through
**2026-10-12**. Only the 2003–2007 warm-up results were supplied from the public
nflverse schedule download. No production data was refreshed. Complete seasons
2008–2025 are the main comparison; partial 2026 is reported separately.

## Results with seasonal retuning

Lower log loss and Brier score are better. Accuracy is the fraction of correctly
predicted non-tied winners.

| Evaluation period | Games | Expanding log loss | Rolling-five log loss | Expanding accuracy | Rolling-five accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2008–2025, complete | 4,883 | 0.628857 | 0.628725 | 64.63% | 64.69% |
| 2021–2025, complete | 1,420 | 0.639541 | 0.639072 | 63.73% | 63.87% |
| 2026, partial | 78 | 0.635478 | 0.635401 | 62.82% | 62.82% |

The rolling policy changed the selected K in 2021 (51 to 48) and 2023 (48 to
45). Other K, home-advantage, and carryover selections were unchanged. For
2026, MOV scale is **2.249689** expanding versus **2.202424** rolling.

With the saved K/HFA/carryover held fixed, rolling scale alone slightly worsened
complete-period log loss: **0.628857 to 0.628874**. The improvement in the full
comparison comes with the retuned settings, not from a consistent benefit of
changing the normalizer alone.

## Interpretation

A five-season scale responds more quickly to changes in score margins; the
expanding scale is steadier. The scale need not use the same window as parameter
tuning. Because MOV scale divides the rating update while K multiplies it,
changing the scale also changes the effective update size, making retuning
important when comparing the two policies.

The full-period retuned improvement is **0.000132 log loss**. Rolling was better
in 9 of 18 complete seasons. A descriptive paired season-block bootstrap
(10,000 resamples, seed 42) gives a rolling-minus-expanding interval of
approximately **[-0.000303, +0.000001]**, which includes zero.

The recent-five-season difference is **-0.000469**, with rolling better in 4 of
5 seasons. That is encouraging but based on only five season blocks. These
historical results are exploratory: the same historical years have already
informed model development, and this is not a newly untouched holdout.

## Verification and files

- Reconstructed baseline matched saved production pre/postgame Elo to within
  **4.55e-13** and home-win probability to within **7.22e-16**.
- Twelve candidate-score checks against the original source implementation,
  across both window policies and early/latest tuning windows, matched exactly.
- Retuned expanding parameters reproduced the production parameter selections.
- Production file hashes were identical before and after the experiment.

The experiment runner is `run_experiment.py`. Outputs in this directory:

- `summary.csv`: aggregate metrics for all four variants.
- `by_season.csv`: season-level metrics.
- `comparisons.csv`: paired differences and descriptive bootstrap intervals.
- `parameters.csv`: season-specific settings for all variants.
- `predictions.csv`: pre/postgame Elo and probabilities for all variants.
- `rating_carry_examples.csv`: exact year-to-year rating examples.
- `validation.json`: input hashes, scope, and verification results.
- `input_games.csv`: the combined game snapshot used by this run.
- `source_games.csv`: the public schedule download used for warm-up seasons.

Run from the repository root with a Python environment containing numpy/pandas:

```powershell
python -B experiments/mov_scale_rolling/run_experiment.py
```

The runner reads the production history and parameter files but writes only
inside this experiment directory. Re-running after production inputs change
will evaluate the newer snapshot.
