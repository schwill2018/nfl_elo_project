# NFL Elo

Prospective NFL Elo model for team ratings, win probabilities, rankings, and game projections. The notebook is the readable model-development version; `nfl_elo.py` is the repeatable production script.

## Overview

- Pulls current NFL schedules and results with `nflreadpy`.
- Updates team ratings through all completed games.
- Produces current rankings, daily/weekly projections, remaining-season projections, and historical game-level output.
- Uses the general Elo framework represented by FiveThirtyEight's NFL Elo model and public NFL Elo applications such as nfelo, while using this project's own tuning and production logic.

## Model

- Starting rating: 1,000
- Logistic win probability with a tuned home-field advantage
- Margin-of-victory adjustment using a log margin and strength-based correction
- Offseason regression toward the league mean
- Prospective expanding-window parameter tuning

The margin-of-victory adjustment gives a larger rating update for a larger win, while reducing the adjustment when a strong favorite defeats a much weaker opponent. `mov_scale` is estimated from prior training games.

## Tuning

Parameters for an evaluation season are selected using earlier seasons only. The evaluation season is never used to tune its own parameters.

The saved 2026 parameters were selected using 2018–2025 data:

```text
K = 49        home_adv = 31
carryover = 0.65
mov_scale = 2.209776
```

Those values are reused throughout 2026. When a new season appears without a saved parameter row, the script tunes it once using prior seasons and saves the result in `elo_parameters.csv`.

## Spread projections

The spread calculation follows the pandas/NumPy implementation in `NFL_elo_v3.ipynb`, using RMSE as the fitting objective. For each evaluation season, it fits one multiplier using pregame Elo probabilities and actual scoring margins from the five preceding seasons. The 2026 multiplier uses 2021–2025 only and stays fixed throughout 2026.

`spread_parameters.csv` records the multiplier, training years, number of games, last training game, training RMSE, and fit timestamp. The initial local setup supplies this file. Daily local and GitHub Actions runs reuse saved rows without calling the spread optimizer or rewriting the cache. A missing season is fitted once and appended. Actions saves the cache alongside the existing Elo parameters.

The notebook's conversion is unchanged:

```python
elo_home_margin = spread_slope * np.log(p_home / (1 - p_home))
model_home_spread = -(np.round(elo_home_margin * 2) / 2)
```

`model_home_spread` appears immediately after `p_home` in the projection CSVs and before `outcome` in the weekly CSV. A negative value favors the home team; a positive value favors the away team. The corresponding away spread is its negative. Probabilities already account for neutral venues before being converted to points.

New pregame spreads are saved with the existing prediction archive and freeze at kickoff. Older archived predictions without a recorded spread retain blank spread values.

Saved seasons are never silently refitted. To deliberately refit a season, back up `spread_parameters.csv` and remove that season's row before running. Changing the training window or fitting objective requires an explicit cache rebuild; incompatible saved settings raise an error.

## Performance

Current prospective evaluation output:

```text
Completed games: 857
Accuracy:        65.3%
Log loss:        0.634
```

Only two completed 2026 games are included in this snapshot, so the overall result is primarily a 2023–2025 evaluation.

## Usage

```bash
python -m pip install -r requirements.txt
python nfl_elo.py
```

The script downloads the current schedule/results data, updates Elo, prints the evaluation summary, and writes the output CSVs beside the script.

## Outputs

| File | Contents |
| --- | --- |
| `nfl_elo_history.csv` | Game-level pregame/postgame ratings, probabilities, results, and parameters used. |
| `current_elo_rankings.csv` | Current team ratings and rank order. |
| `daily_projections.csv` | Projections for games scheduled today. |
| `weekly_projections.csv` | Projections for the current/upcoming NFL week. |
| `future_projections.csv` | Projections for every remaining scheduled game using current ratings. |
| `elo_parameters.csv` | Saved season-specific tuning selections. |
| `spread_parameters.csv` | Saved season-specific RMSE spread multipliers using five prior seasons. |
| `historical_predictions.csv` | Latest recorded pregame probabilities and spreads, frozen after kickoff. |
| `performance_games.csv` | Game-level probability evaluation inputs for the website. |

## Verification

```bash
python -m unittest discover -s tests -v
```

The spread checks cover prior-season training, cached runs with no optimizer calls, new-season appends, incompatible caches, neutral venues, CSV column order, and preservation of frozen pregame records. They use temporary output folders without refreshing production CSVs.

## References

- [FiveThirtyEight NFL Elo game repository](https://github.com/fivethirtyeight/nfl-elo-game)
- [nfelo NFL power ratings](https://www.nfeloapp.com/)
- [andr3w321, “Elo Ratings Part 2 – Margin of Victory Adjustments”](https://andr3w321.com/elo-ratings-part-2-margin-of-victory-adjustments/)
