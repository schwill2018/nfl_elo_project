"""NFL Elo: standalone version of NFL_elo(4).ipynb, final fit in cell 38.

Install once: python -m pip install numpy pandas nflreadpy pyarrow scikit-learn tzdata
Run:          python nfl_elo.py

Writes five CSVs beside this script, replacing the previous run's outputs.
Keeps seasonal tuning selections in elo_parameters.csv beside this script.
Existing seasons reuse their saved parameters; missing seasons tune once using
prior seasons and the final notebook grid. Delete a season's row to retune it
deliberately after changing model settings or correcting historical data.
Model choices, tuning grids, sorting, overtime coding and MOV formula are
preserved. Historical output begins in 2023; training begins in 2018.
Current ratings come directly from the final fit (the notebook's separate
rankings replay incorrectly begins in 1999 instead of 2018).
"""
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import accuracy_score, log_loss



def null_coalesce(x, y):
    return y if x is None else x

def compress_exp_grid(min_val, max_val, n=10, curve=2):
    grid = np.unique(np.round(
        min_val + (max_val - min_val) *
        (np.exp(curve * np.linspace(0, 1, n)) - 1) / (np.exp(curve) - 1)
    ))
    return grid

def calc_mov_scale(games):
    mean_log_margin = np.log(
        (games["home_score"] - games["away_score"]).abs() + 1
    ).mean()
    return mean_log_margin

def nfl_home_result(home_score, away_score, overtime):
    conditions = [
        (home_score > away_score) & (overtime == 0),
        (home_score < away_score) & (overtime == 0),
        (home_score > away_score) & (overtime == 1),
        (home_score < away_score) & (overtime == 1),
        home_score == away_score
    ]
    choices = [1.00, 0.00, 0.75, 0.25, 0.50]
    return np.select(conditions, choices, default=np.nan)

def elo_season(games, K, ratings=None, scale=400, start_rating=1000,
               use_mov=True, mov_scale=1, use_home_adv=True, home_adv=0):

    ratings = {} if ratings is None else ratings.copy()
    n = len(games)

    if use_mov and (pd.isna(mov_scale) or mov_scale <= 0):
        mov_scale = 1
    if not use_home_adv:
        home_adv = 0

    elo_home_pre = np.zeros(n)
    elo_away_pre = np.zeros(n)
    elo_home_post = np.zeros(n)
    elo_away_post = np.zeros(n)
    p_home = np.zeros(n)

    for i in range(n):
        g = games.iloc[i]

        hid = str(g["home_team"])
        aid = str(g["away_team"])

        Ra = ratings.get(hid, start_rating)
        Rb = ratings.get(aid, start_rating)

        elo_home_pre[i] = Ra
        elo_away_pre[i] = Rb

        p = 1 / (1 + 10 ** ((Rb - Ra - home_adv) / scale))
        p_home[i] = p

        y = nfl_home_result(
            home_score=g["home_score"],
            away_score=g["away_score"],
            overtime=g["overtime"]
        )

        if pd.isna(y):
            raise ValueError(f"Invalid NFL result coding for game_id: {g['game_id']}")

        point_diff = abs(g["home_score"] - g["away_score"])
        if g["home_score"] > g["away_score"]:
          winner_elo_diff = (Ra + home_adv) - Rb
        elif g["home_score"] < g["away_score"]:
          winner_elo_diff = Rb - (Ra + home_adv)
        else:
          winner_elo_diff = 0

        if not use_mov:
          mov_mult = 1
        elif g["home_score"] == g["away_score"]:
          mov_mult = 1
        else:
          mov_mult = np.log(point_diff + 1) * (2.2 / (0.001 * winner_elo_diff + 2.2)) / mov_scale

        delta = K * (y - p) * mov_mult

        ratings[hid] = Ra + delta
        ratings[aid] = Rb - delta

        elo_home_post[i] = ratings[hid]
        elo_away_post[i] = ratings[aid]

    games_out = games.copy()
    games_out["elo_home_pre"] = elo_home_pre
    games_out["elo_away_pre"] = elo_away_pre
    games_out["elo_home_post"] = elo_home_post
    games_out["elo_away_post"] = elo_away_post
    games_out["p_home"] = p_home
    games_out["p_away"] = 1 - p_home

    return {"games": games_out, "ratings": ratings}


def carry_over(ratings, carry=0.6):
    if len(ratings) == 0:
        return ratings

    mu = np.mean(list(ratings.values()))
    return {team: carry * rating + (1 - carry) * mu
            for team, rating in ratings.items()}

def logloss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def score_elo_params(all_games, seasons_tune, K, home_adv,
                     use_mov=True, mov_scale=1, use_home_adv=True,
                     scale=400, start_rating=1000, carry=0.6):

  ratings = {}
  ll_vec = []

  if not use_home_adv:
      home_adv = 0

  for s in seasons_tune:
      g_s = (all_games[all_games["season"] == s]
              .sort_values(["gameday", "game_id"])
              .reset_index(drop=True))

      res = elo_season(games=g_s, ratings=ratings, K=K, scale=scale,
                       start_rating=start_rating, use_mov=use_mov,
                       mov_scale=mov_scale, use_home_adv=use_home_adv,
                       home_adv=home_adv)

      games_res = res["games"]
      non_ties = games_res["home_score"] != games_res["away_score"]

      p = games_res.loc[non_ties, "p_home"].to_numpy()
      y = (games_res.loc[non_ties, "home_score"] >
            games_res.loc[non_ties, "away_score"]).astype(float).to_numpy()

      ll_vec.extend(logloss(p, y))
      ratings = carry_over(res["ratings"], carry=carry)

  return np.nanmean(ll_vec)


def tune_elo_params(all_games, seasons_tune, K_grid, home_adv_grid,
                    use_mov=True, mov_scale=1, use_home_adv=True,
                    scale=400, start_rating=1000, carry=0.6,
                    use_carry_grid=True,
                    carryover_grid=(0.35, 0.50, 0.65, 0.80)):

    if not use_home_adv:
        home_adv_grid = [0]

    carry_grid = carryover_grid if use_carry_grid else [carry]
    results = []

    for K, home_adv, carry_val in product(K_grid, home_adv_grid, carry_grid):
        ll = score_elo_params(
            all_games=all_games, seasons_tune=seasons_tune,
            K=K, home_adv=home_adv,
            use_mov=use_mov, mov_scale=mov_scale,
            use_home_adv=use_home_adv, scale=scale,
            start_rating=start_rating, carry=carry_val
        )

        results.append({"K": K, "home_adv": home_adv, "carry": carry_val,
                        "logloss": ll})

    results = pd.DataFrame(results).sort_values("logloss").reset_index(drop=True)

    best = results.iloc[0]

    return {"K_best": best["K"], "home_adv_best": best["home_adv"],
            "carry_best": best["carry"], "logloss": best["logloss"],
            "results": results}


def run_expanding_k_elo(all_games, first_train_year=2012, first_eval_year=2019,
                        K_grid=None, home_adv_grid=None, scale=400,
                        start_rating=1000, carry=0.6, use_carry_grid=True,
                        carryover_grid=(0.35, 0.50, 0.65, 0.80),
                        param_tune_window=None, use_mov=True, use_home_adv=True,
                        projection_season=None, parameter_file=None):

    if K_grid is None:
        K_grid = compress_exp_grid(1, 50, n=10, curve=3)
    if home_adv_grid is None:
        home_adv_grid = compress_exp_grid(-4, 40, n=10, curve=2)
    if not use_home_adv:
        home_adv_grid = [0]

    K_grid = np.atleast_1d(K_grid)
    home_adv_grid = np.atleast_1d(home_adv_grid)
    fixed_param_mode = len(K_grid) == 1 and len(home_adv_grid) == 1 and not use_carry_grid

    seasons = sorted(all_games["season"].unique())
    eval_years = [s for s in seasons if s >= first_eval_year]

    # An upcoming season can have a schedule but no completed games yet.
    if projection_season is not None and projection_season not in eval_years:
        eval_years = sorted(set(eval_years + [projection_season]))
    if not eval_years:
        raise ValueError("No evaluation seasons available.")

    saved_params = pd.DataFrame()
    if parameter_file is not None:
        parameter_file = Path(parameter_file)
        if parameter_file.exists():
            saved_params = pd.read_csv(parameter_file, float_precision="round_trip")
            if saved_params["eval_year"].duplicated().any():
                raise ValueError("elo_parameters.csv has duplicate season rows.")
            saved_params = saved_params.set_index("eval_year")

    all_out = []
    param_out = []

    for yr in eval_years:
        train_years = [s for s in seasons if first_train_year <= s < yr]
        if len(train_years) == 0:
            raise ValueError(f"No training seasons available before eval year: {yr}")

        param_tune_years = (
            train_years if param_tune_window is None
            else train_years[-param_tune_window:]
        )

        train_games = all_games[all_games["season"].isin(train_years)]
        mov_scale = calc_mov_scale(train_games) if use_mov else np.nan

        if yr in saved_params.index:
            print(f"Using saved parameters for NFL season {yr}.", flush=True)
            saved = saved_params.loc[yr]
            if saved["train_end"] >= yr:
                raise ValueError(f"Saved parameters for {yr} include non-prior training seasons.")
            K_best = saved["K_best"]
            home_adv_best = saved["home_adv_best"]
            carry_best = saved["carry_best"]
            mov_scale = saved["mov_scale"]
            best_logloss = saved["logloss"]
        elif fixed_param_mode:
            K_best = K_grid[0]
            home_adv_best = home_adv_grid[0] if use_home_adv else 0
            carry_best = carry
            best_logloss = np.nan
        else:
            print(f"No saved parameters for {yr}; tuning on prior seasons...", flush=True)
            param_fit = tune_elo_params(
                all_games=all_games, seasons_tune=param_tune_years,
                K_grid=K_grid, home_adv_grid=home_adv_grid,
                use_mov=use_mov, mov_scale=mov_scale,
                use_home_adv=use_home_adv, scale=scale,
                start_rating=start_rating, carry=carry,
                use_carry_grid=use_carry_grid,
                carryover_grid=carryover_grid
            )

            K_best = param_fit["K_best"]
            home_adv_best = param_fit["home_adv_best"] if use_home_adv else 0
            carry_best = param_fit["carry_best"] if use_carry_grid else carry
            best_logloss = param_fit["logloss"]

        ratings = {}

        for s in train_years:
            season_games = (all_games[all_games["season"] == s]
                            .sort_values(["gameday", "gametime", "game_id"])
                            .reset_index(drop=True))

            res = elo_season(
                games=season_games, ratings=ratings, K=K_best,
                scale=scale, start_rating=start_rating,
                use_mov=use_mov, mov_scale=mov_scale,
                use_home_adv=use_home_adv, home_adv=home_adv_best
            )

            ratings = carry_over(res["ratings"], carry=carry_best)

        eval_games = (all_games[all_games["season"] == yr]
                      .sort_values(["gameday", "gametime", "game_id"])
                      .reset_index(drop=True))

        res_eval = elo_season(
            games=eval_games, ratings=ratings, K=K_best,
            scale=scale, start_rating=start_rating,
            use_mov=use_mov, mov_scale=mov_scale,
            use_home_adv=use_home_adv, home_adv=home_adv_best
        )

        games_out = res_eval["games"].copy()
        games_out["K_used"] = K_best
        games_out["home_adv_used"] = home_adv_best
        games_out["carry_used"] = carry_best
        games_out["mov_scale_used"] = mov_scale
        games_out["use_mov"] = use_mov
        games_out["use_home_adv"] = use_home_adv
        games_out["train_start"] = min(train_years)
        games_out["train_end"] = max(train_years)
        games_out["param_tune_start"] = min(param_tune_years)
        games_out["param_tune_end"] = max(param_tune_years)
        games_out["param_tune_n"] = len(param_tune_years)
        games_out["param_tune_window"] = param_tune_window

        all_out.append(games_out)

        param_out.append({
            "eval_year": yr,
            "train_start": min(train_years),
            "train_end": max(train_years),
            "param_tune_start": min(param_tune_years),
            "param_tune_end": max(param_tune_years),
            "param_tune_n": len(param_tune_years),
            "param_tune_window": param_tune_window,
            "K_best": K_best,
            "home_adv_best": home_adv_best,
            "carry_best": carry_best,
            "mov_scale": mov_scale,
            "use_mov": use_mov,
            "use_home_adv": use_home_adv,
            "use_carry_grid": use_carry_grid,
            "logloss": best_logloss
        })

    if parameter_file is not None:
        # Retain other saved seasons and add any selections made during this run.
        parameters = pd.concat([saved_params.reset_index(), pd.DataFrame(param_out)],
                               ignore_index=True) if not saved_params.empty else pd.DataFrame(param_out)
        parameters = parameters.drop_duplicates("eval_year", keep="last").sort_values("eval_year")
        parameters.to_csv(parameter_file, index=False)

    return {
        "games": pd.concat(all_out, ignore_index=True),
        "K_by_year": pd.DataFrame(param_out),
        "ratings": res_eval["ratings"],
        "current_season": eval_years[-1]
    }


def main():
    # A new download each run; don't reuse nflreadpy's previously cached data.
    nfl.clear_cache()
    games = nfl.load_schedules(seasons=True).to_pandas()
    season_games = games[games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    if season_games.empty:
        raise ValueError("The NFL source returned no regular-season/playoff schedule.")
    if season_games["game_id"].duplicated().any():
        raise ValueError("Duplicate game IDs in the NFL schedule; Elo was not updated.")
    if (season_games["home_score"].notna() != season_games["away_score"].notna()).any():
        raise ValueError("A game has only one score populated; retry after the source updates.")

    # Same completed-game definition as the notebook: both scores present.
    elo_df = season_games[
        season_games["home_score"].notna() & season_games["away_score"].notna()
    ].copy()
    current_season = int(season_games["season"].max())
    # NFL season comes from the schedule, not the calendar year (January playoffs).
    output_dir = Path(__file__).resolve().parent
    nfl_elo = run_expanding_k_elo(
        all_games=elo_df, first_train_year=2003, first_eval_year=2008,
        K_grid=compress_exp_grid(35, 60, n=7, curve=1),
        home_adv_grid=compress_exp_grid(20, 40, n=7, curve=1),
        scale=400, start_rating=1000, use_carry_grid=True,
        carryover_grid=[0.5, 0.65, 0.7],
        param_tune_window=5, use_mov=True, use_home_adv=True,
        projection_season=current_season,
        parameter_file=output_dir / "elo_parameters.csv"
    )

    historical_elo = nfl_elo["games"]
    params = nfl_elo["K_by_year"].set_index("eval_year").loc[current_season]
    ratings = nfl_elo["ratings"]
    current_teams = pd.unique(games.loc[
        games["season"] == current_season, ["home_team", "away_team"]
    ].values.ravel())
    # A newly appearing team receives the model's existing start_rating of 1000.
    ratings = {team: ratings.get(team, 1000) for team in current_teams}
    current_ratings = pd.Series(ratings, name="elo").rename_axis("team").reset_index()
    current_ratings = current_ratings.sort_values("elo", ascending=False).reset_index(drop=True)
    current_ratings.insert(0, "rank", range(1, len(current_ratings) + 1))

    as_of_date = pd.Timestamp.now(tz="America/Chicago").date()
    rank_map = current_ratings.set_index("team")["rank"]
    future_projections = season_games[
        (season_games["season"] == current_season) &
        season_games["home_score"].isna() & season_games["away_score"].isna()
    ].copy()
    future_projections["home_elo"] = future_projections["home_team"].map(ratings)
    future_projections["away_elo"] = future_projections["away_team"].map(ratings)
    future_projections["home_rank"] = future_projections["home_team"].map(rank_map)
    future_projections["away_rank"] = future_projections["away_team"].map(rank_map)
    # Freeze today's ratings for every remaining game; do not simulate results.
    future_projections["p_home"] = 1 / (1 + 10 ** (
        (future_projections["away_elo"] - future_projections["home_elo"] -
         params["home_adv_best"]) / 400
    ))
    future_projections["p_away"] = 1 - future_projections["p_home"]
    future_projections["as_of_date"] = as_of_date
    future_projections = future_projections[
        ["as_of_date", "week", "game_id", "gameday", "gametime", "away_team", "home_team",
         "away_rank", "home_rank", "away_elo", "home_elo", "p_away", "p_home"]
    ].sort_values(["gameday", "gametime"]).reset_index(drop=True)
    daily_projections = future_projections[
        pd.to_datetime(future_projections["gameday"]).dt.date == as_of_date
    ].reset_index(drop=True)
    # --- Preserve latest pregame prediction for each game ---
    history_path = output_dir / "historical_predictions.csv"
    
    run_timestamp = pd.Timestamp.now(tz="America/New_York")
    kickoff_timestamps = pd.to_datetime(
        future_projections["gameday"].astype(str) + " " +
        future_projections["gametime"].astype(str),
        errors="coerce"
    ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    # Missing/invalid kickoffs are ineligible too; scores do not control this cutoff.
    current_predictions = future_projections.loc[
        kickoff_timestamps > run_timestamp
    ].copy()
    current_predictions["prediction_timestamp"] = run_timestamp.isoformat()
    
    if history_path.exists():
        prediction_history = pd.read_csv(
            history_path,
            float_precision="round_trip"
        )
    
        prediction_history = prediction_history[
            ~prediction_history["game_id"].isin(current_predictions["game_id"])
        ]
    
        prediction_history = pd.concat(
            [prediction_history, current_predictions],
            ignore_index=True
        )
    
    else:
        prediction_history = current_predictions

    prediction_history = prediction_history.drop_duplicates("game_id", keep="last")

    # Select the full NFL week from schedule results, not remaining predictions.
    weekly_schedule = season_games[
        season_games["season"] == current_season
    ].copy()
    # Require published result fields as well as scores; live scores alone are
    # insufficient. result is the home margin and total is the combined score.
    completed = (
        weekly_schedule["result"].notna() & weekly_schedule["total"].notna() &
        weekly_schedule["home_score"].notna() & weekly_schedule["away_score"].notna()
    )
    weekly_schedule["outcome"] = "TBD"
    weekly_schedule.loc[completed & (weekly_schedule["result"] > 0), "outcome"] = (
        weekly_schedule["home_team"]
    )
    weekly_schedule.loc[completed & (weekly_schedule["result"] < 0), "outcome"] = (
        weekly_schedule["away_team"]
    )
    weekly_schedule.loc[completed & (weekly_schedule["result"] == 0), "outcome"] = "TIE"
    unfinished_weeks = weekly_schedule.loc[~completed, "week"]
    # After the season ends, retain its last week until a new season is available.
    active_week = (unfinished_weeks.min() if not unfinished_weeks.empty
                   else weekly_schedule["week"].max())
    schedule_columns = ["week", "game_id", "gameday", "gametime", "away_team", "home_team"]
    prediction_columns = [column for column in future_projections.columns
                          if column not in schedule_columns]
    # The archive above already contains current-run upcoming predictions and
    # frozen pregame predictions. A missing archive row stays blank, never refit.
    weekly_projections = weekly_schedule.loc[
        weekly_schedule["week"] == active_week, schedule_columns + ["outcome"]
    ].merge(
        prediction_history[["game_id"] + prediction_columns],
        on="game_id", how="left", validate="one_to_one"
    )
    weekly_projections = weekly_projections[
        list(future_projections.columns) + ["outcome"]
    ].sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)

    # Game-level inputs for interactive performance metrics on the website.
    performance_columns = [
        "season", "week", "game_id", "gameday", "away_team", "home_team",
        "p_home", "actual_home_win", "evaluation_type"
    ]
    historical_performance = historical_elo.loc[
        (historical_elo["season"] < current_season) &
        historical_elo["home_score"].notna() & historical_elo["away_score"].notna() &
        (historical_elo["home_score"] != historical_elo["away_score"]),
        ["season", "week", "game_id", "gameday", "away_team", "home_team", "p_home",
         "home_score", "away_score"]
    ].copy()
    historical_performance["actual_home_win"] = (
        historical_performance["home_score"] > historical_performance["away_score"]
    ).astype(int)
    historical_performance["evaluation_type"] = "historical"
    historical_performance = historical_performance[performance_columns]

    deployed_results = weekly_schedule.loc[
        completed & (weekly_schedule["home_score"] != weekly_schedule["away_score"]),
        ["season", "week", "game_id", "gameday", "away_team", "home_team",
         "home_score", "away_score"]
    ].copy()
    deployed_performance = deployed_results.merge(
        prediction_history[["game_id", "p_home"]],
        on="game_id", how="inner", validate="one_to_one"
    )
    deployed_performance["actual_home_win"] = (
        deployed_performance["home_score"] > deployed_performance["away_score"]
    ).astype(int)
    deployed_performance["evaluation_type"] = "deployed"
    deployed_performance = deployed_performance[performance_columns]

    performance_games = pd.concat(
        [historical_performance, deployed_performance], ignore_index=True
    ).sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    if performance_games.duplicated(["evaluation_type", "game_id"]).any():
        raise ValueError("Duplicate evaluation_type + game_id in performance output.")
    if performance_games[["p_home", "actual_home_win"]].isna().any().any():
        raise ValueError("Missing prediction or result in performance output.")
    if not performance_games["p_home"].between(0, 1, inclusive="both").all():
        raise ValueError("Invalid probability in performance output.")
    if not performance_games["actual_home_win"].isin([0, 1]).all():
        raise ValueError("Invalid binary result in performance output.")

    # Retain the final notebook's performance reporting without adding more files.
    eval_games = historical_elo.query("home_score != away_score").copy()
    eval_games["actual"] = (eval_games["home_score"] > eval_games["away_score"]).astype(int)
    eval_games["pred"] = (eval_games["p_home"] >= 0.5).astype(int)
    eval_games["correct"] = eval_games["pred"] == eval_games["actual"]
    eval_games["ll"] = -(eval_games["actual"] * np.log(eval_games["p_home"]) +
                         (1 - eval_games["actual"]) * np.log1p(-eval_games["p_home"]))
    print(nfl_elo["K_by_year"].to_string(index=False))
    print(eval_games.groupby("season").agg(
        games=("game_id", "size"), accuracy=("correct", "mean"),
        logloss=("ll", "mean")).to_string())
    print("Accuracy:", accuracy_score(eval_games["actual"], eval_games["pred"]))
    print("Log Loss:", log_loss(eval_games["actual"], eval_games["p_home"], labels=[0, 1]))

    outputs = {"current_elo_rankings.csv": current_ratings,
               "daily_projections.csv": daily_projections,
               "weekly_projections.csv": weekly_projections,
               "future_projections.csv": future_projections,
               "historical_predictions.csv": prediction_history,
               "performance_games.csv": performance_games,
               "nfl_elo_history.csv": historical_elo,}
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
        print(f"Saved {filename}: {len(frame):,} rows")
    print(f"Season: {current_season}; as of: {as_of_date}; output folder: {output_dir}")
    return nfl_elo, outputs


# Run the script when called from the command line; importing it only loads functions.
if __name__ == "__main__":
    main()
