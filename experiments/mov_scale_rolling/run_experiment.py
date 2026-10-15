"""Compare expanding and five-season MOV normalization without changing production.

Uses the saved evaluation-game snapshot and public nflverse schedules for the
2003-2007 warm-up. The parameter grid and Elo formulas come from nfl_elo.py.
Run with Python plus numpy/pandas; outputs stay beside this experiment.
"""
import ast
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "nfl_elo.py"
PRODUCTION_FILES = [SOURCE, *ROOT.glob("*.csv")]


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PRODUCTION_FILES}


before_hashes = hashes()
namespace = dict(np=np, pd=pd, Path=Path, product=product)
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
definitions = ast.Module(
    body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
    type_ignores=[],
)
exec(compile(definitions, str(SOURCE), "exec"), namespace)

history = pd.read_csv(ROOT / "nfl_elo_history.csv", float_precision="round_trip")
saved = pd.read_csv(ROOT / "elo_parameters.csv", float_precision="round_trip").set_index("eval_year")
raw = pd.read_csv(OUT / "source_games.csv", low_memory=False)
columns = ["game_id", "season", "game_type", "week", "gameday", "gametime",
           "home_team", "away_team", "home_score", "away_score", "overtime", "location"]
warmup = raw.loc[
    raw.season.between(2003, 2007)
    & raw.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])
    & raw.home_score.notna() & raw.away_score.notna(), columns
]
games = pd.concat([warmup, history[columns]], ignore_index=True)
assert not games.game_id.duplicated().any()
assert games.home_score.notna().all() and games.away_score.notna().all()
games.to_csv(OUT / "input_games.csv", index=False)
years = sorted(games.season.unique())
eval_years = [s for s in years if s >= 2008]
current_year = max(eval_years)
teams = sorted(set(games.home_team) | set(games.away_team))
team_index = {name: i for i, name in enumerate(teams)}
prepared = {}

for year in years:
    for order in ("main", "tuning"):
        sort = ["gameday", "gametime", "game_id"] if order == "main" else ["gameday", "game_id"]
        frame = games[games.season.eq(year)].sort_values(sort).reset_index(drop=True)
        hs, aws = frame.home_score.to_numpy(), frame.away_score.to_numpy()
        prepared[year, order] = dict(
            frame=frame,
            home=frame.home_team.map(team_index).to_numpy(),
            away=frame.away_team.map(team_index).to_numpy(),
            neutral=frame.location.eq("Neutral").to_numpy(),
            home_win=hs > aws, tie=hs == aws,
            y=namespace["nfl_home_result"](hs, aws, frame.overtime.to_numpy()),
            log_margin=np.log(np.abs(hs - aws) + 1),
        )

grid = np.array(list(product(
    namespace["compress_exp_grid"](40, 65, n=8, curve=1),
    namespace["compress_exp_grid"](25, 65, n=8, curve=1),
    [0.5, 0.575, 0.65],
)))


def simulate(seasons, params, mov_scale, initial=None, collect=False, order="main"):
    """Evaluate all candidate settings in parallel, with the source Elo equations."""
    params = np.atleast_2d(params)
    k, advantage, carry = params.T
    start = np.full(len(teams), np.nan) if initial is None else initial
    ratings = np.broadcast_to(start, (len(params), len(teams))).copy()
    losses = []
    outputs = []
    for season in seasons:
        data = prepared[season, order]
        active = ~np.isnan(ratings)
        mean = np.nansum(ratings, axis=1) / np.maximum(active.sum(axis=1), 1)
        ratings = np.where(active, carry[:, None] * ratings + (1-carry[:, None]) * mean[:, None], np.nan)
        trace = np.zeros((len(data["frame"]), 5)) if collect else None
        for i, (hid, aid) in enumerate(zip(data["home"], data["away"])):
            ra = np.nan_to_num(ratings[:, hid], nan=1000.0)
            rb = np.nan_to_num(ratings[:, aid], nan=1000.0)
            ha = np.zeros(len(params)) if data["neutral"][i] else advantage
            p = 1 / (1 + 10 ** ((rb - ra - ha) / 400))
            if data["tie"][i]:
                multiplier = 1
            else:
                winner_diff = ra + ha - rb if data["home_win"][i] else rb - ra - ha
                multiplier = data["log_margin"][i] * (2.2 / (0.001 * winner_diff + 2.2)) / mov_scale
                clipped = np.clip(p, 1e-12, 1-1e-12)
                y = float(data["home_win"][i])
                losses.append(-(y * np.log(clipped) + (1-y) * np.log1p(-clipped)))
            delta = k * (data["y"][i] - p) * multiplier
            ratings[:, hid] = ra + delta
            ratings[:, aid] = rb - delta
            if collect:
                assert len(params) == 1
                trace[i] = [ra[0], rb[0], ratings[0, hid], ratings[0, aid], p[0]]
        if collect:
            output = data["frame"].copy()
            output[["elo_home_pre", "elo_away_pre", "elo_home_post", "elo_away_post", "p_home"]] = trace
            outputs.append(output)
    loss = np.mean(np.array(losses), axis=0)
    output = pd.concat(outputs, ignore_index=True) if collect else None
    return ratings, loss, output


def scale_for(year, window):
    prior = [s for s in years if s < year]
    if window is not None:
        prior = prior[-window:]
    return namespace["calc_mov_scale"](games[games.season.isin(prior)])


def run_variant(name, window, retune):
    ratings = None
    snapshots = {}
    outputs, selections = [], []
    for year in eval_years:
        scale = scale_for(year, window)
        if window is None:
            assert abs(scale - saved.loc[year, "mov_scale"]) < 1e-12
            # Match production's saved scale exactly, including float serialization.
            scale = saved.loc[year, "mov_scale"]
        tuning_years = [s for s in years if s < year][-5:]
        if retune:
            _, scores, _ = simulate(tuning_years, grid, scale,
                                    initial=snapshots.get(tuning_years[0]), order="tuning")
            selected_index = pd.DataFrame({"loss": scores}).sort_values("loss").index[0]
            selected = grid[selected_index]
            tuning_loss = float(scores[selected_index])
        else:
            selected = saved.loc[year, ["K_best", "home_adv_best", "carry_best"]].to_numpy(dtype=float)
            tuning_loss = np.nan
        if year == eval_years[0]:
            for season in [s for s in years if s < year]:
                snapshots[season] = None if ratings is None else ratings.copy()
                result, _, _ = simulate([season], selected, scale, ratings)
                ratings = result[0]
        snapshots[year] = ratings.copy()
        result, _, output = simulate([year], selected, scale, ratings, collect=True)
        ratings = result[0]
        output["variant"] = name
        outputs.append(output)
        selections.append(dict(variant=name, eval_year=int(year), K=float(selected[0]),
                               home_adv=float(selected[1]), carry=float(selected[2]),
                               mov_scale=float(scale), tuning_logloss=tuning_loss,
                               mov_start=int(min(years) if window is None else tuning_years[0]),
                               mov_end=int(year-1)))
        print(f"{name}: {year}, K={selected[0]:g}, HFA={selected[1]:g}, carry={selected[2]:g}", flush=True)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(selections), snapshots


baseline, baseline_params, baseline_snapshots = run_variant("saved_expanding", None, False)
verify_columns = ["elo_home_pre", "elo_away_pre", "elo_home_post", "elo_away_post", "p_home"]
match = baseline.set_index("game_id")[verify_columns].sort_index()
expected = history.set_index("game_id")[verify_columns].sort_index()
max_errors = (match - expected).abs().max().to_dict()
assert np.allclose(match, expected, rtol=0, atol=1e-9), max_errors

# Check the accelerated candidate scorer against the actual source implementation,
# using both the initial tuning window and the latest one with inherited ratings.
score_checks = []
for year in (2008, current_year):
    tune_years = [s for s in years if s < year][-5:]
    start = baseline_snapshots.get(tune_years[0])
    initial = {} if start is None else {t: float(start[i]) for t, i in team_index.items() if not np.isnan(start[i])}
    for window in (None, 5):
        scale = scale_for(year, window)
        for index in (0, len(grid)//2, len(grid)-1):
            k, ha, carry = grid[index]
            _, accelerated, _ = simulate(tune_years, grid[index], scale, start, order="tuning")
            reference = namespace["score_elo_params"](
                games, tune_years, K=k, home_adv=ha, carry=carry,
                mov_scale=scale, initial_ratings=initial,
            )
            error = abs(float(accelerated[0]) - reference)
            assert error < 1e-12, (year, window, index, error)
            score_checks.append(dict(year=int(year), window=window, candidate=index, absolute_error=error))
print("PASS: baseline matches saved production output; candidate scorer matches source.", flush=True)

variant_outputs = [baseline]
variant_params = [baseline_params]
for name, window, retune in [
    ("saved_rolling5", 5, False),
    ("retuned_expanding", None, True),
    ("retuned_rolling5", 5, True),
]:
    output, params, _ = run_variant(name, window, retune)
    variant_outputs.append(output)
    variant_params.append(params)

predictions = pd.concat(variant_outputs, ignore_index=True)
parameters = pd.concat(variant_params, ignore_index=True)
non_ties = predictions.home_score.ne(predictions.away_score)
scored = predictions[non_ties].copy()
scored["actual"] = scored.home_score.gt(scored.away_score).astype(int)
scored["loss"] = -(scored.actual * np.log(scored.p_home) + (1-scored.actual) * np.log1p(-scored.p_home))
scored["correct"] = scored.p_home.ge(0.5).eq(scored.actual.astype(bool))
scored["brier"] = (scored.p_home-scored.actual)**2
periods = {
    f"complete_2008_{current_year-1}": (2008, current_year-1),
    f"recent_{current_year-5}_{current_year-1}": (current_year-5, current_year-1),
    f"partial_{current_year}": (current_year, current_year),
}
summaries = []
for period, (first, last) in periods.items():
    group = scored[scored.season.between(first, last)]
    for variant, frame in group.groupby("variant", sort=False):
        summaries.append(dict(period=period, variant=variant, games=len(frame),
                              log_loss=frame.loss.mean(), accuracy=frame.correct.mean(), brier=frame.brier.mean()))
summary = pd.DataFrame(summaries)
yearly = scored.groupby(["variant", "season"]).agg(
    games=("game_id", "size"), log_loss=("loss", "mean"),
    accuracy=("correct", "mean"), brier=("brier", "mean"),
).reset_index()

# Paired season-block resampling: descriptive uncertainty, not a fresh holdout.
comparisons = []
for candidate, control in [("saved_rolling5", "saved_expanding"),
                           ("retuned_rolling5", "retuned_expanding")]:
    for period, (first, last) in list(periods.items())[:2]:
        a = scored[scored.variant.eq(candidate) & scored.season.between(first, last)]
        b = scored[scored.variant.eq(control) & scored.season.between(first, last)]
        paired = a[["game_id", "season", "loss"]].merge(
            b[["game_id", "loss"]], on="game_id", suffixes=("_candidate", "_control"), validate="one_to_one")
        paired["difference"] = paired.loss_candidate - paired.loss_control
        blocks = paired.groupby("season").difference.agg(["sum", "count", "mean"])
        rng = np.random.default_rng(42)
        draws = rng.integers(0, len(blocks), size=(10000, len(blocks)))
        samples = blocks["sum"].to_numpy()[draws].sum(axis=1) / blocks["count"].to_numpy()[draws].sum(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        comparisons.append(dict(candidate=candidate, control=control, period=period,
                                delta_log_loss=paired.difference.mean(),
                                season_bootstrap_low=low, season_bootstrap_high=high,
                                better_seasons=int((blocks["mean"] < 0).sum()), seasons=len(blocks)))

proofs = []
for year, team in [(2009, "NYG"), (current_year, "BUF")]:
    starting = baseline_snapshots[year]
    league_mean = np.nanmean(starting)
    carry = saved.loc[year, "carry_best"]
    prior = starting[team_index[team]]
    frame = baseline[baseline.season.eq(year) & (baseline.home_team.eq(team) | baseline.away_team.eq(team))]
    first = frame.iloc[0]
    observed = first.elo_home_pre if first.home_team == team else first.elo_away_pre
    calculated = carry*prior + (1-carry)*league_mean
    assert abs(observed-calculated) < 1e-9
    proofs.append(dict(season=int(year), team=team, prior_final_elo=prior,
                       league_mean=league_mean, carry=carry, first_pregame_elo=observed,
                       calculated_from_prior_elo=calculated))

summary.to_csv(OUT / "summary.csv", index=False)
yearly.to_csv(OUT / "by_season.csv", index=False)
parameters.to_csv(OUT / "parameters.csv", index=False)
predictions.to_csv(OUT / "predictions.csv", index=False)
pd.DataFrame(comparisons).to_csv(OUT / "comparisons.csv", index=False)
pd.DataFrame(proofs).to_csv(OUT / "rating_carry_examples.csv", index=False)
assert hashes() == before_hashes, "Production files changed during the experiment"
manifest = dict(
    source_url="https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    evaluation_snapshot="nfl_elo_history.csv (not refreshed from the network)",
    latest_game_date=str(games.gameday.max()), games=len(games),
    tuning_window=5, candidates=len(grid),
    production_hashes_before_and_after=before_hashes,
    baseline_max_absolute_errors=max_errors,
    reference_score_checks=score_checks,
    completed_seasons=list(range(2008, current_year)), partial_season=int(current_year),
    caveat="Exploratory retrospective comparison; historical seasons have already informed model development.",
)
(OUT / "validation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(summary.to_string(index=False), flush=True)
print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
print("PASS: nfl_elo.py and all production CSVs are unchanged.", flush=True)
