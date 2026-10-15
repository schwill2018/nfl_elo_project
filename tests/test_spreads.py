"""Spread fitting, seasonal reuse, and pregame archive integration checks."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import nfl_elo as elo


def completed_games(last_season=2026):
    games = []
    for season in range(2018, last_season + 1):
        for i, p_home in enumerate([0.2, 0.4, 0.5, 0.6, 0.8]):
            margin = 10 * np.log(p_home / (1 - p_home))
            games.append({
                "season": season, "game_type": "REG", "week": 6,
                "game_id": f"{season}_06_B_A_{i}",
                "gameday": f"{season}-10-10", "gametime": "13:00",
                "home_team": "A", "away_team": "B", "location": "Home",
                "home_score": 20 + margin, "away_score": 20,
                "overtime": 0, "result": margin, "total": 40 + margin,
                "p_home": p_home, "p_away": 1 - p_home,
            })
    return pd.DataFrame(games)


class SpreadParameterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.assertTrue(Path(self.directory.name).resolve().is_relative_to(Path(__file__).resolve().parent))
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "spread_parameters.csv"
        self.games = completed_games()

    def fit(self, games=None, season=2026, window=5):
        with contextlib.redirect_stdout(io.StringIO()):
            return elo.fit_spread_params(
                self.games if games is None else games, season, window, self.path
            )

    def test_five_prior_seasons_and_independent_rmse_solution(self):
        parameters = self.fit()
        current = parameters.set_index("eval_year").loc[2026]
        self.assertEqual(current["train_start"], 2021)
        self.assertEqual(current["train_end"], 2025)
        self.assertEqual(current["train_games"], 25)
        self.assertEqual(current["fit_loss"], "RMSE")
        self.assertAlmostEqual(current["spread_slope"], 10, places=4)

        self.path.unlink()
        changed = self.games.copy()
        changed.loc[changed["season"] == 2026, "home_score"] += 100
        refitted = self.fit(changed).set_index("eval_year")
        self.assertEqual(current["spread_slope"], refitted.loc[2026, "spread_slope"])

    def test_cached_run_never_calls_optimizer_or_rewrites_file(self):
        initial = self.fit()
        before = self.path.read_bytes()
        with patch.object(elo, "minimize_scalar", side_effect=AssertionError("Unexpected refit")):
            reused = self.fit()
        self.assertEqual(before, self.path.read_bytes())
        pd.testing.assert_frame_equal(initial, reused, check_exact=True)

    def test_new_season_appends_once_and_preserves_old_fits(self):
        initial = self.fit()
        with patch.object(elo, "minimize_scalar", wraps=elo.minimize_scalar) as optimizer:
            updated = self.fit(completed_games(2027), season=2027)
        self.assertEqual(optimizer.call_count, 1)
        pd.testing.assert_frame_equal(
            initial, updated[updated["eval_year"] <= 2026].reset_index(drop=True),
            check_exact=True
        )
        new = updated.set_index("eval_year").loc[2027]
        self.assertEqual((new["train_start"], new["train_end"]), (2022, 2026))

    def test_new_season_can_fit_before_any_of_its_games_are_played(self):
        current = self.fit(completed_games(2025)).set_index("eval_year").loc[2026]
        self.assertEqual(current["train_end"], 2025)
        self.assertAlmostEqual(current["spread_slope"], 10, places=4)

    def test_incompatible_or_leaking_cache_is_rejected_without_refit(self):
        initial = self.fit()
        cases = [
            ("train_end", 2026), ("training_window", 4),
            ("fit_loss", "MAE"), ("spread_slope", 21),
            ("spread_slope", np.inf), ("train_rmse", -1),
        ]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                invalid = initial.copy()
                invalid.loc[invalid["eval_year"] == 2026, column] = value
                invalid.to_csv(self.path, index=False)
                before = self.path.read_bytes()
                with patch.object(elo, "minimize_scalar", side_effect=AssertionError("Unexpected refit")):
                    with self.assertRaises(ValueError):
                        self.fit()
                self.assertEqual(before, self.path.read_bytes())

        pd.concat([initial, initial.tail(1)]).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.fit()

    def test_missing_training_season_is_rejected(self):
        incomplete = self.games[self.games["season"] != 2022]
        with self.assertRaisesRegex(ValueError, "Missing prior seasons"):
            self.fit(incomplete)
        self.assertFalse(self.path.exists())


class SpreadOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.assertTrue(Path(self.directory.name).resolve().is_relative_to(Path(__file__).resolve().parent))
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.history = completed_games()
        upcoming = []
        for game_id, day, location in [
            ("2026_06_B_A_upcoming", "2026-10-18", "Home"),
            ("2026_06_B_A_neutral", "2026-10-19", "Neutral"),
            ("2026_06_B_A_started", "2026-10-15", "Home"),
        ]:
            upcoming.append({
                "season": 2026, "game_type": "REG", "week": 6,
                "game_id": game_id, "gameday": day,
                "gametime": "10:00" if "started" in game_id else "13:00",
                "home_team": "A", "away_team": "B", "location": location,
                "home_score": np.nan, "away_score": np.nan,
                "overtime": np.nan, "result": np.nan, "total": np.nan,
            })
        self.schedule = pd.concat([self.history, pd.DataFrame(upcoming)], ignore_index=True)
        self.fixed_now = pd.Timestamp("2026-10-15 12:00", tz="America/New_York")
        self.archive = pd.DataFrame([
            self.archive_row("2026_06_B_A_0", "2026-10-10", 0.3),
            self.archive_row("2026_06_B_A_1", "2026-10-10", 0.4, 3.0),
            self.archive_row("2026_06_B_A_started", "2026-10-15", 0.25, 4.5),
            self.archive_row("2026_06_B_A_upcoming", "2026-10-18", 0.35, 2.0),
        ])
        self.archive.to_csv(self.folder / "historical_predictions.csv", index=False)

    def archive_row(self, game_id, day, p_home, spread=np.nan):
        return {
            "as_of_date": "2026-10-09", "week": 6, "game_id": game_id,
            "gameday": day, "gametime": "10:00" if "started" in game_id else "13:00",
            "away_team": "B", "home_team": "A", "away_rank": 2, "home_rank": 1,
            "away_elo": 1000.0, "home_elo": 1050.0,
            "p_away": 1 - p_home, "p_home": p_home,
            "model_home_spread": spread,
            "prediction_timestamp": "2026-10-09T12:00:00-04:00", "extra_saved_field": "keep",
        }

    def run_main(self):
        clock = Mock(wraps=pd.Timestamp)
        clock.now.side_effect = lambda tz=None: (
            self.fixed_now.tz_convert(tz) if tz else self.fixed_now.tz_localize(None)
        )
        model_result = {
            "games": self.history.copy(),
            "K_by_year": pd.DataFrame([{"eval_year": 2026, "home_adv_best": 29.0}]),
            "ratings": {"A": 1050.0, "B": 1000.0},
        }
        with patch.object(elo, "__file__", str(self.folder / "nfl_elo.py")), \
                patch.object(elo.nfl, "clear_cache"), \
                patch.object(elo.nfl, "load_schedules", return_value=SimpleNamespace(
                    to_pandas=lambda: self.schedule.copy())), \
                patch.object(elo, "run_expanding_k_elo", return_value=model_result), \
                patch.object(elo.pd, "Timestamp", clock), \
                contextlib.redirect_stdout(io.StringIO()):
            return elo.main()[1]

    def test_spread_order_neutral_venue_and_frozen_records(self):
        outputs = self.run_main()
        weekly = outputs["weekly_projections.csv"]
        self.assertEqual(list(weekly.columns[-4:]), ["p_away", "p_home", "model_home_spread", "outcome"])

        future = outputs["future_projections.csv"].set_index("game_id")
        slope = pd.read_csv(self.folder / "spread_parameters.csv").set_index("eval_year").loc[2026, "spread_slope"]
        for game_id, advantage in [("2026_06_B_A_upcoming", 29), ("2026_06_B_A_neutral", 0)]:
            p_home = 1 / (1 + 10 ** ((1000 - 1050 - advantage) / 400))
            self.assertAlmostEqual(future.loc[game_id, "p_home"], p_home)
            margin = slope * np.log(p_home / (1 - p_home))
            self.assertEqual(future.loc[game_id, "model_home_spread"], -np.round(margin * 2) / 2)

        archived = outputs["historical_predictions.csv"].set_index("game_id")
        frozen_ids = ["2026_06_B_A_0", "2026_06_B_A_1", "2026_06_B_A_started"]
        original = self.archive.set_index("game_id")
        pd.testing.assert_frame_equal(
            original.loc[frozen_ids].sort_index(axis=1),
            archived.loc[frozen_ids, original.columns].sort_index(axis=1),
            check_dtype=False
        )
        self.assertTrue(pd.isna(archived.loc["2026_06_B_A_0", "model_home_spread"]))
        self.assertNotEqual(archived.loc["2026_06_B_A_upcoming", "p_home"], 0.35)
        missing_id = "2026_06_B_A_2"
        self.assertTrue(pd.isna(weekly.set_index("game_id").loc[missing_id, "model_home_spread"]))

        with patch.object(elo, "minimize_scalar", side_effect=AssertionError("Unexpected daily refit")):
            self.run_main()

    def test_legacy_archive_and_no_remaining_games(self):
        self.schedule = self.history.copy()
        legacy = self.archive.iloc[:1].drop(columns="model_home_spread")
        legacy.to_csv(self.folder / "historical_predictions.csv", index=False)
        outputs = self.run_main()
        self.assertTrue(outputs["future_projections.csv"].empty)
        self.assertTrue(outputs["daily_projections.csv"].empty)
        archive = outputs["historical_predictions.csv"]
        self.assertEqual(len(archive), 1)
        self.assertTrue(archive["model_home_spread"].isna().all())
        self.assertTrue(outputs["weekly_projections.csv"]["model_home_spread"].isna().all())


if __name__ == "__main__":
    unittest.main()
