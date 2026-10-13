"""Package 16.3: forecasting. Backtests never see the future (changing every target after an origin changes none of that
origin's forecasts), a gap moves the first forecast date and the training targets, covariates unknown at the moment are
kept out (read from the moment, not from the replica's answer), a declared one that moves with the target is shown to
confirm, text covariates are coded, every model is compared with the seasonal naive, intervals are scaled per series and
measured on a later origin, and the kind is refused by the engine until the engine runs it."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import forecast  # noqa: E402
from dclab_core.replicas import store_demand  # noqa: E402

ARGS = dict(series="series", time="date", target="units", horizon=7, known_ahead=["on_promotion"], season=7, origins=3)


def forecasts(result, origin):
    return {(m, r["series"]): r["pred"] for m, rows in result["per_origin"].items() for r in rows if r["origin"] == origin}


class BacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = store_demand.generate(0, stores=2, products=3, days=160).sales
        cls.result = forecast.backtest(cls.frame, **ARGS)

    def test_an_origin_never_sees_a_target_after_it(self):
        first = self.result["origins"][0]
        later = self.frame["date"] > first  # every target after the origin, inside its window too
        changed = self.frame.copy()
        changed.loc[later, "units"] = changed.loc[later, "units"] * 10 + 999
        again = forecast.backtest(changed, **ARGS)
        before, after = forecasts(self.result, first), forecasts(again, first)
        self.assertTrue({m for m, _ in before} >= {"naive", "seasonal_naive", "global_gbm", "per_series"})
        self.assertEqual(before, after)  # a model that read one day ahead would move here
        for origin, seen in self.result["target_seen_until"].items():
            self.assertLessEqual(seen, origin)
        self.assertEqual(self.result["future_seen"], [])

    def test_a_gap_moves_the_first_forecast_date_and_the_training_targets(self):
        gapped = forecast.backtest(self.frame, **{**ARGS, "gap": 2, "origins": 2})
        for rows in gapped["per_origin"].values():
            for r in rows:
                self.assertEqual(pd.Timestamp(r["dates"][0]), pd.Timestamp(r["origin"]) + pd.Timedelta(days=3))
        history = pd.DataFrame({"units": np.arange(40.0)}, index=pd.date_range("2025-01-01", periods=40))
        rows = forecast._training_rows(history, target="units", known=[], season=7, horizon=3, gap=2)
        first = rows.iloc[0]  # the first cut keeps 8 rows (0..7): its first target is 2 days after the last kept
        self.assertEqual((first["last"], first["units"]), (7.0, 10.0))
        self.assertEqual(first["lag_7"], 3.0)  # 7 days before day 10, counted through the gap

    def test_every_model_is_scored_against_the_seasonal_naive_with_measured_intervals(self):
        r = self.result
        self.assertIn("seasonal_naive", r["scores"])
        self.assertTrue({"naive", "moving_average", "global_gbm"} <= set(r["beats_seasonal_naive"]))
        self.assertTrue(r["beats_seasonal_naive"]["global_gbm"])
        self.assertEqual(sorted(r["intervals"]["by_step"]), list(range(1, 8)))
        measured = r["intervals"]["measured"]
        self.assertEqual(set(measured), {"global_gbm", "seasonal_naive"})  # both, on the last origin only
        self.assertEqual(measured["global_gbm"]["origin"], r["origins"][-1])
        self.assertTrue(0.0 <= measured["global_gbm"]["coverage"] <= 1.0)
        self.assertLess(r["scores"]["global_gbm"]["pinball"], r["scores"]["seasonal_naive"]["pinball"])

    def test_a_covariate_not_declared_is_kept_out_and_one_declared_by_mistake_is_shown_to_confirm(self):
        out = forecast.covariate_check(self.frame, series="series", time="date", target="units", known_ahead=["on_promotion"])
        self.assertIn("rain_mm", out["kept_out"])
        self.assertIn("on_promotion", out["kept"])
        self.assertTrue({"store", "product"} <= set(out["kept"]))  # the same on every row of a series
        undeclared = forecast.covariate_check(self.frame, series="series", time="date", target="units")
        self.assertIn("on_promotion", undeclared["kept_out"])  # nothing is known ahead until the person says so
        self.assertNotIn("outcome", undeclared["why"]["on_promotion"])
        mistaken = forecast.covariate_check(self.frame, series="series", time="date", target="units", known_ahead=["on_promotion", "rain_mm"])
        self.assertIn("rain_mm", mistaken["to_confirm"])
        self.assertIn("confirm", mistaken["why"]["rain_mm"])

    def test_a_text_covariate_known_ahead_is_coded_from_the_training_rows(self):
        frame = self.frame.copy()
        frame["holiday"] = np.where(pd.to_datetime(frame["date"]).dt.dayofweek == 6, "sun", "none")
        result = forecast.backtest(frame, **{**ARGS, "known_ahead": ["on_promotion", "holiday"], "origins": 2})
        self.assertIn("holiday", result["known_ahead"])
        self.assertIn("global_gbm", result["scores"])


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_kind_is_declared_never_assumed_from_a_pack(self):
        base = {"target": "y", "task": "regression", "prediction_moment": "Each Sunday for the week ahead"}
        self.assertEqual(self.kinds.resolve(base, "timeseries").key, "regression")  # a time-series pack alone keeps regression
        self.assertEqual(self.kinds.resolve({**base, "kind": "forecasting"}).key, "forecasting")

    def test_known_ahead_is_read_from_the_moment_not_from_the_replica(self):
        from dclab_rnd.kinds.forecasting import known_from_moment

        columns = ["store", "product", "on_promotion", "rain_mm"]
        moment = "Each Sunday evening, for the next 7 days; promotions are planned weeks ahead, the day's rain is measured on the day."
        self.assertEqual(set(known_from_moment(columns, moment)), {"on_promotion"})
        careless = "Each Sunday evening; promotions and rain are known ahead."
        self.assertEqual(set(known_from_moment(columns, careless)), {"on_promotion", "rain_mm"})  # what the person says is used…
        kind = self.kinds.get("forecasting")
        case = kind.case(0)
        run = kind.solve(type(case)(**{**case.__dict__, "moment": careless}))
        self.assertIn("rain_mm", run["covariates"]["to_confirm"])  # …and shown to confirm
        self.assertFalse(all(c["ok"] for c in kind.accept(run)))  # the acceptance can fail

    def test_the_engine_refuses_the_kind_until_it_runs_it_and_the_matrix_says_partial(self):
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio.solution import Solution

        with self.assertRaises(ValueError) as refused:
            Solution(target="units", task="regression", prediction_moment="Each Sunday for the week ahead", kind="forecasting",
                     kind_fields={"series_column": "series", "horizon": 7})
        self.assertIn("does not run", str(refused.exception))
        passing = {"kind": "forecasting", "status": "supported", "missing": [], "checks": [{"decision": "d", "ok": True}]}
        row = {r["kind"]: r for r in acceptance.matrix([passing])}["forecasting"]
        self.assertEqual(row["status"], "partial")
        self.assertIn("engine does not run it yet", " ".join(row["missing"]))


if __name__ == "__main__":
    unittest.main()
