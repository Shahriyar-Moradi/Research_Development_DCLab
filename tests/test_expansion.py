"""Fast, offline tests for the expansion_v1 campaign (no downloads, no data/external)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dclab_rnd.expansion import datasets as ds
from dclab_rnd.expansion import runner

HAVE_GBDT = all(importlib.util.find_spec(name) is not None for name in ("lightgbm", "xgboost"))


def _fraud_bundle(n: int = 500) -> ds.TaskBundle:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172800, n))})
    for i in range(1, 5):
        frame[f"V{i}"] = rng.normal(size=n)
    frame["Amount"] = rng.exponential(80, n)
    logit = -2.6 + 1.8 * frame["V1"] - 1.2 * frame["V2"]
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return ds.prepare_bundle(ds.SPECS["credit_card_fraud"], frame, y, cv_folds=2)


def _letter_bundle(n: int = 240) -> ds.TaskBundle:
    rng = np.random.default_rng(1)
    columns = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
    y = pd.Series(np.repeat([0, 1, 2], n // 3))
    frame = pd.DataFrame(rng.integers(0, 12, size=(n, len(columns))), columns=columns)
    frame["width"] += y.to_numpy() * 2
    frame["x-ege"] += (y.to_numpy() == 2) * 3
    return ds.prepare_bundle(ds.SPECS["letter_recognition"], frame, y, class_labels=["1", "2", "3"], cv_folds=2)


def _bike_bundle(n: int = 160) -> ds.TaskBundle:
    rng = np.random.default_rng(2)
    dates = pd.date_range("2011-01-03", periods=n, freq="D")
    trend = np.linspace(1500, 4500, n)
    season = 800 * np.sin(2 * np.pi * np.asarray(dates.dayofyear) / 365.25)
    temp = 15 + 10 * np.sin(2 * np.pi * (np.asarray(dates.dayofyear) - 100) / 365.25) + rng.normal(0, 2, n)
    cnt = np.round(trend + season + 40 * temp + rng.normal(0, 200, n)).clip(50)
    casual = np.round(cnt * rng.uniform(0.1, 0.3, n))
    frame = pd.DataFrame(
        {
            "instant": np.arange(3, n + 3),
            "dteday": dates.strftime("%Y-%m-%d").astype(object),
            "season": pd.Series(np.where(dates.month <= 3, "WINTER", "SPRING"), dtype=object),
            "yr": dates.year,
            "mnth": pd.Series(dates.strftime("%b").str.upper(), dtype=object),
            "holiday": "N",
            "weekday": pd.Series(dates.strftime("%a").str.upper(), dtype=object),
            "temp": temp,
            "atemp": temp + 15,
            "hum": rng.uniform(30, 90, n),
            "windspeed": rng.uniform(2, 25, n),
            "casual": casual.astype(int),
            "registered": (cnt - casual).astype(int),
            "workday": pd.Series(np.where(dates.dayofweek < 5, "Y", "N"), dtype=object),
            "weather": pd.Series(rng.choice(["GOOD", "MISTY"], n), dtype=object),
            "days_since_2011": np.arange(2, n + 2),
            "cnt_2d_bfr": np.r_[[1000, 1000], cnt[:-2]].astype(int),
        }
    )
    return ds.prepare_bundle(ds.SPECS["bike_sharing_daily"], frame, pd.Series(cnt, dtype=float), cv_folds=2)


def _reviews_bundle(n: int = 320) -> ds.TaskBundle:
    rng = np.random.default_rng(3)
    good = ["love", "perfect", "great", "flattering", "soft", "comfortable"]
    bad = ["return", "cheap", "tight", "itchy", "disappointed", "awful"]
    neutral = ["dress", "top", "fabric", "color", "size", "ordered", "the", "and", "fit"]
    y = (rng.uniform(size=n) < 0.75).astype(int)
    texts, titles = [], []
    for label in y:
        pool = good if label else bad
        words = list(rng.choice(pool, 3)) + list(rng.choice(neutral, 6))
        if rng.uniform() < 0.15:
            words += list(rng.choice(bad if label else good, 1))
        rng.shuffle(words)
        texts.append(" ".join(words) if rng.uniform() > 0.03 else np.nan)
        titles.append(" ".join(rng.choice(pool + neutral, 2)) if rng.uniform() > 0.15 else np.nan)
    frame = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "Clothing ID": rng.integers(0, 30, n),
            "Age": rng.integers(18, 80, n),
            "Title": pd.Series(titles, dtype=object),
            "Review Text": pd.Series(texts, dtype=object),
            "Rating": np.where(y == 1, rng.choice([4, 5], n), rng.choice([1, 2, 3], n)),
            "Positive Feedback Count": rng.poisson(2, n),
            "Division Name": pd.Series(rng.choice(["General", "Petite"], n), dtype=object),
            "Department Name": pd.Series(rng.choice(["Tops", "Dresses", "Bottoms"], n), dtype=object),
            "Class Name": pd.Series(rng.choice(["Knits", "Blouses", "Dresses", "Pants"], n), dtype=object),
        }
    )
    return ds.prepare_bundle(ds.SPECS["ecommerce_clothing_reviews"], frame, pd.Series(y), cv_folds=2)


BUILDERS = {
    "credit_card_fraud": _fraud_bundle,
    "letter_recognition": _letter_bundle,
    "bike_sharing_daily": _bike_bundle,
    "ecommerce_clothing_reviews": _reviews_bundle,
}


class HelperTests(unittest.TestCase):
    def test_adjusted_score_orientation(self) -> None:
        self.assertAlmostEqual(runner.adjusted_score(0.8, 0.04, True), 0.79)
        self.assertAlmostEqual(runner.adjusted_score(100.0, 8.0, False), 102.0)
        self.assertGreater(runner.oriented_gain(90.0, 100.0, False), 0)
        self.assertGreater(runner.oriented_gain(0.9, 0.8, True), 0)

    @staticmethod
    def _row(name: str, mean: float, std: float, features: int, metric: str = "m") -> dict:
        return {
            "config_id": name,
            "recipe": name,
            "feature_count_mean": features,
            "elapsed_seconds": 1.0,
            "metrics": {metric: {"mean": mean, "std": std}},
        }

    def test_select_within_tolerance_prefers_smallest(self) -> None:
        rows = [self._row("big", 0.900, 0.01, 100), self._row("small", 0.899, 0.01, 10), self._row("bad", 0.80, 0.0, 5)]
        self.assertEqual("small", runner.select_within_tolerance(rows, "m", True, 0.002)["recipe"])
        mae = [self._row("a", 100.0, 1, 50), self._row("b", 100.8, 1, 20), self._row("c", 110.0, 1, 5)]
        self.assertEqual("b", runner.select_within_tolerance(mae, "m", False, 0.01, relative=True)["recipe"])

    def test_rank_rows_penalizes_variance(self) -> None:
        rows = [self._row("noisy", 0.90, 0.20, 1), self._row("stable", 0.88, 0.0, 1)]
        self.assertEqual("stable", runner.rank_rows(rows, "m", True)[0]["config_id"])
        mae = [self._row("noisy", 100.0, 80.0, 1), self._row("stable", 110.0, 0.0, 1)]
        self.assertEqual("stable", runner.rank_rows(mae, "m", False)[0]["config_id"])

    def test_tuning_decision_requires_margin(self) -> None:
        base = self._row("base", 0.900, 0.0, 1)
        small = self._row("cand", 0.9005, 0.0, 1)
        big = self._row("cand2", 0.910, 0.0, 1)
        self.assertFalse(runner.tuning_decision(base, [small], "m", True, 0.001)["accepted"])
        decision = runner.tuning_decision(base, [small, big], "m", True, 0.001)
        self.assertTrue(decision["accepted"])
        self.assertEqual("cand2", decision["selected"]["config_id"])
        mae_base = self._row("base", 100.0, 0.0, 1)
        self.assertFalse(runner.tuning_decision(mae_base, [self._row("c", 99.5, 0, 1)], "m", False, 0.01, relative=True)["accepted"])
        self.assertTrue(runner.tuning_decision(mae_base, [self._row("c", 98.0, 0, 1)], "m", False, 0.01, relative=True)["accepted"])

    def test_binary_metrics_and_recall_at_precision(self) -> None:
        y = np.array([0] * 990 + [1] * 10)
        scores = np.r_[np.linspace(0, 0.4, 990), np.linspace(0.5, 0.9, 10)]
        metrics = runner.binary_metrics(y, scores)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["majority_baseline_accuracy"], 0.99)
        self.assertAlmostEqual(metrics["recall_at_precision_target"], 1.0)
        self.assertAlmostEqual(metrics["precision_at_k_positives"], 1.0)
        none = runner.recall_at_precision(np.array([0, 1, 0, 1]), np.array([0.9, 0.1, 0.8, 0.2]), 0.9)
        self.assertFalse(none["reached"])

    def test_multiclass_and_regression_metrics(self) -> None:
        y = np.array([0, 1, 2, 2])
        proba = np.eye(3)[[0, 1, 2, 1]]
        metrics = runner.multiclass_metrics(y, proba, 3, detail=True)
        self.assertEqual(3, len(metrics["per_class_f1"]))
        self.assertIn(metrics["worst_3_classes"][0]["class_index"], {1, 2})
        self.assertEqual(1.0, metrics["per_class_f1"][0])
        reg = runner.regression_metrics([100, 200], [110, 180])
        self.assertAlmostEqual(reg["mae"], 15.0)
        self.assertAlmostEqual(reg["mape"], 0.1)

    def test_prior_correction_restores_base_rate(self) -> None:
        self.assertAlmostEqual(float(runner.prior_correct(np.array([0.5]), 0.25)[0]), 0.2)
        rows = runner._fit_rows(np.array([1] * 5 + [0] * 100), 0.25, seed=1)
        self.assertEqual(5 + 25, len(rows))

    def test_bootstrap_modes(self) -> None:
        y = np.arange(100, dtype=float)
        pred = y + np.random.default_rng(0).normal(0, 5, 100)
        fn = runner.primary_metric_fn("timeseries_regression", "mae")
        for kwargs in ({}, {"block_length": 7}, {"groups": np.repeat(np.arange(20), 5)}):
            ci = runner.bootstrap_ci(y, pred, fn, draws=50, **kwargs)
            self.assertLessEqual(ci["low"], ci["point"])
            self.assertGreaterEqual(ci["high"], ci["low"])

    def test_past_only_lags_never_look_ahead(self) -> None:
        dates = pd.date_range("2012-01-01", periods=40, freq="D").strftime("%Y-%m-%d")
        y = np.arange(40, dtype=float) * 10
        table = runner.past_only_lag_table(dates, y, min_lag=2)
        changed = y.copy()
        changed[30:] = 99999.0  # perturb day 30 onward
        table2 = runner.past_only_lag_table(dates, changed, min_lag=2)
        # Features for days up to 31 (= 30 + min_lag - 1) must be unaffected.
        pd.testing.assert_frame_equal(table.iloc[:32], table2.iloc[:32])
        self.assertEqual(table.loc[dates[20], "past_lag3"], y[17])

    def test_identity_detector_and_canary(self) -> None:
        bundle = _bike_bundle()
        findings = runner.suspicious_features(bundle.X_train, bundle.y_train, "timeseries_regression", declared=["casual", "registered"])
        reasons = {f["feature"]: f["reasons"] for f in findings}
        self.assertIn("arithmetic_target_identity", reasons["casual"])
        self.assertTrue(runner._canary_check(bundle.X_train.iloc[:, :3], bundle.y_train, "timeseries_regression", [])["passed"])

    def test_dataset_cards_are_static(self) -> None:
        cards = ds.dataset_cards()
        self.assertEqual(4, len(cards))
        required = {"key", "name", "description", "task_type", "rows", "features", "source", "decision_time_contract", "blocked_features"}
        for card in cards:
            self.assertTrue(required <= set(card))
            self.assertEqual("positive_rate" in card, card["task_type"] in {"binary_imbalanced", "text_tabular_binary"})

    def test_experiment_numbering_and_manifest(self) -> None:
        self.assertEqual("EXP-051", runner.experiment_id("credit_card_fraud", "data_understanding"))
        self.assertEqual("EXP-070", runner.experiment_id("ecommerce_clothing_reviews", "optimization_reliability"))
        manifest = runner.build_manifest()
        self.assertEqual(20, manifest["experiment_count"])

    def test_splits_respect_structure(self) -> None:
        fraud = _fraud_bundle()
        self.assertGreater(fraud.X_test["Time"].min(), fraud.X_train["Time"].max() - 1e-9)
        for fit, valid in fraud.cv_splits:
            self.assertLess(fit.max(), valid.min())
        reviews = _reviews_bundle()
        self.assertFalse(set(reviews.groups("train")) & set(reviews.groups("test")))


@unittest.skipUnless(HAVE_GBDT, "lightgbm/xgboost not installed")
class StageWritingTests(unittest.TestCase):
    def test_all_stages_write_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, runner.fast_profile():
            root = Path(tmp)
            out_dir = root / "campaign"
            runner.run_campaign(
                root,
                out_dir=out_dir,
                loader=lambda _root, key: BUILDERS[key](),
                log=lambda _msg: None,
            )
            files = sorted((out_dir / "results").glob("EXP-*.json"))
            self.assertEqual(20, len(files))
            for path in files:
                result = json.loads(path.read_text())
                self.assertEqual([], runner.validate_result(result), path.name)
                for key in runner.REQUIRED_KEYS:
                    self.assertIn(key, result)
                self.assertNotIn("llm_review", result)
                self.assertEqual("expansion_v1", result["campaign_id"])
                if result["kind"] == "leakage_audit":
                    self.assertIsInstance(result["evidence"]["apparent_lift"], float)
                if result["kind"] == "optimization_reliability":
                    self.assertTrue(result["evidence"]["holdout_consumed"])
                    self.assertIn("holdout_primary_metric_ci", result["evidence"])
            memory = (out_dir / "agent_memory.jsonl").read_text().splitlines()
            self.assertTrue(memory)
            self.assertNotIn("provenance", json.loads(memory[0]))
            self.assertIn("expansion_v1", (out_dir / "CAMPAIGN_REPORT.md").read_text())
            # resumable: a second run skips everything
            executed = runner.run_campaign(root, out_dir=out_dir, loader=lambda _r, k: BUILDERS[k](), log=lambda _m: None)
            self.assertEqual([], executed)


if __name__ == "__main__":
    unittest.main()
