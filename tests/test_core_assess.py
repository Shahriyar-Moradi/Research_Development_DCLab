"""Package 17.5: a better workflow per task (dclab_core.assess and the engine's final stage).

Thresholds and intervals are chosen on out-of-fold predictions and read once on the holdout; multiclass results name
the confused classes and the tiny ones; regression reports intervals, error by range and segment, and the cap.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


class AssessTest(unittest.TestCase):
    def test_binary_threshold_from_the_budget_is_chosen_out_of_fold(self):
        from dclab_core.assess import assess

        rng = np.random.default_rng(0)
        y_oof = (rng.random(5000) < 0.02).astype(int)
        p_oof = np.clip(0.02 + 0.5 * y_oof + rng.normal(0, 0.1, 5000), 0, 1)
        y_test = (rng.random(2000) < 0.02).astype(int)
        p_test = np.clip(0.02 + 0.5 * y_test + rng.normal(0, 0.1, 2000), 0, 1)
        out = assess("binary", y_oof, p_oof, y_test, p_test, budget=0.01, partial_labels=True)
        threshold = out["threshold"]
        self.assertAlmostEqual(threshold["value"], round(float(np.quantile(p_oof, 0.99)), 6))
        self.assertEqual(threshold["chosen_on"], "out-of-fold training predictions")
        self.assertAlmostEqual(threshold["training"]["alert_share"], 0.01, places=2)
        self.assertIn("lower bound", out["precision_note"])
        self.assertIn("expected_calibration_error", out["calibration"])

    def test_binary_threshold_from_costs_moves_with_the_costs(self):
        from dclab_core.assess import assess

        rng = np.random.default_rng(1)
        y = (rng.random(4000) < 0.1).astype(int)
        p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.15, 4000), 0, 1)
        cheap_misses = assess("binary", y, p, y, p, costs={"false_positive": 10, "false_negative": 1})["threshold"]["value"]
        dear_misses = assess("binary", y, p, y, p, costs={"false_positive": 1, "false_negative": 50})["threshold"]["value"]
        self.assertLess(dear_misses, cheap_misses)  # when a miss costs more, alert on more cases

    def test_multiclass_names_the_confused_pairs_and_top_k(self):
        from dclab_core.assess import assess

        classes = ["billing", "login", "legal"]
        y = np.array(["billing"] * 50 + ["login"] * 45 + ["legal"] * 5)
        proba = np.zeros((100, 3))
        proba[:50, 0] = 0.9
        proba[50:80, 1] = 0.9
        proba[80:95, 0] = 0.9          # 15 login tickets routed to billing
        proba[95:, 1] = 0.9            # the 5 legal tickets routed to login
        proba = proba + 0.01
        out = assess("multiclass", y, proba, y, proba, classes=classes, tiny_classes=["legal"])
        self.assertEqual(out["most_confused"][0], {"true": "login", "predicted": "billing", "rows": 15, "share_of_true": 0.3333})
        self.assertEqual(out["tiny_classes"][0]["recall"], 0.0)
        self.assertIn("top_2", out["top_k_accuracy"])

    def test_regression_intervals_cover_and_the_cap_is_reported(self):
        from dclab_core.assess import assess

        rng = np.random.default_rng(2)
        truth = rng.gamma(4, 8, 6000)
        pred = truth + rng.normal(0, 5, 6000)
        capped = np.minimum(truth, 60)
        from dclab_core.analyse import target_cap

        out = assess("regression", capped[:4000], pred[:4000], capped[4000:], pred[4000:],
                     segments={"weather": rng.choice(["rain", "clear"], 2000).tolist()}, cap=target_cap(capped[:4000]))
        self.assertAlmostEqual(out["intervals"]["80%"]["holdout_coverage"], 0.8, delta=0.05)
        self.assertEqual(out["at_cap"]["cap"], 60.0)
        self.assertEqual(set(out["error_by_segment"]["weather"]), {"rain", "clear"})
        self.assertEqual(len(out["error_by_prediction_range"]), 5)

    def test_the_threshold_can_alert_on_most_rows_when_misses_are_dear(self):
        from dclab_core.assess import assess

        rng = np.random.default_rng(3)
        y = (rng.random(8000) < 0.3).astype(int)
        p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 8000), 0, 1)
        costs = {"false_positive": 1, "false_negative": 20}
        chosen = assess("binary", y, p, y, p, costs=costs)["threshold"]["training"]
        best = min(costs["false_positive"] * ((p >= t) & (y == 0)).sum() + costs["false_negative"] * ((p < t) & (y == 1)).sum()
                   for t in np.unique(p))
        self.assertEqual(chosen["false_positives"] + 20 * chosen["false_negatives"], best)  # the true optimum, not a floor at the median
        self.assertGreater(chosen["alert_share"], 0.6)
        common = (rng.random(8000) < 0.6).astype(int)
        q = np.clip(0.5 + 0.1 * common + rng.normal(0, 0.2, 8000), 0, 1)
        self.assertGreater(assess("binary", common, q, common, q)["threshold"]["training"]["alert_share"], 0.55)

    def test_macro_recall_counts_every_class_and_a_discrete_target_is_not_capped(self):
        from dclab_core.analyse import target_cap
        from dclab_core.assess import assess

        classes = [f"c{i}" for i in range(40)]
        y = np.array([c for i, c in enumerate(classes) for _ in range(100 - 2 * i)])
        pred = np.where(np.isin(y, classes[:30]), y, "c0")  # the 10 smallest classes are never right
        out = assess("multiclass", y, pred, y, pred)
        self.assertEqual(len(out["per_class"]), 30)
        self.assertAlmostEqual(out["macro_recall"], 30 / 40, places=3)
        self.assertIsNone(target_cap(np.ones(500)))
        self.assertIsNone(target_cap(np.random.default_rng(0).integers(1, 6, 500)))  # a 1-5 rating is a scale
        self.assertNotIn("at_cap", assess("regression", np.ones(50), np.ones(50), np.ones(50), np.ones(50)))


class PlanInputsTest(unittest.TestCase):
    def plan(self, policy, coverage="complete"):
        return {"decisions": [{"key": "labels.coverage", "choice": coverage}, {"key": "threshold.policy", "choice": policy}]}

    def test_a_review_capacity_becomes_a_share_of_rows(self):
        from dclab_core.assess import threshold_inputs

        percent = threshold_inputs(self.plan({"kind": "review_budget", "capacity": "at most 0.5% of payments"}), 10_000, None)
        self.assertAlmostEqual(percent["budget"], 0.005)
        daily = threshold_inputs(self.plan({"kind": "review_budget", "capacity": "200 rides a day"}, "partial"), 100_000, 50)
        self.assertAlmostEqual(daily["budget"], 0.1)
        self.assertTrue(daily["partial_labels"])
        no_time = threshold_inputs(self.plan({"kind": "review_budget", "capacity": "200 rides a day"}), 100_000, None)
        self.assertIsNone(no_time["budget"])
        self.assertIn("ask", no_time["note"])

    def test_costs_need_a_number_for_both_errors(self):
        from dclab_core.assess import threshold_inputs

        both = threshold_inputs(self.plan({"kind": "expected_cost", "costs": "A false alarm costs 5 euros and a missed fraud costs 300 euros."}), 1000, None)
        self.assertEqual(both["costs"], {"false_positive": 5.0, "false_negative": 300.0})
        vague = threshold_inputs(self.plan({"kind": "expected_cost", "costs": "Missed fraud costs us a lot."}), 1000, None)
        self.assertIsNone(vague["costs"])
        self.assertIn("ask", vague["note"])
        self.assertEqual(threshold_inputs(None, 1000, None), {"budget": None, "costs": None, "partial_labels": False, "note": None})


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class EngineTest(unittest.TestCase):
    """The engine's final stage records the task's assessment on real runs of the replicas."""

    def run_project(self, frame, solution, plan=None):
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Replica", "general", "Assess")
        path = store.data_dir(project["id"]) / "data.csv"
        frame.to_csv(path, index=False)
        loaded = sd.load_table(path)
        project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path),
                           "profile": sd.profile_table(loaded)}
        project["solution"] = Solution(**solution).model_dump()
        project["settings"]["quick"] = True
        if plan is not None:
            project["plan"] = plan
        store.save(project)
        with runner.fast_profile():
            engine.run_all(store, project["id"])
        return store.read_stage(project["id"], "final")["evidence"]

    def test_tickets_get_per_class_results(self):
        from dclab_core.replicas import ticket_routing

        frame = ticket_routing.generate(seed=0, n=1500).tickets.drop(columns=["resolved_by", "first_response_minutes"])
        evidence = self.run_project(frame, {"target": "assigned_team", "task": "multiclass", "identifiers": ["ticket_id"],
                                            "prediction_moment": "When the ticket is created, from its subject, body and plan.",
                                            "time_column": "created_at"})
        assessment = evidence["task_assessment"]
        self.assertEqual(assessment["task"], "multiclass")
        self.assertTrue(assessment["per_class"])
        self.assertIn("billing", {c["class"] for c in assessment["per_class"]})

    def test_deliveries_get_intervals(self):
        from dclab_core.replicas import delivery_time

        frame = delivery_time.generate(seed=0, n=1500).orders.drop(columns=["delivered_at", "courier_rating"])
        evidence = self.run_project(frame, {"target": "actual_minutes", "task": "regression", "identifiers": ["order_id"],
                                            "prediction_moment": "When the customer places the order.", "group_column": "restaurant_id"})
        assessment = evidence["task_assessment"]
        self.assertEqual(assessment["task"], "regression")
        self.assertGreater(assessment["intervals"]["80%"]["holdout_coverage"], 0.5)
        self.assertIn("at_cap", assessment)

    def binary_frame(self):
        import pandas as pd

        rng = np.random.default_rng(4)
        n = 3000
        x = rng.normal(size=(n, 2))
        y = (rng.random(n) < 1 / (1 + np.exp(-(x[:, 0] * 2 - 3)))).astype(int)
        start = pd.Timestamp("2026-01-01")
        return pd.DataFrame({"case_id": range(n), "created_at": [start + pd.Timedelta(minutes=48 * i) for i in range(n)],
                             "x1": x[:, 0], "x2": x[:, 1], "flagged": y})

    def test_a_binary_project_takes_its_threshold_and_label_coverage_from_the_plan(self):
        plan = {"decisions": [{"key": "labels.coverage", "choice": "partial"},
                              {"key": "threshold.policy", "choice": {"kind": "review_budget", "capacity": "5 cases a day"}}]}
        evidence = self.run_project(self.binary_frame(), {"target": "flagged", "task": "binary", "identifiers": ["case_id"], "positive_label": "1",
                                                          "prediction_moment": "When the case is opened.", "time_column": "created_at"}, plan)
        assessment = evidence["task_assessment"]
        self.assertIn("review budget", assessment["threshold"]["chosen_by"])
        self.assertAlmostEqual(assessment["threshold"]["training"]["alert_share"], 5 * 80 / 2400, delta=0.02)  # 5 a day over ~80 days
        self.assertIn("lower bound", assessment["precision_note"])

    def test_an_assessment_that_fails_leaves_the_stage_as_it_was(self):
        from unittest import mock

        with mock.patch("dclab_core.assess.assess", side_effect=ValueError("broken")):
            evidence = self.run_project(self.binary_frame(), {"target": "flagged", "task": "binary", "identifiers": ["case_id"], "positive_label": "1",
                                                              "prediction_moment": "When the case is opened.", "time_column": "created_at"})
        self.assertNotIn("task_assessment", evidence)
        self.assertIn("holdout_metrics", evidence)


class MissingClassTest(unittest.TestCase):
    """Found by the ticket replica: a class missing from a training fold crashed the engine (XGBoost wants 0..k-1)."""

    def test_a_fold_without_a_class_still_fits_and_predicts_every_class(self):
        import pandas as pd

        from dclab_rnd.expansion import runner

        for family in ("xgboost", "logistic_regression"):
            with self.subTest(family=family):
                model = runner.make_model(family, "multiclass", None)
                rng = np.random.default_rng(0)
                X = rng.normal(size=(300, 3))
                y = rng.choice([0, 1, 3, 4], 300)  # class 2 is absent from this fold
                present = np.unique(y)
                model.fit(X, np.searchsorted(present, y))
                model.dclab_classes_ = present
                proba = runner.predict_scores(model, X, "multiclass", n_classes=5)
                self.assertEqual(proba.shape, (300, 5))
                self.assertTrue(np.allclose(proba[:, 2], 0))
                self.assertTrue(np.allclose(proba.sum(axis=1), 1))
                self.assertIsInstance(pd.Series(proba.argmax(axis=1)).iloc[0], (int, np.integer))

    def test_the_exported_notebook_fits_and_scores_a_fold_without_a_class(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import export

        namespace = {"np": np, "labels": ["a", "b", "c", "d", "e"]}
        exec(export.MULTICLASS_HELPERS, namespace)  # noqa: S102 - the notebook's own helper code
        pipeline = runner.make_model("xgboost", "multiclass", None)
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(200, 3)), rng.choice([0, 1, 3, 4], 200)
        namespace["fit"](pipeline, X, y)
        scores = namespace["class_scores"](pipeline, X)
        self.assertEqual(scores.shape, (200, 5))
        self.assertTrue(set(scores.argmax(axis=1)) <= {0, 1, 3, 4})
        self.assertEqual(export._fit_code("multiclass", "X_train", "y_train"), "fit(pipeline, X_train, y_train)")
        self.assertIn("class_scores(pipeline, X_test)", export._score_code("multiclass", "m"))


if __name__ == "__main__":
    unittest.main()
