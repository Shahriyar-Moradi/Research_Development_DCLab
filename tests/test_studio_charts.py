"""D9: a project's charts from its stage records (the owner, 2026-10-27: "show plot" got bars in text, "plot accuracy"
was refused while the record kept it). Every number comes from a record; a chart the records cannot give is named."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core.model_evaluation import evaluate  # noqa: E402
from dclab_rnd.intern.tools import RESULT_CHARS, compact_record  # noqa: E402
from dclab_rnd.studio.charts import charts  # noqa: E402


def _stage(stage, evidence, metric="roc_auc"):
    return {"stage": stage, "title": stage, "primary_metric": metric, "setup_summary": "", "claims": [], "notes": [], "evidence": evidence}


def _records(final_evidence, metric="roc_auc", leakage=None):
    m = lambda mean, std: {metric: {"mean": mean, "std": std}}  # noqa: E731
    return {
        "leakage": _stage("leakage", leakage or {"identifier_exclusions": {"Loan_ID": "declared identifier"}, "declared_leakage_features": [],
                                                 "heuristic_review_candidates": [], "apparent_lift": 0.0, "apparent_lift_metric": metric,
                                                 "detector_canary_passed": True}, metric),
        "features": _stage("features", {"selected_recipe": "raw", "selection_rule": "smallest within tolerance",
                                        "stage_results": [{"recipe": "raw", "feature_count_mean": 55.0, "metrics": m(0.72, 0.03)},
                                                          {"recipe": "poly2", "feature_count_mean": 61.0, "metrics": m(0.721, 0.04)}]}, metric),
        "models": _stage("models", {"selected_model": "logistic_regression", "selection_rule": "best mean",
                                    "ranking": [{"model": "logistic_regression", "mean": 0.787, "std": 0.017}, {"model": "xgboost", "mean": 0.705, "std": 0.02}]}, metric),
        "final": _stage("final", final_evidence, metric),
    }


def _binary_final(keep_scores=True):
    rng = np.random.default_rng(5)
    y = (rng.random(1600) < 0.13).astype(int)
    s = np.clip(0.3 * y + rng.random(1600) * 0.7, 0, 1)
    held = evaluate("binary_imbalanced", y, s, rows="holdout")
    ev = {"model": "logistic_regression", "feature_recipe": "raw", "holdout_rows": 1600, "holdout_uses_in_this_project": 1,
          "holdout_metrics": {k: v for k, v in held.metrics.items() if v is not None} | {"threshold": 0.5},
          "holdout_primary_metric_ci": {"point": held.metrics["roc_auc"], "low": 0.76, "high": 0.83, "method": "iid row bootstrap"},
          "cv_selected_metric_mean": 0.786, "baselines": {"majority_class": {"accuracy": 0.87, "roc_auc": 0.5, "average_precision_random_ranking": 0.13}},
          "tuning_decision": {"accepted": False}, "calibration_bins": [{"lower": 0.0, "upper": 0.5, "count": 1500, "predicted": 0.1, "observed": 0.11}]}
    if keep_scores:
        ev["evaluation"] = {"holdout": held.model_dump()}
    return ev


class ChartsTest(unittest.TestCase):
    def test_a_binary_project_gets_every_chart_from_its_records(self):
        out = charts(_records(_binary_final()), "p1")
        ids = [c["id"] for c in out["charts"]]
        self.assertEqual(ids, ["leakage", "features", "models", "holdout", "measures", "budgets", "gains", "pr", "calibration"])
        self.assertEqual(out["missing"], [])
        by = {c["id"]: c for c in out["charts"]}
        self.assertEqual([r["chosen"] for r in by["models"]["rows"]], [True, False])
        self.assertEqual(by["holdout"]["low"], 0.76)
        self.assertIn({"label": "chance", "value": 0.5}, by["holdout"]["marks"])
        self.assertEqual(by["leakage"]["verdict"], "no leak found")
        self.assertEqual(by["budgets"]["source"], "scores")
        self.assertLessEqual(max(len(s["points"]) for s in by["gains"]["series"]), 21)
        self.assertLess(len(json.dumps(out)), RESULT_CHARS, "whole within the intern's tool-result limit")

    def test_an_older_record_without_scores_names_what_it_cannot_draw(self):
        out = charts(_records(_binary_final(keep_scores=False)), "p1")
        ids = [c["id"] for c in out["charts"]]
        self.assertIn("measures", ids)
        self.assertIn("calibration", ids)
        self.assertNotIn("gains", ids)
        self.assertTrue(any("review budgets" in m for m in out["missing"]))

    def test_regression_gets_the_generic_charts_and_a_leak_suspect_shows(self):
        final = {"model": "ridge", "feature_recipe": "raw", "holdout_rows": 300, "holdout_metrics": {"mae": 2.1, "rmse": 3.0},
                 "holdout_primary_metric_ci": {"point": 2.1, "low": 1.9, "high": 2.4}, "cv_selected_metric_mean": 2.2, "baselines": {}, "tuning_decision": {"accepted": False}}
        leak = {"identifier_exclusions": {}, "declared_leakage_features": ["casual"], "apparent_lift": 0.4, "apparent_lift_metric": "mae",
                "heuristic_review_candidates": [{"feature": "registered", "reasons": ["extreme univariate signal"]}], "detector_canary_passed": True}
        out = charts(_records(final, metric="mae", leakage=leak), "p2")
        by = {c["id"]: c for c in out["charts"]}
        self.assertEqual(set(by), {"leakage", "features", "models", "holdout"})
        self.assertEqual(by["models"]["better"], "lower")
        self.assertEqual(by["leakage"]["verdict"], "leak suspects to review")
        self.assertEqual([k["column"] for k in by["leakage"]["kept_out"]], ["casual"])

    def test_review_findings_uplift_rows_declared_columns_caught_leaks_and_direction(self):
        # an uplift final record keeps the holdout's row positions, not a count: nothing may raise on it
        uplift = {"model": "two_model", "feature_recipe": "raw", "holdout_rows": [3, 7, 9], "holdout_metrics": {"qini_coefficient": 0.12},
                  "holdout_primary_metric_ci": {"low": 0.05, "high": 0.19}, "cv_selected_metric_mean": 0.1, "baselines": {}, "tuning_decision": {"accepted": False}}
        none = {"identifier_exclusions": {}, "declared_leakage_features": [], "heuristic_review_candidates": [], "apparent_lift": None,
                "detector_canary_passed": None}
        out = charts(_records(uplift, metric="qini_coefficient", leakage=none), "u")
        self.assertIn("holdout", [c["id"] for c in out["charts"]])
        self.assertEqual(out["charts"][0]["verdict"], "no detector run: the kind's own checks only", "never 'no leak found' without a detector")
        self.assertEqual(compact_record(_records(uplift, metric="qini_coefficient")["final"])["numbers"]["model"], "two_model")
        # a declared column the engine also lists among its candidates is kept out, not a suspect; its lift is a leak caught
        caught = {"identifier_exclusions": {}, "declared_leakage_features": ["duration"], "apparent_lift": 0.21, "apparent_lift_metric": "roc_auc",
                  "heuristic_review_candidates": [{"feature": "duration", "reasons": ["declared_post_outcome_or_contested"]}], "detector_canary_passed": True}
        lk = charts(_records(_binary_final(), leakage=caught), "c")["charts"][0]
        self.assertEqual((lk["verdict"], lk["suspects"]), ("leak caught and kept out", []))
        self.assertEqual(charts(_records(_binary_final(), metric="mase"), "f")["charts"][2]["better"], "lower")

    def test_get_results_of_one_stage_says_where_the_project_stands(self):
        from dclab_rnd.intern.tools import Toolbox

        records = _records(_binary_final())

        class Projects:
            def get(self, pid):
                return {"id": pid}

            def records(self, pid):
                return records

        box = Toolbox.__new__(Toolbox)
        box.projects = Projects()
        one = Toolbox.get_results(box, "p1", "models")
        self.assertEqual(one["stages_run"], ["leakage", "features", "models", "final"])
        self.assertIn("Running the final stage again reads the holdout again", one["final_already_ran"])
        self.assertIn("get_charts", one["final_already_ran"])
        self.assertNotIn("final_already_ran", Toolbox.get_results(box, "p1", "final"))

    def test_get_results_keeps_the_holdout_measures_and_budgets(self):
        numbers = compact_record(_records(_binary_final())["final"])["numbers"]
        self.assertIn("accuracy", numbers["holdout_measures"])
        self.assertIn("recall", numbers["holdout_measures"])
        self.assertEqual(len(numbers["review_budgets"]), 5)


if __name__ == "__main__":
    unittest.main()
