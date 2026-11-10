"""Tests for Knowledge, Memory and Verified Skill Reuse (MEM-01, MEM-02, MEM-05)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dclab_rnd.agent_eval.memory_ablation import ABLATION_CASES, MemoryAblationEvaluator
from dclab_rnd.lessons import applicability_of, check_applicability
from dclab_rnd.memory.experiment_retrieval import ExperimentRetrievalIndex, PastExperimentRecord


class TestMemoryKnowledgeReuse(unittest.TestCase):
    def test_experiment_retrieval_indexing_and_query(self) -> None:
        index = ExperimentRetrievalIndex()
        index.add(PastExperimentRecord(
            experiment_id="EXP-01",
            project_id="P1",
            problem_kind="binary_classification",
            dataset_name="churn_data",
            rows=2000,
            columns=10,
            metric="roc_auc",
            recipe="tabular",
            family="lightgbm",
            params={"learning_rate": 0.05},
            status="accepted",
            score=0.85,
            baseline_score=0.80,
            lift=0.05,
            has_time=True,
            rationale="Feature engineering with lag columns improved AUC",
        ))
        index.add(PastExperimentRecord(
            experiment_id="EXP-02",
            project_id="P2",
            problem_kind="regression",
            dataset_name="sales_forecast",
            rows=50000,
            columns=8,
            metric="mae",
            recipe="tabular",
            family="xgboost",
            params={},
            status="accepted",
            score=12.5,
            baseline_score=15.0,
            lift=2.5,
            has_time=True,
            rationale="XGBoost on seasonal trends",
        ))

        # Query matching binary_classification with temporal structure
        res = index.query(problem_kind="binary_classification", rows=2500, has_time=True, top_k=1)
        self.assertEqual(len(res), 1)
        best_rec, score = res[0]
        self.assertEqual(best_rec.experiment_id, "EXP-01")
        self.assertEqual(best_rec.family, "lightgbm")

        # Persistence test
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "exp_index.json"
            index.save(save_path)
            loaded = ExperimentRetrievalIndex.load(save_path)
            self.assertEqual(loaded.count(), 2)

    def test_lesson_applicability_guards(self) -> None:
        # Lesson extracted from a time-ordered churn dataset with 5,000 rows
        scope = {
            "project": "Churn Predictor",
            "task": "binary",
            "problem_kind": "binary_classification",
            "rows": 5000,
            "split": "time-ordered holdout",
        }
        app = applicability_of(scope)
        self.assertTrue(app["requires_time"])
        self.assertEqual(app["min_rows"], 500)

        lesson = {"scope": scope, "applicability": app}

        # Case 1: Matching context -> Applicable
        ok_context = {
            "task": "binary",
            "problem_kind": "binary_classification",
            "rows": 3000,
            "has_time": True,
        }
        is_app, reason = check_applicability(lesson, ok_context)
        self.assertTrue(is_app)

        # Case 2: Incompatible task (regression) -> Blocked
        bad_task = {
            "task": "regression",
            "problem_kind": "regression",
            "rows": 3000,
            "has_time": True,
        }
        is_app, reason = check_applicability(lesson, bad_task)
        self.assertFalse(is_app)
        self.assertIn("Incompatible task", reason)

        # Case 3: Missing required time structure -> Blocked
        no_time = {
            "task": "binary",
            "problem_kind": "binary_classification",
            "rows": 3000,
            "has_time": False,
        }
        is_app, reason = check_applicability(lesson, no_time)
        self.assertFalse(is_app)
        self.assertIn("requires temporal structure", reason)

        # Case 4: Insufficient sample size (e.g. 50 rows) -> Blocked
        too_small = {
            "task": "binary",
            "problem_kind": "binary_classification",
            "rows": 40,
            "has_time": True,
        }
        is_app, reason = check_applicability(lesson, too_small)
        self.assertFalse(is_app)
        self.assertIn("Insufficient sample size", reason)

        # Case 5: Sample size exceeds max_rows (e.g. 100,000 rows vs max 50,000) -> Blocked
        too_large = {
            "task": "binary",
            "problem_kind": "binary_classification",
            "rows": 100000,
            "has_time": True,
        }
        is_app, reason = check_applicability(lesson, too_large)
        self.assertFalse(is_app)
        self.assertIn("Sample size exceeds lesson scope", reason)

    def test_memory_ablation_evaluator_comparison(self) -> None:
        evaluator = MemoryAblationEvaluator()
        summary = evaluator.run_suite()

        self.assertIn("disabled", summary)
        self.assertIn("brief_only", summary)
        self.assertIn("augmented", summary)

        disabled_rate = summary["disabled"]["safety_rate"]
        brief_rate = summary["brief_only"]["safety_rate"]
        aug_rate = summary["augmented"]["safety_rate"]

        # Arm C (augmented) should have the highest safety rate
        self.assertGreater(aug_rate, disabled_rate)
        self.assertGreaterEqual(aug_rate, brief_rate)

        # Zero inapplicable transfers occurred in Arm C
        self.assertEqual(summary["augmented"]["inapplicable_transfers"], 0)


if __name__ == "__main__":
    unittest.main()
