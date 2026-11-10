"""Tests for LeakageContrast contract and contrastive shadow comparison (SCI-02)."""

from __future__ import annotations

import unittest
from pydantic import ValidationError

from dclab_core.leakage_contrast import LeakageContrast, build_leakage_contrast, compute_oriented_gain


class TestLeakageContrast(unittest.TestCase):
    def test_leakage_contrast_contract_validation(self) -> None:
        contrast = build_leakage_contrast(
            project="test_proj",
            target="churn",
            leakage_features=["future_cancellation_date"],
            metric="roc_auc",
            higher_is_better=True,
            clean_cv_metric=0.75,
            leakage_cv_metric=0.92,
            clean_holdout_metric=0.74,
            leakage_holdout_metric=0.79,
            decision_time_contract="Prediction at contract sign-up.",
        )
        self.assertEqual(contrast.project, "test_proj")
        self.assertEqual(contrast.target, "churn")
        self.assertEqual(contrast.leakage_features, ["future_cancellation_date"])
        self.assertAlmostEqual(contrast.cv_apparent_lift, 0.17, places=4)
        self.assertAlmostEqual(contrast.holdout_apparent_lift, 0.05, places=4)
        self.assertAlmostEqual(contrast.generalization_drop, 0.12, places=4)
        self.assertTrue(contrast.is_generalization_hazard())
        self.assertIn("apparent CV roc_auc lift of +0.1700", contrast.lesson)

        # Frozen pydantic model should reject mutation
        with self.assertRaises((ValidationError, TypeError)):
            contrast.cv_apparent_lift = 0.50

    def test_lower_is_better_metric_contrast(self) -> None:
        contrast = build_leakage_contrast(
            project="reg_proj",
            target="delivery_time",
            leakage_features=["customer_delivered_timestamp"],
            metric="mae",
            higher_is_better=False,
            clean_cv_metric=10.5,
            leakage_cv_metric=2.1,  # Much lower error on CV
            clean_holdout_metric=10.6,
            leakage_holdout_metric=9.8,  # Only small drop on holdout
        )
        self.assertAlmostEqual(contrast.cv_apparent_lift, 8.4, places=4)
        self.assertAlmostEqual(contrast.holdout_apparent_lift, 0.8, places=4)
        self.assertAlmostEqual(contrast.generalization_drop, 7.6, places=4)
        self.assertTrue(contrast.is_generalization_hazard())

    def test_compute_oriented_gain(self) -> None:
        # higher is better: 0.8 vs 0.7 -> +0.1
        self.assertAlmostEqual(compute_oriented_gain(0.8, 0.7, True), 0.1)
        # lower is better: 5.0 vs 8.0 -> +3.0 gain (improvement)
        self.assertAlmostEqual(compute_oriented_gain(5.0, 8.0, False), 3.0)

    def test_stage_leakage_generates_contrast(self) -> None:
        from dclab_rnd.studio import ProjectStore, data as sd, engine, solution as sc
        import tempfile
        from pathlib import Path
        import numpy as np
        import pandas as pd

        n = 300
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "age": rng.integers(18, 80, n),
            "monthly_fee": rng.uniform(10, 120, n).round(2),
            "refund_after_churn": rng.uniform(5, 50, n),  # Leakage feature
        })
        y = rng.choice(["yes", "no"], n)
        frame["churned"] = y

        with tempfile.TemporaryDirectory() as tmp:
            st = ProjectStore(Path(tmp))
            project = st.create("Test churn project", "telecom", "Predict churn")
            path = st.data_dir(project["id"]) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = sd.load_table(path)
            project["data"] = {
                "filename": "data.csv",
                "rows": len(loaded),
                "columns": list(loaded.columns),
                "sha256": sd.sha256(path),
                "profile": sd.profile_table(loaded),
            }
            solution_dict = {
                "target": "churned",
                "task": "binary",
                "positive_label": "yes",
                "identifiers": ["customer_id"],
                "prediction_moment": "Before renewal; refunds are given only after cancellation.",
                "forbidden": [{"column": "refund_after_churn", "reason": "refund occurs after churn"}],
            }
            project["solution"] = sc.Solution(**solution_dict).model_dump()
            project["settings"]["quick"] = True
            st.save(project)

            p = engine.prepare(st, project)
            evidence, claims, summary = engine.stage_leakage(p, "E-LEAK")
            self.assertIn("leakage_contrast", evidence)
            contrast_data = evidence["leakage_contrast"]
            self.assertIsNotNone(contrast_data)
            self.assertEqual(contrast_data["leakage_features"], ["refund_after_churn"])
            self.assertTrue(isinstance(contrast_data["cv_apparent_lift"], float))


if __name__ == "__main__":
    unittest.main()
