"""Tests for multi-column and difference identity leakage detection, and time column preservation."""

import unittest
import numpy as np
import pandas as pd

from dclab_core.analyse import analyse, _sum_identity
from dclab_core.contracts import ProblemBrief, TextFact
from dclab_rnd.studio import leakage_review


class MultiColumnLeakageTest(unittest.TestCase):

    def test_difference_identity_leakage_detected(self):
        rng = np.random.default_rng(42)
        n = 1000
        target = np.round(rng.gamma(3, 10, n), 2)
        base = np.round(rng.uniform(50, 100, n), 2)
        # feat_a - feat_b = target  => feat_a = base + target, feat_b = base
        feat_a = np.round(base + target, 2)
        feat_b = base
        noise = rng.normal(size=n)
        df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "noise": noise, "target": target})

        findings = analyse({"main": df}, target="target", task="regression").findings
        ids = {f.id for f in findings}
        self.assertIn("F-leak-diff-feat_a-feat_b", ids)

        leak = next(f for f in findings if f.id == "F-leak-diff-feat_a-feat_b")
        self.assertEqual(leak.severity, "critical")
        self.assertEqual(leak.evidence["columns"], ["feat_a", "feat_b"])
        self.assertIn("features.forbidden", leak.affects)

    def test_difference_identity_reversed_detected(self):
        rng = np.random.default_rng(43)
        n = 1000
        target = np.round(rng.gamma(3, 10, n), 2)
        base = np.round(rng.uniform(50, 100, n), 2)
        # feat_b - feat_a = target => feat_b = base + target, feat_a = base
        feat_b = np.round(base + target, 2)
        feat_a = base
        noise = rng.normal(size=n)
        df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "noise": noise, "target": target})

        findings = analyse({"main": df}, target="target", task="regression").findings
        ids = {f.id for f in findings}
        self.assertIn("F-leak-diff-feat_b-feat_a", ids)

    def test_three_column_sum_identity_detected(self):
        rng = np.random.default_rng(44)
        n = 1000
        target = np.round(rng.gamma(5, 10, n), 2)
        c1 = np.round(target * 0.3, 2)
        c2 = np.round(target * 0.4, 2)
        c3 = np.round(target - c1 - c2, 2)
        noise = rng.normal(size=n)
        df = pd.DataFrame({"part_1": c1, "part_2": c2, "part_3": c3, "noise": noise, "target": target})

        findings = analyse({"main": df}, target="target", task="regression").findings
        ids = {f.id for f in findings}
        self.assertTrue(any(f.id.startswith("F-leak-sum-part_1-part_2-part_3") for f in findings), f"Expected 3-part sum leak in {ids}")

    def test_row_sum_identity_detected(self):
        rng = np.random.default_rng(45)
        n = 500
        c1 = np.round(rng.uniform(1, 10, n), 2)
        c2 = np.round(rng.uniform(1, 10, n), 2)
        c3 = np.round(rng.uniform(1, 10, n), 2)
        c4 = np.round(rng.uniform(1, 10, n), 2)
        target = np.round(c1 + c2 + c3 + c4, 2)
        df = pd.DataFrame({"c1": c1, "c2": c2, "c3": c3, "c4": c4, "target": target})

        findings = _sum_identity(df, df["target"])
        self.assertTrue(len(findings) > 0)
        self.assertTrue(any("features.forbidden" in f.affects for f in findings))

    def test_time_column_not_flagged_as_leakage(self):
        rng = np.random.default_rng(46)
        n = 1000
        # Time column with post-moment-like name or high correlation with target trend
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        target = np.arange(n, dtype=float) + rng.normal(0, 1, n)
        df = pd.DataFrame({"created_time": dates, "val": rng.normal(size=n), "target": target})

        # Passing time_column="created_time"
        brief = ProblemBrief(
            target_column=TextFact(value="target", status="inferred", reason="declared"),
            time_column=TextFact(value="created_time", status="inferred", reason="split timestamp"),
        )
        findings = analyse({"main": df}, target="target", task="regression", brief=brief).findings
        ids = {f.id for f in findings}
        self.assertFalse(any(f.id == "F-leak-created_time" for f in findings), f"Time column flagged as leak in {ids}")

    def test_leakage_review_skips_declared_time_column(self):
        profile = {
            "columns": [
                {"name": "order_timestamp", "type": "datetime"},
                {"name": "feature_x", "type": "numeric"},
                {"name": "delivered_date", "type": "datetime"},
            ]
        }
        moment = "At order placement, before delivery or payment settlement"
        flagged = [{"column": "delivered_date", "reason": "delivered after order", "proof": ["DCLAB-R05"]}]

        # With time_column="order_timestamp", order_timestamp should never be flagged as "after the moment"
        out = leakage_review.review(
            profile,
            moment=moment,
            flagged=flagged,
            target="target",
            identifiers=["order_id"],
            time_column="order_timestamp",
        )
        flagged_cols = {item["column"] for item in out["items"]}
        self.assertNotIn("order_timestamp", flagged_cols)


if __name__ == "__main__":
    unittest.main()
