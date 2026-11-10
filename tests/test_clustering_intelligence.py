"""Tests for clustering intelligence and segmentation problem intake (Package 16.6 enhancement).

Verifies:
1. Hopkins statistic clusterability calculation on clustered vs uniform noise data.
2. Clustering strategy recommendation engine across data profiles.
3. Problem intake intelligence and disambiguation in Segmentation.detect().
4. HierarchicalModel adapter and multi-strategy clustering execution in segment.segment().
"""

import unittest
import numpy as np
import pandas as pd

from dclab_core import segment
from dclab_core.segment import HierarchicalModel, clusterability, recommend_clustering_strategy
from dclab_rnd.kinds.segmentation import Segmentation


class TestClusterability(unittest.TestCase):
    def test_clustered_blobs_yield_high_hopkins(self):
        rng = np.random.default_rng(42)
        # 3 separated compact Gaussian clusters
        c1 = rng.normal(loc=[-5, -5], scale=0.5, size=(100, 2))
        c2 = rng.normal(loc=[5, 5], scale=0.5, size=(100, 2))
        c3 = rng.normal(loc=[5, -5], scale=0.5, size=(100, 2))
        X = np.vstack([c1, c2, c3])

        result = clusterability(X, seed=42)
        self.assertGreater(result["hopkins"], 0.70)
        self.assertEqual(result["tendency"], "high")
        self.assertIn("strong natural cluster structure", result["description"])

    def test_uniform_noise_yields_low_hopkins(self):
        rng = np.random.default_rng(42)
        # Uniform noise over a bounding box
        X = rng.uniform(low=-10, high=10, size=(300, 2))

        result = clusterability(X, seed=42)
        self.assertLess(result["hopkins"], 0.65)
        self.assertIn(result["tendency"], ("uniform_noise", "moderate"))

    def test_small_sample_edge_case(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = clusterability(X)
        self.assertEqual(result["hopkins"], 0.5)
        self.assertEqual(result["tendency"], "uniform_noise")


class TestStrategyRecommendation(unittest.TestCase):
    def test_recommendation_on_small_structured_data(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "feature1": rng.normal(0, 1, 200),
            "feature2": rng.normal(5, 1, 200),
        })
        X, record = segment.prepare(df)
        strategy = recommend_clustering_strategy(df, X, record, seed=0)

        self.assertIn("kmeans", strategy["recommended_methods"])
        self.assertIn("mixture", strategy["recommended_methods"])
        # For small N=200, hierarchical is recommended as well
        self.assertIn("hierarchical", strategy["recommended_methods"])
        self.assertIn("clusterability", strategy)
        self.assertIn("rationales", strategy)

    def test_categorical_heavy_dataset_recommendation(self):
        df = pd.DataFrame({
            "num1": np.arange(100, dtype=float),
            "cat1": ["A", "B", "C", "D"] * 25,
            "cat2": ["X", "Y"] * 50,
        })
        X, record = segment.prepare(df)
        strategy = recommend_clustering_strategy(df, X, record, seed=0)

        self.assertGreater(strategy["data_profile"]["categorical_ratio"], 0.40)
        self.assertIn("categorical_note", strategy["rationales"])


class TestSegmentationIntentDetection(unittest.TestCase):
    def setUp(self):
        self.kind = Segmentation()

    def test_detects_conversational_grouping_intent(self):
        queries = [
            "We want to find natural groupings in our customer base based on transaction history",
            "Identify user personas and behavioral clusters from activity logs",
            "Divide our accounts into clusters by usage patterns",
            "Unsupervised grouping of client behavior",
            "Find archetypes and customer segments",
        ]
        for q in queries:
            detected = self.kind.detect(q)
            self.assertIsNotNone(detected, f"Failed to detect segmentation for query: {q}")

    def test_disambiguates_predictive_queries(self):
        predictive_queries = [
            "Predict churn for each customer segment",
            "Forecast whether users in group A will cancel",
            "Estimate default probability per persona",
            "Predict which clients will buy next month",
        ]
        for q in predictive_queries:
            detected = self.kind.detect(q)
            self.assertIsNone(detected, f"Should reject predictive query as segmentation: {q}")

    def test_disambiguates_simple_aggregations(self):
        aggregation_queries = [
            "How many users by region",
            "Count of transactions per store",
            "Sum of sales by state",
        ]
        for q in aggregation_queries:
            detected = self.kind.detect(q)
            self.assertIsNone(detected, f"Should reject simple aggregation query as segmentation: {q}")

    def test_disambiguates_fraud_and_anomalies(self):
        anomaly_queries = [
            "Detect fraudulent payments in our data",
            "Spot anomalous transactions",
            "Find rare fraud events",
        ]
        for q in anomaly_queries:
            detected = self.kind.detect(q)
            self.assertIsNone(detected, f"Should reject fraud/anomaly query as segmentation: {q}")

    def test_profile_driven_detection_on_neutral_query(self):
        # Ambiguous prompt without explicit 'segment' word, but table profile has no target and behavioral metrics
        problem = "Explore patterns and understand how our users divide based on their activity"
        profile_no_target = {
            "columns": ["user_id", "session_count", "total_spend", "avg_duration"],
            "target": None,
        }
        detected = self.kind.detect(problem, profile_no_target)
        self.assertIsNotNone(detected)
        self.assertIn("no target column", detected)


class TestHierarchicalModelAndPipeline(unittest.TestCase):
    def test_hierarchical_model_fit_and_predict(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(loc=[-4, -4], size=(50, 2)),
            rng.normal(loc=[4, 4], size=(50, 2)),
        ])
        model = HierarchicalModel(k=2).fit(X)
        self.assertIsNotNone(model.cluster_centers_)
        self.assertEqual(model.cluster_centers_.shape, (2, 2))

        # Predict on new points
        new_points = np.array([[-4.0, -4.0], [4.0, 4.0]])
        preds = model.predict(new_points)
        self.assertEqual(len(preds), 2)
        self.assertNotEqual(preds[0], preds[1])

    def test_end_to_end_segmentation_includes_strategy(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "recency": rng.uniform(1, 30, 150),
            "frequency": rng.poisson(5, 150),
            "spend": rng.exponential(50, 150),
        })
        res = segment.segment(df, ks=[2, 3], resamples=5, sample=100)
        self.assertIn("strategy_recommendation", res)
        self.assertIn("clusterability", res)
        self.assertIsNotNone(res["clusterability"])
        self.assertIn("hopkins", res["clusterability"])


if __name__ == "__main__":
    unittest.main()
