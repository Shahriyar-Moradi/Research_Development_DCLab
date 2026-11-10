"""Unit tests for accurate and robust clustering / KMeans in dclab_core.segment.

Tests:
- k-means++ convergence and reproducibility
- Multi-metric evaluation (Davies-Bouldin, Calinski-Harabasz, Silhouette, Inertia)
- Degenerate cluster rejection (min_size threshold)
- Inverted unscaled centroids recovery in original feature units
- Row-level assignment confidence scoring in [0, 1]
- Leakage-safe holdout transform & assignment
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import segment  # noqa: E402


class ClusteringRobustnessTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        # Create 3 well-separated clusters
        c1 = rng.normal(loc=[10.0, 100.0], scale=[1.0, 10.0], size=(100, 2))
        c2 = rng.normal(loc=[30.0, 500.0], scale=[1.5, 20.0], size=(120, 2))
        c3 = rng.normal(loc=[60.0, 1000.0], scale=[2.0, 30.0], size=(110, 2))
        data = np.vstack([c1, c2, c3])
        self.frame = pd.DataFrame(data, columns=["frequency", "spend"])

    def test_kmeanspp_reproducibility(self):
        X, _ = segment.prepare(self.frame)
        m1 = segment._fit("kmeans", 3, X, seed=42)
        m2 = segment._fit("kmeans", 3, X, seed=42)
        np.testing.assert_allclose(m1.cluster_centers_, m2.cluster_centers_)

    def test_multi_metric_evaluation(self):
        result = segment.segment(self.frame, ks=[2, 3, 4], resamples=6, seed=42)
        chosen = result.get("chosen")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["k"], 3)
        self.assertEqual(chosen["method"], "kmeans")

        # Verify multi-metric evaluation presence
        self.assertIn("silhouette", chosen)
        self.assertIn("davies_bouldin", chosen)
        self.assertIn("calinski_harabasz", chosen)
        self.assertIn("inertia", chosen)
        self.assertIn("mean_assignment_confidence", chosen)

        self.assertGreater(chosen["silhouette"], 0.5)
        self.assertGreater(chosen["calinski_harabasz"], 100.0)
        self.assertLess(chosen["davies_bouldin"], 1.0)  # Lower is better for DB
        self.assertGreater(chosen["mean_assignment_confidence"], 0.4)

    def test_unscaled_centroids_recovery(self):
        result = segment.segment(self.frame, ks=[3], resamples=4, seed=42)
        centroids = result.get("unscaled_centroids")
        self.assertIsNotNone(centroids)
        self.assertEqual(len(centroids), 3)

        # Cluster means in raw units should approximate ~10, ~30, ~60 for frequency
        freq_centroids = sorted([v["frequency"] for v in centroids.values()])
        self.assertAlmostEqual(freq_centroids[0], 10.0, delta=2.0)
        self.assertAlmostEqual(freq_centroids[1], 30.0, delta=3.0)
        self.assertAlmostEqual(freq_centroids[2], 60.0, delta=4.0)

        # Profiles should contain unscaled centroids and mean comparisons
        profiles = result.get("profiles", [])
        self.assertEqual(len(profiles), 3)
        for p in profiles:
            self.assertIn("unscaled_centroids", p)
            self.assertIn("sets_it_apart", p)
            apart = p["sets_it_apart"]
            self.assertTrue(any("segment_mean" in item for item in apart))

    def test_assignment_confidence_scoring(self):
        X, record = segment.prepare(self.frame)
        model = segment._fit("kmeans", 3, X, seed=42)
        conf = segment.assignment_confidence(model, X)

        self.assertEqual(len(conf), len(self.frame))
        self.assertTrue(np.all(conf >= 0.0) and np.all(conf <= 1.0))
        # Well-separated cluster centres should have high mean assignment confidence
        self.assertGreater(float(np.mean(conf)), 0.45)

    def test_degenerate_cluster_rejection(self):
        # 100 points in one tight cluster, 2 solitary outlier points far away
        rng = np.random.default_rng(99)
        cluster = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(100, 2))
        outliers = np.array([[50.0, 50.0], [55.0, 55.0]])
        imbalanced = pd.DataFrame(np.vstack([cluster, outliers]), columns=["f1", "f2"])

        # With k=2, the 2 outlier points form a degenerate cluster (size 2 < min threshold max(5, 0.01*102)=5)
        found = segment.search(imbalanced, ["f1", "f2"], ks=[2], resamples=4, seed=0)
        k2_info = found["final"]["table"].get("kmeans:2")
        self.assertIsNotNone(k2_info)
        self.assertFalse(k2_info["balanced"])
        self.assertFalse(k2_info["separated"])

    def test_leakage_safe_holdout_transform(self):
        train_df = self.frame.iloc[:200]
        holdout_df = self.frame.iloc[200:]

        # Prepare strictly on train
        X_train, record = segment.prepare(train_df)
        model = segment._fit("kmeans", 3, X_train, seed=42)

        # Transform holdout using train record (no centroid re-fit)
        X_hold = segment.transform(holdout_df, record)
        preds = model.predict(X_hold)
        conf = segment.assignment_confidence(model, X_hold)

        self.assertEqual(len(preds), len(holdout_df))
        self.assertEqual(len(conf), len(holdout_df))
        self.assertTrue(set(preds).issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
