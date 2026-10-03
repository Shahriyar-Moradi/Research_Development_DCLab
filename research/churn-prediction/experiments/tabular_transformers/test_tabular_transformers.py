"""Lightweight tests for tabular Transformer churn pipeline (no full DL training)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from saint_tabular import SaintClassifier, SaintConfig, SaintPreprocessor
from tabular_transformer_churn import (
    CATEGORICAL_COLS,
    DEFAULT_CSV,
    NUMERICAL_COLS,
    TARGET_COL,
    evaluate_probs,
    load_and_prepare,
    split_frames,
)


class TestTelcoTabularTransformers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_and_prepare(DEFAULT_CSV)

    def test_csv_exists(self):
        self.assertTrue(Path(DEFAULT_CSV).is_file())

    def test_prepare_fills_total_charges(self):
        self.assertEqual(int(self.data["TotalCharges"].isna().sum()), 0)
        self.assertTrue(set(self.data[TARGET_COL].unique()) <= {"Yes", "No"})
        self.assertEqual(self.data["SeniorCitizen"].dtype, object)

    def test_split_is_stratified_and_complete(self):
        train, val, test = split_frames(self.data, seed=42)
        self.assertEqual(len(train) + len(val) + len(test), len(self.data))
        overall = self.data[TARGET_COL].eq("Yes").mean()
        for part in (train, val, test):
            self.assertAlmostEqual(part[TARGET_COL].eq("Yes").mean(), overall, delta=0.03)

    def test_saint_forward_shape(self):
        train, _, _ = split_frames(self.data, seed=42)
        pre = SaintPreprocessor(CATEGORICAL_COLS, NUMERICAL_COLS).fit(train.head(64))
        x_cat, x_num = pre.transform(train.head(16))
        model = SaintClassifier(pre.cardinalities, x_num.shape[1], SaintConfig(dim=16, depth=1, num_heads=2))
        logits = model(torch.as_tensor(x_cat), torch.as_tensor(x_num), apply_intersample=True)
        self.assertEqual(tuple(logits.shape), (16, 2))

    def test_evaluate_probs(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = evaluate_probs(y, p, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["roc_auc"], 0.99)


if __name__ == "__main__":
    unittest.main()
