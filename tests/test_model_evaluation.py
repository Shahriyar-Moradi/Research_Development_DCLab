"""A model's evaluation (the owner, 2026-10-20: precision, recall, F1, the other important measures and a confusion
matrix): measures and curves from scores alone, the confusion counts at every cutoff, every list bounded."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core.model_evaluation import BUDGETS, CUTOFFS, MAX_POINTS, Evaluation, at_cutoff, evaluate, from_record, review_budgets  # noqa: E402


class ModelEvaluationTest(unittest.TestCase):
    def test_binary_measures_match_sklearn_and_every_cutoff_has_its_confusion(self):
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 5000)
        s = np.clip(0.3 * y + rng.random(5000) * 0.7, 0, 1)
        ev = evaluate("binary_imbalanced", y, s, rows="holdout", cutoff=0.4)
        flags = (s >= 0.4).astype(int)
        self.assertAlmostEqual(ev.metrics["precision"], precision_score(y, flags), places=5)
        self.assertAlmostEqual(ev.metrics["recall"], recall_score(y, flags), places=5)
        self.assertAlmostEqual(ev.metrics["f1"], f1_score(y, flags), places=5)
        self.assertAlmostEqual(ev.metrics["roc_auc"], roc_auc_score(y, s), places=5)
        self.assertEqual(sum(ev.confusion[k] for k in ("tp", "fp", "fn", "tn")), 5000)
        self.assertEqual(len(ev.cutoffs), len(CUTOFFS))  # 0.4 is on the grid
        row = next(r for r in ev.cutoffs if r["t"] == 0.4)
        self.assertEqual(at_cutoff(row)["precision"], ev.metrics["precision"])
        self.assertLessEqual(len(ev.curves["roc"]), MAX_POINTS)
        self.assertLessEqual(len(ev.curves["gains"]), MAX_POINTS)
        self.assertEqual(sum(ev.curves["histogram"]["positive"]) + sum(ev.curves["histogram"]["negative"]), 5000)

    def test_review_budgets_are_exact_from_the_ranked_rows(self):
        rng = np.random.default_rng(3)
        y = (rng.random(2000) < 0.12).astype(int)
        s = np.clip(0.35 * y + rng.random(2000) * 0.65, 0, 1)
        ev = evaluate("binary_imbalanced", y, s, rows="holdout")
        rows = review_budgets(ev)
        self.assertEqual([r["share"] for r in rows], list(BUDGETS))
        order = np.argsort(-s, kind="stable")
        for r in rows:
            k = int(round(r["share"] * 2000))
            self.assertEqual(r["flagged"], k)
            self.assertEqual(r["caught"], int(y[order][:k].sum()))
            self.assertAlmostEqual(r["recall"], r["caught"] / y.sum(), places=5)
            self.assertAlmostEqual(r["precision"], r["caught"] / k, places=5)
            self.assertEqual(r["source"], "scores")
        self.assertGreater(rows[0]["lift"], 1.5, "a model with signal flags the cases first")
        self.assertEqual([r["recall"] for r in rows], sorted(r["recall"] for r in rows), "a bigger budget never catches fewer")

    def test_review_budgets_of_a_kept_evaluation_without_them_come_from_its_gains_curve(self):
        # A project built before the budgets were kept: its gains curve at 1 % steps (1,600 rows, 204 cases)
        gains = [[0.0, 0.0]] + [[round(i / 100, 2), min(1.0, round(3.5 * i / 100, 6))] for i in range(1, 101)]
        kept = Evaluation(kind="binary", rows="holdout", n=1600, metrics={"positive_rate": 0.1275}, curves={"gains": gains})
        rows = review_budgets(kept)
        ten = next(r for r in rows if r["share"] == 0.1)
        self.assertEqual((ten["flagged"], ten["caught"], ten["source"]), (160, 71, "gains"))  # 0.35 x 204 = 71.4
        self.assertAlmostEqual(ten["precision"], 71 / 160, places=5)
        self.assertEqual(review_budgets(Evaluation(kind="binary", rows="holdout", n=1600, metrics={"positive_rate": 0.1275})), [],
                         "no curve, no budgets: nothing is made up")
        self.assertEqual(review_budgets(Evaluation(kind="regression", rows="holdout", n=10)), [])

    def test_one_class_is_said_never_a_crash(self):
        ev = evaluate("binary_imbalanced", np.zeros(50), np.linspace(0, 1, 50), rows="validation")
        self.assertIsNone(ev.metrics["roc_auc"])
        self.assertTrue(ev.notes)

    def test_multiclass_per_class_and_confusion_matrix(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 600)
        proba = rng.random((600, 3)); proba[np.arange(600), y] += 0.8
        ev = evaluate("multiclass", y, proba / proba.sum(1, keepdims=True), rows="holdout", labels=["a", "b", "c"])
        self.assertEqual([c["label"] for c in ev.per_class], ["a", "b", "c"])
        self.assertEqual(sum(map(sum, ev.confusion["matrix"])), 600)
        self.assertEqual(sum(c["support"] for c in ev.per_class), 600)

    def test_regression_measures_and_bins(self):
        rng = np.random.default_rng(2)
        y = rng.normal(10, 2, 400)
        ev = evaluate("regression", y, y + rng.normal(0, 1, 400), rows="validation")
        self.assertGreater(ev.metrics["r2"], 0.5)
        self.assertEqual(sum(b["count"] for b in ev.curves["by_prediction"]), 400)
        self.assertEqual(sum(ev.curves["residuals"]["counts"]), 400)

    def test_an_older_record_gives_back_exact_confusion_counts_from_its_own_numbers(self):
        # 1,600 rows, 123 cases; the record kept the cases' share, the cutoffs' alerts and true positives, precision and recall
        evidence = {"holdout_rows": 1600,
                    "holdout_metrics": {"positive_rate": 123 / 1600, "recall": 6 / 123, "precision": 6 / 13, "threshold": 0.5, "f1": 0.088, "accuracy": 0.9225, "roc_auc": 0.72},
                    "threshold_analysis": [{"threshold": 0.1, "alerts": 340, "true_positives": 63}, {"threshold": 0.5, "alerts": 13, "true_positives": 6}],
                    "calibration_bins": [{"lower": 0.0, "upper": 0.2, "count": 1463, "predicted": 0.04, "observed": 0.06}]}
        ev = from_record(evidence)
        self.assertEqual((ev.kind, ev.source, ev.n), ("binary", "record", 1600))
        self.assertEqual(ev.confusion["tp"], 6)
        self.assertEqual(ev.confusion["fp"], 7)
        self.assertEqual(ev.confusion["fn"], 117)
        self.assertEqual(sum(ev.confusion[k] for k in ("tp", "fp", "fn", "tn")), 1600)
        self.assertEqual([c["t"] for c in ev.cutoffs], [0.1, 0.5])
        self.assertIsNotNone(ev.metrics["specificity"])
        self.assertTrue(ev.notes and "scores" in ev.notes[0])
        self.assertIsNone(from_record({"holdout_metrics": {}}))

    def test_a_record_that_kept_the_full_evaluation_returns_it(self):
        y = np.array([0, 1] * 50); s = np.linspace(0, 1, 100)
        full = evaluate("binary", y, s, rows="holdout").model_dump(mode="json")
        ev = from_record({"evaluation": {"holdout": full}, "holdout_metrics": {"roc_auc": 0.5}})
        self.assertEqual(ev.source, "scores")
        self.assertIn("roc", ev.curves)

    def test_a_multiclass_record_gives_each_class_its_results(self):
        evidence = {"holdout_rows": 20, "holdout_metrics": {"accuracy": 0.65, "macro_f1": 0.52, "per_class_f1": [0.75, 0.0, 0.8]},
                    "task_assessment": {"per_class": [{"class": "a", "recall": 1.0, "support": 11, "precision": 0.6111}, {"class": "b", "recall": 0.0, "support": 6, "precision": None}],
                                        "macro_recall": 0.55, "most_confused": [{"rows": 6, "true": "b", "predicted": "a", "share_of_true": 1.0}], "top_k_accuracy": {"top_2": 0.9}}}
        ev = from_record(evidence)
        self.assertEqual([c["label"] for c in ev.per_class], ["a", "b"])
        self.assertEqual(ev.per_class[1]["f1"], 0.0)
        self.assertEqual(ev.metrics["top_2_accuracy"], 0.9)

    def test_the_final_stage_keeps_the_holdouts_and_the_validation_folds_evaluation(self):
        import tempfile

        import pandas as pd

        from dclab_rnd.studio import ProjectStore, data, engine

        rng = np.random.default_rng(3)
        n = 600
        frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "segment": rng.choice(["x", "y", "z"], n)})
        frame["outcome"] = (frame["a"] + rng.normal(scale=1.0, size=n) > 0.8).astype(int)
        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("Eval", "general", "Predict the outcome")["id"]
        frame.to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        project = store.get(pid)
        project["solution"] = {"target": "outcome", "task": "binary", "positive_label": "1", "prediction_moment": "At the time of the visit.", "forbidden": [],
                               "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": "roc_auc", "notes": ""}
        project["settings"]["quick"] = True
        store.save(project)
        for stage in ("data", "leakage", "features", "models", "final"):
            engine.execute(store, pid, stage, "human")
        evidence = store.read_stage(pid, "final")["evidence"]
        kept = evidence["evaluation"]
        holdout, validation = from_record(evidence), kept["validation"]
        self.assertEqual(holdout.source, "scores")
        self.assertEqual(holdout.n, evidence["holdout_rows"])
        self.assertEqual(sum(holdout.confusion[k] for k in ("tp", "fp", "fn", "tn")), evidence["holdout_rows"])
        self.assertAlmostEqual(holdout.metrics["roc_auc"], evidence["holdout_metrics"]["roc_auc"], places=4)  # the same number the stage recorded
        self.assertEqual(validation["rows"], "validation")
        self.assertEqual(validation["n"], evidence["train_rows"])
        self.assertIn("roc", holdout.curves)
        self.assertNotIn("segment", str(kept))  # counts and curves only: no column's value is in it


if __name__ == "__main__":
    unittest.main()
