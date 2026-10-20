"""Package 32.20: the experiments themselves.

A project's models stage yields one experiment record per model fitted, keyed to its table by the table's hash, its shape in
bands and counts (never a row or a column's values), every fold's scores, and the holdout's metrics only for the model
chosen; the records are a set of the M3 model type.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class ExperimentsTest(unittest.TestCase):
    def test_every_model_fitted_is_one_record_with_its_tables_shape_and_no_row(self):
        from dclab_rnd.store import Layout, Store, shapers
        from dclab_rnd.store.sets import experiments

        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(layout)
        project = {"id": "p1", "industry": "telecom", "data": {"rows": 7043, "columns": ["customerID", "tenure", "Churn"], "sha256": "a" * 64,
                                                                "profile": {"rows": 7043, "column_count": 3, "columns": [{"name": "tenure", "kind": "number", "preview": [1, 2]}]}}}
        models = {"stage": "models", "task": "binary", "primary_metric": "roc_auc", "elapsed_seconds": 11.8, "evidence": {
            "selected_model": "boosting", "model_results": [
                {"config_id": "logistic", "model": "logistic_regression", "model_params": {"C": 1.0}, "cv": "StratifiedKFold(5)", "folds": 5,
                 "metrics": {"roc_auc": {"mean": 0.84, "std": 0.01}}},
                {"config_id": "boosting", "model": "hist_gradient_boosting", "model_params": {"max_depth": 3}, "cv": "StratifiedKFold(5)", "folds": 5,
                 "metrics": {"roc_auc": {"mean": 0.85, "std": 0.01}}}]}}
        final = {"stage": "final", "evidence": {"holdout_metrics": {"roc_auc": 0.848}}}
        store.keep("workflows", "dev", "projects/p1/project.json", json.dumps(project).encode(), consent="train")
        store.keep("workflows", "dev", "projects/p1/stages/models.json", json.dumps(models).encode(), consent="train")
        store.keep("workflows", "dev", "projects/p1/stages/final.json", json.dumps(final).encode(), consent="train")
        shapers.shape(layout)
        rows = experiments.records(layout)
        self.assertEqual([r.model for r in rows], ["logistic_regression", "hist_gradient_boosting"])
        chosen = next(r for r in rows if r.selected)
        self.assertEqual((chosen.model, chosen.holdout), ("hist_gradient_boosting", {"roc_auc": 0.848}))
        self.assertEqual(next(r for r in rows if not r.selected).holdout, {})  # the holdout only for the model chosen
        self.assertEqual((chosen.table["rows"], chosen.table["sha256"], chosen.table["kinds"]), ("small", "a" * 64, {"number": 1}))
        self.assertNotIn("customerID", json.dumps([r.model_dump(mode="json") for r in rows]))  # no column's name or value of the table
        report = experiments.build(layout)
        self.assertIn("m3_ml_decisions/experiments/binary", report["sets"])


if __name__ == "__main__":
    unittest.main()
