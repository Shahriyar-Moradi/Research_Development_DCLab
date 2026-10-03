import importlib.util
import unittest as _ut

if importlib.util.find_spec("dclab_rnd.system_eval") is None:
    raise _ut.SkipTest("module not committed yet: dclab_rnd/system_eval.py")

"""Independent scorecard checks against a small controlled evidence archive."""

import hashlib
import json
from pathlib import Path
from statistics import mean
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dclab_rnd.system_eval import METRICS, _scores, audit_run, wilson_interval


class SystemEvaluationTests(unittest.TestCase):
    def test_wilson_interval_counts_attempts(self):
        self.assertIsNone(wilson_interval(0, 0))
        self.assertLess(wilson_interval(3, 6)[0], 0.5)
        self.assertGreater(wilson_interval(3, 6)[1], 0.5)

    def test_saved_predictions_expose_false_repeat_summary_and_blocked_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run1"
            trial = root / "trial-001"
            trial.mkdir(parents=True)
            source = Path(directory) / "dummy.parquet"
            source.write_bytes(b"fixed source")
            hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
            X = pd.DataFrame({"balance": list(range(6))})
            y = pd.Series([0, 1, 0, 1, 0, 1])
            row_ids = np.arange(6)
            probabilities = [.1, .9, .4, .6, .7, .3]
            oof = []
            folds = []
            digest = hashlib.sha256()
            for repeat in range(2):
                for fold in range(3):
                    indices = [fold * 2, fold * 2 + 1]
                    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
                    group = []
                    for i in indices:
                        record = {"row_id": i, "repeat": repeat, "fold": fold,
                                  "target": int(y.iloc[i]), "probability": probabilities[i]}
                        oof.append(record)
                        group.append(record)
                    folds.append({"repeat": repeat, "fold": fold, **_scores(group)})
            plan = {"dataset": "bank_marketing", "model": "logistic_regression"}
            result = {
                "dataset": "bank_marketing", "plan": plan, "rows": 6, "folds": folds,
                "repeat_summary": [{"repeat": repeat, **_scores([r for r in oof if r["repeat"] == repeat])} for repeat in range(2)],
                "metrics": {name: mean(f[name] for f in folds) for name in METRICS},
                "split_hash": digest.hexdigest(), "data_hashes": hashes,
                "pipeline_source_columns": ["balance"], "retained_columns": ["balance"],
                "excluded_by_policy": ["duration"], "production_approved": False,
            }
            recipe = {"plan": plan, "max_rows": 6, "repeats": 2, "data_hashes": hashes}
            (trial / "result.json").write_text(json.dumps(result))
            (trial / "recipe.json").write_text(json.dumps(recipe))
            (trial / "oof_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in oof))
            trajectory = {"training_ready": False, "run": {"id": "run1", "status": "completed"},
                          "events": [
                              {"kind": "proposal", "seq": 1, "payload": {"evidence_ids": []}},
                              {"kind": "trial", "seq": 2, "payload": {"id": "run1:trial-001", "status": "completed", "plan": plan}},
                              {"kind": "critique", "seq": 3, "payload": {"evidence_ids": ["run1:trial-001"]}},
                              {"kind": "synthesis", "seq": 4, "payload": {"lessons": [{"evidence_ids": ["run1:trial-001"]}]}},
                          ]}
            (root / "trajectory.json").write_text(json.dumps(trajectory))
            with patch("dclab_rnd.system_eval.load", return_value=(X, y, [], row_ids, hashes)), patch(
                "dclab_rnd.system_eval.source_paths", return_value=[source]
            ):
                clean = audit_run(root)
                self.assertEqual(clean["issues"], [])
                self.assertTrue(clean["trials"][0]["passed"])
                for item in result["repeat_summary"]:
                    item.update({name: mean(f[name] for f in folds if f["repeat"] == item["repeat"]) for name in METRICS})
                result["pipeline_source_columns"] = ["balance", "duration"]
                (trial / "result.json").write_text(json.dumps(result))
                corrupted = audit_run(root)
                codes = {i["code"] for i in corrupted["issues"]}
                self.assertIn("repeat_summary_uses_fold_mean", codes)
                self.assertIn("blocked_feature_used", codes)
                self.assertFalse(corrupted["trials"][0]["passed"])
                self.assertEqual(len(corrupted["trials"][0]["repeat_corrections"]), 2)
                trajectory["events"][2]["payload"]["evidence_ids"] = ["invented:trial-999"]
                (root / "trajectory.json").write_text(json.dumps(trajectory))
                self.assertIn("unsupported_citation", {i["code"] for i in audit_run(root)["issues"]})


if __name__ == "__main__":
    unittest.main()
