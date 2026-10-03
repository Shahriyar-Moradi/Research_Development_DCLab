import json
from pathlib import Path
import tempfile
import unittest

from dclab_rnd.replay_compare import compare
from dclab_rnd.system_eval import _scores


class ReplayComparisonTests(unittest.TestCase):
    def test_equal_predictions_do_not_hide_code_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("old", "new")]
            rows = [{"row_id": 0, "repeat": 0, "fold": 0, "target": 0, "probability": 0.1},
                    {"row_id": 1, "repeat": 0, "fold": 0, "target": 1, "probability": 0.9}]
            scores = _scores(rows)
            for path in roots:
                path.mkdir()
                (path / "oof_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
                (path / "result.json").write_text(json.dumps({"data_hashes": {"x": "fixed"},
                    "split_hash": "same", "metrics": scores, "folds": [{"repeat": 0, "fold": 0, **scores}]}))
                (path / "recipe.json").write_text(json.dumps({"plan": {"model": "dummy"},
                    "worker_hash": "old-version", "catalog_hash": "old-version"}))
            report = compare(*roots)
            self.assertTrue(report["same_oof_predictions_bytes"])
            self.assertTrue(all(f["metrics_match"] for f in report["candidate_folds_independently_recomputed"]))
            self.assertFalse(report["current_code_matches_reference"])
            self.assertFalse(report["exact_version_replay"])
