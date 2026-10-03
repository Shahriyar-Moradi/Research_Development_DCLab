import json
import unittest

from dclab_rnd.notebook_assist import MANIFEST_CANDIDATES, manifest_path
from dclab_rnd.notebook_eval import evaluate

MISSING = f"frozen manifest not committed yet: {MANIFEST_CANDIDATES[0].name}"


@unittest.skipIf(manifest_path() is None, MISSING)
class NotebookEvaluationTests(unittest.TestCase):
    def test_frozen_real_notebooks_show_known_miss_and_proof_gap(self):
        manifest = json.loads(manifest_path().read_text())
        rows = evaluate(manifest)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(len(rows), 7)
        self.assertFalse(by_id["hyperack_full_data_clustering"]["pass"])
        self.assertTrue(by_id["hyperack_scale_before_split"]["pass"])
        self.assertFalse(by_id["churn_cancellation_date_availability"]["proof_rule_pass"])
        self.assertFalse(by_id["churn_cancellation_date_availability"]["safe_wording"])

    def test_changed_cell_is_not_silently_scored(self):
        manifest = json.loads(manifest_path().read_text())
        manifest["cases"][0]["cell_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Frozen cell changed"):
            evaluate(manifest)
