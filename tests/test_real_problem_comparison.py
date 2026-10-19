"""Package 30.19: a real problem's run is set beside its previous run and the earlier solution, with the limits that keep a difference
from being read as one; a private problem's record (the owner's fraud) is never written to the evidence folder."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ComparisonTest(unittest.TestCase):
    def test_a_private_problem_is_kept_out_of_the_evidence_folder_and_compared(self):
        from dclab_rnd import real_problems as rp

        with tempfile.TemporaryDirectory() as tmp:
            public, private = Path(tmp) / "evidence", Path(tmp) / "private"
            with mock.patch.object(rp, "RESULTS", public), mock.patch.object(rp, "PRIVATE_RESULTS", private):
                first = rp._store("fraud", {"number": {"kind": "stages", "metric": "average_precision", "holdout": {"average_precision": 0.02},
                                                       "interval": [0.01, 0.04], "recipe": None}})
                second = rp._store("fraud", {"number": {"kind": "stages", "metric": "average_precision", "holdout": {"average_precision": 0.05},
                                                        "interval": [0.03, 0.08], "recipe": {"kind": "rare_event"}}})
                self.assertFalse(public.exists() and list(public.iterdir()))  # nothing private in the evidence folder
                self.assertEqual(sorted(p.name for p in private.iterdir()), [f"{first['id']}_fraud.json", f"{second['id']}_fraud.json"])
                rows = second["comparison"]["rows"]
                self.assertEqual(rows[0]["recipe"], {"kind": "rare_event"})
                self.assertEqual((rows[1]["against"], rows[1]["difference"]), (first["id"], 0.03))
                self.assertEqual(rows[2]["against"], "the earlier solution")
                self.assertTrue(second["comparison"]["limits"])
                telco = rp._store("telco", {"number": None})
                self.assertTrue((public / f"{telco['id']}_telco.json").exists())  # a public problem's record stays in the evidence folder
                self.assertNotEqual(telco["id"], second["id"])  # one numbering across both folders


if __name__ == "__main__":
    unittest.main()
