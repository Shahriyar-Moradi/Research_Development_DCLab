"""The owner's decision on a provider model's outputs (2026-10-25): stored for evaluation and for training, silver and apart.

A set of Claude's or GPT's outputs takes its consent from ``dclab_rnd/store/owner_decisions.json``; without a recorded yes
it is evaluation only (``improve``), which an export for training refuses.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProviderOutputsTest(unittest.TestCase):
    def test_the_recorded_decision_sets_the_consent(self):
        from dclab_rnd.store import provider_outputs

        d = Path(tempfile.mkdtemp())
        yes, no, bad = d / "yes.json", d / "no.json", d / "bad.json"
        yes.write_text(json.dumps({"provider_outputs": {"for_training": True}}), encoding="utf-8")
        no.write_text(json.dumps({"provider_outputs": {"for_training": False}}), encoding="utf-8")
        bad.write_text("{", encoding="utf-8")
        self.assertEqual(provider_outputs.consent(yes), "train")
        self.assertEqual(provider_outputs.consent(no), "improve")
        self.assertEqual(provider_outputs.consent(bad), "improve")  # no readable yes: evaluation only
        self.assertEqual(provider_outputs.consent(d / "missing.json"), "improve")

    def test_the_owners_record_says_yes_with_its_conditions(self):
        from dclab_rnd.store import provider_outputs

        record = json.loads(provider_outputs.DECISIONS.read_text(encoding="utf-8"))["provider_outputs"]
        self.assertTrue(record["for_training"])
        self.assertEqual(record["decided"], "2026-10-25")
        self.assertIn("silver", record["kept"])
        self.assertIn("terms", record["check_before_use"])


if __name__ == "__main__":
    unittest.main()
