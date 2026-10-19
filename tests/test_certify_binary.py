"""Package 30.19: the binary kind certified on the recipes' two real datasets (Telco, HyperAck), each with its own traps.

The kind the engine runs without a machine of its own is certified on its replica's seeds and on the real datasets ``REAL_FOR`` names;
each real case carries its traps (Telco's identifier, HyperAck's fares paid after the order), and detection routes each case to the
kind from its own data card. The full run (the engine and the code export on every case) is the campaign's record, CRT-003.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
except ImportError as error:  # .venv-agent
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class BinaryCertificationTest(unittest.TestCase):
    def test_binary_is_certified_on_its_replica_and_both_real_datasets_with_their_traps(self):
        from dclab_rnd import certify

        entries = certify.entries_for("binary", seeds=(0,))
        names = [name for name, *_ in entries]
        self.assertEqual([n for n, source, *_ in entries if source == "replica"], ["binary replica s0"])
        self.assertIn("binary telco", names)
        self.assertIn("binary hyperack", names)
        traps = {name: entry["traps"] for name, _, _, entry in entries if entry is not None}
        self.assertEqual(traps["binary telco"], {"identifiers": ["customerID"]})
        self.assertEqual(traps["binary hyperack"], {"blocked": ["final_customer_fare", "final_biker_fare"]})
        self.assertFalse(any("telco" in n for n, *_ in certify.entries_for("matching", seeds=(0,))))  # REAL_FOR is the binary kind's only

    def test_detection_routes_the_real_churn_table_to_binary_from_its_own_card(self):
        from dclab_rnd import certify, kinds

        telco = next(case for name, _, case, _ in certify.entries_for("binary", seeds=()) if name == "binary telco")
        self.assertTrue(certify._routed(kinds.get("binary"), telco))


if __name__ == "__main__":
    unittest.main()
