"""T2 of the team workspace (2026-11-23): a correction of a team's answer names the report revision it is based on.

``dclab_core.team_work.correction_revision`` is the input revision of a corrected estimate: the snapshot, the corrected
field and its value, and the base report revision, so an explicit zero is its own revision (never the base snapshot) and
the same value on another base is another correction. ``read_correction`` reads one back. ``accept_correction`` is the
rule: a correction based on the current revision applies, one already applied is a duplicate, and one based on an older
revision is stale (a late correction never overwrites a newer answer).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import team_work as tw  # noqa: E402

SNAP = "dispatch:1042@18:00"


class CorrectionRevisionTests(unittest.TestCase):
    def test_the_revision_names_the_snapshot_the_field_the_value_and_the_base(self):
        self.assertEqual(tw.correction_revision(SNAP, "kitchen_minutes", 8, 1), f"{SNAP};kitchen_minutes=8@r1")

    def test_an_explicit_zero_is_its_own_revision_never_the_base_snapshot(self):
        zero = tw.correction_revision(SNAP, "kitchen_minutes", 0, 2)
        self.assertEqual(zero, f"{SNAP};kitchen_minutes=0@r2")
        self.assertNotEqual(zero, SNAP)

    def test_the_same_value_on_another_base_is_another_correction(self):
        self.assertNotEqual(tw.correction_revision(SNAP, "kitchen_minutes", 8, 1),
                            tw.correction_revision(SNAP, "kitchen_minutes", 8, 3))

    def test_it_reads_back_and_a_plain_snapshot_is_no_correction(self):
        self.assertEqual(tw.read_correction(tw.correction_revision(SNAP, "kitchen_minutes", 0, 2)), ("kitchen_minutes", "0", 2))
        self.assertIsNone(tw.read_correction(SNAP))
        self.assertIsNone(tw.read_correction(""))

    def test_a_bad_input_is_refused(self):
        for args in (("", "kitchen_minutes", 8, 1), (SNAP, "Kitchen minutes", 8, 1), (SNAP, "kitchen_minutes", None, 1),
                     (SNAP, "kitchen_minutes", 8, 0), (SNAP, "kitchen_minutes", "8;x", 1), (SNAP, "kitchen_minutes", True, 1),
                     ("x" * 120, "kitchen_minutes", 8, 1), ("a;b", "kitchen_minutes", 8, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tw.correction_revision(*args)


class AcceptCorrectionTests(unittest.TestCase):
    def test_a_correction_on_the_current_revision_applies(self):
        self.assertEqual(tw.accept_correction(1, 1), "apply")
        self.assertEqual(tw.accept_correction(3, 3), "apply")

    def test_a_correction_already_applied_is_a_duplicate_whatever_its_base(self):
        self.assertEqual(tw.accept_correction(2, 1, already_applied=True), "duplicate")
        self.assertEqual(tw.accept_correction(2, 2, already_applied=True), "duplicate")

    def test_a_correction_on_an_older_revision_is_stale(self):
        self.assertEqual(tw.accept_correction(3, 2), "stale_base")
        self.assertEqual(tw.accept_correction(4, 1), "stale_base")

    def test_a_base_that_does_not_exist_is_refused(self):
        for current, base in ((1, 2), (0, 1), (2, 0), (3, -1)):
            with self.subTest(current=current, base=base), self.assertRaises(ValueError):
                tw.accept_correction(current, base)

    def test_the_verdicts_are_the_ledgers(self):
        from dclab_rnd.ledger import capture

        self.assertEqual(tw.CORRECTION_VERDICTS, ("apply", "duplicate", "stale_base"))
        self.assertEqual(tw.CORRECTION_VERDICTS, capture.CORRECTION_VERDICTS)


if __name__ == "__main__":
    unittest.main()
