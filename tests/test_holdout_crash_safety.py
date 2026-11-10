"""Unit tests for holdout crash-safety, atomic receipts, and split governance."""

import json
from pathlib import Path
import tempfile
import unittest

from dclab_core.holdout import (
    check_receipt,
    mark_consumed,
    read_receipt,
    record_intent,
    split_fingerprint,
    verify_split,
)


class HoldoutCrashSafetyTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.receipt_path = self.dir_path / "holdout_receipt.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_fingerprint_deterministic_and_canonical(self):
        train_a = [10, 2, 5, 8]
        train_b = [2, 5, 8, 10]
        holdout = [1, 9, 3]

        fp_a = split_fingerprint(train_a, holdout)
        fp_b = split_fingerprint(train_b, holdout)
        self.assertEqual(fp_a, fp_b, "Sorted canonical order must yield identical fingerprint")

        # Modifying one index alters fingerprint
        fp_diff = split_fingerprint([10, 2, 5, 7], holdout)
        self.assertNotEqual(fp_a, fp_diff)

    def test_verify_split(self):
        fp = split_fingerprint([1, 2, 3], [4, 5])
        verify_split([1, 2, 3], [4, 5], fp)  # Should not raise

        with self.assertRaises(ValueError):
            verify_split([1, 2, 999], [4, 5], fp)

    def test_atomic_intent_and_consumption_lifecycle(self):
        fp = split_fingerprint([1, 2], [3])
        receipt = record_intent(self.receipt_path, project_id="p-12345", split_fingerprint=fp, holdout_uses=1)
        self.assertTrue(self.receipt_path.exists())
        self.assertFalse(receipt.consumed)
        self.assertIsNone(receipt.consumed_at)

        # File on disk matches
        disk_receipt = read_receipt(self.receipt_path)
        self.assertIsNotNone(disk_receipt)
        self.assertEqual(disk_receipt.receipt_id, receipt.receipt_id)
        self.assertFalse(disk_receipt.consumed)

        # Mark consumed
        consumed_receipt = mark_consumed(self.receipt_path, receipt)
        self.assertTrue(consumed_receipt.consumed)
        self.assertIsNotNone(consumed_receipt.consumed_at)

        disk_consumed = read_receipt(self.receipt_path)
        self.assertTrue(disk_consumed.consumed)

    def test_crash_safety_blocks_replay_without_reuse_reason(self):
        fp = split_fingerprint([10, 20], [30])
        # Simulate crash: intent was recorded, but process crashed before mark_consumed
        record_intent(self.receipt_path, project_id="p-crash", split_fingerprint=fp, holdout_uses=1)

        # Attempting to re-run without reuse_reason must be blocked
        allowed, reason = check_receipt(self.receipt_path, current_fingerprint=fp, reuse_reason=None)
        self.assertFalse(allowed)
        self.assertIn("crash", reason.lower())

        # With explicit reuse_reason, re-run is allowed
        allowed_retry, _ = check_receipt(self.receipt_path, current_fingerprint=fp, reuse_reason="recovered from worker reboot")
        self.assertTrue(allowed_retry)

    def test_consumed_holdout_blocks_replay_without_reuse_reason(self):
        fp = split_fingerprint([1, 2, 3], [4, 5])
        receipt = record_intent(self.receipt_path, project_id="p-done", split_fingerprint=fp, holdout_uses=1)
        mark_consumed(self.receipt_path, receipt)

        # Re-running consumed holdout without reason is blocked
        allowed, reason = check_receipt(self.receipt_path, current_fingerprint=fp, reuse_reason=None)
        self.assertFalse(allowed)
        self.assertIn("already been consumed", reason)

        # Re-running with reuse_reason is allowed
        allowed_reuse, _ = check_receipt(self.receipt_path, current_fingerprint=fp, reuse_reason="owner approved model revision")
        self.assertTrue(allowed_reuse)

    def test_split_drift_detected_and_rejected(self):
        fp_original = split_fingerprint([1, 2, 3], [4, 5])
        record_intent(self.receipt_path, project_id="p-drift", split_fingerprint=fp_original, holdout_uses=1)

        fp_modified = split_fingerprint([1, 2, 3, 4], [5])
        allowed, reason = check_receipt(self.receipt_path, current_fingerprint=fp_modified, reuse_reason="rerun")
        self.assertFalse(allowed)
        self.assertIn("split drift detected", reason.lower())


if __name__ == "__main__":
    unittest.main()
