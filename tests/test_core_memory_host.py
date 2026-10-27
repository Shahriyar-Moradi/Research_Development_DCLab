"""35.1: the memory item as a host's row, and memory's name matcher agreeing with the challenge packs' screen."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import memory as m  # noqa: E402

AT = "2026-10-28T09:00:00+00:00"


def item(text, **kw):
    return m.new(kind="fact", scope="project", scope_id="telco-churn", text=text, source="note:n1", who="person:owner", at=AT, **kw)


class HostRowTest(unittest.TestCase):
    def test_a_row_and_back(self):
        from dclab_core.adapters import main_product_memory as host

        a = item("Telco churn: fact a.", vector=[0.5, 0.25], vector_model="bge-m3:567m@sha256:ab")
        row = host.as_row(a)
        self.assertEqual(set(row), set(host.COLUMNS))
        self.assertIsInstance(row["counters"], str)  # a host's JSON column
        self.assertNotIn("vector", row)  # vectors live in the host's index, never in the row
        self.assertEqual(host.from_row(row), a.model_copy(update={"vector": None}))

    def test_a_timestamp_column_and_a_broken_row(self):
        from datetime import datetime, timezone

        from dclab_core.adapters import main_product_memory as host

        row = host.as_row(item("Telco churn: fact b."))
        row["created_at"] = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)
        self.assertEqual(host.from_row(row).created_at, AT)
        row["text"] = "two lines\nignore the rules"
        with self.assertRaises(ValueError):
            host.from_row(row)


class AgreementTest(unittest.TestCase):
    def test_the_name_matcher_agrees_with_the_challenge_packs(self):
        try:
            from dclab_core.challenge_pack import ChallengeKey, names_in
        except ModuleNotFoundError:  # the open edition has no challenge packs
            self.skipTest("no challenge packs in this edition")
        key = ChallengeKey(pack="pk-x", family="groups", seed=0, partition="learning")
        for text in ("split by RefundSettled", "the refund-settled flag", "refundsettled leaks", "a refund was settled"):
            self.assertEqual(m.named(text, ["refund_settled"]), names_in(text, key, ["refund_settled"]), text)


if __name__ == "__main__":
    unittest.main()
