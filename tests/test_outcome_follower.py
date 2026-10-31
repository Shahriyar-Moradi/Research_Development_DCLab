"""Package 37.16 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): a call list's results land on the decisions that made it, as a
measured business outcome; a group under the floor is never measured; the inbox is followed on a schedule."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.ledger import follow  # noqa: E402


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)
        for point, value in (("split.strategy", "time"), ("metric.primary", "roc_auc")):
            ledger.decide(point, ("project", "p1"), answers=[ledger.answer("rule", "rule:test", value)], value_used=value, used="rule", project="p1")
        self.results = pd.DataFrame({"customer_id": [f"C{i}" for i in range(40)], "stayed": [1] * 15 + [0] * 5 + [1] * 8 + [0] * 12})

    def test_a_call_lists_results_land_on_every_decision_that_made_it(self):
        out = follow.list_result("p1", [f"C{i}" for i in range(20)], self.results, id_column="customer_id", outcome_column="stayed")
        self.assertEqual((out["rate_listed"], out["rate_others"], out["decisions"], out["verdict"]), (0.75, 0.4, 2, "well"))
        rows = self.book.outcomes(project="p1")
        self.assertEqual({r.record for r in rows}, {r.id for r in self.book.records(project="p1")})
        self.assertEqual({(r.metric, r.observed) for r in rows}, {("list_outcome_rate", "business")})

    def test_a_group_under_the_floor_is_never_measured(self):
        out = follow.list_result("p1", ["C0", "C1"], self.results, id_column="customer_id", outcome_column="stayed")
        self.assertFalse(out["written"])
        self.assertIn("too few", out["error"])
        self.assertEqual(self.book.outcomes(project="p1"), [])

    def test_the_inbox_is_followed_and_each_folder_moves_to_done(self):
        folder = self.home / "outcomes_inbox" / "p1"
        folder.mkdir(parents=True)
        pd.DataFrame({"customer_id": [f"C{i}" for i in range(20)]}).to_csv(folder / "list.csv", index=False)
        self.results.to_csv(folder / "results.csv", index=False)
        (folder / "follow.json").write_text(json.dumps({"id_column": "customer_id", "outcome_column": "stayed"}))
        rows = follow.follow_inbox(self.home)
        self.assertEqual((rows[0]["project"], rows[0]["decisions"]), ("p1", 2))
        self.assertTrue((self.home / "outcomes_inbox" / "done" / "p1" / "followed.json").exists())
        self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
