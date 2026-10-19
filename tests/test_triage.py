"""Package 30.24: delivering many projects (the owner's pattern 11).

Fifteen written requests and capacity for five: every request scored with its reasons, the five chosen by the total, none without a
success number agreed before building, the reuse among the chosen said; and a project with no use for the period flagged to retire, one in
use never.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIFTEEN = [
    ("R01", "Churn of monthly subscribers", "binary", 30_000, ["customers", "billing"], True, True, False, True, False, "recall of churners at top 10% >= 0.5"),
    ("R02", "Delivery fraud screening", "rare_event", 60_000, ["rides", "cases"], True, True, False, False, False, "precision at 200 a day >= 0.4"),
    ("R03", "Demand forecast per store", "forecasting", 25_000, ["sales", "stores"], True, None, False, False, False, "MAPE below 15%"),
    ("R04", "Why sales dropped (BI chat)", "metric_change", 8_000, ["sales", "footfall"], True, None, True, False, False, "answers right on 20 questions"),
    ("R05", "Invoice reading", None, 6_000, [], False, None, True, False, False, "fields right on 95% of invoices"),
    ("R06", "Price changes applied automatically", "regression", 40_000, ["sales", "prices"], True, True, False, False, True, "margin up 2%"),
    ("R07", "Customer segments for marketing", "segmentation", 4_000, ["customers"], True, None, False, True, False, "segments stable over 3 months"),
    ("R08", "Ticket routing", "multiclass", 12_000, ["tickets"], True, True, False, True, False, "macro F1 >= 0.7"),
    ("R09", "Driver rating prediction", "regression", 500, ["rides"], True, True, False, True, False, None),  # no success number agreed
    ("R10", "Next best offer", "ranking", 20_000, ["customers", "orders", "offers"], False, None, False, True, False, "uplift in take rate"),
    ("R11", "Machine failure alerts", "streams", 15_000, ["sensors", "incidents"], True, True, True, False, False, "alerts 2 days ahead"),
    ("R12", "Campaign uplift", "uplift", 18_000, ["customers", "campaigns"], True, True, False, True, False, "incremental revenue"),
    ("R13", "Late order prediction", "binary", 9_000, ["rides"], True, True, False, False, False, "recall of late orders >= 0.6"),
    ("R14", "Duplicate customers across systems", "matching", 7_000, ["customers", "crm"], True, True, False, True, False, None),  # no success number
    ("R15", "Employee attrition", "binary", 3_000, ["employees"], False, False, False, True, False, "flag 60% of leavers"),
]


def requests():
    from dclab_core import triage

    return [triage.Request(id=i, title=t, kind=k, value_eur_month=v, tables=tb, tables_exist=ex, labelled=lab, new_kind=new, personal_data=pd_,
                           acts_on_its_own=auto, success=ok) for i, t, k, v, tb, ex, lab, new, pd_, auto, ok in FIFTEEN]


class TriageTest(unittest.TestCase):
    def test_five_are_chosen_of_fifteen_by_their_scores_with_reasons_and_reuse(self):
        from dclab_core import triage

        p = triage.choose(requests(), 5)
        self.assertEqual(len(p.chosen), 5)
        self.assertEqual(sorted(p.chosen), ["R01", "R02", "R03", "R06", "R13"])  # the five, pinned: a change of weights is seen here
        self.assertEqual(sorted(p.not_ready), ["R09", "R14"])  # no success number: never chosen
        self.assertFalse(set(p.chosen) & set(p.not_ready))
        by = {s.request: s for s in p.scores}
        self.assertTrue(all(len(s.reasons) == 4 for s in p.scores))
        ranked = sorted((r for r in requests() if r.success), key=lambda r: -by[r.id].total)
        self.assertGreaterEqual(min(by[i].total for i in p.chosen), max(by[i].total for i in p.waiting))  # the chosen score at least as high
        self.assertIn("R02", p.chosen)  # the most valuable, ready and labelled
        self.assertNotIn("R06", p.chosen[:1])  # valuable, but its prices are applied with no person: its risk weighs
        self.assertGreaterEqual(by["R06"].risk, 3)
        shared = {tuple(sorted(r["requests"])) for r in p.reuse}
        self.assertTrue(p.reuse and all(r["tables"] or r["kind"] for r in p.reuse), shared)
        self.assertEqual(triage.Portfolio.read(p.model_dump_json()), p)
        self.assertEqual(len(ranked), 13)

    def test_a_project_nobody_uses_is_flagged_and_one_in_use_never(self):
        from dclab_core import triage

        used = {"old": "2026-05-01T00:00:00+00:00", "fresh": "2026-10-15T00:00:00+00:00", "never": None, "new": None, "undated": None}
        made = {"never": "2026-01-01T00:00:00+00:00", "new": "2026-10-19T00:00:00+00:00"}
        flagged = triage.retire(used, now="2026-10-20T00:00:00+00:00", days=90, created=made)
        self.assertEqual([f["project"] for f in flagged], ["never", "old"])  # one made yesterday, or with no date, is never flagged
        self.assertIn("never used", flagged[0]["why"])

    def test_the_studio_reads_each_projects_last_use_from_its_records(self):
        import tempfile
        from types import SimpleNamespace

        from dclab_rnd import experience, triage as studio

        made = "2026-01-01T00:00:00+00:00"
        rows = [{"id": "old", "created": made, "updated": made}, {"id": "used", "created": made, "updated": made}]
        services = SimpleNamespace(projects=SimpleNamespace(list=lambda: rows))
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        with experience.bind(project="used"):
            self.assertIsNotNone(experience.record("outcome.observed", "person", {"metric": "orders_saved", "verdict": "kept"}, log=log))
        flagged = studio.unused(services, days=90, now=experience.now(), log=log)
        self.assertEqual([f["project"] for f in flagged], ["old"])  # an outcome observed today is a use; the other was last touched in January

if __name__ == "__main__":
    unittest.main()
