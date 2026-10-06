"""The leakage reviewer (package A3.2): proposals with a reason and a record, code decides, measured on the studied samples."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio import leakage_review  # noqa: E402

MOMENT = "Immediately before a marketing call. Duration is post-call and forbidden."
PROFILE = {"columns": [{"name": n, "kind": k, "missing_rate": 0.0, "unique": u, "preview": ["SECRET-VALUE"]}
                       for n, k, u in [("age", "numeric", 70), ("duration", "numeric", 1500), ("pdays", "numeric", 27), ("campaign", "numeric", 40),
                                       ("poutcome", "categorical", 3), ("y", "categorical", 2)]]}
FLAGGED = [{"column": "duration", "reason": "post-outcome name; known DCLab leakage precedent", "proof": ["LEAK-bank_marketing", "DCLAB-R04"]}]


class Scripted:
    def __init__(self, content):
        self.content, self.seen, self.verdicts = content, [], []

    def complete(self, messages, tools=None, **_):
        self.seen.append(json.dumps(messages))
        return {"content": self.content, "tool_calls": [], "assistant_message": {"role": "assistant", "content": self.content}}

    def output(self, ok, reason=""):
        self.verdicts.append((ok, reason))


class ReviewerTests(unittest.TestCase):
    def test_on_the_studied_samples_it_finds_the_rds_forbidden_columns(self):
        keys = ["bank_marketing", "online_shoppers", "hyperack", "credit_card_fraud", "bike_sharing_daily", "ecommerce_clothing_reviews"]
        m = leakage_review.measure(keys)  # the six samples where the R&D forbade columns; no model
        by = {r["sample"]: r for r in m["samples"]}
        self.assertIn("duration", by["bank_marketing"]["found"])
        self.assertEqual((m["found"], m["rd_forbidden"]), (14, 17))
        # what only a reader of the moment finds: "completed-session counts" name no column
        self.assertEqual(by["online_shoppers"]["missed"], ["Administrative", "Informational", "ProductRelated"])
        self.assertIn("BounceRates", by["online_shoppers"]["found"])  # "Exclude … bounce/exit rates" in the moment

    def test_only_the_audits_flags_are_ticked(self):
        out = leakage_review.review(PROFILE, MOMENT, FLAGGED, "y")
        self.assertEqual([(i["column"], i["source"], i["apply"]) for i in out["items"]], [("duration", "audit", True)])
        self.assertEqual(out["items"][0]["records"], ["LEAK-bank_marketing", "DCLAB-R04"])
        moment = "Session-start purchase prediction. Exclude durations and bounce rates"
        profile = {"columns": [{"name": "BounceRates"}, {"name": "Region"}]}
        item, = leakage_review.review(profile, moment, [], "Revenue")["items"]
        self.assertEqual((item["column"], item["source"], item["apply"], item["records"]), ("BounceRates", "moment", False, ["DCLAB-R01"]))

    def test_a_models_items_need_a_reason_that_quotes_the_moment_and_real_records(self):
        model = Scripted(json.dumps({"columns": [
            {"column": "pdays", "verdict": "after the moment", "reason": "Counted after the contact, not immediately before a marketing call.", "records": ["DCLAB-R01"]},
            {"column": "campaign", "verdict": "after the moment", "reason": "", "records": ["DCLAB-R01"]},  # no reason
            {"column": "poutcome", "verdict": "after the moment", "reason": "Known only before a marketing call ends.", "records": ["LEAK-nowhere"]},  # no such record
            {"column": "age", "verdict": "after the moment", "reason": "Age changes over time.", "records": ["DCLAB-R01"]},  # does not quote the moment
            {"column": "income", "verdict": "after the moment", "reason": "before a marketing call", "records": ["DCLAB-R01"]},  # no such column
            {"column": "duration", "verdict": "available", "reason": "The call length is planned before a marketing call.", "records": ["DCLAB-R01"]},
        ]}))
        out = leakage_review.review(PROFILE, MOMENT, FLAGGED, "y", client=model)
        self.assertEqual(out["mode"], "model")
        self.assertEqual(out["dropped"], 4)
        self.assertEqual([(i["column"], i["source"], i["apply"]) for i in out["items"]], [("duration", "audit", True), ("pdays", "model", False)])
        self.assertEqual([d["column"] for d in out["disagreements"]], ["duration"])  # reported, and the flag stays ticked
        self.assertEqual(model.verdicts[0][0], False)
        self.assertNotIn("SECRET-VALUE", model.seen[0])  # names and summaries, never values

    def test_no_moment_no_model_and_a_quote_needs_a_content_word(self):
        model = Scripted(json.dumps({"columns": [{"column": "age", "verdict": "after the moment", "reason": "Anything at all.", "records": ["DCLAB-R01"]}]}))
        out = leakage_review.review(PROFILE, None, FLAGGED, "y", client=model)
        self.assertEqual((out["mode"], model.seen), ("rules", []))  # nothing to judge against: no request is paid for
        self.assertFalse(leakage_review.quotes_moment("Age is recorded at the end of the year.", "At the monthly snapshot of each customer."))
        self.assertTrue(leakage_review.quotes_moment("Not known at the monthly snapshot.", "At the monthly snapshot of each customer."))
        twice = Scripted(json.dumps({"columns": [{"column": "duration", "verdict": "available", "reason": f"Planned {w} a marketing call.", "records": ["DCLAB-R01"]}
                                                 for w in ("before", "ahead of")]}))
        self.assertEqual(len(leakage_review.review(PROFILE, MOMENT, FLAGGED, "y", client=twice)["disagreements"]), 1)

    def test_the_moment_rule_reads_only_the_excluding_part(self):
        profile = {"columns": [{"name": "region"}, {"name": "age"}, {"name": "outcome_code"}]}
        out = leakage_review.review(profile, "At signup; region and age are known but the outcome is not available", [], "y")
        self.assertEqual([i["column"] for i in out["items"]], ["outcome_code"])

    def test_an_unreadable_or_failing_model_leaves_the_rules(self):
        class Broken:
            def complete(self, *a, **k):
                raise RuntimeError("BudgetExceeded: the cap is reached")
        for client in (Scripted("Sure, duration leaks."), Broken()):
            out = leakage_review.review(PROFILE, MOMENT, FLAGGED, "y", client=client)
            self.assertEqual([i["column"] for i in out["items"]], ["duration"])

    def test_the_intern_reviews_a_projects_columns(self):
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore

        box = Toolbox(ProjectStore(Path(tempfile.mkdtemp()) / "projects"))
        pid = box.call("create_project", {"name": "Calls", "goal": "Predict term deposits"})["project_id"]
        box.call("use_sample", {"project_id": pid, "key": "bank_marketing"})
        out = box.call("review_leakage", {"project_id": pid, "target": "target"})
        self.assertIn("duration", [i["column"] for i in out["forbid"]])
        self.assertTrue(out["prediction_moment"].startswith("Immediately before a marketing call"))  # the sample's own moment
        self.assertTrue(all(not i["apply"] for i in out["consider"]))


if __name__ == "__main__":
    unittest.main()
