"""The owner's Telco session (2026-10-29): "how can I analyze my customer", A business question, the Telco churn table, then
"show my customer segments" typed twice. Home offered SeniorCitizen as the measure of success and not Churn, asked for two
periods with invented dates on a table with no date column, and did not reply at all to the typed messages."""
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft import brief as briefs  # noqa: E402
from dclab_rnd.draft import business, pipeline  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

FIRST = "how can I analyze my customer"
LATER = "show my customer segments"


def telco(rows=80):
    rng = random.Random(7)
    return pd.DataFrame({
        "customerID": [f"{1000 + i}-ABCD" for i in range(rows)],
        "gender": [rng.choice(["Female", "Male"]) for _ in range(rows)],
        "SeniorCitizen": [rng.choice([0, 1]) for _ in range(rows)],
        "tenure": [rng.randint(1, 72) for _ in range(rows)],
        "Contract": [rng.choice(["Month-to-month", "One year", "Two year"]) for _ in range(rows)],
        "MonthlyCharges": [round(rng.uniform(18, 119), 2) for _ in range(rows)],
        "TotalCharges": [round(rng.uniform(18, 8600), 2) for _ in range(rows)],
        "Churn": [rng.choice(["Yes", "No"]) for _ in range(rows)],
    })


class Script:
    """A model that answers with these replies in order: a dict (content and tool calls) or a plain text."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append(messages)
        reply = self.replies.pop(0) if self.replies else ""
        out = dict(reply) if isinstance(reply, dict) else {"content": reply}
        calls = out.get("tool_calls") or []
        out.setdefault("assistant_message", {"role": "assistant", "content": out.get("content") or "",
                                             **({"tool_calls": [{"id": c["id"], "type": "function", "function": {
                                                 "name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})})
        return out


class TelcoTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def business_draft(self, client=None):
        d = self.store.create(FIRST)
        agent = HomeAgent(self.store, client, on_request=lambda *a: None)
        agent.start(d["id"])
        agent.choose_intent(d["id"], "business")
        path = self.store.data_dir(d["id"]) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        telco().to_csv(path, index=False)
        asset = pipeline.new_asset(self.store, d["id"], "upload", path.name, path.name)
        pipeline.run(self.store, d["id"], asset["id"], agent, None)
        return agent, d["id"]

    def said(self, draft_id):
        return [m for m in self.store.get(draft_id)["messages"] if m.get("role") != "user"]

    def test_no_periods_are_asked_of_a_table_without_dates_and_no_date_is_invented(self):
        agent, draft_id = self.business_draft()
        text = " ".join(m.get("text") or "" for m in self.said(draft_id))
        self.assertNotIn("March 2026", text)
        self.assertNotIn("which two periods", text)
        self.assertIn("no date column", text)
        self.assertIn("Churn", text)  # what it can do instead names the table's own outcome
        options = [o for q in self.store.get(draft_id)["questions"] for o in q.get("options") or []]
        self.assertNotIn("SeniorCitizen", options)

    def test_the_period_example_comes_from_the_tables_own_dates(self):
        frame = pd.DataFrame({"order_date": pd.date_range("2025-01-05", periods=200, freq="D"), "amount": range(200)})
        spec, missing = business.read("why did amount change", frame, "amount")
        self.assertIsNone(spec)
        self.assertIn("July 2025", missing)  # the latest month in the table, and the one before it
        self.assertIn("June 2025", missing)
        self.assertNotIn("2026", missing)
        spec, missing = business.read("why did revenue drop in March 2026 compared with February 2026", telco(), "MonthlyCharges")
        self.assertEqual(missing, business.NO_DATES)

    def test_a_measure_of_success_offers_the_outcome_and_never_a_flag_or_a_key(self):
        agent, draft_id = self.business_draft()
        draft = self.store.get(draft_id)
        f = briefs.field("business", "success_measure")
        offered = briefs.options(f, briefs.columns_of(draft), (draft.get("analysis") or {}).get("profile"), "")
        self.assertEqual(offered[0], "Churn")
        self.assertNotIn("SeniorCitizen", offered)
        self.assertNotIn("customerID", offered)
        offered = briefs.options(f, briefs.columns_of(draft), (draft.get("analysis") or {}).get("profile"), "is it the monthly charges?")
        self.assertEqual(offered[0], "MonthlyCharges")  # the person's own words rank first

    def test_a_typed_message_always_gets_a_reply(self):
        refused = {"content": "", "tool_calls": [{"id": "c1", "name": "ask_user", "arguments": {"question": "Which periods?"}}]}
        agent, draft_id = self.business_draft(Script(refused, "", "", "", "", ""))
        for _ in range(2):
            before = len(self.store.get(draft_id)["messages"])
            agent.reply(draft_id, "what can you tell me about them?")
            after = self.store.get(draft_id)["messages"]
            self.assertGreater(len(after), before + 1)
            self.assertNotEqual(after[-1].get("role"), "user")

    def test_a_later_message_naming_another_need_is_offered_as_a_switch(self):
        agent, draft_id = self.business_draft()
        agent.reply(draft_id, LATER)
        offer = [m for m in self.said(draft_id) if m.get("kind") == "route_suggestion"]
        self.assertTrue(offer)
        self.assertEqual(offer[-1]["suggestion"]["track"], "analysis")
        self.assertEqual(HomeAgent.track(self.store.get(draft_id)), "business")  # never switched silently
        agent.choose_intent(draft_id, "analysis")  # the Switch button
        question = [m for m in self.said(draft_id) if m.get("kind") == "question"][-1]
        self.assertEqual(question["question"]["field"], "segments_route")  # the segments' two ways, from their words

    def test_the_switch_is_offered_once_and_never_for_an_explanation_or_on_a_prediction(self):
        agent, draft_id = self.business_draft()
        agent.reply(draft_id, "explain why churn went up")  # the business question itself, never advice
        self.assertFalse([m for m in self.said(draft_id) if m.get("kind") == "route_suggestion" and m["suggestion"].get("rule") == "later_message"])
        agent.reply(draft_id, LATER)
        agent.reply(draft_id, LATER)  # the owner typed it twice: offered once, then answered inside the track
        offers = [m for m in self.said(draft_id) if m.get("kind") == "route_suggestion" and m["suggestion"].get("rule") == "later_message"]
        self.assertEqual(len(offers), 1)
        self.assertNotIn("the a ", offers[0]["text"])
        self.assertNotEqual(self.store.get(draft_id)["messages"][-1].get("role"), "user")
        self.assertTrue(self.store.get(draft_id).get("suggestion_earlier"))  # the record read when the track opened is kept
        d = self.store.create("Predict which customers will cancel next month")
        HomeAgent(self.store, None, on_request=lambda *a: None).start(d["id"])
        before = len(self.store.get(d["id"])["messages"])
        HomeAgent(self.store, None, on_request=lambda *a: None).reply(d["id"], "We will send a monthly report of the customers at risk to the team")
        self.assertFalse([m for m in self.store.get(d["id"])["messages"][before:] if m.get("kind") == "route_suggestion"])


if __name__ == "__main__":
    unittest.main()
