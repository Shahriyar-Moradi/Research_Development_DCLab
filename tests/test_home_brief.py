"""A brief per track, and questions that matter (package H4): what Home reads from the request for each kind of help,
the defaults it takes and shows, the one question it asks only when a field has no safe default, a "not sure" that
never erases what was understood, and a column named in the request that the table does not have.

Sentences are written for these tests; the H1 dev requests are read through the suite's own measurement.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dclab_rnd.draft import analyze, brief, lang, plan  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

ZWNJ = "‌"


class ExtractTests(unittest.TestCase):
    def test_an_analysis_request_names_its_fields(self):
        text = "What was the total refund amount per store in 2025? Only online orders."
        found = brief.extract("analysis", text)
        self.assertEqual(found["measure"][0], "total refund amount")
        self.assertEqual(found["breakdown"][0], "store")
        self.assertEqual(found["period"][0], "2025")
        self.assertEqual(found["filters"][0], "online orders")
        for value, quote in found.values():
            self.assertIn(value, quote)
            self.assertTrue(plan.quoted(quote, [text]), quote)  # every field is backed by the person's words

    def test_each_other_track(self):
        self.assertEqual(brief.extract("reporting", "A monthly dashboard of signups, churn and revenue for the product team")["kpis"][0],
                         "signups, churn and revenue")
        self.assertEqual(brief.extract("reporting", "A monthly dashboard of signups for the product team")["refresh"][0], "monthly")
        self.assertEqual(brief.extract("business", "Should we close the Friday shift?")["decision"][0], "close the Friday shift")
        self.assertEqual(brief.extract("business", "Why did bookings drop in May compared with April?")["success_measure"][0], "bookings")
        self.assertEqual(brief.extract("preparation", "Merge the invoices with the clients on client_id, then export as a CSV file")["destination"][0],
                         "as a CSV file")
        self.assertEqual(brief.extract("advice", "What's the difference between ROC AUC and PR AUC?")["topic"][0], "ROC AUC and PR AUC")

    def test_persian_fields(self):
        text = "میانگین مبلغ سفارش" + ZWNJ + "ها به تفکیک شهر در سال ۱۴۰۲ چقدر بود؟"
        found = brief.extract("analysis", text)
        self.assertIn("میانگین", found["measure"][0])
        self.assertEqual(found["breakdown"][0], "شهر")
        self.assertIn("۱۴۰۲", found["period"][0])  # the person's digits
        self.assertEqual(brief.extract("reporting", "یک داشبورد هفتگی از درآمد هر کانال برای مدیر فروش")["audience"][0], "مدیر فروش")

    def test_words_that_do_not_say_a_field_are_not_taken(self):
        self.assertNotIn("measure", brief.extract("analysis", "Show the breakdown by sales region"))
        self.assertNotIn("breakdown", brief.extract("analysis", "How many customers did we lose by the end of the year?"))
        self.assertEqual(brief.extract("business", "Which store should we close first?")["decision"][0], "Which store should we close first")
        self.assertEqual(brief.extract("reporting", "A dashboard with revenue, orders, and returns, refreshed daily for the ops team")["kpis"][0],
                         "revenue, orders, and returns")
        self.assertNotIn("kpis", brief.extract("reporting", "I need to track how our app is doing"))
        self.assertIn("customers.csv", brief.extract("preparation", "Combine orders.csv and customers.csv on customer_id and remove duplicate rows")["rules"][0])
        self.assertEqual(brief.extract("analysis", "میانگین فروش بر اساس منطقه")["measure"][0], "میانگین فروش")

    def test_nothing_is_claimed_without_its_words(self):
        self.assertEqual(brief.extract("analysis", "Show me something interesting"), {})
        self.assertNotIn("decision", brief.extract("business", "Why did revenue drop in March?"))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))

    def start(self, text):
        d = self.store.create(text)
        HomeAgent(self.store, None).start(d["id"])
        return self.store.get(d["id"])

    def test_a_request_that_says_enough_asks_nothing(self):
        d = self.start("How much revenue did we make per region last quarter? Only paid orders.")
        self.assertEqual(d["questions"], [])
        entries = d["track_brief"]["analysis"]
        self.assertEqual({k: e["source"] for k, e in entries.items()}, {"measure": "problem", "breakdown": "problem", "period": "problem", "filters": "problem"})

    def test_a_field_with_a_default_is_shown_not_asked(self):
        d = self.start("Merge the two customer exports and remove duplicate rows")
        self.assertEqual(d["questions"], [])
        self.assertEqual(d["track_brief"]["preparation"]["destination"], {"value": "a cleaned table to download", "status": "inferred", "source": "default", "quote": None})

    def test_only_a_field_with_no_safe_default_is_asked_and_not_sure_moves_on(self):
        d = self.start("Make a report for the board")
        self.assertEqual([(q["field"], q.get("track")) for q in d["questions"]], [("kpis", "reporting")])
        HomeAgent(self.store, None).reply(d["id"], "Not sure")
        d = self.store.get(d["id"])
        self.assertEqual(d["track_brief"]["reporting"]["kpis"]["status"], "unknown")
        self.assertEqual(len(d["questions"]), 1)  # asked once
        self.assertEqual(d["track_brief"]["reporting"]["audience"]["value"], "board")

    def test_an_answer_fills_the_field(self):
        d = self.start("Make a report for the board")
        HomeAgent(self.store, None).reply(d["id"], "Revenue and margin by month")
        d = self.store.get(d["id"])
        self.assertEqual((d["track_brief"]["reporting"]["kpis"]["value"], d["track_brief"]["reporting"]["kpis"]["source"]),
                         ("Revenue and margin by month", "answer"))

    def test_the_budget_holds(self):
        d = self.store.create("Should we change something?")
        agent = HomeAgent(self.store, None)
        agent.start(d["id"])
        for _ in range(6):
            if agent.pending(self.store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "Not sure")
        d = self.store.get(d["id"])
        self.assertLessEqual(sum(1 for q in d["questions"] if q.get("track") == d["intent"]["track"]), brief.BUDGET[d["intent"]["track"]])

    def test_a_column_the_table_does_not_have_is_not_taken(self):
        d = self.start("Show me the average lifetime_spend by segment.")
        frame = pd.DataFrame({"spend": np.arange(40.0), "segment": ["a", "b"] * 20})
        self.store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        HomeAgent(self.store, None).data_ready(d["id"], self.store.get(d["id"])["assets"][0])
        d = self.store.get(d["id"])
        self.assertEqual(d["track_brief"]["analysis"]["measure"]["status"], "unknown")
        self.assertEqual(d["track_brief"]["analysis"]["breakdown"]["value"], "segment")
        q = d["questions"][-1]
        self.assertEqual(q["field"], "measure")
        self.assertIn("lifetime_spend", q["text"])
        self.assertIn("spend", q["options"])  # the table's numeric columns, by name

    def add_table(self, d, frame):
        self.store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        HomeAgent(self.store, None).data_ready(d["id"], self.store.get(d["id"])["assets"][0])
        return self.store.get(d["id"])

    def test_the_missing_column_rule_reads_only_column_fields_and_keeps_defaults(self):
        d = self.start("Join orders_2024 and customers on id and remove duplicate rows")
        d = self.add_table(d, pd.DataFrame({"id": np.arange(20), "amount": np.arange(20.0)}))
        entries = d["track_brief"]["preparation"]
        self.assertEqual(entries["rules"]["source"], "problem")  # a file name in a step is not a missing column
        self.assertEqual(entries["inputs"]["source"], "problem")
        self.assertFalse([q for q in d["questions"] if q.get("track")])

    def test_a_dropped_column_is_named_against_its_own_field(self):
        d = self.start("How much revenue did we make by sales_channel?")
        d = self.add_table(d, pd.DataFrame({"revenue": np.arange(40.0), "channel": ["web", "app"] * 20}))
        self.assertEqual(d["track_brief"]["analysis"]["breakdown"]["source"], "default")
        said = [m["text"] for m in d["messages"] if "can't find" in (m.get("text") or "") and "sales_channel" in m["text"]]
        self.assertEqual(len(said), 1)
        self.assertIn("breakdown", said[0])
        self.assertFalse(any("sales_channel" in q["text"] and q["field"] == "measure" for q in d["questions"]))

    def test_an_answer_naming_a_missing_column_is_not_taken(self):
        d = self.start("Make a report for the board")
        d = self.add_table(d, pd.DataFrame({"revenue": np.arange(40.0), "region": ["n", "s"] * 20}))
        HomeAgent(self.store, None).reply(d["id"], "lifetime_spend")
        d = self.store.get(d["id"])
        self.assertNotEqual(d["track_brief"]["reporting"]["kpis"]["value"], "lifetime_spend")
        self.assertIn("lifetime_spend", d["messages"][-1]["text"])

    def test_an_answer_that_is_another_kind_of_help_moves_there(self):
        d = self.start("Make a report for the board")
        HomeAgent(self.store, None).reply(d["id"], "Actually I'd rather predict which customers will ask for a refund next month")
        d = self.store.get(d["id"])
        self.assertEqual(d["intent"]["track"], "prediction")
        self.assertIsNone(d["track_brief"]["reporting"]["kpis"]["value"])

    def test_a_recorded_answer_is_said_back(self):
        d = self.start("Make a report for the board")
        HomeAgent(self.store, None).reply(d["id"], "Revenue and margin by month")
        self.assertIn("Revenue and margin by month", self.store.get(d["id"])["messages"][-1]["text"])

    def test_moving_to_another_track_closes_the_brief_question(self):
        d = self.start("Make a report for the board")
        HomeAgent(self.store, None).choose_intent(d["id"], "prediction")
        d = self.store.get(d["id"])
        self.assertTrue(all(q.get("answered") for q in d["questions"] if q.get("track") == "reporting"))


class NotSureTests(unittest.TestCase):
    def test_not_sure_in_both_languages(self):
        for text in ("Not sure", "I'm not sure", "no idea", "I don't know", "مطمئن نیستم", "نمی" + ZWNJ + "دانم", "نمیدونم"):
            self.assertTrue(lang.not_sure(text), text)
        for text in ("revenue", "the churned column", "مطمئنم که درآمد است"):
            self.assertFalse(lang.not_sure(text), text)

    def test_an_answer_that_starts_unsure_is_still_an_answer(self):
        for text in ("Not sure, but probably whether they cancel within 30 days", "No idea yet; we score every Monday morning",
                     "نمی" + ZWNJ + "دانم دقیقا، ولی لغو اشتراک در ۳۰ روز"):
            self.assertFalse(lang.not_sure(text), text)

    def test_after_not_sure_the_outcome_is_asked_again_with_the_columns(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Help me build a prediction model for my subscription shop")
        agent = HomeAgent(store, None)
        agent.start(d["id"])
        self.assertEqual(agent.pending(store.get(d["id"]))["field"], "target")
        agent.reply(d["id"], "Not sure")
        frame = pd.DataFrame({"churned": np.arange(40) % 2, "tenure": np.arange(40.0), "plan": ["a", "b"] * 20})
        store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        agent.data_ready(d["id"], store.get(d["id"])["assets"][0])
        asked_again = None
        for _ in range(6):  # the question open when the data came is answered first; the outcome comes back with columns
            q = agent.pending(store.get(d["id"]))
            if q is None:
                break
            if q["field"] == "target" and q.get("options"):
                asked_again = q
                break
            agent.reply(d["id"], "Not sure")
        self.assertIsNotNone(asked_again)
        self.assertIn("churned", asked_again["options"])

    def test_with_a_model_not_sure_is_still_taken_by_code(self):
        class Recorder:  # a model that would record the words as the outcome
            def complete(self, messages, tools=None, **kw):
                return {"content": "", "tool_calls": [{"id": "c1", "name": "record", "arguments": {"field": "target", "value": "Not sure"}}],
                        "assistant_message": {"role": "assistant", "content": ""}}
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Predict which customers will cancel within 30 days, scored on the first day of each month")
        frame = pd.DataFrame({"churned": np.arange(40) % 2, "tenure": np.arange(40.0)})
        store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        HomeAgent(store, None).start(d["id"])
        understood = store.get(d["id"])["understanding"]["target"]
        HomeAgent(store, Recorder()).reply(d["id"], "Not sure")
        self.assertEqual(store.get(d["id"])["understanding"]["target"], understood)

    def test_not_sure_never_erases_an_understood_outcome(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Predict which customers will cancel within 30 days, scored on the first day of each month")
        frame = pd.DataFrame({"churned": np.arange(40) % 2, "tenure": np.arange(40.0), "plan": ["a", "b"] * 20})
        store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        agent = HomeAgent(store, None)
        agent.start(d["id"])
        understood = store.get(d["id"])["understanding"]["target"]
        for _ in range(6):
            if agent.pending(store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "Not sure")
        self.assertEqual(store.get(d["id"])["understanding"]["target"], understood)


if __name__ == "__main__":
    unittest.main()
