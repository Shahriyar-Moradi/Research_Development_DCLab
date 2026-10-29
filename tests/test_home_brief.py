"""A brief per track (package H4): fields read from the person's words (English and Persian), defaults, the question
only when it matters, a column that is not there refused, "Not sure" keeping what is known, and Home's flow.

The sentences here are written for these tests; none is a request of the H1 suite (dev or sealed).
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft import brief, plan  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

COLUMNS = ["ticket_id", "team", "opened_at", "resolution_hours", "priority", "satisfaction"]


def values(track, text, columns=None):
    return {k: v["value"] for k, v in brief.read(track, text, columns)["fields"].items()}


class ReadTests(unittest.TestCase):
    def test_analysis_fields_in_english(self):
        got = values("analysis", "What was the average resolution_hours per team last month? Only urgent tickets.", COLUMNS)
        self.assertEqual(got["measure"], "average resolution_hours")
        self.assertEqual(got["breakdown"], "team")
        self.assertEqual(got["period"], "last month")
        self.assertEqual(got["filters"], "urgent tickets")

    def test_analysis_fields_in_persian(self):
        text = "میانگین زمان پاسخ به تفکیک تیم در سال ۲۰۲۵ چقدر بود؟"
        got = values("analysis", text)
        self.assertTrue(got["measure"].startswith("میانگین زمان"))
        self.assertEqual(got["breakdown"], "تیم")
        self.assertEqual(got["period"], "سال ۲۰۲۵")
        for v in got.values():
            self.assertIn(v, text)  # every value is the person's words, in their own script

    def test_reporting_and_business_and_preparation(self):
        got = values("reporting", "A daily report of tickets opened and satisfaction for the support managers.")
        self.assertEqual((got["kpis"], got["audience"], got["refresh"]), ("tickets opened and satisfaction", "support managers", "daily"))
        got = values("business", "Why did satisfaction fall in April compared with March?")
        self.assertEqual((got["success_measure"], got["period_compared"]), ("satisfaction", "April compared with March"))
        got = values("business", "Should we hire two more agents for the night shift?")
        self.assertEqual(got["decision"], "hire two more agents for the night shift")
        got = values("preparation", "Drop the test tickets and export the rest as a CSV file.")
        self.assertEqual((got["rules"], got["destination"]), ("Drop the test tickets", "CSV file"))

    def test_a_column_that_is_not_there_is_refused_never_taken(self):
        result = brief.read("analysis", "Show the average first_reply_minutes by team.", COLUMNS)
        self.assertNotIn("measure", result["fields"])
        self.assertEqual(result["missing"], {"measure": ["first_reply_minutes"]})
        self.assertEqual(result["fields"]["breakdown"]["value"], "team")
        self.assertIn("measure", brief.read("analysis", "Show the average first_reply_minutes by team.")["fields"])  # no data: no check

    def test_no_guess_when_the_words_say_nothing(self):
        got = values("analysis", "Tell me something interesting about my tickets.")
        self.assertEqual(got, {})  # an analysis's question stays its default, "the request as written"
        self.assertEqual(values("reporting", "I need a dashboard."), {})

    def test_every_quote_passes_the_products_quote_rule(self):
        text = "Compare the total resolution_hours by priority over 2025 for the leadership team, weekly report please."
        for track in ("analysis", "reporting"):
            for entry in brief.read(track, text, COLUMNS)["fields"].values():
                self.assertTrue(plan.quoted(entry["quote"], [text]) or len(entry["quote"]) < 8, entry["quote"])


class BriefTests(unittest.TestCase):
    def test_defaults_fill_what_is_safe_and_the_rest_is_asked(self):
        filled = brief.fill("analysis", brief.read("analysis", "Tell me about my tickets.")["fields"])
        self.assertEqual(filled["period"], {"value": "all rows", "status": "inferred", "source": "default", "quote": None})
        self.assertEqual(filled["measure"]["status"], "unknown")
        self.assertEqual(brief.asks("analysis", filled, 0).name, "measure")
        self.assertIsNone(brief.asks("analysis", filled, brief.BUDGET["analysis"]))  # the budget is spent

    def test_a_request_that_names_measure_and_breakdown_asks_nothing(self):
        filled = brief.fill("analysis", brief.read("analysis", "Average satisfaction by team.", COLUMNS)["fields"])
        self.assertIsNone(brief.asks("analysis", filled, 0))

    def test_not_sure_records_the_default_and_keeps_the_rest(self):
        filled = brief.fill("reporting", brief.read("reporting", "A dashboard of satisfaction for the support team.")["fields"])
        self.assertEqual(brief.asks("reporting", filled, 0).name, "refresh")
        after = brief.answered("reporting", filled, "refresh", "Not sure", not_known=False)
        self.assertEqual((after["refresh"]["status"], after["refresh"]["value"]), ("unknown", None))
        self.assertTrue(after["refresh"]["asked"])  # never asked twice
        self.assertEqual(after["kpis"], filled["kpis"])  # what was understood stays
        self.assertIsNone(brief.asks("reporting", after, 1))

    def test_a_known_field_is_never_replaced_by_a_reading_or_a_default(self):
        before = {"measure": {"value": "satisfaction", "status": "stated", "source": "user", "quote": "satisfaction"}}
        filled = brief.fill("analysis", {"measure": {"value": "number of tickets", "source": "rules"}}, before)
        self.assertEqual(filled["measure"]["value"], "satisfaction")

    def test_options_come_from_the_columns_names_only(self):
        columns = [{"name": "resolution_hours", "kind": "numeric", "unique": 900}, {"name": "team", "kind": "categorical", "unique": 6},
                   {"name": "ticket_id", "kind": "categorical", "unique": 5000}]
        self.assertEqual(brief.options(brief.field("analysis", "measure"), columns), ["resolution_hours", "Number of rows", "Not sure"])
        self.assertEqual(brief.options(brief.field("analysis", "breakdown"), columns), ["team", "Not sure"])  # not an identifier


class ReviewOfH4Tests(unittest.TestCase):
    """The review's findings (H4), each with the sentence that showed it."""

    def test_a_prediction_said_with_explore_or_profile_stays_a_prediction(self):
        from dclab_rnd.draft import intent
        for sentence in ("Help me find out which loans default, explore the data",
                         "I want to explore the data and find which customers churn",
                         "Profile my users and find who is at risk",
                         "Use statistics to figure out which customers are at risk of churning"):
            # never read as an analysis; one the rules cannot read is asked openly (2026-10-29), never guessed
            self.assertIn(intent.read(sentence)["track"], ("prediction", None), sentence)
        self.assertEqual(intent.read("I want an analysis of my tickets.")["track"], "analysis")  # asked outright: still strong

    def test_words_that_are_not_fields(self):
        cases = [("analysis", "tickets closed by the end of the month", "breakdown"),
                 ("analysis", "by the way, how are we doing", "breakdown"),
                 ("analysis", "a trip across the bridge", "breakdown"),
                 ("analysis", "I may want to march on with this", "period"),
                 ("reporting", "a dashboard for my boss", "kpis"),
                 ("analysis", "I am only interested in refunds", "filters"),
                 ("preparation", "Every customer has many orders; join them", "output_grain")]
        for track, sentence, field in cases:
            self.assertNotIn(field, values(track, sentence), sentence)
        self.assertEqual(values("analysis", "the number of the team members")["measure"], "number of the team members")


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home)
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def start(self, text, table=False):
        d = self.store.create(text)
        if table:  # a described table (names and kinds only): the measure question has the table's columns as buttons
            self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "resolution_hours", "kind": "numeric", "unique": 80},
                                                                               {"name": "region", "kind": "categorical", "unique": 4}]}))
        agent = HomeAgent(self.store, None)
        agent.start(d["id"])
        return agent, self.store.get(d["id"])

    def test_no_measure_question_before_the_table(self):  # 2026-10-29: "What should I measure?" with only "Number of rows"
        agent, d = self.start("I want an analysis of my tickets.")
        self.assertIsNone(agent.pending(d))
        agent, d = self.start("I want an analysis of my tickets.", table=True)
        q = agent.pending(d)
        self.assertEqual(q["field"], "measure")
        self.assertIn("resolution_hours", q["options"])

    def test_an_analysis_request_that_says_everything_asks_nothing(self):
        agent, d = self.start("What was the average satisfaction per team last month?")
        self.assertEqual(d["intent"]["track"], "analysis")
        b = d["track_brief"]["analysis"]
        self.assertEqual((b["measure"]["value"], b["breakdown"]["value"], b["period"]["value"]), ("average satisfaction", "team", "last month"))
        self.assertIsNone(agent.pending(d))

    def test_an_unclear_analysis_asks_one_question_with_a_not_sure_button_and_not_sure_moves_on(self):
        agent, d = self.start("I want an analysis of my tickets.", table=True)
        q = agent.pending(d)
        self.assertEqual((q["field"], q["brief"]), ("measure", "analysis"))
        self.assertEqual(q["options"][-1], "Not sure")
        self.assertLessEqual(len(q["options"]), 4)
        agent.reply(d["id"], "Not sure")
        d = self.store.get(d["id"])
        self.assertIsNone(agent.pending(d))  # no second ask of the same field, and nothing else worth asking
        self.assertEqual(d["track_brief"]["analysis"]["measure"]["status"], "unknown")
        self.assertEqual(d["track_brief"]["analysis"]["period"]["source"], "default")

    def test_an_answer_is_the_persons_words(self):
        agent, d = self.start("A dashboard of satisfaction for the support team.")
        self.assertEqual(agent.pending(d)["field"], "refresh")
        agent.reply(d["id"], "Every Monday morning")
        b = self.store.get(d["id"])["track_brief"]["reporting"]
        self.assertEqual((b["refresh"]["value"], b["refresh"]["source"]), ("Every Monday morning", "user"))
        self.assertEqual(b["kpis"]["value"], "satisfaction")

    def test_a_prediction_asks_todays_questions(self):
        agent, d = self.start("Predict which tickets will breach their deadline within 2 days, scored when the ticket is opened.")
        self.assertEqual(d["intent"]["track"], "prediction")
        self.assertNotIn("prediction", d.get("track_brief") or {})
        q = agent.pending(d)
        self.assertIsNotNone(q)
        self.assertNotIn("brief", q)

    def test_changing_to_a_prediction_closes_the_brief_question(self):
        agent, d = self.start("I want an analysis of my tickets.", table=True)
        self.assertEqual(agent.pending(d)["brief"], "analysis")
        agent.choose_intent(d["id"], "prediction")
        q = agent.pending(self.store.get(d["id"]))
        self.assertTrue(q is None or "brief" not in q)

    def test_another_request_while_a_brief_question_is_open_is_never_its_answer(self):
        agent, d = self.start("I want an analysis of my tickets.", table=True)
        self.assertEqual(agent.pending(d)["field"], "measure")
        agent.reply(d["id"], "Actually I want to predict which customers will churn next month")
        d = self.store.get(d["id"])
        self.assertNotEqual((d["track_brief"]["analysis"]["measure"] or {}).get("source"), "user")  # not taken as the measure
        agent2, d2 = self.start("I want an analysis of my tickets.", table=True)
        agent2.reply(d2["id"], "Can you make synthetic data for me?")
        self.assertNotEqual((self.store.get(d2["id"])["track_brief"]["analysis"]["measure"] or {}).get("source"), "user")
        agent3, d3 = self.start("I want an analysis of my tickets.", table=True)
        agent3.reply(d3["id"], "resolution time")  # a short answer is the answer
        self.assertEqual(self.store.get(d3["id"])["track_brief"]["analysis"]["measure"]["value"], "resolution time")

    def test_a_sheet_edit_closes_the_open_question_on_that_field(self):
        agent, d = self.start("I want an analysis of my tickets.", table=True)
        self.assertEqual(agent.pending(d)["field"], "measure")
        agent.set_brief(d["id"], "analysis", "measure", "resolution time")
        agent.reply(d["id"], "and split it by region please")
        b = self.store.get(d["id"])["track_brief"]["analysis"]
        self.assertEqual(b["measure"]["value"], "resolution time")  # the next message never overwrites the edit

    def test_the_person_edits_a_field_on_the_sheet(self):
        agent, d = self.start("What was the average satisfaction per team last month?")
        agent.set_brief(d["id"], "analysis", "period", "the whole of 2025")
        b = self.store.get(d["id"])["track_brief"]["analysis"]
        self.assertEqual((b["period"]["value"], b["period"]["source"]), ("the whole of 2025", "user"))
        with self.assertRaises(ValueError):
            agent.set_brief(d["id"], "analysis", "target", "x")  # a prediction's field is not on this sheet


if __name__ == "__main__":
    unittest.main()
