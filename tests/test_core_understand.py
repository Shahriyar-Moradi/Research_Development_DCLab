"""Package 17.2: understand the problem from its description, before any data (dclab_core.understand).

Four briefs, written for these tests (none is a real document): a fraud challenge with a rubric, a churn problem, a
delivery-time regression and a ticket-routing multiclass problem. Each test pins what a careful data scientist takes
from the words alone, and that nothing is filled in without the words it came from (or a stated reason).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""

FRAUD = """Fraud detection exercise.
Build an algorithm that, given the rides of one city, detects fraudulent activity by couriers and by customers.
A sample of known forged rides is supplied in known_frauds.csv so you can test how well the algorithm works.
Any technique is allowed, but you must justify the choice and the selection of the best model.
Write everything in Python following good practice (modular, testable, type annotations).
Deliver a report as a PDF that explains every step.
Scoring: innovative feature extraction (3 pts.), an API server with Flask and Gunicorn (2 pts.), dockerize the code (1 pt.),
use source control (1 pt.).
The data you receive must not be disclosed or distributed, in whole or in part."""
FRAUD_FILES = {"rides.csv": ["created_date", "account_id", "biker_id", "order_id", "biker_fare", "sender_phone",
                             "source_lat", "source_lon", "destination_lat", "destination_lon"],
               "known_frauds.csv": ["order_id", "biker_id", "reason"]}

CHURN = """We run a subscription video service. On the first day of each month, using data up to the end of the previous
month, we want to know which subscribers will cancel within the next 30 days, so the retention team can call the top 500.
The table has one row per subscriber per month; churned_next_30d is the label. A call costs 4 euros and a lost
subscriber costs about 60 euros."""
CHURN_FILES = {"subscribers.csv": ["subscriber_id", "month", "plan", "watch_hours", "tickets", "churned_next_30d"]}

DELIVERY = """Estimate how many minutes a food delivery will take, to show the customer when they place the order.
orders.csv holds the order time, the restaurant, the distance, the couriers online, the weather and actual_minutes.
The estimate should be within 5 minutes for at least 80% of orders, and it must answer in under 100 ms."""
DELIVERY_FILES = {"orders.csv": ["order_time", "restaurant_id", "distance_km", "couriers_online", "weather", "actual_minutes"]}

TICKETS = """Route each incoming support ticket to one of 12 teams at the moment it is created, from its subject, its body and
the customer's plan. tickets.csv records the team that finally handled it in assigned_team. Some teams receive very
few tickets. A wrong route costs a handoff of about two hours."""
TICKETS_FILES = {"tickets.csv": ["ticket_id", "created_at", "subject", "body", "plan", "resolved_by", "assigned_team"]}


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class UnderstandTest(unittest.TestCase):
    def understand(self, text, files=None, advisor=None):
        from dclab_core.understand import understand

        return understand(text, files=files, advisor=advisor)

    def assertGrounded(self, brief, text):
        """Every stated fact quotes the description (or a file name); every inferred one has a reason."""
        flat = " ".join(text.split()).lower()
        facts = {name: getattr(brief, name) for name in ("goal", "action", "unit", "task", "target_column", "target_definition",
                                                          "label_source", "primary_metric",
                                                          "label_coverage", "prediction_moment", "horizon", "entities",
                                                          "time_column")}
        for name, fact in facts.items():
            with self.subTest(field=name):
                if fact.status == "stated" and fact.source == "text":
                    self.assertIn(" ".join(fact.quote.split()).lower(), flat)
                if fact.status == "inferred":
                    self.assertTrue(fact.reason)

    def test_the_fraud_challenge(self):
        brief = self.understand(FRAUD, FRAUD_FILES)
        self.assertGrounded(brief, FRAUD)
        self.assertEqual(brief.task.value, "binary")
        self.assertEqual(brief.unit.value, "ride")
        self.assertEqual(brief.label_source.value, "file")
        self.assertIn("known_frauds.csv", brief.label_source.quote)
        self.assertEqual(brief.label_coverage.value, "partial")       # "a sample of known forged rides"
        self.assertFalse(brief.target_column.known)                     # no column holds the label ...
        self.assertIn("known_frauds.csv", brief.target_definition.value)  # ... the file of known cases defines it
        self.assertEqual(sorted(brief.entities.value), ["biker_id"])   # couriers -> the biker column; customers have none
        kinds = {c.kind for c in brief.constraints}
        self.assertIn("confidential", kinds)
        points = {d.name: d.points for d in brief.deliverables}
        self.assertEqual(points.get("feature extraction"), 3)
        self.assertEqual(points.get("api"), 2)
        self.assertEqual(points.get("docker"), 1)
        self.assertEqual(points.get("source control"), 1)
        self.assertIn("report", points)
        # the prediction moment is not said: it is the first question, and the most important one
        self.assertFalse(brief.prediction_moment.known)
        self.assertEqual(brief.open_questions[0].field, "prediction_moment")
        self.assertEqual(brief.open_questions[0].impact, 1)

    def test_churn(self):
        brief = self.understand(CHURN, CHURN_FILES)
        self.assertGrounded(brief, CHURN)
        self.assertEqual(brief.task.value, "binary")
        self.assertEqual(brief.target_column.value, "churned_next_30d")
        self.assertEqual(brief.target_column.status, "stated")
        self.assertEqual(brief.label_source.value, "column")
        self.assertIn("first day of each month", brief.prediction_moment.quote)
        self.assertIn("30 days", brief.horizon.quote)
        self.assertEqual(brief.entities.value, ["subscriber_id"])
        self.assertEqual(brief.time_column.value, "month")
        capacity = [c for c in brief.constraints if c.kind == "review_capacity"]
        self.assertTrue(capacity and "500" in capacity[0].text)
        self.assertTrue(any(c.kind == "budget" and "60" in c.text for c in brief.constraints))  # the cost of each error
        self.assertNotIn("prediction_moment", [q.field for q in brief.open_questions])

    def test_delivery_time_regression(self):
        brief = self.understand(DELIVERY, DELIVERY_FILES)
        self.assertGrounded(brief, DELIVERY)
        self.assertEqual(brief.task.value, "regression")
        self.assertEqual(brief.target_column.value, "actual_minutes")
        self.assertEqual(brief.unit.value, "order")
        self.assertIn("when they place the order", brief.prediction_moment.quote)
        self.assertTrue(any(c.kind == "latency" and "100 ms" in c.text for c in brief.constraints))
        self.assertTrue(any("80%" in (s.quote or "") for s in brief.success_criteria))

    def test_ticket_routing_multiclass(self):
        brief = self.understand(TICKETS, TICKETS_FILES)
        self.assertGrounded(brief, TICKETS)
        self.assertEqual(brief.task.value, "multiclass")
        self.assertEqual(brief.target_column.value, "assigned_team")
        self.assertEqual(brief.unit.value, "ticket")
        self.assertIn("at the moment it is created", brief.prediction_moment.quote)
        self.assertTrue(any(c.kind == "budget" and "two hours" in c.text for c in brief.constraints))

    def test_nothing_is_invented_from_an_empty_description(self):
        brief = self.understand("", {})
        for name in ("goal", "task", "target_column", "target_definition", "unit", "prediction_moment"):
            self.assertFalse(getattr(brief, name).known, name)
        self.assertEqual([q.field for q in brief.open_questions][:2], ["goal", "target"])

    def test_an_advisor_fills_gaps_only_with_the_persons_words_that_support_its_value(self):
        class Advisor:
            def ask(self, purpose, evidence):
                return {"action": {"value": "test how well the algorithm works", "quote": "so you can test how well the algorithm works"},
                        "prediction_moment": {"value": "after the ride is completed and paid", "quote": "detects fraudulent activity by couriers"},
                        "horizon": {"value": "next week", "quote": "e"},
                        "unit": {"value": "invented", "quote": "words that are not in the text"}}

        brief = self.understand(FRAUD, FRAUD_FILES, advisor=Advisor())
        self.assertEqual((brief.action.status, brief.action.source), ("inferred", "advisor"))  # quoted and supported: kept as an inference
        self.assertFalse(brief.prediction_moment.known)   # quoted, but the quote does not say it: dropped
        self.assertFalse(brief.horizon.known)             # "e" is not a quote
        self.assertEqual(brief.unit.value, "ride")        # the rules' answer is never replaced

    def test_a_scripted_advisor_changes_nothing_the_rules_found_on_any_brief(self):
        class Advisor:
            def ask(self, purpose, evidence):
                return {f: {"value": "something else", "quote": "something else entirely"} for f in evidence["fields_to_propose"]}

        for text, files in ((FRAUD, FRAUD_FILES), (CHURN, CHURN_FILES), (DELIVERY, DELIVERY_FILES), (TICKETS, TICKETS_FILES)):
            with self.subTest(text=text[:30]):
                self.assertEqual(self.understand(text, files, advisor=Advisor()), self.understand(text, files))

    def test_a_broken_or_wrong_advisor_is_no_advisor(self):
        class Broken:
            def ask(self, purpose, evidence):
                raise TimeoutError("down")

        class Wrong:
            def ask(self, purpose, evidence):
                return {"task": {"value": "clustering", "quote": "detects fraudulent activity"},
                        "target_column": {"value": "not_a_column", "quote": "known forged rides"},
                        "action": {"value": "x" * 5000, "quote": "detects fraudulent activity"}}

        plain = self.understand(FRAUD, FRAUD_FILES)
        self.assertEqual(self.understand(FRAUD, FRAUD_FILES, advisor=Broken()), plain)
        self.assertEqual(self.understand(FRAUD, FRAUD_FILES, advisor=Wrong()), plain)

    def test_the_advisor_sees_every_paragraph_of_a_long_brief(self):
        seen = {}

        class Recorder:
            def ask(self, purpose, evidence):
                seen.update(evidence)
                return None

        long_text = "\n\n".join(f"Paragraph {i}: we want to detect fraudulent rides in region {i}. " + "Details follow. " * 60 for i in range(30))
        self.understand(long_text, {"r" * 150 + ".csv": ["a", "b"]}, advisor=Recorder())
        self.assertEqual(len(seen["description_paragraphs"]), 20)
        self.assertTrue(all(len(p) <= 2000 for p in seen["description_paragraphs"]))

    def test_long_pasted_text_is_clipped_never_an_error(self):
        bullet = "- confidential pricing data, the cost of a wrong decision is high " + "and more words " * 400
        text = "\n".join([bullet] * 3 + [f"A wrong flag costs {i} euros." for i in range(40)]
                         + [f"Reach at least {i}% recall." for i in range(30)])
        brief = self.understand(text)
        self.assertLessEqual(len(brief.constraints), 30)
        self.assertLessEqual(len(brief.success_criteria), 20)

    def test_the_brief_is_the_same_text_the_same_answer(self):
        self.assertEqual(self.understand(CHURN, CHURN_FILES), self.understand(CHURN, CHURN_FILES))
        self.assertEqual(len(self.understand(CHURN, CHURN_FILES).source_digest), 64)

    def test_no_moment_is_read_from_the_outcome_or_from_a_description_of_the_present(self):
        for sentence in ("Predict which customers will cancel before the renewal.",
                         "Predict whether a purchase is made at checkout.",
                         "Predict whether the order will arrive late when it is delivered.",
                         "Predict which users run every day.",
                         "At the moment we use a rule-based system that flags 3% of rides.",
                         "Every day we receive 10,000 orders.",
                         "Find tickets that get escalated when the ticket is closed."):
            with self.subTest(sentence=sentence):
                self.assertFalse(self.understand(sentence).prediction_moment.known)
        brief = self.understand("Predict which customers will cancel within 30 days, scored before the renewal call.")
        self.assertEqual(brief.prediction_moment.value, "before the renewal call")

    def test_a_column_is_the_target_only_when_the_sentence_binds_it(self):
        cases = [("The label is whether they upgrade their plan within 30 days.", {"t.csv": ["plan", "upgraded"]}),
                 ("Predict which customers will churn next month. Drop refund_amount: it is written after the outcome.",
                  {"t.csv": ["refund_amount", "churned"]}),
                 ("orders.csv records each order placed in 2023 with its amount and status; predict whether it is returned.",
                  {"orders.csv": ["amount", "status", "returned"]})]
        for text, files in cases:
            with self.subTest(text=text[:40]):
                self.assertFalse(self.understand(text, files).target_column.known)

    def test_volume_and_unrelated_costs_are_not_constraints(self):
        brief = self.understand("We process 50,000 rides a day and want to detect fraudulent rides.")
        self.assertEqual([c.kind for c in brief.constraints], [])
        self.assertIn("action", [q.field for q in brief.open_questions])
        self.assertEqual(self.understand("Estimate the cost of each repair from the machine's history.").constraints, [])

    def test_the_task_comes_from_the_goal_sentence_first(self):
        self.assertEqual(self.understand("Predict which customers will churn next month. We do not know how much history matters.").task.value, "binary")
        self.assertEqual(self.understand("Detect fraudulent rides; one of 3 analysts reviews each flagged ride.").task.value, "binary")


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class HomeTest(unittest.TestCase):
    """Home keeps the brief on the draft and takes from it only what is safe: a bound target and the action."""

    def start(self, sentence):
        import tempfile

        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create(sentence)
        HomeAgent(store, None).start(draft["id"])
        return store.get(draft["id"])

    def test_home_keeps_the_brief_and_takes_the_action_from_it(self):
        draft = self.start("Estimate how many minutes a food delivery will take, to show the customer when they place the order.")
        self.assertEqual(draft["brief"]["schema_version"], "dclab_core.problem_brief.v1")
        self.assertEqual(draft["brief"]["task"]["value"], "regression")
        self.assertEqual(draft["brief"]["prediction_moment"]["value"], "when they place the order")
        self.assertEqual(draft["plan"]["action"]["status"], "stated")
        self.assertEqual(draft["plan"]["action"]["quote"], "to show the customer when they place the order")

    def test_home_takes_no_moment_from_the_brief(self):
        draft = self.start("Estimate how many minutes a food delivery will take, to show the customer when they place the order.")
        self.assertNotEqual(draft["plan"]["prediction_moment"].get("source"), "brief")
        for sentence in ("Predict which customers will cancel before the renewal.", "Predict whether a purchase is made at checkout."):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.start(sentence)["plan"]["prediction_moment"]["status"], "unknown")

    def test_home_does_not_take_volume_as_the_action(self):
        draft = self.start("We process 50,000 rides a day and want to detect fraudulent rides.")
        self.assertEqual(draft["plan"]["action"]["status"], "unknown")


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class InternToolTest(unittest.TestCase):
    def test_the_intern_reads_a_project_into_a_brief(self):
        import tempfile

        from dclab_rnd.agents import default_registry
        from dclab_rnd.studio import ProjectStore

        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Delivery", "food", DELIVERY)
        project["data"] = {"filename": "orders.csv", "columns": DELIVERY_FILES["orders.csv"]}
        store.save(project)
        registry = default_registry()
        tool = registry.get("understand_problem")
        self.assertEqual((tool.scope, tool.effect), ("project", "read"))
        from dclab_rnd.intern.tools import Toolbox

        out = Toolbox(store).understand_problem(project["id"])
        self.assertEqual(out["brief"]["task"]["value"], "regression")
        self.assertEqual(out["brief"]["target_column"]["value"], "actual_minutes")
        compact = Toolbox._compact_brief(store.get(project["id"]))  # what describe_data hands the intern first
        self.assertEqual(compact["known"]["task"], {"value": "regression", "how": "inferred"})
        self.assertLessEqual(len(compact["open_questions"]), 4)


if __name__ == "__main__":
    unittest.main()
