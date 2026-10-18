"""Package 28.4: the understanding critic: its rules read the card against the data card; a model may add concerns, only
about card fields and columns the tables hold; each concern a question the interviewer asks next; a field never sent back
twice, at most two rounds; the walk goes on after it."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.understanding_critic import FIELDS, ROUNDS, concerns, critique
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""

COLUMNS = [{"name": "order_id", "purpose": "identifier", "role": "identifier"}, {"name": "created_at", "purpose": "time", "role": "datetime"},
           {"name": "amount", "purpose": "feature", "role": "numeric"}, {"name": "is_fraud", "purpose": "outcome", "role": "categorical_code"}]


@unittest.skipIf(MISSING, MISSING)
class RuleTests(unittest.TestCase):
    def test_an_identifier_as_the_outcome_and_a_missing_moment_are_sent_back(self):
        got = concerns({"outcome": "order_id", "moment": None}, COLUMNS)
        self.assertEqual([c["field"] for c in got], ["target", "prediction_moment"])
        self.assertIn("order_id", got[0]["question"])
        self.assertEqual(got[1]["columns"], ["created_at"])

    def test_a_card_that_holds_raises_nothing_and_a_mismatch_is_carried(self):
        self.assertEqual(concerns({"outcome": "is_fraud", "moment": "when the order is placed"}, COLUMNS), [])
        got = concerns({"outcome": "is_fraud", "moment": "x"}, COLUMNS, [{"code": "target_absent", "question": "Which column?"}])
        self.assertEqual((got[0]["field"], got[0]["question"]), ("target", "Which column?"))


@unittest.skipIf(MISSING, MISSING)
class ModelTests(unittest.TestCase):
    def test_a_model_adds_concerns_after_the_rules_and_one_per_field(self):
        advisor = ScriptedAdvisor({"understanding_critic": {"concerns": [
            {"field": "prediction_moment", "question": "model's moment question", "why": "w", "columns": ["created_at"]},
            {"field": "action", "question": "What will the team do with each flagged order?", "why": "the budget decides the threshold", "columns": []}]}})
        out = critique(advisor, {"outcome": "order_id", "moment": None}, COLUMNS)
        self.assertEqual(out.source, "advisor")
        self.assertEqual([c["field"] for c in out.output], ["target", "prediction_moment", "action"])
        self.assertNotEqual(out.output[1]["question"], "model's moment question")  # the rules' concern stands for its field
        from dclab_core.advisor import check_evidence

        check_evidence(advisor.requests[0][1])  # a summary, never a table: the guard lets it through

    def test_a_concern_naming_an_absent_column_or_field_refuses_the_answer(self):
        for bad in ({"field": "action", "question": "q", "columns": ["driver_rating"]}, {"field": "budget_of_mars", "question": "q"}):
            out = critique(ScriptedAdvisor({"understanding_critic": {"concerns": [bad]}}), {"outcome": "is_fraud", "moment": "m"}, COLUMNS)
            self.assertEqual((out.source, out.output), ("fallback", []))  # refused: the rules stand

    def test_the_fields_and_rounds_are_bounded(self):
        self.assertIn("target", FIELDS)
        self.assertEqual(ROUNDS, 2)

    def test_a_name_in_a_concerns_words_must_be_a_column(self):  # the review of 28.4, finding 9
        bad = {"field": "entity", "question": "Is it per driver_rating?", "columns": []}
        out = critique(ScriptedAdvisor({"understanding_critic": {"concerns": [bad]}}), {"outcome": "is_fraud", "moment": "m"}, COLUMNS)
        self.assertEqual((out.source, out.output), ("fallback", []))
        fine = {"field": "entity", "question": "Is each order_id one case, or can an order be scored twice?", "columns": []}
        out = critique(ScriptedAdvisor({"understanding_critic": {"concerns": [fine]}}), {"outcome": "is_fraud", "moment": "m"}, COLUMNS)
        self.assertEqual([c["field"] for c in out.output], ["entity"])

    def test_each_mismatch_is_sent_back_on_the_field_it_names(self):  # the review of 28.4: time and grain were "family"
        got = concerns({"outcome": "is_fraud", "moment": "m"}, COLUMNS, [{"code": "time_absent", "question": "When?"},
                                                                         {"code": "unit_grain", "question": "Per what?"},
                                                                         {"code": "labels_contradict", "question": "Labels?"}])
        self.assertEqual([c["field"] for c in got], ["prediction_moment", "entity", "labels"])


@unittest.skipIf(MISSING, MISSING)
class WalkTests(unittest.TestCase):
    def test_the_critic_sends_back_once_the_person_answers_and_the_walk_builds(self):
        import tempfile

        from dclab_rnd import experience
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.walk import Walk, drive
        from dclab_rnd.studio import ProjectStore, team
        from tests.test_coordinator import ANSWERS, PROBLEM, dfr, upload

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "w")
        experience.install(log)
        self.addCleanup(experience.install, None)
        store, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        agent = HomeAgent(store, None)
        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        orders, cases = replica.rides, replica.sample.assign(final_status="DELIVERED")
        d1 = store.create(PROBLEM)["id"]
        agent.start(d1)
        upload(store, agent, d1, "part2-dataset.csv", orders)
        d2 = store.create(PROBLEM)["id"]
        agent.start(d2)
        upload(store, agent, d2, "part2-fraud-sample.csv", cases)
        advisor = ScriptedAdvisor({"understanding_critic": [{"concerns": [{"field": "action", "question": "What does the team do with each flagged order, and how many can it review a day?",
                                                                           "why": "the review budget sets the threshold", "columns": []}]},
                                                            {"concerns": [{"field": "action", "question": "again", "columns": []}]}]})
        asked = []
        person = team.Person(answer=lambda field, q: asked.append((field, q)) or ANSWERS.get(field) or "We review the top 80 a day",
                             approve=lambda gate, summary: True, who="p")
        with checkpoints.open_checkpointer(home, database_url="") as saver:
            out = drive(Walk(Coordinator(store, projects, agent), saver, home=home, critic_advisor=advisor), d2, person)
        self.assertEqual(out["next"], "done", out["says"])
        critic_steps = [s for s in out["steps"] if s["action"] == "critic"]
        self.assertEqual(critic_steps[0]["result"]["sent_back"], ["action"])
        self.assertTrue(any(q.startswith("What does the team do") for f, q in asked))
        self.assertEqual(sum(1 for f, q in asked if q == "again"), 0)  # a field is never sent back twice
        own = store.get(d2)["coordinator"]["critic"]
        self.assertEqual((own["passed"], own["asked"]), (True, ["action"]))
        handoffs = [e.payload["what"] for e in log.read() if e.kind == "team.handoff"]
        self.assertIn("send_back", handoffs)
        self.assertIn("critique_passed", handoffs)
        self.assertEqual([h["sent_back"] for h in own["history"]], ["action", None])  # every round kept with its reasons
        self.assertEqual(own["history"][1]["refused"][0]["refused"], "the person answered it already")  # never asked twice


@unittest.skipIf(MISSING, MISSING)
class RoundTests(unittest.TestCase):
    """The critic in the walk on the cases draft, a model raising a new concern every round."""

    @classmethod
    def setUpClass(cls):
        from tests.test_coordinator import dfr

        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        cls.orders, cls.cases = replica.rides, replica.sample.assign(final_status="DELIVERED")

    silent = False  # a person who cannot answer the critic's questions (28.7's measurement)

    def walk(self, rounds, approve):
        import tempfile

        from dclab_rnd import experience
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.walk import Walk, drive
        from dclab_rnd.studio import ProjectStore, team
        from tests.test_coordinator import ANSWERS, PROBLEM, upload

        home = Path(tempfile.mkdtemp())
        experience.install(experience.FileLog(home / "experience", "w"))
        self.addCleanup(experience.install, None)
        store, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        agent = HomeAgent(store, None)
        orders = store.create(PROBLEM)["id"]  # the table the cases belong to, in another draft (brought by the walk)
        agent.start(orders)
        upload(store, agent, orders, "part2-dataset.csv", self.orders)
        d = store.create(PROBLEM)["id"]
        agent.start(d)
        upload(store, agent, d, "part2-fraud-sample.csv", self.cases)
        advisor = ScriptedAdvisor({"understanding_critic": [{"concerns": [{"field": f, "question": f"Round {i}: is the {f} right?", "columns": []}]}
                                                            for i, f in enumerate(rounds)]})
        self.asked, self.gates = [], []
        person = team.Person(answer=lambda field, q: self.asked.append(q) or ((None if q.startswith("Round") else ANSWERS.get(field)) if self.silent else (ANSWERS.get(field) or "Each order")),
                             approve=lambda gate, summary: self.gates.append((gate, summary)) or approve(gate), who="p")
        from unittest import mock

        from dclab_rnd.orchestra import walk as W

        with checkpoints.open_checkpointer(home, database_url="") as saver, mock.patch.object(W, "_answered", return_value=set()):
            out = drive(Walk(Coordinator(store, projects, agent), saver, home=home, critic_advisor=advisor), d, person)
        return out, store.get(d), store  # (the fields the Home agent asked are open to the model's doubts here: the bound alone)

    def test_at_most_two_rounds_then_the_person_decides(self):  # the review of 28.4: a third round was possible
        out, draft, _ = self.walk(["entity", "labels", "family", "entity", "labels"], approve=lambda gate: True)
        self.assertEqual(out["next"], "done", out["says"])
        self.assertEqual(sum(q.startswith("Round") for q in self.asked), ROUNDS)  # two send-backs, never a third
        own = draft["coordinator"]["critic"]
        self.assertEqual((own["rounds"], own["overridden"], own["passed"]), (ROUNDS, True, True))
        self.assertIn("understanding", [g for g, _ in self.gates])  # the concern that stood was shown to the person

    def test_a_critic_question_with_no_answer_is_declined_and_the_person_decides_at_the_gate(self):  # 28.7: it stopped six of fifteen
        self.silent = True
        out, draft, _ = self.walk(["entity", "entity"], approve=lambda gate: True)
        self.assertEqual(out["next"], "done", out["says"])  # never "Waiting for the person"
        declined = [q for q in draft["questions"] if q.get("declined")]
        self.assertEqual([q["field"] for q in declined], ["entity"])
        self.assertIn("understanding", [g for g, _ in self.gates])  # the concern stood and was shown before the build
        self.assertTrue(draft["coordinator"]["critic"]["overridden"])

    def test_a_declined_understanding_gate_stops_with_the_concerns(self):
        out, draft, _ = self.walk(["entity", "labels", "family", "entity", "labels"], approve=lambda gate: gate != "understanding")
        self.assertEqual(out["next"], "stop")
        self.assertIn("concerns stand", out["says"])
        self.assertFalse(draft.get("project_id"))
        self.assertTrue(draft["coordinator"]["critic"]["awaiting"])

    def test_yes_to_a_critics_family_question_keeps_the_family(self):  # the review of 28.4, finding 3
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.card import card_of

        out, draft, store = self.walk([], approve=lambda gate: True)
        d = draft["id"]
        before = (draft.get("understanding") or {}).get("family")
        status = card_of(draft).family.status
        agent = HomeAgent(store, None)
        agent.ask(d, "family", f"Is this {before}?", source="critic")
        agent.reply(d, "Yes.")
        after = store.get(d)
        self.assertEqual((after.get("understanding") or {}).get("family"), before)
        self.assertEqual(card_of(after).family.status, status)


if __name__ == "__main__":
    unittest.main()
