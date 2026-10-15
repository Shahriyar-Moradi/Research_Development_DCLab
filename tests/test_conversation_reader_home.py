"""20.10 v4: the conversation reader on Home. It reads the problem sentence and the answers that settle the card; every
value it gives quotes the person's words (or is refused); a family the person names, or a rule read outright, is never
replaced; and it is on only when switched on and a model serves it. No model is reached: the advisor is scripted."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.agents import ScriptedAdvisor
    from dclab_rnd.draft import reader as home_reader
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


class FakeGateway:
    def __init__(self, serves=True):
        self.serves = serves

    def available(self, purpose):
        return self.serves and purpose == "conversation_reader"


def scripted_reader(script):
    """The product's reader with a scripted advisor in place of the gateway's."""
    advisor = ScriptedAdvisor({"conversation_reader": script})
    advisor.spent_eur = lambda: 0.0
    with mock.patch("dclab_rnd.core_advisor.GatewayAdvisor", return_value=advisor):
        return home_reader.reader(FakeGateway(), force=True), advisor


@unittest.skipIf(MISSING, MISSING)
class ReaderOnHomeTest(unittest.TestCase):
    def agent(self, reader, problem):
        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None, reader=reader)
        draft_id = store.create(problem)["id"]
        return store, agent, draft_id

    def test_the_problem_sentence_is_read_and_its_family_quotes_the_person(self):
        problem = "Last time half the list got the coupon. Next time I only want to give it where it changes something."
        read, advisor = scripted_reader(lambda e: {"family": {"value": "uplift", "quote": "where it changes something"}}
                                        if e["question_field"] == "problem" else {})
        store, agent, draft_id = self.agent(read, problem)
        agent.read_problem(draft_id)
        u = store.get(draft_id)["understanding"]
        self.assertEqual((u["family"], u["family_by"], u["family_shape_only"]), ("uplift", "reader", "no"))
        self.assertEqual(advisor.requests[0][1]["question_field"], "problem")

    def test_a_family_without_the_persons_words_is_refused(self):
        read, _ = scripted_reader({"family": {"value": "uplift", "quote": "words nobody wrote"}})
        store, agent, draft_id = self.agent(read, "Which customers will leave us next month?")
        agent.read_problem(draft_id)
        self.assertNotIn("family", store.get(draft_id).get("understanding") or {})

    def test_a_shape_gives_way_to_the_family_the_reader_quotes_and_a_named_family_never_does(self):
        read, _ = scripted_reader({"family": {"value": "text_or_images", "quote": "from what the note says"}})
        store, agent, draft_id = self.agent(read, "Triage our notes")
        agent.read_card_answer(draft_id, "family", "a yes or no per note, from what the note says")
        self.assertEqual(store.get(draft_id)["understanding"]["family"], "text_or_images")
        store, agent, draft_id = self.agent(read, "Triage our notes")
        agent.read_card_answer(draft_id, "family", "a classification, from what the note says")
        self.assertEqual(store.get(draft_id)["understanding"]["family"], "classification")  # the person named it

    def test_the_readers_own_shape_stays_a_shape_and_a_ranking_takes_no_outcome_column(self):
        read, _ = scripted_reader({"family": {"value": "classification", "quote": "yes or no per note"}})
        store, agent, draft_id = self.agent(read, "Can a computer read the notes first?")
        agent.read_card_answer(draft_id, "family", "yes or no per note, from what it says")
        self.assertEqual(store.get(draft_id)["understanding"]["family_shape_only"], "yes")  # the data may still say long text
        agent.read_problem(draft_id)
        read, _ = scripted_reader({"target_column": {"value": "event", "quote": "the event column"}})
        store, agent, draft_id = self.agent(read, "I want build a recommender system, top-10 for each user")
        store.update(draft_id, lambda d: d.update(analysis={"profile": {"columns": [{"name": n} for n in ("user_id", "item_id", "event")]}},
                                                  understanding={"family": "ranking"}))
        agent.read_card_answer(draft_id, "target", "purchase events in the event column")
        self.assertNotIn("target", store.get(draft_id)["understanding"])

    def test_an_outcome_column_the_reader_quotes_is_the_target(self):
        read, _ = scripted_reader({"target_column": {"value": "days_subscribed", "quote": "the days_subscribed numbers"}})
        store, agent, draft_id = self.agent(read, "How long do members stay?")
        store.update(draft_id, lambda d: d.update(analysis={"profile": {"columns": [{"name": n} for n in ("days_subscribed", "cancelled")]}}))
        agent.read_card_answer(draft_id, "target", "mostly we look at the days_subscribed numbers")
        self.assertEqual(store.get(draft_id)["understanding"]["target"], "days_subscribed")

    def test_off_unless_switched_on_served_and_measured(self):
        from dclab_rnd.agent_eval import journeys

        with mock.patch.dict(os.environ, {"DCLAB_CONVERSATION_READER": ""}), mock.patch.object(journeys, "reader_passed", return_value=True):
            self.assertIsNone(home_reader.reader(FakeGateway()))
        with mock.patch.dict(os.environ, {"DCLAB_CONVERSATION_READER": "1"}):
            with mock.patch.object(journeys, "reader_passed", return_value=True):
                self.assertIsNone(home_reader.reader(FakeGateway(serves=False)))
                self.assertIsNotNone(home_reader.reader(FakeGateway()))
            with mock.patch.object(journeys, "reader_passed", return_value=False):
                self.assertIsNone(home_reader.reader(FakeGateway()))  # switched on, but not measured better: off

    def test_a_column_mentioned_as_known_later_is_never_the_outcome_even_when_quoted(self):
        read, _ = scripted_reader({"target_column": {"value": "final_refund_amount", "quote": "final_refund_amount is paid"}})
        store, agent, draft_id = self.agent(read, "Find fake orders")
        store.update(draft_id, lambda d: d.update(analysis={"profile": {"columns": [{"name": n} for n in ("final_refund_amount", "is_fraud")]}}))
        agent.read_card_answer(draft_id, "target", "fraud; we only know it once final_refund_amount is paid")
        self.assertNotIn("target", store.get(draft_id).get("understanding") or {})

    def test_a_family_the_person_stated_is_never_replaced_by_one_read_in_another_answer(self):
        read, _ = scripted_reader(lambda e: {"family": {"value": "rare_event", "quote": "the rare cases we confirmed"}}
                                  if e["question_field"] == "outcome" else {})
        store, agent, draft_id = self.agent(read, "Who churns?")
        agent.read_card_answer(draft_id, "family", "a classification")
        agent.read_card_answer(draft_id, "target", "churned, the rare cases we confirmed")
        self.assertEqual(store.get(draft_id)["understanding"]["family"], "classification")

    def test_the_readers_labels_reading_is_recorded_where_the_rules_read_none(self):
        read, _ = scripted_reader({"labels": {"value": "file", "quote": "the audit team's spreadsheet"}})
        store, agent, draft_id = self.agent(read, "Find fake rides")
        store.update(draft_id, lambda d: d.update(understanding={"labels": "the audit team's spreadsheet"}))
        agent.read_card_answer(draft_id, "labels", "the audit team's spreadsheet")
        self.assertEqual(store.get(draft_id)["understanding"]["labels"], HomeAgent.LABEL_PHRASES["file"])

    def test_a_failing_reader_leaves_the_rules_reading(self):
        def broken(field, text, columns):
            raise RuntimeError("down")

        store, agent, draft_id = self.agent(broken, "Find the rides that look strange; nothing is labelled")
        agent.read_card_answer(draft_id, "family", "the unusual ones")
        self.assertEqual(store.get(draft_id)["understanding"]["family"], "anomaly")


@unittest.skipIf(MISSING, MISSING)
class CardPrecedenceTest(unittest.TestCase):
    """The card weighs a shape and the reader's family below what the person stated and what the data says."""

    def card(self, problem, understanding, columns=("late", "minutes_late"), label_file=False):
        from dclab_rnd.draft.card import card_of

        draft = {"problem": problem, "understanding": understanding, "solution": {},
                 "analysis": {"profile": {"columns": [{"name": c} for c in columns]}},
                 "assets": [{"id": "a", "status": "ready", "name": "t.csv"}] + ([{"id": "l", "status": "ready", "kind": "labels"}] if label_file else []),
                 "active_asset": "a"}
        with mock.patch("dclab_rnd.draft.card.plans.label_asset", return_value={"id": "l"} if label_file else None):
            return card_of(draft)

    def test_a_shape_the_inference_contradicts_stands(self):
        c = self.card("How many minutes late will each delivery be?", {"family": "classification", "family_shape_only": "yes"})
        self.assertEqual(c.family.value, "classification")

    def test_the_readers_family_gives_way_to_a_label_file(self):
        c = self.card("Find the unusual rides", {"family": "anomaly", "family_by": "reader"}, columns=("fare", "order_id"), label_file=True)
        self.assertEqual(c.family.value, "rare_event")


if __name__ == "__main__":
    unittest.main()
