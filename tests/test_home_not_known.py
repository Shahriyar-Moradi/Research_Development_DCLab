"""A person who does not know says so (S8 B3, live 2026-10-25): DCLab recorded "I don't know." as the answer (46
prediction moments, 33 outcomes and 31 families of 589 written people; 4 projects built with it as their moment). Such
an answer is never a value: the question is closed as declined and never asked again, the field stays unknown, and
only the prediction moment is assumed, said on the card as an assumption, never as the person's words."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft import chat  # noqa: E402
from dclab_rnd.draft.card import card_of  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class NotKnownTest(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.agent = chat.HomeAgent(self.store, None)
        self.draft = self.store.create("Predict which customers cancel")

    def test_i_dont_know_is_never_recorded_as_an_answer(self):
        for said in ("I don't know.", "i dont know", "Not sure!", "no idea", "I do not know yet"):
            self.assertTrue(chat.not_known(said), said)
        for said in ("Later, when the order is delivered", "I don't know the courier yet, but at checkout", "Each morning at 8"):
            self.assertFalse(chat.not_known(said), said)  # a real answer that mentions not knowing is an answer
        self.agent.record(self.draft["id"], "target", "I don't know.")
        self.agent.note(self.draft["id"], "family", "I don't know", "stated", "person", "I don't know")
        u = self.store.get(self.draft["id"])["understanding"]
        self.assertNotIn("target", u)
        self.assertNotIn("family", u)
        self.assertEqual(u["not_known"], ["family", "target"])

    def test_an_unknown_moment_is_assumed_and_said_so_and_the_question_is_closed(self):
        def ask(d):
            d.setdefault("questions", []).append({"id": "q1", "field": "prediction_moment", "text": "When is the prediction made?"})
        self.store.update(self.draft["id"], ask)
        self.agent.record(self.draft["id"], "prediction_moment", "I don't know.")
        d = self.store.get(self.draft["id"])
        self.assertEqual(d["understanding"]["prediction_moment"], chat.ASSUMED_MOMENT)
        q = d["questions"][0]
        self.assertTrue(q["declined"])
        fact = card_of(d).brief.prediction_moment
        self.assertEqual((fact.status, fact.source), ("inferred", "rule"))  # an assumption, never the person's words
        self.agent.record(self.draft["id"], "prediction_moment", "At checkout, before a courier is picked")
        fact = card_of(self.store.get(self.draft["id"])).brief.prediction_moment
        self.assertEqual(fact.status, "stated")  # a moment the person then says is theirs

    def test_an_outcome_said_as_no_outcome_reads_the_family_where_the_card_read_none(self):
        """S9.4 live: Claude recorded "No outcome column. Unsupervised clustering to discover customer types" as the
        outcome; the card read no family from it and waited on a question the person could not answer."""
        def put(d):
            d["analysis"] = {"profile": {"rows": 300, "columns": [{"name": c} for c in ("customer_id", "orders_per_month", "avg_basket")]}}
            d.setdefault("understanding", {})["target"] = "No outcome column. Unsupervised clustering to discover customer types"
        self.store.update(self.draft["id"], put)
        self.assertEqual(card_of(self.store.get(self.draft["id"])).family.value, "segmentation")
        self.store.update(self.draft["id"], lambda d: d["understanding"].update(target="orders_per_month"))
        self.assertNotEqual(card_of(self.store.get(self.draft["id"])).family.value, "segmentation")  # a column: never re-read

    def test_a_column_named_in_quotes_is_the_outcome(self):  # S9.4 live: 'The "urgent" column (0/1)' stayed words
        d = {"analysis": {"columns": [{"name": c} for c in ("item", "author", "text", "urgent")]}}
        self.assertEqual(chat.HomeAgent._as_column(d, "target", 'The "urgent" column (0/1): whether the email is urgent'), "urgent")
        self.assertEqual(chat.HomeAgent._as_column(d, "target", "the \u201curgent\u201d flag"), "urgent")
        self.assertEqual(chat.HomeAgent._as_column(d, "target", "Whether the email is urgent"), "Whether the email is urgent")  # only mentioned

    def test_i_dont_know_to_a_models_question_closes_it_before_the_models_turn(self):
        """The 45 live people again (2026-10-26): Claude asked the review budget, the person said they did not know, the
        model's turn recorded nothing and the walk waited on the open question for ever."""
        from unittest import mock

        def ask(d):
            d.setdefault("questions", []).append({"id": "q1", "field": "budget", "text": "How many flags could your team review a day?"})
        self.store.update(self.draft["id"], ask)
        agent = chat.HomeAgent(self.store, object())  # a model is configured: its turn is replaced below
        with mock.patch.object(chat.HomeAgent, "model_turn", lambda *a, **k: None):
            agent.reply(self.draft["id"], "I don't know.")
        d = self.store.get(self.draft["id"])
        self.assertTrue(d["questions"][0]["declined"])
        open_now = agent.pending(d)  # since 2026-10-29 (C1, never silent) Home goes on to the next open question
        self.assertTrue(open_now is None or open_now["field"] != "budget", open_now)  # never the declined one again
        self.assertNotIn("budget", d["understanding"])  # never "I don't know" as the budget

    def test_a_declined_field_is_never_asked_again_by_the_model(self):
        """The 45 again (2026-10-26): after "I don't know" to the entity, the model asked it again in other words, and the
        walk stopped for no progress."""
        from dclab_rnd.draft.chat import DraftGuard

        def put(d):
            d.setdefault("questions", []).append({"id": "q1", "field": "entity", "text": "Who could commit it?", "answered": "I don't know.", "declined": True})
            d.setdefault("understanding", {}).update(target="is_fraud", not_known=["entity"])
        self.store.update(self.draft["id"], put)
        agent = chat.HomeAgent(self.store, None)
        tool = type("T", (), {"move": "ask", "name": "ask_user"})()
        turn = type("Turn", (), {"agent": agent, "draft_id": self.draft["id"]})()
        got = DraftGuard().refusal(tool, turn, {"field": "entity", "question": "What does one row stand for?"})
        self.assertIsNotNone(got)
        self.assertIn("do not ask it again", got["error"])

    def test_im_not_sure_is_not_knowing_and_never_becomes_the_models_answer(self):
        """HOME-003 (the model run of the Home suite, 2026-10-28): to "I'm not sure", the model recorded the outcome in its own
        words as the person's answer, its value as the quote; and "I'm not sure" was not read as not knowing at all."""
        for said in ("I'm not sure", "Im not sure", "I am not sure."):
            self.assertTrue(chat.not_known(said), said)
        self.assertFalse(chat.not_known("I'm not sure, maybe the churn column"))  # an answer with something in it is an answer

        from dclab_rnd.draft.chat import Turn, record

        def ask(d):
            d.setdefault("questions", []).append({"id": "q1", "field": "target", "text": "What should be predicted?"})
            d["messages"] += [{"id": "a1", "role": "agent", "kind": "question", "text": "What should be predicted?", "question": {"id": "q1"}},
                              {"id": "u1", "role": "user", "kind": "text", "text": "I'm not sure"}]
        self.store.update(self.draft["id"], ask)
        agent = chat.HomeAgent(self.store, None)
        out = record(Turn(agent, self.draft["id"]), field="target", value="Customers who cancel within 30 days")
        d = self.store.get(self.draft["id"])
        self.assertTrue(out.get("not_known"))
        self.assertNotIn("target", d["understanding"])  # never the model's words as the person's answer
        self.assertTrue(d["questions"][0]["declined"])

    def test_an_answer_recorded_by_the_model_is_quoted_from_the_persons_reply(self):
        from dclab_rnd.draft.chat import Turn, record

        def ask(d):
            d.setdefault("questions", []).append({"id": "q1", "field": "target", "text": "What should be predicted?"})
            d["messages"] += [{"id": "a1", "role": "agent", "kind": "question", "text": "What should be predicted?", "question": {"id": "q1"}},
                              {"id": "u1", "role": "user", "kind": "text", "text": "the ones who cancel in the next month"}]
        self.store.update(self.draft["id"], ask)
        record(Turn(chat.HomeAgent(self.store, None), self.draft["id"]), field="target", value="Customers who cancel within 30 days")
        d = self.store.get(self.draft["id"])
        self.assertEqual(d["plan"]["target"]["quote"], "the ones who cancel in the next month")  # the person's words


if __name__ == "__main__":
    unittest.main()
