"""Package 20.2: the understanding conversation. The next question is the one that unblocks the most; an answer is
read into the card by rules ("it is anomaly detection" states the family whatever was asked); a model reader is kept
only behind its validator; written conversations, one per family, reach the right card in at most five questions on
dev, and the sealed sets were each run once and are refused a second time; Home records a family said in an answer
and shows the card to confirm."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import conversation as cv
    from dclab_core.agents import ScriptedAdvisor, run_role
    from dclab_rnd.agent_eval import conversation as harness
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class RulesTest(unittest.TestCase):
    def test_an_answer_is_read_into_card_fields(self):
        self.assertEqual(cv.read_answer("outcome", "it is anomaly detection"), {"family": "anomaly"})  # the morning's answer
        self.assertEqual(cv.read_answer("labels", "A file lists the known ones"), {"labels": "file"})
        self.assertEqual(cv.read_answer("labels", "Nothing is labelled"), {"labels": "none"})
        self.assertEqual(cv.read_answer("outcome", "the churned column", ["id", "churned"]), {"target_column": "churned"})
        self.assertEqual(cv.read_answer("outcome", "I don't know"), {})
        self.assertEqual(cv.read_answer("prediction_moment", "how much is known at noon"), {"prediction_moment": "how much is known at noon"})

    def test_a_word_in_passing_never_states_a_kind_of_answer(self):
        for field, text in (("labels", "No known cases"), ("outcome", "whether a customer cancels in the next month"),
                            ("action", "the team reviews the unusual ones"), ("prediction_moment", "when the customer segment is assigned"),
                            ("prediction_moment", "at the time to approve the loan"), ("action", "the top ranked ones are reviewed"),
                            ("action", "we recommend a follow-up call"), ("action", "our team decides who to contact first")):
            with self.subTest(text=text):
                self.assertNotIn("family", cv.read_answer(field, text))
        self.assertEqual(cv.read_answer("labels", "No known cases"), {"labels": "none"})  # never rare events without cases
        self.assertNotIn("family", cv.read_answer("family", "rare cases we learn from, but none are confirmed"))

    def test_the_kind_of_answer_is_read_in_the_right_order(self):
        for text, family in (("how many orders are late, a yes or no for each", "classification"), ("whether the price goes up", "classification"),
                             ("a number, the price", "regression"), ("forecast how many calls next week", "forecasting"),
                             ("groups, nothing to predict", "segmentation"), ("the unusual ones", "anomaly")):
            with self.subTest(text=text):
                self.assertEqual(cv.read_answer("family", text).get("family"), family)

    def test_labels_and_outcome_columns_are_read_whole(self):
        self.assertEqual(cv.read_answer("labels", "No file, but the is_fraud column marks them", ["is_fraud"]), {"labels": "column"})
        self.assertEqual(cv.read_answer("labels", "the profile says the type"), {})  # "profile" holds no "file"
        self.assertEqual(cv.read_answer("outcome", "the anomaly_score column", ["anomaly_score"]), {"target_column": "anomaly_score"})

    def test_durations_that_are_not_events_stay_numbers(self):
        from dclab_core.card import WORDS

        for text in ("Predict the time to process returned parcels", "the time to refuel diesel trucks", "the time until breakfast orders are ready"):
            self.assertFalse(WORDS["time_to_event"].search(text), text)
        for text in ("Estimate the time until each customer cancels", "the time before each pump breaks down", "How long until each machine fails"):
            self.assertTrue(WORDS["time_to_event"].search(text), text)

    def test_the_question_that_unblocks_the_most_comes_first(self):
        card = cv.card_for("I want build anomaly detection", {}, ["customer_id", "fare"])
        self.assertEqual(cv.next_question(card).field, "labels")  # rare events or anomalies: the labels decide
        open_kind = cv.card_for("Tell which claims will be disputed", {}, ["claim_id", "disputed"])
        self.assertEqual(cv.next_question(open_kind).field, "outcome")  # its values settle the kind of answer
        self.assertEqual(cv.next_question(card, contradictions=(type("C", (), {"code": "labels_contradict"})(),)).code, "labels_contradict")
        files_first = cv.card_for("Tell which claims will be disputed; a file lists the known ones", {}, ["claim_id"], data={"tables": 1})
        self.assertEqual(cv.next_question(files_first).field, "data")  # the file before anything it would settle
        ready = cv.card_for("Predict churn", {"target_column": "churned", "prediction_moment": "monthly"}, ["churned"], kinds={"churned": "binary"})
        self.assertIsNone(cv.next_question(ready))
        self.assertEqual(cv.summary(ready)["family"], "classification")

    def test_a_conversation_is_never_longer_than_five_questions(self):
        played = cv.converse("Something about our data", {}, ["a", "b"])
        self.assertLessEqual(len(played["asked"]), cv.LIMIT)


@unittest.skipIf(MISSING, MISSING)
class ReaderTest(unittest.TestCase):
    def run_reader(self, reply, answer="we never confirmed any of them, they are just odd", columns=("fare",)):
        advisor = ScriptedAdvisor({"conversation_reader": reply})
        return run_role(cv.READER, advisor, {"field": "labels", "answer": answer, "columns": list(columns)})

    def test_a_proposal_quoted_from_the_answer_is_kept(self):
        out = self.run_reader({"labels": {"value": "none", "quote": "we never confirmed any of them"},
                               "family": {"value": "anomaly", "quote": "they are just odd"}})
        self.assertEqual((out.source, out.output.get("labels"), out.output.get("family")), ("advisor", "none", "anomaly"))

    def test_what_the_person_did_not_say_is_refused(self):
        for reply in ({"labels": {"value": "none", "quote": "nothing at all"}},       # a quote not in the answer
                      {"family": {"value": "astrology", "quote": "they are just odd"}},  # not a family
                      {"target_column": {"value": "is_fraud", "quote": "odd"}},      # not a column of the table
                      {"labels": {"value": "none", "quote": "are"}},                  # a quote too short to say anything
                      {"family": {"value": "anomaly", "quote": "a"}},
                      {"target_column": {"value": "fare", "quote": "they are just odd"}},  # a column its quote does not name
                      {"prediction_moment": {"value": "after the final fare is known", "quote": "they are just odd"}}):  # a moment the model wrote
            with self.subTest(reply=reply):
                self.assertEqual(self.run_reader(reply).source, "fallback")


@unittest.skipIf(MISSING, MISSING)
class HarnessTest(unittest.TestCase):
    def test_the_score_and_the_verdict(self):
        case = {"id": "x", "family": "anomaly", "gold": {"family": "anomaly", "labels": "none"}}
        good = {"asked": ["labels"], "summary": {"family": "anomaly", "labels": "none", "ready": True}}
        self.assertTrue(harness.score(case, good)["right"])
        for why, played in (("labels", {**good, "summary": {**good["summary"], "labels": "file"}}),
                            ("not ready", {**good, "summary": {**good["summary"], "ready": False}}),
                            ("too many questions", {**good, "asked": ["a"] * 6})):
            self.assertEqual(harness.score(case, played)["why"], why)
        rules = [{"id": "a", "right": True}, {"id": "b", "right": False}]
        self.assertEqual(harness.verdict(rules, "sealed"), "fail")  # the rules must get every one
        self.assertEqual(harness.verdict([{"id": "a", "right": True}], "sealed"), "pass")
        self.assertEqual(harness.verdict([{"id": "a", "right": False}, {"id": "b", "right": True}], "sealed", rules), "fail")  # turned one wrong
        self.assertEqual(harness.verdict([{"id": "a", "right": True}, {"id": "b", "right": True}], "sealed", rules), "pass")

    def test_dev_reaches_every_right_card_in_five_questions(self):
        rows = harness.measure("dev")
        self.assertGreaterEqual(len({r["family"] for r in rows}), 9)  # one per family at least
        self.assertEqual([r["id"] for r in rows if not r["right"]], [])
        self.assertTrue(all(r["questions"] <= 5 for r in rows))

    def test_the_sealed_sets_are_pinned_and_run_once(self):
        for split in ("sealed", "sealed_v2"):
            self.assertEqual(len(harness.load(split)), 9)
            self.assertTrue(harness.sealed_runs("rules", split), split)  # run, and stored
        results = {p.name: json.loads(p.read_text()) for p in harness.RESULTS.glob("CONV-*_sealed*_rules.json")}
        self.assertEqual(sorted(r["verdict"] for r in results.values()), ["fail", "fail"])  # both failed, and stay recorded
        from dclab_rnd.agent_eval.conversation.__main__ import main

        for arm in ("rules", "reader"):  # spent for every arm, not only the arm that ran it
            self.assertEqual(main(["--split", "sealed_v2", "--arm", arm, "--yes"]), 2)

    def test_a_changed_sealed_file_is_refused(self):
        import shutil

        tmp = Path(tempfile.mkdtemp())
        for name in ("conversations_sealed.json", "sealed.sha256"):
            shutil.copy(harness.HERE / name, tmp / name)
        (tmp / "conversations_sealed.json").write_text("[]")
        old = harness.HERE
        harness.HERE = tmp
        try:
            with self.assertRaises(ValueError):
                harness.load("sealed")
        finally:
            harness.HERE = old


@unittest.skipIf(MISSING, MISSING)
class HomeTest(unittest.TestCase):
    def test_a_family_said_in_an_answer_is_recorded_and_the_card_is_shown(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None)
        d = store.create("Look at our orders")["id"]
        agent.start(d)
        first = store.get(d)["questions"][0]
        agent.reply(d, "it is anomaly detection")  # typed as the outcome, the morning of 2026-10-13
        self.assertEqual(store.get(d)["understanding"].get("family"), "anomaly", first)
        for answer in ("every night", "we review them", "later"):
            agent.reply(d, answer)
        summary = next(m for m in reversed(store.get(d)["messages"]) if m.get("kind") == "summary")
        self.assertEqual(summary["card"]["family"], "anomaly")
        self.assertEqual(summary["card"]["family_how"], "stated")


@unittest.skipIf(MISSING, MISSING)
class HomeGuardTest(unittest.TestCase):
    def store(self):
        from dclab_rnd.draft.store import DraftStore

        return DraftStore(Path(tempfile.mkdtemp()))

    def test_no_known_cases_in_a_fraud_draft_is_never_a_rare_event_built(self):
        from dclab_rnd.draft.card import card_of
        from dclab_rnd.draft.chat import HomeAgent

        store = self.store()
        agent = HomeAgent(store, None)
        d = store.create("Find fraudulent rides")["id"]
        agent.start(d)
        store.update(d, lambda x: x["questions"].append({"id": "qL", "field": "labels", "text": "?", "options": [], "why": "", "answered": None}))
        agent.reply(d, "No known cases")
        draft = store.get(d)
        self.assertNotIn("family", draft["understanding"])
        self.assertNotEqual(card_of(draft).family.value, "rare_event")

    def test_a_model_turn_reads_the_persons_own_words_for_the_card(self):
        from dclab_rnd.draft.chat import HomeAgent
        from tests.test_draft_flow import ScriptedClient

        store = self.store()
        d = store.create("Look at our orders")["id"]
        HomeAgent(store, None).start(d)
        field = store.get(d)["questions"][0]["field"]
        agent = HomeAgent(store, ScriptedClient([("record", {"field": field, "value": "anomaly detection"})]))
        agent.reply(d, "it is anomaly detection")
        self.assertEqual(store.get(d)["understanding"].get("family"), "anomaly")


if __name__ == "__main__":
    unittest.main()
