"""Package 37.9 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): Home, conversation first. With DCLAB_CASE_AGENT on, the lead agent
leads Home's draft conversation; with it off, or with no model, the rules-first Home does. Scripted models only."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft.case_home import CaseHome  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.draft.work import DraftWork  # noqa: E402

FULL = {"gets": "a ranked weekly list", "tradeoff": "needs past cancellations", "effort": "a day"}
OPTIONS = [{"title": "Rank members by risk", "what": "a score for each member", "track": "prediction", "tool": "ml_project", **FULL},
           {"title": "Who cancelled last year", "what": "cancellations by plan", "track": "analysis", "tool": "query", **FULL}]


class Script:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append(messages)
        text, calls = self.replies.pop(0) if self.replies else ("", [])
        calls = [{"id": f"c{len(self.seen)}-{i}", "name": f"case_{n}", "arguments": a} for i, (n, a) in enumerate(calls)]
        msg = {"role": "assistant", "content": text, **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})}
        return {"content": text, "tool_calls": calls, "assistant_message": msg}


class Models:
    def __init__(self, script):
        self.script = script

    def client(self, purpose, draft_id=None, **k):
        return self.script if purpose == "case_agent" else None

    def available(self, purpose):
        return purpose == "case_agent"


class NoModels:
    """A gateway that serves no purpose (no key, or DCLAB_NO_LIVE_MODELS=1)."""

    def client(self, purpose, draft_id=None, **k):
        return None

    def available(self, purpose):
        return False


class CaseHomeTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.store = DraftStore(self.folder / "drafts")

    def test_the_switch_chooses_the_lead_agent_and_off_keeps_the_rules_first_home(self):
        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": "1"}):
            self.assertIsInstance(DraftWork(self.store, Models(Script())).agent("d"), CaseHome)
        for off in ("0", "off", "false"):
            with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": off}):
                agent = DraftWork(self.store, Models(Script())).agent("d")
                self.assertNotIsInstance(agent, CaseHome)
                self.assertIsInstance(agent, HomeAgent)

    def test_the_lead_agent_is_the_default_where_a_model_serves_it_and_the_rules_first_home_without_one(self):
        """37.11 (the owner, 2026-11-01): unset, the lead agent leads Home when a model serves its purpose; with no model (the
        tests, the open edition) the rules-first Home serves."""
        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": ""}):
            self.assertIsInstance(DraftWork(self.store, Models(Script())).agent("d"), CaseHome)
            agent = DraftWork(self.store, NoModels()).agent("d")
            self.assertNotIsInstance(agent, CaseHome)
            self.assertIsInstance(agent, HomeAgent)

    def test_rules_first_is_the_old_home_whatever_the_switch_says(self):
        """The conversation exam's arm "home" must stay today's Home after the default changed."""
        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": "1"}):
            agent = DraftWork(self.store, Models(Script())).rules_first("d")
            self.assertNotIsInstance(agent, CaseHome)
            self.assertIsInstance(agent, HomeAgent)

    def test_the_exams_home_arm_is_the_rules_first_home_with_a_model_serving_the_lead_agent(self):
        from dclab_rnd.agent_eval.conversation_exam.arms import HomeArm

        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": ""}):
            arm = HomeArm(self.folder / "arm", Models(Script()))
            arm.open({"id": "d-01", "opening": "our members keep cancelling"})
            self.assertNotIsInstance(arm.agent, CaseHome)
            self.assertIsInstance(arm.agent, HomeAgent)

    def test_an_unknown_switch_value_is_the_default(self):
        from dclab_rnd.agents import case_agent

        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": "maybe"}):
            self.assertTrue(case_agent.on(Models(Script())))
            self.assertFalse(case_agent.on(NoModels()))
            self.assertFalse(case_agent.on(None))

    def test_a_draft_keeps_the_home_that_started_it(self):
        """The review of 37.11: a draft the rules-first Home answered stays with it; one with a case stays with the lead agent."""
        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": ""}):
            old = self.store.create("predict churn")
            self.store.update(old["id"], lambda d: d["messages"].append({"role": "agent", "kind": "text", "text": "What is the outcome?"}))
            agent = DraftWork(self.store, Models(Script())).agent(old["id"])
            self.assertNotIsInstance(agent, CaseHome)
            new = self.store.create("predict churn")
            self.store.update(new["id"], lambda d: d.update(case_state={"case": {}}))
            self.store.update(new["id"], lambda d: d["messages"].append({"role": "agent", "kind": "text", "text": "Who leaves?"}))
            self.assertIsInstance(DraftWork(self.store, Models(Script())).agent(new["id"]), CaseHome)

    def _home_with_builds(self, *results):
        calls = []

        def on_request(draft_id, what, args):
            calls.append(what)
            return results[len(calls) - 1] if len(calls) <= len(results) else None
        return CaseHome(self.store, Models(Script()), on_request=on_request), calls

    def _yes(self, state, client, words):
        state.build = {"tool": "ml_project", "quote": "yes"}
        return {"text": "Your yes is recorded."}

    def test_the_persons_yes_starts_the_build_and_an_outcome_named_in_their_answer_is_read_by_code(self):
        """The review of 37.11: the yes the agent records starts the build; the build's outcome question is answered by code."""
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which members cancel")
        self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "member_id"}, {"name": "churned"}]}))
        home, calls = self._home_with_builds({"refused": "Which column is the outcome the model should learn?", "rule": "outcome column"},
                                             {"refused": "When is the prediction made?", "rule": "prediction moment"},
                                             {"refused": "not yet", "rule": None})
        with mock.patch.object(A, "turn", self._yes):
            home.start(d["id"])
        self.assertEqual(calls, ["build"])
        self.assertEqual(self.store.get(d["id"]).get("build_waiting"), "outcome column")
        texts = [m.get("text") for m in self.store.get(d["id"])["messages"]]
        self.assertIn("Which column is the outcome the model should learn?", texts)
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the build's answer")):
            home.reply(d["id"], "the outcome is Churned")
        draft = self.store.get(d["id"])
        self.assertEqual(draft["understanding"]["target"], "churned")
        self.assertEqual(calls, ["build", "build"])
        self.assertEqual(draft.get("build_waiting"), "prediction moment")
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the build's answer")):
            home.reply(d["id"], "at signup, before the first month is paid")
        draft = self.store.get(d["id"])
        self.assertNotIn("prediction_moment", draft["understanding"])  # said back first: the contract is the person's yes
        self.assertEqual(draft.get("build_waiting"), "confirm moment")
        self.assertIn("at signup, before the first month is paid", draft["messages"][-1]["text"])
        self.assertEqual(calls, ["build", "build"])
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the build's answer")):
            home.reply(d["id"], "yes")
        draft = self.store.get(d["id"])
        self.assertEqual(draft["understanding"]["prediction_moment"], "at signup, before the first month is paid")
        self.assertEqual(calls, ["build", "build", "build"])
        self.assertIsNone(draft.get("build_waiting"))

    def test_a_question_or_a_sentence_naming_a_column_in_passing_is_never_taken_as_the_build_answer(self):
        """The review of 37.11: "What do you mean by prediction moment?" became the moment; "revenue by region" made region the outcome."""
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict fraud")
        self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "amount"}, {"name": "is_fraud"}, {"name": "region"}]},
                                                     build_waiting="prediction moment"))
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", return_value={"text": "It is the moment the prediction is made."}) as turn:
            home.reply(d["id"], "What do you mean by prediction moment?")
        self.assertTrue(turn.called)
        self.assertNotIn("prediction_moment", self.store.get(d["id"]).get("understanding") or {})
        self.store.update(d["id"], lambda x: x.update(build_waiting="outcome column"))
        for words in ("Actually I just want revenue by region", "Is amount the right one?"):
            with mock.patch.object(A, "turn", return_value={"text": "Let us look at that."}) as turn:
                home.reply(d["id"], words)
            self.assertTrue(turn.called, words)
            self.assertNotIn("target", self.store.get(d["id"]).get("understanding") or {})
        self.assertEqual(calls, [])

    def test_a_yes_with_a_correction_says_the_new_moment_back_instead_of_building(self):
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which orders are returned")
        self.store.update(d["id"], lambda x: x.update(build_waiting="confirm moment", moment_said="at signup"))
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read it")):
            home.reply(d["id"], "Yes, but it is at checkout, not signup")
        draft = self.store.get(d["id"])
        self.assertEqual(calls, [])
        self.assertEqual(draft.get("build_waiting"), "confirm moment")
        self.assertIn("checkout", draft["moment_said"])
        self.assertNotIn("prediction_moment", draft.get("understanding") or {})

    def test_a_long_yes_to_the_moment_builds_and_is_never_said_back_as_a_new_moment(self):
        """The review of 37.11 in a cloud session: "Yes, that is correct, go ahead" was said back as the moment
        ("So the prediction is made that is correct, go ahead"), and a person answering in full sentences looped."""
        from dclab_rnd.agents import case_agent as A

        for words in ("Yes, that is correct, go ahead", "Yes that's right, build it now please", "Yes, no other changes"):
            d = self.store.create("predict which orders are returned")
            self.store.update(d["id"], lambda x: x.update(build_waiting="confirm moment", moment_said="at checkout"))
            home, calls = self._home_with_builds()
            with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read it")):
                home.reply(d["id"], words)
            draft = self.store.get(d["id"])
            self.assertEqual(calls, ["build"], words)
            self.assertEqual(draft["understanding"]["prediction_moment"], "at checkout", words)

    def test_a_no_with_a_new_moment_is_said_back_and_a_hold_builds_nothing(self):
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which orders are returned")
        self.store.update(d["id"], lambda x: x.update(build_waiting="confirm moment", moment_said="at signup"))
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read it")):
            home.reply(d["id"], "No, it is at checkout")
        draft = self.store.get(d["id"])
        self.assertEqual((calls, draft.get("build_waiting")), ([], "confirm moment"))
        self.assertIn("checkout", draft["moment_said"])
        with mock.patch.object(A, "turn", return_value={"text": "Of course, tell me when."}) as turn:
            home.reply(d["id"], "Yes that's right, but hold off on the build for now")
        self.assertTrue(turn.called)
        draft = self.store.get(d["id"])
        self.assertEqual(calls, [])
        self.assertIsNone(draft.get("build_waiting"))
        self.assertNotIn("prediction_moment", draft.get("understanding") or {})

    def test_groups_then_a_prediction_asks_for_the_outcome_again(self):
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("group my members")
        self.store.update(d["id"], lambda x: x.update(understanding={"family": "segmentation", "family_by": "answer"}))
        home, _ = self._home_with_builds()

        def to_prediction(state, client, words):
            state.case = state.case.model_copy(update={"route": C.CaseRoute(track="prediction", tool="ml_project", by="agent", reasons="")})
            return {"text": "ok"}
        with mock.patch.object(A, "turn", to_prediction):
            home.reply(d["id"], "actually, predict who churns")
        self.assertNotIn("family", self.store.get(d["id"])["understanding"])

    def test_a_second_yes_after_a_refused_build_builds_again(self):
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which members cancel")
        home, calls = self._home_with_builds({"refused": "There is no table to train on yet", "rule": None}, {"refused": "not yet", "rule": None})
        with mock.patch.object(A, "turn", self._yes):
            home.start(d["id"])
            home.reply(d["id"], "yes, build it")
        self.assertEqual(calls, ["build", "build"])

    def test_a_route_away_from_a_build_drops_the_open_build_question_and_groups_need_no_outcome(self):
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which members cancel")
        self.store.update(d["id"], lambda x: x.update(build_waiting="outcome column"))
        home, _ = self._home_with_builds()

        def to(tool):
            def run(state, client, words):
                state.case = state.case.model_copy(update={"route": C.CaseRoute(track=C.TRACK_OF_TOOL[tool], tool=tool, by="agent", reasons="")})
                return {"text": "ok"}
            return run
        with mock.patch.object(A, "turn", to("query")):
            home.reply(d["id"], "I would rather see cancellations by plan")
        self.assertIsNone(self.store.get(d["id"]).get("build_waiting"))
        with mock.patch.object(A, "turn", to("segmentation_project")):
            home.reply(d["id"], "group my members")
        self.assertEqual(self.store.get(d["id"])["understanding"]["family"], "segmentation")

    def test_the_lead_agents_turns_hold_the_drafts_turn_lock(self):
        for name in ("start", "reply", "data_ready"):
            self.assertTrue(hasattr(getattr(CaseHome, name), "__wrapped__"), name)

    def test_words_that_name_no_column_go_back_to_the_agent(self):
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which members cancel")
        self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "churned"}]}, build_waiting="outcome column"))
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", return_value={"text": "Which of your columns says who left?"}) as turn:
            home.reply(d["id"], "I am not sure what you mean")
        self.assertTrue(turn.called)
        self.assertEqual(calls, [])

    def test_a_column_check_answer_is_read_by_code_and_the_build_goes_on(self):
        from dclab_rnd.agents import case_agent as A
        from dclab_rnd.draft import column_check as checks

        d = self.store.create("predict which members cancel")
        self.store.update(d["id"], lambda x: x.update(
            intent={"track": "prediction", "status": "chosen"}, case_state={"case": {}},
            column_check_pending={"item": {"column": "last_login"}, "moment": "at signup", "target": "churned"},
            questions=x.get("questions", []) + [{"id": "q1", "field": checks.FIELD, "question": "Is last_login known at signup?",
                                                "options": list(checks.OPTIONS), "column": "last_login"}]))
        home, calls = self._home_with_builds({"refused": "not yet", "rule": None})
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the column answer")):
            home.reply(d["id"], checks.OPTIONS[0])
        draft = self.store.get(d["id"])
        self.assertIn("last_login", draft["understanding"]["column_checks"])
        self.assertEqual(calls, ["build"])

    def test_a_turn_writes_the_agents_words_the_case_file_and_its_state(self):
        d = self.store.create("our members keep cancelling and I do not know who will go next")
        script = Script(("You want to know who will cancel next.", [("note", {"field": "goal", "value": "know who will cancel",
                                                                              "status": "said", "quote": "who will go next"}),
                                                                     ("ask", {"question": "What would you do with that list?"})]))
        CaseHome(self.store, Models(script)).start(d["id"])
        draft = self.store.get(d["id"])
        texts = [m["text"] for m in draft["messages"] if m.get("role") == "agent"]
        self.assertTrue(any("What would you do with that list?" in t for t in texts))
        self.assertEqual(draft["case"]["facts"]["goal"]["quote"], "who will go next")
        self.assertIn("case", draft["case_state"])

    def test_options_are_cards_and_the_persons_pick_becomes_the_drafts_kind_of_help(self):
        d = self.store.create("members cancel, what can you do")
        home = CaseHome(self.store, Models(Script(("", [("propose", {"options": OPTIONS})]), ("Two ways to start.", []))))
        home.start(d["id"])
        msg = next(m for m in self.store.get(d["id"])["messages"] if m.get("kind") == "case_options")
        self.assertEqual([o["title"] for o in msg["options"]], ["Rank members by risk", "Who cancelled last year"])
        self.assertEqual(msg["options"][0]["gets"], "a ranked weekly list")
        home.models.script = Script(("I will rank your members.", []))
        home.reply(d["id"], "Rank members by risk")
        draft = self.store.get(d["id"])
        self.assertEqual(draft["case"]["route"]["track"], "prediction")
        self.assertEqual((draft["intent"]["track"], draft["intent"]["status"]), ("prediction", "chosen"))
        self.assertEqual(draft["messages"][-2]["text"], "Rank members by risk")  # the person's pick is in the conversation

    def test_with_no_model_the_rules_first_home_answers(self):
        class NoModels:
            def client(self, *a, **k):
                return None

            def available(self, *a):
                return False

        d = self.store.create("I need help with my data")
        CaseHome(self.store, NoModels()).start(d["id"])
        draft = self.store.get(d["id"])
        self.assertNotIn("case", draft)
        self.assertTrue(draft["messages"])  # the rules-first Home spoke

    def test_the_state_carries_from_turn_to_turn(self):
        d = self.store.create("our members keep cancelling and I do not know who will go next")
        CaseHome(self.store, Models(Script(("", [("ask", {"question": "Who acts on the list?"})])))).start(d["id"])
        CaseHome(self.store, Models(Script(("", [("note", {"field": "user", "value": "the club managers", "status": "said",
                                                            "quote": "the club managers"}), ("ask", {"question": "What data do you have?"})])))).reply(d["id"], "the club managers call them")
        draft = self.store.get(d["id"])
        self.assertEqual(draft["case"]["facts"]["user"]["value"], "the club managers")
        self.assertEqual(len(draft["case_state"]["turns"]), 4)  # two turns of the person and the agent, kept across instances


if __name__ == "__main__":
    unittest.main()
