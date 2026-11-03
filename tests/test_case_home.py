"""Package 37.9 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): Home, conversation first. With DCLAB_CASE_AGENT on, the lead agent
leads Home's draft conversation; with it off, or with no model, the rules-first Home does. Scripted models only."""
import json
import re
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd.agents import case_agent as A  # noqa: E402
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
        self.assertTrue(any(t.startswith("Which column is the outcome the model should learn?") for t in texts), texts)  # the table's own candidates are named now
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

    def _confirm(self, words, moment="at signup", agent_text="Let us look at that."):
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which orders are returned")
        self.store.update(d["id"], lambda x: x.update(build_waiting="confirm moment", moment_said=moment))
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", return_value={"text": agent_text}) as turn:
            home.reply(d["id"], words)
        return self.store.get(d["id"]), calls, turn.called

    def test_a_yes_naming_another_moment_never_builds_the_old_one(self):
        """The reviewer of 37.11b: the first fix built any yes, so "Yes, it is at checkout, not at signup" built at signup."""
        for words in ("Yes, it is at checkout, not at signup", "Yes - wait, not at signup, at checkout", "Yes, wait no, at checkout",
                      "Sure, it should be at checkout after payment", "Okay, I think it is when the order ships",
                      "Not quite, it's at checkout", "Yes, at checkout"):
            draft, calls, _ = self._confirm(words)
            self.assertEqual(calls, [], words)
            self.assertNotIn("prediction_moment", draft.get("understanding") or {}, words)
            self.assertEqual(draft.get("build_waiting"), "confirm moment", words)
            self.assertIn("checkout" if "checkout" in words else "ships", draft["moment_said"], words)
            self.assertFalse(re.match(r"^\W*(?:quite|wait|no|not)\b", draft["moment_said"], re.I), draft["moment_said"])

    def test_a_yes_with_words_that_are_not_a_moment_is_asked_again_and_never_said_back(self):
        """The reviewer of 37.11b: "Yes, the moment is checkout, not signup" and "Actually yes, but I'll explain" must not
        become the moment; they are asked again with the old moment kept, and nothing is built."""
        for words in ("Yes, the plan is checkout, not signup", "Yes, but my colleague disagrees"):
            draft, calls, agent = self._confirm(words)
            self.assertEqual((calls, agent), ([], False), words)
            self.assertEqual((draft.get("build_waiting"), draft.get("moment_said")), ("confirm moment", "at signup"), words)
            self.assertIn("not sure that is a yes", draft["messages"][-1]["text"], words)

    def test_a_trailing_but_or_a_question_without_its_mark_never_builds(self):
        """The re-review of 37.11b: "Yes, but..." built before the correction was typed; "Is that right" built."""
        for words in ("Yes, but...", "Okay, but", "Correct. But", "Yes — but"):
            draft, calls, agent = self._confirm(words)
            self.assertEqual((calls, agent), ([], False), words)
            self.assertEqual((draft.get("build_waiting"), draft.get("moment_said")), ("confirm moment", "at signup"), words)
        for words in ("Is that right", "should it be later"):
            draft, calls, agent = self._confirm(words)
            self.assertEqual((calls, agent), ([], True), words)
            self.assertNotIn("prediction_moment", draft.get("understanding") or {}, words)

    def test_curly_apostrophes_from_a_phone_keyboard_read_as_plain_ones(self):
        for words in ("Yes, that’s correct, go ahead", "That’s right", "It’s fine"):
            draft, calls, _ = self._confirm(words)
            self.assertEqual(calls, ["build"], words)
        draft, calls, agent = self._confirm("Don’t build it yet")
        self.assertEqual((calls, agent), ([], True))

    def test_at_once_or_at_all_is_not_a_moment(self):
        draft, calls, _ = self._confirm("Yes, build it at once")
        self.assertEqual(calls, ["build"])
        draft, calls, agent = self._confirm("Not at all")
        self.assertEqual(calls, [])
        self.assertNotEqual(draft.get("moment_said"), "at all")

    def test_plain_approval_in_many_words_builds(self):
        for words in ("No problem, go ahead", "Actually yes, build it", "Yes, nothing else to add but go ahead", "sure, no problem",
                      "That sounds right"):
            draft, calls, agent = self._confirm(words)
            self.assertEqual((calls, agent), (["build"], False), words)
            self.assertEqual(draft["understanding"]["prediction_moment"], "at signup", words)

    def test_a_hold_on_the_build_or_other_words_go_to_the_agent_and_build_nothing(self):
        for words in ("Yes that's right, but I'll build later", "Don't build it yet", "Let us look at groups instead"):
            draft, calls, agent = self._confirm(words)
            self.assertEqual((calls, agent), ([], True), words)
            self.assertIsNone(draft.get("build_waiting"), words)
            self.assertNotIn("prediction_moment", draft.get("understanding") or {}, words)

    def _events(self, draft_id, kind=None):
        return [e for e in self.store.events(draft_id, 0) if kind is None or e["kind"] == kind]

    def test_the_persons_words_are_sent_to_the_page_on_every_path_under_the_lead_agent(self):
        """The owner, 2026-11-02: no user prompt in the chat. The page draws the chat from what is sent; the lead agent's Home only
        saved the person's message, so it never showed (and the question's buttons stayed live)."""
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which members cancel")
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", return_value={"text": "Understood."}):
            home.reply(d["id"], "I would call them before they leave")  # the ordinary path
        self.store.update(d["id"], lambda x: x.update(build_waiting="prediction moment"))
        with mock.patch.object(A, "turn", return_value={"text": "Understood."}):
            home.reply(d["id"], "What do you mean by the moment?")  # a build question's answer path
        users = [e["data"] for e in self._events(d["id"], "chat") if e["data"].get("role") == "user"]
        self.assertEqual([u["text"] for u in users], ["I would call them before they leave", "What do you mean by the moment?"])
        self.assertTrue(all(u.get("id") and u.get("at") and u["kind"] == "text" for u in users))  # as the rules-first Home's own

    def test_a_pick_of_an_option_is_sent_to_the_page_too(self):
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent as A

        d = self.store.create("predict which members cancel")
        home, calls = self._home_with_builds()
        case = C.new("c", ["x"])
        case = C.apply(case, "options", [C.CaseOption(title="Rank members", what="w", gets="a list", tradeoff="needs history", effort="a day",
                                                      track="prediction", tool="ml_project")], by="agent", why="t")
        from dclab_rnd.agents import case_session

        state = A.CaseState(case=case)
        self.store.update(d["id"], lambda x: x.update(case_state=case_session.snapshot(state)))
        with mock.patch.object(A, "pick", return_value={"text": "Starting."}):
            home.reply(d["id"], "Rank members")
        users = [e["data"]["text"] for e in self._events(d["id"], "chat") if e["data"].get("role") == "user"]
        self.assertEqual(users, ["Rank members"])

    def test_a_prediction_route_draws_the_workflow_once_and_other_routes_draw_none(self):
        """The owner's screenshot: a segment report showed an empty Solution workflow card. The page learns the workflow only from
        what is sent; a prediction route draws it (once), a report or an analysis does not."""
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent as A

        def route(track, tool):
            def turn(state, client, words):
                state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="agent"), by="agent", why="t")
                return {"text": "Started."}
            return turn

        d = self.store.create("predict which members cancel")
        home, calls = self._home_with_builds()
        with mock.patch.object(A, "turn", route("prediction", "ml_project")):
            home.reply(d["id"], "go on")
        self.assertEqual(len(self._events(d["id"], "workflow")), 1)
        with mock.patch.object(A, "turn", return_value={"text": "More."}):
            home.reply(d["id"], "and what else")
        self.assertEqual(len(self._events(d["id"], "workflow")), 1)  # not redrawn on every turn
        self.assertTrue(self.store.get(d["id"]).get("workflow"))
        other = self.store.create("break churn down by contract")
        with mock.patch.object(A, "turn", route("analysis", "query")):
            home.reply(other["id"], "go on")
        self.assertEqual(self._events(other["id"], "workflow"), [])

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


def telco_frame():
    import pandas as pd

    rows = []
    for contract, total, left in (("Month-to-month", 40, 20), ("One year", 30, 6), ("Two year", 30, 1)):
        rows += [{"customerID": f"{contract[:2]}{i:03d}", "Contract": contract, "Churn": "Yes" if i < left else "No", "tenure": i + 1} for i in range(total)]
    return pd.DataFrame(rows)


class CardsTests(unittest.TestCase):
    """2026-11-02 (the owner's Telco session): "show my report and give dashboard" ended in "I've switched your report into a
    dashboard" and nothing on the page. Now an answer or a dashboard the agent asks for is drawn by Home's renderers under its
    message, from what code computed."""

    def setUp(self):
        from dclab_rnd import ledger
        from dclab_rnd.draft import pipeline

        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.store = DraftStore(self.folder / "drafts")
        ledger.install(ledger.FileLedger(self.folder / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)
        self.draft = self.store.create("I face losing my customer I want know which segment of my customer I lost more")
        path = Path(self.store.data_dir(self.draft["id"]))
        path.mkdir(parents=True, exist_ok=True)
        telco_frame().to_csv(path / "telco.csv", index=False)
        asset = pipeline.new_asset(self.store, self.draft["id"], "upload", "telco.csv", "telco.csv")
        pipeline.run(self.store, self.draft["id"], asset["id"], HomeAgent(self.store, None, on_request=lambda *a: None), None)

    def home(self, *replies):
        return CaseHome(self.store, Models(Script(*replies)), on_request=lambda *a: None)

    def kinds(self):
        return [m.get("kind") for m in self.store.get(self.draft["id"])["messages"] if m.get("role") == "agent"]

    SHARE = {"group_by": ["Contract"], "measures": [{"name": "customers", "op": "count"}, {"name": "churn_rate", "op": "rate", "column": "Churn"}],
             "sort": [{"by": "churn_rate", "desc": True}]}

    def test_an_answer_the_agent_asks_for_is_drawn_as_a_chart_before_its_words(self):
        home = self.home(("", [("answer", self.SHARE)]), ("Month-to-month loses the most: 50.0% of its 40 customers left.", []))
        home._run(self.draft["id"], "which segment do I lose most? show me")
        messages = [m for m in self.store.get(self.draft["id"])["messages"] if m.get("role") == "agent"][-2:]  # the pipeline's own cards come before
        self.assertEqual([m["kind"] for m in messages], ["answer", "text"])  # the card first, then what it shows
        chart = messages[0]["answer"]["chart"]
        self.assertEqual(chart["kind"], "bar")
        self.assertEqual(chart["x"][0], "Month-to-month")
        self.assertIn("50.0%", messages[1]["text"])  # said exactly as the result gave it: not dropped by the number guard

    def test_a_dashboard_the_agent_asks_for_is_drawn_and_kept_so_it_can_be_refreshed(self):
        tiles = [{"group_by": ["Contract"], "measures": [{"name": "customers", "op": "count"}]},
                 {"group_by": ["Contract"], "measures": [{"name": "churn_rate", "op": "rate", "column": "Churn"}]}]
        home = self.home(("", [("dashboard", {"title": "Where we lose customers", "tiles": tiles})]), ("Two tiles by contract: customers and the share that left.", []))
        home._run(self.draft["id"], "now show my report and give dashboard")
        draft = self.store.get(self.draft["id"])
        self.assertEqual(self.kinds()[-2:], ["dashboard", "text"])
        board = next(m for m in draft["messages"] if m.get("kind") == "dashboard")["dashboard"]
        self.assertEqual([t["chart"]["kind"] for t in board["tiles"]], ["bar", "bar"])
        self.assertTrue(draft["dashboard"]["board"])  # the Refresh button has its board
        home._run(self.draft["id"], "show it again")  # the same board is not drawn twice
        self.assertEqual(self.kinds().count("dashboard"), 1)

    def test_the_owners_telco_session_ends_in_figures_and_a_dashboard_not_in_questions(self):
        """The 12 messages of 2026-11-02, replayed with a model that does what the real one did: it asks first, writes a share as
        {op: rate, column: Churn}, and promises a dashboard. Now the question is refused, the share is read as Churn = Yes, the
        chart is drawn in the first turn, and the dashboard exists when it is asked for."""
        said = "I face losing my customer I want know which segment of my customer I lost more"
        home = self.home(
            ("", [("note", {"field": "goal", "value": "which segment of my customer I lost more", "status": "said", "quote": "which segment of my customer"}),
                  ("ask", {"question": "What will you do with the answer: a retention offer, a price change, or something else?"})]),
            ("", [("answer", self.SHARE)]),
            ("Month-to-month loses the most: 50.0% of its 40 customers left, against 20.0% of 30 on one year.", []),
            ("", [("dashboard", {"title": "Where we lose customers", "tiles": [
                {"group_by": ["Contract"], "measures": [{"name": "customers", "op": "count"}]},
                {"group_by": ["Contract"], "measures": [{"name": "churn_rate", "op": "rate", "column": "Churn"}]}]})]),
            ("The dashboard shows customers and the share that left, by contract.", []))
        home._run(self.draft["id"], said)
        first = [m for m in self.store.get(self.draft["id"])["messages"] if m.get("role") == "agent"][-2:]
        self.assertEqual([m["kind"] for m in first], ["answer", "text"])  # figures in the first turn
        self.assertNotIn("?", first[1]["text"])
        home._run(self.draft["id"], "now show my report and give dashboard")
        draft = self.store.get(self.draft["id"])
        self.assertEqual([m["kind"] for m in draft["messages"] if m.get("role") == "agent"][-2:], ["dashboard", "text"])
        every = " ".join(m.get("text") or "" for m in draft["messages"] if m.get("role") == "agent")
        for stalling in ("next turn", "reply \"go\"", "Reply go", "I'll retry", "\u2588"):
            self.assertNotIn(stalling, every)
        self.assertEqual(draft["case"]["facts"]["goal"]["status"], "said")
        self.assertLessEqual(draft["case_state"]["case"]["asked"], 0)  # the one question was refused until the figures were shown

    def test_a_tile_the_table_refuses_is_a_refusal_the_agent_reads_and_nothing_is_drawn(self):
        bad = [{"group_by": ["Plan"], "measures": [{"name": "n", "op": "count"}]}]
        home = self.home(("", [("dashboard", {"tiles": bad})]), ("I could not use that column.", []))
        home._run(self.draft["id"], "give dashboard")
        self.assertNotIn("dashboard", self.kinds())



class BuildFlowTests(CardsTests):
    """2026-11-02 (the owner's second Telco session): "now build model" redrew the report, "make prediction models" got "Shall I build
    it now?", a bare "yes" got "reply build the model", and the build asked for the outcome the person had confirmed."""

    SAID = ("which segment of my customer I lost more, churn is the column that says who left, I decide about contract deals myself")

    def home_with_builds(self, *replies):
        calls = []

        def on_request(draft_id, what, args):
            calls.append(what)
            return {"refused": "At what moment will the prediction be made?", "rule": "prediction moment"}
        return CaseHome(self.store, Models(Script(*replies)), on_request=on_request), calls

    def prepare(self, home):
        """The report first (the figures asked for), the case ready with the decision and the user said."""
        home._run(self.draft["id"], self.SAID)

    NOTES = [("note", {"field": "goal", "value": "which segment of my customer I lost more", "status": "said", "quote": "which segment of my customer"}),
             ("note", {"field": "decision", "value": "contract deals", "status": "said", "quote": "about contract deals"}),
             ("note", {"field": "user", "value": "I decide myself", "status": "said", "quote": "I decide about contract deals"}),
             ("answer", CardsTests.SHARE)]

    def test_an_order_to_build_a_model_starts_the_build_at_once_with_the_outcome_the_person_named_and_draws_no_report(self):
        home, calls = self.home_with_builds(
            ("", self.NOTES), ("Month-to-month loses the most: 50.0% of its 40 customers left.", []),
            ("", [("answer", self.SHARE), ("start", {"track": "prediction", "tool": "ml_project", "quote": "now build model"}),
                  ("request_build", {"quote": "now build model", "outcome": "Churn"})]),
            ("Your order is recorded and the build starts now.", []))
        self.prepare(home)
        before = self.kinds().count("answer")
        home._run(self.draft["id"], "now build model")
        draft = self.store.get(self.draft["id"])
        self.assertEqual(calls, ["build"])
        self.assertEqual(self.kinds().count("answer"), before)  # no report drawn again
        self.assertEqual(draft["understanding"]["target"], "Churn")  # not asked again
        self.assertEqual(draft.get("build_waiting"), "prediction moment")  # the moment keeps its protocol

    def test_a_bare_yes_after_the_agents_question_starts_the_build_and_asks_for_no_phrase(self):
        home, calls = self.home_with_builds(
            ("", self.NOTES), ("Month-to-month loses the most.", []),
            ("", [("start", {"track": "prediction", "tool": "ml_project", "quote": "a model predict who leaves"})]),
            ("A model would score each customer's chance of leaving.", [("ask", {"question": "Shall I build one for you?"})]),
            ("", [("request_build", {"quote": "yes", "outcome": "Churn"})]), ("The build starts now.", []))
        self.prepare(home)
        home._run(self.draft["id"], "could a model predict who leaves?")
        self.assertEqual(calls, [])
        home._run(self.draft["id"], "yes")
        self.assertEqual(calls, ["build"])
        every = " ".join(m.get("text") or "" for m in self.store.get(self.draft["id"])["messages"] if m.get("role") == "agent")
        self.assertNotIn('reply "build the model"', every.lower())

    def test_make_prediction_models_is_an_order_so_the_agent_may_not_ask_whether_to_build(self):
        home, calls = self.home_with_builds(
            ("", self.NOTES), ("Month-to-month loses the most.", []),
            ("", [("ask", {"question": "Shall I build it now?"})]),
            ("", [("start", {"track": "prediction", "tool": "ml_project", "quote": "make prediction models"}),
                  ("request_build", {"quote": "make prediction models", "outcome": "Churn"})]), ("The build starts now.", []))
        self.prepare(home)
        home._run(self.draft["id"], "make prediction models")
        self.assertEqual(calls, ["build"])  # the question was refused; the agent acted on the order

    def test_you_decide_on_the_outcome_question_uses_the_one_column_the_persons_words_name(self):
        home, calls = self.home_with_builds(("", self.NOTES), ("Month-to-month loses the most.", []))
        self.prepare(home)
        self.store.update(self.draft["id"], lambda d: d.update(build_waiting="outcome column"))
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the answer to the build's question")):
            home.reply(self.draft["id"], "you decide")
        draft = self.store.get(self.draft["id"])
        self.assertEqual(draft["understanding"]["target"], "Churn")
        self.assertEqual(calls, ["build"])

    def test_you_decide_inside_a_refusal_builds_nothing(self):
        home, calls = self.home_with_builds(("", self.NOTES), ("Month-to-month loses the most.", []), ("Understood, I will wait.", []))
        self.prepare(home)
        self.store.update(self.draft["id"], lambda d: d.update(build_waiting="outcome column"))
        home.reply(self.draft["id"], "don't build anything yet, up to you")
        self.assertEqual(calls, [])
        self.assertFalse(self.store.get(self.draft["id"]).get("understanding", {}).get("target"))

    def test_a_column_only_mentioned_is_never_the_outcome(self):
        home, calls = self.home_with_builds(("", self.NOTES), ("Month-to-month loses the most.", []))
        self.store.update(self.draft["id"], lambda d: d.update(case_state={"case": C.hear(C.new("c"), "Is it different by gender? Which contract loses most?").model_dump(mode="json")}))
        home._build(self.draft["id"])
        self.assertFalse(self.store.get(self.draft["id"]).get("understanding", {}).get("target"))
        self.assertEqual(calls, ["build"])  # asked the build's own question, with the table's candidates

    def test_with_no_column_named_the_outcome_question_names_the_tables_own_candidates(self):
        home, calls = self.home_with_builds()
        asset = self.store.get(self.draft["id"])["active_asset"]
        self.store.update(self.draft["id"], lambda d: d.update(case_state={"case": C.new("c").model_dump(mode="json")}))
        home._build(self.draft["id"])
        texts = [m.get("text") or "" for m in self.store.get(self.draft["id"])["messages"] if m.get("role") == "agent"]
        self.assertFalse(self.store.get(self.draft["id"]).get("understanding", {}).get("target"))
        self.assertEqual(calls, ["build"])


if __name__ == "__main__":
    unittest.main()
