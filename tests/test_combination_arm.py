"""Package 26.5: the decision models inside the workflow model's loop (decide_point at Home) and the third arm."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # .venv-agent
    HAVE_ML = False


class DecideWorkflowTests(unittest.TestCase):
    def test_a_workflow_point_is_answered_with_the_callers_rule(self):
        from dclab_rnd import decide

        state = {"open_questions": 2, "step": 1, "outcome_known": True, "moment_known": False, "pack": "tabular"}
        with mock.patch("dclab_rnd.ledger.shadow.answerers", return_value=()):
            out = decide.ask("workflow.next_question", state, rule_value="prediction_moment", rule_name="rule:home", routing_doc={})
        self.assertEqual((out["answer"], out["decided_by"], out["rule"]["answerer"]), ("prediction_moment", "rule", "rule:home"))
        self.assertIn("entity", out["options"])  # every field Home can ask, not the plan's five
        self.assertEqual(out["options"][-1], "none")

    def test_without_a_rules_answer_nothing_is_invented(self):
        from dclab_rnd import decide

        with mock.patch("dclab_rnd.ledger.shadow.answerers", return_value=()):
            out = decide.ask("problem.family", {"labels": "column", "open_questions": 1, "outcome_known": True}, routing_doc={})
        self.assertIsNone(out["answer"])
        self.assertEqual(out["decided_by"], "none")

    def test_a_point_outside_both_lists_is_refused(self):
        from dclab_rnd import decide

        with self.assertRaises(decide.DecideError):
            decide.ask("workflow.move", {"stage": "data"})

    def test_a_routed_model_never_frees_a_column_the_rules_hold_back(self):
        from dclab_core.ledger import Answer
        from dclab_rnd import decide

        class Freer:
            name, version, points = "m", "1", frozenset({"column.is_identifier"})

            def answer(self, point, state, options):
                return Answer(by="classifier", answerer="m@1", value=False, confidence=0.99)

        doc = {"decisions": {"column.is_identifier": {"answerer": "m@1", "level": 3}}}
        state = {"column": "order_id", "name_tokens": ["order", "id"], "dtype": "integer", "uniqueness": "all"}
        with mock.patch("dclab_rnd.ledger.shadow.answerers", return_value=(Freer(),)):
            out = decide.ask("column.is_identifier", state, routing_doc=doc)
        self.assertIs(out["answer"], True)
        self.assertEqual(out["decided_by"], "rule")
        self.assertIn("never frees", out["refused_by_validator"])

    def test_a_journeys_arm_routing_is_the_one_read(self):
        from dclab_rnd import decide
        from dclab_rnd.ledger import use

        doc = {"decisions": {"problem.ready": {"answerer": "nobody@1", "level": 3}}}
        with use.scope(doc), mock.patch("dclab_rnd.ledger.shadow.answerers", return_value=()):
            out = decide.ask("problem.ready", {"labels": "file", "open_questions": 0, "outcome_known": True, "family": "rare_event"},
                             rule_value=True)
        self.assertEqual((out["routing"], out["level"], out["answer"]), ("scope", 3, True))  # the routed model is absent: the rule stands


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class HomeToolTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.draft.chat import HomeAgent, Turn
        from dclab_rnd.draft.store import DraftStore

        self.store = DraftStore(Path(tempfile.mkdtemp()) / "drafts")
        self.agent = HomeAgent(self.store, None)
        self.draft = self.store.create("predict which customers churn next month")["id"]
        self.agent.start(self.draft)
        self.turn = Turn(self.agent, self.draft)

    def test_decide_point_is_a_draft_read_tool_the_home_agent_is_offered(self):
        from dclab_rnd.agents import default_registry

        tool = default_registry().get("decide_point")
        self.assertEqual((tool.scope, tool.effect), ("draft", "read"))
        self.assertIn("decide_point", self.agent.policy().tools)

    def test_the_next_question_is_answered_from_the_drafts_state(self):
        from dclab_rnd.draft.chat import decide_point

        with mock.patch("dclab_rnd.ledger.shadow.answerers", return_value=()):
            waiting = decide_point(self.turn, point="workflow.next_question")
            self.assertIsNone(waiting["rule"])  # the opening question waits for its answer: the rules decide nothing yet
            self.store.update(self.draft, lambda d: [q.update(answered="the churners") for q in d["questions"]])
            out = decide_point(self.turn, point="workflow.next_question")
        self.assertNotIn("error", out)
        self.assertEqual(out["rule"]["answerer"], "rule:home")
        from dclab_core.workflow_labels import QUESTIONS

        self.assertIn(out["answer"], QUESTIONS)

    def test_a_column_point_needs_a_column_of_the_prepared_table(self):
        from dclab_rnd.draft.chat import decide_point

        out = decide_point(self.turn, point="column.is_identifier", column="customer_id")
        self.assertIn("error", out)  # no table yet
        self.assertIn("error", decide_point(self.turn, point="workflow.move"))


class LiveArmsPlanTests(unittest.TestCase):
    def test_the_live_arms_only_print_their_plan_without_yes_and_refuse_a_sealed_split(self):
        from dclab_rnd.agent_eval.journeys import __main__ as cli

        out = io.StringIO()
        with mock.patch("sys.argv", ["journeys", "--live-arms", "--cap-eur", "1"]), redirect_stdout(out), \
                mock.patch.object(cli, "measure", side_effect=AssertionError("nothing is played without --yes")):
            self.assertEqual(cli.main(), 0)
        self.assertIn("three arms", out.getvalue())
        with mock.patch("sys.argv", ["journeys", "--live-arms", "--split", "sealed_v2", "--spent"]), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(), 2)


if __name__ == "__main__":
    unittest.main()
