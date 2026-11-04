"""Home's chips (the owner, 2026-11-04): the person picks the kind of help before writing; the lead agent starts it at once, refuses
another kind, and asks once only when the person's words clearly point elsewhere (the report-not-model trap stays caught)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd.agents import case_agent as A  # noqa: E402
from dclab_rnd.agents import case_session  # noqa: E402

VIEW = [{"name": "plan", "kind": "category"}, {"name": "churn", "kind": "binary"}, {"name": "tenure", "kind": "number"}]


def state(words, chip):
    s = A.CaseState(case=C.new("chip-case", words), table=A.table_of("customers.csv", VIEW, 1200))
    s.chip = chip
    return s


def option(track, tool, title):
    return {"title": title, "what": f"{title} on the shared table", "gets": "an answer they can use", "tradeoff": "less detail than a full study", "effort": "a few minutes", "track": track, "tool": tool}


class ChipRuleTests(unittest.TestCase):
    def test_the_words_disagree_only_when_they_clearly_point_elsewhere(self):
        self.assertEqual(C.chip_conflict("prediction", ["what was our churn per plan last month"]), "analysis")
        self.assertEqual(C.chip_conflict("prediction", ["need a churn prediction for the board, what our churn was per plan last month"]), "analysis")
        self.assertIsNone(C.chip_conflict("prediction", ["which customers will leave next month"]))
        self.assertIsNone(C.chip_conflict("prediction", ["churn model for our telco data"]))
        self.assertEqual(C.chip_conflict("reporting", ["which customers will churn next quarter"]), "prediction")
        self.assertIsNone(C.chip_conflict("analysis", ["why did revenue drop last month"]))
        self.assertEqual(set(C.CHIPS), set(C.TRACKS))


class ChipRouteTests(unittest.TestCase):
    def test_a_chip_starts_its_route_at_once_without_the_readiness_wait(self):
        s = state(["churn model for our telco data"], "prediction")
        out = A._start(s, "prediction", "ml_project")
        self.assertIn("started", out)
        self.assertEqual((s.case.route.track, s.case.route.by), ("prediction", "person"))

    def test_another_kind_of_help_is_refused(self):
        s = state(["a weekly dashboard of sales by region"], "reporting")
        self.assertIn("picked Report", A._start(s, "analysis", "query")["error"])
        self.assertIsNone(s.case.route)

    def test_a_conflict_asks_once_with_two_options_then_the_pick_decides(self):
        s = state(["what was our churn per plan last month"], "prediction")
        self.assertIn("propose the two once", A._start(s, "prediction", "ml_project")["error"])
        self.assertIn("propose exactly two options", A.chip_note(s))
        three = [option("prediction", "ml_project", "A churn model"), option("analysis", "query", "Churn per plan"),
                 option("reporting", "dashboard", "A dashboard")]
        self.assertIn("offer only", A._propose(s, three)["error"])
        out = A._propose(s, three[:2])
        self.assertIn("shown", out, out)
        self.assertTrue(s.chip_checked)  # asked once
        self.assertIn("error", A._start(s, "analysis", "query"))  # the chip's track stands until they pick the other

    def test_without_a_chip_nothing_changes(self):
        s = state(["churn model for our telco data"], None)
        self.assertIsNone(A.chip_note(s))
        self.assertIn("error", A._start(s, "prediction", "ml_project"))  # the old readiness wait still holds

    def test_the_chip_is_kept_with_the_case(self):
        s = state(["churn model"], "prediction")
        s.chip_checked = True
        doc = case_session.snapshot(s)
        t = state(["churn model"], None)
        case_session.restore(t, doc)
        self.assertEqual((t.chip, t.chip_checked), ("prediction", True))



class ExamArmTests(unittest.TestCase):
    """The exam's chip arm hands each person their true kind of help as a chip (the best case of chips)."""

    def test_the_chip_arm_starts_with_the_persons_true_track(self):
        from dclab_rnd.agent_eval import conversation_exam as X
        from dclab_rnd.agent_eval.conversation_exam.arms import CaseArm

        self.assertIn("case_chip", X.ARMS)
        arm = CaseArm(client=None, chip=True)
        captured = {}
        original = A.turn
        A.turn = lambda state, client, words, registry=None: captured.setdefault("chip", state.chip) and {"text": "", "cards": []}
        try:
            try:
                arm.open({"id": "r-01", "opening": "which customers will leave next month", "route": "prediction"})
            except Exception:  # noqa: BLE001 - only the state given to the turn matters here
                pass
        finally:
            A.turn = original
        self.assertEqual(captured.get("chip"), "prediction")

class ReviewFixTests(unittest.TestCase):
    """The reviewer's findings on the chips (2026-11-04), one test each."""

    def test_the_person_can_change_their_mind_in_their_latest_words(self):
        s = state(["predict which customers will leave", "actually I want a dashboard of churn by plan instead"], "prediction")
        self.assertEqual(C.chip_conflict("prediction", s.case.words), "reporting")
        out = A._start(s, "reporting", "dashboard", quote="I want a dashboard of churn by plan instead")
        self.assertIn("started", out, out)
        self.assertEqual((s.chip, s.case.route.track), ("reporting", "reporting"))

    def test_a_learn_chip_with_data_work_asks_once(self):
        s = state(["send me the list of customers who churned last month"], "advice")
        self.assertIn("propose the two once", A._start(s, "advice", "cited_answer")["error"])
        self.assertIn("work on data", A.chip_note(s))

    def test_a_model_asked_for_by_its_verb_is_not_the_past_trap(self):
        self.assertIsNone(C.chip_conflict("prediction", ["forecast sales by region"]))
        self.assertIsNone(C.chip_conflict("prediction", ["score each customer by churn risk using last year's data"]))
        self.assertEqual(C.chip_conflict("prediction", ["I will present it to the board: churn rate per plan last month"]), "analysis")
        self.assertEqual(C.chip_conflict("prediction", ["send me the list of who churned"]), "analysis")

    def test_figures_shown_on_the_way_never_move_a_chips_route(self):
        s = state(["churn model for our telco data"], "prediction")
        s.wants_figures = True
        A._worked(s, "analysis", "query")
        self.assertIsNone(s.case.route)

    def test_a_data_chip_with_no_table_asks_for_it(self):
        s = A.CaseState(case=C.new("no-table", ["churn model please"]))
        s.chip = "prediction"
        self.assertIn("share the table", A._start(s, "prediction", "ml_project")["error"])
        self.assertIsNone(s.case.route)

    def test_the_benchmark_page_lists_the_chip_arm(self):
        from dclab_rnd.agent_eval.conversation_exam import benchmark
        self.assertIn("case_chip", benchmark.ARMS)

class SecondReviewTests(unittest.TestCase):
    """The second review of the chips (2026-11-04)."""

    def test_a_base_rate_or_an_also_after_the_chip_started_is_not_a_change_of_mind(self):
        s = state(["churn model for our telco data"], "prediction")
        A._start(s, "prediction", "ml_project")
        for later in ("first, what was our churn rate last month?", "can you also show me a chart of churn by plan?"):
            s.case.words.append(later)
            self.assertIsNone(A.changed_mind(s))
            self.assertIsNone(A.chip_note(s))  # work under way: no "propose the two" interruption
        self.assertEqual(s.case.route.track, "prediction")

    def test_an_order_for_a_model_under_another_chip_is_a_conflict(self):
        self.assertEqual(C.chip_conflict("analysis", ["churn per plan", "actually, build me a churn model instead"]), "prediction")
        self.assertIsNone(C.chip_conflict("prediction", ["build a churn model"]))

    def test_a_change_to_a_kind_needing_a_table_is_kept_until_the_table_comes(self):
        s = A.CaseState(case=C.new("no-table", ["churn per plan", "predict which customers will churn instead"]))
        s.chip, s.chip_intent = "analysis", "analysis"
        out = A._start(s, "prediction", "ml_project", quote="predict which customers will churn instead")
        self.assertIn("share the table", out["error"])
        self.assertEqual((s.chip, s.chip_intent), ("prediction", "analysis"))  # the press it came from is unchanged: not re-read


if __name__ == "__main__":
    unittest.main()
