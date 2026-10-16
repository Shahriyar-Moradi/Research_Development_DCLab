"""The evaluation dashboard (package 25.8): precision, recall, F1 and accuracy of every answerer, checked against cases
counted by hand; read from twin steps and the decision ledger; served to the owner's Admin page."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import evaluation as E  # noqa: E402
from dclab_core import ledger as L  # noqa: E402
from dclab_core import twin as T  # noqa: E402

AT = "2026-10-16T12:00:00+00:00"


class MetricsByHandTest(unittest.TestCase):
    def test_a_flag(self):
        # truth T,T,F,F,T ; answers T,F,T,F,(none): tp 1, fn 2 (one missed, one unanswered), fp 1, tn 1
        s = E.score("rules", "feature.leakage_suspect", "flag", "planted", [(True, True), (False, True), (True, False), (False, False), (None, True)])
        self.assertEqual(s.confusion, {"tp": 1, "fp": 1, "fn": 2, "tn": 1})
        self.assertEqual((s.accuracy, s.precision, s.recall, s.f1, s.coverage, s.abstained), (0.4, 0.5, 0.3333, 0.4, 0.8, 1))
        self.assertEqual(s.accuracy_interval, E.wilson(2, 5))

    def test_a_choice_macro_over_the_truths_classes(self):
        s = E.score("m", "split.strategy", "choice", "planted", [("time", "time"), ("random", "time"), ("random", "random"), (None, "group")])
        by = {c.label: c for c in s.classes}
        self.assertEqual((by["time"].precision, by["time"].recall), (1.0, 0.5))
        self.assertEqual((by["random"].precision, by["random"].recall), (0.5, 1.0))
        self.assertEqual((by["group"].precision, by["group"].recall), (0.0, 0.0))  # never named: 0, as scikit-learn's zero_division=0
        self.assertEqual(s.accuracy, 0.5)
        self.assertEqual((s.precision, s.recall), (0.5, 0.5))
        self.assertEqual(s.f1, round((0 + 0.6667 + 0.6667) / 3, 4))

    def test_a_set_micro_over_items(self):
        s = E.score("claude", "solution.columns:leakage", "set", "planted", [(["a", "b"], ["a"]), ([], ["c"]), (["d"], ["d"])])
        self.assertEqual(s.confusion, {"tp": 2, "fp": 1, "fn": 1})
        self.assertEqual((s.precision, s.recall, s.f1, s.accuracy), (0.6667, 0.6667, 0.6667, 0.3333))
        nothing = E.score("claude", "x", "set", "planted", [([], []), ([], [])])
        self.assertEqual((nothing.precision, nothing.recall, nothing.accuracy), (1.0, 1.0, 1.0))


class ReviewFindingsTest(unittest.TestCase):
    """The 25.8 review's findings on the metrics."""

    def test_one_rule_for_undefined_ratios_and_the_macro_over_every_class(self):
        missed = E.score("m", "p", "flag", "planted", [(False, True)] * 5)
        self.assertEqual((missed.precision, missed.recall, missed.f1), (0.0, 0.0, 0.0))  # positives existed, none predicted: 0 (zero_division=0)
        union = E.score("m", "p", "choice", "planted", [("x", "a"), ("a", "a")])
        self.assertEqual(union.precision, 0.5)  # sklearn's macro: x (precision 0) counts
        never = E.score("m", "p", "set", "planted", [(None, []), (None, [])])
        self.assertEqual((never.precision, never.recall, never.f1), (None, None, None))  # no answer, no number
        wide = E.score("m", "p", "choice", "planted", [(f"c{i}", f"c{i}") for i in range(60)] + [(None, "c60")])
        self.assertEqual((len(wide.classes), wide.recall), (50, round(60 / 61, 4)))
        odd = E.score("m", "p", "flag", "planted", [("no", True), (True, True)])
        self.assertEqual((odd.answered, odd.confusion["tp"], odd.confusion["fn"]), (1, 1, 1))  # "no" is not a flag: no answer

    def test_blind_labels_weigh_and_a_correction_of_what_was_shown_is_left_out(self):
        recs, labs = [], []
        for i, (rule, truth, source, weight) in enumerate([(True, True, "blind", 1.0), (False, True, "blind", 9.0), (True, True, "user_decision_exposed", None)]):
            recs.append(L.DecisionRecord(id=f"dr-wgt-{i:04d}", point="column.is_identifier", subject=L.Subject(kind="column", key=f"c{i}"), at=AT,
                                         answers=[L.Answer(by="rule", answerer="rule:k", value=rule)], value_used=rule, used="rule", project="p1"))
            labs.append(L.DecisionLabel(id=f"dl-wgt-{i:04d}", record=f"dr-wgt-{i:04d}", point="column.is_identifier", value=truth, source=source,
                                        by="reviewer", at=AT, **({"weight": weight} if weight else {})))
        s = E.from_ledger(recs, labs).scores[0]
        self.assertEqual((s.n, s.accuracy), (2, 0.1))  # weights 1 and 9: one right of ten weighed; the exposed one is out
        score_point = L.DecisionRecord(id="dr-score-0001", point="proposal.completeness", subject=L.Subject(kind="proposal", key="p"), at=AT,
                                       answers=[L.Answer(by="rule", answerer="rule:c", value=1)], value_used=1, used="rule")
        lab = L.DecisionLabel(id="dl-score-0001", record="dr-score-0001", point="proposal.completeness", value=1.0, source="benchmark", by="r", at=AT)
        self.assertEqual(E.from_ledger([score_point], [lab]).scores[0].accuracy, 1.0)  # a level is a number, not a string


class FromRecordsTest(unittest.TestCase):
    def test_both_arms_of_twin_steps_are_scored_on_the_same_cases(self):
        steps = []
        for i, (sys_leak, ai_leak) in enumerate([(["fare"], ["fare"]), ([], ["fare"]), (["fare", "id"], ["fare"])]):
            steps.append(T.step(f"s{i}", "run", "solution.columns", AT, {"i": i},
                                T.ArmAnswer(arm="system", answerer="rules", value={"leakage": sys_leak, "target": "y"}),
                                T.ArmAnswer(arm="ai_first", answerer="claude-sonnet-5-5", value={"leakage": ai_leak, "target": "y"}),
                                truth=T.Truth(value={"leakage": ["fare"], "target": "y"}, source="planted")))
        board = E.from_twin_steps(steps)
        by = {(s.answerer, s.point): s for s in board.scores}
        system, claude = by[("system", "solution.columns:leakage")], by[("claude-sonnet-5-5", "solution.columns:leakage")]
        self.assertEqual((system.n, system.precision, system.recall, system.accuracy), (3, 0.6667, 0.6667, 0.3333))
        self.assertEqual((claude.precision, claude.recall, claude.accuracy), (1.0, 1.0, 1.0))
        self.assertEqual(by[("system", "solution.columns:target")].accuracy, 1.0)
        self.assertEqual(E.EvaluationBoard.read(board.model_dump(mode="json")), board)

    def test_every_answer_on_a_labelled_decision_is_scored_against_its_effective_label(self):
        records, labels = [], []
        for i, (rule, clf, truth) in enumerate([(True, True, True), (False, True, True), (True, False, False), (False, False, False)]):
            records.append(L.DecisionRecord(id=f"dr-rec-{i:04d}", point="column.is_identifier", subject=L.Subject(kind="column", key=f"c{i}"), at=AT,
                                            answers=[L.Answer(by="rule", answerer="rule:key_named", value=rule),
                                                     L.Answer(by="classifier", answerer="identifier-gbm@1", value=clf, source=L.SHADOW)],
                                            value_used=rule, used="rule", project="p1"))
            labels.append(L.DecisionLabel(id=f"dl-lab-{i:04d}", record=f"dr-rec-{i:04d}", point="column.is_identifier", value=truth, source="benchmark", by="replica", at=AT))
        board = E.from_ledger(records, labels)
        by = {s.answerer: s for s in board.scores}
        self.assertEqual((by["rule:key_named"].accuracy, by["rule:key_named"].confusion), (0.5, {"tp": 1, "fp": 1, "fn": 1, "tn": 1}))
        self.assertEqual((by["identifier-gbm@1"].accuracy, by["identifier-gbm@1"].f1), (1.0, 1.0))
        self.assertEqual(by["identifier-gbm@1"].truth, "benchmark")


class DashboardTest(unittest.TestCase):
    def test_the_board_reads_the_twin_folder_and_the_route_is_the_reviewers(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd import evaluation
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:  # pragma: no cover
            self.skipTest(str(error))
        home = Path(tempfile.mkdtemp())
        folder = home / "twin" / "sessions" / "2026" / "10" / "16" / "draft-abc"
        folder.mkdir(parents=True)
        first = T.step("x:1", "draft-abc", "home.pack", AT, {"t": 1}, T.ArmAnswer(arm="system", answerer="home", value={"pack": "tabular"}),
                       T.ArmAnswer(arm="ai_first", answerer="ai_first@home", value={"pack": "churn"}))
        picked = T.with_truth(first, T.Truth(value={"pack": "churn"}, source="person_pick"))
        (folder / "steps.jsonl").write_text(first.model_dump_json() + "\n" + picked.model_dump_json() + "\n")
        result = evaluation.board(home)
        self.assertEqual(result["sources"]["twin_steps"], 1)  # the pick replaced the earlier line of the same step
        self.assertEqual(result["board"]["scores"], [])  # a pick is a preference, not a truth the models are scored against
        self.assertEqual((result["twin"]["steps"][0]["judged"], result["twin"]["steps"][0]["mode"]), (1, "undecided"))  # the matrix keeps it
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            client = TestClient(create_app(home))
            out = client.get("/api/evaluation").json()
        self.assertEqual(out["summary"]["judged_twin_steps"], 1)
        self.assertEqual(out["summary"]["cases"], 0)
        from dclab_rnd.accounts import guard

        self.assertEqual(guard.permission("GET", "/api/evaluation"), "review")


if __name__ == "__main__":
    unittest.main()
