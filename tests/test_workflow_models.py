"""Package 26.3: decision models for the workflow, labelled from journeys, trained and judged once."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dclab_core import workflow_labels as wl
from dclab_core.ledger import POINTS

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    HAVE_ML = True
except ImportError:  # .venv-agent
    HAVE_ML = False


def _row(i, group, state, label, value_used, split="train", point="workflow.next_question"):
    return {"record": f"r{i}", "point": point, "state": state, "options": ["target", "prediction_moment", "action", "labels", "none"],
            "value_used": value_used, "used": "rule", "group": group, "source": "dev", "seed": 0, "split": split,
            "value_raw": value_used, "label": {"value": label, "accepted": [label], "source": "benchmark", "why": "test", "by": wl.VERSION}}


def _rows(n_people=12, split="train"):
    rows = []
    for p in range(n_people):
        for step, (known, label) in enumerate(((False, "target"), (True, "prediction_moment"), (True, "none"))):
            state = {"open_questions": 3 - step, "step": step + 1, "outcome_known": known, "moment_known": step == 2, "pack": "tabular"}
            rows.append(_row(len(rows), f"{split}-P{p}", state, label, label, split))
    return rows


class LabelTests(unittest.TestCase):
    truth = wl.Truth(families=("rare_event", "anomaly"), kind="rare_event", answers=frozenset({"target", "labels", "prediction_moment"}))

    def test_the_family_label_never_follows_dclabs_answer(self):
        for answered in ("anomaly", "classification", None):
            got = wl.label("problem.family", answered, {}, [], self.truth)
            self.assertEqual((got.value, got.accepted), ("rare_event", ("rare_event", "anomaly")))
        self.assertEqual(wl.canonical("anomaly", got), "rare_event")  # an accepted answer is scored as the label
        self.assertEqual(wl.canonical("classification", got), "classification")

    def test_ready_is_right_only_with_the_persons_family(self):
        self.assertIs(wl.label("problem.ready", True, {"family": "rare_event"}, [], self.truth).value, True)
        self.assertIs(wl.label("problem.ready", True, {"family": "classification"}, [], self.truth).value, False)
        self.assertIs(wl.label("problem.ready", False, {"family": "classification"}, [], self.truth).value, False)
        self.assertIsNone(wl.label("problem.ready", False, {"family": "rare_event"}, [], self.truth))  # held back with the right family: unknown

    def test_a_question_is_accepted_when_the_person_can_answer_it(self):
        opening = {"open_questions": 3, "outcome_known": False, "moment_known": False}
        got = wl.label("workflow.next_question", "action", opening, [], self.truth)
        self.assertEqual(got.value, "target")
        self.assertEqual(set(got.accepted), {"target", "prediction_moment", "labels"})  # never what they cannot answer, never stopping
        self.assertEqual(wl.canonical("labels", got), "target")
        self.assertEqual(wl.canonical("labels_check", got), "target")  # asked under a second name
        self.assertEqual(wl.canonical("none", got), "none")
        settled = {"open_questions": 1, "outcome_known": True, "moment_known": True}  # only the action is open, and they cannot answer it
        done = wl.label("workflow.next_question", "labels", settled, [], self.truth)
        self.assertEqual((done.value, done.accepted), ("none", ("none", "labels")))  # what is learned once settled: to stop

    def test_the_conversations_own_rules_are_not_judged(self):
        from dclab_rnd.draft import chat

        self.assertEqual(wl.QUESTION_LIMIT, chat.MAX_QUESTIONS)
        capped = {"open_questions": 1, "outcome_known": True, "moment_known": True, "step": 5, "steps_left": 0}
        self.assertIsNone(wl.label("workflow.next_question", "none", capped, [], self.truth))  # the cap: stopping is the product's rule
        again = {"open_questions": 1, "outcome_known": True, "moment_known": True, "step": 2, "steps_left": 3}
        self.assertIsNone(wl.label("workflow.next_question", "target", again, [], self.truth))  # asked again with the columns
        cases = {"open_questions": 2, "outcome_known": False, "moment_known": True, "step": 3, "steps_left": 2, "labels": "file"}
        self.assertEqual(wl.label("workflow.next_question", "none", cases, [], self.truth).value, "none")  # a file of cases defines the outcome

    def test_a_route_is_labelled_by_what_the_journey_expects_and_a_refusal_is_not(self):
        programme = wl.Truth(families=("anomaly",), approaches=("anomaly", "rare_event", "review_budgeted"))
        got = wl.label("solution.route", "anomaly", {"parts": 4}, ["rare_event", "anomaly"], programme)
        self.assertEqual((got.value, got.accepted), ("rare_event", ("rare_event", "anomaly")))
        self.assertIsNone(wl.label("solution.route", None, {"parts": 4}, ["review_budgeted"], programme))  # every approach blocked
        self.assertIsNone(wl.label("solution.route", "entity_tables", {"parts": 4}, ["entity_tables"], programme))
        single = wl.Truth(families=("classification",), kind="binary")
        self.assertEqual(wl.label("solution.route", "multiclass", {"parts": 1}, ["binary", "multiclass"], single).value, "binary")
        self.assertIsNone(wl.label("solution.route", "anomaly", {"parts": 4}, ["anomaly"], single))  # one kind says nothing of four parts

    def test_every_workflow_point_is_learned_or_says_why_not(self):
        workflow = {k for k, p in POINTS.items() if p.kind == "workflow" and not k.startswith("ops.")}
        self.assertEqual(workflow - set(wl.POINTS) - set(wl.NOT_LEARNED), set())
        self.assertEqual(set(wl.POINTS) & set(wl.NOT_LEARNED), set())

    def test_truth_of_reads_a_programme_journey(self):
        t = wl.truth_of({"id": "JP", "answers": {"target": "x", "budget": ""}, "expect": {"families": ["anomaly"], "approaches": ["anomaly", None]}})
        self.assertEqual((t.families, t.approaches, t.answers), (("anomaly",), ("anomaly",), frozenset({"target"})))


class RunnerLabelTests(unittest.TestCase):
    def test_a_stated_family_is_left_out_and_the_rules_are_scored_by_what_is_accepted(self):
        from dclab_rnd import workflow_models as W

        truths = {"dev:JD-9": wl.Truth(families=("classification", "text_or_images"))}
        raw = [{"record": "dev:JD-9:s0:0", "point": "problem.family", "state": {}, "options": [], "value_raw": "text_or_images", "used": "rule",
                "group": "dev:JD-9", "journey": "JD-9", "source": "dev", "seed": 0},
               {"record": "dev:JD-9:s0:1", "point": "problem.family", "state": {}, "options": [], "value_raw": "classification", "used": "person",
                "group": "dev:JD-9", "journey": "JD-9", "source": "dev", "seed": 0}]
        rows = W.label_rows(raw, truths)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["value_raw"], rows[0]["value_used"], rows[0]["label"]["value"]), ("text_or_images", "classification", "classification"))

    def test_collecting_never_opens_the_sealed_journeys(self):
        from dclab_rnd import workflow_models as W

        with mock.patch.object(W, "SEALED_FILE", Path("/nonexistent/sealed.json")):
            items = W.journeys()
        self.assertTrue(items and not any(s == "workflow_sealed" for s, _ in items))
        self.assertEqual(len({f"{s}:{j['id']}" for s, j in items}), len(items))  # a person is a set and an id: no two share a group


class SealedSetTests(unittest.TestCase):
    def test_the_sealed_journeys_are_the_ones_pinned(self):
        from dclab_rnd import workflow_models as W

        self.assertEqual(hashlib.sha256(W.SEALED_FILE.read_bytes()).hexdigest(), W.SEALED_HASH.read_text(encoding="utf-8").strip())


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class TrainAndGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        from dclab_rnd import workflow_models as W

        self.W = W
        folder = self.tmp / "sets" / "decisions" / "workflow.next_question"
        folder.mkdir(parents=True)
        rows = _rows(12) + _rows(4, "dev")
        folder.joinpath("rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_a_card_per_point_says_its_journeys_limits_and_a_point_without_rows_is_skipped(self):
        got = self.W.train(sets=self.tmp / "sets", out=self.tmp / "models")
        card = got["cards"]["workflow.next_question"]
        self.assertEqual(card.limits, self.W.LIMITS)
        self.assertIsNotNone(card.dev and card.rule_dev)
        self.assertIn("problem.family", got["skipped"])
        fitted = json.loads((self.tmp / "models" / "fitted.json").read_text())
        self.assertEqual(set(fitted), {"workflow.next_question"})

    def test_training_refuses_a_sealed_row(self):
        from dclab_core.decision_models import train

        with self.assertRaises(ValueError):
            train("workflow.next_question", _rows(4) + [dict(_rows(1)[0], split="sealed")])

    def test_the_sealed_journeys_are_judged_once(self):
        self.W.train(sets=self.tmp / "sets", out=self.tmp / "models")
        sealed = [_row(i, f"S{i // 3}", r["state"], r["label"]["value"], r["value_used"], "sealed") for i, r in enumerate(_rows(10))]
        played = [{"id": "S", "rows": sealed, "error": None}]
        campaign = self.tmp / "campaign"
        paths = self.W.judge(models=self.tmp / "models", campaign=campaign, played=played)
        doc = json.loads(paths[0].read_text())
        self.assertEqual((doc["point"], doc["sealed"]["decisions"], doc["gate"]["decision_gate"]), ("workflow.next_question", 30, "decision_gate_v1"))
        self.assertIn(doc["result"], ("pass", "fail"))
        self.assertTrue(self.W.spent(campaign))
        with self.assertRaises(ValueError):
            self.W.judge(models=self.tmp / "models", campaign=campaign, played=played)

    def test_a_sealed_journey_that_did_not_play_fails_every_point(self):
        self.W.train(sets=self.tmp / "sets", out=self.tmp / "models")
        sealed = [_row(i, f"S{i // 3}", r["state"], r["label"]["value"], r["value_used"], "sealed") for i, r in enumerate(_rows(10))]
        played = [{"id": "S", "source": "workflow_sealed", "rows": sealed, "error": None},
                  {"id": "T", "source": "workflow_sealed", "rows": [], "error": "RuntimeError: broke"}]
        doc = json.loads(self.W.judge(models=self.tmp / "models", campaign=self.tmp / "campaign", played=played)[0].read_text())
        self.assertEqual(doc["result"], "fail")
        self.assertEqual(doc["sealed"]["failed_plays"], ["workflow_sealed:T"])

    def test_a_crash_after_the_sealed_set_is_opened_still_spends_it(self):
        self.W.train(sets=self.tmp / "sets", out=self.tmp / "models")
        campaign = self.tmp / "campaign"
        with mock.patch.object(self.W, "_played", side_effect=RuntimeError("crashed while playing")), \
                mock.patch.object(self.W, "journeys", return_value=[]):
            with self.assertRaises(RuntimeError):
                self.W.judge(models=self.tmp / "models", campaign=campaign)
        self.assertTrue(self.W.spent(campaign))
        self.assertEqual(list((campaign / "results").glob("DWM-*.json")) if (campaign / "results").exists() else [], [])

    def test_a_model_whose_bytes_changed_is_never_judged(self):
        self.W.train(sets=self.tmp / "sets", out=self.tmp / "models")
        (self.tmp / "models" / "workflow.next_question.joblib").write_bytes(b"swapped")
        with self.assertRaises(ValueError):
            self.W.judge(models=self.tmp / "models", campaign=self.tmp / "campaign", played=[])

    def test_a_workflow_model_answers_in_shadow_and_a_column_folder_alone_stays_alone(self):
        from dclab_core.ledger import DecisionRecord
        from dclab_rnd.ledger import shadow

        self.W.train(sets=self.tmp / "sets", out=self.tmp / "models")
        shadow.answerers.cache_clear()
        try:
            with mock.patch.object(shadow, "MODELS", self.tmp / "none"), mock.patch.object(shadow, "WORKFLOW_MODELS", self.tmp / "models"):
                models = shadow.answerers()
            self.assertEqual([m.point for m in models], ["workflow.next_question"])
            self.assertEqual(shadow.answerers(str(self.tmp / "none")), ())
            state = {"open_questions": 3, "step": 1, "outcome_known": False, "moment_known": False, "pack": "tabular"}
            record = DecisionRecord(id="dec-000001", point="workflow.next_question", subject={"kind": "conversation", "key": "c1"}, state=state,
                                    options=["target", "prediction_moment", "action", "labels", "none"],
                                    answers=[{"by": "rule", "answerer": "rule:plan_fields", "value": "target"}], value_used="target", used="rule",
                                    at="2026-10-17T00:00:00+00:00")
            self.assertTrue(shadow.answerable(record))
            column = record.model_copy(update={"point": "column.is_identifier", "state": {"column": "customer_id"}})
            self.assertFalse(shadow.answerable(column))  # a column decision with only a name: still nothing to answer from
            out = shadow.add_many([record], models)
            self.assertEqual([a.source for a in out[0].answers][-1], "shadow")
            self.assertEqual(out[0].value_used, "target")  # a shadow is never used
        finally:
            shadow.answerers.cache_clear()


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class PlayTests(unittest.TestCase):
    def test_a_journey_leaves_labelled_workflow_decisions_up_to_its_build(self):
        from dclab_rnd import workflow_models as W

        journey = next(j for s, j in W.journeys() if j["id"] == "JD-02")
        got = W.play(("dev", journey, 0))
        self.assertIsNone(got["error"])
        points = {r["point"] for r in got["rows"]}
        self.assertTrue({"problem.family", "problem.ready", "workflow.next_question"} <= points)
        self.assertTrue(all(r["group"] == "dev:JD-02" and "label" not in r for r in got["rows"]))  # labelled apart, by label_rows
        rows = W.label_rows(got["rows"], W.truths([("dev", journey)]))
        family = next(r for r in rows if r["point"] == "problem.family")
        self.assertEqual(family["label"]["value"], "classification")
        self.assertTrue(all(isinstance(r["state"], dict) and "words" not in r["state"] for r in got["rows"]))



@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class RoutingTests(unittest.TestCase):
    def test_a_workflow_model_is_a_candidate_only_after_its_sealed_pass(self):
        from dclab_rnd.ledger import routing, shadow

        tmp = Path(tempfile.mkdtemp())
        with mock.patch.object(shadow, "GATE", tmp / "none"), mock.patch.object(shadow, "WORKFLOW_GATE", tmp):
            self.assertEqual(routing.passed(), {})
            doc = {"point": "workflow.next_question", "result": "fail", "model": {"name": "wf", "version": "abc"}, "gate": {"version": "g"}}
            (tmp / "DWM-001_workflow.next_question.json").write_text(json.dumps(doc))
            self.assertEqual(routing.passed(), {})
            (tmp / "DWM-002_workflow.next_question.json").write_text(json.dumps({**doc, "result": "pass", "model": {"name": "wf", "version": "def"}}))
            self.assertEqual(list(routing.passed()), ["wf@def"])


if __name__ == "__main__":
    unittest.main()
