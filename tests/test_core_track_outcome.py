"""36.2: what a person did with what Home produced. The verdict is read by rule from the draft's ordered actions (the Home
session's rule, folded in here), recorded as one event and, when the person kept or changed it, as a person's label on
the draft's routing and reading; the store's Home conversation carries it as its outcome. Names and counts only."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core import track_outcome as O  # noqa: E402
from dclab_core.contracts import CONTRACTS  # noqa: E402


class VerdictTests(unittest.TestCase):
    def read(self, *actions, track="analysis"):
        return O.read(track, list(actions))

    def test_nothing_produced_is_no_outcome(self):
        self.assertIsNone(self.read())
        self.assertIsNone(self.read("save", "undo"))  # a verdict on nothing is not an outcome

    def test_a_result_nobody_judged_is_open_and_by_the_rules(self):
        o = self.read("answer")
        self.assertEqual((o.verdict, o.by, o.produced), ("open", "rule", ["answer"]))

    def test_the_person_keeping_changing_or_refusing_decides(self):
        self.assertEqual((self.read("answer", "save").verdict, self.read("answer", "save").by), ("kept", "person"))
        self.assertEqual(self.read("answer", "edit").verdict, "changed")
        self.assertEqual(self.read("table", "step_no", track="preparation").verdict, "refused")
        self.assertEqual(self.read("table", "undo", track="preparation").verdict, "refused")
        self.assertEqual(self.read("table", "export", track="preparation").verdict, "kept")
        self.assertEqual(self.read("answer", "save", "unsave").verdict, "refused")  # the last decisive action wins

    def test_a_thumb_is_a_verdict_and_a_dashboard_refresh_is_not(self):
        self.assertEqual(self.read("dashboard", "thumb_up", track="reporting").verdict, "kept")
        self.assertEqual(self.read("answer", "thumb_down").verdict, "refused")
        o = self.read("dashboard", "refresh", track="reporting")  # recomputing the tiles says nothing about them
        self.assertEqual((o.verdict, o.by), ("open", "rule"))

    def test_a_second_result_without_judging_the_first_is_asked_again_by_the_rules(self):
        o = self.read("answer", "answer")
        self.assertEqual((o.verdict, o.by), ("asked_again", "rule"))
        self.assertEqual(self.read("answer", "answer", "save").verdict, "kept")  # judged later: the person decided

    def test_a_kept_result_stays_kept_while_the_next_one_is_pending(self):
        self.assertEqual(self.read("answer", "save", "answer").verdict, "kept")
        self.assertEqual(self.read("answer", "save", "answer", "answer").verdict, "asked_again")

    def test_unknown_actions_are_ignored_and_the_evidence_is_names_only(self):
        o = self.read("answer", "wiggle", "save", *["save"] * 8)
        self.assertEqual(o.verdict, "kept")
        self.assertNotIn("wiggle", o.evidence)
        self.assertLessEqual(len(o.evidence), 5)
        self.assertEqual(o.actions, 10)

    def test_the_rules_never_keep_change_or_refuse_and_the_person_never_asks_again(self):
        for verdict, by in (("kept", "rule"), ("changed", "rule"), ("refused", "rule"), ("asked_again", "person"), ("open", "person")):
            with self.assertRaises(ValueError, msg=(verdict, by)):
                O.TrackOutcome(track="analysis", produced=["answer"], verdict=verdict, by=by, actions=2)

    def test_the_contract_is_registered_and_reads_back(self):
        self.assertIn("track_outcome", CONTRACTS)
        o = self.read("brief", "notebook", track="business")
        self.assertEqual(O.TrackOutcome.read(o.model_dump_json()), o)


class EventTests(unittest.TestCase):
    def test_the_event_kind_is_in_the_closed_vocabulary_and_holds_no_content(self):
        from dclab_core import experience

        fields = experience.KINDS["track.outcome"]
        self.assertTrue(all(c in (experience.M, experience.C) for c in fields.values()))  # names and counts only
        self.assertIn("dclab_core.experience_event.v15", experience.READ_AS_CURRENT)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import experience, ledger

        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.ledger = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.ledger)
        self.addCleanup(ledger.install, None)
        self.log = experience.open_log(self.home)

    def routed(self, track="analysis", kind="analysis"):
        from dclab_rnd.ledger import capture

        intent = capture.intent_read("d1", {"track": track, "rule_track": track}, has_data=True, words=6)
        exit_ = capture.request_exit("d1", track, kind, None)
        return intent, exit_

    def labels(self, point):
        return [(lb.value, lb.source) for lb in self.ledger.labels(point=point)]

    def test_kept_by_the_person_labels_the_exit_and_the_reading(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        n = capture.track_outcome("d1", O.read("analysis", ["answer", "save"]), exit_record=exit_, exit_kind="analysis", intent_record=intent,
                                  log=self.log)
        self.assertEqual(n, 2)
        self.assertIn(("analysis", "user_decision_exposed"), self.labels("request.exit"))
        self.assertIn(("analysis", "user_decision_exposed"), self.labels("request.intent"))

    def test_refused_or_asked_again_never_becomes_a_guessed_label(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        for actions in (["answer", "unsave"], ["answer", "answer"], ["answer"]):
            self.assertEqual(capture.track_outcome("d1", O.read("analysis", actions), exit_record=exit_, exit_kind="analysis",
                                                   intent_record=intent, log=self.log), 0, actions)
        self.assertEqual(self.labels("request.exit"), [])
        self.assertEqual(self.labels("request.intent"), [])

    def current(self, point):
        labels = self.ledger.labels(point=point)
        gone = {lb.supersedes for lb in labels if lb.supersedes}
        return [(lb.value, lb.by) for lb in labels if lb.id not in gone and lb.value is not None]

    def test_a_keep_then_a_refusal_leaves_no_confirmed_route_and_a_second_keep_restores_it(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        ids = dict(exit_record=exit_, exit_kind="analysis", intent_record=intent, log=self.log)
        self.assertEqual(capture.track_outcome("d1", O.read("analysis", ["answer", "save"]), **ids), 2)
        self.assertEqual(capture.track_outcome("d1", O.read("analysis", ["answer", "save"]), **ids), 0)  # no duplicate label
        self.assertEqual(capture.track_outcome("d1", O.read("analysis", ["answer", "save", "unsave"]), **ids), 2)  # two retractions
        self.assertEqual(self.current("request.exit"), [])
        self.assertEqual(self.current("request.intent"), [])
        self.assertEqual(capture.track_outcome("d1", O.read("analysis", ["answer", "save", "unsave", "save"]), **ids), 2)
        self.assertEqual(self.current("request.exit"), [("analysis", capture.OUTCOME_BY)])

    def test_the_persons_own_track_choice_is_never_retracted(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        capture.intent_chosen(intent, "analysis")
        ids = dict(exit_record=exit_, exit_kind="analysis", intent_record=intent, log=self.log)
        capture.track_outcome("d1", O.read("analysis", ["answer", "save"]), **ids)
        capture.track_outcome("d1", O.read("analysis", ["answer", "save", "undo"]), **ids)
        self.assertEqual(self.current("request.intent"), [("analysis", "person")])

    def test_an_unknown_exit_or_track_labels_nothing(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        self.assertEqual(capture.track_outcome("d1", O.read("knitting", ["answer", "save"]), exit_record=exit_, exit_kind="spaceship",
                                               intent_record=intent, log=self.log), 0)

    def test_an_unjudged_result_is_an_event_without_a_label(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        capture.track_outcome("d1", O.read("analysis", ["answer"]), exit_record=exit_, exit_kind="analysis", intent_record=intent, log=self.log)
        self.assertEqual([e.payload["verdict"] for e in self.log.read() if e.kind == "track.outcome"], ["open"])
        self.assertEqual(self.labels("request.exit"), [])

    def test_the_event_is_recorded_with_names_and_counts(self):
        from dclab_rnd.ledger import capture

        intent, exit_ = self.routed()
        capture.track_outcome("d1", O.read("analysis", ["answer", "edit"]), exit_record=exit_, exit_kind="analysis", intent_record=intent,
                              log=self.log)
        events = [e for e in self.log.read() if e.kind == "track.outcome"]
        self.assertEqual(len(events), 1)
        p = events[0].payload
        self.assertEqual((p["track"], p["exit"], p["verdict"], p["by"], p["produced"]), ("analysis", "analysis", "changed", "person", ["answer"]))
        self.assertEqual(events[0].ids.conversation, "d1")

    def test_nothing_to_judge_records_nothing(self):
        from dclab_rnd.ledger import capture

        self.assertEqual(capture.track_outcome("d1", None, exit_record=None, exit_kind=None, intent_record=None, log=self.log), 0)
        self.assertEqual([e for e in self.log.read() if e.kind == "track.outcome"], [])


class StoreTests(unittest.TestCase):
    def record(self, body):
        from dclab_core.store import Provenance, StoreRecord

        return StoreRecord(id="r1", kind="conversations", type="draft", at="2026-10-28T10:00:00+00:00", body=body, split="train",
                           provenance=Provenance(workspace="dev", source="drafts/d1.json", entry="e:d1", at="2026-10-28T10:00:00+00:00",
                                                 sha256="0" * 64, consent="train", draft="d1"))

    def body(self, **extra):
        return {"id": "d1", "intent": {"track": "analysis"}, "updated": "2026-10-28T10:00:00+00:00",
                "messages": [{"role": "person", "text": "Sales by month?"}, {"role": "agent", "text": "Here they are."}], **extra}

    def test_the_conversation_carries_its_outcome_and_who_judged_it(self):
        from dclab_rnd.store.sets import llm

        ex = llm.from_draft(self.record(self.body(outcome=O.read("analysis", ["answer", "save"]).model_dump())))
        self.assertEqual((ex.check["outcome"], ex.check["outcome_by"]), ("kept", "person"))
        asked = llm.from_draft(self.record(self.body(outcome=O.read("analysis", ["answer", "answer"]).model_dump())))
        self.assertEqual((asked.check["outcome"], asked.check["outcome_by"]), ("asked_again", "rule"))

    def test_an_open_or_missing_outcome_is_no_outcome(self):
        from dclab_rnd.store.sets import llm

        self.assertNotIn("outcome", llm.from_draft(self.record(self.body())).check)
        open_ = llm.from_draft(self.record(self.body(outcome=O.read("analysis", ["answer"]).model_dump())))
        self.assertNotIn("outcome", open_.check)  # produced, not judged yet: nothing to learn from

    def test_the_store_reads_only_labels_no_later_label_supersedes(self):
        from dclab_core.store import Provenance, StoreRecord
        from dclab_rnd.store.sets import decisions

        def lab(i, value, supersedes=None):
            return StoreRecord(id=i, kind="decisions", type="label", at="2026-10-28T10:00:00+00:00", split="train",
                               body={"id": i, "value": value, "source": "user_decision_exposed", "supersedes": supersedes},
                               provenance=Provenance(workspace="dev", source="ledger/labels.jsonl", entry=i, at="2026-10-28T10:00:00+00:00",
                                                     sha256="0" * 64, consent="train"))
        measured = StoreRecord(id="m", kind="decisions", type="label", at="2026-10-28T10:00:00+00:00", split="train",
                               body={"id": "m", "value": "analysis", "source": "measured"},
                               provenance=Provenance(workspace="dev", source="ledger/labels.jsonl", entry="m", at="2026-10-28T10:00:00+00:00",
                                                     sha256="0" * 64, consent="train"))
        self.assertEqual(decisions._best_label([lab("a", "analysis")]), ("analysis", "person"))
        self.assertEqual(decisions._best_label([lab("a", "analysis"), lab("b", None, "a")]), (None, None))  # kept, then refused
        self.assertEqual(decisions._best_label([measured, lab("a", "analysis"), lab("b", None, "a")]), ("analysis", "measured"))

    def test_coverage_names_the_track_outcome_capture_for_every_track_that_is_not_prediction(self):
        from dclab_rnd.store import coverage

        for track, cap in coverage.TRACK_CAPTURE.items():
            if track not in ("prediction", "advice"):
                self.assertIn("event:track.outcome", cap.paths, track)


if __name__ == "__main__":
    unittest.main()
