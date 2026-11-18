"""The owner, 2026-11-18: everything Claude wrote (labels, answers, thinking, and the words of written people and requests)
is kept and prepared, and no training run or export uses it until the owner records a rights review. One module says who
wrote every kind of stored row."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.store import writers as W  # noqa: E402

DEC = "dclab_core.decision_example.v4"
RT = "dclab_core.routing_example.v1"


def routing(**k):
    return {"schema_version": RT, "label": "analysis", **k}


class WriterTests(unittest.TestCase):
    def setUp(self):
        hold = mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=False)  # the owner's hold, whatever the file says
        hold.start()
        self.addCleanup(hold.stop)

    def test_a_decision_rows_teacher_label_is_a_models_and_a_measured_one_is_not(self):
        self.assertEqual(W.writer_of({"schema_version": DEC, "label": True, "label_source": "teacher"}), "frontier:teacher")
        for source in ("person", "measured", "benchmark", "blind"):
            self.assertIsNone(W.writer_of({"schema_version": DEC, "label": True, "label_source": source}), source)
        self.assertIsNone(W.writer_of({"schema_version": DEC, "label": None, "label_source": "teacher"}))  # no label: nothing trains on it

    def test_a_routing_rows_label_and_words_name_their_writer(self):
        self.assertEqual(W.writer_of(routing(label_source="silver", label_by="model:claude-sonnet-5-5:routing_labeler")), "claude:claude-sonnet-5-5")
        self.assertEqual(W.writer_of(routing(label_source="silver", label_by="model:model:claude-sonnet-5-5")), "claude:claude-sonnet-5-5")
        qwen = [{"by": "model", "answerer": "home_intent", "model": "qwen/qwen3.8-flash"}]
        self.assertIsNone(W.writer_of(routing(label_source="silver", label_by="model:home_intent", readings=qwen)))  # the owner's yes to Qwen
        self.assertEqual(W.writer_of(routing(label_source="silver", label_by="model:home_intent")), "frontier:unknown")
        self.assertEqual(W.writer_of(routing(label_source="written", label_by="written:intent_requests_v1:a", origin="written")), "claude:claude-sonnet-5-5")
        self.assertEqual(W.writer_of(routing(label_source="written", label_by="eval:requests_dev", origin="eval")), "claude:claude_code")
        self.assertEqual(W.writer_of(routing(label_source="written", label_by="conversation_exam:people", origin="eval")), "frontier:exam_person")
        self.assertEqual(W.writer_of(routing(label=None, origin="scripted")), "claude:claude_code")  # a test's scripted person
        self.assertIsNone(W.writer_of(routing(label_source="real", label_by="person:card", origin="real")))

    def test_a_models_reading_is_held_unless_its_model_is_the_approved_qwen(self):
        self.assertTrue(W.reading_held({"by": "model", "model": "claude-sonnet-5-5"}))
        self.assertTrue(W.reading_held({"by": "model"}))  # no model named: the safe side
        self.assertFalse(W.reading_held({"by": "model", "model": "qwen/qwen3.8-flash"}))
        self.assertFalse(W.reading_held({"by": "rule", "model": "claude-sonnet-5-5"}))

    def test_judged_episode_preference_retrieval_and_experiment_rows(self):
        judged = {"schema_version": "dclab_core.judged_example.v1"}
        self.assertEqual(W.writer_of({**judged, "candidate_by": "model:claude-haiku-5-5"}), "claude:claude-haiku-5-5")
        self.assertEqual(W.writer_of({**judged, "candidate_by": "reviewer"}), "claude:claude_code")
        self.assertEqual(W.writer_of({**judged, "candidate_by": "development"}), "claude:claude_code")
        self.assertEqual(W.writer_of({**judged, "candidate_by": "human", "judge": "developer"}), "claude:claude_code")
        self.assertIsNone(W.writer_of({**judged, "candidate_by": "human", "judge": "validator"}))
        self.assertEqual(W.writer_of({"schema_version": "dclab_core.episode.v3", "process": "dclab_code"}), "claude:claude_code")
        self.assertIsNone(W.writer_of({"schema_version": "dclab_core.episode.v3", "process": "walk"}))
        self.assertIsNone(W.writer_of({"schema_version": "dclab_core.experiment_record.v1"}))
        search = {"schema_version": "dclab_core.retrieval_example.v1", "provenance": {"rule": "capture_rules_v2:agent_step"}}
        self.assertEqual(W.writer_of(search), "frontier:agent_search")

    def test_a_chat_row_from_a_test_workspace_is_held_and_one_from_the_owners_workspace_is_not(self):
        chat = {"schema_version": "dclab_core.chat_example.v3", "task": "t", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(W.writer_of({**chat, "provenance": {"workspace": "live_tests__2026-10-30-journeys"}}), "claude:claude_code")
        self.assertIsNone(W.writer_of({**chat, "provenance": {"workspace": "dev"}}))
        self.assertEqual(W.writer_of({**chat, "provenance": {"workspace": "dev"}, "check": {"model": "claude-sonnet-5-5"}}), "claude:claude-sonnet-5-5")

    def test_written_sets_name_their_writer(self):
        self.assertEqual(W.set_writer("intent_requests_v1"), "claude:claude-sonnet-5-5")
        self.assertEqual(W.set_writer("homework_v1"), "claude:claude-sonnet-5-5")
        self.assertEqual(W.set_writer("homework_v2"), "frontier:journey_writer_b")
        self.assertEqual(W.set_writer("sealed_v3"), "claude:claude_code")  # any other written set: a Claude Code session
        self.assertIsNone(W.set_writer(None))

    def test_held_follows_the_owners_switch(self):
        row = {"schema_version": DEC, "label": True, "label_source": "teacher"}
        self.assertTrue(W.held(row))
        with mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=True):
            self.assertFalse(W.held(row))  # after the rights review: every row trains again, no code change
            self.assertFalse(W.held_writer("claude:x"))
        self.assertTrue(W.held_writer("claude:x"))
        self.assertFalse(W.held_writer("openrouter:qwen/qwen3.8-flash"))
        self.assertFalse(W.held_writer(None))

    def test_the_audit_counts_every_set_and_never_reads_a_sealed_file(self):
        from dclab_rnd.store import Layout

        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        base = layout.sets("m1_students", "column.time", "v1")
        base.mkdir(parents=True)
        rows = [{"schema_version": DEC, "label": True, "label_source": "teacher"}, {"schema_version": DEC, "label": True, "label_source": "person"}]
        (base / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (base / "sealed.jsonl").write_text("not json: the audit must never read this file\n", encoding="utf-8")
        report = W.audit(layout)
        self.assertEqual(report, {"m1_students/column.time/v1": {"rows": 2, "provider": {"frontier": 1}}})


if __name__ == "__main__":
    unittest.main()
