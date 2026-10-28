"""36.2 on Home: each result Home produces and each act of the person's on it is a name in the draft's ordered actions;
the verdict read from them (dclab_core.track_outcome) is kept on the draft and recorded through capture.track_outcome when
it changes. A refresh is no verdict and never a second result; a model's save on its own is not the person's act."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.ledger import capture  # noqa: E402


class ActedTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        self.ledger = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.ledger)
        self.addCleanup(ledger.install, None)
        self.agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        d = self.store.create("Show sales by month.")
        self.id = d["id"]
        intent = capture.intent_read(self.id, {"track": "analysis", "rule_track": "analysis"}, has_data=True, words=4)
        exit_ = capture.request_exit(self.id, "analysis", "analysis", None)
        self.store.update(self.id, lambda x: x.update(intent={"track": "analysis", "record": intent},
                                                     exit={"kind": "analysis", "track": "analysis", "record": exit_}))

    def current(self, point):
        labels = self.ledger.labels(point=point)
        gone = {lb.supersedes for lb in labels if lb.supersedes}
        return [lb.value for lb in labels if lb.id not in gone and lb.value is not None]

    def test_a_kept_answer_is_the_verdict_on_the_draft_and_a_label_on_its_route(self):
        self.agent.acted(self.id, "answer")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "open")
        self.agent.acted(self.id, "save")
        d = self.store.get(self.id)
        self.assertEqual((d["actions"], d["outcome"]["verdict"], d["outcome"]["by"]), (["answer", "save"], "kept", "person"))
        self.assertEqual(self.current("request.exit"), ["analysis"])
        self.assertEqual(self.current("request.intent"), ["analysis"])
        self.agent.acted(self.id, "unsave")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "refused")
        self.assertEqual(self.current("request.exit"), [])

    def test_nothing_produced_keeps_no_outcome(self):
        self.agent.acted(self.id, "save")
        self.assertNotIn("outcome", self.store.get(self.id))

    def test_the_action_list_is_capped_and_names_only(self):
        for _ in range(300):
            self.agent.acted(self.id, "answer")
        self.agent.acted(self.id, "not a real action")
        actions = self.store.get(self.id)["actions"]
        self.assertLessEqual(len(actions), 200)
        self.assertNotIn("not a real action", actions)

    def test_a_refresh_is_never_a_second_result_and_the_same_verdict_is_recorded_once(self):
        self.agent.acted(self.id, "dashboard")
        self.agent.acted(self.id, "refresh")  # not an action the rule reads
        self.agent.acted(self.id, "thumb_up")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "kept")
        self.agent.acted(self.id, "thumb_up")
        exits = [lb for lb in self.ledger.labels(point="request.exit") if lb.by == capture.OUTCOME_BY]
        self.assertEqual(len(exits), 1)  # kept twice: one label

    def test_only_a_whole_no_on_a_step_is_a_refusal(self):
        self.store.update(self.id, lambda x: x.update(questions=[{"id": "q1", "step": 0}, {"id": "q2", "step": 0}],
                                                     recipe={"planned": [], "decisions": {}}))
        self.agent.acted(self.id, "table")
        self.agent.confirm_step(self.id, {"id": "q1", "step": 0}, "not sure")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "open")  # skipped, but no verdict
        self.agent.confirm_step(self.id, {"id": "q2", "step": 0}, "no")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "refused")

    def test_a_save_by_the_person_counts_and_a_models_save_alone_does_not(self):
        self.store.update(self.id, lambda x: x.update(answers=[{"n": 0, "plan": {}, "by": "rules"}]))
        self.agent.acted(self.id, "answer")
        self.agent.save_answer(self.id, 0, True, by="model")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "open")
        self.agent.save_answer(self.id, 0, True)
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "kept")
        self.agent.save_answer(self.id, 0, False)
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "refused")



class ThumbTest(ActedTest):
    """A thumb up or down from someone who may change or review the work, on the reply that delivered a result, is the
    person's verdict; a thumb on any other reply, a viewer's thumb, or "wrong" (a correction) is not."""

    def thumb(self, ref, rating, role=None):
        from types import SimpleNamespace
        from unittest import mock

        from dclab_rnd import context
        from dclab_rnd.agentic.pages import ledger as page
        from dclab_rnd.draft.work import DraftWork

        services = SimpleNamespace(ledger=self.ledger, draft_work=DraftWork(self.store, SimpleNamespace(client=lambda *a, **k: None, available=lambda *a, **k: False)))
        who = SimpleNamespace(role=role, user_id="u1") if role else None
        with context.using(services), mock.patch("dclab_rnd.accounts.principal.current", lambda: who):
            return page.feedback({"on": "message", "ref": ref, "rating": rating, "draft": self.id}, self.ledger)

    def board(self, kind="dashboard"):
        self.agent.acted(self.id, "dashboard")
        return self.agent.say(self.id, "Your dashboard is ready.", kind=kind)["id"]

    def test_a_thumb_on_the_reply_that_delivered_the_result_is_the_verdict(self):
        mid = self.board()
        self.thumb(mid, "helpful")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "kept")
        self.assertEqual(self.current("request.exit"), ["analysis"])
        self.thumb(mid, "unhelpful")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "refused")
        self.assertEqual(self.current("request.exit"), [])

    def test_a_thumb_on_another_reply_or_wrong_is_no_verdict(self):
        self.board()
        other = self.agent.say(self.id, "Which period do you mean?", kind="text")["id"]
        self.thumb(other, "helpful")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "open")
        mid = self.agent.say(self.id, "Here it is again.", kind="dashboard")["id"]
        self.thumb(mid, "wrong")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "open")

    def test_a_viewers_thumb_is_no_verdict(self):
        mid = self.board()
        self.thumb(mid, "helpful", role="viewer")
        self.assertEqual(self.store.get(self.id)["outcome"]["verdict"], "open")


if __name__ == "__main__":
    unittest.main()
