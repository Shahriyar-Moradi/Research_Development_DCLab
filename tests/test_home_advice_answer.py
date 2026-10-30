"""2026-10-30: the advice track's own product. Home answers an advice request from the Evidence library (the closest
records with no model; a model's answer only through the citation check), says it with its sources, records it as a
produced answer joined to the draft, and the person's thumb on it is the track's outcome."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class AdviceAnswerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        before = ledger.current_ledger()
        self.addCleanup(ledger.install, before)
        ledger.install(ledger.FileLedger(self.tmp / "ledger", "wtest"))
        self.store = DraftStore(self.tmp / "drafts")
        self.agent = HomeAgent(self.store, None)
        self.draft = self.store.create("How should I avoid leakage when a column is filled in after the outcome?")
        self.store.update(self.draft["id"], lambda d: d.update(intent={"track": "advice", "tracks": ["advice"], "status": "confirmed"}))

    def test_an_advice_request_gets_a_cited_answer_the_person_can_judge(self):
        msg = self.agent.advise(self.draft["id"])
        self.assertIsNotNone(msg)
        self.assertEqual((msg["kind"], msg["track"]), ("answer", "advice"))
        self.assertTrue(msg["sources"])  # the records it rests on
        d = self.store.get(self.draft["id"])
        self.assertEqual((d["outcome"]["verdict"], d["outcome"]["produced"]), ("open", ["answer"]))  # produced, not yet judged
        self.agent.acted(self.draft["id"], "thumb_up")  # what the feedback route's _thumb_outcome does on this reply
        d = self.store.get(self.draft["id"])
        self.assertEqual((d["outcome"]["verdict"], d["outcome"]["by"]), ("kept", "person"))
        import json
        rows = [json.loads(line) for f in (self.tmp / "ledger").rglob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
        search = [r for r in rows if r.get("point") == "evidence.records"]
        self.assertTrue(search and search[0].get("draft") == self.draft["id"])  # the search joined to its conversation

    def test_the_question_is_the_persons_latest_words(self):
        """The review: the start of everything said was searched; a later question was cut off."""
        from unittest import mock

        from dclab_rnd.agentic.pages import evidence

        self.store.update(self.draft["id"], lambda d: d["messages"].append({"id": "u2", "role": "user", "kind": "text", "text": "How do I validate a churn model over time?"}))
        seen = []
        with mock.patch.object(evidence, "ask", lambda q, client, draft_id=None: seen.append(q) or {"records": [], "covered": False, "note": "n"}):
            self.assertIsNone(self.agent.advise(self.draft["id"]))  # nothing covered: nothing produced
        self.assertEqual(seen, ["How do I validate a churn model over time?"])

if __name__ == "__main__":
    unittest.main()
