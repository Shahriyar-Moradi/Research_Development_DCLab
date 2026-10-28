"""A memory of the whole Home conversation (package H5): a fact settled early is still in the model's view at the
thirtieth message, and the summary says nothing the draft's records do not."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft import memory  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class Seeing:
    """A model client that keeps every request it was sent and answers plainly."""

    def __init__(self):
        self.seen = []

    def complete(self, messages, tools=None, **kw):
        self.seen.append(messages)
        return {"content": "Noted.", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}}


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home)
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def test_a_fact_from_the_second_message_is_in_view_at_the_thirtieth(self):
        d = self.store.create("A dashboard of satisfaction for the support team.")
        HomeAgent(self.store, None).start(d["id"])
        rules = HomeAgent(self.store, None)
        rules.reply(d["id"], "Weekly")  # message 2 or so: the refresh, settled early
        self.assertEqual(self.store.get(d["id"])["track_brief"]["reporting"]["refresh"]["value"], "Weekly")

        def chatter(x):
            for i in range(30):
                x["messages"].append({"id": f"u{i}", "role": "user", "kind": "text", "text": f"Another remark number {i}."})
        self.store.update(d["id"], chatter)
        client = Seeing()
        try:
            HomeAgent(self.store, client).model_turn(d["id"], text="Another remark")
        except Exception:  # noqa: BLE001 - only what the model was shown matters here
            pass
        self.assertTrue(client.seen, "the model was never asked")
        shown = json.dumps(client.seen[0], ensure_ascii=False)
        self.assertIn("Weekly", shown)  # settled at the start, still in view
        self.assertNotIn("Another remark number 0.", shown)  # an old remark is not repeated whole
        self.assertLessEqual(sum(1 for m in client.seen[0] if m["role"] != "system"), memory.RECENT)

    def test_the_summary_holds_only_what_the_records_say(self):
        draft = {"messages": [{"role": "user", "text": "hi"}], "intent": {"track": "analysis", "status": "stated", "source": "rules"},
                 "track_brief": {"analysis": {"measure": {"value": "average spend", "source": "rules"}, "period": {"value": "all rows", "source": "default"}}},
                 "questions": [{"id": "q1", "field": "breakdown", "text": "Split it by what?", "answered": None},
                               {"id": "q2", "field": "measure", "text": "What should I measure?", "answered": "average spend"}],
                 "analysis": {"summary": {"rows": 1200, "columns": 9}}}
        s = memory.summary(draft)
        self.assertEqual(s["track"], "analysis")
        self.assertEqual(s["known"]["measure"], {"value": "average spend", "from": "rules", "track": "analysis"})
        self.assertEqual(s["known"]["period"]["from"], "default")
        self.assertEqual(s["open"], [{"field": "breakdown", "question": "Split it by what?"}])
        self.assertEqual(s["answered"], [{"field": "measure", "answer": "average spend"}])
        self.assertEqual(s["data"], {"rows": 1200, "columns": 9})
        import re
        said = set(re.findall(r"\d+", json.dumps(draft)))
        self.assertTrue(set(re.findall(r"\d+", json.dumps({k: v for k, v in s.items() if k != "messages"}))) <= said)  # no number of its own


    def test_what_the_person_stated_becomes_the_projects_memory_and_nothing_else(self):
        from dclab_rnd import memory as project_memory

        drafts = DraftStore(self.home / "drafts")
        d = drafts.create("Predict which customers cancel within 30 days")
        drafts.update(d["id"], lambda x: x.update(
            understanding={"target": "which customers cancel within 30 days", "prediction_moment": "At the time each row is recorded",
                           "assumed": {"prediction_moment": "At the time each row is recorded"}},
            plan={"target": {"status": "stated", "source": "text", "quote": "which customers cancel within 30 days"},
                  "prediction_moment": {"status": "inferred", "source": "default"}},
            track_brief={"reporting": {"refresh": {"value": "Weekly", "source": "user", "quote": "Weekly"},
                                       "audience": {"value": "you", "source": "default", "quote": None}}}))
        self.assertEqual(memory.facts(drafts.get(d["id"])), ["Outcome predicted: which customers cancel within 30 days", "Updates: Weekly"])
        written = memory.hand_off(drafts, drafts.get(d["id"]), "p1")
        self.assertEqual(written, 2)
        items = [i for i in project_memory.open_memory(self.home).log() if i.scope_id == "p1"]
        self.assertEqual(sorted(i.text for i in items), ["Outcome predicted: which customers cancel within 30 days", "Updates: Weekly"])
        self.assertTrue(all(i.source == f"home brief:{d['id']}" for i in items))


if __name__ == "__main__":
    unittest.main()
