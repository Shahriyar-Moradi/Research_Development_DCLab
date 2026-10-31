"""Package 37.9 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): Home, conversation first. With DCLAB_CASE_AGENT on, the lead agent
leads Home's draft conversation; with it off, or with no model, the rules-first Home does. Scripted models only."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


class CaseHomeTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.store = DraftStore(self.folder / "drafts")

    def test_the_switch_chooses_the_lead_agent_and_off_keeps_the_rules_first_home(self):
        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": "1"}):
            self.assertIsInstance(DraftWork(self.store, Models(Script())).agent("d"), CaseHome)
        with mock.patch.dict(os.environ, {"DCLAB_CASE_AGENT": ""}):
            agent = DraftWork(self.store, Models(Script())).agent("d")
            self.assertNotIsInstance(agent, CaseHome)
            self.assertIsInstance(agent, HomeAgent)

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


if __name__ == "__main__":
    unittest.main()
