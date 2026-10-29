"""36.3: written people for the kinds of help with no runs yet (analysis, preparation, reporting, business), played
through Home: dev only, English, nothing sent without --yes and a cap, every conversation kept whole in a workspace
layout and marked scripted (their acts are a script's, never a person's label)."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.agent_eval.home import sparse, tables
    from dclab_rnd.store.keepers import workspaces
    MISSING = ""
except ImportError as error:  # the agent environment lacks pandas
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class PeopleTests(unittest.TestCase):
    def test_four_written_people_per_sparse_track_in_english_on_known_tables(self):
        by_track = {}
        for p in sparse.PEOPLE:
            by_track.setdefault(p["track"], []).append(p)
            self.assertIn(p["table"], tables.TABLES)
            self.assertTrue(p["text"].isascii(), p["id"])  # English first (the owner, 2026-10-28)
        self.assertEqual(sorted(by_track), ["analysis", "business", "preparation", "reporting"])
        self.assertTrue(all(len(v) == 4 for v in by_track.values()))
        self.assertEqual(len({p["id"] for p in sparse.PEOPLE}), len(sparse.PEOPLE))

    def test_nothing_is_sent_without_yes_and_a_cap(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(sparse.main([]), 0)
            self.assertEqual(sparse.main(["--yes"]), 0)
        self.assertIn("nothing was sent", out.getvalue())


@unittest.skipIf(MISSING, MISSING)
class PlayTests(unittest.TestCase):
    def test_a_rules_run_keeps_every_conversation_whole_and_scripted(self):
        keep = Path(tempfile.mkdtemp()) / "run"
        self.addCleanup(shutil.rmtree, keep.parent, True)
        people = [next(p for p in sparse.PEOPLE if p["track"] == t) for t in ("analysis", "preparation")]
        rows = sparse.play(people, keep, client=lambda purpose, draft_id: None)  # no model: the rules alone
        self.assertEqual([r["id"] for r in rows], [p["id"] for p in people])
        self.assertTrue(all(r["error"] is None for r in rows), rows)
        drafts = sorted((keep / "drafts").glob("*/draft.json"))
        self.assertEqual(len(drafts), 2)
        first = json.loads(drafts[0].read_text(encoding="utf-8"))
        self.assertGreaterEqual(sum(1 for m in first["messages"] if m.get("role") in ("user", "person")), 1)
        self.assertTrue(list((keep / "ledger").rglob("*.jsonl")))  # its decisions, for the decision models
        self.assertTrue(workspaces.scripted(keep / "drafts", keep.parent))  # a script's acts, never a person's label
        self.assertEqual({r["track"] for r in rows}, {"analysis", "preparation"})


@unittest.skipIf(MISSING, MISSING)
class FirstRequestTests(unittest.TestCase):
    """36.3's finding: a first request the rules cannot read was answered "say it again" although a model serves Home."""

    def setUp(self):
        from dclab_rnd import ledger
        from dclab_rnd.draft.store import DraftStore

        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def run_request(self, text, table, track, client, brief=None, model_says="I'll read that with my tools."):
        from unittest import mock

        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.chat import HomeAgent

        def model_turn(agent, draft_id, *args, **kwargs):  # the model's turn, scripted: it answers, or stays silent
            if model_says:
                agent.say(draft_id, model_says, kind="text")

        d = self.store.create(text)
        agent = HomeAgent(self.store, client, on_request=lambda *a: None)
        with mock.patch.object(HomeAgent, "model_turn", autospec=True, side_effect=model_turn) as turn:
            agent.start(d["id"])
            if agent.track(self.store.get(d["id"])) != track:
                agent.choose_intent(d["id"], track)
            name = tables.write(table, self.store.data_dir(d["id"]))
            pipeline.run(self.store, d["id"], pipeline.new_asset(self.store, d["id"], "upload", name, name)["id"], agent, None)
            if brief is not None and (agent.pending(self.store.get(d["id"])) or {}).get("brief"):
                agent.reply(d["id"], brief)  # the track's brief question first, as the person answers it
        said = " ".join(m.get("text") or "" for m in self.store.get(d["id"])["messages"])
        return turn, said

    def test_unread_preparation_steps_go_to_the_model_not_back_to_the_person(self):
        text = "Give me only the bookings with a price above 500 and drop the nationality column."
        turn, said = self.run_request(text, "bookings", "preparation", client=object())
        self.assertEqual(turn.call_count, 1)  # the model reads the request with its tools (propose_step)
        self.assertNotIn("Say them one at a time", said)
        turn, said = self.run_request(text, "bookings", "preparation", client=object(), model_says="")  # the model said nothing
        self.assertEqual(turn.call_count, 1)
        self.assertIn("Say them one at a time", said)  # the rules then say what they need (the review of 36.3)
        turn, said = self.run_request(text, "bookings", "preparation", client=None)  # no model: the rules say what they need
        self.assertEqual(turn.call_count, 0)
        self.assertIn("Say them one at a time", said)

    def test_unread_periods_go_to_the_model(self):
        text = "Is revenue in the app channel doing better than in the store lately?"
        turn, said = self.run_request(text, "sales", "business", client=object(), brief="Total amount.")
        self.assertEqual(turn.call_count, 1)  # compare_periods is the model's
        self.assertNotIn("To do it here I need", said)

    def test_never_a_model_turn_inside_a_model_turn(self):
        from dclab_rnd.draft import chat

        token = chat._MODEL_TURN.set(True)
        self.addCleanup(chat._MODEL_TURN.reset, token)
        turn, said = self.run_request("Give me only the bookings with a price above 500 and drop the nationality column.", "bookings",
                                      "preparation", client=object())
        self.assertEqual(turn.call_count, 0)
        self.assertIn("Say them one at a time", said)


if __name__ == "__main__":
    unittest.main()
