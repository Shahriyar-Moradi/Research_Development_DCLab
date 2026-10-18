"""Package 28.2: the board and the hand-off contracts: framework-free, a board that would hold a table's value refused,
every hand-off and every pause an experience event with the board's version before and after, the whole board kept at
every hand-off, a walk's record kept."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError as error:  # the agent environment has no pandas: only that skips
    MISSING = str(error)
else:
    MISSING = ""
if not MISSING:
    from dclab_core import orchestra as core
    from dclab_rnd import experience
    from dclab_rnd.orchestra import board as B


@unittest.skipIf(MISSING, MISSING)
class ContractTests(unittest.TestCase):
    def test_a_board_holds_names_codes_and_counts_and_reads_its_version(self):
        b = core.Board(thread="draft:d1", problem={"family": "rare_event", "outcome": "is_fraud"}, data={"tables": 2, "rows": 6000})
        self.assertEqual(core.Board.read(b.model_dump_json()).problem["outcome"], "is_fraud")
        with self.assertRaises(ValueError):
            core.Board(thread="draft:d1", problem={"action": "x" * 400})
        with self.assertRaises(ValueError):
            core.Board.read({**json.loads(b.model_dump_json()), "schema_version": "dclab_core.orchestra_board.v0"})

    def test_advance_raises_the_version_and_records_the_handoff_validated(self):
        b = core.Board(thread="draft:d1")
        after = core.advance(b, "analyst", "brief", "2026-10-18T00:00:00+00:00", to="confirm", problem={"family": "rare_event"})
        self.assertEqual((after.version, after.handoffs[-1].board_before, after.handoffs[-1].board_after), (1, 0, 1))
        self.assertEqual((after.handoffs[-1].from_node, after.handoffs[-1].to_node), ("analyst", "confirm"))
        with self.assertRaises(ValueError):  # validated again: a node never leaves a board its contract refuses
            core.advance(after, "analyst", "brief", "t", problem={"action": "x" * 400})

    def test_a_board_that_would_hold_a_tables_value_is_refused(self):
        frame = pd.DataFrame({"biker_id": ["BK-77812", "BK-10023"], "city": ["Lisbon", "Porto"], "is_fraud": [True, False]})
        values = B.distinct_values(frame)
        self.assertIn("BK-77812", values)
        self.assertNotIn("biker_id", values)  # a column's name is not a value
        b = core.Board(thread="draft:d1", problem={"outcome": "is_fraud", "unit": "BK-77812"})
        self.assertEqual(core.values_in(b, values), ["problem.unit"])
        with self.assertRaises(ValueError):
            B.advance(core.Board(thread="draft:d1"), "analyst", "brief", "understanding", frame=frame, problem={"unit": "Lisbon"})

    def test_every_text_field_is_read_inside_longer_text_and_the_vocabulary_is_not(self):  # the review of 28.2
        values = {"BK-77812", "Maria Silva", "none", "False"}
        b = core.Board(thread="draft:d1", problem={"unit": "each biker, e.g. BK-77812", "labels": "none", "ready": "False"},
                       open_questions=["Is BK-77812 a test account?"], blockers=["Maria Silva"],
                       parts=[core.Part(part="p1", workflow=["score Maria Silva"])])
        self.assertEqual(core.values_in(b, values), ["problem.unit", "open_questions[0]", "blockers[0]", "parts[0].workflow"])
        with self.assertRaises(ValueError):
            core.Board(thread="d", open_questions=["x" * 50_000])  # every list item is bounded

    def test_a_two_line_answer_is_one_line_on_the_board(self):
        b = core.Board(thread="d", problem={"moment": "When the order is placed,\nbefore a courier is assigned"})
        self.assertEqual(b.problem["moment"], "When the order is placed, before a courier is assigned")

    def test_a_pause_answered_raises_the_version_and_its_event_says_both(self):
        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "w")
        experience.install(log)
        self.addCleanup(experience.install, None)
        b = core.Board(thread="draft:d7")
        opened = B.interrupt(b, "question", "interviewer", "understanding", field="target", status="open", home=home)
        self.assertEqual(opened.version, 0)  # the walk waiting: an event, not a change
        answered = B.interrupt(b, "question", "interviewer", "understanding", field="target", status="answered", home=home)
        self.assertEqual((answered.version, len(answered.interrupts)), (1, 1))
        events = [e for e in log.read() if e.kind == "team.interrupt"]
        self.assertEqual([(e.payload["board_before"], e.payload["board_after"]) for e in events], [(0, 0), (0, 1)])
        self.assertEqual({e.ids.conversation for e in events}, {"d7"})  # the draft's: deleting it deletes them

    def test_deleting_a_draft_removes_its_team_records(self):
        from dclab_rnd.experience import retention

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "w")
        experience.install(log)
        self.addCleanup(experience.install, None)
        b = B.advance(core.Board(thread="draft:gone"), "analyst", "brief", "understanding", home=home)
        B.advance(core.Board(thread="draft:kept"), "analyst", "brief", "understanding", home=home)
        B.record("draft:gone", "understanding", [], b, "completed", "t", home=home)
        log.home = home
        out = retention.forget(draft="gone", log=log)
        self.assertGreaterEqual(out.get("orchestra", 0), 2)
        boards = [json.loads(x) for x in (home / "orchestra" / "boards.jsonl").read_text().splitlines()]
        self.assertEqual([x["board"]["thread"] for x in boards], ["draft:kept"])
        self.assertEqual([e for e in log.read() if e.kind == "team.handoff" and e.ids.conversation == "gone"], [])

    def test_parts_solved_in_parallel_merge_by_part(self):
        merged = B.merge_parts([{"part": "p1", "status": "open"}, {"part": "p2", "status": "open"}],
                               [{"part": "p2", "status": "routed", "route": "classifier"}])
        self.assertEqual(merged, [{"part": "p1", "status": "open"}, {"part": "p2", "status": "routed", "route": "classifier"}])


@unittest.skipIf(MISSING, MISSING)
class RoundTripTests(unittest.TestCase):
    def test_every_handoff_and_pause_is_an_event_and_the_whole_board_is_kept(self):
        from dclab_rnd.agent_eval import journeys
        from dclab_rnd.orchestra import checkpoints, spike

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        person = next(j for j in journeys.load("dev") if j["id"] == "JD-02")
        with checkpoints.open_checkpointer(home, database_url="") as saver:
            s = spike.Spike(saver, home=home)
            out = s.run("d1", person["words"])
            while out["status"] == "paused" and out["asks"]["kind"] == "question":
                out = s.run("d1", resume=person["answers"].get(out["asks"]["field"], "I don't know"))
            out = s.run("d1", resume=True)
        board = core.Board.model_validate(out["board"])
        events = log.read()
        handoffs = [e.payload for e in events if e.kind == "team.handoff"]
        answered = [i for i in board.interrupts if i.status != "open"]
        self.assertEqual(len(handoffs) + len(answered), board.version)  # every hand-off and every answered pause raised it once
        self.assertTrue(all(h["board_after"] == h["board_before"] + 1 for h in handoffs))
        pauses_raised = [e.payload for e in events if e.kind == "team.interrupt" and e.payload["status"] != "open"]
        steps = sorted([h["board_after"] for h in handoffs] + [p["board_after"] for p in pauses_raised])
        self.assertEqual(steps, list(range(1, board.version + 1)))  # one version per change, none skipped, none twice
        self.assertEqual(board.state, "map")  # confirmed: the walk moves on
        self.assertTrue(board.problem.get("family"))
        pauses = [e.payload for e in events if e.kind == "team.interrupt"]
        self.assertEqual({p["status"] for p in pauses}, {"open", "answered"})
        self.assertEqual(sum(p["interrupt"] == "confirmation" for p in pauses), 2)  # opened, then answered
        kept = [json.loads(x) for x in (home / "orchestra" / "boards.jsonl").read_text().splitlines()]
        self.assertEqual(len(kept), len(handoffs))  # every hand-off's whole board
        self.assertEqual(kept[-1]["board"]["version"], board.version)
        rec = B.record(board.thread, "understanding", [{"node": "interviewer", "steps": 3}, {"node": "analyst"}], board, "completed",
                       "2026-10-18T00:00:00+00:00", home=home)
        self.assertEqual((rec.handoffs, rec.board_version), (len(board.handoffs), board.version))
        self.assertEqual(core.TeamRecord.read((home / "orchestra" / "teams.jsonl").read_text().splitlines()[0]).team, "understanding")


if __name__ == "__main__":
    unittest.main()
