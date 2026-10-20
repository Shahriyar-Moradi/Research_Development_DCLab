"""Package 32.19: corrections and failures.

A person changing the agent's solution, a stage approved with another choice than the rule's, and a map edited are each
kept whole as before and after; each correction becomes a preference pair (the person's chosen over DCLab's rejected) and a
judged example; a move the validator refused becomes a failure, judged by the validator; nothing is recorded when nothing
changed.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class CorrectionsTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp()) / "agent_runs"
        self.home.mkdir()
        log = experience.open_log(self.home)
        experience.install(log)
        self.addCleanup(experience.install, None)

    def lines(self):
        return [json.loads(line) for p in sorted((self.home / "ledger" / "corrections").rglob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]

    def test_a_persons_changes_are_kept_whole_and_become_preferences_and_failures_judgments(self):
        from dclab_rnd import corrections
        from dclab_rnd.studio import memory

        project = {"id": "p1", "solution_saved_by": "agent"}
        before = {"target": "late", "task": "binary", "forbidden": [], "identifiers": []}
        after = {"target": "late", "task": "binary", "forbidden": [{"column": "fare_final"}], "identifiers": []}
        memory.solution_saved(project, before, after, "human")
        memory.stage_approved(project, "models", {"selected": "logistic", "rule": "best fold score"}, "boosting", "human")
        self.assertIsNone(corrections.record("card", {"a": 1}, {"a": 1}))  # nothing changed: nothing kept
        got = self.lines()
        self.assertEqual([c["what"] for c in got], ["solution", "stage_choice:models"])
        self.assertEqual((got[0]["before"], got[0]["after"], got[0]["source"]), (before, after, "person"))
        self.assertEqual((got[1]["before"], got[1]["after"]), ("logistic", "boosting"))

        from dclab_rnd.store import Layout, Store, shapers
        from dclab_rnd.store.sets import others

        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(layout)
        path = next((self.home / "ledger" / "corrections").rglob("*.jsonl"))
        store.keep_log("decisions", "dev", path.relative_to(self.home).as_posix(), path, consent="train")
        move = {"at": "2026-10-20T10:00:00+00:00", "actor": "agent", "move": "read_holdout", "from": "WF-02", "to": "WF-02", "status": "blocked", "project": "p1"}
        store.keep_log("workflows", "dev", "projects/p1/transitions.jsonl", (json.dumps(move) + "\n").encode(), consent="train")
        shapers.shape(layout)
        prefs = others.preferences(layout)
        self.assertEqual(len(prefs), 2)
        solution = next(p for p in prefs if p.task == "correction:solution")
        self.assertIn("fare_final", solution.chosen.content)
        self.assertNotIn("fare_final", solution.rejected.content)
        self.assertEqual(solution.decided_by, "person")
        refused = next(j for j in others.judged(layout) if j.judged == "step")
        self.assertEqual((refused.score, refused.judge, refused.evidence), (0.0, "validator", "verdict"))


if __name__ == "__main__":
    unittest.main()
