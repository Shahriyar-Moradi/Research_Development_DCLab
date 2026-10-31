"""The lead agent against Claude alone, person by person: wins, ties and losses on the same people, the paired difference
with its range, and which measures the judge marks (not yet trusted) and which code counts."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.conversation_exam import headtohead  # noqa: E402


def person(pid, arm, understanding, safety, seconds):
    return {"id": pid, "arm": arm, "understanding": understanding, "safety_failures": safety, "seconds_per_turn": seconds,
            "route_reached": True, "spent_eur": 0.1, "asked": 2, "useful_questions": 2, "already_answered": 0}


class HeadToHeadTests(unittest.TestCase):
    def test_wins_ties_losses_and_who_counts_each_measure(self):
        lead = [person(f"p{i}", "case", 1.0, 0, 30) for i in range(12)]
        plain = [person(f"p{i}", "plain", 0.5, 3, 15) for i in range(12)] + [person("only-plain", "plain", 0.0, 0, 1)]
        rows = {m["measure"]: m for m in headtohead.compare(lead, plain)}
        u = rows["understanding"]
        self.assertEqual((u["people"], u["lead_better"], u["lead_worse"], u["winner"], u["by"]), (12, 12, 0, "lead agent", "judge"))
        self.assertEqual((rows["invented numbers and broken rules"]["winner"], rows["invented numbers and broken rules"]["by"]), ("lead agent", "code"))
        self.assertEqual(rows["seconds per reply"]["winner"], "Claude alone")
        self.assertEqual(rows["right route reached"]["winner"], "no clear difference")

    def test_build_reads_only_runs_where_both_arms_met_the_same_people(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        (root / "CEX-009_dev.json").write_text(json.dumps({"scores": {"case": [person("p1", "case", 1.0, 0, 30)], "plain": [person("p1", "plain", 0.5, 1, 10)]}}))
        doc = headtohead.build(root)
        self.assertEqual([r["run"] for r in doc["runs"]], ["CEX-009"])
        self.assertIn("lead agent", headtohead.render(doc).lower())


if __name__ == "__main__":
    unittest.main()
