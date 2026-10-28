"""31.4: a family's scoreboard row, each number traced to the results and records it was computed from."""
import json
import tempfile
import unittest
from pathlib import Path


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")


class ScoreboardRowTests(unittest.TestCase):
    def test_the_row_takes_the_newest_zero_the_spend_and_the_lessons(self):
        from dclab_rnd.learning.scoreboard import family_row

        root = Path(tempfile.mkdtemp())
        home, results = root / "home", root / "results"
        for rid, split, passed, at in (("LRN-001", "check", 7, "2026-10-23T12:00:00+00:00"), ("LRN-005", "check", 9, "2026-10-29T03:00:00+00:00"),
                                       ("LRN-002", "learning", 3, "2026-10-23T12:01:00+00:00")):
            write(results / f"{rid}_groups_{split}.json", {"id": rid, "kind": "learning_baseline", "family": "groups", "split": split, "at": at,
                                                            "spent_eur": 0.0, "rows": [], "summary": {"passed": passed, "solvable": 20}})
        write(results / "LRN-003_groups_pilot.json", {"id": "LRN-003", "kind": "learning_pilot", "family": "groups", "spent_eur": 0.25})
        write(results / "LRN-004_other_check.json", {"id": "LRN-004", "kind": "learning_baseline", "family": "other", "split": "check", "spent_eur": 9.0})
        lessons = home / "learning" / "lessons.jsonl"
        lessons.parent.mkdir(parents=True)
        lessons.write_text("".join(json.dumps({"family": "groups", "gap_type": g}) + "\n" for g in ("rule", "rule", "decision")), encoding="utf-8")
        row = family_row("groups", home, results)
        self.assertEqual((row["zero"]["check"]["result"], row["zero"]["check"]["passed"]), ("LRN-005", 9))  # the newest zero
        self.assertEqual(row["zero"]["learning"]["result"], "LRN-002")
        self.assertIsNone(row["zero"]["sealed"])
        self.assertFalse(row["sealed_read"])
        self.assertEqual(row["spent_eur"], 0.25)  # another family's spend is not this one's
        self.assertEqual(row["lessons_by_gap"], {"rule": 2, "decision": 1})
        self.assertEqual(sorted(row["results"]), ["LRN-001", "LRN-002", "LRN-003", "LRN-005"])
        self.assertEqual(row["replay_problems"], [])


if __name__ == "__main__":
    unittest.main()
