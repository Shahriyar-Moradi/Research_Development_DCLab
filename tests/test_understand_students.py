"""The understanding students with more homework (S8 B2, 2026-10-25): a person's split is the store's (its sealed share
dev, never trained on); the text student reads a person's words and is measured on dev people beside the rule on the
same rows; the rule's accuracy on the new people is reported apart from the old (the plan's 'too easy' check)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import understand_students as U  # noqa: E402


def row(group, split, label, rule, source="homework_v1"):
    return {"record": f"{group}:s0:0", "point": "problem.family", "state": {}, "options": [], "group": group, "source": source,
            "split": split, "value_used": rule, "label": {"value": label, "accepted": [label]}}


class UnderstandTest(unittest.TestCase):
    def test_a_persons_split_is_the_stores_and_sealed_people_are_never_trained_on(self):
        from dclab_core.datasets import split_of

        groups = [f"homework_v1:JH-binary-0{i:02d}" for i in range(200)]
        store = {g: split_of(f"dataset:journey:{g}") for g in groups}
        mine = {g: U.person_split(g) for g in groups}
        self.assertTrue(any(s == "sealed" for s in store.values()))
        self.assertTrue(all(mine[g] == ("dev" if store[g] == "sealed" else store[g]) for g in groups))
        self.assertNotIn("sealed", set(mine.values()))

    def test_the_text_student_reads_words_and_is_measured_beside_the_rule(self):
        words = {"classification": "who will cancel churn leave renew subscription quit",
                 "regression": "how many minutes price amount estimate number how much",
                 "forecasting": "next week sales per store forecast units future demand"}
        people, rows = [], []
        for fam, w in words.items():
            for i in range(30):
                gid = f"{fam[:4]}{i}"
                people.append(("homework_v1", {"id": gid, "words": f"{w.split()[i % 7]} {w.split()[(i + 3) % 7]} please", "answers": {}}))
                rows.append(row(f"homework_v1:{gid}", "train" if i < 20 else "dev", fam, "classification"))
        with mock.patch.object(U, "items", return_value=people):
            _, m = U.text_student(rows)
        self.assertEqual((m["train_people"], m["dev_rows"]), (60, 30))
        self.assertGreater(m["student_dev_accuracy"], m["rule_dev_accuracy"])  # the rule says one family for all here
        self.assertAlmostEqual(m["rule_dev_accuracy"], round(10 / 30, 4))

    def test_the_sealed_people_are_read_by_no_answerer_before_their_gate(self):
        from dclab_rnd.lake import answers

        with mock.patch.object(U, "spent", return_value=False):
            sets = {r.get("set") for r in answers._journeys(Path("."))}
        self.assertNotIn("understand_sealed", sets)
        homework = [r for r in answers._journeys(Path(".")) if r.get("set") == "homework_v1"]
        self.assertTrue(all(r["split"] == U.person_split(f"homework_v1:{r['group'].split(':')[-1]}") for r in homework))

    def test_the_rule_on_the_new_people_is_reported_apart(self):
        sets = Path(tempfile.mkdtemp())
        folder = sets / "decisions" / "problem.family"
        folder.mkdir(parents=True)
        rows = [row("homework_v1:a", "train", "regression", "classification"), row("homework_v1:b", "dev", "regression", "regression"),
                row("dev:JD-01", "train", "rare_event", "rare_event", source="dev")]
        (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        got = U.rule_on_new_people(sets)["problem.family"]
        self.assertEqual((got["new_rows"], got["rule_on_new"], got["old_rows"], got["rule_on_old"]), (2, 0.5, 1, 1.0))


if __name__ == "__main__":
    unittest.main()
