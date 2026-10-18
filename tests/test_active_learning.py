"""Package 29.11: students ask for the labels they need most. The order a label teaches in (a disagreement with the rule
first, the least sure answer first, the point needing most labels first); the experiment that measures it against
labels drawn at random; and the rule that labels chosen to teach train a student but never estimate its accuracy."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import sklearn  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


class OrderTests(unittest.TestCase):
    def test_a_disagreement_first_then_the_least_sure_then_the_point_that_needs_more(self):
        from dclab_core.active import priority, select

        pool = [{"point": "a", "student": True, "rule": True, "confidence": 0.55},   # agrees, unsure
                {"point": "a", "student": True, "rule": False, "confidence": 0.95},  # disagrees, sure
                {"point": "a", "student": False, "rule": True, "confidence": 0.60},  # disagrees, unsure
                {"point": "a", "student": True, "rule": True, "confidence": 0.99}]   # agrees, sure
        self.assertEqual(select(pool, 4), [2, 1, 0, 3])
        self.assertEqual(select(pool, 2), [2, 1])
        self.assertLess(priority(True, False, 0.9, need=20), priority(True, False, 0.9, need=5))  # a point further from acting first
        self.assertEqual(priority(None, True, None)[0], 1)  # no answer from the student: not a disagreement

    def test_a_label_chosen_to_teach_never_estimates(self):
        from types import SimpleNamespace

        from dclab_core.active import split_labels

        drawn = SimpleNamespace(source="blind", weight=1.67)
        taught = SimpleNamespace(source="blind", weight=None)
        bench = SimpleNamespace(source="benchmark", weight=None)
        self.assertEqual(split_labels([drawn, taught, bench]), ([drawn, bench], [taught]))


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class TeachingOrderTests(unittest.TestCase):
    """The queue's teaching order: any decision a point's student answered, a disagreement first and the least sure first;
    its labels carry no weight, train a student and never count toward its acting or its demotion."""

    POINT, NAME = "feature.leakage_suspect", "student@v1"

    def setUp(self):
        from dclab_rnd import ledger

        self.book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        self.records = []
        for i in range(40):  # half disagree with the rule; the student's confidence rises with i
            rule, student = False, i % 2 == 0
            self.records.append(ledger.decide(self.POINT, ("column", f"c{i}"), answers=[ledger.answer("rule", "rule:leak", rule),
                                              ledger.answer("classifier", self.NAME, student, source="shadow", confidence=round(0.5 + i / 100, 3))],
                                              value_used=rule, state={"column": f"c{i}", "name_tokens": ["c"], "dtype": "float"}, options=["yes", "no"],
                                              ledger=self.book, project="p1"))

    def test_the_teaching_order_alternates_the_least_sure_and_a_random_one_never_from_the_draw(self):
        from dclab_core import ledger as contract
        from dclab_rnd.ledger import labels

        got = labels.queue(self.book, self.POINT, limit=100, current={self.POINT: self.NAME}, order="teach")
        by_id = {r.id: r for r in self.records}
        free = sorted(r.answers[1].confidence for r in self.records if not contract.blind_draw(r, "blind-v1"))
        sure = [by_id[i["record"]].answers[1].confidence for i in got["items"]]
        self.assertTrue(got["items"])
        self.assertFalse(any(contract.blind_draw(by_id[i["record"]], "blind-v1") for i in got["items"]))  # the draw is the fair queue's
        half = len(sure) // 2
        self.assertEqual(sorted(sure[0::2])[:half], free[:half][: len(sure[0::2])])  # every other one: the least sure outside the draw
        again = labels.queue(self.book, self.POINT, limit=100, current={self.POINT: self.NAME}, order="teach")
        self.assertEqual([i["record"] for i in again["items"]], [i["record"] for i in got["items"]])  # the seed's: the same each time
        self.assertTrue(all(i["taught"] and i["weight"] is None and "answers" not in i for i in got["items"]))  # still blind

    def test_a_taught_label_has_no_weight_trains_and_never_counts_toward_acting_or_demotion(self):
        from dclab_core.active import TEACH_NOTE
        from dclab_rnd.ledger import labels, shadow

        class Student:
            by, name, version, points = "classifier", "student", "v1", frozenset({"feature.leakage_suspect"})

        got = labels.queue(self.book, self.POINT, limit=100, current={self.POINT: self.NAME}, order="teach")
        for item in got["items"]:
            labels.blind_label(item["record"], False, "reviewer:u1", ledger=self.book, taught=True)
        written = self.book.labels()
        self.assertTrue(all(x.weight is None and x.weight_note == TEACH_NOTE for x in written))
        row = shadow.measure(self.book.records(), written, [Student()])[self.NAME]
        disagreeing = sum(1 for i in got["items"] if next(r for r in self.records if r.id == i["record"]).answers[1].value is True)
        self.assertEqual((row["labelled_disagreements"], row["taught_disagreements"]), (0, disagreeing))  # trains, never estimates
        drawn = next(r for r in self.records if __import__("dclab_core.ledger", fromlist=["blind_draw"]).blind_draw(r, "blind-v1"))
        with self.assertRaises(ValueError):  # a drawn decision is labelled from the fair sample, with its weight
            labels.blind_label(drawn.id, False, "reviewer:u1", ledger=self.book, taught=True)

    def test_a_taught_label_never_displaces_a_truth_an_estimate_reads(self):
        from dclab_rnd import ledger
        from dclab_rnd.ledger import shadow

        class Student:
            by, name, version, points = "classifier", "student", "v1", frozenset({"feature.leakage_suspect"})

        r = self.records[0]  # a disagreement (the student says yes, the rule no)
        ledger.label(r, True, "benchmark", "replica:test", ledger=self.book)
        ledger.label(r, False, "blind", "reviewer:u2", weight_note="chosen to teach (test)", ledger=self.book)
        row = shadow.measure([r], self.book.labels(), [Student()])[self.NAME]
        self.assertEqual((row["labelled_disagreements"], row["right_on_disagreements"], row["taught_disagreements"]), (1, 1, 0))  # the benchmark counts


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class ExperimentTests(unittest.TestCase):
    def test_both_arms_share_their_start_and_their_budget_and_the_sealed_rows_are_never_read(self):
        from dclab_rnd import active_learning as A

        sets = Path(tempfile.mkdtemp())
        folder = sets / "feature.leakage_suspect"
        folder.mkdir()
        rng = random.Random(0)
        rows = []
        for i in range(160):
            leak = i % 3 == 0
            rows.append({"record": f"r{i:04d}", "point": "feature.leakage_suspect", "group": f"dataset:t{i % 20}", "split": "dev" if i % 5 == 0 else "train",
                         "state": {"dtype": "float", "uniqueness": rng.choice(["all", "under_1pct"]), "name_tokens": ["refund" if leak else "amount"]},
                         "options": ["yes", "no"], "answers": [{"by": "rule", "value": False}], "value_used": False,
                         "label": {"value": leak, "source": "benchmark"}})
        (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (folder / "sealed.jsonl").write_text("not json: reading this file would fail\n")
        small = {**A.GATE, "points": ["feature.leakage_suspect"], "seeds": [0], "budget": 40, "start": 10, "batch": 10}
        with mock.patch.object(A, "GATE", small):
            train, _ = A.load("feature.leakage_suspect", sets)
            ordered = A._order(train, "feature.leakage_suspect", 0)
            asked = A.asked_for("feature.leakage_suspect", ordered)
            doc = A.run(sets=sets, log=lambda *a: None)
        self.assertEqual(len(asked), 40)
        self.assertEqual([r["record"] for r in asked[:10]], [r["record"] for r in ordered[:10]])  # the same start as the random arm
        got = doc["points"]["feature.leakage_suspect"]
        self.assertEqual(len(got["seeds"]), 1)
        self.assertIn("pass", doc["verdict"])
        self.assertEqual(doc["gate"]["version"], "active_learning_gate_v1")


if __name__ == "__main__":
    unittest.main()
