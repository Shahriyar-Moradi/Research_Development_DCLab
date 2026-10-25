"""Step 5 (2026-10-25): a trained student of one point answering another whose truth follows from its answer (the split
student for groups and time order, the cleaning-step student for each step kept), on homework whose states production's
own capture wrote, saved like any student and loaded by the shadow only when its bytes match."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.ledger.mapped import MAPS, MappedAnswerer  # noqa: E402

SPLIT = {"rows": "small", "columns": 6, "date_columns": 0, "id_columns": 1, "repeat_share": 0.0}


def split_student():
    from dclab_core.answerers import ClassifierAnswerer

    rows = []
    for i in range(80):
        truth = ["stratified", "time", "group", "time_group"][i % 4]
        state = {**SPLIT, "has_time": truth in ("time", "time_group"), "has_groups": truth in ("group", "time_group"), "date_columns": i % 3}
        rows.append((state, truth))
    return ClassifierAnswerer("split.strategy", name="split.strategy.test", version="t1").fit(rows)


def step_student():
    from dclab_core.answerers import ClassifierAnswerer

    rows = [({"column": f"c{i}", "dtype": ["string", "float"][i % 2], "value_pattern": ["short_text", "digits"][i % 2]},
             ["trim", "numbers"][i % 2]) for i in range(60)]
    return ClassifierAnswerer("data.cleaning_step", name="data.cleaning_step.test", version="t1").fit(rows)


class MappedTest(unittest.TestCase):
    def test_the_split_students_answer_becomes_groups_and_time_order(self):
        base = split_student()
        both = {**SPLIT, "has_time": True, "has_groups": True, "task": "binary_classification", "strength": 0.4}  # the modifier's own fields too
        groups, time_order = MappedAnswerer(base, "modifier.groups"), MappedAnswerer(base, "modifier.time_order")
        self.assertEqual(base.answer("split.strategy", both, list(MAPS["modifier.groups"][1])).value, "time_group")
        self.assertIs(groups.answer("modifier.groups", both, ["yes", "no"]).value, True)
        self.assertIs(time_order.answer("modifier.time_order", {**both, "has_groups": False}, ["yes", "no"]).value, True)
        self.assertIs(groups.answer("modifier.groups", {**both, "has_groups": False}, ["yes", "no"]).value, False)
        self.assertIsNone(groups.answer("modifier.time_order", both, ["yes", "no"]))  # another point: no answer
        a = groups.answer("modifier.groups", both, ["yes", "no"])
        self.assertTrue(a.answerer.startswith("modifier.groups.from_split.strategy@t1+map1") and a.reason == "split.strategy: time_group")

    def test_a_step_is_kept_when_it_is_the_one_the_column_needs_numbers_after_markers_too(self):
        keep = MappedAnswerer(step_student(), "data.cleaning_keep")
        text = {"column": "c0", "dtype": "string", "uniqueness": "under_1pct", "value_pattern": "short_text"}
        self.assertIs(keep.answer("data.cleaning_keep", {**text, "cleaning": "trim"}, ["yes", "no"]).value, True)
        self.assertIs(keep.answer("data.cleaning_keep", {**text, "cleaning": "numbers"}, ["yes", "no"]).value, False)
        self.assertIsNone(keep.answer("data.cleaning_keep", text, ["yes", "no"]))  # no step named: nothing to judge
        from dclab_rnd.ledger import mapped

        self.assertTrue(mapped._keep("missing_markers", {"cleaning": "numbers"}))  # a column of numbers with "N/A" needs both

    def test_only_the_right_student_maps_and_the_map_survives_pickling(self):
        import pickle

        with self.assertRaises(ValueError):
            MappedAnswerer(step_student(), "modifier.groups")
        with self.assertRaises(ValueError):
            MappedAnswerer(split_student(), "column.semantic_role")
        m = pickle.loads(pickle.dumps(MappedAnswerer(split_student(), "modifier.groups")))
        self.assertIs(m.answer("modifier.groups", {**SPLIT, "has_time": False, "has_groups": True}, ["yes", "no"]).value, True)


    def test_without_its_own_fields_the_mapped_student_abstains(self):  # the review of step 5: missing fields read as zeros
        groups, keep = MappedAnswerer(split_student(), "modifier.groups"), MappedAnswerer(step_student(), "data.cleaning_keep")
        self.assertIsNone(groups.answer("modifier.groups", {"rows": "small", "has_time": True, "has_groups": True}, ["yes", "no"]))
        self.assertIsNone(keep.answer("data.cleaning_keep", {"rows": "small", "cleaning": "duplicates"}, ["yes", "no"]))  # a table-wide step
        self.assertIsNone(keep.answer("data.cleaning_keep", {"rows": "small", "cleaning": "trim"}, ["yes", "no"]))  # no column bands


class HomeworkTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_production_capture_writes_the_states_and_the_planted_truths_are_the_labels(self):
        from dclab_core import benchgen2 as g

        from dclab_rnd import decisions_gen as G

        seed = G.V3_FIRST + 1
        records, labels = G.decide_v3(seed)
        points = {r.point for r in records}
        self.assertEqual(points, set(G.V3_POINTS))
        self.assertTrue(all(r.id.startswith("g3") for r in records))
        truth = {l.record: l.value for l in labels}
        _, facts = g.split_table(seed)
        for r in records:
            if r.point == "modifier.groups":
                self.assertEqual(truth[r.id], facts["has_entity"])
                self.assertTrue({"columns", "date_columns", "id_columns", "repeat_share"} <= set(r.state))  # the split student's fields
            if r.point == "data.cleaning_keep":
                self.assertIn("cleaning", r.state)
                self.assertIs(r.value_used, True)  # the rule keeps every step
        self.assertEqual({r.dataset for r in records if r.point.startswith("modifier.")}, {f"split3-s{seed}"})  # named and hidden: one group

    def test_a_tables_column_count_never_says_its_truth(self):  # the review of step 5: v2 always drew three features
        from dclab_core import benchgen2 as g

        from dclab_rnd import decisions_gen as G

        widths = {True: set(), False: set()}
        for seed in range(G.V3_FIRST, G.V3_FIRST + 80):
            frame, facts = g.split_table(seed)
            plain = G.with_plain_columns(frame, seed)
            self.assertEqual(list(plain.columns[: frame.shape[1]]), list(frame.columns))  # the table's own columns kept as they were
            widths[facts["has_entity"]].add(plain.shape[1])
        self.assertGreaterEqual(len(widths[True] & widths[False]), 4)  # the same widths with and without an entity

    def test_an_entity_recurring_over_the_period_has_no_truth_for_groups(self):  # the product's rule asks the person there
        from dclab_core import benchgen2 as g

        from dclab_rnd import decisions_gen as G

        seed = next(s for s in range(G.V3_FIRST, G.V3_FIRST + 40) if s % 4 == 3)  # a row time and an entity
        _, facts = g.split_table(seed)
        self.assertTrue(facts["has_row_time"] and facts["has_entity"])
        records, _ = G.decide_v3(seed)
        self.assertNotIn("modifier.groups", {r.point for r in records})
        self.assertIn("modifier.time_order", {r.point for r in records})

    def test_a_mapped_student_is_saved_like_any_and_its_base_read_only_when_its_bytes_match(self):
        import joblib

        from dclab_rnd import decision_models_build as B
        from dclab_rnd.ledger import shadow

        bases, sets, out = (Path(tempfile.mkdtemp()) for _ in range(3))
        for name, student in (("split.strategy", split_student()), ("data.cleaning_step", step_student())):
            joblib.dump(student, bases / f"{name}.joblib")
            (bases / f"{name}.json").write_text(json.dumps({"name": student.name, "version": student.version, "family": "logistic", "strength": 1.0,
                                                            "features": 3, "trained_on": 80, "trained_set": "x", "groups": 40}))
        B.record_fitted(bases)
        for point in MAPS:
            (sets / point).mkdir()
            row = {"record": "r1", "split": "dev", "group": "g1", "state": {**SPLIT, "has_time": True, "has_groups": False, "cleaning": "trim",
                                                                         "column": "c0", "dtype": "string", "value_pattern": "short_text"},
                   "options": ["yes", "no"], "value_used": False if point.startswith("modifier") else True, "label": {"value": True, "source": "benchmark"}}
            (sets / point / "rows.jsonl").write_text(json.dumps(row) + "\n")
        cards = B.build_mapped(sets, out, bases)
        self.assertEqual(set(cards), set(MAPS))
        self.assertEqual(cards["modifier.time_order"].dev.accuracy, 1.0)
        self.assertIn("not trained on this point", cards["modifier.groups"].chosen_by)
        loaded = shadow.answerers(str(out))
        self.assertEqual({m.point for m in loaded}, set(MAPS))  # the shadow loads them like any student
        (bases / "split.strategy.joblib").write_bytes(b"swapped")
        with self.assertRaises(ValueError):
            B.build_mapped(sets, out, bases)  # a base whose bytes moved is never unpickled


if __name__ == "__main__":
    unittest.main()
