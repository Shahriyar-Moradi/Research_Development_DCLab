"""Package 29.12: harder homework where a student fails. A round's new tables are of the failing kinds, only in groups the
datasets' hash puts in train, built as the parent's decisions were; the parent's dev rows and sealed file never change; a
dev row's kind is read from its table's seed."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


class GeneratorTests(unittest.TestCase):
    def test_a_kind_asked_for_is_made_and_the_default_is_the_seeds(self):
        from dclab_core import benchgen2 as g

        self.assertEqual(g.task_table(7)[2]["kind"], g.TASK_KINDS[7 % len(g.TASK_KINDS)])  # the sets as they were built
        self.assertEqual(g.task_table(7, kind="regression_small_counts")[2]["task"], "regression")
        with self.assertRaises(ValueError):
            g.task_table(7, kind="no_such_kind")


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class RoundTests(unittest.TestCase):
    def test_new_tables_teach_only_and_a_dev_rows_kind_is_its_seeds(self):
        from dclab_core import benchgen2 as g
        from dclab_core.datasets import split_of
        from dclab_rnd import curriculum as C

        rows = C.new_rows(["multiclass_codes", "regression_small_counts"], 1, 3)
        self.assertEqual(len(rows), 12)  # 3 tables of each kind, each with its hidden-name twin
        self.assertEqual({r["kind"] for r in rows}, {"multiclass_codes", "regression_small_counts"})
        self.assertTrue(all(r["split"] == "train" and split_of(r["group"]) == "train" for r in rows))  # never a dev or sealed group
        self.assertEqual({r["label"]["value"] for r in rows}, {"multiclass", "regression"})
        self.assertTrue(all(r["record"].startswith("g2") and r["options"] == ["binary", "multiclass", "regression"] for r in rows))
        self.assertEqual(C.kind_of({"group": "dataset:task-s13"}), g.TASK_KINDS[13 % len(g.TASK_KINDS)])
        self.assertIsNone(C.kind_of({"group": "dataset:nothing"}))


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class RunTests(unittest.TestCase):
    """A whole run on a small parent: one register line per round, an exam on none of the rounds' students, the last round
    published, and the verdict's parts (a changed sealed file is never untouched)."""

    def test_a_run_records_each_round_and_never_gives_a_round_an_exam_it_did_not_sit(self):
        import json
        import tempfile
        from unittest import mock

        from dclab_core.decision_models import train
        from dclab_rnd import curriculum as C

        tmp = Path(tempfile.mkdtemp())
        parent, home = tmp / "generated_v2", tmp / "home"
        rows = C.new_rows(["binary_bool", "multiclass_codes", "regression_small_counts", "multiclass_names", "regression_float"], 9, 6)
        for i, r in enumerate(rows):
            r.pop("kind")  # a parent row carries no kind: it is read from its group's seed
            r["split"] = "dev" if i % 4 == 0 else "train"
        folder = parent / "decisions" / C.POINT
        folder.mkdir(parents=True)
        (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (folder / "sealed.jsonl").write_text("sealed: never opened\n")
        quick = lambda train_rows, dev: train(C.POINT, train_rows, dev, families=("logistic",), strengths=(1.0,))  # noqa: E731
        with mock.patch.object(C, "PARENT", parent), mock.patch.object(C, "OUT", tmp / "out"), mock.patch.object(C, "MODELS", tmp / "models"), \
                mock.patch.object(C, "fit", side_effect=quick), mock.patch.object(C, "kind_of", side_effect=lambda r: r.get("kind") or "multiclass_codes"):
            doc = C.run(rounds=2, per_kind=2, home=home, log=lambda *a: None)
        rounds = [h["round"] for h in doc["history"]]
        self.assertEqual(rounds[0], 0)
        self.assertTrue(doc["fingerprints"]["unchanged"])
        lines = [json.loads(x) for x in (home / "students" / "register.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), len(rounds))  # one line per round, in the register's history
        self.assertTrue(all(x["points"][C.POINT]["exam"] is None for x in lines[1:]))  # a round never sat the point's exam
        if len(rounds) > 1:
            self.assertEqual((tmp / "models" / "latest" / "ROUND").read_text().strip(), str(rounds[-1]))
        before = C.fingerprints
        calls = iter([{"rows": "a", "sealed": "b"}, {"rows": "a", "sealed": "changed"}])
        with mock.patch.object(C, "PARENT", parent), mock.patch.object(C, "OUT", tmp / "out2"), mock.patch.object(C, "MODELS", tmp / "models2"), \
                mock.patch.object(C, "fit", side_effect=quick), mock.patch.object(C, "fingerprints", side_effect=lambda *a: next(calls)):
            again = C.run(rounds=0, per_kind=2, home=None, log=lambda *a: None, persist=False)
        self.assertFalse(again["verdict"]["untouched"])  # a sealed file that changed is never untouched
        self.assertIs(C.fingerprints, before)


class HistoryTests(unittest.TestCase):
    def test_a_return_to_the_points_earlier_student_is_no_slip_but_a_worse_new_one_is(self):
        import json
        import tempfile

        from dclab_rnd import students

        home = Path(tempfile.mkdtemp())
        (home / "students").mkdir()
        lines = [("task@a", 0.9156), ("task@b", 0.9362), ("task@a", 0.9156), ("task@c", 0.89)]  # a round not adopted, then a worse new one
        (home / "students" / "register.jsonl").write_text("".join(json.dumps({"at": f"2026-10-19T0{i}:00:00+00:00", "points": {"task.framing":
                                                          {"student": who, "dev_accuracy": d}}}) + "\n" for i, (who, d) in enumerate(lines)))
        got = students.history(home)["task.framing"]
        self.assertEqual([h["slipped"] for h in got], [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
