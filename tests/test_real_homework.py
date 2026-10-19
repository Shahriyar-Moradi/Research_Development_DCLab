"""Package 29.15: real homework from public data. A real column's state is built exactly as the invented sets build
theirs (and as the lake's own state); each truth is the dataset's declaration (the row identifier, the declared type,
the outcome's type), an outcome whose declaration and data disagree left out; each table sits in the split the lake
gives it, the sealed rows in a file of their own with their fingerprint; scoring reads both sets' dev rows and never
the sealed files."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


def _table(seed: int = 0, binary: bool = True):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 240
    frame = pd.DataFrame({"row_no": np.arange(n), "age": rng.integers(18, 80, n), "plan": rng.choice(["basic", "plus", "max"], n),
                          "monthly_fee": rng.normal(40, 9, n).round(2), "region_note": rng.choice(["north", "south"], n)})
    frame["churned"] = rng.choice(["yes", "no"], n) if binary else rng.integers(0, 2, n)
    rec = {"did": seed, "name": f"t{seed}", "target": "churned", "row_id": "row_no", "ignore": "region_note", "rows": n,
           "columns": [{"name": "row_no", "openml_type": "numeric", "is_target": False, "is_row_identifier": True},
                       {"name": "age", "openml_type": "numeric", "is_target": False, "is_row_identifier": False},
                       {"name": "plan", "openml_type": "nominal", "is_target": False, "is_row_identifier": False},
                       {"name": "monthly_fee", "openml_type": "numeric", "is_target": False, "is_row_identifier": False},
                       {"name": "region_note", "openml_type": "nominal", "is_target": False, "is_row_identifier": False, "is_ignore": True},
                       {"name": "churned", "openml_type": "nominal" if binary else "numeric", "is_target": True, "is_row_identifier": False}]}
    return rec, frame


class CoreTests(unittest.TestCase):
    def test_a_declaration_is_a_truth_only_where_the_data_agrees(self):
        from dclab_core import declared as D

        self.assertEqual([D.task("numeric", 40), D.task("numeric", 2), D.task("nominal", 2), D.task("nominal", 5), D.task("nominal", 1)],
                         ["regression", None, "binary", "multiclass", None])
        self.assertIsNone(D.task("string", 30))
        self.assertEqual([D.role("nominal"), D.role("numeric", True), D.role("unknown")], ["categorical_code", "identifier", None])
        self.assertEqual([D.identifier(True), D.identifier(False)], [True, False])


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class DeclarationTests(unittest.TestCase):
    def test_a_real_columns_state_is_the_invented_sets_and_each_truth_is_the_declaration(self):
        from dclab_rnd import real_homework as R

        rec, frame = _table(0)
        got = R.derive(rec, frame)
        self.assertEqual(got["differ_from_lake"], 0)  # benchgen.decide's state is the lake's column_state
        self.assertNotIn("churned", got["columns"])  # the outcome is never a column decision
        self.assertEqual(set(got["task"]["state"]) >= {"rows", "columns", "classes", "integer_valued"}, True)
        records, labels, left = R.decisions_of(rec, got)
        truth = {(r.point, r.subject.key): x.value for r, x in zip(records, labels)}
        self.assertIs(truth[("column.is_identifier", "row_no")], True)
        self.assertIs(truth[("column.is_identifier", "age")], False)
        self.assertEqual(truth[("column.semantic_role", "plan")], "categorical_code")
        self.assertEqual(truth[("column.semantic_role", "row_no")], "identifier")
        self.assertEqual(truth[("task.framing", "openml:0")], "binary")
        self.assertNotIn(("column.is_identifier", "region_note"), truth)  # an ignored column: its role is not said
        self.assertEqual(left["an ignored column (its role is not said)"], 1)
        state = next(r.state for r in records if r.point == "column.is_identifier" and r.subject.key == "age")
        self.assertEqual(state["target"], "churned")
        self.assertTrue(all(x.source == "benchmark" and x.by == "public:openml_declared" for x in labels))

    def test_a_dataset_declaring_no_row_identifier_says_nothing_of_identifiers(self):
        from dclab_core import declared
        from dclab_rnd import real_homework as R

        rec, frame = _table(2)
        rec["columns"][0]["is_row_identifier"] = False  # row_no left as a feature, undeclared: as most OpenML keys are
        records, _, left = R.decisions_of(rec, R.derive(rec, frame))
        self.assertFalse([r for r in records if r.point == "column.is_identifier"])
        self.assertTrue([r for r in records if r.point == "column.semantic_role"])  # its declared types still say roles
        self.assertEqual(left[declared.NO_IDENTIFIER], 4)
        rec["columns"][-1]["openml_type"] = "string"  # an outcome whose type names no task: not a disagreement
        _, _, left = R.decisions_of(rec, R.derive(rec, frame))
        self.assertEqual((left[declared.NO_TASK], left[declared.DISAGREE]), (1, 0))

    def test_a_numeric_outcome_with_two_values_is_left_out(self):
        from dclab_rnd import real_homework as R

        rec, frame = _table(1, binary=False)  # declared numeric, two values in the data: neither is a truth
        records, _, left = R.decisions_of(rec, R.derive(rec, frame))
        self.assertFalse([r for r in records if r.point == "task.framing"])
        self.assertEqual(left["an outcome whose declaration and data disagree"], 1)


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class BuildAndScoreTests(unittest.TestCase):
    def _build(self, tmp: Path):
        from dclab_core import lake as L
        from dclab_rnd import real_homework as R

        by_split: dict[str, list[int]] = {}
        for did in range(1, 400):
            by_split.setdefault(L.split_of(f"openml:{did}"), []).append(did)
        dids = by_split["train"][:6] + by_split["dev"][:3] + by_split["sealed"][:2]
        recs, derived = [], tmp / "derived"
        derived.mkdir()
        for did in dids:
            rec, frame = _table(did)
            rec["did"] = did
            recs.append(rec)
            (derived / f"{did}.json").write_text(json.dumps(R.derive(rec, frame)))
        with mock.patch("dclab_rnd.lake.openml.records", side_effect=lambda: iter(recs)):
            summary = R.build(out=tmp / "real_v1", derived=derived)
        return summary, dids

    def test_each_table_sits_in_its_lake_split_and_the_sealed_rows_in_their_own_file(self):
        from dclab_core import lake as L
        from dclab_rnd.decision_gate_run import sealed_fingerprint

        tmp = Path(tempfile.mkdtemp())
        summary, dids = self._build(tmp)
        self.assertEqual(summary["tables"], len(dids))
        self.assertEqual(summary["states_differing_from_the_lake"], 0)
        for point in ("column.is_identifier", "column.semantic_role", "task.framing"):
            folder = tmp / "real_v1" / "decisions" / point
            rows = [json.loads(x) for x in (folder / "rows.jsonl").read_text().splitlines()]
            sealed = [json.loads(x) for x in (folder / "sealed.jsonl").read_text().splitlines()]
            self.assertTrue(sealed and all(r["split"] in ("train", "dev") for r in rows) and all(r["split"] == "sealed" for r in sealed))
            for r in rows + sealed:
                self.assertEqual(r["split"], L.split_of(r["group"].removeprefix("dataset:")))  # the lake's split, not the sets' own hash
            self.assertEqual(summary["sets"][point]["sealed_fingerprint"], sealed_fingerprint(sealed))
            self.assertEqual(summary["sets"][point]["splits"].get("sealed"), len(sealed))
            self.assertIn("sealed (15%)", json.loads((folder / "manifest.json").read_text())["policy"])  # the lake's split, said

    def test_scoring_reads_both_sets_dev_rows_and_never_a_sealed_file(self):
        from dclab_core.decision_models import train
        from dclab_rnd import real_homework as R

        tmp = Path(tempfile.mkdtemp())
        self._build(tmp)
        invented = tmp / "decisions" / "generated_v1" / "decisions" / "column.is_identifier"
        invented.mkdir(parents=True)
        real_rows = (tmp / "real_v1" / "decisions" / "column.is_identifier" / "rows.jsonl").read_text().splitlines()
        moved = [dict(json.loads(x), group=f"dataset:inv-{i % 5}", split="dev" if i % 5 == 0 else "train") for i, x in enumerate(real_rows)]
        (invented / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in moved))
        for f in (tmp / "real_v1").rglob("sealed.jsonl"):
            f.write_text("sealed: never opened\n")
        (invented / "sealed.jsonl").write_text("sealed: never opened\n")
        quick = lambda point, rows, dev, **kw: train(point, rows, dev, families=("logistic",), strengths=(1.0,), **kw)  # noqa: E731
        doc = R.score(["column.is_identifier", "task.framing"], real=tmp / "real_v1", decisions=tmp / "decisions", models=tmp / "models",
                      fit=quick, log=lambda *a: None, lake=tmp / "lake")  # an empty lake: never the real one
        entry = doc["points"]["column.is_identifier"]
        self.assertIsNone(entry["student"])  # no student folder here
        for name in ("invented", "real"):
            self.assertIsNotNone(entry[name]["rule"]["accuracy"])
            self.assertIsNotNone(entry[name]["with_real_homework"]["accuracy"])
            self.assertIsNotNone(entry[name]["baseline"]["commonest_accuracy"])  # every score beside the commonest label's
        card = json.loads((tmp / "models" / "column.is_identifier.json").read_text())
        self.assertTrue(any("real OpenML tables" in x for x in card["limits"]))  # the card says what it trained on
        self.assertTrue((tmp / "models" / "column.is_identifier.joblib").exists() and (tmp / "models" / "fitted.json").exists())
        self.assertEqual(doc["points"]["task.framing"], {"skipped": "no invented set"})


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class GlinerTests(unittest.TestCase):
    def test_a_captured_answer_counts_only_for_the_input_it_was_asked(self):
        from dclab_core import lake as L
        from dclab_rnd import real_homework as R
        from dclab_rnd.lake import answers

        lake = Path(tempfile.mkdtemp())
        rows = [{"group": "dataset:openml:5", "subject": {"key": k}, "state": {"column": k, "dtype": "integer", "target": "y"}, "options": ["yes", "no"]}
                for k in ("a", "b", "c")]
        path = answers.path_of(R.GLINER, "lake", "column.is_identifier", "dev", lake=lake)
        path.parent.mkdir(parents=True)
        lines = []
        for r, same in zip(rows[:2], (True, False)):
            asked = answers.fingerprint(*answers.render("column.is_identifier", r["state"] if same else {**r["state"], "dtype": "float"}, ["yes", "no"]))
            lines.append(json.dumps({"sample_id": f"openml5:{L.fingerprint(r['subject']['key'])[:12]}:id", "input_fingerprint": asked, "value": True}))
        path.write_text("".join(x + "\n" for x in lines))
        self.assertEqual(R.gliner("column.is_identifier", rows, lake=lake), [True, None, None])  # asked another state; never asked
        self.assertIsNone(R.gliner("task.framing", rows, lake=lake))


class RegisterTests(unittest.TestCase):
    def test_the_register_shows_real_scores_only_for_the_version_that_was_scored(self):
        from dclab_rnd import students

        tmp = Path(tempfile.mkdtemp())
        results = tmp / "real_homework_v1" / "results"
        results.mkdir(parents=True)
        entry = {"student": "task.framing.boosting@aaa", "with_real_homework": "task.framing.boosting@bbb",
                 "invented": {"rule": {"accuracy": 0.79}, "student": {"accuracy": 0.91}, "with_real_homework": {"accuracy": 0.90}},
                 "real": {"tables": 69, "baseline": {"commonest_accuracy": 0.48}, "rule": {"accuracy": 0.70},
                          "student": {"accuracy": 0.62, "recall": 0.5}, "with_real_homework": {"accuracy": 0.81}}}
        (results / "RHW-001_real_v1.json").write_text(json.dumps({"id": "RHW-001", "points": {"task.framing": entry,
                                                                                             "column.time": {"skipped": "no real set"}}}))
        with mock.patch.object(students, "CAMPAIGNS", tmp):
            got = students.real()
            self.assertEqual(set(got), {"task.framing"})
            self.assertEqual((got["task.framing"]["real_dev_accuracy"], got["task.framing"]["rule_real_dev_accuracy"]), (0.62, 0.70))
            self.assertEqual((got["task.framing"]["commonest_real_dev_accuracy"], got["task.framing"]["real_dev_recall"]), (0.48, 0.5))
            for version, shown in (("aaa", True), ("ccc", False)):
                card = {"task.framing": {"answerer": f"task.framing.boosting@{version}", "family": "boosting"}}
                with mock.patch.object(students, "students", return_value=card):
                    row = next(r for r in students.register()["items"] if r["point"] == "task.framing")
                self.assertEqual(row["real"] is not None, shown)  # another version's real scores are never shown as this one's


if __name__ == "__main__":
    unittest.main()
