"""Package 22.2: classical decision classifiers. Wilson intervals; accuracy counts an abstention wrong; a choice's macro
figures come with each class's; the candidate is chosen on folds grouped by table (a memorised table cannot score);
sealed rows are refused; the build reads rows.jsonl only and writes a card that validates."""

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import decision_models as dm
    from dclab_core.contracts import CONTRACTS
    from dclab_rnd import decision_models_build
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def row(group, value, split="train", **state):
    return {"record": f"r{random.random()}", "group": group, "split": split, "state": state,
            "label": {"value": value, "source": "benchmark"}, "value_used": value, "options": [True, False]}


def signal_rows(n_groups=20, per=10, seed=0, split="train"):
    """The label follows one band; every table also has a token of its own."""
    rng = random.Random(seed)
    out = []
    for g in range(n_groups):
        for _ in range(per):
            y = rng.random() < 0.5
            out.append(row(f"t{seed}-{g}", y, split, uniqueness="high" if y else "low", name_tokens=[f"tok{g}"]))
    return out


@unittest.skipIf(MISSING, MISSING)
class MetricsTest(unittest.TestCase):
    def test_wilson_matches_the_known_interval(self):
        lo, hi = dm.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        self.assertEqual(dm.wilson(0, 0), (None, None))

    def test_an_abstention_is_wrong_and_precision_is_of_the_positive_answer(self):
        s = dm.metrics([True, True, False, False], [True, None, True, False])
        self.assertEqual((s.accuracy, s.answered), (0.5, 3))
        self.assertEqual((s.precision, s.recall), (0.5, 0.5))
        self.assertEqual(len(s.precision_ci), 2)

    def test_a_choice_gives_each_class_with_its_interval_and_a_clustered_macro_interval(self):
        s = dm.metrics(["a", "a", "b", "c"], ["a", "b", "b", "c"], groups=["g1", "g1", "g2", "g3"])
        self.assertEqual(s.per_class["a"]["recall"], 0.5)
        self.assertEqual(s.per_class["b"]["precision"], 0.5)
        self.assertEqual(len(s.per_class["b"]["precision_ci"]), 2)
        self.assertAlmostEqual(s.recall, round((0.5 + 1 + 1) / 3, 4))
        self.assertEqual(len(s.recall_ci), 2)
        self.assertLessEqual(s.recall_ci[0], s.recall)
        self.assertIn("table group", s.intervals)

    def test_a_class_never_answered_or_never_true_is_not_dropped_from_the_macro(self):
        # as scikit-learn's macro with zero_division=0: every class true or answered counts
        self.assertAlmostEqual(dm.metrics(["a", "a", "b"], ["a", "a", "z"]).precision, round(1 / 3, 4))
        self.assertAlmostEqual(dm.metrics(["a", "a", "b"], ["a", "a", "a"]).precision, round((2 / 3) / 2, 4))

    def test_calibration_error_is_the_weighted_gap_between_confidence_and_accuracy(self):
        self.assertEqual(dm.calibration_error([0.95, 0.95], [True, True]), 0.05)
        self.assertEqual(dm.calibration_error([0.9, 0.9, 0.9, 0.9], [True, False, True, False]), 0.4)
        self.assertIsNone(dm.calibration_error([], []))


@unittest.skipIf(MISSING, MISSING)
class TrainTest(unittest.TestCase):
    def test_sealed_rows_are_refused(self):
        rows = signal_rows() + [row("s", True, "sealed", uniqueness="high")]
        with self.assertRaises(ValueError):
            dm.train("column.is_identifier", rows, folds=3, families=("logistic",), strengths=(1.0,))
        with self.assertRaises(ValueError):  # nor measured on here: 22.3 reads them once
            dm.train("column.is_identifier", signal_rows(), [row("s", True, "sealed", uniqueness="high")], folds=3,
                     families=("logistic",), strengths=(1.0,))

    def test_one_table_is_refused_and_a_tie_goes_to_the_smaller_strength(self):
        with self.assertRaises(ValueError):
            dm.train("column.is_identifier", [dict(r, group="one") for r in signal_rows()], folds=3, families=("logistic",))
        _, card = dm.train("column.is_identifier", signal_rows(), folds=4, families=("logistic",), strengths=(10.0, 1.0))
        self.assertEqual(card.oof_accuracy["logistic:1.0"], card.oof_accuracy["logistic:10.0"])  # the band settles it
        self.assertEqual(card.strength, 1.0)

    def test_a_fold_that_cannot_be_fitted_counts_its_rows_wrong(self):
        rows = [row(f"t{g}", True, uniqueness="high") for g in range(4) for _ in range(10)]
        rows += [row("lone", False, uniqueness="low") for _ in range(10)]  # its fold leaves one value to train on
        self.assertEqual(dm._oof("column.is_identifier", "logistic", 1.0, rows, (), 5), 0.8)  # skipping it would say 1.0

    def test_chosen_on_table_folds_so_a_memorised_table_does_not_score(self):
        rng = random.Random(1)  # each table's label is one coin: only its own token predicts it
        rows = []
        for g in range(30):
            y = rng.random() < 0.5
            rows += [row(f"g{g}", y, name_tokens=[f"tok{g}"]) for _ in range(8)]
        _, card = dm.train("column.is_identifier", rows, folds=5, families=("logistic",), strengths=(10.0,))
        self.assertLess(card.oof_accuracy["logistic:10.0"], 0.75)  # random folds would read each table's token: ~1.0

    def test_the_card_compares_with_the_rule_on_dev_and_validates(self):
        dev = signal_rows(8, 10, seed=5, split="dev")
        for r in dev[:20]:
            r["value_used"] = not r["label"]["value"]  # the rule is wrong on these
        model, card = dm.train("column.is_identifier", signal_rows(), dev, folds=4, families=("logistic", "boosting"),
                               strengths=(0.1, 1.0))
        self.assertEqual(card.dev.accuracy, 1.0)
        self.assertEqual(card.rule_dev.accuracy, 0.75)
        self.assertEqual(card.groups, 20)
        self.assertEqual(list(card.dev_by_source), ["benchmark"])
        self.assertIsNotNone(card.dev_calibration_error)
        self.assertIn("Platt", card.calibration)
        self.assertIn("CalibratedClassifierCV", type(model.model.steps[-1][1]).__name__)
        self.assertEqual(model.version, card.version)
        self.assertEqual(card.fields, ["name_tokens", "uniqueness"])
        _, again = dm.train("column.is_identifier", signal_rows(seed=3), folds=4, families=("logistic", "boosting"),
                            strengths=(0.1, 1.0))
        self.assertEqual(again.chosen_by, card.chosen_by)
        self.assertNotEqual(again.version, card.version)  # another training set, another model
        self.assertEqual(set(card.oof_accuracy), {"logistic:0.1", "logistic:1.0", "boosting:0.1", "boosting:1.0"})
        self.assertIs(CONTRACTS["decision_model_card"], dm.ModelCard)
        dm.ModelCard.model_validate_json(card.model_dump_json())
        self.assertEqual(model.answer("column.is_identifier", {"uniqueness": "high"}, [True, False]).value, True)


@unittest.skipIf(MISSING, MISSING)
class ImportTest(unittest.TestCase):
    def test_every_module_imports_first_in_a_fresh_interpreter(self):
        # contracts registers its cards by importing modules that import others: no first import may meet a cycle
        # (the ledger imported first once failed through contracts -> decision_models -> answerers -> ledger)
        from concurrent.futures import ThreadPoolExecutor

        modules = ["dclab_core"] + sorted(f"dclab_core.{p.stem}" for p in (ROOT / "dclab_core").glob("*.py") if p.stem != "__init__")
        modules += sorted(f"dclab_core.{p.parent.name}" for p in (ROOT / "dclab_core").glob("*/__init__.py"))

        def first(module):
            done = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT, capture_output=True, text=True)
            missing = "No module named" in done.stderr and "dclab_core" not in done.stderr.split("No module named")[-1]
            return module, done.returncode, done.stderr[-600:], missing  # an optional library absent here is not a cycle

        with ThreadPoolExecutor(8) as pool:
            for module, code, err, missing in pool.map(first, modules):
                if not missing:
                    self.assertEqual(code, 0, f"{module}: {err}")


@unittest.skipIf(MISSING, MISSING)
class BuildTest(unittest.TestCase):
    def test_the_build_refuses_measured_labels_without_their_fields_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            sets = Path(tmp) / "sets"
            (sets / "column.is_identifier").mkdir(parents=True)
            data = signal_rows()
            data[0]["label"]["source"] = "measured"
            (sets / "column.is_identifier" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            with self.assertRaises(ValueError):
                decision_models_build.build(quick=True, sets=sets, out=Path(tmp) / "out")

    def test_the_build_reads_rows_only_and_writes_a_card_and_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            sets, out = Path(tmp) / "sets", Path(tmp) / "out"
            (sets / "column.is_identifier").mkdir(parents=True)
            data = signal_rows() + signal_rows(6, 10, seed=9, split="dev")
            (sets / "column.is_identifier" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            (sets / "column.is_identifier" / "sealed.jsonl").write_text("not json: opening it would fail\n")
            cards = decision_models_build.build(quick=True, sets=sets, out=out)
            self.assertEqual(cards["column.is_identifier"].trained_on, 200)
            self.assertTrue((out / "column.is_identifier.joblib").exists())
            card = json.loads((out / "column.is_identifier.json").read_text())
            self.assertEqual(card["schema_version"], "dclab_core.decision_model_card.v1")
            self.assertEqual(card["dev"]["n"], 60)


if __name__ == "__main__":
    unittest.main()
