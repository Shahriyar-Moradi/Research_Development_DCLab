"""Package 16.15 C, long texts in the engine: a table of support notes with a text column, an author and a label runs
the embedded machine. The holdout is whole authors chosen from their names alone; near copies across the holdout line
are counted from the texts; the regularisation is chosen on folds grouped by author; the holdout's labels are read
once, in the final stage. Images run in the engine too since 26.8 (tests/test_images_engine.py): their checks are named
"images: …" and asserted here beside the texts'."""

import dataclasses
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_core import embeddings
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.studio.solution import Solution
    from tests.test_engine_machines import before_final, run_case
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FIELDS = {"source_column": "author", "item_kind": "text", "text_column": "text"}
MOMENT = "When a support note arrives: is it urgent?"


@unittest.skipIf(MISSING, MISSING)
class TextsInTheEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kind = kinds.get("embedded")
        cls.case = cls.kind.case(0)
        cls.store, cls.pid, cls.records = run_case("embedded", cls.case)

    def test_every_engine_decision_holds_and_only_the_pretrained_encoder_is_missing(self):
        result = acceptance.run(self.kind)
        self.assertEqual(result["missing"], ["a pretrained local encoder is measured on the replicas"], result)
        engine = [c for c in result["checks"] if c["decision"].startswith(("the engine", "in the engine"))]
        self.assertEqual(len(engine), 5)
        self.assertTrue(all(c["ok"] for c in engine), engine)
        images = [c for c in result["checks"] if c["decision"].startswith("images:")]  # 26.8
        self.assertEqual(len(images), 5)
        self.assertTrue(all(c["ok"] for c in images), images)

    def test_the_holdout_labels_reach_the_final_stage_only(self):
        held = set(self.records["final"]["evidence"]["holdout_row_positions"])
        table = self.case.table.copy()
        flip = table.index.isin(sorted(held))
        table.loc[flip, "urgent"] = 1 - table.loc[flip, "urgent"]
        _, _, other = run_case("embedded", dataclasses.replace(self.case, table=table))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"the {stage} stage read the holdout's labels")
        self.assertNotEqual(self.records["final"]["evidence"]["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"])
        self.assertEqual(self.records["final"]["evidence"]["holdout_uses_in_this_project"], 1)

    def test_the_holdout_rows_are_never_fitted(self):
        import numpy as np

        X, _ = embeddings.encode(self.case.table["text"].tolist(), embeddings.HashingText())
        sources = self.case.table["author"]
        parts = embeddings.split(sources)
        test = embeddings.holdout_mask(sources, parts)
        y = self.case.table["urgent"].to_numpy()
        flipped = np.where(test, 1 - y, y)  # the holdout's labels changed: a model that never fitted them predicts the same
        a, b = embeddings.score(X, y, sources, parts, "1.0"), embeddings.score(X, flipped, sources, parts, "1.0")
        self.assertEqual(a["holdout_predictions"], b["holdout_predictions"])
        self.assertNotEqual(a["holdout"], b["holdout"])

    def test_the_near_copy_threshold_is_calibrated_on_training_texts_only(self):
        held = set(self.records["final"]["evidence"]["holdout_row_positions"])
        table = self.case.table.copy()
        rows = table.index.isin(sorted(held))
        table.loc[rows, "text"] = table.loc[rows, "text"] + " Please call back."
        _, _, other = run_case("embedded", dataclasses.replace(self.case, table=table))
        self.assertEqual(self.records["leakage"]["evidence"]["calibration"], other["leakage"]["evidence"]["calibration"])
        self.assertEqual(self.records["models"]["evidence"]["ranking"], other["models"]["evidence"]["ranking"])

    def test_whole_authors_on_one_side_and_the_choice_on_folds_by_author(self):
        final, models = self.records["final"]["evidence"], self.records["models"]["evidence"]
        self.assertEqual(final["sources_on_both_sides"], 0)
        self.assertEqual(final["fitted_rows"], final["train_rows"])
        self.assertEqual(final["model"], models["selected_model"])
        self.assertEqual({r["model"] for r in models["ranking"]}, {str(c) for c in embeddings.CANDIDATE_C})
        self.assertNotIn("_held", self.records["data"]["evidence"]["split"])  # the authors' names never enter the record

    def test_the_final_stage_refuses_held_out_sources_that_changed(self):
        from dclab_rnd.studio.machine import prepare

        project = self.store.get(self.pid)
        machine = prepare(self.store, project, Solution(**project["solution"]), self.kind).machine
        moved = {"data": {"evidence": {"split": {**self.records["data"]["evidence"]["split"], "holdout_fingerprint": "0" * 16}}}}
        with self.assertRaises(ValueError):
            machine.final("E-1", {**self.records, **moved})

    def test_the_final_figure_is_the_scored_models_own(self):
        import copy

        store, pid, records = self.store, self.pid, self.records
        from dclab_rnd.studio.machine import prepare

        project = store.get(pid)
        machine = prepare(store, project, Solution(**project["solution"]), kinds.get("embedded")).machine
        moved = copy.deepcopy(records)
        moved["models"]["evidence"]["ranking"] = [{"model": "0.1", "mean": 0.5, "std": 0.0}, {"model": "1.0", "mean": 0.9, "std": 0.0}]
        moved["models"]["decision"]["chosen"] = "0.1"  # a person chose another C at approval
        evidence = machine.final("E-1", moved)[0]
        self.assertEqual(evidence["model"], "0.1")
        self.assertEqual(evidence["cv_selected_metric_mean"], 0.5)

    def test_exports_are_refused(self):
        from dclab_rnd.studio import export

        with self.assertRaises(export.ExportUnsupported):
            export.notebook(self.store.get(self.pid), self.records)


def machine_on(table, **solution):
    """The embedded machine the engine would build on ``table`` (no stage run)."""
    from dclab_rnd.studio.machine import MachinePrepared  # noqa: F401 — the engine's own wrapper is not needed here
    from dclab_rnd.kinds.embedded import EmbeddedMachine

    kind = kinds.get("embedded")
    sol = Solution(target="urgent", task="binary", prediction_moment=MOMENT, kind="embedded", kind_fields=FIELDS, **solution)
    return EmbeddedMachine(kind, sol, table, {"id": "p"})


@unittest.skipIf(MISSING, MISSING)
class TextMachineEdgesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = kinds.get("embedded").case(0).table

    def test_a_date_as_the_source_holds_out_the_same_sources(self):
        days = pd.Timestamp("2024-01-01") + pd.to_timedelta(self.table["author"].str[1:].astype(int), unit="D")
        machine = machine_on(self.table.assign(author=days))
        self.assertEqual(int(machine.test.sum()) > 0, True)
        self.assertEqual(len(set(machine.rows.loc[machine.test, "author"])), machine.parts["holdout_sources"])

    def test_the_positive_class_comes_from_the_training_rows_or_the_person(self):
        labels = self.table.assign(urgent=self.table["urgent"].map({0: "no", 1: "yes"}))
        machine = machine_on(labels)
        held = machine.index[machine.test]
        odd = labels.copy()
        odd.loc[held[0], "urgent"] = "maybe"  # a held-out value the training rows never had (it sorts first)
        other = machine_on(odd)
        self.assertEqual(machine.classes(), other.classes())
        self.assertTrue((machine.y()[~machine.test] == other.y()[~other.test]).all())
        with self.assertRaises(ValueError):
            other.final("E-1", {"data": {"evidence": {"split": other.lock()}}, "models": {"decision": {"chosen": "1.0"}}})
        self.assertEqual(machine_on(labels, positive_label="no").classes(), ("yes", "no"))

    def test_a_missing_label_never_moves_a_source(self):
        machine = machine_on(self.table)
        author = machine.rows.loc[machine.test, "author"].iloc[0]
        quiet = self.table.astype({"urgent": float})
        quiet.loc[quiet["author"] == author, "urgent"] = None  # every label of one held-out author missing
        other = machine_on(quiet)
        self.assertEqual(machine.parts["holdout_fingerprint"], other.parts["holdout_fingerprint"])
        self.assertEqual(other.dropped["without_label"], int((self.table["author"] == author).sum()))

    def test_positions_in_the_record_are_the_tables(self):
        gaps = self.table.copy()
        gaps.loc[[0, 1, 2], "text"] = None
        machine = machine_on(gaps)
        self.assertEqual(machine.dropped["without_text_or_source"], 3)
        self.assertNotIn(0, set(machine.index))
        self.assertEqual(set(machine.index[machine.test]) & {0, 1, 2}, set())
        self.assertTrue(all(gaps.loc[i, "author"] in set(machine.rows.loc[machine.test, "author"]) for i in machine.index[machine.test]))

    def test_one_class_folds_and_a_one_class_holdout_are_said_not_scored(self):
        import numpy as np

        X, _ = embeddings.encode(self.table["text"].tolist(), embeddings.HashingText())
        sources = self.table["author"]
        y = self.table["urgent"].to_numpy()
        rare = np.where(sources.isin(["A000", "A001"]), y, 0)  # positives on two authors only
        with self.assertRaises(ValueError):
            embeddings.choose(X, rare, sources)
        parts = embeddings.split(sources)
        test = embeddings.holdout_mask(sources, parts)
        calm = np.where(test, 0, y)  # the held-out authors wrote only routine notes
        out = embeddings.score(X, calm, sources, parts, "1.0")
        self.assertIsNone(out["holdout"])
        self.assertIn("one class", out["why_no_score"])


@unittest.skipIf(MISSING, MISSING)
class TextSolutionTest(unittest.TestCase):
    def test_the_engine_takes_texts_in_a_column_and_refuses_images(self):
        Solution(target="urgent", task="binary", prediction_moment=MOMENT, kind="embedded", kind_fields=FIELDS)
        for fields, extra, why in ((dict(FIELDS, item_kind="image"), {}, "images need local weights"),
                                   (dict(FIELDS, text_column=None), {}, "no text column"),
                                   (dict(FIELDS, text_column="author"), {}, "the text is the source"),
                                   (FIELDS, {"forbidden": [{"column": "text", "reason": "x"}]}, "the text forbidden"),
                                   (FIELDS, {"metric": "f1"}, "a metric the machine ignores")):
            with self.assertRaises(ValueError, msg=why):
                Solution(target="urgent", task="binary", prediction_moment=MOMENT, kind="embedded", kind_fields=fields, **extra)

    def test_the_held_out_sources_come_from_their_names_alone(self):
        sources = [f"A{i:03d}" for i in range(60) for _ in range(3)]
        once = embeddings.split(sources)
        again = embeddings.split(sources + sources[:30])  # more rows of the same authors
        self.assertEqual(once["_held"], again["_held"])
        self.assertEqual(once["holdout_sources"], 12)
        with self.assertRaises(ValueError):
            embeddings.split(["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
