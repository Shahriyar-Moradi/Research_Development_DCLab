"""Package 16.15 B: the library kinds in the engine. Uplift (then ranking and segmentation) runs as its own machine
through the five stages: the holdout locked in the data stage from what is not an outcome, the choice made on
validation, the holdout read once in the final stage; changing only the holdout's outcomes leaves every stage before
the final as it was; the library's own numbers do not move."""

import dataclasses
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_rnd import kinds
    from dclab_rnd.studio import engine
    from tests.test_engine_machines import before_final, run_case
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


class SharedReaders:
    """The exports refuse, and every shared reader reads the machine's records; the holdout answer is the holdout's own."""

    key = ""
    holdout_metric = ""

    def test_exports_refuse_and_the_shared_readers_work(self):
        from dclab_core.report import to_markdown
        from dclab_rnd.intern.tools import compact_record
        from dclab_rnd.studio import agent, code, export
        from dclab_rnd.studio.solution_report import build as report_build

        project, records = self.store.get(self.pid), self.store.records(self.pid)
        with self.assertRaises(export.ExportUnsupported):
            export.notebook(project, records)
        with self.assertRaises(code.CodeRefused):
            code.export(self.store, self.pid, Path(tempfile.mkdtemp()))
        for record in records.values():
            self.assertIsNotNone(compact_record(record)["numbers"])
        answer = agent.answer("how good is the holdout score?", project, records, self.key)["answer"]
        self.assertIn(self.holdout_metric, answer)  # the holdout's own number, by its own name
        self.assertIn(self.holdout_metric, json.dumps(records["final"]["evidence"]["holdout_metrics"]))
        self.assertTrue(to_markdown(report_build(self.store, self.pid, analyse=False)))


@unittest.skipIf(MISSING, MISSING)
class UpliftInTheEngineTest(SharedReaders, unittest.TestCase):
    key, holdout_metric = "uplift", "qini_coefficient"
    @classmethod
    def setUpClass(cls):
        cls.case = kinds.get("uplift").case(0)
        cls.store, cls.pid, cls.records = run_case("uplift", cls.case)

    def test_its_decisions_hold_on_the_engines_records(self):
        from dclab_rnd.kinds.acceptance import Run

        project = self.store.get(self.pid)
        checks = kinds.get("uplift").accept_engine(Run(self.case, project["solution"], self.records))
        self.assertEqual([c["decision"] for c in checks if not c["ok"]], [], checks)

    def test_the_holdout_is_locked_and_a_moved_one_is_refused(self):
        prepared = engine.prepare(self.store, self.store.get(self.pid))
        previous = {s: self.records[s] for s in engine.STAGE_KEYS if s != "final"}
        moved = {**previous["data"], "evidence": {**previous["data"]["evidence"], "split": {**previous["data"]["evidence"]["split"], "holdout_fingerprint": "0" * 16}}}
        with self.assertRaises(ValueError) as caught:
            prepared.machine.final("x", {**previous, "data": moved})
        self.assertIn("rerun the data stage", str(caught.exception))

    def test_the_holdouts_outcomes_reach_only_the_final(self):
        from dclab_core import uplift

        f = self.case.solution["kind_fields"]
        held = uplift.split(self.case.table, treatment=f["treatment_column"])["holdout"]
        table = self.case.table.copy()
        table.loc[held, self.case.target] = 1 - table.loc[held, self.case.target]
        _, _, other = run_case("uplift", dataclasses.replace(self.case, table=table))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"uplift: the {stage} stage read the holdout's outcomes")
        self.assertNotEqual(self.records["final"]["evidence"]["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"])

    def test_the_interval_is_95_and_the_overlap_reads_only_kept_columns(self):
        final, data = self.records["final"]["evidence"], self.records["data"]["evidence"]
        ci = final["holdout_primary_metric_ci"]
        self.assertEqual([ci["low"], ci["high"]], final["models"][final["model"]]["qini_coefficient_ci95"])
        table = self.case.table.copy()
        table["redeemed_offer"] = (table[self.case.solution["kind_fields"]["treatment_column"]] == 1).astype(float) * 3.0  # after the offer
        altered = dataclasses.replace(self.case, table=table, solution={**self.case.solution,
                                                                         "forbidden": [{"column": "redeemed_offer", "reason": "after the offer"}]})
        _, _, other = run_case("uplift", altered)
        self.assertNotIn("redeemed_offer", other["leakage"]["evidence"]["features"])
        self.assertTrue(other["data"]["evidence"]["wording"]["causal"])  # the trial stays randomised: the forbidden column is not read
        self.assertEqual(other["data"]["evidence"]["overlap"], data["overlap"])

    def test_a_solution_states_the_machines_metric(self):
        from dclab_rnd.studio.solution import Solution

        good = {"target": "bought", "task": "binary", "kind": "uplift", "prediction_moment": "When the campaign list is drawn",
                "kind_fields": {"treatment_column": "got_offer", "randomised": True}}
        Solution(**good)
        with self.assertRaises(ValueError):
            Solution(**{**good, "metric": "roc_auc"})
        with self.assertRaises(ValueError):
            Solution(**{**good, "kind_fields": {"treatment_column": "bought", "randomised": True}})



@unittest.skipIf(MISSING, MISSING)
class RankingInTheEngineTest(SharedReaders, unittest.TestCase):
    key, holdout_metric = "ranking", "ndcg_at_k"
    @classmethod
    def setUpClass(cls):
        cls.case = kinds.get("ranking").case(0)
        cls.store, cls.pid, cls.records = run_case("ranking", cls.case)

    def test_its_decisions_hold_on_the_engines_records(self):
        from dclab_rnd.kinds.acceptance import Run

        checks = kinds.get("ranking").accept_engine(Run(self.case, self.store.get(self.pid)["solution"], self.records))
        self.assertEqual([c["decision"] for c in checks if not c["ok"]], [], checks)

    def test_a_moved_holdout_is_refused(self):
        prepared = engine.prepare(self.store, self.store.get(self.pid))
        previous = {s: self.records[s] for s in engine.STAGE_KEYS if s != "final"}
        moved = {**previous["data"], "evidence": {**previous["data"]["evidence"], "split": {**previous["data"]["evidence"]["split"], "fingerprint": "0" * 16}}}
        with self.assertRaises(ValueError) as caught:
            prepared.machine.final("x", {**previous, "data": moved})
        self.assertIn("rerun the data stage", str(caught.exception))

    def test_its_outcome_said_in_words_is_one(self):
        from dclab_core.card import problem_card
        from dclab_core.understand import understand

        words = "Rank the items each customer sees so the ones they will buy come first"
        brief = understand(words, {"interactions.csv": ["user_id", "item_id", "time", "event"]})
        brief = brief.model_copy(update={"target_definition": brief.target_definition.model_copy(update={"value": "a purchase in the next two weeks", "status": "stated"})})
        card = problem_card(brief, words=words, data={"columns": True}, family="ranking")
        self.assertEqual(card.outcome, "a purchase in the next two weeks")  # an event in the window, never a column of the log

    def test_the_windows_purchases_reach_only_the_final(self):
        f = self.case.solution["kind_fields"]
        table = self.case.table.copy()
        after = pd.to_datetime(table[f["time_column"]]) > pd.Timestamp(f["moment"])
        bought = table.index[after & (table[f["event_column"]] == f["relevant_event"])]
        table.loc[bought[::2], f["event_column"]] = "view"  # half the window's purchases were only views
        self.assertTrue(len(bought) > 10)
        _, _, other = run_case("ranking", dataclasses.replace(self.case, table=table))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"ranking: the {stage} stage read the window after the moment")
        self.assertNotEqual(self.records["final"]["evidence"]["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"])



@unittest.skipIf(MISSING, MISSING)
class SegmentationInTheEngineTest(SharedReaders, unittest.TestCase):
    key, holdout_metric = "segmentation", "silhouette"
    @classmethod
    def setUpClass(cls):
        cls.case = kinds.get("segmentation").case(0)
        cls.store, cls.pid, cls.records = run_case("segmentation", cls.case)

    def test_its_decisions_hold_on_the_engines_records(self):
        from dclab_rnd.kinds.acceptance import Run

        checks = kinds.get("segmentation").accept_engine(Run(self.case, self.store.get(self.pid)["solution"], self.records))
        self.assertEqual([c["decision"] for c in checks if not c["ok"]], [], checks)

    def test_a_moved_holdout_is_refused(self):
        prepared = engine.prepare(self.store, self.store.get(self.pid))
        previous = {s: self.records[s] for s in engine.STAGE_KEYS if s != "final"}
        moved = {**previous["data"], "evidence": {**previous["data"]["evidence"], "split": {**previous["data"]["evidence"]["split"], "fingerprint": "0" * 16}}}
        with self.assertRaises(ValueError) as caught:
            prepared.machine.final("x", {**previous, "data": moved})
        self.assertIn("rerun the data stage", str(caught.exception))

    def test_the_held_out_rows_reach_only_the_final(self):
        from dclab_rnd.kinds.segmentation import SegmentationMachine

        table = self.case.table.copy()
        order = __import__("numpy").random.default_rng(0).permutation(len(table))
        held = sorted(order[: int(round(len(table) * SegmentationMachine.HOLDOUT))])
        numbers = [c for c in table.columns if c != "renewed" and pd.api.types.is_float_dtype(table[c])]
        table.loc[held, numbers] = table.loc[held, numbers] * 3 + 5  # the held-out rows' own values
        _, _, other = run_case("segmentation", dataclasses.replace(self.case, table=table))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"segmentation: the {stage} stage read the held-out rows")
        self.assertNotEqual(self.records["final"]["evidence"]["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"])

    def test_the_final_scales_and_fits_on_the_training_rows_only(self):
        from sklearn.metrics import silhouette_score

        from dclab_core import segment

        prepared = engine.prepare(self.store, self.store.get(self.pid))
        machine, final = prepared.machine, self.records["final"]["evidence"]
        method, k = final["model"].split(":")[0], int(final["model"].split(":")[1])
        X_train, record = segment.prepare(machine.training_rows(), self.records["models"]["evidence"]["columns_used"])
        labels = segment._fit(method, k, X_train, 0).predict(segment.transform(machine.data_frame, record))
        held = segment.transform(machine.data_frame, record)[machine.holdout]
        self.assertAlmostEqual(final["holdout_metrics"]["silhouette"], round(float(silhouette_score(held, labels[machine.holdout])), 3), places=3)
        fitted_on = []
        real = segment.prepare
        def watched(frame, *args, **kwargs):
            fitted_on.append(len(frame))
            return real(frame, *args, **kwargs)
        previous = {s: self.records[s] for s in engine.STAGE_KEYS if s != "final"}
        with mock.patch.object(segment, "prepare", watched):
            machine.final("x", previous)
        self.assertTrue(fitted_on)
        self.assertLessEqual(max(fitted_on), len(machine.training_rows()))  # no scaling learned from a held-out row

    def test_the_outcome_reaches_only_the_final(self):
        table = self.case.table.copy()
        table["renewed"] = table["renewed"].iloc[::-1].to_numpy()  # another outcome, the same columns
        _, _, other = run_case("segmentation", dataclasses.replace(self.case, table=table))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"segmentation: the {stage} stage read the outcome")
        self.assertNotEqual(self.records["final"]["evidence"]["outcome"], other["final"]["evidence"]["outcome"])


def notes_case(target: str, task: str):
    """The notes replica with its topic (seven classes) or its minutes to resolve (a number) as the outcome."""
    from dclab_core.replicas import shapes_and_notes

    case = kinds.get("embedded").case(0)
    notes = shapes_and_notes.notes(0, outcomes=True)
    table = notes.table.assign(text=notes.items).drop(columns=[c for c in ("urgent", "topic", "minutes_to_resolve") if c != target])
    return dataclasses.replace(case, table=table, target=target, task=task, solution={**case.solution, "task": task})


@unittest.skipIf(MISSING, MISSING)
class LongTextsAsClassesAndNumbersTest(unittest.TestCase):
    """16.15 C's open half: the embedded machine answers a class or a number from long texts, sources kept whole, the
    holdout's labels read once in the final stage, scored against what no text gives."""

    @classmethod
    def setUpClass(cls):
        cls.topic = notes_case("topic", "multiclass")
        cls.minutes = notes_case("minutes_to_resolve", "regression")
        cls.runs = {name: run_case("embedded", case) for name, case in (("topic", cls.topic), ("minutes", cls.minutes))}

    def test_a_topic_is_read_from_the_words_beyond_the_majority_class(self):
        final = self.runs["topic"][2]["final"]["evidence"]
        self.assertEqual(final["metric"], "accuracy")
        self.assertEqual(final["sources_on_both_sides"], 0)
        majority = final["baselines"]["majority_class"]["accuracy"]
        self.assertGreater(final["holdout_metrics"]["accuracy"], majority + 0.2)  # the topic is planted in the words
        self.assertGreater(len(self.runs["topic"][2]["data"]["evidence"]["classes"]), 2)

    def test_minutes_are_read_from_the_words_beyond_the_training_mean(self):
        final = self.runs["minutes"][2]["final"]["evidence"]
        self.assertEqual(final["metric"], "r2")
        self.assertGreater(final["holdout_metrics"]["r2"], final["baselines"]["training_mean"]["r2"] + 0.2)

    def test_the_holdouts_labels_reach_only_the_final(self):
        for name, case in (("topic", self.topic), ("minutes", self.minutes)):
            store, pid, records = self.runs[name]
            final = records["final"]["evidence"]
            held = final["holdout_row_positions"]
            table = case.table.copy()
            if name == "topic":
                table.loc[held, case.target] = "question"  # a class the training rows have, on every held-out row
            else:
                table.loc[held, case.target] = table.loc[held, case.target] * 3 + 5
            _, _, other = run_case("embedded", dataclasses.replace(case, table=table))
            self.assertEqual(before_final(records), before_final(other), name)
            self.assertNotEqual(final["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"], name)

    def test_a_held_out_class_the_training_rows_never_had_is_refused(self):
        table = self.topic.table.copy()
        held = self.runs["topic"][2]["final"]["evidence"]["holdout_row_positions"]
        table.loc[held[:3], "topic"] = "never_seen"
        with self.assertRaises(Exception) as caught:
            run_case("embedded", dataclasses.replace(self.topic, table=table))
        self.assertIn("never had", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
