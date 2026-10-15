"""Package 16.15: the engine runs the library kinds' own machines (forecasting and time to event first). The same five
stages, graph, gates and record shape; the holdout locked by the kind's rule in the data stage and read once in the final
stage, after the choice was made on validation; a reuse needs a person's reason; exports refuse; the shared record
readers (the intern's compact record, the answers, the solution report) work on these records."""

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine, graph
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def built(key: str, case=None):
    tmp = tempfile.mkdtemp()
    return acceptance.build(case or kinds.get(key).case(0), Path(tmp))


def run_case(key: str, case=None):
    store, pid = built(key, case)
    with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    return store, pid, store.records(pid)


VOLATILE = {"elapsed_seconds"}


def before_final(records):
    """Every stage's evidence before the final, without the timings."""
    return {s: {k: v for k, v in records[s]["evidence"].items() if k not in VOLATILE} for s in engine.STAGE_KEYS if s != "final"}


@unittest.skipIf(MISSING, MISSING)
class ForecastInTheEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.pid = built("forecasting")
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            cls.records = {stage: engine.execute(cls.store, cls.pid, stage) for stage in engine.STAGE_KEYS}

    def test_every_stage_has_the_record_shape_and_says_its_machine(self):
        for stage, record in self.records.items():
            self.assertEqual(record["machine"], "forecasting", stage)
            for key in ("experiment_id", "setup_summary", "evidence", "claims", "notes", "provenance", "primary_metric", "decision_time_rule"):
                self.assertIn(key, record, stage)
            for c in record["claims"]:
                self.assertEqual(set(c), {"claim_id", "kind", "statement", "evidence", "limitations"})
            for note in record["notes"]:
                self.assertTrue(note["proof"], f"{stage}: {note['title']}")
            self.assertEqual(record["primary_metric"], "mase")

    def test_the_holdout_is_locked_first_and_read_once_after_the_choice(self):
        data, models, final = (self.records[s]["evidence"] for s in ("data", "models", "final"))
        self.assertEqual(final["holdout_origin"], data["holdout_origin"])  # locked in the data stage, by the kind's rule
        self.assertLess(max(models["validation_origins"]), data["holdout_origin"])  # every validation origin strictly before it
        self.assertEqual(final["model"], self.records["models"]["decision"]["chosen"])
        self.assertEqual(final["holdout_uses_in_this_project"], 1)
        self.assertTrue(final["holdout_consumed"])
        self.assertFalse(final["production_approved"])

    def test_a_reuse_needs_a_person_and_the_agent_never_may(self):
        options = [o["id"] for o in self.records["models"]["decision"]["options"]]
        other = next(o for o in options if o != self.records["models"]["decision"]["chosen"])
        engine.approve(self.store, self.pid, "models", other)
        with self.assertRaises(graph.GraphBlocked):
            engine.execute(self.store, self.pid, "final")
        with self.assertRaises(graph.GraphBlocked):
            engine.execute(self.store, self.pid, "final", actor="agent", reuse_reason="a better score")
        with private_scope(True):
            again = engine.execute(self.store, self.pid, "final", reuse_reason="the owner wants the per-series model")
        self.assertEqual(again["evidence"]["model"], other)
        self.assertEqual(again["evidence"]["holdout_uses_in_this_project"], 2)

    def test_exports_refuse_and_the_shared_readers_work(self):
        from dclab_rnd.intern.tools import compact_record
        from dclab_rnd.studio import agent, export
        from dclab_core.report import to_markdown
        from dclab_rnd.studio.solution_report import build as report_build

        project = self.store.get(self.pid)
        records = self.store.records(self.pid)
        with self.assertRaises(export.ExportUnsupported):
            export.notebook(project, records)
        for record in records.values():
            self.assertTrue(compact_record(record)["numbers"] is not None)
        for question in ("which model is best?", "is rain a leak?", "how good is the holdout score?"):
            self.assertTrue(agent.answer(question, project, records, "forecasting")["answer"])
        text = to_markdown(report_build(self.store, self.pid, analyse=False))
        self.assertIn("mase", text.lower())  # the report reads the machine's final record


@unittest.skipIf(MISSING, MISSING)
class SurvivalInTheEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.pid, cls.records = run_case("time_to_event")

    def test_the_survival_machine_runs_every_stage(self):
        records = self.records
        self.assertEqual({r["machine"] for r in records.values()}, {"time_to_event"})
        self.assertIn("months_billed", records["leakage"]["evidence"]["kept_out"])
        self.assertEqual(records["final"]["evidence"]["model"], records["models"]["decision"]["chosen"])
        self.assertTrue(records["final"]["evidence"]["beats_kaplan_meier"])
        titles = [n["title"] for n in records["final"]["notes"]]
        self.assertIn("Not production-approved", titles)
        self.assertIn("The holdout has now been used once", titles)

    def test_the_service_export_refuses_and_the_sft_examples_and_the_report_read_the_records(self):
        from dclab_rnd.studio import service, sft
        from dclab_rnd.studio.solution_report import findings_for

        engine.approve(self.store, self.pid, "final")
        with self.assertRaises(service.ServiceRefused):  # a refusal, not a crash on the missing tabular bundle
            service.export(self.store, self.pid, Path(tempfile.mkdtemp()))
        project = self.store.get(self.pid)
        self.assertTrue(project["solution"].get("forbidden"))  # the path that read the tabular leakage fields
        self.assertTrue(sft.examples_from_project(project, self.store.records(self.pid)))
        findings, note = findings_for(self.store, project)
        self.assertIsNotNone(findings)
        self.assertEqual(note, "the run's training rows; the holdout is left out")

    def test_the_final_stage_refuses_a_holdout_that_moved_since_it_was_locked(self):
        from dclab_rnd.studio import machine
        from dclab_rnd.studio.solution import Solution

        project = self.store.get(self.pid)
        prepared = machine.prepare(self.store, project, Solution(**project["solution"]), kinds.get("time_to_event"))
        previous = {s: self.records[s] for s in engine.STAGE_KEYS if s != "final"}
        moved = {**previous["data"], "evidence": {**previous["data"]["evidence"], "split": {**previous["data"]["evidence"]["split"], "holdout": 1}}}
        with self.assertRaises(ValueError) as refused:
            prepared.machine.final("x", {**previous, "data": moved})
        self.assertIn("rerun the data stage", str(refused.exception))


@unittest.skipIf(MISSING, MISSING)
class TheHoldoutReachesOnlyTheFinalTest(unittest.TestCase):
    """The holdout's targets changed, nothing else: every stage before the final must be the same, the final not."""

    def test_forecasting(self):
        from dclab_core import forecast

        case = kinds.get("forecasting").case(0)
        time, target, f = case.solution["time_column"], case.target, case.solution["kind_fields"]
        origin = forecast.holdout_origin(case.table, time=time, horizon=f["horizon"], gap=f.get("gap", 0))
        table = case.table.copy()
        later = pd.to_datetime(table[time]) > origin
        table.loc[later, target] = table.loc[later, target] * 10 + 5
        self.assertTrue(later.any())
        self._same_until_the_final("forecasting", case, dataclasses.replace(case, table=table))

    def test_time_to_event(self):
        from dclab_core import survival

        case = kinds.get("time_to_event").case(0)
        f = case.solution["kind_fields"]
        d, e, s = f["duration_column"], f["event_column"], f["start_column"]
        frm = survival.split(case.table, duration=d, event=e, start=s)["split"]["holdout_from"]
        table = case.table.copy()
        later = pd.to_datetime(table[s]).dt.normalize() >= pd.Timestamp(frm)
        table.loc[later, d] = table.loc[later, d] * 3 + 1
        table.loc[later, e] = 1 - table.loc[later, e]
        self.assertTrue(later.any())
        self._same_until_the_final("time_to_event", case, dataclasses.replace(case, table=table))

    def _same_until_the_final(self, key, case, altered):
        _, _, a = run_case(key, case)
        _, _, b = run_case(key, altered)  # the proposal made before the project reads the earliest rows only: it is part of the test
        first, second = before_final(a), before_final(b)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"{key}: the {stage} stage read the holdout's targets")
        self.assertNotEqual(a["final"]["evidence"]["holdout_metrics"], b["final"]["evidence"]["holdout_metrics"])


@unittest.skipIf(MISSING, MISSING)
class TheClassBalanceIsReadFromTrainingRowsTest(unittest.TestCase):
    def test_the_holdouts_labels_never_pick_the_metric(self):
        """With no metric chosen, the default metric and the rare-event screen follow the class balance: it is read
        from the training rows, so labels on the holdout's rows (the latest by time) change neither."""
        case = kinds.get("binary").case(0)

        def prepared(table):
            store, pid = built("binary", dataclasses.replace(case, table=table))
            return engine.prepare(store, store.get(pid)).bundle

        first = prepared(case.table)
        latest = case.table["signup_time"].rank(method="first") > len(case.table) - len(first.test_idx)
        table = case.table.copy()
        table.loc[latest, "churned"] = "yes"  # every held-out subscriber churned: 18% positive overall would become 34%
        second = prepared(table)
        self.assertEqual(first.spec.positive_rate, second.spec.positive_rate)
        self.assertEqual(first.spec.positive_rate, float(first.y_train.mean()))
        self.assertEqual(first.spec.primary_metric, second.spec.primary_metric)


@unittest.skipIf(MISSING, MISSING)
class SolutionTest(unittest.TestCase):
    def test_a_machine_kinds_solution_says_what_its_machine_needs(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "units", "task": "regression", "prediction_moment": "Each Sunday for the week ahead", "kind": "forecasting",
                "kind_fields": {"series_column": "series", "horizon": 7}}
        self.assertEqual(Solution(**{**base, "time_column": "date"}).problem_kind().key, "forecasting")
        refusals = [
            ({**base}, "needs its time column"),  # no time column: a forecast cannot order its days
            ({**base, "time_column": "date", "kind_fields": {"series_column": "units", "horizon": 7}}, "cannot be the target"),
            ({**base, "time_column": "series"}, "cannot be the target or the time column"),
        ]
        for fields, reason in refusals:  # each refused for its own reason, not for another field
            with self.assertRaises(ValueError) as refused:
                Solution(**fields)
            self.assertIn(reason, str(refused.exception))
        ranking = dict(task="binary", prediction_moment="On the day the list is shown", kind="ranking",
                       kind_fields={"user_column": "u", "item_column": "i", "time_column": "t", "event_column": "e",
                                    "relevant_event": "purchase", "moment": "2025-04-10"})
        Solution(**ranking)  # 16.15 B: the engine runs it
        with mock.patch.object(kinds.get("ranking"), "runs_in_engine", False), self.assertRaises(ValueError) as refused:
            Solution(**ranking)  # a kind the engine does not run is refused, whichever it is
        self.assertIn("does not run it yet", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
