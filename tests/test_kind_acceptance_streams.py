"""Package 16.15 C, streams in the engine: machine readings as the project's table and its incidents as a related table
run the streams machine. The holdout is the latest windows, cut from their dates alone; training reads only the
readings before the cut and the incidents known by it; the method and its alert threshold are set on validation; the
holdout's windows are built and scored once, in the final stage."""

import dataclasses
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.studio.solution import Solution
    from tests.test_engine_machines import before_final, built, run_case
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FIELDS = {"unit_column": "machine_id", "time_column": "at", "sensor_columns": ["temperature", "vibration"], "incident_time_column": "failed_at"}
MOMENT = "Every six hours, for each machine: raise an alert now?"


@unittest.skipIf(MISSING, MISSING)
class StreamsInTheEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kind = kinds.get("streams")
        cls.case = cls.kind.case(0)
        cls.store, cls.pid, cls.records = run_case("streams", cls.case)

    def test_the_replica_ends_with_incidents_caught_within_the_budget(self):
        result = acceptance.run(self.kind)
        self.assertEqual(result["status"], "supported", result)
        self.assertTrue(all(c["ok"] for c in result["checks"]), [c for c in result["checks"] if not c["ok"]])

    def test_the_holdout_reaches_the_final_stage_only(self):
        f = self.case.facts
        cut = pd.Timestamp(self.records["data"]["evidence"]["split"]["cut"])
        readings, incidents = self.case.table.copy(), self.case.tables["incidents"].copy()
        later = pd.to_datetime(readings[f["time"]]) > cut
        for s in f["sensors"]:  # every reading after the cut changed
            readings.loc[later, s] = readings.loc[later, s] * 3 + 7
        after = pd.to_datetime(incidents[f["incident_time"]]) > cut
        self.assertGreater(int(after.sum()), 5)
        machines = readings[f["unit"]].unique()  # the holdout's incidents happen to other machines
        incidents.loc[after, f["unit"]] = np.random.default_rng(1).choice(machines, int(after.sum()))
        _, _, other = run_case("streams", dataclasses.replace(self.case, table=readings, tables={"incidents": incidents}))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"the {stage} stage read the holdout period")
        self.assertNotEqual(self.records["final"]["evidence"]["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"])
        self.assertEqual(self.records["final"]["evidence"]["holdout_uses_in_this_project"], 1)

    def test_the_method_is_chosen_on_validation_and_no_window_reads_past_the_cut(self):
        models, final, feats = self.records["models"]["evidence"], self.records["final"]["evidence"], self.records["features"]["evidence"]
        self.assertEqual(final["model"], models["selected_model"])
        self.assertTrue({"isolation_forest", "own_change", "percentile_rules", "ecod", "supervised"} <= {r["model"] for r in models["ranking"]})
        self.assertEqual(feats["readings_after_the_cut_read"], 0)
        self.assertEqual(feats["future_check"]["changed_by_the_future"], 0)
        self.assertGreater(feats["future_check"]["windows_before_the_cut"], 0)
        self.assertLessEqual(pd.Timestamp(self.records["leakage"]["evidence"]["train_labels_until"]),
                             pd.Timestamp(self.records["data"]["evidence"]["split"]["cut"]))

    def test_machines_left_out_are_never_fitted_on(self):
        final = self.records["final"]["evidence"]
        self.assertTrue(final["held_out_units"] and final["unseen_fitted_on_units"])
        self.assertFalse(set(final["held_out_units"]) & set(final["unseen_fitted_on_units"]))

    def test_an_incident_not_known_by_the_cut_is_never_read_before_the_final(self):
        from dclab_core import streams

        f = self.case.facts
        parts = {**self.records["data"]["evidence"]["split"], "label_delay_hours": 72}
        incidents = self.case.tables["incidents"]
        _, known = streams.known_by(self.case.table, incidents, parts, time=f["time"], incident_time=f["incident_time"])
        at = pd.to_datetime(incidents[f["incident_time"]])
        cut = pd.Timestamp(parts["cut"])
        self.assertGreater(int(((at > cut - pd.Timedelta(hours=72)) & (at <= cut)).sum()), 0)  # some happened, not yet known
        self.assertEqual(len(known), int((at + pd.Timedelta(hours=72) <= cut).sum()))
        self.assertTrue((pd.to_datetime(known[f["incident_time"]]) + pd.Timedelta(hours=72) <= cut).all())

    def test_the_final_stage_refuses_a_holdout_the_data_stage_did_not_lock(self):
        from dclab_rnd.studio.machine import prepare

        project = self.store.get(self.pid)
        machine = prepare(self.store, project, Solution(**project["solution"]), self.kind).machine
        moved = {"data": {"evidence": {"split": {**self.records["data"]["evidence"]["split"], "cut": "2025-02-01 00:00:00"}}}}
        with self.assertRaises(ValueError):
            machine.final("E-1", {**self.records, **moved})

    def test_the_acceptance_fails_when_a_window_moves_with_the_future(self):
        import copy
        from types import SimpleNamespace

        records = copy.deepcopy(self.records)
        records["features"]["evidence"]["future_check"]["changed_by_the_future"] = 4

        def verdict(r):
            run = SimpleNamespace(records=r, case=self.case, evidence=lambda stage: r[stage]["evidence"])
            return self.kind.accept_engine(run)[1]["ok"]

        self.assertTrue(verdict(self.records))
        self.assertFalse(verdict(records))

    def test_a_sensor_written_knowing_the_failure_is_kept_out_until_confirmed(self):
        f = self.case.facts
        readings = self.case.table.copy()
        at = pd.to_datetime(readings[f["time"]])
        inc = self.case.tables["incidents"].assign(_t=pd.to_datetime(self.case.tables["incidents"][f["incident_time"]])).sort_values("_t")
        nxt = pd.merge_asof(readings.assign(_i=np.arange(len(readings)), _at=at).sort_values("_at"), inc[[f["unit"], "_t"]],
                            left_on="_at", right_on="_t", by=f["unit"], direction="forward").sort_values("_i")
        readings["service_counter"] = ((nxt["_t"] - nxt["_at"]).dt.total_seconds() / 3600).clip(upper=300).fillna(300).to_numpy()
        fields = {**self.case.solution["kind_fields"], "sensor_columns": [*f["sensors"], "service_counter"]}
        case = dataclasses.replace(self.case, table=readings, solution={**self.case.solution, "kind_fields": fields})
        _, _, records = run_case("streams", case)
        self.assertEqual(records["leakage"]["evidence"]["sensors_kept_out"], ["service_counter"])
        self.assertFalse(any(c.startswith("service_counter_") for c in records["features"]["evidence"]["features"]))
        confirmed = dataclasses.replace(case, solution={**case.solution, "kind_fields": {**fields, "confirmed_sensors": ["service_counter"]}})
        _, _, records = run_case("streams", confirmed)
        self.assertEqual(records["leakage"]["evidence"]["sensors_kept_out"], [])
        self.assertTrue(any(c.startswith("service_counter_") for c in records["features"]["evidence"]["features"]))

    def test_a_delayed_label_is_known_by_the_cut_in_the_engine(self):
        case = dataclasses.replace(self.case, solution={**self.case.solution, "kind_fields": {**self.case.solution["kind_fields"], "label_delay_hours": 36}})
        _, _, records = run_case("streams", case)
        split = records["data"]["evidence"]["split"]
        self.assertEqual(split["label_delay_hours"], 36)
        self.assertLessEqual(pd.Timestamp(split["train_labels_until"]), pd.Timestamp(split["cut"]))
        self.assertIsNotNone(records["final"]["evidence"]["holdout_primary_metric_ci"]["low"])

    def test_exports_are_refused(self):
        from dclab_rnd.studio import export

        project = self.store.get(self.pid)
        with self.assertRaises(export.ExportUnsupported):
            export.notebook(project, self.records)

    def test_a_project_without_its_incidents_table_is_refused(self):
        store, pid = built("streams", dataclasses.replace(self.case, tables={}))
        from dclab_rnd.studio import data as sd
        from dclab_rnd.studio.machine import prepare

        project = store.get(pid)
        with self.assertRaises(sd.DataError):
            prepare(store, project, Solution(**project["solution"]), self.kind)


@unittest.skipIf(MISSING, MISSING)
class StreamSolutionTest(unittest.TestCase):
    def test_a_solution_names_distinct_columns_and_no_forbidden_sensor(self):
        Solution(task="binary", prediction_moment=MOMENT, kind="streams", kind_fields=FIELDS)
        for fields, extra, why in ((dict(FIELDS, sensor_columns=["temperature", "temperature"]), {}, "a sensor twice"),
                                   (dict(FIELDS, sensor_columns=["machine_id"]), {}, "the unit as a sensor"),
                                   (FIELDS, {"forbidden": [{"column": "vibration", "reason": "written by the repair"}]}, "a forbidden sensor"),
                                   (FIELDS, {"metric": "roc_auc"}, "a metric the machine ignores")):
            with self.assertRaises(ValueError, msg=why):
                Solution(task="binary", prediction_moment=MOMENT, kind="streams", kind_fields=fields, **extra)

    def test_the_kind_names_its_incidents_table(self):
        kind = kinds.get("streams")
        solution = Solution(task="binary", prediction_moment=MOMENT, kind="streams", kind_fields=dict(FIELDS, incidents_table="failures"))
        self.assertEqual(kind.tables_of(solution), ["failures"])
        self.assertEqual(kind.table_columns_of(solution), {"failures": ["machine_id", "failed_at"]})


if __name__ == "__main__":
    unittest.main()
