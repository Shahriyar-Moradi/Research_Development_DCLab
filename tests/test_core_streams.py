"""Package 16.11: streams. Windows read only the past of their end (changing every reading after a cut changes no window
ending before it, on regular and on gappy readings), labels come from incidents in the horizon after the end and drop
the downtime, metrics count incidents (an alert after an incident, or before the previous one's repair, catches nothing),
thresholds and the method are set on the validation period, the holdout and the unseen machines are scored once, and a
delayed label is not used before it is known."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import streams  # noqa: E402
from dclab_core.replicas import machine_sensors  # noqa: E402


class WindowTest(unittest.TestCase):
    def test_window_features_by_hand(self):
        t = pd.date_range("2025-01-01", periods=10, freq="h")
        readings = pd.DataFrame({"m": "a", "t": t.strftime("%Y-%m-%d %H:%M:%S"), "x": np.arange(10.0)})  # x rises one per hour
        w = streams.windows(readings, unit="m", time="t", sensors=["x"], window_hours=4, step_hours=2, baseline_hours=4)
        row = w[w["end"] == pd.Timestamp("2025-01-01 08:00")].iloc[0]
        self.assertEqual(row["x_last"], 8.0)  # the reading at the end itself is known at the end
        self.assertEqual(row["x_mean"], np.mean([5, 6, 7, 8]))  # (end - 4 h, end]
        self.assertAlmostEqual(row["x_slope"], 1.0)
        self.assertTrue(np.isnan(w.iloc[0]["x_vs_own"]))  # no baseline yet: no change score, not a huge one

    def test_the_change_score_reads_the_baseline_before_the_window(self):
        t = pd.date_range("2025-01-01", periods=12, freq="h")
        x = [0, 2, 0, 2, 0, 2, 0, 2, 9, 9, 9, 9]
        w = streams.windows(pd.DataFrame({"m": "a", "t": t.strftime("%Y-%m-%d %H:%M:%S"), "x": x}), unit="m", time="t", sensors=["x"],
                            window_hours=3.5, step_hours=0.5, baseline_hours=6)
        row = w[w["end"] == pd.Timestamp("2025-01-01 11:00")].iloc[0]
        # window (07:30, 11:00]: 9, 9, 9, 9; baseline as it stood at 07:00: hours 02-07, mean 1 and spread 1
        self.assertAlmostEqual(row["x_vs_own"], (9 - 1) / 1)

    def test_gappy_readings_never_read_past_an_end(self):
        rng = np.random.default_rng(0)
        t = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 400, 300)), unit="h")
        t = t[(t < pd.Timestamp("2025-01-06")) | (t > pd.Timestamp("2025-01-06 12:00"))]  # a 12-hour gap that is not an incident
        readings = pd.DataFrame({"m": "a", "t": t.strftime("%Y-%m-%d %H:%M:%S"), "x": rng.normal(size=len(t))})
        w = streams.windows(readings, unit="m", time="t", sensors=["x"], window_hours=6, step_hours=3)
        self.assertTrue((pd.to_datetime(w["_last_read"]) <= w["end"]).all())
        cut = w["end"].quantile(0.5)
        later = readings.copy()
        later.loc[pd.to_datetime(later["t"]) > cut, "x"] += 50
        again = streams.windows(later, unit="m", time="t", sensors=["x"], window_hours=6, step_hours=3)
        pd.testing.assert_frame_equal(w[w["end"] <= cut].reset_index(drop=True), again[again["end"] <= cut].reset_index(drop=True))

    def test_labels_and_incident_metrics_by_hand(self):
        ends = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 12:00", "2025-01-02 00:00", "2025-01-02 06:00"])
        rows = pd.DataFrame({"m": "a", "end": ends})
        inc = pd.DataFrame({"m": ["a"], "at": ["2025-01-02 01:00"]})
        lab = streams.label_windows(rows, inc, unit="m", time="at", horizon_hours=24, downtime_hours=12)
        self.assertEqual(list(lab["end"]), list(ends[:3]))  # the 06:00 window is in the downtime after the incident
        self.assertEqual(list(lab["target"]), [0, 1, 1])  # within 24 h before the incident; midnight of the 1st is 25 h before
        m = streams.incident_metrics(lab, np.array([False, True, False]), inc, unit="m", time="at", horizon_hours=24,
                                     start=pd.Timestamp("2024-12-31"), stop=pd.Timestamp("2025-01-03"))
        self.assertEqual((m["incidents"], m["caught"], m["lead_time_hours_median"]), (1, 1, 13.0))
        late = pd.DataFrame({"m": "a", "end": pd.to_datetime(["2025-01-02 03:00"]), "target": [0]})
        self.assertEqual(streams.incident_metrics(late, np.array([True]), inc, unit="m", time="at", horizon_hours=24,
                                                  start=pd.Timestamp("2024-12-31"), stop=pd.Timestamp("2025-01-03"))["caught"], 0)  # after it: nothing caught

    def test_an_alert_before_the_previous_repair_ended_catches_nothing(self):
        inc = pd.DataFrame({"m": ["a", "a"], "at": ["2025-01-02 00:00", "2025-01-03 06:00"]})  # the second, 30 h after the first
        rows = pd.DataFrame({"m": "a", "end": pd.to_datetime(["2025-01-01 20:00"]), "target": [1]})
        m = streams.incident_metrics(rows, np.array([True]), inc, unit="m", time="at", horizon_hours=48, downtime_hours=24,
                                     start=pd.Timestamp("2024-12-31"), stop=pd.Timestamp("2025-01-04"))
        self.assertEqual((m["incidents"], m["caught"]), (2, 1))  # one alert, before the first: it cannot warn of the second too


class ReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = machine_sensors.generate(0, machines=16, days=75)
        cls.args = dict(unit="machine", time="time", sensors=cls.r.facts["sensors"], incident_time="failed_at")
        cls.result = streams.evaluate(cls.r.readings, cls.r.incidents, **cls.args)

    def test_no_window_moves_with_the_future_and_labels_end_before_the_holdout(self):
        r = self.result
        self.assertEqual(r["future_check"]["changed_by_the_future"], 0)
        self.assertGreater(r["future_check"]["windows_before_the_cut"], 0)
        self.assertLessEqual(pd.Timestamp(r["split"]["train_labels_until"]), pd.Timestamp(r["split"]["cut"]))

    def test_thresholds_and_the_choice_do_not_read_the_holdout(self):
        cut = pd.Timestamp(self.result["split"]["cut"])
        later = self.r.incidents[pd.to_datetime(self.r.incidents["failed_at"]) <= cut]  # every incident after the cut removed
        again = streams.evaluate(self.r.readings, later, **self.args)
        self.assertEqual(again["validation"], self.result["validation"])
        self.assertEqual({n: m["threshold"] for n, m in again["methods"].items()}, {n: m["threshold"] for n, m in self.result["methods"].items()})

    def test_incidents_are_caught_early_within_the_budget(self):
        r = self.result
        for name, m in r["validation"].items():  # thresholds set on these rows: they keep the budget there
            self.assertAlmostEqual(m["alerts_per_unit_day"], r["budget_per_unit_day"], delta=0.03, msg=name)
        best = r["methods"][r["best"]]
        self.assertGreaterEqual(best["incident_recall"], 0.75)
        self.assertGreater(best["lead_time_hours_median"], 6)
        self.assertLessEqual(best["alerts_per_unit_day"], 1.5 * r["budget_per_unit_day"])
        self.assertGreater(r["unseen_units"]["incidents"], 0)

    def test_a_delayed_label_is_not_used_before_it_is_known(self):
        delayed = streams.evaluate(self.r.readings, self.r.incidents, **self.args, label_delay_hours=72)
        self.assertLessEqual(pd.Timestamp(delayed["split"]["train_labels_until"]), pd.Timestamp(delayed["split"]["cut"]))
        self.assertLess(delayed["split"]["fit_windows"], self.result["split"]["fit_windows"])

    def test_too_few_matched_incidents_are_refused(self):
        with self.assertRaises(ValueError):
            streams.evaluate(self.r.readings, self.r.incidents.head(1), **self.args)
        unmatched = self.r.incidents.assign(machine="ZZ")  # incidents for a unit the readings do not have
        with self.assertRaises(ValueError):
            streams.evaluate(self.r.readings, unmatched, **self.args)



class EngineSplitTest(unittest.TestCase):  # 16.15 C: what the engine's stages call
    @classmethod
    def setUpClass(cls):
        r = machine_sensors.generate(0)
        cls.r, cls.f = r, r.facts
        cls.kw = dict(unit=r.facts["unit"], time=r.facts["time"], sensors=r.facts["sensors"])
        cls.parts = streams.split(r.readings, **cls.kw)

    def test_downtime_after_the_cut_never_moves_the_line(self):
        cut = pd.Timestamp(self.parts["cut"])
        at = pd.to_datetime(self.r.readings[self.f["time"]])
        quiet = self.r.readings[~((at > cut + pd.Timedelta(days=3)) & (at < cut + pd.Timedelta(days=6)))]  # every machine silent three days later
        again = streams.split(quiet, **self.kw)
        self.assertEqual((again["cut"], again["validation_from"]), (self.parts["cut"], self.parts["validation_from"]))

    def test_no_incident_in_the_validation_period_is_refused(self):
        p = self.parts
        inc = self.r.incidents
        t = pd.to_datetime(inc[self.f["incident_time"]])
        kept = inc[~((t > pd.Timestamp(p["validation_from"])) & (t <= pd.Timestamp(p["cut"])))]
        view = streams.training_view(self.r.readings, kept, p, incident_time=self.f["incident_time"], **self.kw)
        with self.assertRaises(ValueError):
            streams.choose(view, kept, p, unit=self.f["unit"], incident_time=self.f["incident_time"])

    def test_machines_left_out_without_validation_windows_are_not_scored(self):
        view = streams.training_view(self.r.readings, self.r.incidents, self.parts, incident_time=self.f["incident_time"], **self.kw)
        unit = self.f["unit"]
        units = np.array(sorted(view["fit"][unit].unique()))
        held = set(np.random.default_rng(0).choice(units, max(1, len(units) // 4), replace=False))  # score's own draw
        only_held = {**view, "validation": view["validation"][view["validation"][unit].isin(held)]}  # the others have no validation window
        out = streams.score(self.r.readings, self.r.incidents, only_held, self.parts, "own_change", incident_time=self.f["incident_time"], **self.kw)
        self.assertIsNone(out["unseen_units"])  # not measured, and said, rather than a crash after the holdout was read
        self.assertIn("own_change", out["holdout"])
        full = streams.score(self.r.readings, self.r.incidents, view, self.parts, "own_change", incident_time=self.f["incident_time"], **self.kw)
        self.assertIsNotNone(full["unseen_units"])

    def test_the_wilson_interval(self):
        self.assertEqual(streams.wilson(22, 22), (0.8513, 1.0))
        low, high = streams.wilson(5, 10)
        self.assertTrue(low < 0.5 < high)
        self.assertEqual(streams.wilson(0, 0), (None, None))

class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_acceptance_holds_without_reading_the_answer(self):
        kind = self.kinds.get("streams")
        case = kind.case(0)
        run = kind.solve(type(case)(**{**case.__dict__, "facts": {}}))
        decisions = kind.accept({**run, "facts": case.facts})
        self.assertTrue(all(c["ok"] for c in decisions), decisions)
        peeked = {**run, "evaluation": {**run["evaluation"], "future_check": {**run["evaluation"]["future_check"], "changed_by_the_future": 3}}}
        self.assertFalse(kind.accept(peeked)[0]["ok"])

    def test_detected_by_its_words_and_run_by_the_engine_without_a_target(self):  # 16.15 C
        from dclab_rnd.studio.solution import Solution

        self.assertEqual(self.kinds.detect("Alert before a machine failure from the sensors' readings")["key"], "streams")
        self.assertNotEqual(self.kinds.detect("Which customers will churn next month?")["key"], "streams")
        self.assertEqual(self.kinds.detect("Survival analysis of machine failures: time to failure for each pump, with censored machines")["key"],
                         "time_to_event")
        fields = {"unit_column": "m", "time_column": "t", "sensor_columns": ["x"], "incident_time_column": "at"}
        with self.assertRaises(ValueError):  # the target is built from the incidents
            Solution(target="incident", task="binary", prediction_moment="Every six hours", kind="streams", kind_fields=fields)
        Solution(task="binary", prediction_moment="Every six hours", kind="streams", kind_fields=fields)
        self.assertTrue(self.kinds.get("streams").runs_in_engine)


if __name__ == "__main__":
    unittest.main()
