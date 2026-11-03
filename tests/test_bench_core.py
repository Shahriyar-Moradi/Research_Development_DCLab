"""DCLab Bench B1 (2026-11-03): one shape for every evaluation result, the verdict read from every shape the suites wrote,
counts with their interval, and the rule that says a layer's state from its suites' latest results."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import bench  # noqa: E402


class VerdictTests(unittest.TestCase):
    def test_every_shape_the_suites_wrote(self):
        cases = [
            ({"verdict": "pass: 17 of 17"}, True),
            ({"verdict": "FAIL (3 of 10)"}, False),
            ({"verdict": "ok"}, True),
            ({"verdict": "complete"}, None),
            ({"verdict": "too few cases"}, None),
            ({"verdict": {"gate": "g", "status": "pass (dev: the gate is read on sealed)"}}, True),
            ({"verdict": {"gate": "g", "status": "fail"}}, False),
            ({"verdict": {"gate": "g", "status": "not scored (rules only)"}}, None),
            ({"verdict": {"gate": "decision_gate_v1", "pass": False, "criteria": []}}, False),
            ({"verdict": {"passed": True, "right": 3}}, True),
            ({"verdict": {"daily": {}, "pooled": {}}}, None),
            ({"verdict": {"pooled": "fail", "daily": "fail"}}, False),  # review F3: time_decision_v1
            ({"verdict": {"pooled": "pass", "daily": "fail"}}, False),  # time_decision_v2
            ({"verdict": {"daily": "pass"}}, True),  # time_decision_v5
            ({"verdict": {"daily": "pass", "note": "kept"}}, None),  # a word that is not a verdict: not scored
            ({"verdict": "ok: baseline recorded, today's DCLab passes 3 of 20"}, None),  # review F8: a record, not a gate
            ({"result": "fail", "rule": "one run"}, False),
            ({"result": "solved"}, None),
            ({"passes": []}, False),
            ({"passes": ["standard_plan"]}, True),
            ({"status": "completed"}, None),
            ({}, None),
        ]
        for doc, want in cases:
            with self.subTest(doc=doc):
                self.assertIs(bench.verdict_of(doc)[0], want)

    def test_the_split_comes_from_the_result_then_its_name(self):
        self.assertEqual(bench.split_of({"split": "sealed_v3"}), "sealed")
        self.assertEqual(bench.split_of({"gate": {"split": "dev"}}), "dev")
        self.assertEqual(bench.split_of({}, "BEN4-002_test_layer"), "test")
        self.assertEqual(bench.split_of({}, "AEV-103"), "unknown")


class CountTests(unittest.TestCase):
    def test_a_stored_interval_is_kept_and_a_missing_one_is_wilson(self):
        doc = {"leaks_caught": [42, 64], "leaks_caught_ci95": [0.53, 0.76], "summary": {"right": [8, 10], "cases": 10, "note": "x"},
               "not_a_count": [3, 2], "flag": [True, False]}
        got = {c["metric"]: c for c in bench.counts(doc)}
        self.assertEqual((got["leaks_caught"]["k"], got["leaks_caught"]["n"], got["leaks_caught"]["lower"]), (42, 64, 0.53))
        self.assertAlmostEqual(got["summary.right"]["lower"], 0.4902, places=3)  # Wilson 8 of 10
        self.assertEqual(got["summary.cases"]["value"], 10.0)
        self.assertNotIn("not_a_count", got)  # k above n is not a count
        self.assertNotIn("flag", got)
        self.assertNotIn("summary.note", got)

    def test_a_result_becomes_rows_that_carry_its_context(self):
        doc = {"verdict": "fail", "split": "sealed_v6", "spent_eur": 0.41, "model": "claude-sonnet-5-5", "at": "2026-11-01T10:00:00+00:00",
               "gate": {"version": "journeys_gate_v1"}, "journeys": [21, 30]}
        rows = bench.rows_of(doc, suite="journeys_v1", ref="JRN-025_sealed_v6", layer="agents", commit="abc")
        self.assertEqual(rows[0].metric, "verdict")
        self.assertIs(rows[0].passed, False)
        self.assertEqual({r.split for r in rows}, {"sealed"})
        self.assertEqual((rows[1].k, rows[1].n, rows[1].gate, rows[1].cost_eur), (21, 30, "journeys_gate_v1", 0.41))
        self.assertEqual(json.loads(rows[0].model_dump_json())["schema_version"], "dclab_core.bench_row.v1")


class LayerTests(unittest.TestCase):
    def row(self, suite, ref, passed, at, n=None, verdict=""):
        out = [bench.BenchRow(suite=suite, ref=ref, layer="agents", passed=passed, at=at, verdict=verdict)]
        if n:
            out.append(bench.BenchRow(suite=suite, ref=ref, layer="agents", metric="right", k=n // 2, n=n, at=at))
        return out

    def test_the_latest_result_of_each_suite_decides(self):
        old_fail = self.row("a", "A-1", False, "2026-10-01", 40)
        new_pass = self.row("a", "A-2", True, "2026-10-02", 40)
        self.assertEqual(bench.layer_status(old_fail + new_pass)["status"], "green")
        self.assertEqual(bench.layer_status(new_pass + self.row("b", "B-1", False, "2026-10-03", 40, "fail: 3 of 10"))["status"], "red")
        amber = bench.layer_status(new_pass + self.row("c", "C-1", None, "2026-10-03"))
        self.assertEqual((amber["status"], amber["unscored"]), ("amber", 1))
        thin = bench.layer_status(self.row("a", "A-2", True, "2026-10-02", 6))
        self.assertEqual(thin["status"], "thin")
        self.assertTrue(any("fewer than" in r for r in thin["reasons"]))
        self.assertEqual(bench.layer_status([])["status"], "empty")

    def test_each_decision_point_and_split_is_its_own_track(self):
        """Review F1: one decision point's later result hid eleven failing points; a later dev pass hid a sealed fail."""
        rows = (self.row("decision_models_v1", "DMG-013_column.time", False, "2026-10-14", 40)
                + self.row("decision_models_v1", "DMG-014_column.semantic_role", True, "2026-10-14", 40))
        self.assertEqual(bench.layer_status(rows)["status"], "red")
        self.assertEqual(bench.track_of("JRN-025_sealed_v6", "sealed"), ("sealed", ""))
        self.assertEqual(bench.track_of("JRN-024_sealed_v6", "sealed"), bench.track_of("JRN-025_sealed_v6", "sealed"))
        self.assertEqual(bench.track_of("DMG-014_column.semantic_role"), ("unknown", "column.semantic_role"))
        self.assertEqual(bench.track_of("CEX-034_dev_real", "dev"), ("dev", "real"))

    def test_an_undated_result_is_ordered_by_its_number(self):
        """Review F2: search_v3's decision had no date and lost to an older dated result."""
        older = self.row("search_v3", "SRCH-009", True, "2026-10-01", 40)
        newer = self.row("search_v3", "SRCH-010", False, "", 40)
        self.assertEqual(bench.layer_status(older + newer)["status"], "red")

    def test_a_red_layer_names_the_failing_result(self):
        state = bench.layer_status(self.row("journeys_v1", "JRN-025", False, "2026-11-01", 30, "fail: 21 of 30"))
        self.assertIn("journeys_v1 JRN-025: failed (fail: 21 of 30)", state["reasons"])


if __name__ == "__main__":
    unittest.main()
