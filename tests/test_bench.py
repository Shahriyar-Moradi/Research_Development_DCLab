"""DCLab Bench B3 (2026-11-03): every campaign in a layer, the free runner recorded once and never over a record, sealed
suites never run, and the summary a layer state with its reasons."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import bench  # noqa: E402


class LayerMapTests(unittest.TestCase):
    def test_every_campaign_has_a_layer(self):
        _, unassigned = bench.evidence_rows()
        self.assertEqual(unassigned, [], "a new campaign needs its line in dclab_rnd.bench.LAYER_OF")

    def test_the_runner_never_runs_a_sealed_or_live_suite(self):
        for layer, commands in bench.COMMANDS.items():
            for name, argv in commands:
                words = " ".join(argv).lower()
                for banned in ("sealed", "split=test", "--yes", "--live", "live", "cap-eur"):
                    self.assertNotIn(banned, words, f"{layer}/{name}")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.campaigns = Path(tempfile.mkdtemp())
        (self.campaigns / "journeys_v1" / "results").mkdir(parents=True)
        (self.campaigns / "journeys_v1" / "results" / "JRN-001_dev.json").write_text(json.dumps(
            {"at": "2026-10-01T00:00:00+00:00", "verdict": "pass: 30 of 30", "journeys": [30, 30]}))
        (self.campaigns / "journeys_v1" / "results" / "JRN-002_sealed_v1.json").write_text(json.dumps(
            {"at": "2026-10-02T00:00:00+00:00", "verdict": "fail: 21 of 30", "split": "sealed_v1", "journeys": [21, 30]}))
        self.runs = Path(tempfile.mkdtemp())
        self.calls = []

    def fake(self, argv):
        self.calls.append(argv)
        return 0 if "agent-eval" in argv else 2

    def test_a_run_is_recorded_once_with_each_command_timed(self):
        path = bench.run(["agents"], runs=self.runs, execute=self.fake, campaigns=self.campaigns)
        doc = json.loads(path.read_text())
        self.assertEqual([c["suite"] for c in doc["commands"]], ["agent-eval", "harness-eval", "journeys-dev"])
        self.assertEqual([c["exit"] for c in doc["commands"]], [0, 2, 2])
        self.assertEqual(doc["rows"][0]["passed"], True)
        self.assertEqual(doc["rows"][1]["verdict"], "fail: exit 2")
        self.assertIn("agents", doc["states"])
        self.assertIn("commit", doc)
        before = path.read_text()
        original = bench.datetime

        class Frozen(original):  # the same second again: the same file name
            @classmethod
            def now(cls, tz=None):
                return original.strptime(path.name.split("_")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=tz)

        bench.datetime = Frozen
        try:
            with self.assertRaises(FileExistsError):
                bench.run(["agents"], runs=self.runs, execute=self.fake, campaigns=self.campaigns)
        finally:
            bench.datetime = original
        self.assertEqual(path.read_text(), before)  # the record is untouched

    def test_the_summary_reads_the_latest_result_and_names_why_a_layer_is_red(self):
        s = bench.summary(campaigns=self.campaigns, runs=self.runs)
        agents = s["layers"]["agents"]
        self.assertEqual(agents["status"], "red")  # the latest result, the sealed one, failed
        self.assertTrue(any("JRN-002_sealed_v1" in r for r in agents["reasons"]))
        tracks = {t["track"]: t for t in agents["suites"]}
        self.assertEqual(set(tracks), {"dev", "sealed"})  # a dev result and a sealed one are separate tracks
        sealed = tracks["sealed"]
        self.assertEqual((sealed["ref"], sealed["passed"], sealed["results"], sealed["sealed_results"]), ("JRN-002_sealed_v1", False, 1, 1))
        self.assertEqual(sealed["count"]["n"], 30)
        self.assertEqual(s["layers"]["software"]["status"], "empty")
        self.assertEqual(s["history"], [])

    def test_a_later_dev_pass_never_hides_a_failed_sealed_result(self):
        """Review F1: journeys_v1 had sealed v6 failing and later dev runs passing; the suite read as a pass."""
        (self.campaigns / "journeys_v1" / "results" / "JRN-003_dev.json").write_text(json.dumps(
            {"at": "2026-10-03T00:00:00+00:00", "verdict": "pass: 30 of 30", "journeys": [30, 30]}))
        agents = bench.summary(campaigns=self.campaigns, runs=self.runs)["layers"]["agents"]
        self.assertEqual(agents["status"], "red")
        self.assertTrue(any("JRN-002_sealed_v1" in r for r in agents["reasons"]))

    def test_an_unknown_layer_is_refused_before_anything_runs(self):
        """Review F5: make bench LAYERS=agent recorded an empty run."""
        with self.assertRaises(ValueError):
            bench.run(["agent"], runs=self.runs, execute=self.fake)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.runs.glob("*.json")), [])

    def test_the_route_is_the_owners_only(self):
        """Review F7: without its guard entry any member could read it (a GET defaults to read)."""
        from dclab_rnd.accounts import guard, roles

        self.assertEqual(guard.permission("GET", "/api/admin/bench"), "admin")
        self.assertTrue(roles.allows("owner", "admin"))
        for role in ("data_scientist", "reviewer", "viewer"):
            self.assertFalse(roles.allows(role, "admin"), role)

    def test_a_recorded_run_joins_the_history_and_its_layer(self):
        bench.run(["software"], runs=self.runs, execute=lambda argv: 0, campaigns=self.campaigns)
        s = bench.summary(campaigns=self.campaigns, runs=self.runs)
        self.assertEqual(len(s["history"]), 1)
        self.assertEqual(s["layers"]["software"]["suites"][0]["suite"], "cmd:product-e2e")
        self.assertEqual(s["layers"]["software"]["status"], "thin")  # one command passing is not 20 cases

    def test_task_planning_v2_and_v3_verdicts_are_read_into_system_tasks(self):
        rows, unassigned = bench.evidence_rows()
        self.assertEqual(unassigned, [])
        tp3 = [r for r in rows if r.suite == "task_planning_v3" and r.ref == "TP3-001_live" and r.metric == "verdict"]
        self.assertEqual(len(tp3), 1)
        self.assertIn("gate_v2: PASS", tp3[0].verdict)
        self.assertEqual(tp3[0].gate, "tp06.gate.v2")
        self.assertEqual(tp3[0].model, "qwen/qwen3.8-flash")
        self.assertGreater(tp3[0].cost_eur or 0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
