"""Research lab and Benchmark routes: every number is read from the repository, nothing is typed in."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class LabRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        cls.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
        cls.env.start()
        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.summary = cls.client.get("/api/lab/summary").json()
        cls.bench = cls.client.get("/api/lab/benchmark").json()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.env.stop()

    def test_registry_and_champions_match_the_status_command(self):
        from dclab_rnd.analysis import build_evidence
        from dclab_rnd.registry import collect_registry
        records, issues = collect_registry(ROOT)
        evidence = build_evidence(ROOT, records, issues)
        reg = self.summary["registry"]
        self.assertEqual(reg["experiments"], evidence["summary"]["experiments"])
        self.assertEqual(reg["eligible"], evidence["summary"]["deployment_eligible_completed"])
        self.assertEqual(reg["datasets"], evidence["summary"]["datasets"])
        champs = {c["dataset"]: c for c in self.summary["champions"]}
        self.assertEqual(set(champs), {c["dataset"] for c in evidence["champions"]})
        for c in evidence["champions"]:
            self.assertAlmostEqual(champs[c["dataset"]]["value"], c["roc_auc"])
            self.assertEqual(champs[c["dataset"]]["run_id"], c["run_id"])

    def test_campaigns_come_from_the_campaign_folders(self):
        folders = sorted(p.name for p in (ROOT / "evidence/campaigns").iterdir() if p.is_dir())
        campaigns = {c["id"]: c for c in self.summary["campaigns"]}
        self.assertEqual(sorted(campaigns), folders)
        for name, c in campaigns.items():
            results = sorted((ROOT / "evidence/campaigns" / name / "results").glob("*.json"))
            self.assertEqual(c["experiments"], len(results), name)
            self.assertEqual(sorted(c["record_ids"]), sorted(json.loads(p.read_text())["experiment_id"] for p in results))
            self.assertIn(c["status"], ("done", "in progress", "failed", "planned"))
            self.assertTrue(c["objective"], name)
        index = [json.loads(l) for l in (ROOT / "evidence/knowledge/rag/records.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(self.summary["index"]["records"], len(index))
        self.assertEqual(self.summary["index"]["campaign_records"],
                         sum(1 for r in index if (r.get("metadata") or {}).get("campaign") in campaigns))

    def test_critic_gate_numbers_are_the_gate_output(self):
        from dclab_rnd import critic_gate
        stats = critic_gate.summary(critic_gate.gate_campaigns(ROOT))
        c = self.summary["critic"]
        self.assertEqual(c["challenges"], stats["critic_challenges"])
        self.assertEqual(c["contradicted"], stats["contradicted_challenges"])
        self.assertEqual(c["kept"] + c["contradicted"], c["challenges"])
        self.assertEqual(len(c["examples"]), c["contradicted"])
        self.assertTrue(all(e["consistent"] for e in c["examples"]), "a contradicted challenge disputes a confirmed rule")

    def test_pitfalls_report_their_measured_cost(self):
        files = sorted((ROOT / "evidence/campaigns/pitfalls_v1/results").glob("PIT-*.json"))
        pits = {p["id"]: p for p in self.summary["pitfalls"]}
        self.assertEqual(sorted(pits), [json.loads(f.read_text())["experiment_id"] for f in files])
        for f in files:
            r = json.loads(f.read_text())
            p = pits[r["experiment_id"]]
            self.assertTrue(p["cost"]["text"], p["id"])
            rows = [d for d in r["evidence"].get("datasets", []) if isinstance(d, dict) and "inflation" in d]
            if rows:  # the headline is the largest measured inflation, not a typed number
                self.assertAlmostEqual(p["cost"]["value"], max((d["inflation"]["mean"] for d in rows), key=abs))

    def test_auditor_replay_is_the_stored_result(self):
        stored = json.loads((ROOT / "evidence/campaigns/agent_verification_v1/results/VER-001_auditor_blind_replay.json").read_text())
        head, full = self.summary["auditor"], self.bench["auditor"]
        self.assertEqual((head["found"], head["known"], head["datasets"]), (stored["found"], stored["known_leaks"], stored["datasets"]))
        self.assertNotIn("cases", head)
        self.assertEqual(sum(len(c["known"]) for c in full["cases"]), stored["known_leaks"])
        self.assertEqual(sum(len(c["found"]) for c in full["cases"]), stored["found"])

    def test_benchmark_plan_is_marked_planned_with_no_scores(self):
        self.assertEqual(self.bench["suites"]["status"], "planned")
        self.assertTrue(all(s["status"] == "planned" and s["scored_cases"] == 0 for s in self.bench["suites"]["items"]))
        self.assertEqual(self.bench["suites"]["planned_cases"], sum(s["planned_cases"] for s in self.bench["suites"]["items"]))
        # policies: the standard plan and two scripted references are scored on the scripted judgment suite (A4.1);
        # the model policies and the human reviewer are not
        items = {p["name"]: p for p in self.bench["policies"]["items"]}
        judgment = self.bench["judgment"]
        if judgment:
            self.assertEqual(items["Standard plan (no model)"]["scores"]["leaks_caught"], judgment["summary"]["standard"]["leaks_caught"])
            self.assertEqual(self.bench["policies"]["scored"], 3)
        for name in ("General LLM (open model, router)", "DCLab policy model", "Human reviewer"):
            self.assertIsNone(items[name]["scores"], name)

    def test_notebook_pilot_follows_the_committed_manifest(self):
        from dclab_rnd.notebook_assist import MANIFEST_CANDIDATES
        pilot = self.bench["notebook_pilot"]
        self.assertEqual(pilot["status"] == "not_committed", not MANIFEST_CANDIDATES[0].is_file())
        if pilot["status"] == "scored":
            self.assertEqual(pilot["passed"], sum(c["pass"] for c in pilot["cases"]))

    def test_commands_come_from_the_makefile(self):
        make = (ROOT / "Makefile").read_text()
        commands = self.summary["commands"]["campaign"]
        self.assertTrue(commands)
        for c in commands:
            self.assertIn(f"\n{c['target']}:", "\n" + make)
            self.assertTrue(c["command"].startswith("make "))

    def test_track_statuses_cover_the_research_folders(self):
        tracks = {p.parent.name for p in (ROOT / "research").glob("*/README.md") if not p.parent.name.startswith("_")}
        status = self.summary["track_status"]
        self.assertEqual(set(status), tracks)
        self.assertTrue(all(s["kind"] in ("active", "planned", "proposed", "open", "paused", "concluded", "unknown") for s in status.values()))


class PilotAndCacheTests(unittest.TestCase):
    def test_missing_manifest_is_an_honest_empty_state(self):
        from dclab_rnd.agentic.pages import lab
        with mock.patch("dclab_rnd.notebook_assist.MANIFEST_CANDIDATES", (Path(tempfile.mkdtemp()) / "missing.json",)):
            pilot = lab._notebook_pilot()
        self.assertEqual(pilot["status"], "not_committed")
        self.assertNotIn("passed", pilot)

    def test_cache_follows_file_modification_times(self):
        from dclab_rnd.agentic.pages import lab
        path = Path(tempfile.mkdtemp()) / "x.json"
        path.write_text("1")
        calls = []
        compute = lambda: calls.append(1) or len(calls)  # noqa: E731
        self.assertEqual(lab._cached("test-x", [path], compute), 1)
        self.assertEqual(lab._cached("test-x", [path], compute), 1)
        later = time.time() + 5
        os.utime(path, (later, later))
        self.assertEqual(lab._cached("test-x", [path], compute), 2)


if __name__ == "__main__":
    unittest.main()
