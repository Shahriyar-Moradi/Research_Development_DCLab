"""Policy model and Domain packs routes (dclab_rnd/agentic/pages/learn.py): corpus, runs, project data, pack usage."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agentic.pages import learn  # noqa: E402
from dclab_rnd.draft.pack import PACKS  # noqa: E402
from dclab_rnd.draft.store import DraftStore
from dclab_rnd.storage import open_stores  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402

SOLUTION = {"target": "churn", "task": "binary", "positive_label": "1", "prediction_moment": "At the monthly billing run, before the call.",
            "forbidden": [], "identifiers": []}


def record(stage):
    return {"status": "completed", "stage": stage, "task": "demo", "task_type": "binary", "primary_metric": "roc_auc", "experiment_id": f"P-{stage}",
            "setup_summary": "Profiled 100 training rows.", "question": "What does the data contain?",
            "claims": [{"kind": "fact", "statement": "The data has 100 training rows and 5 columns.", "limitations": []}],
            "notes": [{"severity": "info", "source": "rule", "title": "Size", "text": "Small data.", "proof": ["DCLAB-R03"]}]}


def move(actor, status, at, name="run_stage"):
    return {"at": at, "actor": actor, "move": name, "args": {}, "from": "WF-04", "to": "WF-04", "state": "dddd------",
            "status": status, "message": status, "failed_checks": [], "rules": [], "evidence": [], "side_effects": []}


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def write(self, train, val, manifest=None):
        def row(task, content):
            return json.dumps({"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": content}, {"role": "assistant", "content": "ok"}],
                               "metadata": {"task": task, "dataset": "adult"}})
        (self.dir / "train.chat.jsonl").write_text("".join(row(t, c) + "\n" for t, c in train), encoding="utf-8")
        (self.dir / "val.chat.jsonl").write_text("".join(row(t, c) + "\n" for t, c in val), encoding="utf-8")
        (self.dir / "MANIFEST.json").write_text(json.dumps(manifest or {
            "total": len(train) + len(val), "train": len(train), "val": len(val), "by_task": {"grounded_qa": 2, "rule_reasoning": 1},
            "validation_policy": "datasets held out entirely: heart_disease, spambase; dataset-free examples use a 12% hash split",
            "quality_gates": {"challenges_kept": 3, "contradicted_challenges_dropped": 1}}), encoding="utf-8")

    def test_manifest_lines_and_one_truncated_example_per_task(self):
        self.write([("grounded_qa", "x" * (learn.EXAMPLE_CHARS + 50)), ("grounded_qa", "second")], [("rule_reasoning", "r")])
        out = learn.corpus(self.dir)
        c = out["corpus"]
        self.assertTrue(c["available"])
        self.assertEqual((c["total"], c["train"], c["val"]), (3, 2, 1))
        self.assertEqual(c["held_out_datasets"], ["heart_disease", "spambase"])
        self.assertEqual(c["lines"], {"train.chat.jsonl": 2, "val.chat.jsonl": 1})
        self.assertTrue(c["lines_match_manifest"])
        self.assertEqual([e["task"] for e in out["examples"]], ["grounded_qa", "rule_reasoning"])
        user = out["examples"][0]["messages"][1]
        self.assertTrue(user["truncated"])
        self.assertEqual(len(user["content"]), learn.EXAMPLE_CHARS)
        self.assertEqual(user["length"], learn.EXAMPLE_CHARS + 50)
        self.assertEqual(out["examples"][1]["split"], "val")

    def test_lines_that_disagree_with_the_manifest_are_reported(self):
        self.write([("grounded_qa", "a")], [], manifest={"total": 5, "train": 4, "val": 1, "by_task": {}})
        self.assertFalse(learn.corpus(self.dir)["corpus"]["lines_match_manifest"])

    def test_missing_corpus(self):
        out = learn.corpus(self.dir)
        self.assertFalse(out["corpus"]["available"])
        self.assertEqual(out["examples"], [])
        steps = learn.curriculum(out["corpus"], {"runs": []}, {"trajectories": 0, "moves": 0, "by_status": {}})
        self.assertFalse(steps[0]["ready"])

    def test_the_repository_corpus_matches_its_manifest(self):
        c = learn.corpus()["corpus"]
        if not c["available"]:
            self.skipTest("corpus v3 not built")
        self.assertEqual(c["total"], c["train"] + c["val"])
        self.assertEqual(sum(c["by_task"].values()), c["total"])
        self.assertTrue(c["lines_match_manifest"])


class RunsTests(unittest.TestCase):
    def test_no_run_then_one_run(self):
        base = Path(tempfile.mkdtemp())
        self.assertEqual(learn.training_runs((base / "missing", base))["runs"], [])
        run = base / "qwen-dclab"
        run.mkdir()
        (base / "empty-folder").mkdir()
        (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "r": 16}))
        (run / "trainer_state.json").write_text(json.dumps({"epoch": 3.0, "global_step": 51, "log_history": [{"eval_loss": 0.9}, {"loss": 0.5}, {"eval_loss": 0.7}]}))
        runs = learn.training_runs((base,))["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["name"], runs[0]["base_model"], runs[0]["rank"], runs[0]["eval_loss"], runs[0]["adapter"]),
                         ("qwen-dclab", "Qwen/Qwen2.5-1.5B-Instruct", 16, 0.7, True))


class ProjectDataTests(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(Path(tempfile.mkdtemp()))

    def test_examples_from_projects(self):
        empty = self.store.create("No solution")
        done = self.store.create("Churn")
        done["solution"] = SOLUTION
        self.store.save(done)
        self.store.write_stage(done["id"], "data", record("data"))
        broken = self.store.create("Old format")
        broken["solution"] = SOLUTION
        self.store.save(broken)
        self.store.write_stage(broken["id"], "data", {"status": "completed"})
        out = learn.project_examples(self.store)
        items = {i["name"]: i for i in out["items"]}
        self.assertEqual((out["projects"], out["with_examples"], out["examples"]), (3, 1, 1))
        self.assertEqual(items["Churn"]["stages"], ["data"])
        self.assertEqual(out["by_stage"], {"data": 1})
        self.assertEqual(items["No solution"]["why"], "no saved solution")
        self.assertIn("could not read", items["Old format"]["why"])
        self.assertEqual(empty["id"], items["No solution"]["id"])

    def test_trajectories_count_actors_and_show_the_latest_blocked_move(self):
        a = self.store.create("A")
        for m in [move("human", "allowed", "2026-10-01T10:00:00+00:00"), move("agent", "blocked", "2026-10-01T10:01:00+00:00", "approve_gate"),
                  move("agent", "allowed", "2026-10-01T10:02:00+00:00"), move("human", "needs_approval", "2026-10-01T10:03:00+00:00")]:
            self.store.transition(a["id"], m)
        b = self.store.create("B")
        self.store.transition(b["id"], move("human", "allowed", "2026-10-02T10:00:00+00:00"))
        self.store.create("No log")
        t = learn.trajectories(self.store)
        self.assertEqual((t["projects_with_log"], t["trajectories"], t["moves"]), (2, 1, 5))
        self.assertEqual(t["by_status"], {"allowed": 3, "blocked": 1, "needs_approval": 1})
        self.assertEqual(t["by_actor"]["agent"], {"moves": 2, "allowed": 1, "blocked": 1, "needs_approval": 0})
        self.assertEqual(t["by_actor"]["human"]["moves"], 3)
        self.assertEqual((t["sample"]["project"]["name"], t["sample"]["index"], t["sample"]["of"], t["sample"]["record"]["move"]), ("A", 2, 4, "approve_gate"))
        steps = learn.curriculum({"available": True, "total": 10}, {"runs": []}, t)
        self.assertEqual([s["ready"] for s in steps], [True, False, False, False])
        self.assertEqual(steps[1]["status"], "needs the graph log")
        logs_only = {**t, "trajectories": learn.TRAJECTORY_TARGET}  # 50 logs alone are not enough (A6.2): the opted-in count decides
        self.assertFalse(learn.curriculum({"available": True, "total": 10}, {"runs": []}, logs_only)[1]["ready"])
        ready = {"trajectories": 50, "projects": 10, "target": 50, "min_projects": 10, "min_decisions": 3, "ready": True, "held_out_projects": 2}
        self.assertTrue(learn.curriculum({"available": True, "total": 10}, {"runs": []}, {**t, "usable": ready})[1]["ready"])

    def test_no_moves_no_sample(self):
        self.store.create("Fresh")
        t = learn.trajectories(self.store)
        self.assertEqual((t["moves"], t["sample"]), (0, None))


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        for key in ("DCLAB_STUDIO_HOME", "DCLAB_DRAFT_HOME", "DCLAB_INTERN_HOME"):
            os.environ.pop(key, None)
        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_policy_route(self):
        projects = open_stores(self.home)[0]  # the app's own store
        p = projects.create("Churn")
        projects.transition(p["id"], move("agent", "blocked", "2026-10-01T10:00:00+00:00"))
        r = self.client.get("/api/learn/policy")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        for key in ("corpus", "examples", "critic_gate", "training", "projects", "trajectories", "curriculum", "review"):
            self.assertIn(key, d)
        self.assertEqual([s["stage"] for s in d["curriculum"]], [1, 2, 3, 4])
        self.assertEqual(d["review"]["queue"], [])
        self.assertEqual(d["trajectories"]["by_actor"]["agent"]["blocked"], 1)
        self.assertEqual(d["projects"]["items"][0]["why"], "no saved solution")
        self.assertIsInstance(d["training"]["runs"], list)
        self.assertGreater(d["critic_gate"]["experiments"], 0)

    def test_pack_usage_route(self):
        projects, drafts, _ = open_stores(self.home)
        drafts.create("Detect pedestrians in camera frames", pack="vision")
        drafts.create("Something")
        p = projects.create("Frames")
        p["draft"] = {"id": "x", "pack": {"key": "vision", "source": "user"}}
        projects.save(p)
        projects.create("Direct project")
        r = self.client.get("/api/learn/packs")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(set(d["packs"]), {p["key"] for p in PACKS})
        self.assertEqual((d["packs"]["vision"]["drafts"], d["packs"]["vision"]["open_drafts"], d["packs"]["vision"]["projects"]), (1, 1, 1))
        self.assertEqual(d["packs"]["vision"]["project_list"][0]["name"], "Frames")
        self.assertEqual((d["drafts"], d["projects"], d["drafts_without_pack"], d["projects_without_pack"]), (2, 2, 1, 1))
        self.assertEqual(d["packs"]["tabular"]["projects"], 0)


if __name__ == "__main__":
    unittest.main()
