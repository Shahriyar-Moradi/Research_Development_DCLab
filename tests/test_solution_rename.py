"""The prediction contract is now called the solution; old project files, names and routes keep working."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio import ProjectStore, graph  # noqa: E402
from dclab_rnd.studio.store import migrate  # noqa: E402
from dclab_rnd.intern.tools import Toolbox  # noqa: E402

OLD = {"target": "y", "task": "binary", "positive_label": "yes", "prediction_moment": "Right before the marketing call is placed.",
       "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": None, "notes": ""}


def old_project(pid: str) -> dict:
    return {"id": pid, "name": "old", "industry": "Banking", "goal": "", "data": None, "contract": dict(OLD), "proposal": None,
            "settings": {"max_rows": 20000, "quick": False}, "stages": {}, "decisions": {}, "holdout_uses": 0, "running": None,
            "policy": {"require_contract_signoff": True},
            "signoff": {"gate": "contract", "contract_hash": "abc"},
            "approvals": [{"gate": "contract", "contract_hash": "abc"}, {"gate": "holdout"}]}


class MigrationTests(unittest.TestCase):
    def test_migrate_renames_every_old_key(self):
        p = old_project("p1")
        self.assertTrue(migrate(p))
        self.assertNotIn("contract", p)
        self.assertEqual(p["solution"]["target"], "y")
        self.assertEqual(p["policy"], {"require_solution_signoff": True})
        self.assertEqual(p["signoff"], {"gate": "solution", "solution_hash": "abc"})
        self.assertEqual(p["approvals"][0], {"gate": "solution", "solution_hash": "abc"})
        self.assertFalse(migrate(p))  # idempotent

    def test_store_rewrites_old_files_on_read(self):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("x", "Banking", "")["id"]
        path = store.directory(pid) / "project.json"
        path.write_text(json.dumps(old_project(pid)), encoding="utf-8")
        self.assertEqual(store.get(pid)["solution"]["target"], "y")
        self.assertNotIn('"contract"', path.read_text(encoding="utf-8"))
        self.assertIn("solution", store.list()[0])


class LegacyNameTests(unittest.TestCase):
    def test_old_move_gate_and_policy_names(self):
        p = old_project("p2")
        migrate(p)
        self.assertEqual(graph.check(p, "set_contract", "human", target="y").move, "set_solution")
        self.assertTrue(graph.policy({"policy": {"require_contract_signoff": True}})["require_solution_signoff"])
        verdict = graph.check(p, "approve_gate", "human", gate="contract")
        self.assertEqual(verdict.args.get("gate"), "solution")

    def test_old_tool_names_reach_the_new_tools(self):
        box = Toolbox(ProjectStore(Path(tempfile.mkdtemp())))
        self.assertIn("set_solution", box.names())
        self.assertNotIn("set_contract", box.names())
        self.assertIn("propose_solution", box.call("propose_contract", {})["error"])


class RouteTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.headers = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_old_and_new_solution_routes(self):
        c = self.client
        pid = c.post("/api/projects", json={"name": "Bank", "industry": "Banking", "goal": "who subscribes"}, headers=self.headers).json()["id"]
        sample = c.post(f"/api/projects/{pid}/data/sample", json={"key": "bank_marketing"}, headers=self.headers)
        self.assertEqual(sample.status_code, 200, sample.text)
        target = sample.json()["suggestion"]["target"]
        body = {**OLD, "target": target, "positive_label": None, "task": "binary"}
        for route in ("solution", "contract"):  # "contract" is the old name, kept for one release
            r = c.put(f"/api/projects/{pid}/{route}", json=body, headers=self.headers)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["solution"]["target"], target)
            self.assertNotIn("contract", r.json())


if __name__ == "__main__":
    unittest.main()
