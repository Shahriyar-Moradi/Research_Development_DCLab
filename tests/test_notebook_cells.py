"""Running the project's notebook cells on the Notebook page (the owner, 2026-10-20).

A cell runs in the project's Jupyter kernel: the source the server generates for that index, never code from the page.
The kernel's folder holds the project's table with the engine's training rows only, under the name the notebook reads,
so the cells run as they are and the holdout is never in the kernel; the cell that scores the holdout is refused. Every
run is kept in the workspace's code runs; the kernel's environment has no key, token or database address.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOLUTION = {"task": "binary", "positive_label": None, "prediction_moment": "Right before the marketing call is placed.",
            "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": None, "notes": ""}


class NotebookCellsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jupyter_client  # noqa: F401
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio or Jupyter dependencies not installed: {error}")
        cls.home = Path(tempfile.mkdtemp())
        cls.client = TestClient(create_app(cls.home))
        cls.headers = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        c = cls.client
        cls.pid = c.post("/api/projects", json={"name": "Bank", "industry": "Banking", "goal": "who subscribes"}, headers=cls.headers).json()["id"]
        sample = c.post(f"/api/projects/{cls.pid}/data/sample", json={"key": "bank_marketing"}, headers=cls.headers)
        assert sample.status_code == 200, sample.text
        c.patch(f"/api/projects/{cls.pid}", json={"settings": {"max_rows": 1500}}, headers=cls.headers)  # a small table: fast cells
        saved = c.put(f"/api/projects/{cls.pid}/solution", json={**SOLUTION, "target": sample.json()["suggestion"]["target"]}, headers=cls.headers)
        assert saved.status_code == 200, saved.text
        cls.cells = c.get(f"/api/projects/{cls.pid}/review").json()["cells"]

    @classmethod
    def tearDownClass(cls):
        cls.client.app.state.services.kernels.shutdown_all()

    def run_cell(self, index):
        return self.client.post(f"/api/projects/{self.pid}/notebook/run", json={"cell": index}, headers=self.headers)

    def test_the_cells_run_up_to_the_holdout_on_the_training_rows_only(self):
        roles = {c.get("role"): i for i, c in enumerate(self.cells) if c.get("role")}
        self.assertIn("holdout", roles)
        self.assertIn("split", roles)
        code = [i for i, c in enumerate(self.cells) if c["type"] == "code" and i < roles["holdout"]]
        for i in code:
            r = self.run_cell(i)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["status"], "ok", json.dumps(r.json())[:2000])
        refused = self.run_cell(roles["holdout"])
        self.assertEqual(refused.status_code, 409)
        self.assertIn("read once", refused.json()["detail"])

        from dclab_rnd.studio import data as studio_data, engine, kernels

        services = self.client.app.state.services
        project = services.projects.get(self.pid)
        prepared = engine.prepare(services.projects, project)
        table, name = kernels.training_rows(services.projects, project)
        self.assertEqual(len(table), len(prepared.bundle.train_idx))  # the training rows, and only them
        written = studio_data.load_table(services.kernels.folder(self.pid) / name)
        self.assertEqual(len(written), len(prepared.bundle.train_idx))
        whole = studio_data.load_table(studio_data.data_path(services.projects, project))
        holdout = whole.iloc[sorted(int(i) for i in prepared.source_rows[prepared.bundle.test_idx])]
        self.assertEqual(len(written.merge(holdout.drop_duplicates(), how="inner")), 0)  # no holdout row reached the kernel

        runs = [json.loads(line) for p in (self.home / "code_runs").rglob("*.jsonl") for line in p.read_text().splitlines()]
        kept = [r for r in runs if r.get("kind") == "notebook_cell"]
        self.assertEqual(len(kept), len(code))  # every run kept whole
        self.assertTrue(all(r["project"] == self.pid and r["code"] for r in kept))
        status = self.client.get(f"/api/projects/{self.pid}/notebook/kernel").json()
        self.assertTrue(status["running"])
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/notebook/restart", headers=self.headers).json(), {"restarted": True})
        self.assertFalse(self.client.get(f"/api/projects/{self.pid}/notebook/kernel").json()["running"])

    def test_a_markdown_cell_or_a_wrong_index_is_refused(self):
        markdown = next(i for i, c in enumerate(self.cells) if c["type"] == "markdown")
        self.assertEqual(self.run_cell(markdown).status_code, 409)
        self.assertEqual(self.run_cell(10_000).status_code, 409)
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/notebook/run", json={"cell": "x"}, headers=self.headers).status_code, 422)

    def test_the_kernel_never_gets_a_key_or_a_database_address(self):
        from dclab_rnd.studio import kernels

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "DCLAB_DATABASE_URL": "postgresql://x", "GITHUB_TOKEN": "t",
                                          "HOME": "/home/x"}):
            env = kernels._env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("DCLAB_DATABASE_URL", env)
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertEqual(env["HOME"], "/home/x")


if __name__ == "__main__":
    unittest.main()
