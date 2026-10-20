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


def model_env():
    """A scripted model needs a tier with a key and nothing live: what the other tests of a purpose's data do (the .env's own
    key is blanked by the runner, so a test must bring its own)."""
    from dclab_rnd.models import settings

    blank = ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "DCLAB_LOG_PROMPTS",
             "DCLAB_INTERN_MODEL", "DCLAB_NO_LIVE_MODELS")
    return {**{k: "" for k in blank}, **{f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")},
            "OPENAI_API_KEY": "k"}


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

    def notebook_runs(self):
        runs = [json.loads(line) for p in sorted((self.home / "code_runs").rglob("*.jsonl")) for line in p.read_text().splitlines()]
        return sorted((r for r in runs if r.get("kind") == "notebook_cell"), key=lambda r: str(r.get("at") or r.get("started") or ""))

    def run_cell(self, index):
        return self.client.post(f"/api/projects/{self.pid}/notebook/run", json={"cell": index}, headers=self.headers)

    def test_the_cells_run_up_to_the_holdout_on_the_training_rows_only(self):
        roles = {c.get("role"): i for i, c in enumerate(self.cells) if c.get("role")}
        self.assertIn("holdout", roles)
        self.assertIn("split", roles)
        code = [i for i, c in enumerate(self.cells) if c["type"] == "code" and i < roles["holdout"]]
        before = len(self.notebook_runs())  # another test of this class may have run cells in the same workspace
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
        written = studio_data.load_table(services.kernels.folder(services.projects, self.pid) / name)
        self.assertEqual(len(written), len(prepared.bundle.train_idx))
        whole = studio_data.load_table(studio_data.data_path(services.projects, project))
        holdout = whole.iloc[sorted(int(i) for i in prepared.source_rows[prepared.bundle.test_idx])]
        self.assertEqual(len(written.merge(holdout.drop_duplicates(), how="inner")), 0)  # no holdout row reached the kernel

        kept = self.notebook_runs()[before:]
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

    def test_the_agent_writes_a_cell_it_is_checked_accepted_and_run_with_its_plot(self):
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage

        def reply(code):
            return {"content": json.dumps({"code": code, "explanation": "the folds"}), "tool_calls": [], "usage": {}}

        holdout = "print(pipeline.fit(X_train, y_train).score(X_test, y_test))"
        plot = ("from sklearn.model_selection import cross_val_score\nimport matplotlib.pyplot as plt\n"
                "s = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')\n"
                "plt.plot(range(1, len(s) + 1), s, marker='o'); plt.title('Accuracy per fold'); plt.xlabel('fold'); plt.ylabel('accuracy')\nplt.show()")
        replies = [reply(holdout), reply(holdout), reply("x = (") , reply(plot)]

        class Scripted:
            def complete(self, messages, tools=None, max_tokens=1800, **options):
                return replies.pop(0)

        env = mock.patch.dict(os.environ, model_env())
        env.start()
        self.addCleanup(env.stop)
        services = self.client.app.state.services
        kept = services.gateway
        services.gateway = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Scripted())
        self.addCleanup(setattr, services, "gateway", kept)
        url = f"/api/projects/{self.pid}/notebook"
        self.assertTrue(self.client.get(f"/api/projects/{self.pid}/review").json()["writes_cells"])

        refused = self.client.post(f"{url}/write", json={"request": "score it on the test set"}, headers=self.headers).json()
        self.assertEqual(refused["status"], "refused")  # twice the holdout: never offered
        self.assertTrue(any("X_test" in x for x in refused["problems"]))
        self.assertEqual(self.client.post(f"{url}/cells/{refused['id']}/answer", json={"accepted": True}, headers=self.headers).status_code, 409)

        cell = self.client.post(f"{url}/write", json={"request": "show accuracy plot"}, headers=self.headers).json()
        self.assertEqual(cell["status"], "proposed", cell)  # a cell that did not parse, then a good one
        self.assertEqual(self.client.post(f"{url}/run", json={"cell": cell["id"]}, headers=self.headers).status_code, 409)  # not accepted yet
        accepted = self.client.post(f"{url}/cells/{cell['id']}/answer", json={"accepted": True}, headers=self.headers).json()
        self.assertEqual(accepted["status"], "accepted")
        self.assertIn(cell["id"], [c["id"] for c in self.client.get(f"/api/projects/{self.pid}/review").json()["agent_cells"]])
        self.assertNotIn(refused["id"], [c["id"] for c in self.client.get(f"/api/projects/{self.pid}/review").json()["agent_cells"]])

        for i, c in enumerate(self.cells):  # the notebook's cells first, as the page does on a fresh kernel
            if c["type"] != "code" or c.get("role") == "holdout":
                if c.get("role") == "holdout":
                    break
                continue
            self.assertEqual(self.run_cell(i).json()["status"], "ok")
        ran = self.client.post(f"{url}/run", json={"cell": cell["id"]}, headers=self.headers).json()
        self.assertEqual(ran["status"], "ok", json.dumps(ran)[:2000])
        self.assertTrue(any(o["type"] == "image" for o in ran["outputs"]))  # the plot comes back as a picture

        tampered = [dict(c, code=holdout) if c["id"] == cell["id"] else c for c in services.projects.get(self.pid)["notebook_agent_cells"]]
        services.projects.update(self.pid, notebook_agent_cells=tampered)
        self.assertEqual(self.client.post(f"{url}/run", json={"cell": cell["id"]}, headers=self.headers).status_code, 409)  # checked again before it runs

    def _time_project(self, newest_first=True):
        import numpy as np
        import pandas as pd

        from dclab_rnd.studio import data as studio_data

        store = self.client.app.state.services.projects
        pid = store.create("Days", "general", "Predict the outcome")["id"]
        days = np.arange(600)[::-1] if newest_first else np.arange(600)
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"day": days, "x": rng.normal(size=600), "outcome": rng.integers(0, 2, 600)})
        frame.to_csv(store.data_dir(pid) / "days.csv", index=False)
        studio_data.attach_data(store, pid, "days.csv")
        p = store.get(pid)
        p["solution"] = {**SOLUTION, "target": "outcome", "positive_label": "1", "time_column": "day"}
        store.save(p)
        return store, pid

    def test_a_time_split_keeps_the_latest_rows_out_of_the_kernel_whatever_the_file_order(self):
        from dclab_rnd.studio import kernels

        for newest_first in (True, False):
            store, pid = self._time_project(newest_first)
            table, _ = kernels.training_rows(store, store.get(pid))
            self.assertEqual(len(table), 480)
            self.assertEqual(sorted(table["day"]), list(range(480)), "the kernel holds the earliest 80% by day, never a later one")

    def test_a_kernel_on_an_old_solution_is_restarted_with_the_new_rows(self):
        from dclab_rnd.studio import kernels

        services = self.client.app.state.services
        store, pid = self._time_project()
        project = store.get(pid)
        k = services.kernels._kernel(store, project)
        project["solution"] = {**project["solution"], "time_column": None}  # a random split now: other rows train
        store.save(project)
        k2 = services.kernels._kernel(store, store.get(pid))
        self.assertIsNot(k, k2)
        self.assertFalse(k.manager.is_alive())
        self.assertNotEqual(kernels.fingerprint(project), k.fingerprint)
        services.kernels.restart(pid)

    def test_a_restart_during_a_run_ends_the_run_at_once(self):
        import threading
        import time as clock

        services = self.client.app.state.services
        store, pid = self._time_project()
        kernel = services.kernels._kernel(store, store.get(pid))
        got = {}
        t = threading.Thread(target=lambda: got.update(kernel.execute("import time\ntime.sleep(60)", timeout=120)))
        t.start()
        clock.sleep(2)
        started = clock.time()
        services.kernels.restart(pid)
        t.join(30)
        self.assertFalse(t.is_alive(), "the run kept waiting after its kernel stopped")
        self.assertLess(clock.time() - started, 30)
        self.assertEqual(got["status"], "error")
        self.assertEqual(got["outputs"][-1]["ename"], "KernelStopped")

    def test_overlapping_requests_keep_every_answer_and_accepted_cells_are_never_dropped(self):
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage
        from dclab_rnd.studio import notebook_agent

        services = self.client.app.state.services
        store, pid = self._time_project()
        first = {"id": "a0000000001", "request": "r", "code": "print(1)", "explanation": "", "problems": [], "status": "proposed", "at": "t"}
        store.update(pid, notebook_agent_cells=[first])

        class AcceptsMeanwhile:  # while the model writes, the person accepts the first cell
            def complete(self, messages, tools=None, max_tokens=1800, **options):
                r = self_client.post(f"/api/projects/{pid}/notebook/cells/{first['id']}/answer", json={"accepted": True}, headers=headers)
                assert r.status_code == 200, r.text
                return {"content": json.dumps({"code": "print(2)", "explanation": "two"}), "tool_calls": [], "usage": {}}

        self_client, headers = self.client, self.headers
        env = mock.patch.dict(os.environ, model_env())
        env.start()
        self.addCleanup(env.stop)
        kept = services.gateway
        services.gateway = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: AcceptsMeanwhile())
        self.addCleanup(setattr, services, "gateway", kept)
        self.client.post(f"/api/projects/{pid}/notebook/write", json={"request": "two"}, headers=self.headers)
        cells = store.get(pid)["notebook_agent_cells"]
        self.assertEqual([c["status"] for c in cells], ["accepted", "proposed"])  # the answer given meanwhile is kept

        many = [dict(first, id=f"a{i:010d}", status="refused") for i in range(60)]
        trimmed = notebook_agent.keep([dict(first, status="accepted")] + many)
        self.assertEqual(len(trimmed), notebook_agent.MAX_CELLS)
        self.assertEqual(trimmed[0]["status"], "accepted")

    def test_an_agent_cell_on_a_fresh_kernel_runs_after_the_notebook_and_is_kept_whole(self):
        services = self.client.app.state.services
        services.kernels.restart(self.pid)
        long = {"id": "a00000000ff", "request": "long", "code": "print(len(X_train)); print('x' * 30000)", "explanation": "", "problems": [],
                "status": "accepted", "at": "t"}
        project = services.projects.get(self.pid)
        services.projects.update(self.pid, notebook_agent_cells=list(project.get("notebook_agent_cells") or []) + [long])
        before = len(self.notebook_runs())
        ran = self.client.post(f"/api/projects/{self.pid}/notebook/run", json={"cell": long["id"]}, headers=self.headers).json()
        self.assertEqual(ran["status"], "ok", json.dumps(ran)[:1500])  # X_train exists: the notebook's cells ran first
        self.assertGreater(ran["prepared"], 0)
        self.assertLessEqual(max(len(o["text"]) for o in ran["outputs"] if o["type"] == "text"), 20_000)  # the page gets it cut
        mine = [r for r in self.notebook_runs()[before:] if r.get("cell") == long["id"]]
        self.assertEqual(len(mine), 1)
        self.assertIn("x" * 30000, mine[0]["stdout"])  # the record keeps it whole

    def test_the_kernel_lives_in_the_project_and_goes_with_it(self):
        services = self.client.app.state.services
        store, pid = self._time_project()
        kernel = services.kernels._kernel(store, store.get(pid))
        folder = services.kernels.folder(store, pid)
        self.assertTrue(folder.is_dir())
        self.assertEqual(folder.parent.parent, Path(store.data_dir(pid)).parent)  # inside the project's own folder, under data/
        self.assertEqual(folder.name, "data")
        self.assertEqual(self.client.delete(f"/api/projects/{pid}", headers=self.headers).status_code, 204)
        self.assertFalse(folder.exists())
        self.assertFalse(kernel.manager.is_alive())

    def test_the_store_keeps_a_cells_run_without_its_printed_rows_or_pictures(self):
        from dclab_rnd.store.keepers import standard

        line = {"kind": "notebook_cell", "code": "frame.head()", "stdout": "   a  b\n0  1  2\n1  3  4\n", "pictures": ["iVBORw0KGgo="]}
        other = {"kind": "pipeline", "code": "x", "stdout": "ok"}
        kept = [json.loads(x) for x in standard.code_runs((json.dumps(line) + "\n" + json.dumps(other) + "\n").encode()).decode().splitlines()]
        self.assertEqual(kept[0]["pictures"], 1)
        self.assertNotIn("iVBORw0KGgo=", json.dumps(kept))
        self.assertEqual(kept[0]["code"], "frame.head()")
        self.assertEqual(kept[1], other)

    def test_a_cell_shows_a_chart_as_data_the_page_draws_and_a_bad_one_is_dropped(self):
        from dclab_rnd.studio import kernels

        good = {"kind": "line", "title": "T", "series": [{"name": "a", "pts": [[1, 0.5], [2, 0.7]]}], "xlabel": "x", "ylabel": "y", "diag": True}
        self.assertEqual(kernels.clean_chart(good)["series"][0]["pts"], [[1.0, 0.5], [2.0, 0.7]])
        for bad in ({"kind": "script", "series": []}, {"kind": "line", "series": [{"name": "a", "pts": [[1, float("nan")]]}]},
                    {"kind": "line", "series": [{"name": "a", "pts": [[1, 2]] * (kernels.MAX_POINTS + 1)}]}, {"kind": "bars", "labels": ["a"], "values": [1, 2]}, "chart", None):
            self.assertIsNone(kernels.clean_chart(bad), bad)
        self.assertEqual(kernels.clean_chart({"kind": "bars", "labels": ["a", "b"], "values": [1, 2.5], "title": "x" * 500})["title"], "x" * 80)

        services = self.client.app.state.services
        services.kernels.restart(self.pid)
        code = ("chart_line({'acc': [(1, 0.9), (2, 0.92), (3, 0.91)]}, title='Accuracy per fold', xlabel='fold', ylabel='accuracy')\n"
                "chart_bars(['a', 'b'], [1, 2], title='Bars')\n"
                "from IPython.display import display\n"
                "display({'application/vnd.dclab.chart+json': {'kind': 'line', 'series': [{'pts': [[0, 'x']]}]}}, raw=True)")
        cell = {"id": "a00000000cc", "request": "chart", "code": code, "explanation": "", "problems": [], "status": "accepted", "at": "t"}
        project = services.projects.get(self.pid)
        services.projects.update(self.pid, notebook_agent_cells=list(project.get("notebook_agent_cells") or []) + [cell])
        ran = self.client.post(f"/api/projects/{self.pid}/notebook/run", json={"cell": cell["id"]}, headers=self.headers).json()
        self.assertEqual(ran["status"], "ok", json.dumps(ran)[:1500])
        charts = [o for o in ran["outputs"] if o["type"] == "chart"]
        self.assertEqual([c["spec"]["kind"] for c in charts], ["line", "bars"])
        self.assertTrue(any("could not draw" in o.get("text", "") or "cannot draw" in o.get("text", "") for o in ran["outputs"] if o["type"] == "text"))  # the bad one said so
        self.assertEqual(services.kernels.status(self.pid)["count"] > 0, True)

    def test_a_cell_the_person_writes_is_checked_kept_and_run(self):
        url = f"/api/projects/{self.pid}/notebook"
        empty = self.client.post(f"{url}/cells", json={"code": "  "}, headers=self.headers)
        self.assertEqual(empty.status_code, 422)
        holdout = self.client.post(f"{url}/cells", json={"code": "print(y_test.mean())"}, headers=self.headers)
        self.assertEqual(holdout.status_code, 422)
        self.assertIn("X_test or y_test", holdout.json()["detail"])  # the holdout is read once, by the workflow
        broken = self.client.post(f"{url}/cells", json={"code": "x = ("}, headers=self.headers)
        self.assertEqual(broken.status_code, 422)
        made = self.client.post(f"{url}/cells", json={"code": "print('mine', 6 * 7)"}, headers=self.headers)
        self.assertEqual(made.status_code, 200, made.text)
        cell = made.json()
        self.assertEqual((cell["status"], cell["author"]), ("accepted", "person"))  # no one else proposed it: nothing to accept
        ran = self.client.post(f"{url}/run", json={"cell": cell["id"]}, headers=self.headers).json()
        self.assertEqual(ran["status"], "ok", json.dumps(ran)[:1500])
        self.assertIn("mine 42", "".join(o.get("text", "") for o in ran["outputs"]))
        kept = [c["id"] for c in self.client.get(f"/api/projects/{self.pid}/review").json()["agent_cells"]]
        self.assertIn(cell["id"], kept)

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
