"""Package 20.9, code for each step: a project's pipeline written as code, one block per step, that computes its
holdout number again from the data it was given. Here on three kinds in this interpreter (isolated, from an unrelated
folder); ``make code-check`` runs every kind the engine runs in a clean Python 3.12."""

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import kinds
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import code, engine
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


def ran(key):
    store, pid = acceptance.build(kinds.get(key).case(0), Path(tempfile.mkdtemp()))
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    return store, pid


def run(folder):
    out = subprocess.run([sys.executable, "-I", str(folder / "pipeline.py")], capture_output=True, text=True, cwd=tempfile.mkdtemp(), timeout=900)
    return out.returncode, json.loads((out.stdout.strip().splitlines() or ["{}"])[-1]) if out.stdout.strip() else {}, out.stderr[-1500:]


@unittest.skipIf(MISSING, MISSING)
class CodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projects = {key: ran(key) for key in ("binary", "anomaly", "embedded")}
        cls.folders = {}
        for key, (store, pid) in cls.projects.items():
            cls.folders[key] = Path(tempfile.mkdtemp()) / "code"
            code.export(store, pid, cls.folders[key])

    def test_the_exported_code_reproduces_the_holdout_number(self):
        for key, folder in self.folders.items():
            status, out, err = run(folder)
            self.assertEqual(status, 0, (key, out, err))
            self.assertTrue(out["same"], (key, out))
            store, pid = self.projects[key]
            final = store.read_stage(pid, "final")
            self.assertEqual(out["expected"], final["evidence"]["holdout_metrics"][final["primary_metric"]])

    def test_one_block_per_step_in_the_module_and_the_notebook(self):
        for key, folder in self.folders.items():
            source = (folder / "pipeline.py").read_text()
            for step in ("data", "models", "final"):
                self.assertIn(f"# ---- {step}:", source, key)
            nb = json.loads((folder / "pipeline.ipynb").read_text())
            self.assertGreaterEqual(sum(c["cell_type"] == "code" for c in nb["cells"]), 5, key)
        self.assertIn("anomaly.choose(", (self.folders["anomaly"] / "pipeline.py").read_text())  # the choice made again, in the library
        expected = json.loads((self.folders["binary"] / "expected.json").read_text())
        self.assertEqual(expected["engine_profile"], "fast")  # the project ran with the engine's small model sizes, and so does its code
        self.assertIn('rn._PROFILE["fast"] = True', (self.folders["binary"] / "pipeline.py").read_text())

    def test_the_notebooks_cells_run_in_order_and_reproduce_it(self):
        folder = self.folders["embedded"]
        nb = json.loads((folder / "pipeline.ipynb").read_text())
        script = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        out = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, cwd=folder, timeout=900)
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        self.assertTrue(json.loads(out.stdout.strip().splitlines()[-1])["same"])

    def test_a_number_that_differs_from_the_record_fails_the_run(self):
        folder = Path(tempfile.mkdtemp()) / "code"
        store, pid = self.projects["binary"]
        code.export(store, pid, folder)
        expected = json.loads((folder / "expected.json").read_text())
        (folder / "expected.json").write_text(json.dumps({**expected, "value": expected["value"] + 0.01}))
        status, out, _ = run(folder)
        self.assertEqual(status, 1)
        self.assertFalse(out["same"])

    def test_the_code_reads_only_what_the_folder_holds(self):
        folder = self.folders["anomaly"]
        self.assertTrue((folder / "vendor" / "dclab_core").is_dir())
        self.assertFalse((folder / "vendor" / "dclab_rnd").exists())  # a kind with its own machine needs only dclab_core
        self.assertTrue((self.folders["binary"] / "vendor" / "dclab_rnd" / "expansion" / "runner.py").exists())
        reqs = (folder / "requirements.txt").read_text()
        self.assertIn("scikit-learn==", reqs)
        self.assertNotIn("flask", reqs)

    def test_a_table_stored_as_parquet_is_read_as_it_was(self):
        import dataclasses

        case = kinds.get("anomaly").case(0)

        def files(directory):
            case.table.to_parquet(directory / "table.parquet", index=False)
            return "table.parquet", None

        store, pid = acceptance.build(dataclasses.replace(case, files=files), Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, folder)
        self.assertIn("pyarrow", (folder / "requirements.txt").read_text())
        status, out, err = run(folder)
        self.assertEqual((status, out.get("same")), (0, True), err)

    def test_a_persons_choice_is_scored_and_the_rules_pick_checked(self):
        store, pid = acceptance.build(kinds.get("anomaly").case(0), Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            for stage in ("data", "leakage", "features", "models"):
                engine.execute(store, pid, stage)
            record = store.read_stage(pid, "models")
            other = next(o["id"] for o in record["decision"]["options"] if o["id"] != record["decision"]["chosen"])
            engine.approve(store, pid, "models", choice=other)
            engine.execute(store, pid, "final")
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, folder)
        source = (folder / "pipeline.py").read_text()
        self.assertIn(f"a person chose {other!r}", source)
        status, out, err = run(folder)
        self.assertEqual((status, out.get("same")), (0, True), err)

    def test_rows_in_neither_part_stay_out_of_training(self):
        import dataclasses

        case = kinds.get("binary").case(0)
        table = case.table.copy()
        table["household"] = [f"h{i // 3}" for i in range(len(table))]
        table.loc[table.index[-40:], "household"] = "h0"  # late rows of a household seen early: in neither part
        case = dataclasses.replace(case, table=table, solution={**case.solution, "group_column": "household"})
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, folder)
        import pandas as pd

        parts = pd.read_pickle(folder / "data" / "prepared.pkl")["__part__"].value_counts().to_dict()
        self.assertGreater(parts.get("none", 0), 0)  # a time-and-group split leaves later rows of seen groups in neither part
        status, out, err = run(folder)
        self.assertEqual((status, out.get("same")), (0, True), (parts, err))

    def test_numbers_keep_every_digit(self):
        import dataclasses

        case = kinds.get("binary").case(0)
        table = case.table.copy()
        numeric = [c for c in table.select_dtypes("number").columns if c != case.target][:3]
        table[numeric] = table[numeric] * 1e-11  # values a ten-digit text format would round to zero
        store, pid = acceptance.build(dataclasses.replace(case, table=table), Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, folder)
        status, out, err = run(folder)
        self.assertEqual((status, out.get("same")), (0, True), err)

    def test_a_project_without_a_final_stage_is_refused(self):
        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        with self.assertRaises(code.CodeRefused):
            code.export(store, pid, Path(tempfile.mkdtemp()) / "code")

    def test_the_route_gives_a_zip(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        store = client.app.state.services.projects
        _, pid = acceptance.build(kinds.get("embedded").case(0), None, store=store)
        r = client.get(f"/api/projects/{pid}/export/code", headers=h)
        self.assertEqual(r.status_code, 409, r.text)  # no final stage yet
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        r = client.get(f"/api/projects/{pid}/export/code", headers=h)
        self.assertEqual(r.status_code, 200, r.text[:300])
        import io

        names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        self.assertTrue({"pipeline.py", "pipeline.ipynb", "expected.json", "requirements.txt"} <= set(names))
        project = store.get(pid)
        project["name"] = "پروژه یادداشت‌ها"  # a name outside Latin-1 in the download's header
        store.save(project)
        r = client.get(f"/api/projects/{pid}/export/code", headers=h)
        self.assertEqual(r.status_code, 200, r.text[:300])
        self.assertIn("filename*=UTF-8''", r.headers["content-disposition"])
        total, rows = client.app.state.services.audit.query(10, 0, kind="data_export")  # the folder holds data: who took it is kept
        self.assertGreaterEqual(total, 2)
        self.assertTrue(any(name.startswith("data/") for name in rows[0]["detail"]["files"]))


if __name__ == "__main__":
    unittest.main()
