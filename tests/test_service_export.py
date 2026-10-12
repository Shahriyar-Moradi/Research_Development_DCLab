"""Package 15.10: a finished project exported as a scoring service (dclab_rnd/studio/service.py)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd.expansion import runner
    from dclab_rnd.studio import ProjectStore, data as sd, engine, service
    from dclab_rnd.studio.solution import Solution
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class ServiceExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = dfr.generate(seed=0)
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Rides", "delivery", "Find fraudulent rides")
        paths = dfr.save(cls.replica, Path(tempfile.mkdtemp()))
        (store.data_dir(project["id"]) / dfr.RIDE_FILE).write_bytes(paths["rides"].read_bytes())
        sd.attach_data(store, project["id"], dfr.RIDE_FILE)
        (store.data_dir(project["id"]) / dfr.SAMPLE_FILE).write_bytes(paths["sample"].read_bytes())
        project = sd.attach_labels(store, project["id"], dfr.SAMPLE_FILE, target="is_fraud")
        project["solution"] = Solution(**{**project["suggestion"], "prediction_moment": "After delivery, in a daily audit of the day's rides.",
                                          "group_column": "biker_id", "alert_budget": 0.005}).model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile():
            engine.run_all(store, project["id"])
        cls.store, cls.pid = store, project["id"]

    def test_an_unapproved_project_is_refused(self):
        with self.assertRaises(service.ServiceRefused):
            service.export(self.store, self.pid, Path(tempfile.mkdtemp()) / "svc")

    def test_the_service_scores_as_the_engine_did_and_holds_no_data(self):
        engine.approve(self.store, self.pid, "final")
        out = Path(tempfile.mkdtemp()) / "svc"
        with runner.fast_profile():
            described = service.export(self.store, self.pid, out)
            p = engine.prepare(self.store, self.store.get(self.pid))
            final = self.store.read_stage(self.pid, "final")["evidence"]
            expected = runner.fit_and_predict(p.bundle, final["feature_recipe"], final["model"], final.get("selected_model_params") or None,
                                              p.bundle.X_train, p.bundle.y_train, p.bundle.X_test)["predictions"]
        self.assertTrue({"model.pkl", "model.json", "app.py", "scorer.py", "Dockerfile", "requirements.txt", "tests", "vendor"}
                        <= {f.name for f in out.iterdir()})
        rows = Path(tempfile.mkdtemp()) / "rows.json"
        p.bundle.X_test.to_json(rows, orient="records", date_format="iso")
        script = ("import json, sys; sys.path.insert(0, sys.argv[1]); from scorer import Scorer; s = Scorer(key=sys.argv[3]); "
                  "rows = json.load(open(sys.argv[2])); print(json.dumps([r['score'] for r in s.predict(rows)])); "
                  "print(json.dumps(s.predict(rows[:1])[0].get('note')))")
        # a Python that cannot import the repository: only the exported folder and its vendored code (and the key, given apart)
        result = subprocess.run([sys.executable, "-I", "-c", script, str(out), str(rows), described["key"]], cwd=tempfile.mkdtemp(),
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        lines = result.stdout.strip().splitlines()
        scores = np.array(json.loads(lines[-2]))
        self.assertIn("single row has no history", json.loads(lines[-1]))  # said, not hidden
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-9)  # the same features, the same model
        blob = b"".join(f.read_bytes() for f in out.rglob("*") if f.is_file() and "vendor" not in f.parts)
        self.assertNotIn(described["key"].encode(), b"".join(f.read_bytes() for f in out.rglob("*") if f.is_file()))  # the key is not in it
        for value in list(self.replica.rides["order_id"].astype(str).head(30)) + list(self.replica.rides["sender_phone"].dropna().astype(str).head(30)):
            self.assertNotIn(value.encode(), blob)  # no row of the table is in the service
        self.assertEqual(described["meta"]["task"], "binary")
        self.assertEqual(described["meta"]["threshold"], final["task_assessment"]["threshold"]["value"])  # the stated budget's threshold
        recorded = final["holdout_metrics"]["average_precision"]  # the export refuses a refit that does not reproduce the record
        self.assertAlmostEqual(float(runner.binary_metrics(p.bundle.y_test, expected)["average_precision"]), recorded, places=6)
        self.assertIn("no data file is copied", (out / "Dockerfile").read_text())


if __name__ == "__main__":
    unittest.main()
