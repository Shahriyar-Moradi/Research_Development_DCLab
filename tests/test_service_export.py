"""Package 15.10: a finished project exported as a scoring service (dclab_rnd/studio/service.py)."""

import json
import os
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

    def test_every_export_of_a_project_gives_the_same_log_key_kept_where_nothing_copies_it(self):
        """34.9's open item: without a stable key the log's hashes changed with each export, so an outcome posted to a new
        export never joined a prediction the last one logged."""
        from dclab_rnd.testers import NEVER

        if self.store.get(self.pid)["stages"]["final"].get("status") != "approved":
            engine.approve(self.store, self.pid, "final")
        with runner.fast_profile():
            first = service.export(self.store, self.pid, Path(tempfile.mkdtemp()) / "a")
            second = service.export(self.store, self.pid, Path(tempfile.mkdtemp()) / "b")
        # I1: repeat exports of the same approved final run recover the exact artifact and its serving key.
        self.assertEqual(first["key"], second["key"])
        self.assertEqual(first["meta"]["model_version"], second["meta"]["model_version"])
        self.assertNotEqual(first["key"], first["log_key"])
        self.assertEqual(first["log_key"], second["log_key"])  # the log's key stays stable across exports and retrains
        self.assertEqual(len(first["log_key"]), 32)
        self.assertRegex(service.LOG_KEY_FILE.as_posix(), NEVER)  # the store and the testers' kit skip it by its name
        for folder in (Path(first["folder"]), Path(second["folder"])):
            text = "".join(p.read_text(errors="ignore") for p in folder.rglob("*") if p.is_file())
            self.assertNotIn(first["log_key"], text)  # never in the export's folder
        records = "".join(p.read_text(errors="ignore") for p in Path(self.store.directory(self.pid)).rglob("*")
                          if p.is_file() and "secrets" not in p.parts)
        self.assertNotIn(first["log_key"], records)  # nor in the project's records and logs

    def test_the_log_key_needs_no_store_folder_method_and_is_made_for_the_owner_only(self):
        """The review of item 11: the PostgreSQL store has no ``directory``; the key is found through ``export_dir``."""
        import stat

        home = Path(tempfile.mkdtemp())

        class PgLike:  # only what both backends share
            def export_dir(self, project_id):
                path = home / "projects" / project_id / "exports"
                path.mkdir(parents=True, exist_ok=True)
                return path

        first = service.project_log_key(PgLike(), "abcdef012345")
        self.assertEqual(service.project_log_key(PgLike(), "abcdef012345"), first)
        path = home / "projects" / "abcdef012345" / service.LOG_KEY_FILE
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode) & 0o077, 0)
        path.write_text("short")  # a broken key is made again, never used
        self.assertEqual(len(service.project_log_key(PgLike(), "abcdef012345")), 32)

    def test_the_service_logs_every_prediction_takes_outcomes_and_the_logs_join_by_key_without_a_value(self):
        """34.9: a batch scored as JSON and one as CSV, their outcomes posted by the caller's own key (the order id), a
        batch with no key gets keys made for it; the two logs join by key and neither holds a value of the rows."""
        from dclab_core import serving

        if self.store.get(self.pid)["stages"]["final"].get("status") != "approved":
            engine.approve(self.store, self.pid, "final")
        out = Path(tempfile.mkdtemp()) / "svc"
        with runner.fast_profile():
            described = service.export(self.store, self.pid, out)
            p = engine.prepare(self.store, self.store.get(self.pid))
        meta, key = described["meta"], described["key"]
        self.assertEqual((meta["model_id"], len(meta["model_version"])), (self.pid, 16))
        self.assertIn("predictions.jsonl", (out / "README.md").read_text())
        self.assertIn("POST /outcome", (out / "README.md").read_text())
        rows = p.bundle.X_test.head(70).copy()
        rows["order_id"] = p.frame.loc[rows.index, "order_id"].to_numpy()
        labels = p.bundle.y_test.loc[rows.index].tolist()
        scratch = Path(tempfile.mkdtemp())
        (scratch / "rows.json").write_text(rows.to_json(orient="records", date_format="iso"), encoding="utf-8")
        (scratch / "labels.json").write_text(json.dumps([str(v) for v in labels]), encoding="utf-8")
        stub = scratch / "stub"
        stub.mkdir()
        (stub / "flask.py").write_text(FLASK_STUB, encoding="utf-8")  # the routes of app.py, called without Flask installed
        logs = scratch / "logs"
        result = subprocess.run([sys.executable, "-I", "-c", SERVICE_SCRIPT, str(stub), str(out), str(scratch)], cwd=str(scratch),
                                capture_output=True, text=True, timeout=600,
                                env={**os.environ, "DCLAB_SERVICE_KEY": key, "DCLAB_SERVICE_LOG_DIR": str(logs)})
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        said = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(said["statuses"], {"json": 200, "csv": 200, "no_key": 200, "outcomes": 200, "bad_outcome": 400, "no_rows": 400,
                                            "monitor": 200, "bad_window": 400})
        self.assertEqual(said["recorded"], 70 + 1)
        predictions = serving.read_log(logs / "predictions.jsonl", serving.PredictionRecord)  # every line strictly the contract
        outcomes = serving.read_log(logs / "outcomes.jsonl", serving.OutcomeRecord)
        self.assertEqual((len(predictions), len(outcomes)), (40 + 30 + 2, 70 + 1))  # the refused outcome left nothing
        self.assertEqual(len({r.batch for r in predictions}), 3)
        self.assertEqual({(r.model_id, r.model_version, r.task) for r in predictions}, {(self.pid, meta["model_version"], "binary")})
        self.assertEqual([r.score for r in predictions[:40]], said["scores_json"])  # the score logged is the score answered
        joined = serving.join(predictions, outcomes)
        self.assertEqual(len(joined), 70 + 1)  # every keyed row and the made-up key's row met their outcome
        positive = str(meta["positive_label"])
        want = {serving.key_hash(order, key): float(str(label) == positive) for order, label in zip(rows["order_id"], labels)}
        got = dict(zip(joined["key"], joined["outcome"]))
        self.assertEqual({k: got[k] for k in want}, want)
        text = (logs / "predictions.jsonl").read_text() + (logs / "outcomes.jsonl").read_text()
        for column in list(rows.columns):
            values = rows[column].dropna().astype(str)
            for value in set(values if column == "order_id" or rows[column].dtype == object else []):
                if len(value) >= 4:
                    self.assertNotIn(value, text, f"a value of {column} is in a log")
        self.assertNotIn(key, text)
        from dclab_core import live  # 34.10: one profile a batch, the monitor reads the three logs, no value in a profile

        profiles = serving.read_log(logs / "profiles.jsonl", live.BatchProfile)
        self.assertEqual([p.rows for p in profiles], [40, 30])  # the two-row request waits: no profile of fewer than twenty rows
        self.assertTrue(all("batch" not in p.model_dump() for p in profiles))  # linked to no request
        watched = said["monitor"]
        self.assertEqual(watched["schema_version"], "dclab_core.live_report.v1")
        self.assertEqual(len(watched["windows"]), 3)
        self.assertEqual(sum(w["labelled"] for w in watched["windows"]), 70 + 1)
        self.assertEqual(watched["holdout"]["metric"], "roc_auc")
        self.assertEqual(meta["live_reference"]["metric"], "roc_auc")
        shared = (logs / "profiles.jsonl").read_text()
        for column in list(rows.columns):
            for value in set(rows[column].dropna().astype(str)):
                if len(value) >= 4 and not value.replace(".", "").isdigit():
                    self.assertNotIn(value, shared, f"a value of {column} is in a profile")
        self.assertIn("GET /monitor", (out / "README.md").read_text())

    def test_exported_service_ships_version_aware_alert_history(self):
        # Verify the exported service bundle contains and executes the fixed serve_log.py template.
        if not self.store.get(self.pid).get("approvals"):
            engine.approve(self.store, self.pid, "final")
        out = Path(tempfile.mkdtemp()) / "svc"
        with runner.fast_profile():
            service.export(self.store, self.pid, out)
        logs = Path(tempfile.mkdtemp())
        check = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from types import SimpleNamespace\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import serve_log\n"
            "class S:\n"
            "    def __init__(self, v): self.meta={'model_id':'m','model_version':v,'task':'binary'}; self.key='k'\n"
            "rep = SimpleNamespace(window='day', windows=[SimpleNamespace(window='2026-11-19', alerts=['drift'])])\n"
            "serve_log.Log(S('v1'))._alerts(rep)\n"
            "serve_log.Log(S('v2'))._alerts(rep)\n"
            "serve_log.Log(S('v1'))._alerts(rep)\n"
            "rows = [json.loads(l) for l in (Path(sys.argv[2])/'alerts.jsonl').read_text().splitlines() if l.strip()]\n"
            "assert len(rows) == 2 and [r['model_version'] for r in rows] == ['v1', 'v2'], rows\n"
        )
        run = subprocess.run([sys.executable, "-c", check, str(out), str(logs)],
                             env={**os.environ, "DCLAB_SERVICE_LOG_DIR": str(logs)}, capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_export_never_prepares_refits_or_reads_holdout_and_matches_repeat_exports(self):
        from unittest import mock

        if self.store.get(self.pid)["stages"]["final"].get("status") != "approved":
            engine.approve(self.store, self.pid, "final")
        out_a = Path(tempfile.mkdtemp()) / "svc_a"
        out_b = Path(tempfile.mkdtemp()) / "svc_b"

        def forbid(*args, **kwargs):
            raise AssertionError("export must never prepare, refit, score, or read raw/holdout data")

        with mock.patch.object(engine, "prepare", side_effect=forbid), \
                mock.patch.object(runner, "fit_and_predict", side_effect=forbid), \
                mock.patch.object(runner, "compute_metrics", side_effect=forbid), \
                mock.patch.object(sd, "load_table", side_effect=forbid):
            first = service.export(self.store, self.pid, out_a)
            second = service.export(self.store, self.pid, out_b)
        self.assertEqual((out_a / "model.pkl").read_bytes(), (out_b / "model.pkl").read_bytes())
        self.assertEqual(first["meta"]["model_version"], second["meta"]["model_version"])
        self.assertEqual(first["meta"].get("calibrator"), second["meta"].get("calibrator"))
        self.assertEqual(first["meta"].get("live_reference"), second["meta"].get("live_reference"))
        self.assertEqual((out_a / "monitoring.json").read_text(), (out_b / "monitoring.json").read_text())
        self.assertTrue(first["checked"]["checked"] and second["checked"]["checked"])

    def test_tampered_unapproved_legacy_incomplete_or_failing_export_protects_previous_output(self):
        import copy
        from unittest import mock

        from dclab_rnd.storage.files import files_for

        if self.store.get(self.pid)["stages"]["final"].get("status") != "approved":
            engine.approve(self.store, self.pid, "final")
        out = Path(tempfile.mkdtemp()) / "svc_protected"
        good = service.export(self.store, self.pid, out)
        sentinel = (out / "model.pkl").read_bytes()
        original_record = copy.deepcopy(self.store.read_stage(self.pid, "final"))
        original_project = copy.deepcopy(self.store.get(self.pid))
        art = original_record["evidence"]["service_artifact"]
        files = files_for(self.store)
        artifact_path = files.local_path(art["artifact_key"])
        original_bytes = artifact_path.read_bytes()
        try:
            # 1. Corrupted artifact bytes are refused and previous export stays intact.
            artifact_path.write_bytes(original_bytes + b"tampered")
            with self.assertRaises(service.ServiceRefused):
                service.export(self.store, self.pid, out)
            self.assertEqual((out / "model.pkl").read_bytes(), sentinel)
            artifact_path.write_bytes(original_bytes)

            # 2. Swapped approved_artifact digest is refused.
            proj = copy.deepcopy(original_project)
            proj["stages"]["final"]["approved_artifact"]["artifact_sha256"] = "0" * 64
            self.store.save(proj)
            with self.assertRaises(service.ServiceRefused):
                service.export(self.store, self.pid, out)
            self.assertEqual((out / "model.pkl").read_bytes(), sentinel)

            # 3. Revoked approval is refused.
            proj = copy.deepcopy(original_project)
            proj["stages"]["final"]["status"] = "completed"
            self.store.save(proj)
            with self.assertRaises(service.ServiceRefused):
                service.export(self.store, self.pid, out)

            # 4. Write/reload failure during repeat export preserves previous export at out.
            self.store.save(copy.deepcopy(original_project))
            with mock.patch.object(service, "_reload_check", side_effect=service.ServiceRefused("injected reload fail")):
                with self.assertRaises(service.ServiceRefused):
                    service.export(self.store, self.pid, out)
            self.assertEqual((out / "model.pkl").read_bytes(), sentinel)

            # 5. Legacy approved record lacking service_artifact gets actionable refusal without refit.
            legacy = copy.deepcopy(original_record)
            legacy["evidence"].pop("service_artifact", None)
            self.store.write_stage(self.pid, "final", legacy)
            proj = copy.deepcopy(original_project)
            proj["stages"]["final"].pop("approved_artifact", None)
            self.store.save(proj)
            with mock.patch.object(engine, "prepare", side_effect=AssertionError("legacy export must not refit")):
                with self.assertRaises(service.ServiceRefused) as ctx:
                    service.export(self.store, self.pid, out)
            self.assertIn("Approved artifact unavailable", str(ctx.exception))
            self.assertIn(art["run_id"], str(ctx.exception))

            # 6. Incomplete capture is never described as exportable.
            incomplete = copy.deepcopy(original_record)
            incomplete["evidence"]["service_artifact"] = {"schema_version": service.ARTIFACT_SCHEMA, "format_version": 1,
                                                          "run_id": art["run_id"], "complete": False, "error": "disk full"}
            self.store.write_stage(self.pid, "final", incomplete)
            with self.assertRaises(service.ServiceRefused) as ctx2:
                service.export(self.store, self.pid, out)
            self.assertIn("disk full", str(ctx2.exception))
        finally:
            artifact_path.write_bytes(original_bytes)
            self.store.write_stage(self.pid, "final", original_record)
            self.store.save(original_project)
        self.assertEqual(good["meta"]["model_version"], art["model_version"])

    def test_regression_and_multiclass_export_without_refit_and_object_storage_fetch(self):
        from unittest import mock

        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.storage.files import S3Files

        class _FakeBody:
            def __init__(self, data):
                self._data = data

            def read(self, n=-1):
                return self._data if n < 0 else self._data[:n]

        class _FakeS3:
            def __init__(self):
                self.objects = {}

            def put_object(self, Bucket, Key, Body, ContentType="", Metadata=None):
                self.objects[(Bucket, Key)] = Body.read()

            def get_object(self, Bucket, Key):
                data = self.objects[(Bucket, Key)]
                return {"ContentLength": len(data), "Body": _FakeBody(data)}

            def head_object(self, Bucket, Key):
                if (Bucket, Key) not in self.objects:
                    raise KeyError(Key)
                return {"ContentLength": len(self.objects[(Bucket, Key)])}

            def delete_object(self, Bucket, Key):
                self.objects.pop((Bucket, Key), None)

        for key in ("regression", "multiclass"):
            with self.subTest(key):
                root = Path(tempfile.mkdtemp())
                store, pid = acceptance.build(kinds.get(key).case(0), root)
                if key == "regression":
                    store.file_storage = S3Files(root, "s3://test-bucket/prefix", client=_FakeS3())
                with runner.fast_profile():
                    engine.run_all(store, pid)
                engine.approve(store, pid, "final")
                if key == "regression":
                    # Remove local cached artifact so export must fetch and verify bytes from object storage.
                    art = store.read_stage(pid, "final")["evidence"]["service_artifact"]
                    store.file_storage.local_path(art["artifact_key"]).unlink()
                    self.assertFalse(store.file_storage.local_path(art["artifact_key"]).exists())
                with mock.patch.object(engine, "prepare", side_effect=AssertionError("must not prepare")), \
                        mock.patch.object(runner, "fit_and_predict", side_effect=AssertionError("must not refit")), \
                        mock.patch.object(sd, "load_table", side_effect=AssertionError("must not read data")):
                    out = Path(tempfile.mkdtemp()) / f"svc_{key}"
                    done = service.export(store, pid, out)
                self.assertTrue(done["checked"]["checked"])
                self.assertEqual(done["meta"]["task"], key)


FLASK_STUB = '''
"""A stand-in for Flask with the parts app.py uses: routes kept by method and path, one request at a time."""


class _Upload:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _Request:
    def __init__(self):
        self.set()

    def set(self, json=None, args=None, files=None, form=None):
        self._json, self.args, self.form = json, dict(args or {}), dict(form or {})
        self.files = {k: _Upload(v) for k, v in (files or {}).items()}

    def get_json(self, force=False, silent=False):
        return self._json


request = _Request()


def jsonify(obj=None, **kwargs):
    return obj if obj is not None else kwargs


class Flask:
    def __init__(self, name):
        self.routes = {}

    def _route(self, method, path):
        def keep(function):
            self.routes[(method, path)] = function
            return function
        return keep

    def get(self, path):
        return self._route("GET", path)

    def post(self, path):
        return self._route("POST", path)
'''

SERVICE_SCRIPT = r'''
import io, json, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import flask, app
import pandas as pd

scratch = Path(sys.argv[3])
rows = json.loads((scratch / "rows.json").read_text())
labels = json.loads((scratch / "labels.json").read_text())


def call(method, path, **request):
    flask.request.set(**request)
    out = app.app.routes[(method, path)]()
    return out if isinstance(out, tuple) else (out, 200)


statuses = {}
body, statuses["json"] = call("POST", "/predict", json={"rows": rows[:40], "key": "order_id"})
scores = [p["score"] for p in body["predictions"]]
assert [p["key"] for p in body["predictions"]] == [str(r["order_id"]) for r in rows[:40]], body["predictions"][:2]
csv = pd.DataFrame(rows[40:70]).to_csv(index=False).encode()
body, statuses["csv"] = call("POST", "/predict/csv", files={"file": csv}, form={"key": "order_id"})
assert len(body["predictions"]) == 30
body, statuses["no_key"] = call("POST", "/predict", json=rows[:2])
made = body["predictions"][0]["key"]
assert len(made) == 32 and body["batch"]
from datetime import datetime, timezone
known = datetime.now(timezone.utc).isoformat()  # known now: an outcome known later than the reading is not yet read (34.10)
posted = [{"key": r["order_id"], "outcome": label, "known_at": known} for r, label in zip(rows, labels)]
posted.append({"key": made, "outcome": labels[0]})
body, statuses["outcomes"] = call("POST", "/outcome", json={"outcomes": posted})
recorded = body["recorded"]
_, statuses["bad_outcome"] = call("POST", "/outcome", json={"outcomes": [{"key": "x", "outcome": "maybe"}]})
_, statuses["no_rows"] = call("POST", "/outcome", json={"nothing": 1})
flask.request.set(args={"window": "batch"})
watched, statuses["monitor"] = app.app.routes[("GET", "/monitor")]() if True else None, 200
if isinstance(watched, tuple):
    watched, statuses["monitor"] = watched
flask.request.set(args={"window": "hour"})
bad = app.app.routes[("GET", "/monitor")]()
statuses["bad_window"] = bad[1] if isinstance(bad, tuple) else 200
print(json.dumps({"statuses": statuses, "scores_json": scores, "recorded": recorded, "monitor": watched}))
'''


class _FakeScorer:
    def __init__(self, model_id, model_version, plan=None):
        self.meta = {"model_id": model_id, "model_version": model_version, "task": "binary", "live_reference": None}
        self.key = "test-secret"
        self._plan = plan

    def monitoring_plan(self):
        return self._plan


@unittest.skipIf(MISSING, MISSING)
class AlertHistoryTests(unittest.TestCase):
    """I2 (2026-11-19): version-aware, window-typed, process-safe alert deduplication in serve_log.py."""

    def _report(self, window_type, windows, model_id="m1", model_version="v1"):
        from types import SimpleNamespace

        return SimpleNamespace(
            window=window_type,
            model_id=model_id,
            model_versions=[model_version],
            windows=[SimpleNamespace(window=w, alerts=list(a)) for w, a in windows],
        )

    def _rows(self, folder):
        path = folder / "alerts.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []

    def test_two_versions_keep_distinct_alerts_and_repeat_keeps_first_seen(self):
        from unittest import mock

        from dclab_rnd.studio.service_template import serve_log

        folder = Path(tempfile.mkdtemp())
        log_v1 = serve_log.Log(_FakeScorer("rides", "v1"))
        log_v2 = serve_log.Log(_FakeScorer("rides", "v2"))
        rep1 = self._report("day", [("2026-11-19", ["score shifted (PSI 0.310)"])], "rides", "v1")
        rep2 = self._report("day", [("2026-11-19", ["score shifted (PSI 0.310)"])], "rides", "v2")
        with mock.patch.dict(os.environ, {"DCLAB_SERVICE_LOG_DIR": str(folder)}):
            log_v1._alerts(rep1)
            first_seen_v1 = self._rows(folder)[0]["first_seen"]
            log_v2._alerts(rep2)
            log_v1._alerts(rep1)
        rows = self._rows(folder)
        self.assertEqual(len(rows), 2)
        self.assertEqual([(r["model_id"], r["model_version"], r["window"]) for r in rows],
                         [("rides", "v1", "2026-11-19"), ("rides", "v2", "2026-11-19")])
        self.assertEqual(rows[0]["first_seen"], first_seen_v1)

    def test_different_model_ids_with_same_version_remain_distinct(self):
        from unittest import mock

        from dclab_rnd.studio.service_template import serve_log

        folder = Path(tempfile.mkdtemp())
        a = serve_log.Log(_FakeScorer("model-a", "v1"))
        b = serve_log.Log(_FakeScorer("model-b", "v1"))
        rep = self._report("day", [("2026-11-19", ["distance shifted"])])
        with mock.patch.dict(os.environ, {"DCLAB_SERVICE_LOG_DIR": str(folder)}):
            a._alerts(rep)
            b._alerts(rep)
        rows = self._rows(folder)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["model_id"] for r in rows}, {"model-a", "model-b"})

    def test_window_types_and_in_report_duplicates(self):
        from unittest import mock

        from dclab_rnd.studio.service_template import serve_log

        folder = Path(tempfile.mkdtemp())
        log = serve_log.Log(_FakeScorer("rides", "v1"))
        day_rep = self._report("day", [("2026-11-19", ["alert-x", "alert-x"])])
        batch_rep = self._report("batch", [("2026-11-19", ["alert-x"])])
        with mock.patch.dict(os.environ, {"DCLAB_SERVICE_LOG_DIR": str(folder)}):
            log._alerts(day_rep)
            log._alerts(batch_rep)
        rows = self._rows(folder)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r.get("window_type") for r in rows], ["day", "batch"])

    def test_legacy_unversioned_row_remains_readable_and_does_not_suppress_new_version(self):
        from unittest import mock

        from dclab_rnd.studio.service_template import serve_log

        folder = Path(tempfile.mkdtemp())
        legacy = {"window": "2026-11-19", "alert": "legacy alert", "first_seen": "2026-11-01T00:00:00+00:00"}
        (folder / "alerts.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        log = serve_log.Log(_FakeScorer("rides", "v3"))
        rep = self._report("day", [("2026-11-19", ["legacy alert"])])
        with mock.patch.dict(os.environ, {"DCLAB_SERVICE_LOG_DIR": str(folder)}):
            log._alerts(rep)
        rows = self._rows(folder)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("model_version", rows[0])
        self.assertEqual((rows[1]["model_id"], rows[1]["model_version"]), ("rides", "v3"))

    def test_concurrent_processes_converge_to_one_alert_row(self):
        import time

        folder = Path(tempfile.mkdtemp())
        worker_script = (
            "import os, sys, time\n"
            "from types import SimpleNamespace\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from dclab_rnd.studio.service_template import serve_log\n"
            "class S:\n"
            "    meta = {'model_id': 'rides', 'model_version': 'v1', 'task': 'binary'}\n"
            "    key = 'k'\n"
            "log = serve_log.Log(S())\n"
            "rep = SimpleNamespace(window='day', windows=[SimpleNamespace(window='2026-11-19', alerts=['race-alert'])])\n"
            "start = float(sys.argv[1])\n"
            "while time.time() < start:\n"
            "    time.sleep(0.005)\n"
            "for _ in range(15):\n"
            "    log._alerts(rep)\n"
        )
        env = {**os.environ, "DCLAB_SERVICE_LOG_DIR": str(folder)}
        target = str(time.time() + 0.25)
        procs = [subprocess.Popen([sys.executable, "-c", worker_script, target], env=env) for _ in range(3)]
        for proc in procs:
            self.assertEqual(proc.wait(timeout=20), 0)
        rows = self._rows(folder)
        self.assertEqual(len(rows), 1, f"expected 1 deduplicated alert row across workers, got {len(rows)}")

    def test_monitor_repeat_returns_same_report_and_preserves_first_seen(self):
        from unittest import mock

        from dclab_core import live
        from dclab_rnd.studio.service_template import serve_log

        folder = Path(tempfile.mkdtemp())
        log = serve_log.Log(_FakeScorer("rides", "v1", plan=object()))
        fake_report = live.LiveReport(
            model_id="rides", model_versions=["v1"], window="day", as_of="2026-11-19T00:00:00+00:00",
            windows=[live.WindowReading(window="2026-11-19", start="2026-11-19T00:00:00+00:00",
                                        end="2026-11-19T23:59:59+00:00", rows=50, status="alert",
                                        alerts=["score shifted (PSI 0.300)"])],
            first_alert="2026-11-19", alerts=["2026-11-19: score shifted (PSI 0.300)"], status="alert",
        )
        with mock.patch.dict(os.environ, {"DCLAB_SERVICE_LOG_DIR": str(folder)}), \
                mock.patch.object(live, "monitor_log", return_value=fake_report):
            first = log.monitor(window="day")
            first_seen = self._rows(folder)[0]["first_seen"]
            second = log.monitor(window="day")
        self.assertEqual(first, second)
        rows = self._rows(folder)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["first_seen"], first_seen)


if __name__ == "__main__":
    unittest.main()
