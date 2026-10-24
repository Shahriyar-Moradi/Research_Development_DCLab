"""Export a finished project as a scoring service (package 15.10).

Next to the notebook, a small package a team can run: the fitted feature builder and model (``model.pkl``, with no
row of data in it), a Flask app served by Gunicorn (health, the model, predictions for one row, many rows or a CSV), a
Dockerfile that copies no data, and tests on made-up rows built from the profile (never a real value). The fitted
objects come from DCLab's own modules, which are vendored as they were, so the service computes exactly the features
the final stage computed; a test in the repository checks that its predictions equal the engine's.

The project's final stage must be approved: a service is a promise, and an unapproved result is not one. With the
behaviour recipe, a request's rows are counted together, as the audit window was in training; the service keeps no
state between requests (``model.json`` says so).
"""

from __future__ import annotations

import json
import pickle
import secrets
import shutil
import types
from importlib import metadata

import numpy as np
from pathlib import Path
from typing import Any

from ..expansion import runner as rn
from . import engine
from .data import column_kind

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "service_template"
VENDOR = {"dclab_rnd/provenance.py": "dclab_rnd/provenance.py", "dclab_rnd/expansion/__init__.py": "dclab_rnd/expansion/__init__.py",
          "dclab_rnd/expansion/datasets.py": "dclab_rnd/expansion/datasets.py", "dclab_rnd/expansion/runner.py": "dclab_rnd/expansion/runner.py",
          "dclab_rnd/expansion/detectors.py": "dclab_rnd/expansion/detectors.py"}
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "pydantic", "lightgbm", "xgboost")


class ServiceRefused(ValueError):
    """The project cannot be exported as a service yet; the reason says why."""


def _strip(recipe: Any, salt: str) -> tuple[Any, list[str]]:
    """The fitted feature builder without any data: its bundle replaced by the spec alone; the behaviour features'
    per-row training rates and last frame dropped (counts and means come from the scored rows); and every category
    level, feature name built from one, entity value and place under a salted hash, which the scorer applies to its
    input the same way. Returns the recipe and the categorical columns the scorer hashes."""
    import hashlib

    h = lambda v: "h:" + hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:16]  # noqa: E731
    recipe.bundle = types.SimpleNamespace(spec=recipe.bundle.spec, task_type=recipe.bundle.task_type)
    hashed: list[str] = []
    encoder = getattr(recipe, "encoder", None)
    if encoder is not None:
        renamed = {}
        for column, levels in encoder.categories.items():
            for level in levels:
                renamed[f"{column}={level}"] = f"{column}={h(level)}"
            encoder.categories[column] = [h(level) for level in levels]
            hashed.append(column)
        recipe.names = [renamed.get(name, name) for name in recipe.names]
    entity = getattr(recipe, "entity_model", None)
    if entity is not None:
        entity.loo, entity.fit_key = {}, None
        entity.__dict__.pop("current", None)
        entity.hide_keys(salt)
        entity.salt = None  # the key is not kept: the scorer sets it at start from DCLAB_SERVICE_KEY
    return recipe, hashed


def _made_up_rows(profile: dict[str, Any], columns: list[str], n: int = 3) -> list[dict[str, Any]]:
    """Rows for the generated tests, from the profile's kinds only: numbers 0, 1, 2; text 'a', 'b', 'c'; dates in 2020."""
    kinds = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    rows = []
    for i in range(n):
        row = {}
        for column in columns:
            kind = kinds.get(column)
            row[column] = (float(i) if kind in ("numeric", "binary") else f"2020-01-0{i + 1} 10:00:00" if kind == "datetime"
                           else "abc"[i % 3])
        rows.append(row)
    return rows


def export(store, project_id: str, out: Path) -> dict[str, Any]:
    """Write the service package to ``out`` and return its description; refuses an unapproved project."""
    project = store.get(project_id)
    if (project.get("stages", {}).get("final") or {}).get("status") != "approved":
        raise ServiceRefused("Approve the final stage first: a service is built only from an approved result")
    final_record = store.read_stage(project_id, "final")
    final = final_record["evidence"]
    recipe_name, family, params = final["feature_recipe"], final["model"], final.get("selected_model_params") or None
    p = engine.prepare(store, project)
    if not p.kind.export.get("service", True):  # 16.1: the kind says what its export supports (a kind's own machine has no bundle)
        raise ServiceRefused(f"The {p.kind.title.lower()} kind has no service export yet: {p.kind.export.get('why', 'not built')}")
    bundle = p.bundle
    if bundle.task_type == "text_tabular_binary":
        raise ServiceRefused("A text project's vocabulary is learned from its rows: its service export is not built yet")
    fitted = rn.fit_and_predict(bundle, recipe_name, family, params, bundle.X_train, bundle.y_train, bundle.X_test)
    metric = final_record.get("primary_metric") or bundle.spec.primary_metric
    recorded = (final.get("holdout_metrics") or {}).get(metric)
    refit = rn.compute_metrics(bundle.task_type, bundle.y_test, fitted["predictions"], n_classes=rn._n_classes(bundle),
                               alert_budget=bundle.spec.settings.get("alert_budget")).get(metric)
    if recorded is None or refit is None or abs(float(refit) - float(recorded)) > 1e-6:
        raise ServiceRefused(f"The refit does not reproduce the approved holdout {metric} ({refit} against {recorded}): "
                             "the service would not be the approved model")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tests").mkdir(parents=True)
    key = secrets.token_hex(16)  # the hashing key never goes into the folder: the service reads it at start (DCLAB_SERVICE_KEY)
    recipe, hashed = _strip(fitted["recipe"], key)
    y_train = bundle.y_train.to_numpy(dtype=float) if bundle.task_type == "timeseries_regression" else None
    with open(out / "model.pkl", "wb") as handle:
        pickle.dump({"recipe": recipe, "model": fitted["model"], "hashed_columns": hashed}, handle)
    solution = p.solution
    time = solution.time_column
    # an alert only at a budget the person stated; otherwise the score alone (the plan leaves the threshold to the person)
    threshold = ((final.get("task_assessment") or {}).get("threshold") or {}).get("value") if solution.alert_budget else None
    behaviour = bool(getattr(fitted["recipe"], "behaviour", None) or getattr(fitted["recipe"], "programs", None))
    meta = {"project": project["name"], "task": solution.task, "task_type": bundle.task_type, "target": solution.target,
            "recipe": recipe_name, "model": family, "input_columns": [str(c) for c in bundle.X.columns],
            "input_types": {str(c): str(t) for c, t in bundle.X.dtypes.items()},
            "review_change": {k: v for k, v in (params or {}).items() if str(k).startswith("review__")},
            "target_range": [float(np.min(y_train)), float(np.max(y_train))] if y_train is not None else None,
            "behaviour_within_batch": behaviour,
            "time_column": time, "time_is_datetime": bool(time and column_kind(p.frame[time]) == "datetime") if time else False,
            "class_labels": list(p.class_labels) if p.class_labels else None, "n_classes": rn._n_classes(bundle),
            "positive_label": solution.positive_label, "threshold": threshold, "prediction_moment": solution.prediction_moment,
            "holdout": {k: final["holdout_metrics"].get(k) for k in (bundle.spec.primary_metric,)},
            "batches": ("behaviour features are computed within each request, as they were within each fold and the holdout in "
                        "training: send a whole audit window (or a batch in time order) together; a single row gets no history, "
                        "and no state is kept between requests" if behaviour else "each row is scored on its own"),
            "production_approved": False,
            # 34.9: the log's model and version (a fingerprint of model.pkl), and the label an outcome that is not positive has
            "model_id": project_id, "model_version": _fingerprint(out / "model.pkl"),
            "negative_label": _negative_label(p.frame, solution) if solution.task == "binary" else None}
    calibrator = (final.get("calibration_fit") or {}).get("calibrator")
    if calibrator and calibrator.get("method") != "none":  # 30.15: fitted on the out-of-fold scores; served beside the score, never instead of it
        meta["calibrator"] = calibrator
    watch = final.get("monitoring") or {}
    if watch.get("columns"):  # 30.15 (step 12): the training rows' bins and the alert lines, read by Scorer.monitor
        (out / "monitoring.json").write_text(json.dumps(_hashed_plan(watch, key), indent=1, default=str) + "\n", encoding="utf-8")
        meta["monitoring"] = "monitoring.json"  # its category levels hashed with the key, as the model's are: no value of the table
    (out / "model.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    for name in ("scorer.py", "app.py", "serve_log.py"):
        shutil.copyfile(TEMPLATE / name, out / name)
    vendor = out / "vendor"
    for source, target in VENDOR.items():
        (vendor / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, vendor / target)
    (vendor / "dclab_rnd" / "__init__.py").write_text('"""The parts of DCLab R&D a fitted service needs."""\n', encoding="utf-8")
    shutil.copytree(ROOT / "dclab_core", vendor / "dclab_core", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "replicas", "schemas"))
    pins = []
    for package in PACKAGES:
        try:
            pins.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    (out / "requirements.txt").write_text("\n".join(pins + ["flask>=3.0", "gunicorn>=22.0", "pytest>=8.0"]) + "\n", encoding="utf-8")
    (out / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /service\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n"
        "# the model, the code and the tests: no data file is copied\nCOPY model.pkl model.json scorer.py app.py serve_log.py ./\n"
        + ("COPY monitoring.json ./\n" if meta.get("monitoring") else "") + "COPY vendor ./vendor\n"
        "COPY tests ./tests\n# the hashing key is given at run time: docker run -e DCLAB_SERVICE_KEY=... (it is not in the image)\n"
        "# the prediction and outcome logs: mount a volume here (-v ...:/logs), or they go with the container\n"
        "ENV DCLAB_SERVICE_LOG_DIR=/logs\n"
        "EXPOSE 8000\nCMD [\"gunicorn\", \"--bind\", \"0.0.0.0:8000\", \"--workers\", \"2\", \"app:app\"]\n", encoding="utf-8")
    (out / ".dockerignore").write_text("*.csv\n*.parquet\n*.xlsx\n*.json.gz\n*.jsonl\n__pycache__\n", encoding="utf-8")
    rows = _made_up_rows(project["data"].get("profile") or {}, meta["input_columns"])
    (out / "tests" / "test_service.py").write_text(
        '"""Generated with the service: made-up rows from the profile (never a real value)."""\n\nimport io\nimport sys\nfrom pathlib import Path\n\n'
        "import pytest\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\nfrom app import app  # noqa: E402\n\n"
        f"ROWS = {json.dumps(rows)}\n\n\n"
        "@pytest.fixture(autouse=True)\ndef _logs(tmp_path, monkeypatch):\n    monkeypatch.setenv('DCLAB_SERVICE_LOG_DIR', str(tmp_path))  # the tests' rows stay out of the real logs\n\n\n"
        "def test_health_and_model():\n    client = app.test_client()\n    assert client.get('/health').json == {'status': 'ok'}\n"
        "    assert client.get('/model').json['task'] == " + repr(solution.task) + "\n\n\n"
        "def test_one_and_many_rows():\n    client = app.test_client()\n    one = client.post('/predict', json=ROWS[0]).json['predictions']\n"
        "    many = client.post('/predict', json=ROWS).json['predictions']\n    assert len(one) == 1 and len(many) == len(ROWS)\n\n\n"
        "def test_a_csv():\n    import pandas as pd\n    client = app.test_client()\n    data = pd.DataFrame(ROWS).to_csv(index=False).encode()\n"
        "    response = client.post('/predict/csv', data={'file': (io.BytesIO(data), 'rows.csv')})\n"
        "    assert len(response.json['predictions']) == len(ROWS)\n\n\n"
        "def test_a_prediction_is_logged_and_its_outcome_joins_it(tmp_path):\n    from dclab_core import serving\n    client = app.test_client()\n"
        "    made = client.post('/predict', json=ROWS).json['predictions']\n"
        f"    outcome = {_made_up_outcome(solution, p.class_labels)!r}\n"
        "    assert client.post('/outcome', json={'key': made[0]['key'], 'outcome': outcome}).json == {'recorded': 1}\n"
        "    joined = serving.join(serving.read_log(tmp_path / 'predictions.jsonl', serving.PredictionRecord),\n"
        "                          serving.read_log(tmp_path / 'outcomes.jsonl', serving.OutcomeRecord))\n"
        "    assert len(joined) == 1\n", encoding="utf-8")
    (out / "README.md").write_text(
        f"# Scoring service: {project['name']}\n\nExported by DCLab from the approved final stage: recipe `{recipe_name}`, model `{family}`. "
        "No data is in this folder, and the key that hashes category values is not either: it was given to whoever exported "
        "the service, who passes it as `DCLAB_SERVICE_KEY`.\n\n```bash\ndocker build -t dclab-service .\n"
        "docker run -e DCLAB_SERVICE_KEY=... -p 8000:8000 dclab-service\n```\n\n"
        "Endpoints: `GET /health`, `GET /model`, `POST /predict` (one row as an object, or a list), `POST /predict/csv` (form field `file`), "
        "`POST /outcome` (the true outcome of a row once it is known).\n\n"
        "Logs: every row scored is a line of `predictions.jsonl` and every outcome a line of `outcomes.jsonl`, beside the model or in "
        "`DCLAB_SERVICE_LOG_DIR` (the image sets `/logs`: mount a volume there). Keys are hashed with `DCLAB_LOG_KEY` (16 characters or more; without it, `DCLAB_SERVICE_KEY`, which changes with each export): "
        "set it once and keep it across exports, so an outcome that arrives after a retrain still joins its prediction. A line holds the model and its version, the row's key "
        "as a salted hash, the score, the decision and the time; never a value of the row. Name the column that identifies a row "
        "(`{\"rows\": [...], \"key\": \"order_id\"}`, or `?key=order_id`, or the form field `key` with a CSV) and post its outcome "
        "later with the same value: `{\"key\": 123, \"outcome\": ..., \"known_at\": \"2026-10-25T08:00:00+00:00\"}` or "
        "`{\"outcomes\": [...]}`; with no key column, each prediction comes back with a key made for it. The two logs join by key "
        "(`dclab_core.serving.join`), which is what monitoring with outcomes and a retrain read.\n\n"
        f"{meta['batches'].capitalize()}. A score is evidence about the method, not production approval.\n"
        + ("\nMonitoring: `Scorer().monitor(rows, scores)` reads a batch against the training rows' bins in `monitoring.json` "
           "(each column's and the score's PSI, the share the rule flags); an alert says the inputs moved away from what the model "
           "was trained on.\n" if meta.get("monitoring") else ""), encoding="utf-8")
    try:
        checked = _reload_check(out, key, bundle.X_test, fitted["predictions"], bundle.task_type)  # 30.15: saved, reloaded, predicted, matched
    except ServiceRefused:
        _forget(store, project_id, "service")  # the earlier export of this project is gone with its folder: it packages nothing
        raise
    store.log(project_id, "service_exported", {"recipe": recipe_name, "model": family, "folder": out.name, "checked": checked})
    from . import graph

    graph.capture(store, project_id, "human")  # WF-10: the export after the final stage, validated and logged as any move
    project = store.get(project_id)
    project["exports"] = {**(project.get("exports") or {}), "service": {"at": _now(), "folder": out.name, "final_run": (final_record.get("run") or {}).get("run"),
                                                                        **checked}}
    store.save(project)
    return {"folder": str(out), "meta": meta, "key": key, "checked": checked}


def _fingerprint(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _negative_label(frame: Any, solution: Any) -> str | None:
    """A binary target's other label (the one that is not positive), so a posted outcome of either is read; None when the
    target has not exactly two labels or the positive one is not among them."""
    target, positive = solution.target, solution.positive_label
    if positive is None or target not in getattr(frame, "columns", []):
        return None
    from dclab_core.serving import _text

    labels = {_text(v) for v in frame[target].dropna().unique()}
    others = labels - {_text(positive)}
    return others.pop() if len(labels) == 2 and len(others) == 1 else None


def _made_up_outcome(solution: Any, class_labels: Any) -> Any:
    """An outcome for the generated test: the positive label, the first class, or a number (never a row's value)."""
    if solution.task == "binary":
        return solution.positive_label if solution.positive_label is not None else 1
    if solution.task == "multiclass" and class_labels:
        return str(list(class_labels)[0])
    return 1.0


def _forget(store: Any, project_id: str, which: str) -> None:
    project = store.get(project_id)
    if (project.get("exports") or {}).pop(which, None) is not None:
        store.save(project)


def _hashed_plan(plan: dict[str, Any], key: str) -> dict[str, Any]:
    """The monitoring plan as the service keeps it: each category level under the salted hash the scorer gives its inputs
    (the bins' shares unchanged); a numeric column's edges are already midpoints between training values."""
    import hashlib

    from dclab_core.monitoring import MISSING, OTHER

    def h(v: str) -> str:
        return v if v in (OTHER, MISSING) else "h:" + hashlib.sha256(f"{key}|{v}".encode()).hexdigest()[:16]

    out = json.loads(json.dumps(plan, default=str))
    for column in out.get("columns") or []:
        if column.get("kind") == "category":
            column["categories"] = [h(str(v)) for v in column.get("categories") or []]
    out["hashed_levels"] = True
    return out


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reload_check(out: Path, key: str, rows: Any, expected: Any, task_type: str) -> dict[str, Any]:
    """30.15 (step 10): the exported folder loaded in a Python that cannot import the repository (only the folder and its
    vendored code), the rows the engine's refit scored predicted again, and the scores matched; a mismatch refuses the
    export and removes the folder. The rows go through a file outside the folder, deleted after."""
    import subprocess
    import sys
    import tempfile

    scratch = Path(tempfile.mkdtemp())
    try:
        rows_file = scratch / "rows.json"
        rows.to_json(rows_file, orient="records", date_format="iso", double_precision=15)
        field = "probabilities" if task_type == "multiclass" else ("score" if task_type in ("binary_imbalanced",) else "prediction")
        import os

        script = ("import json, sys; sys.path.insert(0, sys.argv[1]); from scorer import Scorer; s = Scorer(); "
                  "rows = json.load(open(sys.argv[2])); out = s.predict(rows); "
                  f"print(json.dumps([list(r['{field}'].values()) if isinstance(r.get('{field}'), dict) else r.get('{field}', r.get('score', r.get('prediction'))) for r in out]))")
        try:  # the key by the environment, never on the command line (a process list shows it)
            done = subprocess.run([sys.executable, "-I", "-c", script, str(out), str(rows_file)], cwd=str(scratch), capture_output=True, text=True,
                                  timeout=900, env={**os.environ, "DCLAB_SERVICE_KEY": key})
            got = np.asarray(json.loads(done.stdout.strip().splitlines()[-1]), dtype=float) if done.returncode == 0 else None
        except (subprocess.TimeoutExpired, ValueError, IndexError) as error:
            shutil.rmtree(out, ignore_errors=True)
            raise ServiceRefused(f"The exported service could not be checked ({type(error).__name__}): not exported") from None
        if got is None:
            shutil.rmtree(out, ignore_errors=True)
            raise ServiceRefused("The exported service did not load and predict on its own: " + done.stderr.strip()[-600:])
        want = np.asarray(expected, dtype=float)
        diff = float(np.max(np.abs(got - want))) if got.shape == want.shape and got.size else float("inf")
        if not diff <= 1e-6:
            shutil.rmtree(out, ignore_errors=True)
            raise ServiceRefused(f"The exported service does not predict what the engine's refit did (largest difference {diff}): not exported")
        return {"checked": True, "rows": int(len(want)), "max_abs_diff": diff, "how": "reloaded in an isolated Python, the refit's rows predicted again"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
