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
            "production_approved": False}
    (out / "model.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    for name in ("scorer.py", "app.py"):
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
        "# the model, the code and the tests: no data file is copied\nCOPY model.pkl model.json scorer.py app.py ./\nCOPY vendor ./vendor\n"
        "COPY tests ./tests\n# the hashing key is given at run time: docker run -e DCLAB_SERVICE_KEY=... (it is not in the image)\n"
        "EXPOSE 8000\nCMD [\"gunicorn\", \"--bind\", \"0.0.0.0:8000\", \"--workers\", \"2\", \"app:app\"]\n", encoding="utf-8")
    (out / ".dockerignore").write_text("*.csv\n*.parquet\n*.xlsx\n*.json.gz\n__pycache__\n", encoding="utf-8")
    rows = _made_up_rows(project["data"].get("profile") or {}, meta["input_columns"])
    (out / "tests" / "test_service.py").write_text(
        '"""Generated with the service: made-up rows from the profile (never a real value)."""\n\nimport io\nimport sys\nfrom pathlib import Path\n\n'
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\nfrom app import app  # noqa: E402\n\n"
        f"ROWS = {json.dumps(rows)}\n\n\n"
        "def test_health_and_model():\n    client = app.test_client()\n    assert client.get('/health').json == {'status': 'ok'}\n"
        "    assert client.get('/model').json['task'] == " + repr(solution.task) + "\n\n\n"
        "def test_one_and_many_rows():\n    client = app.test_client()\n    one = client.post('/predict', json=ROWS[0]).json['predictions']\n"
        "    many = client.post('/predict', json=ROWS).json['predictions']\n    assert len(one) == 1 and len(many) == len(ROWS)\n\n\n"
        "def test_a_csv():\n    import pandas as pd\n    client = app.test_client()\n    data = pd.DataFrame(ROWS).to_csv(index=False).encode()\n"
        "    response = client.post('/predict/csv', data={'file': (io.BytesIO(data), 'rows.csv')})\n"
        "    assert len(response.json['predictions']) == len(ROWS)\n", encoding="utf-8")
    (out / "README.md").write_text(
        f"# Scoring service: {project['name']}\n\nExported by DCLab from the approved final stage: recipe `{recipe_name}`, model `{family}`. "
        "No data is in this folder, and the key that hashes category values is not either: it was given to whoever exported "
        "the service, who passes it as `DCLAB_SERVICE_KEY`.\n\n```bash\ndocker build -t dclab-service .\n"
        "docker run -e DCLAB_SERVICE_KEY=... -p 8000:8000 dclab-service\n```\n\n"
        "Endpoints: `GET /health`, `GET /model`, `POST /predict` (one row as an object, or a list), `POST /predict/csv` (form field `file`).\n\n"
        f"{meta['batches'].capitalize()}. A score is evidence about the method, not production approval.\n", encoding="utf-8")
    store.log(project_id, "service_exported", {"recipe": recipe_name, "model": family, "folder": out.name})
    return {"folder": str(out), "meta": meta, "key": key}
