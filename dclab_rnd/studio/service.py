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

import copy
import hashlib
import json
import os
import pickle
import platform
import secrets
import shutil
import types
from importlib import metadata

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from .. import kinds
from ..expansion import runner as rn
from ..storage.files import FileError, files_for
from . import engine
from .data import column_kind
from .na_codes import codes_kept  # "NA" as a code in the service's CSV, as in training (2026-10-25)
from .solution import Solution

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "service_template"
VENDOR = {"dclab_rnd/provenance.py": "dclab_rnd/provenance.py", "dclab_rnd/expansion/__init__.py": "dclab_rnd/expansion/__init__.py",
          "dclab_rnd/expansion/datasets.py": "dclab_rnd/expansion/datasets.py", "dclab_rnd/expansion/runner.py": "dclab_rnd/expansion/runner.py",
          "dclab_rnd/expansion/detectors.py": "dclab_rnd/expansion/detectors.py"}
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "pydantic", "lightgbm", "xgboost")
ARTIFACT_SCHEMA = "dclab.service_artifact.v1"
ARTIFACT_FORMAT = 1
# what an approval binds of the captured artifact (engine.approve) and the export compares, every field (review of 96e47f43)
BIND_FIELDS = ("run_id", "schema_version", "artifact_key", "artifact_sha256", "meta_key", "meta_sha256", "parity_key", "parity_sha256",
               "monitoring_key", "monitoring_sha256", "key_ref")
# the environment the reload check's Python gets: what an interpreter needs, never the server's keys or database address
CHECK_ENV = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "SYSTEMDRIVE")


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


def _package_versions() -> dict[str, str]:
    """The installed version of each package the model's pickle may need (the ones not installed are left out)."""
    found = {}
    for package in PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def approval_binding(art: Any) -> dict[str, Any] | None:
    """What the final stage's approval binds of a captured artifact: every key, digest and the key's reference, or None
    when the capture did not get as far as its digests (nothing to bind; the export then refuses)."""
    if not isinstance(art, dict) or not all(art.get(f) for f in ("run_id", "artifact_sha256", "meta_sha256", "parity_sha256", "key_ref")):
        return None
    return {field: art.get(field) for field in BIND_FIELDS}


def _training_frame(store: Any, project: dict[str, Any], prepared: Any) -> pd.DataFrame:
    """The training rows as the table holds them (the target's labels and the columns' raw values), found by
    ``prepared.source_rows``: each row of the bundle's X is its place in the table as loaded after a label file's join,
    so ``bundle.train_idx`` (positions in X) maps through it. ``prepared.frame`` cannot be indexed so: it lost the rows
    without a target or a time, and a sample or a time split reorders X (review of 96e47f43: 97 holdout rows were read)."""
    from .data import data_path, load_table

    bundle = prepared.bundle
    rows = getattr(prepared, "source_rows", None)
    if rows is None or len(rows) != len(bundle.X):
        raise ValueError("the training rows cannot be traced back to the table (source_rows does not cover the bundle)")
    table = load_table(data_path(store, project))
    solution = prepared.solution
    if project.get("labels") and project["labels"].get("target") == solution.target:  # as prepare joined it
        from . import labels as label_file

        table = label_file.apply(store, project, table)
    return table.iloc[np.asarray(rows)[np.asarray(bundle.train_idx)]].reset_index(drop=True)


def _made_up_rows(profile: dict[str, Any], columns: list[str], n: int = 3) -> list[dict[str, Any]]:
    """Rows for the generated tests, from the profile's kinds only: numbers 0, 1, 2; text 'a', 'b', 'c'; dates in 2020."""
    kinds_map = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    rows = []
    for i in range(n):
        row = {}
        for column in columns:
            kind = kinds_map.get(column)
            row[column] = (float(i) if kind in ("numeric", "binary") else f"2020-01-{i + 1:02d} 10:00:00" if kind == "datetime"
                           else "abc"[i % 3])
        rows.append(row)
    return rows


def _secret_key_path(store: Any, project_id: str, name: str) -> Path:
    if "/" in name or "\\" in name or ".." in name or not name:
        raise ServiceRefused("Invalid serving secret reference")
    return Path(store.export_dir(project_id)).parent / "secrets" / name


def _write_secret_key(store: Any, project_id: str, name: str, value: str) -> None:
    path = _secret_key_path(store, project_id, name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{name}.{secrets.token_hex(4)}.part")
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(value)
    os.replace(tmp, path)


def _read_secret_key(store: Any, project_id: str, name: str) -> str | None:
    path = _secret_key_path(store, project_id, name)
    if not path.is_file():
        return None
    held = path.read_text(encoding="utf-8").strip()
    return held if len(held) >= 32 else None


def _score_stripped(artifact: dict[str, Any], meta: dict[str, Any], key: str, rows: list[dict[str, Any]]) -> list[Any]:
    """Score synthetic parity rows with the stripped artifact using the exact same transforms as Scorer."""
    frame = pd.DataFrame(rows).copy()
    for column in meta["input_columns"]:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[meta["input_columns"]]
    for column, kind in (meta.get("input_types") or {}).items():
        values = pd.to_numeric(frame[column], errors="coerce") if kind.startswith(("int", "Int", "float")) else None
        if values is not None and kind.startswith("Int"):
            frame[column] = values.round().astype("Int64")
        elif values is not None and kind.startswith("int") and values.notna().all():
            frame[column] = values.round().astype("int64")
        elif values is not None:
            frame[column] = values.astype(float)
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]) or str(frame[column].dtype) == "boolean":
            frame[column] = pd.to_numeric(frame[column].astype("object").map({True: 1.0, False: 0.0}), errors="coerce")
    time_col = meta.get("time_column")
    if time_col and meta.get("time_is_datetime"):
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in artifact.get("hashed_columns") or []:
        frame[column] = rn._as_text(frame[column]).map(lambda v: "h:" + hashlib.sha256(f"{key}|{v}".encode()).hexdigest()[:16])
    recipe = artifact["recipe"]
    entity = getattr(recipe, "entity_model", None)
    if entity is not None:
        entity.salt = key
    try:
        matrix, _ = recipe.transform(frame)
    finally:
        if entity is not None:
            entity.salt = None  # never persist the secret salt inside the artifact
    scores = rn.predict_scores(artifact["model"], matrix, meta["task_type"], meta.get("n_classes"))
    change = meta.get("review_change") or {}
    if change.get("review__target") == "log1p":
        scores = np.expm1(scores)
    if change.get("review__clip") == "fit_range" and meta.get("target_range"):
        scores = np.clip(scores, *meta["target_range"])
    return np.asarray(scores, dtype=float).tolist()


def capture_final_artifact(store: Any, project_id: str, run_id: str, project: dict[str, Any], prepared: Any,
                           final_evidence: dict[str, Any], fitted: dict[str, Any]) -> dict[str, Any] | None:
    """Capture the privacy-safe fitted artifact, serving metadata, and synthetic parity fixture at the original
    authorized final run so service.export never refits or re-reads training/holdout data (I1)."""
    if not prepared.kind.export.get("service", True):
        return None
    bundle = prepared.bundle
    if bundle.task_type == "text_tabular_binary":
        return None
    recipe_name = str(fitted["recipe_name"])
    family = str(fitted["family"])
    params = fitted.get("params") or None
    try:
        key_ref = f"service_artifact_{run_id}.key"
        key = secrets.token_hex(16)
        _write_secret_key(store, project_id, key_ref, key)
        shallow = copy.copy(fitted["recipe"])  # never deep-copy the training bundle only to drop it
        shallow.bundle = types.SimpleNamespace(spec=fitted["recipe"].bundle.spec, task_type=fitted["recipe"].bundle.task_type)
        stripped_recipe, hashed = _strip(copy.deepcopy(shallow), key)
        artifact = {"recipe": stripped_recipe, "model": fitted["model"], "hashed_columns": hashed}
        model_bytes = pickle.dumps(artifact)
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        model_version = model_sha256[:16]
        solution = prepared.solution
        train_frame = _training_frame(store, project, prepared)
        y_train = bundle.y_train.to_numpy(dtype=float) if bundle.task_type == "timeseries_regression" else None
        time_col = solution.time_column
        threshold = ((final_evidence.get("task_assessment") or {}).get("threshold") or {}).get("value") if solution.alert_budget else None
        behaviour = bool(getattr(stripped_recipe, "behaviour", None) or getattr(stripped_recipe, "programs", None))
        input_columns = [str(c) for c in bundle.X.columns]
        meta: dict[str, Any] = {
            "project": project["name"], "task": solution.task, "task_type": bundle.task_type, "target": solution.target,
            "recipe": recipe_name, "model": family, "input_columns": input_columns,
            "input_types": {str(c): str(t) for c, t in bundle.X.dtypes.items()},
            "review_change": {k: v for k, v in (params or {}).items() if str(k).startswith("review__")},
            "target_range": [float(np.min(y_train)), float(np.max(y_train))] if y_train is not None else None,
            "behaviour_within_batch": behaviour,
            "time_column": time_col, "time_is_datetime": bool(time_col and column_kind(train_frame[time_col]) == "datetime") if time_col else False,
            "class_labels": list(prepared.class_labels) if prepared.class_labels else None, "n_classes": rn._n_classes(bundle),
            "positive_label": solution.positive_label, "threshold": threshold, "prediction_moment": solution.prediction_moment,
            "holdout": {k: final_evidence["holdout_metrics"].get(k) for k in (bundle.spec.primary_metric,)},
            "live_reference": _live_reference(solution.task, bundle.y_test, fitted["predictions"],
                                              bundle.groups("test") if bundle.spec.split_strategy in ("group", "time_group") else None),
            "batches": ("behaviour features are computed within each request, as they were within each fold and the holdout in "
                        "training: send a whole audit window (or a batch in time order) together; a single row gets no history, "
                        "and no state is kept between requests" if behaviour else "each row is scored on its own"),
            "production_approved": False,
            "model_id": project_id, "model_version": model_version,
            "negative_label": _negative_label(train_frame, solution) if solution.task == "binary" else None,
            "na_codes": [c for c in codes_kept(train_frame) if c in set(input_columns)],
            # the versions the model was pickled with: the export pins these, not the exporting host's (review of 96e47f43)
            "packages": _package_versions(), "python": platform.python_version(),
        }
        calibrator = (final_evidence.get("calibration_fit") or {}).get("calibrator")
        if calibrator and calibrator.get("method") != "none":
            meta["calibrator"] = calibrator
        watch = final_evidence.get("monitoring") or {}
        monitoring_bytes: bytes | None = None
        if watch.get("columns"):
            monitoring_bytes = (json.dumps(_hashed_plan(watch, key), indent=1, default=str) + "\n").encode("utf-8")
            meta["monitoring"] = "monitoring.json"
        meta_bytes = (json.dumps(meta, indent=1, default=str) + "\n").encode("utf-8")
        profile = (project.get("data") or {}).get("profile") or {}
        parity_rows = _made_up_rows(profile, input_columns, n=5)
        parity_expected = _score_stripped(artifact, meta, key, parity_rows)
        parity_doc = {
            "schema_version": ARTIFACT_SCHEMA,
            "task_type": bundle.task_type,
            "tolerance": 1e-6,
            "rows": parity_rows,
            "expected": parity_expected,
            "test_rows": _made_up_rows(profile, input_columns, n=3),
            "test_outcome": _made_up_outcome(solution, prepared.class_labels),
        }
        parity_bytes = (json.dumps(parity_doc, indent=1, default=str) + "\n").encode("utf-8")
        files = files_for(store)
        prefix = f"projects/{project_id}/artifacts/{run_id}"
        planned = [("artifact", f"{prefix}/model.pkl", model_bytes, "application/octet-stream"),
                   ("meta", f"{prefix}/serving_meta.json", meta_bytes, "application/json"),
                   ("parity", f"{prefix}/parity_fixture.json", parity_bytes, "application/json")]
        if monitoring_bytes is not None:
            planned.append(("monitoring", f"{prefix}/monitoring.json", monitoring_bytes, "application/json"))
        captured: dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA,
            "format_version": ARTIFACT_FORMAT,
            "project_id": project_id,
            "run_id": run_id,
            "kind": prepared.kind.key,
            "task": solution.task,
            "task_type": bundle.task_type,
            "recipe": recipe_name,
            "model": family,
            "model_version": model_version,
            "monitoring_key": None,
            "monitoring_sha256": None,
            "key_ref": key_ref,
        }
        for name, file_key, data, _kind in planned:  # the digests are known before storage is touched
            captured[f"{name}_key"], captured[f"{name}_sha256"] = file_key, hashlib.sha256(data).hexdigest()
    except Exception as exc:  # noqa: BLE001 - capture failure never destroys the already-consumed holdout record
        return {
            "schema_version": ARTIFACT_SCHEMA,
            "format_version": ARTIFACT_FORMAT,
            "project_id": project_id,
            "run_id": run_id,
            "complete": False,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }
    try:
        for _name, file_key, data, _kind in planned:  # the bytes on the local disk first: a storage failure after this is retried from them
            local = files.local_path(file_key)
            local.parent.mkdir(parents=True, exist_ok=True)
            part = local.with_name(f"{local.name}.{secrets.token_hex(4)}.part")
            part.write_bytes(data)
            os.replace(part, local)
        for name, file_key, _data, kind in planned:
            record = files.put(file_key, files.local_path(file_key), kind=kind)
            if record["sha256"] != captured[f"{name}_sha256"]:
                raise FileError(f"{file_key} was stored with another digest")
    except Exception as exc:  # noqa: BLE001 - a transient storage failure: the export retries from the local bytes (no refit)
        return {**captured, "complete": False, "recoverable": True, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {**captured, "complete": True}


def recapture(store: Any, project_id: str) -> dict[str, Any]:
    """Finish a capture a storage failure interrupted: store again the bytes the final run wrote to the local disk, each
    checked against the digest its run record holds (never a refit, never a second read of the holdout). Returns the
    completed artifact record; raises ServiceRefused when the bytes are gone or differ, or storage still fails."""
    final_record = store.read_stage(project_id, "final") or {}
    art = (final_record.get("evidence") or {}).get("service_artifact")
    if not isinstance(art, dict):
        raise ServiceRefused("Approved artifact unavailable: no captured artifact to finish")
    if art.get("complete"):
        return art
    run_id = art.get("run_id") or "unknown"
    if not art.get("recoverable") or approval_binding(art) is None:
        raise ServiceRefused(f"Approved artifact unavailable for final run {run_id}: {art.get('error') or 'the capture did not finish'}. "
                             "Export never refits or re-reads holdout data; a new governed final evaluation is required to produce a verified artifact.")
    if not _read_secret_key(store, project_id, str(art.get("key_ref") or "")):
        raise ServiceRefused(f"Approved artifact unavailable for final run {run_id}: serving key missing from protected store")
    files = files_for(store)
    for name in ("artifact", "meta", "parity", "monitoring"):
        file_key, digest = art.get(f"{name}_key"), art.get(f"{name}_sha256")
        if not file_key:
            continue
        local = files.local_path(str(file_key))
        if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise ServiceRefused(f"Approved artifact unavailable for final run {run_id}: the locally written {name} file is missing or "
                                 "changed since the final run; a new governed final evaluation is required")
        try:
            record = files.put(str(file_key), local, kind="application/octet-stream" if name == "artifact" else "application/json")
        except Exception as exc:  # noqa: BLE001 - still failing: say so, keep the record as it is for the next try
            raise ServiceRefused(f"Approved artifact storage is still failing for final run {run_id} ({type(exc).__name__}); try again") from None
        if record["sha256"] != digest:
            raise ServiceRefused(f"Approved artifact integrity check failed for {name}: digest mismatch")
    done = {k: v for k, v in art.items() if k not in ("recoverable", "error")}
    done.update({"complete": True, "recaptured": {"at": _now(), "after": str(art.get("error") or "")[:200]}})
    final_record.setdefault("evidence", {})["service_artifact"] = done
    store.write_stage(project_id, "final", final_record)
    store.log(project_id, "service_artifact_recaptured", {"run": run_id, "artifact_sha256": done["artifact_sha256"]})
    return done


def _verified_bytes(files: Any, key: str, expected_sha256: str, label: str) -> bytes:
    """Load a host-created artifact file from trusted storage and verify its SHA-256 digest."""
    if not key or not expected_sha256:
        raise ServiceRefused(f"Approved artifact unavailable: missing {label} reference")
    record = files.record(key)
    if record is None:
        raise ServiceRefused(f"Approved artifact unavailable: {label} is not registered in trusted storage ({key})")
    try:
        local = files.path(key, expected=expected_sha256)
        raw = local.read_bytes()
    except (FileError, OSError) as exc:
        raise ServiceRefused(f"Approved artifact unavailable: could not verify {label} ({exc})") from None
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ServiceRefused(f"Approved artifact integrity check failed for {label}: digest mismatch")
    return raw


def _publish_dir(staging: Path, out: Path) -> None:
    """Atomically replace ``out`` with ``staging``, keeping the previous export intact if publishing fails."""
    out.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if out.exists():
        backup = out.parent / f".{out.name}.prev-{secrets.token_hex(4)}"
        os.replace(out, backup)
    try:
        os.replace(staging, out)
    except Exception:
        if backup is not None and backup.exists() and not out.exists():
            os.replace(backup, out)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def export(store, project_id: str, out: Path) -> dict[str, Any]:
    """Write the service package to ``out`` from the verified approved final artifact without refitting or
    re-reading training/holdout data; refuses unapproved, unsupported, legacy, or tampered records."""
    project = store.get(project_id)
    final_stage = project.get("stages", {}).get("final") or {}
    if final_stage.get("status") != "approved":
        raise ServiceRefused("Approve the final stage first: a service is built only from an approved result")
    final_record = store.read_stage(project_id, "final")
    if not final_record:
        raise ServiceRefused("Approved artifact unavailable: the final stage record is missing")
    solution = Solution(**(project.get("solution") or {}))
    kind = kinds.resolve(solution, project.get("pack"))
    if not kind.export.get("service", True):  # 16.1: preserve exact refusal for kinds without service export
        raise ServiceRefused(f"The {kind.title.lower()} kind has no service export yet: {kind.export.get('why', 'not built')}")
    task_type = str(final_record.get("task_type") or "")
    if task_type == "text_tabular_binary":
        raise ServiceRefused("A text project's vocabulary is learned from its rows: its service export is not built yet")
    final = final_record.get("evidence") or {}
    final_run_id = (final_record.get("run") or {}).get("run") or final_stage.get("run")
    art = final.get("service_artifact")
    if isinstance(art, dict) and not art.get("complete") and art.get("recoverable"):
        art = recapture(store, project_id)  # a storage failure at the final run: finished from its local bytes, digests checked
    if not isinstance(art, dict) or not art.get("complete"):
        why = (art.get("error") if isinstance(art, dict) and art.get("error") else "no captured fitted artifact was stored for this final run")
        raise ServiceRefused(
            f"Approved artifact unavailable for final run {final_run_id or 'unknown'}: {why}. "
            "Export never refits or re-reads holdout data; a new governed final evaluation is required to produce a verified artifact."
        )
    if art.get("schema_version") != ARTIFACT_SCHEMA or art.get("format_version") != ARTIFACT_FORMAT:
        raise ServiceRefused(f"Approved artifact unavailable: unsupported artifact schema {art.get('schema_version')!r}")
    if art.get("project_id") != project_id or (final_run_id and art.get("run_id") != final_run_id):
        raise ServiceRefused("Approved artifact binding mismatch: artifact does not belong to the approved final run")
    approved_art = final_stage.get("approved_artifact")
    if not isinstance(approved_art, dict):
        raise ServiceRefused("Approved artifact binding missing: the approval does not name this artifact; approve the final stage again")
    for field in BIND_FIELDS:
        if field not in approved_art:
            raise ServiceRefused(f"Approved artifact binding incomplete (no {field}): approve the final stage again to bind every file")
        if approved_art.get(field) != art.get(field):
            raise ServiceRefused(f"Approved artifact binding mismatch on {field}: candidate changed after approval")
    approved_run = final_stage.get("approved_run")
    if approved_run and art.get("run_id") != approved_run:
        raise ServiceRefused("Approved artifact binding mismatch: final run does not match approved_run")
    key = _read_secret_key(store, project_id, str(art.get("key_ref") or ""))
    if not key:
        raise ServiceRefused(f"Approved artifact unavailable for final run {art.get('run_id')}: serving key missing from protected store")
    files = files_for(store)
    model_bytes = _verified_bytes(files, str(art.get("artifact_key") or ""), str(art.get("artifact_sha256") or ""), "model.pkl")
    meta_bytes = _verified_bytes(files, str(art.get("meta_key") or ""), str(art.get("meta_sha256") or ""), "serving_meta.json")
    parity_bytes = _verified_bytes(files, str(art.get("parity_key") or ""), str(art.get("parity_sha256") or ""), "parity_fixture.json")
    monitoring_bytes = (
        _verified_bytes(files, str(art["monitoring_key"]), str(art.get("monitoring_sha256") or ""), "monitoring.json")
        if art.get("monitoring_key") else None
    )
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
        parity = json.loads(parity_bytes.decode("utf-8"))
    except ValueError as exc:
        raise ServiceRefused(f"Approved artifact unavailable: malformed metadata ({exc})") from None
    expected_version = str(art["artifact_sha256"])[:16]
    if meta.get("model_id") != project_id or meta.get("model_version") != expected_version:
        raise ServiceRefused("Approved artifact metadata mismatch: model identity does not match verified artifact digest")
    meta["project"] = project["name"]
    recipe_name, family = str(meta["recipe"]), str(meta["model"])
    log_key = project_log_key(store, project_id)  # stable across exports and retrains (DCLAB_LOG_KEY)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
    try:
        (staging / "tests").mkdir(parents=True)
        (staging / "model.pkl").write_bytes(model_bytes)
        if monitoring_bytes is not None:
            (staging / "monitoring.json").write_bytes(monitoring_bytes)
        (staging / "model.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
        for name in ("scorer.py", "app.py", "serve_log.py"):
            shutil.copyfile(TEMPLATE / name, staging / name)
        vendor = staging / "vendor"
        for source, target in VENDOR.items():
            (vendor / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, vendor / target)
        (vendor / "dclab_rnd" / "__init__.py").write_text('"""The parts of DCLab R&D a fitted service needs."""\n', encoding="utf-8")
        shutil.copytree(ROOT / "dclab_core", vendor / "dclab_core", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "replicas", "schemas"))
        host = _package_versions()
        at_fit = meta.get("packages") if isinstance(meta.get("packages"), dict) else None
        versions = at_fit if at_fit else host  # a record from before the versions were kept: this host's, said so below
        pins = [f"{package}=={version}" for package, version in versions.items()]
        drift = {package: {"fit": version, "host": host.get(package)} for package, version in (at_fit or {}).items() if host.get(package) != version}
        python_tag = ".".join(str(meta.get("python") or "3.12").split(".")[:2])
        (staging / "requirements.txt").write_text("\n".join(pins + ["flask>=3.0", "gunicorn>=22.0", "pytest>=8.0"]) + "\n", encoding="utf-8")
        (staging / "Dockerfile").write_text(
            f"FROM python:{python_tag}-slim\nWORKDIR /service\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n"
            "# the model, the code and the tests: no data file is copied\nCOPY model.pkl model.json scorer.py app.py serve_log.py ./\n"
            + ("COPY monitoring.json ./\n" if meta.get("monitoring") else "") + "COPY vendor ./vendor\n"
            "COPY tests ./tests\n# the hashing key is given at run time: docker run -e DCLAB_SERVICE_KEY=... (it is not in the image)\n"
            "# the prediction and outcome logs: mount a volume here (-v ...:/logs), or they go with the container\n"
            "ENV DCLAB_SERVICE_LOG_DIR=/logs\n"
            "EXPOSE 8000\nCMD [\"gunicorn\", \"--bind\", \"0.0.0.0:8000\", \"--workers\", \"2\", \"--timeout\", \"120\", \"app:app\"]\n", encoding="utf-8")
        (staging / ".dockerignore").write_text("*.csv\n*.parquet\n*.xlsx\n*.json.gz\n*.jsonl\n__pycache__\n", encoding="utf-8")
        rows = list(parity.get("test_rows") or _made_up_rows((project.get("data") or {}).get("profile") or {}, meta["input_columns"]))
        test_outcome = parity.get("test_outcome", _made_up_outcome(solution, meta.get("class_labels")))
        (staging / "tests" / "test_service.py").write_text(
            '"""Generated with the service: made-up rows from the profile (never a real value)."""\n\nimport io\nimport sys\nfrom pathlib import Path\n\n'
            "import pytest\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\nfrom app import app  # noqa: E402\n\n"
            f"ROWS = {json.dumps(rows)}\n\n\n"
            "@pytest.fixture(autouse=True)\ndef _logs(tmp_path, monkeypatch):\n    monkeypatch.setenv('DCLAB_SERVICE_LOG_DIR', str(tmp_path))  # the tests' rows stay out of the real logs\n\n\n"
            "def test_health_and_model():\n    client = app.test_client()\n    assert client.get('/health').json == {'status': 'ok'}\n"
            "    assert client.get('/model').json['task'] == " + repr(meta["task"]) + "\n\n\n"
            "def test_one_and_many_rows():\n    client = app.test_client()\n    one = client.post('/predict', json=ROWS[0]).json['predictions']\n"
            "    many = client.post('/predict', json=ROWS).json['predictions']\n    assert len(one) == 1 and len(many) == len(ROWS)\n\n\n"
            "def test_a_csv():\n    import pandas as pd\n    client = app.test_client()\n    data = pd.DataFrame(ROWS).to_csv(index=False).encode()\n"
            "    response = client.post('/predict/csv', data={'file': (io.BytesIO(data), 'rows.csv')})\n"
            "    assert len(response.json['predictions']) == len(ROWS)\n\n\n"
            "def test_a_prediction_is_logged_and_its_outcome_joins_it(tmp_path):\n    from dclab_core import serving\n    client = app.test_client()\n"
            "    made = client.post('/predict', json=ROWS).json['predictions']\n"
            f"    outcome = {test_outcome!r}\n"
            "    assert client.post('/outcome', json={'key': made[0]['key'], 'outcome': outcome}).json == {'recorded': 1}\n"
            "    joined = serving.join(serving.read_log(tmp_path / 'predictions.jsonl', serving.PredictionRecord),\n"
            "                          serving.read_log(tmp_path / 'outcomes.jsonl', serving.OutcomeRecord))\n"
            "    assert len(joined) == 1\n", encoding="utf-8")
        (staging / "README.md").write_text(
            f"# Scoring service: {project['name']}\n\nExported by DCLab from the approved final stage: recipe `{recipe_name}`, model `{family}`. "
            "No data is in this folder, and the key that hashes category values is not either: it was given to whoever exported "
            "the service, who passes it as `DCLAB_SERVICE_KEY`. DCLab gives `DCLAB_LOG_KEY` with it, the same for every export of this "
            "project, so the logs of one export and the next join.\n\n```bash\ndocker build -t dclab-service .\n"
            "docker run -e DCLAB_SERVICE_KEY=... -e DCLAB_LOG_KEY=... -p 8000:8000 dclab-service\n```\n\n"
            "Endpoints: `GET /health`, `GET /model`, `POST /predict` (one row as an object, or a list), `POST /predict/csv` (form field `file`), "
            "`POST /outcome` (the true outcome of a row once it is known), `GET /monitor` (the model in use, window by window).\n\n"
            "Logs: every row scored is a line of `predictions.jsonl` and every outcome a line of `outcomes.jsonl`, beside the model or in "
            "`DCLAB_SERVICE_LOG_DIR` (the image sets `/logs`: mount a volume there). Keys are hashed with `DCLAB_LOG_KEY` (the project's own, given "
            "with every export; without it, `DCLAB_SERVICE_KEY`, which changes with each export): pass it every time, so an outcome that arrives "
            "after a retrain still joins its prediction. A line holds the model and its version, the row's key "
            "as a salted hash, the score, the decision and the time; never a value of the row. Name the column that identifies a row "
            "(`{\"rows\": [...], \"key\": \"order_id\"}`, or `?key=order_id`, or the form field `key` with a CSV) and post its outcome "
            "later with the same value: `{\"key\": 123, \"outcome\": ..., \"known_at\": \"2026-10-25T08:00:00+00:00\"}` or "
            "`{\"outcomes\": [...]}`; with no key column, each prediction comes back with a key made for it. The two logs join by key "
            "(`dclab_core.serving.join`), which is what monitoring with outcomes and a retrain read.\n\n"
            f"{meta['batches'].capitalize()}. A score is evidence about the method, not production approval.\n"
            + ("\nMonitoring: `Scorer().monitor(rows, scores)` reads a batch against the training rows' bins in `monitoring.json` "
               "(each column's and the score's PSI, the share the rule flags); an alert says the inputs moved away from what the model "
               "was trained on. `GET /monitor?window=day` (or `week`, `batch`) reads the logs: the score's and the inputs' drift (each "
               "batch's inputs kept in `profiles.jsonl` only as the share of its rows in each training bin), the share the rule flags, and "
               "live performance once outcomes are known, against the holdout's interval in `model.json` (`live_reference`); one worse "
               "window is a warning, two running an alert.\n" if meta.get("monitoring") else "")
            + (f"\nVersions: `requirements.txt` pins the packages the model was fitted and pickled with (Python {meta.get('python')}). "
               "The export's own check ran on the exporting machine's packages, which differ for "
               + ", ".join(f"{p} ({v['fit']} at fit, {v['host']} here)" for p, v in drift.items())
               + ": build the image and run its tests before serving it.\n" if drift else "")
            + ("\nVersions: this model's record is older than the kept package versions; `requirements.txt` pins the exporting "
               "machine's, which may not be the ones it was pickled with.\n" if not at_fit else ""), encoding="utf-8")
        checked = _reload_check(staging, key, parity["rows"], parity["expected"], str(parity.get("task_type") or meta["task_type"]),
                                tol=float(parity.get("tolerance", 1e-6)))
        checked["versions"] = {"pinned_from": "fit" if at_fit else "host (not kept at fit)", "differ": drift}
        if drift:
            checked["how"] += "; on this machine's packages, which differ from the pinned fit-time ones: " + ", ".join(sorted(drift))
        _publish_dir(staging, out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    store.log(project_id, "service_exported", {"recipe": recipe_name, "model": family, "folder": out.name, "checked": checked,
                                               "artifact_sha256": art["artifact_sha256"]})
    from . import graph

    graph.capture(store, project_id, "human")  # WF-10: the export after the final stage, validated and logged as any move
    project = store.get(project_id)
    project["exports"] = {**(project.get("exports") or {}), "service": {"at": _now(), "folder": out.name, "final_run": art["run_id"],
                                                                        "artifact_sha256": art["artifact_sha256"], "model_version": meta["model_version"],
                                                                        **checked}}
    store.save(project)
    return {"folder": str(out), "meta": meta, "key": key, "log_key": log_key, "checked": checked}


LOG_KEY_FILE = Path("secrets") / "service_log.key"  # beside the project; its name says secret, so the store and the testers' kit never copy it


def project_log_key(store: Any, project_id: str) -> str | None:
    """34.9's open item: the key that hashes the service log's row keys, one per project and the same for every export, so an
    outcome posted to a later export still joins the prediction an earlier one logged. Made once and kept beside the project
    (never in a record, an event, a log or the export's folder). The project's folder is found through ``export_dir``, which
    both storage backends have (files and PostgreSQL), and goes with the project when it is deleted. Made with the owner's
    read only, at once (never readable in between), and the first of two exports at the same moment wins for both."""
    import os

    path = Path(store.export_dir(project_id)).parent / LOG_KEY_FILE
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for _ in range(2):
        if path.is_file():
            held = path.read_text(encoding="utf-8").strip()
            if len(held) >= 32:
                return held
            path.unlink()  # a short or broken key is made again, never used
        try:
            handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # another export made it first: read that one
            continue
        made = secrets.token_hex(16)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(made)
        return made
    raise RuntimeError("The project's log key could not be made or read")


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


def _live_reference(task: str, y_test: Any, predictions: Any, groups: Any = None) -> dict[str, Any] | None:
    """34.10: the holdout's ROC AUC, accuracy or mean absolute error with its 90% interval (over whole groups when the split
    keeps them together), from the refit's predictions on the holdout rows (the ones the export just checked against the
    approved record); None when it cannot be read."""
    from dclab_core import live

    try:
        return live.reference(task if task in ("binary", "multiclass") else "regression", np.asarray(y_test, dtype=float), predictions,
                              groups=groups)
    except (ValueError, TypeError):
        return None


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


def _reload_check(out: Path, key: str, rows: Any, expected: Any, task_type: str, *, tol: float = 1e-6) -> dict[str, Any]:
    """30.15 (step 10) + I1: load the staged export in isolated Python, score the synthetic parity rows, and verify
    predictions match the captured artifact within tolerance."""
    import subprocess
    import sys
    import tempfile

    scratch = Path(tempfile.mkdtemp())
    try:
        rows_file = scratch / "rows.json"
        if isinstance(rows, list):
            rows_file.write_text(json.dumps(rows), encoding="utf-8")
        else:
            rows.to_json(rows_file, orient="records", date_format="iso", double_precision=15)
        field = "probabilities" if task_type == "multiclass" else ("score" if task_type in ("binary_imbalanced",) else "prediction")

        script = ("import json, sys; sys.path.insert(0, sys.argv[1]); from scorer import Scorer; s = Scorer(); "
                  "rows = json.load(open(sys.argv[2])); out = s.predict(rows); "
                  f"print(json.dumps([list(r['{field}'].values()) if isinstance(r.get('{field}'), dict) else r.get('{field}', r.get('score', r.get('prediction'))) for r in out]))")
        try:  # the key by the environment, never on the command line (a process list shows it)
            done = subprocess.run([sys.executable, "-I", "-c", script, str(out), str(rows_file)], cwd=str(scratch), capture_output=True, text=True,
                                  timeout=900, env={**{k: os.environ[k] for k in CHECK_ENV if k in os.environ}, "DCLAB_SERVICE_KEY": key})
            got = np.asarray(json.loads(done.stdout.strip().splitlines()[-1]), dtype=float) if done.returncode == 0 else None
        except (subprocess.TimeoutExpired, ValueError, IndexError) as error:
            raise ServiceRefused(f"The exported service could not be checked ({type(error).__name__}): not exported") from None
        if got is None:
            raise ServiceRefused("The exported service did not load and predict on its own: " + done.stderr.strip()[-600:])
        want = np.asarray(expected, dtype=float)
        diff = float(np.max(np.abs(got - want))) if got.shape == want.shape and got.size else float("inf")
        if not diff <= tol:
            raise ServiceRefused(f"The exported service does not match the approved artifact parity fixture (largest difference {diff}): not exported")
        return {"checked": True, "rows": int(len(want)), "max_abs_diff": diff, "how": "reloaded in an isolated Python, synthetic parity rows matched"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
