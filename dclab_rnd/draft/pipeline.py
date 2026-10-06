"""The background data pipeline of a draft: structure → clean → analyse, one asset at a time.

Each step updates the asset's status and emits a ``pipeline`` event, so Home shows progress live while
the user keeps talking with the agent. The cleaned table is saved as ``data/clean.parquet``: that is the
table a project starts from when the draft is built. Cleaning is structural and logged (``clean.py``);
nothing is learned from the data that could leak, and nothing looks at a possible target yet.
"""

from __future__ import annotations

import os

import secrets
from pathlib import Path
from typing import Any

import pandas as pd

from .store import DraftStore, now

STEPS = ("structuring", "cleaning", "analysing")


def new_asset(store: DraftStore, draft_id: str, kind: str, name: str, filename: str, actor: str = "human", **extra: Any) -> dict[str, Any]:
    """A new asset, queued for the pipeline, and its import in the audit (source and SHA-256; it reads the whole file,
    so an async route calls this in a thread)."""
    asset = {"id": "a" + secrets.token_hex(4), "kind": kind, "name": name, "filename": filename, "status": "queued",
             "added": now(), "synthetic": bool(extra.pop("synthetic", False)), **extra}

    def add(d):
        d["assets"].append(asset)
    store.update(draft_id, add)
    store.emit(draft_id, "pipeline", {"asset": asset, "step": "queued", "text": f"Received {name}"})
    from .. import audit

    source = asset.get("source") if isinstance(asset.get("source"), dict) else {}
    audit.record_for(store, "data_import", actor, "Home agent" if actor == "agent" else None, draft_id=draft_id,
                     move="add_asset", status="allowed", source={"kind": kind, **{k: v for k, v in source.items() if k != "sha256"}},
                     filename=filename, sha256=source.get("sha256") or audit.sha256_of(store.data_dir(draft_id) / filename),
                     synthetic=asset["synthetic"], bytes=asset.get("bytes"))
    return asset


def set_asset(store: DraftStore, draft_id: str, asset_id: str, **values: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def put(d):
        for a in d["assets"]:
            if a["id"] == asset_id:
                a.update(values)
                found.update(a)
    store.update(draft_id, put)
    return found


def _record(store: DraftStore, path) -> None:
    """Record a draft file in file storage; a file the storage cannot take (a name it would not write) stays as it is."""
    try:
        from ..storage.files import files_for

        files = files_for(store)
        files.put(files.key_of(path), path)
    except Exception:  # noqa: BLE001 — best effort: a storage that fails (S3 without credentials, the database) never stops the pipeline
        pass


def run(store: DraftStore, draft_id: str, asset_id: str, agent=None, client=None) -> dict[str, Any]:
    """Process one asset. Returns the final asset record (status "ready" or "failed")."""
    from ..jobs import checkpoint  # a stop asked for on the Compute page takes effect between the steps (package 10.3)
    from . import analyze, clean, structure  # imported lazily: the store and the chat do not need pandas

    draft = store.get(draft_id)
    asset = next(a for a in draft["assets"] if a["id"] == asset_id)
    path = store.data_dir(draft_id) / asset["filename"]
    _record(store, path)  # the upload, a sample or a connector's file, as it arrived (package 9.3)
    try:
        checkpoint()
        set_asset(store, draft_id, asset_id, status="structuring")
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "structuring", "text": "Reading the file and turning it into a table"})
        # DCLAB_MODEL_READS_SAMPLE_LINES=0: a file no built-in reader fits is read line by line and nothing is sent
        reader = client if os.environ.get("DCLAB_MODEL_READS_SAMPLE_LINES", "1") != "0" else None
        frame, info = structure.to_table(path, client=reader)
        set_asset(store, draft_id, asset_id, status="cleaning", format=info.get("format"), structure=info)
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "structured", "text": _structured_text(info), "info": info})

        checkpoint()
        frame, log = clean.clean(frame)
        codes = clean.category_code_columns(frame)
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "cleaned",
                                          "text": f"{len(log)} cleaning step{'s' if len(log) != 1 else ''} applied" if log else "No cleaning needed",
                                          "log": log})
        checkpoint()
        set_asset(store, draft_id, asset_id, status="analysing")
        target = (draft.get("understanding") or {}).get("target")
        report = analyze.analyze(frame, target=target if target in frame.columns else None)
        from .chat import rank_targets  # names only: the columns the problem sentence mentions are offered first

        report["profile"]["target_candidates"] = rank_targets(report["profile"], draft.get("problem") or "")
        checkpoint()
        out = store.data_dir(draft_id) / "clean.parquet"
        _to_parquet(frame, out)
        _record(store, out)
        final: dict[str, Any] = {}

        def put(d):  # one write: a reader never sees the asset ready without its analysis
            for a in d["assets"]:
                if a["id"] == asset_id:
                    a.update(status="ready", rows=int(len(frame)), columns=int(frame.shape[1]), category_codes=codes, clean_file="clean.parquet")
                    final.update(a)
            d["cleaning_log"], d["analysis"], d["structure"], d["active_asset"] = log, report, info, asset_id
        store.update(draft_id, put)
        store.emit(draft_id, "analysis", {"asset": asset_id, "analysis": report})
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "ready", "text": "Ready", "rows": int(len(frame)), "columns": int(frame.shape[1])})
        if agent is not None:
            agent.data_ready(draft_id, final)
        return final
    except Exception as exc:  # noqa: BLE001 — any failure is reported to the user in plain words
        message = str(exc) if exc.__class__.__name__ in ("StructureError", "ValueError") else f"{type(exc).__name__}: the file could not be processed"
        final = set_asset(store, draft_id, asset_id, status="failed", error=message)
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "failed", "text": message})
        if agent is not None:
            agent.data_failed(draft_id, final, message)
        return final


def _structured_text(info: dict[str, Any]) -> str:
    fmt = (info.get("format") or "table").replace("_", " ")
    rate = info.get("parse_rate")
    text = f"Read as {fmt}: {info.get('rows', 0):,} rows, {info.get('columns', 0)} columns"
    if rate is not None and rate < 1:
        text += f" ({rate:.0%} of lines parsed)"
    return text


def _to_parquet(frame: pd.DataFrame, path: Path) -> None:
    safe = frame.copy()
    for col in safe.columns:  # parquet needs one type per column; mixed object columns become strings
        if safe[col].dtype == object:
            safe[col] = safe[col].map(_as_text)
    safe.to_parquet(path, index=False)


def _as_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:  # NaN
        return None
    try:
        if pd.isna(value) is True:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
