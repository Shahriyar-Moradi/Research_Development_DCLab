"""The background data pipeline of a draft: structure → clean → analyse, one asset at a time.

Each step updates the asset's status and emits a ``pipeline`` event, so Home shows progress live while
the user keeps talking with the agent. The cleaned table is saved as ``data/clean.parquet``: that is the
table a project starts from when the draft is built. Cleaning is structural and logged (``clean.py``);
nothing is learned from the data that could leak, and nothing looks at a possible target yet.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import pandas as pd

from .store import DraftStore, now

STEPS = ("structuring", "cleaning", "analysing")


def new_asset(store: DraftStore, draft_id: str, kind: str, name: str, filename: str, **extra: Any) -> dict[str, Any]:
    asset = {"id": "a" + secrets.token_hex(4), "kind": kind, "name": name, "filename": filename, "status": "queued",
             "added": now(), "synthetic": bool(extra.pop("synthetic", False)), **extra}

    def add(d):
        d["assets"].append(asset)
    store.update(draft_id, add)
    store.emit(draft_id, "pipeline", {"asset": asset, "step": "queued", "text": f"Received {name}"})
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


def run(store: DraftStore, draft_id: str, asset_id: str, agent=None, client=None) -> dict[str, Any]:
    """Process one asset. Returns the final asset record (status "ready" or "failed")."""
    from . import analyze, clean, structure  # imported lazily: the store and the chat do not need pandas

    draft = store.get(draft_id)
    asset = next(a for a in draft["assets"] if a["id"] == asset_id)
    path = store.data_dir(draft_id) / asset["filename"]
    try:
        set_asset(store, draft_id, asset_id, status="structuring")
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "structuring", "text": "Reading the file and turning it into a table"})
        frame, info = structure.to_table(path, client=client)
        set_asset(store, draft_id, asset_id, status="cleaning", format=info.get("format"), structure=info)
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "structured", "text": _structured_text(info), "info": info})

        frame, log = clean.clean(frame)
        codes = clean.category_code_columns(frame)
        store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "cleaned",
                                          "text": f"{len(log)} cleaning step{'s' if len(log) != 1 else ''} applied" if log else "No cleaning needed",
                                          "log": log})
        set_asset(store, draft_id, asset_id, status="analysing")
        target = (draft.get("understanding") or {}).get("target")
        report = analyze.analyze(frame, target=target if target in frame.columns else None)
        from .chat import rank_targets  # names only: the columns the problem sentence mentions are offered first

        report["profile"]["target_candidates"] = rank_targets(report["profile"], draft.get("problem") or "")
        out = store.data_dir(draft_id) / "clean.parquet"
        _to_parquet(frame, out)
        final = set_asset(store, draft_id, asset_id, status="ready", rows=int(len(frame)), columns=int(frame.shape[1]),
                          category_codes=codes, clean_file="clean.parquet")

        def put(d):
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
