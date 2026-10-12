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
    from .. import audit

    source = extra.get("source") if isinstance(extra.get("source"), dict) else {}
    raw_sha = source.get("sha256") or audit.sha256_of(store.data_dir(draft_id) / filename)
    asset = {"id": "a" + secrets.token_hex(4), "kind": kind, "name": name, "filename": filename, "status": "queued",
             "added": now(), "synthetic": bool(extra.pop("synthetic", False)), "raw_sha256": raw_sha, **extra}

    def add(d):
        d["assets"].append(asset)
    store.update(draft_id, add)
    store.emit(draft_id, "pipeline", {"asset": asset, "step": "queued", "text": f"Received {name}"})
    audit.record_for(store, "data_import", actor, "Home agent" if actor == "agent" else None, draft_id=draft_id,
                     move="add_asset", status="allowed", source={"kind": kind, **{k: v for k, v in source.items() if k != "sha256"}},
                     filename=filename, sha256=raw_sha,
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


LABEL_SHARE = 0.5  # a table at most half the size of another, that joins it by a key, is its label file (15.6)


def _label_role(store: DraftStore, draft_id: str, asset_id: str, frame) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Whether the new table is the label file of the table already here ("labels"), or the table whose label file
    arrived first ("main_after_labels"): the smaller of the two, if a key joins them at about one row per label."""
    import pandas as pd

    from ..studio import labels as label_review

    draft = store.get(draft_id)
    main = next((a for a in reversed(draft["assets"]) if a["id"] != asset_id and a.get("status") == "ready"
                 and a.get("kind") != "labels" and a.get("clean_file")), None)
    if main is None:
        return None, None, None
    other = pd.read_parquet(store.data_dir(draft_id) / main["clean_file"])
    small, big = (frame, other) if len(frame) <= len(other) else (other, frame)
    if len(small) > LABEL_SHARE * len(big):
        return None, None, None
    if not (set(map(str, small.columns)) - set(map(str, big.columns))):
        return None, None, None  # every column is the table's own: a sample of the same table, not a file of known cases
    review = label_review.review(big, small)
    if not review["key"]:
        return None, None, None
    summary = {"key": review["key"], "rows_marked": review["rows_marked"], "rows": int(len(big)), "label_rows": review["label_rows"],
               "not_found": review["not_found"], "disagreeing_columns": review["disagreeing_columns"],
               "refused": [{"key": c["key"], "reason": c["reason"]} for c in review["candidates"] if c["verdict"] == "refused"][:4]}
    return ("labels" if small is frame else "main_after_labels"), summary, main


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
        role, labels_summary, other = _label_role(store, draft_id, asset_id, frame)
        if role == "labels":  # 15.6: a label file next to the table, kept apart; the table stays the one analysed
            out = store.data_dir(draft_id) / "labels_clean.parquet"
            _to_parquet(frame, out)
            _record(store, out)
            final: dict[str, Any] = {}

            def put_labels(d):
                for a in d["assets"]:
                    if a["id"] == asset_id:
                        a.update(status="ready", kind="labels", rows=int(len(frame)), columns=int(frame.shape[1]),
                                 clean_file="labels_clean.parquet", label_review=labels_summary)
                        final.update(a)
                d.setdefault("analysis", {})["labels"] = labels_summary
            store.update(draft_id, put_labels)
            store.emit(draft_id, "pipeline", {"asset": asset_id, "step": "ready", "text": "Ready: a label file for your table",
                                              "rows": int(len(frame)), "columns": int(frame.shape[1])})
            if agent is not None:
                agent.data_ready(draft_id, final)
            return final
        if role == "main_after_labels":  # the label file came first: it moves aside, this table is the one analysed
            (store.data_dir(draft_id) / other["clean_file"]).replace(store.data_dir(draft_id) / "labels_clean.parquet")
            _record(store, store.data_dir(draft_id) / "labels_clean.parquet")  # the moved file, in file storage too
            set_asset(store, draft_id, other["id"], kind="labels", clean_file="labels_clean.parquet", label_review=labels_summary)
        set_asset(store, draft_id, asset_id, status="analysing")
        target = (draft.get("understanding") or {}).get("target")
        report = analyze.analyze(frame, target=target if target in frame.columns else None)
        from .chat import rank_targets  # names only: the columns the problem sentence mentions are offered first

        report["profile"]["target_candidates"] = rank_targets(report["profile"], draft.get("problem") or "")
        if labels_summary is not None:
            report["labels"] = labels_summary
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
