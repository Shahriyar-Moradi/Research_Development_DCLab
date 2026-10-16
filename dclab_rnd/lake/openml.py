"""Real tables for the decision models (26.2): OpenML's public datasets as labelled decision samples.

OpenML describes each dataset's columns itself: which one is the row identifier, which is the default target, which
are to be ignored, and each column's type (numeric, nominal, string, date). Those marks are the labels (origin
``public_catalog``). The features are the same bands DCLab computes on a person's table (``ledger.column_state`` over
``studio.data.profile_table``), so a model learns one input whatever the source: each dataset's data file is fetched
once, profiled, and deleted; only its bands are kept, never a value.

``python -m dclab_rnd.lake openml [--limit 1200]`` fetches into ``data/downloads/openml/`` (not committed; resumable:
a dataset already profiled is not fetched again) and ``make lake`` reads it. The owner's yes to the download:
2026-10-16. Each sample keeps its dataset's licence; a dataset without a licence that allows reuse is skipped.
"""

from __future__ import annotations

import json
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from dclab_core import lake as L

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "downloads" / "openml"
API = "https://www.openml.org/api/v1/json"
AGENT = "DCLab-R&D lake builder (26.2)"
OPEN_LICENCES = ("public", "cc0", "cc-by", "cc by", "cc-by-sa", "pddl", "odc-by", "odbl", "mit", "apache", "gpl", "cc-by-nc")
MAX_CELLS = 2_000_000
ROLE = {"numeric": "numeric", "nominal": "categorical_code", "string": "free_text", "date": "datetime"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _json(url: str) -> dict[str, Any]:
    return json.loads(_get(url))


def _quality(d: dict[str, Any], name: str) -> float | None:
    for q in d.get("quality") or []:
        if q.get("name") == name:
            try:
                return float(q.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def candidates(limit: int) -> list[dict[str, Any]]:
    """Active tabular datasets of a usable size, one version per name, the most-used shapes first."""
    path = CACHE / "list.json"
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows, offset = [], 0
        while True:
            try:
                page = _json(f"{API}/data/list/limit/1000/offset/{offset}/status/active")["data"]["dataset"]
            except (urllib.error.HTTPError, KeyError):
                break
            rows += page
            offset += 1000
            if len(page) < 1000 or offset >= 10000:
                break
            time.sleep(0.5)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
    seen, out = set(), []
    for d in rows:
        n, f = _quality(d, "NumberOfInstances"), _quality(d, "NumberOfFeatures")
        if not n or not f or n < 100 or f < 3 or f > 300 or n * f > MAX_CELLS:
            continue
        key = str(d.get("name", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out[:limit]


def _licence_ok(licence: str | None) -> bool:
    text = str(licence or "").lower()
    return any(k in text for k in OPEN_LICENCES)


def profile_dataset(did: int) -> dict[str, Any] | None:
    """One dataset's description, its columns' OpenML marks and their DCLab bands (the data file read once, deleted)."""
    out_path = CACHE / "datasets" / f"{did}.json"
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    desc = _json(f"{API}/data/{did}")["data_set_description"]
    record: dict[str, Any] = {"did": did, "name": desc.get("name"), "version": desc.get("version"), "licence": desc.get("licence"),
                              "target": desc.get("default_target_attribute"), "row_id": desc.get("row_id_attribute"),
                              "ignore": desc.get("ignore_attribute"), "skipped": None}
    if not _licence_ok(desc.get("licence")):
        record["skipped"] = f"licence {desc.get('licence')!r} not known to allow reuse"
        out_path.write_text(json.dumps(record), encoding="utf-8")
        return record
    features = _json(f"{API}/data/features/{did}")["data_features"]["feature"]
    parquet = desc.get("parquet_url")
    if not parquet:
        record["skipped"] = "no parquet file"
        out_path.write_text(json.dumps(record), encoding="utf-8")
        return record
    import pandas as pd

    from ..ledger import column_state
    from ..studio.data import profile_table

    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "data.parquet"
        file.write_bytes(_get(parquet, timeout=120))
        frame = pd.read_parquet(file)
    profile = profile_table(frame)
    rows = int(len(frame))
    by_name = {c.get("name"): c for c in profile.get("columns") or []}
    record["rows"] = rows
    record["columns"] = [{"name": f.get("name"), "openml_type": f.get("data_type"), "is_target": str(f.get("is_target")) == "true",
                          "is_row_identifier": str(f.get("is_row_identifier")) == "true", "is_ignore": str(f.get("is_ignore")) == "true",
                          "distinct_nominal": len(f.get("nominal_value") or []) if isinstance(f.get("nominal_value"), list) else None,
                          "state": column_state(by_name[f.get("name")], rows) if f.get("name") in by_name else None}
                         for f in features]
    out_path.write_text(json.dumps(record), encoding="utf-8")
    return record


def fetch(limit: int = 1200, log=print) -> dict[str, int]:
    counts: Counter = Counter()
    for i, d in enumerate(candidates(limit)):
        did = int(d["did"])
        try:
            rec = profile_dataset(did)
            counts["skipped" if rec is None or rec.get("skipped") else "profiled"] += 1
        except Exception as exc:  # noqa: BLE001 — one dataset that cannot be read costs itself only
            counts["failed"] += 1
            (CACHE / "datasets").mkdir(parents=True, exist_ok=True)
            (CACHE / "datasets" / f"{did}.json").write_text(json.dumps({"did": did, "skipped": f"{type(exc).__name__}: {str(exc)[:200]}"}), encoding="utf-8")
        if i % 50 == 0:
            log(f"{i} datasets: {dict(counts)}")
        time.sleep(0.2)
    return dict(counts)


def records() -> Iterator[dict[str, Any]]:
    for path in sorted((CACHE / "datasets").glob("*.json")) if (CACHE / "datasets").is_dir() else []:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not rec.get("skipped") and rec.get("columns"):
            yield rec


def samples(dropped: Counter) -> list[L.DecisionSample]:
    """Every profiled dataset's columns as samples of three points, and the dataset as samples of two."""
    from dclab_core.decision_export import request  # noqa: F401  (the same input text as every other source)

    from . import _state_text

    out: list[L.DecisionSample] = []
    for rec in records():
        group = f"openml:{rec['did']}"
        split = L.split_of(group)
        target = rec.get("target") if isinstance(rec.get("target"), str) and "," not in str(rec.get("target")) else None
        ignore = {s.strip() for s in str(rec.get("ignore") or "").split(",") if s.strip()}
        prov = lambda conf=1.0: L.Provenance(source="openml", origin="public_catalog", record_ids=[f"openml:{rec['did']}"],  # noqa: E731
                                             consent="public", licence=str(rec.get("licence") or "")[:120], group=group, split=split)
        states = []
        for c in rec["columns"]:
            state = dict(c.get("state") or {})
            if not state or c.get("is_target"):
                continue
            state["target"] = target
            states.append(state)
            name = str(c.get("name"))
            sid = f"openml{rec['did']}:{L.fingerprint(name)[:12]}"
            if name in ignore and not c.get("is_row_identifier"):
                dropped["an ignored column (its role is not said)"] += 1
                continue
            options = ["yes", "no"]
            out.append(L.DecisionSample(sample_id=sid + ":id", decision_point="column.is_identifier", subject_kind="column", subject_ref=name[:200],
                                        features=state, state_text=_state_text("column.is_identifier", state, options), options=options,
                                        label=bool(c.get("is_row_identifier")), label_confidence=1.0 if c.get("is_row_identifier") else 0.9, provenance=prov()))
            role = "identifier" if c.get("is_row_identifier") else ROLE.get(str(c.get("openml_type")))
            if role:
                options = ["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"]
                out.append(L.DecisionSample(sample_id=sid + ":role", decision_point="column.semantic_role", subject_kind="column", subject_ref=name[:200],
                                            features=state, state_text=_state_text("column.semantic_role", state, options), options=options,
                                            label=role, label_confidence=0.9, provenance=prov()))
        target_col = next((c for c in rec["columns"] if c.get("is_target")), None)
        if target and target_col is not None and states:
            task = ("regression" if target_col.get("openml_type") == "numeric" else
                    "binary" if target_col.get("distinct_nominal") == 2 else "multiclass" if (target_col.get("distinct_nominal") or 0) > 2 else None)
            table = {"rows": rec.get("rows"), "target": target, "target_state": target_col.get("state"), "columns": len(states)}
            if task:
                options = ["binary", "multiclass", "regression"]
                out.append(L.DecisionSample(sample_id=f"openml{rec['did']}:task", decision_point="task.framing", subject_kind="dataset",
                                            subject_ref=f"openml:{rec['did']}", features=table, state_text=_state_text("task.framing", table, options),
                                            options=options, label=task, provenance=prov()))
            listing = {"columns": [{"name_tokens": s.get("name_tokens"), "dtype": s.get("dtype"), "uniqueness": s.get("uniqueness"),
                                    "cardinality": s.get("cardinality")} for s in states][:59] + [{"name_tokens": (target_col.get("state") or {}).get("name_tokens"),
                                    "dtype": (target_col.get("state") or {}).get("dtype"), "uniqueness": (target_col.get("state") or {}).get("uniqueness"),
                                    "cardinality": (target_col.get("state") or {}).get("cardinality")}]}
            names = [str(c.get("name")) for c in rec["columns"] if c.get("state") and not c.get("is_target")][:59] + [target]  # the outcome always among the options
            names = sorted(set(names))  # never in a position a model could learn (the outcome was last)
            out.append(L.DecisionSample(sample_id=f"openml{rec['did']}:target", decision_point="target.column", subject_kind="dataset",
                                        subject_ref=f"openml:{rec['did']}", features=listing, state_text=_state_text("target.column", listing, names),
                                        options=names, label=target, provenance=prov()))
    return out
