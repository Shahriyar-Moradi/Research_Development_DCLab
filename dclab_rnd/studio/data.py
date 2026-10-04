"""Load and profile a table; built-in sample datasets with their known contracts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_rnd.agentic.catalog import DATASETS, ROOT

from .store import INDUSTRIES, safe_name  # noqa: F401  (re-exported for the server)

TARGET_NAMES = re.compile(r"(^|[_\s])(target|label|class|y|outcome|churn(ed)?|default(ed)?|fraud|accepted|converted|"
                          r"response|purchase|survived|diagnosis|risk|price|sales|cnt|count|amount|revenue|demand)([_\s]|$)", re.I)
ID_NAMES = re.compile(r"(^|[_\s])(id|uuid|guid|key|index|number|no|code)$|(^|[_\s])(id|uuid)([_\s]|$)|_id$|id$", re.I)
TIME_NAMES = re.compile(r"(date|time|timestamp|datetime|_at$|created|updated)", re.I)
MAX_PREVIEW_VALUES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path, low_memory=False)


def _datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty or sample.str.len().median() < 6:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def _text_like(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return False
    return bool(sample.str.len().mean() >= 40 and sample.nunique() / len(sample) > 0.5)


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if _datetime_like(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _text_like(series):
        return "text"
    return "categorical"


def profile_table(frame: pd.DataFrame) -> dict[str, Any]:
    """Shape, per-column facts and candidate roles. Nothing here looks at a target."""
    columns = []
    for name in frame.columns:
        series = frame[name]
        kind = column_kind(series)
        non_null = series.dropna()
        unique = int(non_null.nunique())
        unique_ratio = float(unique / max(len(non_null), 1))
        preview = [str(v)[:40] for v in non_null.unique()[:MAX_PREVIEW_VALUES]]
        id_like = bool(unique_ratio >= 0.98 and len(non_null) > 20 and (kind != "numeric" or (pd.to_numeric(non_null, errors="coerce") % 1 == 0).all()))
        columns.append({
            "name": str(name), "kind": kind, "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()), "unique": unique, "unique_ratio": unique_ratio,
            "preview": preview,
            "id_like": id_like and kind not in ("datetime", "text"), "name_id_like": bool(ID_NAMES.search(str(name))) and kind not in ("datetime", "text"),
            "time_like": kind == "datetime" or (bool(TIME_NAMES.search(str(name))) and kind == "numeric"),
            "text_like": kind == "text",
            "target_name_like": bool(TARGET_NAMES.search(str(name))),
            "low_cardinality": unique <= 20,
        })
    targets = sorted(
        [c for c in columns if c["kind"] != "text" and c["unique"] >= 2 and not c["id_like"]],
        key=lambda c: (-int(c["target_name_like"]), -int(c["low_cardinality"] and c["kind"] != "datetime"), c["missing_rate"]),
    )
    return {
        "rows": int(len(frame)), "columns": columns, "column_count": int(frame.shape[1]),
        "memory_mb": float(frame.memory_usage(deep=True).sum() / 1e6),
        "duplicate_row_rate": float(frame.duplicated().mean()) if len(frame) else 0.0,
        "target_candidates": [c["name"] for c in targets[:8]],
        "time_candidates": [c["name"] for c in columns if c["time_like"]],
        "id_candidates": [c["name"] for c in columns if c["id_like"] or c["name_id_like"]],
        "text_candidates": [c["name"] for c in columns if c["text_like"]],
    }


def detect_task(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    series = frame[target].dropna()
    unique = series.nunique()
    numeric = pd.api.types.is_numeric_dtype(series)
    if unique == 2:
        values = sorted(series.unique(), key=str)
        positive = _positive_label(values)
        rate = float((series == positive).mean())
        return {"task": "binary", "classes": [str(v) for v in values], "positive_label": str(positive), "positive_rate": rate,
                "note": f"Two classes; `{positive}` is treated as the positive class ({rate:.1%} of rows)."}
    if unique <= 20 and (not numeric or (series % 1 == 0).all()):
        counts = series.value_counts()
        return {"task": "multiclass", "classes": [str(v) for v in counts.index.tolist()], "n_classes": int(unique),
                "smallest_class": int(counts.min()), "note": f"{unique} classes; the smallest has {int(counts.min())} rows."}
    if numeric:
        return {"task": "regression", "mean": float(series.mean()), "std": float(series.std()),
                "note": f"Continuous target (mean {series.mean():.3g}, std {series.std():.3g})."}
    return {"task": "multiclass", "classes": [str(v) for v in series.value_counts().index[:50].tolist()], "n_classes": int(unique),
            "note": f"{unique} distinct labels; consider grouping rare labels before modeling."}


def _positive_label(values: list[Any]) -> Any:
    text = {str(v).strip().lower(): v for v in values}
    for key in ("1", "true", "yes", "y", "positive", "churn", "fraud", "default", "accepted", "bad", "spam"):
        if key in text:
            return text[key]
    numeric = [v for v in values if isinstance(v, (int, float, np.integer, np.floating))]
    if len(numeric) == 2:
        return max(numeric)
    return values[-1]


# ---------------------------------------------------------------------- samples

SAMPLE_META = {
    "hyperack": {"target": "hyper_ack", "industry": "logistics and delivery", "identifiers": [], "time": "created_date",
                 "goal": "Predict whether a delivery order will be accepted, at the moment the order is created."},
    "telco_churn": {"target": "Churn", "industry": "telecom", "identifiers": ["customerID"], "time": None,
                    "goal": "Predict which customers will churn, from their current account and billing fields."},
}
PUBLIC_GOALS = {
    "adult": "Predict whether income exceeds 50K from census attributes.",
    "bank_marketing": "Predict whether a client subscribes to a term deposit, before the marketing call.",
    "breast_cancer": "Classify a tumor as malignant from cell-nucleus measurements.",
    "heart_disease": "Predict heart disease from clinical test results.",
    "credit_default": "Predict next-month credit card default from billing history.",
    "german_credit": "Predict credit risk from loan application fields.",
    "mushroom": "Classify mushrooms as poisonous from physical traits.",
    "spambase": "Classify email as spam from word and character frequencies.",
    "online_shoppers": "Predict a purchase from session-start information.",
    "wine_quality": "Predict good wine quality from physicochemical tests.",
}


EXPANSION_INDUSTRY = {"credit_card_fraud": "fintech and banking", "letter_recognition": "general",
                      "bike_sharing_daily": "logistics and delivery", "ecommerce_clothing_reviews": "retail and e-commerce"}
EXPANSION_TASK = {"binary_imbalanced": "binary", "multiclass": "multiclass", "timeseries_regression": "regression", "text_tabular_binary": "binary"}


def _expansion_specs():
    from dclab_rnd.expansion import datasets as expansion

    return expansion


def sample_catalog(root: Path = ROOT) -> list[dict[str, Any]]:
    """Datasets the R&D already studied, each with the contract it wrote for them.

    The ten UCI tables and the two project tables ship with the repository. The four
    Kaggle/GitHub datasets of the expansion campaign (imbalanced fraud, multiclass,
    time series, text + tabular) are fetched on first use with a pinned SHA-256.
    """
    out = []
    for key, policy in DATASETS.items():
        if key in SAMPLE_META:
            meta = SAMPLE_META[key]
            source = root / ("data/project/hyperack/hyper_ackt-dataset.csv" if key == "hyperack" else "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
            available = source.exists()
            name = "HyperAck delivery acceptance" if key == "hyperack" else "Telco customer churn"
            goal, industry = meta["goal"], meta["industry"]
        else:
            source = root / "data/public" / key / "X.parquet"
            available = source.exists()
            name = key.replace("_", " ").capitalize()
            goal, industry = PUBLIC_GOALS.get(key, ""), "general"
        out.append({"key": key, "name": name, "goal": goal, "industry": industry, "available": available, "source": "repository",
                    "blocked": policy["blocked"], "decision": policy["decision"], "task": "binary", "rows": None})
    expansion = _expansion_specs()
    for key, spec in expansion.SPECS.items():
        cached = expansion.raw_path(root, spec).exists()
        out.append({"key": key, "name": spec.name, "goal": spec.description, "industry": EXPANSION_INDUSTRY.get(key, "general"),
                    "available": True, "source": "cached" if cached else "download on first use (pinned SHA-256)",
                    "blocked": list(spec.blocked_features), "decision": spec.decision_time_contract,
                    "task": EXPANSION_TASK[spec.task_type], "task_type": spec.task_type, "rows": spec.source_rows, "url": spec.url})
    return out


def load_sample(key: str, root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The sample table as one frame (target included) plus a suggested contract."""
    expansion = _expansion_specs()
    if key in expansion.SPECS:
        return _load_expansion_sample(expansion, key, root)
    if key not in DATASETS:
        raise KeyError(key)
    policy = DATASETS[key]
    if key == "hyperack":
        frame = pd.read_csv(root / "data/project/hyperack/hyper_ackt-dataset.csv")
        target, identifiers, time_column = "hyper_ack", [], "created_date"
        for column in ("first_created_at",):
            if column in frame:
                frame = frame.drop(columns=[column])
    elif key == "telco_churn":
        frame = pd.read_csv(root / "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].astype(str).str.strip(), errors="coerce")
        target, identifiers, time_column = "Churn", ["customerID"], None
    else:
        directory = root / "data/public" / key
        frame = pd.read_parquet(directory / "X.parquet")
        frame["target"] = pd.read_parquet(directory / "y.parquet")["target"].astype(int).to_numpy()
        target, identifiers, time_column = "target", [], None
    suggestion = {
        "target": target,
        "forbidden": [{"column": c, "reason": "Blocked by the DCLab prediction contract for this dataset: " + policy["decision"]} for c in policy["blocked"] if c in frame and c not in identifiers],
        "identifiers": [c for c in identifiers if c in frame],
        "time_column": time_column if time_column in frame else None,
        "prediction_moment": policy["decision"],
    }
    return frame, suggestion


def _load_expansion_sample(expansion, key: str, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = expansion.SPECS[key]
    bundle = expansion.load_bundle(root, key)  # downloads once, verifies the pinned hash, never subsamples here
    frame = bundle.X.copy()
    if bundle.class_labels:
        frame[spec.target] = [bundle.class_labels[int(i)] for i in bundle.y]
    else:
        frame[spec.target] = bundle.y.to_numpy()
    suggestion = {
        "target": spec.target,
        "task": EXPANSION_TASK[spec.task_type],
        "forbidden": [{"column": c, "reason": r} for c, r in spec.blocked_features.items() if c in frame],
        "identifiers": [c for c in spec.identifier_columns if c in frame and c != spec.time_column],
        "time_column": spec.time_column if spec.time_column in frame else None,
        "group_column": spec.group_column if spec.group_column in frame else None,
        "text_columns": [c for c in spec.text_columns if c in frame],
        "prediction_moment": spec.decision_time_contract,
        "metric": spec.primary_metric,
        "positive_label": "1" if EXPANSION_TASK[spec.task_type] == "binary" else None,
    }
    return frame, suggestion


# ---------------------------------------------------------------------- attaching data to a project


class DataError(ValueError):
    """A table the notebook cannot use; the message is safe to show."""


def attach_data(store, project_id: str, filename: str) -> dict[str, Any]:
    """Register ``data/<filename>`` (already written) as the project's table and profile it."""
    path = store.data_dir(project_id) / filename
    try:
        frame = load_table(path)
    except Exception as exc:  # noqa: BLE001 — pandas raises many types; the file is simply not a table
        path.unlink(missing_ok=True)
        raise DataError(f"Could not read the file as a table: {type(exc).__name__}") from None
    if frame.shape[1] < 2 or len(frame) < 30:
        path.unlink(missing_ok=True)
        raise DataError("The table needs at least 2 columns and 30 rows")
    for old in store.data_dir(project_id).iterdir():
        if old.name != filename:
            old.unlink()
    project = store.get(project_id)
    project["data"] = {"filename": filename, "rows": int(len(frame)), "columns": [str(c) for c in frame.columns],
                       "sha256": sha256(path), "profile": profile_table(frame)}
    project["contract"], project["proposal"], project["suggestion"] = None, None, None
    store.save(project)
    store.clear_stages(project_id)
    store.log(project_id, "data_attached", {"filename": filename, "rows": project["data"]["rows"], "columns": len(project["data"]["columns"])})
    return store.get(project_id)


def use_sample(store, project_id: str, key: str, root: Path = ROOT) -> dict[str, Any]:
    """Copy a sample dataset into the project and remember the contract the R&D wrote for it."""
    frame, suggestion = load_sample(key, root)
    name = f"{key}.csv"
    frame.to_csv(store.data_dir(project_id) / name, index=False)
    project = attach_data(store, project_id, name)
    project["suggestion"] = {**suggestion, "sample": key}
    return store.save(project)
