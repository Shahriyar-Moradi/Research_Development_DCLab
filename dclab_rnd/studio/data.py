"""Load and profile a table; built-in sample datasets with their known solutions."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_rnd.agentic.catalog import DATASETS, ROOT
from dclab_rnd.categoricals import declared_categorical

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


def data_card(tables: dict[str, pd.DataFrame], name: str) -> dict[str, Any] | None:
    """The tables' data card (package 19.11, ``dclab_core.datacard``): columns in bands, relations and problems, never a
    row; category names replaced by codes. None when it cannot be built: what it describes stands without it."""
    import sys

    try:
        from dclab_core.datacard import card_of

        return card_of(tables, name=str(name)[:200] or "table").model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        print(f"data card: not built: {type(exc).__name__}", file=sys.stderr, flush=True)
        return None


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
        numbers = pd.to_numeric(non_null, errors="coerce") if kind == "numeric" and id_like else None
        sequence_like = bool(numbers is not None and numbers.notna().all() and unique
                             and float(numbers.max() - numbers.min() + 1) <= 1.05 * unique)  # 1, 2, 3, …: a counter, not an amount
        columns.append({
            "name": str(name), "kind": kind, "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()), "unique": unique, "unique_ratio": unique_ratio,
            "preview": preview,
            "id_like": id_like and kind not in ("datetime", "text"), "name_id_like": bool(ID_NAMES.search(str(name))) and kind not in ("datetime", "text"),
            "sequence_like": sequence_like,
            "time_like": kind == "datetime" or (bool(TIME_NAMES.search(str(name))) and kind == "numeric"),
            "text_like": kind == "text",
            "target_name_like": bool(TARGET_NAMES.search(str(name))),
            "low_cardinality": unique <= 20,
        })
    targets = sorted(  # a column named as a key (customer_id, order_no) is never proposed: a person may still name it
        [c for c in columns if proposable_target(c)],
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


ROW_NUMBER_NAMES = re.compile(r"^(unnamed:?\s*\d+|index|level_\d+|row(_?(id|no|number|num))?)$", re.I)
# a key's name: customer_id, order_request_id, customerId, uuid, order_no; not is_paid, valid or covid (ID_NAMES is wider)
KEY_NAMES = re.compile(r"((^|[_\s.-])(id|ids|uuid|guid|key|no|nr|number|ref|code)$|^(id|uuid|guid)([_\s.-]|$))", re.I)
CAMEL_KEY = re.compile(r"[a-z0-9](Id|ID|Uuid|UUID|Key|No)$")


def key_named(name: Any) -> bool:
    return bool(KEY_NAMES.search(str(name)) or CAMEL_KEY.search(str(name)))


def proposable_target(column: dict[str, Any]) -> bool:
    """Whether DCLab may offer a column as a likely outcome: allowed (:func:`target_refusal`), not text, and not named as a
    key. A key-named column with few values (customer_id with 3) is never offered; a person may still name it."""
    return target_refusal(column) is None and column.get("kind") != "text" and not key_named(column.get("name", ""))


OUTCOME_NAMES = re.compile(r"^(target|label|labels|outcome|y|class|is_case)$", re.I)


def named_outcome(profile: dict[str, Any], text: str = "") -> str | None:
    """The outcome a person named, never a guess: a column whose name appears as whole words in what they wrote, else a
    column the table itself names as its outcome (``target``, ``label``, ``outcome``). Only allowed columns; None when
    nothing is named, and then DCLab asks."""
    columns = [c for c in profile.get("columns") or [] if target_refusal(c) is None and c.get("kind") != "text"]
    words = f" {re.sub(r'[^a-z0-9_]+', ' ', str(text).lower())} "
    said = [c["name"] for c in columns if f" {str(c['name']).lower()} " in words]
    if len(said) == 1:
        return said[0]
    own = [c["name"] for c in columns if OUTCOME_NAMES.match(str(c["name"]))]
    return own[0] if len(own) == 1 else None


def target_refusal(column: dict[str, Any] | None) -> str | None:
    """Why a profiled column can never be the outcome, or None: a value the same on every row, a key different on every
    row, a saved row number, or free text. Read from the profile only: names and counts, no relation to anything."""
    if not column:
        return None
    name = column.get("name", "")
    if (column.get("unique") or 0) < 2:
        return f"`{name}` has one value on every row: there is nothing to predict"
    if ROW_NUMBER_NAMES.match(str(name)):
        return f"`{name}` is a saved row number: its order is how the file was written, not an outcome"
    keyish = key_named(name)
    # different on almost every row: a key when it is text, named as a key, or counts 1, 2, 3, …; whole-number amounts
    # (salary, price) are as distinct and stay allowed
    if column.get("id_like") and (column.get("kind") != "numeric" or keyish or column.get("sequence_like")):
        return f"`{name}` is different on almost every row ({column.get('unique_ratio', 0):.0%} unique): it is a key, not an outcome"
    if keyish and (column.get("unique_ratio") or 0) > 0.5:  # an order id repeated by multi-drop orders
        return f"`{name}` is named as a key and {column.get('unique_ratio', 0):.0%} of its values are distinct: it is a key, not an outcome"
    if column.get("kind") == "text":
        return f"`{name}` is free text: name the outcome it describes as its own column"
    return None


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
    """Datasets the R&D already studied, each with the solution it wrote for them.

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
    """The sample table as one frame (target included) plus a suggested solution."""
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
        "forbidden": [{"column": c, "reason": "Blocked by the R&D solution for this dataset: " + policy["decision"]} for c in policy["blocked"] if c in frame and c not in identifiers],
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


def safe_name(filename: str) -> str:
    """A file name that cannot leave the data folder: the last path part, safe characters only, no leading dots."""
    base = re.split(r"[\\/]", str(filename or ""))[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip("._")[:120]
    return base


def data_path(store, project: dict[str, Any]) -> Path:
    """The project's table, checked against the SHA-256 recorded when it was attached (package 9.3): a file that changed
    on disk, or is missing, is an error here, never other data in a model."""
    from ..storage.files import FileError, files_for

    path = store.data_dir(project["id"]) / project["data"]["filename"]
    try:
        files = files_for(store)
        return files.path(files.key_of(path), expected=project["data"].get("sha256"))
    except FileError as exc:
        raise DataError(str(exc)) from None


def attach_data(store, project_id: str, filename: str, source: dict[str, Any] | None = None, actor: str = "human") -> dict[str, Any]:
    """Register ``data/<filename>`` (already written) as the project's table and profile it. ``source`` says where it
    came from (an upload, a studied sample, a draft); the audit keeps it with the table's SHA-256 (package 10.4)."""
    path = store.data_dir(project_id) / filename
    try:
        frame = load_table(path)
    except Exception as exc:  # noqa: BLE001 — pandas raises many types; the file is simply not a table
        path.unlink(missing_ok=True)
        raise DataError(f"Could not read the file as a table: {type(exc).__name__}") from None
    if frame.shape[1] < 2 or len(frame) < 30:
        path.unlink(missing_ok=True)
        raise DataError("The table needs at least 2 columns and 30 rows")
    from ..storage.files import files_for

    try:  # the new table is recorded first: a storage that cannot take it leaves the project and its old table as they were
        files = files_for(store)
        stored = files.put(files.key_of(path), path)  # key, size, SHA-256, content type (package 9.3)
    except Exception as exc:  # noqa: BLE001 — a name, a folder or a backend the storage refuses
        from ..storage.files import FileError

        path.unlink(missing_ok=True)
        # a storage error says what was wrong with the file; any other error (a database, a bucket) only its type:
        # its text can hold the connection's host, port or user, which never reach a response (CLAUDE.md rule 4)
        raise DataError(str(exc) if isinstance(exc, FileError) else f"The table could not be stored ({type(exc).__name__})") from None
    for old in store.data_dir(project_id).iterdir():
        if old.name != filename and not old.name.endswith(".part"):
            try:
                files.delete(files.key_of(old))  # the file and its record
            except Exception:  # noqa: BLE001 — a leftover the storage never recorded: just remove the file
                old.unlink(missing_ok=True)
    project = store.get(project_id)
    project["data"] = {"filename": filename, "rows": int(len(frame)), "columns": [str(c) for c in frame.columns],
                       "sha256": stored["sha256"], "profile": profile_table(frame), "card": data_card({filename: frame}, filename)}
    project["solution"], project["proposal"], project["suggestion"] = None, None, None
    project["labels"] = None  # its file was removed with the old table, and it was keyed on that table
    if project.get("memory"):
        from . import memory
        memory.data_replaced(project, filename)  # A5.2: what the notes settled was about the old table
    store.save(project)
    store.clear_stages(project_id)
    store.log(project_id, "data_attached", {"filename": filename, "rows": project["data"]["rows"], "columns": len(project["data"]["columns"])})
    from .. import audit

    audit.record_for(store, "data_import", actor, project=project, move="attach_data", status="allowed",
                     source=source or {"kind": "upload"}, filename=filename, sha256=stored["sha256"], rows=project["data"]["rows"],
                     columns=len(project["data"]["columns"]))
    return store.get(project_id)


def attach_labels(store, project_id: str, filename: str, target: str = "is_case", key: list[str] | None = None,
                  per_entity: bool = False, labels: str = "partial", actor: str = "human") -> dict[str, Any]:
    """Register ``data/<filename>`` (already written) as the project's label file (package 15.2). The join key is chosen
    and checked by code (``studio.labels.review``); an inflating key is refused with its numbers. The target column
    ``target`` is 1 where the table's key is in the file. A sample of known cases gives partial labels unless the person
    says the file holds every case (``labels="complete"``). The solution is cleared: the labels define the target."""
    from . import labels as label_review
    from ..storage.files import FileError, files_for

    from . import graph

    project = store.get(project_id)
    verdict = graph.check(project, "attach_labels", actor, filename=filename)  # the move is validated and logged (15.2)
    if not verdict.allowed:
        graph.log(store, project_id, verdict, project)
        raise DataError(verdict.message)
    path = store.data_dir(project_id) / filename
    try:
        table, frame = load_table(data_path(store, project)), load_table(path)
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001 — pandas raises many types; the file is simply not a table
        path.unlink(missing_ok=True)
        raise DataError(f"Could not read the label file as a table: {type(exc).__name__}") from None
    if target in table.columns:
        raise DataError(f"The table already has a column {target!r}: name the label another way")
    try:
        review = label_review.review(table, frame, key=key, per_entity=per_entity)
    except ValueError as exc:
        raise DataError(str(exc)) from None
    from ..ledger import capture  # 19.7: the key that joins the labels, the rule's or the person's

    capture.join_key(review, list(key) if key else None, project=project_id)
    if not review["key"]:
        refused = [c for c in review["candidates"] if c["verdict"] == "refused"][:3]
        raise DataError("No key joins the labels to the table at about one row per label"
                        + ("; refused: " + "; ".join(f"{'+'.join(c['key'])} {c['reason']}" for c in refused) if refused else ""))
    try:
        files = files_for(store)
        stored = files.put(files.key_of(path), path)
    except Exception as exc:  # noqa: BLE001 — as attach_data: a storage error's own words, any other only its type
        path.unlink(missing_ok=True)
        raise DataError(str(exc) if isinstance(exc, FileError) else f"The label file could not be stored ({type(exc).__name__})") from None
    found = review["label_rows_found"]
    provenance = (f"{filename}: {found} of {review['label_rows']} label rows found in the table, joined on {'+'.join(review['key'])} "
                  f"({review['rows_marked']} rows marked)" + ("; a sample of known cases" if labels == "partial" else ""))
    project = store.get(project_id)
    project["labels"] = {"filename": filename, "sha256": stored["sha256"], "target": target, "key": review["key"],
                         "rows": review["label_rows"], "found": found, "not_found": review["not_found"], "rows_marked": review["rows_marked"],
                         "coverage": labels, "per_entity": per_entity, "disagreeing_columns": review["disagreeing_columns"],
                         "candidates": review["candidates"][:12]}
    project["solution"], project["proposal"] = None, None
    from dclab_core.analyse import ROW_NUMBER_NAME

    profile = {c["name"]: c for c in (project["data"].get("profile") or {}).get("columns", [])}
    identifiers = [c for c in table.columns if ROW_NUMBER_NAME.match(str(c)) or (profile.get(c, {}).get("id_like") and c not in review["key"])]
    identifiers += [c for c in review["key"] if c not in identifiers]  # the join key names a case: it identifies, never describes
    project["suggestion"] = {"target": target, "task": "binary", "positive_label": "1", "labels": labels, "label_provenance": provenance,
                             "identifiers": [str(c) for c in identifiers][:50]}
    store.save(project)
    store.clear_stages(project_id)
    graph.log(store, project_id, verdict, store.get(project_id), outcome="done: labels attached")
    store.log(project_id, "labels_attached", {"filename": filename, "key": review["key"], "rows_marked": review["rows_marked"],
                                               "not_found": review["not_found"], "coverage": labels})
    from .. import audit

    audit.record_for(store, "data_import", actor, project=project, move="attach_labels", status="allowed", source={"kind": "labels"},
                     filename=filename, sha256=stored["sha256"], rows=review["label_rows"], columns=int(frame.shape[1]))
    return store.get(project_id)


def sample_categorical(key: str, columns: list[str]) -> list[str]:
    """The columns the R&D declared categorical for a sample (dataset catalog or expansion spec)."""
    expansion = _expansion_specs()
    if key in expansion.SPECS:
        declared = set(expansion.SPECS[key].categorical_columns)
        return [c for c in columns if c in declared]
    return declared_categorical(key, columns) or []


def use_sample(store, project_id: str, key: str, root: Path = ROOT, actor: str = "human") -> dict[str, Any]:
    """Copy a sample dataset into the project and remember the solution the R&D wrote for it."""
    frame, suggestion = load_sample(key, root)
    name = f"{key}.csv"
    frame.to_csv(store.data_dir(project_id) / name, index=False)
    project = attach_data(store, project_id, name, source={"kind": "sample", "key": key}, actor=actor)
    # The cached samples hold factorized category codes; the declaration spares the engine a guess.
    project["data"]["categorical"] = sample_categorical(key, [c for c in frame.columns if c != suggestion["target"]])
    project["suggestion"] = {**suggestion, "sample": key}
    return store.save(project)
