"""Discover and normalize the repository's historical experiment results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .record import ExperimentRecord


RESULT_GLOBS = (
    "general_pipeline/results/*.json",
    "general_pipeline/results_external/*.json",
    "hyperack_exp/results/*.json",
    "safe_leakage_exp/results/*.json",
    "optimized_safe_model/results/*.json",
    "external_projects/*_exp/results/ladder/*.json",
)

SUMMARY_FILES = {"benchmark_dashboard.json", "benchmark_summary.json"}
KNOWN_LEAKAGE_DATASETS = {"hyperack", "bank_marketing", "online_shoppers"}
METRIC_NAMES = (
    "roc_auc",
    "avg_precision",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fit_seconds",
    "threshold",
)
PLAYBOOK_MODEL_BY_EXPERIMENT = {
    "1": "logistic_regression",
    "2": "lightgbm",
    "3": "lightgbm",
    "4": "lightgbm",
    "5": "lightgbm",
    "6": "lightgbm",
    "7": "lightgbm",
    "8": "xgboost",
    "9": "lightgbm",
    "10": "hist_gradient_boosting",
    "11": "extra_trees",
    "12": "lightgbm",
    "13": "stacking_ensemble",
    "14": "soft_vote_blend",
    "15": "lightgbm",
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _metric(value: Any) -> float | None:
    """Normalize harmless floating-point overshoot at probability-metric boundaries."""
    number = _number(value)
    if number is None:
        return None
    if -1e-12 <= number <= 1 + 1e-12:
        return min(1.0, max(0.0, number))
    return number


def _family(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    rules = (
        ("ensemble", ("softvote", "soft_vote", "soft-vote", "stack", "blend", "oof", "bag")),
        ("tabular_transformer", ("tabular_transformer", "fttransformer", "ft-transformer")),
        ("tabpfn", ("tabpfn",)),
        ("lightgbm", ("lightgbm", "lgbm")),
        ("xgboost", ("xgboost", "xgb")),
        ("extra_trees", ("extra_trees", "extratrees")),
        ("random_forest", ("random_forest", "randomforest")),
        ("hist_gradient_boosting", ("hist_gradient", "histgradient", "histgb", "hgb")),
        ("catboost", ("catboost",)),
        ("logistic_regression", ("logistic",)),
        ("linear_svc", ("linear_svc", "linearsvc")),
    )
    for family, tokens in rules:
        if any(token in text for token in tokens):
            return family
    cleaned = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return cleaned[:80] or "unknown"


def _suite(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("general_pipeline/results_external/"):
        return "external_benchmark"
    if value.startswith("general_pipeline/results/"):
        return "hyperack_4way"
    if value.startswith("hyperack_exp/results/"):
        return "hyperack_initial"
    if value.startswith("safe_leakage_exp/results/"):
        return "hyperack_safe"
    if value.startswith("optimized_safe_model/results/"):
        return "hyperack_optimized_safe"
    if value.startswith("external_projects/"):
        return "external_playbook"
    return "unknown"


def _dataset(path: Path, payload: dict[str, Any]) -> str:
    if payload.get("dataset"):
        return str(payload["dataset"])
    match = re.search(r"external_projects/([^/]+)_exp/", path.as_posix())
    return match.group(1) if match else "hyperack"


def _mode(suite: str, payload: dict[str, Any], experiment_name: str) -> str:
    explicit = str(payload.get("mode") or "").lower()
    if explicit:
        return explicit
    if suite in {"hyperack_safe", "hyperack_optimized_safe"}:
        return "safe"
    if suite == "hyperack_initial":
        return "safe" if "leakage_safe" in experiment_name.lower() else "unsafe"
    return "unknown"


def _deployment_eligible(
    dataset: str,
    mode: str,
    suite: str,
    payload: dict[str, Any],
) -> tuple[bool | None, bool | None]:
    raw = payload.get("has_leakage")
    known_leakage = bool(raw) if isinstance(raw, bool) else dataset in KNOWN_LEAKAGE_DATASETS

    if mode == "safe":
        return True, known_leakage
    if mode == "unsafe":
        return (not known_leakage), known_leakage
    if suite == "external_benchmark":
        return (not known_leakage), known_leakage
    return None, known_leakage


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _parse(path: Path, root: Path, payload: dict[str, Any]) -> ExperimentRecord:
    relative = path.relative_to(root)
    suite = _suite(relative)
    metrics = payload.get("metrics", {})
    experiment_id = str(payload.get("exp_id") or relative.stem)
    experiment_name = str(
        payload.get("exp_name") or payload.get("name") or payload.get("model_name") or relative.stem
    )
    dataset = _dataset(relative, payload)
    mode = _mode(suite, payload, experiment_name)
    eligible, known_leakage = _deployment_eligible(dataset, mode, suite, payload)
    model_value = payload.get("model_name") or payload.get("best_model")
    if model_value is None and suite == "external_playbook":
        model_value = PLAYBOOK_MODEL_BY_EXPERIMENT.get(str(experiment_id), experiment_name)
    model = str(model_value or experiment_name)
    optimization = str(
        payload.get("optimization")
        or payload.get("optimization_method")
        or ("optimized" if "optimized" in suite else "unspecified")
    )
    status_value = str(metrics.get("status") or payload.get("status") or "").lower()
    roc_auc = _metric(metrics.get("roc_auc"))
    status = "skipped" if status_value.startswith("skip") else "completed"
    if roc_auc is None and status == "completed":
        status = "incomplete"

    canonical = _canonical_json(payload)
    fingerprint = hashlib.sha256(canonical).hexdigest()
    stable_key = f"{relative.as_posix()}::{experiment_id}"
    run_id = hashlib.sha256(stable_key.encode()).hexdigest()[:16]

    return ExperimentRecord(
        run_id=run_id,
        fingerprint=fingerprint,
        source_path=relative.as_posix(),
        suite=suite,
        dataset=dataset,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        model=model,
        model_family=_family(model, experiment_name, payload.get("strategy")),
        mode=mode,
        optimization=optimization,
        status=status,
        deployment_eligible=eligible,
        has_known_leakage=known_leakage,
        feature_count=_integer(payload.get("feature_count")),
        train_rows=_integer(payload.get("train_rows")),
        test_rows=_integer(payload.get("test_rows")),
        roc_auc=roc_auc,
        avg_precision=_metric(metrics.get("avg_precision")),
        accuracy=_metric(metrics.get("accuracy")),
        precision=_metric(metrics.get("precision")),
        recall=_metric(metrics.get("recall")),
        f1=_metric(metrics.get("f1")),
        fit_seconds=_number(metrics.get("fit_seconds")),
        total_seconds=_number(payload.get("total_elapsed_seconds")),
        threshold=_number(metrics.get("threshold")),
        timestamp=str(payload.get("timestamp") or ""),
        strategy=str(payload.get("strategy") or ""),
        notes=str(payload.get("notes") or ""),
        schema_version=_integer(payload.get("schema_version")),
        has_provenance=isinstance(payload.get("provenance"), dict),
    )


def discover_result_files(root: Path, patterns: Iterable[str] = RESULT_GLOBS) -> list[Path]:
    found = {path.resolve() for pattern in patterns for path in root.glob(pattern)}
    return sorted(found)


def collect_registry(root: Path) -> tuple[list[ExperimentRecord], list[dict[str, str]]]:
    """Return normalized records and validation issues from every known result suite."""
    root = root.resolve()
    records: list[ExperimentRecord] = []
    issues: list[dict[str, str]] = []

    for path in discover_result_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            issues.append({"severity": "error", "path": relative, "message": f"invalid JSON: {exc}"})
            continue
        if path.name in SUMMARY_FILES:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            issues.append({"severity": "warning", "path": relative, "message": "ignored non-experiment JSON"})
            continue
        record = _parse(path, root, payload)
        records.append(record)

        if record.status == "incomplete":
            issues.append({"severity": "warning", "path": relative, "message": "completed result has no ROC-AUC"})
        for metric in ("roc_auc", "avg_precision", "accuracy", "precision", "recall", "f1"):
            value = getattr(record, metric)
            if value is not None and not 0.0 <= value <= 1.0:
                issues.append({"severity": "error", "path": relative, "message": f"{metric}={value} is outside [0, 1]"})

    duplicates: dict[tuple[str, str, str, str], list[ExperimentRecord]] = {}
    for record in records:
        key = (record.suite, record.dataset, record.mode, record.experiment_id)
        duplicates.setdefault(key, []).append(record)
    for key, group in duplicates.items():
        if len(group) > 1:
            paths = ", ".join(item.source_path for item in group)
            issues.append({
                "severity": "warning",
                "path": paths,
                "message": "duplicate canonical experiment key: " + "/".join(key),
            })

    records.sort(key=lambda item: (item.dataset, item.suite, item.mode, item.experiment_id, item.source_path))
    return records, issues
