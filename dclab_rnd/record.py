"""Canonical experiment record used across the repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    fingerprint: str
    source_path: str
    suite: str
    dataset: str
    experiment_id: str
    experiment_name: str
    model: str
    model_family: str
    mode: str
    optimization: str
    status: str
    deployment_eligible: Optional[bool]
    has_known_leakage: Optional[bool]
    feature_count: Optional[int]
    train_rows: Optional[int]
    test_rows: Optional[int]
    roc_auc: Optional[float]
    avg_precision: Optional[float]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    fit_seconds: Optional[float]
    total_seconds: Optional[float]
    threshold: Optional[float]
    timestamp: str
    strategy: str
    notes: str
    schema_version: Optional[int]
    has_provenance: bool

    def to_dict(self) -> dict:
        return asdict(self)

