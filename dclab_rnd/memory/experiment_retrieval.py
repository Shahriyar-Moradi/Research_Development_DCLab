"""Structured past experiment and outcome retrieval (MEM-01).

Allows agents and search operators to query past machine learning experiments, hypotheses,
and verified outcomes across problem kinds, dataset scales, and scientific constraints.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class PastExperimentRecord:
    """Historical record of an attempted ML experiment and its verified outcome."""

    experiment_id: str
    project_id: str
    problem_kind: str
    dataset_name: str
    rows: int
    columns: int
    metric: str
    recipe: str
    family: str
    params: dict[str, Any]
    status: str  # "accepted", "rejected", "failed"
    score: float
    baseline_score: float
    lift: float
    has_time: bool = False
    has_groups: bool = False
    class_imbalance: Optional[float] = None
    rationale: str = ""
    error_message: Optional[str] = None
    timestamp: str = ""


class ExperimentRetrievalIndex:
    """In-memory and file-backed index of past experiment outcomes."""

    def __init__(self, records: Optional[list[PastExperimentRecord]] = None):
        self._records: list[PastExperimentRecord] = list(records or [])

    def add(self, record: PastExperimentRecord) -> None:
        self._records.append(record)

    def count(self) -> int:
        return len(self._records)

    def query(
        self,
        problem_kind: Optional[str] = None,
        rows: Optional[int] = None,
        has_time: Optional[bool] = None,
        has_groups: Optional[bool] = None,
        status: Optional[str] = None,
        family: Optional[str] = None,
        top_k: int = 5,
    ) -> list[tuple[PastExperimentRecord, float]]:
        """Query past experiments by scientific criteria, returning (record, similarity_score) sorted best-first."""
        if not self._records:
            return []

        scored: list[tuple[PastExperimentRecord, float]] = []
        for rec in self._records:
            sim = 1.0

            # Problem kind match (critical constraint)
            if problem_kind is not None:
                if rec.problem_kind.lower() == problem_kind.lower():
                    sim += 2.0
                elif rec.problem_kind.split("_")[0] == problem_kind.split("_")[0]:
                    sim += 0.5
                else:
                    sim -= 2.0

            # Status filter
            if status is not None and rec.status.lower() != status.lower():
                continue

            # Family filter
            if family is not None and rec.family.lower() != family.lower():
                continue

            # Temporal structure match
            if has_time is not None:
                if rec.has_time == has_time:
                    sim += 0.5
                else:
                    sim -= 0.5

            # Group structure match
            if has_groups is not None:
                if rec.has_groups == has_groups:
                    sim += 0.5
                else:
                    sim -= 0.5

            # Row scale similarity (log-scale distance)
            if rows is not None and rows > 0 and rec.rows > 0:
                log_ratio = abs(math.log10(rows) - math.log10(rec.rows))
                scale_sim = max(0.0, 1.0 - (log_ratio / 2.0))
                sim += scale_sim

            # Performance bonus for accepted positive lifts
            if rec.status == "accepted" and rec.lift > 0:
                sim += min(rec.lift, 0.5)

            scored.append((rec, sim))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        """Persist index to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self._records]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExperimentRetrievalIndex:
        """Load index from JSON file."""
        if not path.exists():
            return cls([])
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [PastExperimentRecord(**item) for item in data]
        return cls(records)
