"""The main product's numbers as outcomes of its decisions (package 23.5).

Its ``experiments`` rows carry a ``result`` (numbers of the run) and its ``project_decision_records`` rows name the
``experiment_id`` they belong to: an experiment's result is the outcome of every decision linked to it. A promotion is
a ``model_versions`` row there; that table is not in the field lists read at this commit (``FIELDS_MISSING``), so a
promotion outcome waits for 23.12's refresh. Nothing here reads a table's values: the result is numbers of a metric.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from dclab_core.adapters.main_product import recorded
from dclab_core.adapters.main_product_lineage import check_columns
from dclab_core.outcomes import Outcome, judge

FIELDS_MISSING = {
    "model_versions": "not in the field lists read at this commit (23.5): a promotion outcome is mapped once 23.12 refreshes them",
    "experiments.result.reference": "the result holds the run's numbers only; the verdict is unknown unless the caller passes the "
                                    "cross-validation estimate it was selected on",
}


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def from_experiment(row: Mapping[str, Any], decisions: Iterable[Any], metric: str, *, higher_is_better: Optional[bool] = None,
                    reference: Optional[float] = None, consent: str = "operate") -> list[Outcome]:
    """One ``experiments`` row's ``result[metric]`` as the outcome of every decision record whose ``experiment`` is it."""
    check_columns("experiments", row)
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    value = result.get(metric)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return []
    out = []
    for d in decisions:
        if getattr(d, "experiment", None) != str(row.get("id")):
            continue
        digest = hashlib.sha1(f"{row.get('id')}|{d.id}|{metric}".encode("utf-8")).hexdigest()[:20]
        out.append(Outcome(id=f"mp-oc-{digest}", record=d.id, point=d.point, metric=str(metric)[:80], value=float(value), reference=reference,
                           higher_is_better=higher_is_better, observed="holdout", verdict=judge(value, reference, higher_is_better),
                           by="experiment", workspace=str(row.get("workspace_id") or "") or None, project=str(row.get("project_id") or "") or None,
                           run=str(row.get("id")), at=_iso(row.get("ended_at") or row.get("created_at")), consent=consent))
    return out


__all__ = ["FIELDS_MISSING", "from_experiment", "recorded"]
