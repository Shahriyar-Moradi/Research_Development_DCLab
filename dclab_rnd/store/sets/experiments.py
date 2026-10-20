"""The experiments themselves (32.20): every model fitted by the engine, one :class:`dclab_core.store.ExperimentRecord` each,
keyed to its table by the table's hash and described by its shape in bands (never a row): across hundreds of tables, which
approach wins on which kind of data, for M3, M5 and the service.

From the shaped records only: each project's ``models`` stage record (its every candidate with every fold's scores) and its
``final`` record (the holdout's metrics, for the model chosen), and the project's own record for the table's shape.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from dclab_core.datasets import split_of
from dclab_core.store import ExperimentRecord, StoreRecord

from ..layout import Layout
from . import write_set


def _band(rows: Any) -> str:
    try:
        n = int(rows)
    except (TypeError, ValueError):
        return "unknown"
    return "tiny" if n < 1_000 else "small" if n < 10_000 else "medium" if n < 100_000 else "large" if n < 1_000_000 else "very_large"


def _table(project: dict[str, Any]) -> dict[str, Any]:
    data = project.get("data") or {}
    profile = data.get("profile") or {}
    cols = profile.get("columns") or []
    kinds: dict[str, int] = defaultdict(int)
    for c in cols if isinstance(cols, list) else []:
        if isinstance(c, dict):
            kinds[str(c.get("kind") or c.get("dtype") or "unknown")] += 1
    return {"rows": _band(data.get("rows") or profile.get("rows")), "columns": len(data.get("columns") or []) or profile.get("column_count"),
            "kinds": dict(kinds), "duplicate_row_rate": profile.get("duplicate_row_rate"), "sha256": data.get("sha256"),
            "industry": project.get("industry")}


def records(layout: Layout) -> list[ExperimentRecord]:
    from .. import shapers

    projects: dict[str, StoreRecord] = {}
    stages: dict[tuple[str, str], StoreRecord] = {}
    for r in shapers.records(layout, "workflows"):
        source = r.provenance.source
        parts = source.split("/")
        if len(parts) >= 3 and parts[0] == "projects":
            pid = parts[1]
            if parts[2] == "project.json" and (pid not in projects or r.provenance.entry > projects[pid].provenance.entry):
                projects[pid] = r
            elif parts[2] == "stages" and source.endswith((".json",)):
                stage = parts[-1].rsplit(".", 1)[0]
                key = (pid, stage)
                if key not in stages or r.provenance.entry > stages[key].provenance.entry:
                    stages[key] = r
    out: list[ExperimentRecord] = []
    for (pid, stage), r in sorted(stages.items()):
        if stage != "models":
            continue
        b = r.body
        ev = b.get("evidence") or {}
        final = ((stages.get((pid, "final")) or StoreRecord.model_construct(body={})).body.get("evidence") or {})
        chosen = ev.get("selected_model")
        table = _table(projects[pid].body) if pid in projects else {"sha256": None}
        for n, m in enumerate(ev.get("model_results") or []):
            if not isinstance(m, dict):
                continue
            selected = bool(chosen) and m.get("config_id") == chosen
            out.append(ExperimentRecord(
                id=f"m3_ml_decisions:experiment:{pid}:{n}", model_type="m3_ml_decisions", split=split_of(f"project:{pid}"), provenance=r.provenance,
                task=str(b.get("task") or "unknown")[:40], table=table, model=str(m.get("model") or m.get("config_id") or "unknown")[:120],
                params=m.get("model_params") if isinstance(m.get("model_params"), dict) else {}, recipe=(m.get("recipe") or None) and str(m.get("recipe"))[:120],
                cv=(m.get("cv") or None) and str(m.get("cv"))[:300], folds=m.get("folds") if isinstance(m.get("folds"), int) else None,
                metrics=m.get("metrics") if isinstance(m.get("metrics"), dict) else {}, primary_metric=b.get("primary_metric"), selected=selected,
                holdout=(final.get("holdout_metrics") or {}) if selected else {}, seconds=b.get("elapsed_seconds")))
    return out


def build(layout: Optional[Layout] = None) -> dict[str, Any]:
    layout = layout or Layout()
    groups: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for e in records(layout):
        groups[e.task].append(e)
    report: dict[str, Any] = {"sets": {}}
    for task, rows in sorted(groups.items()):
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in rows}
        m = write_set(layout, "m3_ml_decisions", f"experiments/{task}", rows, sources=sources.values(),
                      about="Every model the engine fitted on one task, with its table's shape in bands, every fold's scores, and the holdout of the one chosen.")
        report["sets"][f"m3_ml_decisions/experiments/{task}"] = m.counts
    return report


__all__ = ["build", "records"]
