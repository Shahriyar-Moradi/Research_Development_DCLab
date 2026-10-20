"""The experiments themselves (32.20): every model fitted by the engine, one :class:`dclab_core.store.ExperimentRecord` each,
keyed to its table by the table's hash and described by its shape in bands (never a row): across hundreds of tables, which
approach wins on which kind of data, for M3, M5 and the service.

From the shaped records: each project's ``models`` stage record (its every candidate with every fold's scores) and its ``final``
record (the holdout's metrics, for the model chosen), and the project's own record for the table's shape; and the research
campaigns' model-selection experiments on public datasets (``campaign_records``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from dclab_core.datasets import split_of
from dclab_core.store import ExperimentRecord

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


def _experiment(ident: str, split: Any, provenance: Any, task: Any, table: dict[str, Any], m: dict[str, Any], primary: Any, selected: bool,
                holdout: dict[str, Any], seconds: Any) -> ExperimentRecord:
    return ExperimentRecord(
        id=ident, model_type="m3_ml_decisions", split=split, provenance=provenance, task=str(task or "unknown")[:40], table=table,
        model=str(m.get("model") or m.get("config_id") or "unknown")[:120], params=m.get("model_params") if isinstance(m.get("model_params"), dict) else {},
        recipe=(m.get("recipe") or None) and str(m.get("recipe"))[:120], cv=(m.get("cv") or None) and str(m.get("cv"))[:300],
        folds=m.get("folds") if isinstance(m.get("folds"), int) else None, metrics=m.get("metrics") if isinstance(m.get("metrics"), dict) else {},
        primary_metric=primary, selected=selected, holdout=holdout if selected else {}, seconds=seconds)


def records(layout: Layout) -> list[ExperimentRecord]:
    """Every candidate of every project's models stage (file or PostgreSQL workspaces, keyed by workspace and project), from
    workspaces whose consent allows training; its split is the project's one group's."""
    from ..service import _latest

    out: list[ExperimentRecord] = []
    for (ws, pid), p in sorted(_latest(layout).items()):
        r = (p.get("records") or {}).get("models")
        if r is None or p.get("consent") != "train":
            continue
        b, final = p["stages"]["models"], (p["stages"].get("final") or {}).get("evidence") or {}
        ev = b.get("evidence") or {}
        chosen = ev.get("selected_model")
        table = _table(p["project"])
        split = r.split or split_of(f"project:{pid}")
        for n, m in enumerate(ev.get("model_results") or []):
            if isinstance(m, dict):
                out.append(_experiment(f"m3_ml_decisions:experiment:{ws}:{pid}:{n}", split, r.provenance, b.get("task"), table, m, b.get("primary_metric"),
                                       bool(chosen) and m.get("config_id") == chosen, final.get("holdout_metrics") or {}, b.get("elapsed_seconds")))
    return out


def campaign_records(root: Any = None) -> list[ExperimentRecord]:
    """Every candidate of the research campaigns' model-selection experiments (``evidence/campaigns``), on public datasets: the
    table's shape from its dataset card, the holdout of the model chosen from the same dataset's optimization result, split by
    dataset so one dataset's experiments never cross train and sealed."""
    import hashlib
    import json

    from dclab_core.store import Provenance

    from ...evidence_index import ROOT, dataset_cards, experiment_files

    root = root or ROOT
    cards = dataset_cards(root)
    results = []
    for path in experiment_files(root):
        try:
            results.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    holdouts = {(d.get("campaign_id"), d.get("dataset")): (d.get("evidence") or {}) for _, d in results if d.get("kind") == "optimization_reliability"}
    out = []
    for path, d in results:
        ev = d.get("evidence") or {}
        if d.get("kind") != "model_selection" or not isinstance(ev.get("model_results"), list):
            continue
        dataset, card = str(d.get("dataset")), cards.get(str(d.get("dataset"))) or {}
        final = holdouts.get((d.get("campaign_id"), d.get("dataset"))) or {}
        chosen = ev.get("selected_model")
        prov = Provenance(entry=f"campaign:{d.get('experiment_id')}", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          source=path.relative_to(root).as_posix(), workspace="rnd", consent="train", at=str(d.get("completed_at") or d.get("started_at") or ""),
                          rule="campaign:model_selection")
        table = {"rows": _band(card.get("rows")), "columns": card.get("features"), "dataset": dataset, "sha256": None, "industry": None}
        for n, m in enumerate(ev["model_results"]):
            if isinstance(m, dict):
                out.append(_experiment(f"m3_ml_decisions:experiment:campaign:{d.get('experiment_id')}:{n}", split_of(f"dataset:{dataset}"), prov,
                                       d.get("task_type") or card.get("task_type"), table, m, d.get("primary_metric"),
                                       bool(chosen) and m.get("config_id") == chosen and final.get("model", chosen) == chosen,
                                       final.get("holdout_metrics") or {}, m.get("elapsed_seconds")))
    return out


def build(layout: Optional[Layout] = None) -> dict[str, Any]:
    layout = layout or Layout()
    groups: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for e in records(layout) + campaign_records():
        groups[e.task].append(e)
    report: dict[str, Any] = {"sets": {}}
    for task, rows in sorted(groups.items()):
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in rows}
        m = write_set(layout, "m3_ml_decisions", f"experiments/{task}", rows, sources=sources.values(),
                      about="Every model the engine fitted on one task, with its table's shape in bands, every fold's scores, and the holdout of the one chosen.")
        report["sets"][f"m3_ml_decisions/experiments/{task}"] = m.counts
    return report


__all__ = ["build", "records", "campaign_records"]
