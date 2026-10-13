"""Decision samples (package A6.5): every typed decision the studio makes, with the evidence it saw and what the person
did with it, kept as training data for DCLab's own decision models.

A decision point is a question the rules answer on a column, a table or a problem: is this column an identifier, is
it a leak suspect, which split, which family, which recipe, what does the contract say the target is. The studio
answers each one deterministically today; a model of DCLab's own (or an open decision model such as Laya) will answer
beside the rule tomorrow. For that it needs, per decision, the evidence the rule saw, the rule's answer and the
person's. This package writes exactly that:

* a **sample** where the decision is computed: the proposal (``record_core``), the brief (``record_brief``), a stage
  (``record_stage``);
* a **label** where a person acts on it: a stage approved or overridden (``label_stage``), a solution saved
  (``label_solution``), a gate approved (``label_gate``).

Rows go to ``decision_samples.jsonl`` and ``decision_labels.jsonl`` in the workspace folder, or to the
``decision_sample`` and ``decision_label`` tables on PostgreSQL. ``python -m dclab_rnd.decisions export`` joins them into
the training-lake sample record, split by dataset. Nothing here reads the holdout or stores a cell value; a write that
fails is reported and the decision it recorded stands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import extract, labels, schema, store
from .export import export, join, split_of
from .schema import POINTS, SAMPLE_FIELDS, LABEL_FIELDS
from .store import FileSamples, PgSamples, SampleStore, for_store, open_samples

__all__ = ["POINTS", "SAMPLE_FIELDS", "LABEL_FIELDS", "FileSamples", "PgSamples", "SampleStore", "for_store", "open_samples", "export",
           "join", "split_of", "record_stage", "record_core", "record_brief", "label_stage", "label_solution", "label_gate", "forget", "stats"]


def _target(project_store: Any) -> SampleStore | None:
    try:
        return for_store(project_store)
    except Exception:  # noqa: BLE001 — a store that cannot be opened records nothing
        return None


def record_stage(project_store: Any, project: dict[str, Any], record: dict[str, Any]) -> int:
    """A completed stage: its column judgments and its choice, with the options it ranked."""
    try:
        rows = extract.from_stage(record, project)
    except Exception as exc:  # noqa: BLE001 — a record this reader does not expect is skipped, never a failed stage
        import sys

        print(f"decisions: could not read stage {record.get('stage')}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 0
    return store.record(_target(project_store), rows)


def record_core(project_store: Any, project: dict[str, Any], findings: Any, plan: Any, columns: list[str], target: str | None,
                profile: dict[str, Any] | None = None, run_ref: str | None = None) -> int:
    """The proposal: dclab_core's findings and plan on the provisional training rows (per-column forbidden, identifier
    and missing-value judgments; the split, families, cleaning, metric and the other plan decisions)."""
    data = project.get("data") or {}
    try:
        rows = extract.from_core(findings, plan, columns=columns, profile=profile or data.get("profile"), target=target,
                                 project_id=project.get("id"), dataset=schema.dataset_ref(data.get("sha256"), data.get("filename")), run_ref=run_ref)
    except Exception as exc:  # noqa: BLE001
        import sys

        print(f"decisions: could not read the plan: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 0
    return store.record(_target(project_store), rows)


def record_brief(any_store: Any, brief: Any, *, project_id: str | None = None, draft_id: str | None = None,
                 columns: list[str] | None = None) -> int:
    """The problem brief: one sample per contract field (task, target, prediction moment, time column, coverage)."""
    try:
        rows = extract.from_brief(brief, project_id=project_id, draft_id=draft_id, columns=columns)
    except Exception as exc:  # noqa: BLE001
        import sys

        print(f"decisions: could not read the brief: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 0
    return store.record(_target(any_store), rows)


def label_stage(project_store: Any, project: dict[str, Any], stage: str, record: dict[str, Any] | None, choice: str | None,
                actor: str = "human") -> int:
    target = _target(project_store)
    return store.label(target, labels.stage_approved(target, project, stage, record, choice, actor))


def label_solution(project_store: Any, project: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any], who: str = "human") -> int:
    target = _target(project_store)
    return store.label(target, labels.solution_saved(target, project, before, after, who))


def label_gate(project_store: Any, project: dict[str, Any], gate: str, by: str, reason: str = "") -> int:
    target = _target(project_store)
    return store.label(target, labels.gate_approved(target, project, gate, by, reason))


def forget(project_store: Any, project_id: str, draft_id: str | None = None) -> None:
    """A deleted project takes its samples and labels with it, and the brief's samples of the draft it was built from
    (export first to keep them)."""
    target = _target(project_store)
    if target is None:
        return
    try:
        target.delete(project_id, draft_id)
    except Exception as exc:  # noqa: BLE001
        import sys

        print(f"decisions: could not forget {project_id}: {type(exc).__name__}", file=sys.stderr, flush=True)


def stats(home: Path) -> dict[str, Any]:
    """Counts per point and how many samples carry a human label (the Admin page and the CLI)."""
    from collections import Counter

    target = open_samples(Path(home))
    rows = join(target.samples(), target.labels())
    return {"samples": len(rows), "labelled": sum(1 for r in rows if r["label"] is not None),
            "by_point": dict(Counter(r["point"] for r in rows)), "projects": len({r.get("project_id") for r in rows if r.get("project_id")})}
