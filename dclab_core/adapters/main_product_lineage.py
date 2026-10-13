"""DCLab's main product's runs as :class:`dclab_core.lineage.RunRecord` (package 19.5).

The main product already records its runs: ``experiments`` (a pipeline run: its dataset, split plan, config, seed, git
commit and result), ``pipeline_stage_runs`` (each stage of one, with input and output summaries and its duration) and
``agent_runs`` (an agent's run: the digests of its context, policy and tool catalog, its prompt release and runtime).
These functions map their rows, field by field from the lists read at one commit (``main_product_fields.json``,
``experience_tables``), onto run records, so both hosts trace a number to its run with the same code
(:func:`dclab_core.lineage.run_for_path`, :func:`~dclab_core.lineage.compare`). No table is added there.

A row whose status says it is still live (queued, running, waiting for a person, created) is not a run yet: ``None``.
What the main product records elsewhere is not guessed: a field it has no column for is left empty (``INPUTS_MISSING``).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Optional

from dclab_core.adapters.main_product import recorded
from dclab_core.lineage import RunCode, RunInputs, RunOutputs, RunRecord, flat, numbers

# agent_runs.status (apps/api/app/domain/agent_records.py AGENT_RUN_STATUSES): the terminal ones; the live ones are absent
AGENT_STATUS = {"completed": "completed", "closed": "completed", "failed": "failed", "rejected_by_validator": "failed",
                "over_budget": "stopped", "timed_out": "stopped", "cancelled": "stopped"}
# experiments.status and pipeline_stage_runs.status are free text there (default "CREATED"): read without case
RUN_STATUS = {"completed": "completed", "succeeded": "completed", "success": "completed", "done": "completed",
              "failed": "failed", "fail": "failed", "error": "failed",
              "cancelled": "stopped", "canceled": "stopped", "stopped": "stopped", "skipped": "stopped", "skip": "stopped"}
INPUTS_MISSING = {
    "experiments": ["data_sha256 (the dataset version's hash is in datasets, not read)", "data_rows", "data_columns",
                    "solution_hash (the problem spec's lock lives in problem_specs)", "sampling"],
    "pipeline_stage_runs": ["every input but the stage's own summary: they are its experiment's"],
    "agent_runs": ["data_sha256", "solution_hash", "seed: an agent run reads digests, not tables"],
}


def check_columns(table: str, row: Mapping[str, Any]) -> None:
    """A row must have only the columns recorded for its table at the read commit: a renamed column fails loudly."""
    unknown = set(row) - set(recorded()["experience_tables"][table])
    if unknown:
        raise ValueError(f"{table}: columns not recorded at the read commit: {sorted(unknown)}; read the main product again")


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return str(value)


def _seconds(started: Any, finished: Any) -> Optional[float]:
    try:
        a, b = datetime.fromisoformat(_iso(started)), datetime.fromisoformat(_iso(finished))
    except (TypeError, ValueError):
        return None
    return max((b - a).total_seconds(), 0.0)


def _str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def from_agent_run(row: Mapping[str, Any]) -> Optional[RunRecord]:
    """One ``agent_runs`` row: what the agent was shown (by digest), what it was for, the code (runtime, prompt release)."""
    check_columns("agent_runs", row)
    status = AGENT_STATUS.get(str(row.get("status") or ""))
    if status is None:
        return None
    started = row.get("started_at") or row.get("created_at")
    refs = {k: str(row[c]) for k, c in (("context", "context_digest"), ("policy", "policy_digest"), ("tool_catalog", "tool_catalog_digest"),
                                         ("dataset", "dataset_id"), ("problem_spec", "problem_spec_id"), ("split_plan", "split_plan_id"),
                                         ("experiment", "experiment_id"), ("model_version", "model_version_id")) if row.get(c)}
    params = {k: row.get(k) for k in ("kind", "purpose", "decision_point_key", "subject_kind", "data_class", "outcome_scope") if row.get(k) is not None}
    runtime = "@".join(str(row[k]) for k in ("runtime", "runtime_version") if row.get(k))
    agent = "@".join(str(row[k]) for k in ("agent_key", "agent_version") if row.get(k))
    return RunRecord(run=str(row["id"]), kind="agent", name=str(row.get("agent_key") or row.get("kind") or "agent"),
                     workspace=_str(row.get("workspace_id")), project=_str(row.get("project_id")), parent=_str(row.get("parent_run_id")),
                     status=status, started=_iso(started), finished=_iso(row.get("finished_at")), seconds=_seconds(started, row.get("finished_at")),
                     inputs=RunInputs(refs=refs), params=params,
                     code=RunCode(runner=" ".join(x for x in (agent, runtime) if x) or None,
                                  prompts={"prompt_release": str(row["prompt_release_id"])} if row.get("prompt_release_id") else {}),
                     error=_str(row.get("error_code")))


def from_experiment(row: Mapping[str, Any]) -> Optional[RunRecord]:
    """One ``experiments`` row (a pipeline run): its dataset and split plan, config, seed, commit and result's numbers."""
    check_columns("experiments", row)
    status = RUN_STATUS.get(str(row.get("status") or "").lower())
    if status is None:
        return None
    started = row.get("started_at") or row.get("created_at")
    refs = {k: str(row[c]) for k, c in (("dataset", "dataset_id"), ("source_dataset", "source_dataset_id"), ("split_plan", "split_plan_id"),
                                         ("pipeline_version", "pipeline_version_id"), ("task", "task_id"), ("workflow_run", "workflow_run_id")) if row.get(c)}
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    return RunRecord(run=str(row["id"]), kind="pipeline", name=str(row.get("pipeline_name") or row.get("pipeline_purpose") or "experiment"),
                     workspace=_str(row.get("workspace_id")), project=_str(row.get("project_id")), parent=_str(row.get("parent_pipeline_run_id")),
                     status=status, started=_iso(started), finished=_iso(row.get("ended_at")), seconds=_seconds(started, row.get("ended_at")),
                     inputs=RunInputs(refs=refs, seed=row.get("seed"), settings=flat(row.get("config") or {}),
                                      decisions=flat(row.get("change_set") or {})),
                     # ``intent`` is the person's own text there (String(2000)): never a run's parameter, never meta
                     params={k: row.get(k) for k in ("branch_key", "run_number") if row.get(k) is not None},
                     outputs=RunOutputs(numbers=numbers(result)), code=RunCode(commit=_str(row.get("git_commit"))),
                     error=_str(row.get("failure_reason"))[:600] if row.get("failure_reason") else None)


def from_stage_run(row: Mapping[str, Any]) -> Optional[RunRecord]:
    """One ``pipeline_stage_runs`` row: a stage of an experiment, its summaries and its duration."""
    check_columns("pipeline_stage_runs", row)
    status = RUN_STATUS.get(str(row.get("status") or "").lower())
    if status is None:
        return None
    started = row.get("started_at") or row.get("created_at")
    duration = row.get("duration_ms")
    summary = row.get("output_summary") if isinstance(row.get("output_summary"), Mapping) else {}
    return RunRecord(run=str(row["id"]), kind="stage", name=str(row.get("stage_key") or row.get("name") or "stage"),
                     workspace=_str(row.get("workspace_id")), project=_str(row.get("project_id")), parent=_str(row.get("pipeline_run_id")),
                     status=status, started=_iso(started), finished=_iso(row.get("completed_at")),
                     seconds=duration / 1000 if isinstance(duration, (int, float)) and duration >= 0 else _seconds(started, row.get("completed_at")),
                     # its input summary may name columns or quote values: only its numbers travel
                     params=numbers(row.get("input_summary")) if isinstance(row.get("input_summary"), Mapping) else {},
                     outputs=RunOutputs(numbers=numbers(summary)), error=_str(row.get("failure_code")))
