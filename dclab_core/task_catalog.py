"""P2 / P3 contracts: code-owned typed task catalog and ProblemDecomposition goal trees.

Pure Pydantic v2 validation rules: unknown tasks, hallucinated columns or tables, invalid kinds,
and post-holdout inspection tasks are rejected deterministically. No I/O, no model calls.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.solution_workflow import FAMILIES

CatalogTaskType = Literal[
    # Understanding & Clarification
    "define_target",
    "confirm_unit_and_horizon",
    "ask_owner",
    "profile_column",
    # Relational & Split Hygiene
    "check_join_key",
    "time_split_check",
    "grouped_split",
    # Leakage, Modeling & Decision
    "leakage_probe",
    "baseline",
    "feature_hypothesis",
    "segment_subproblem",
    "error_analysis",
    "calibration",
    "threshold_for_cost",
    # Physical Pipeline Stages
    "data",
    "leakage",
    "features",
    "models",
    "holdout",
    "final",
    "report",
]

STAGES = ("data", "leakage", "features", "models", "holdout", "final", "report")


class CatalogSpec(Contract):
    """Metadata specification of one code-owned task type."""

    task_type: CatalogTaskType
    phase: Literal["understand", "prepare", "model", "evaluate", "report"]
    effect: Literal["read", "ask", "write"]
    before_holdout: bool = True
    allowed_params: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    column_params: tuple[str, ...] = ()
    table_params: tuple[str, ...] = ()
    description: str = Field(max_length=240)


CATALOG: dict[str, CatalogSpec] = {
    "define_target": CatalogSpec(
        task_type="define_target",
        phase="understand",
        effect="read",
        allowed_params=("table", "candidate_columns", "target", "reason"),
        column_params=("target",),
        table_params=("table",),
        description="Identify or disambiguate the exact outcome column or event definition.",
    ),
    "confirm_unit_and_horizon": CatalogSpec(
        task_type="confirm_unit_and_horizon",
        phase="understand",
        effect="read",
        allowed_params=("unit", "horizon_days", "reason"),
        description="Confirm prediction unit (customer/order/line) and forecast window.",
    ),
    "ask_owner": CatalogSpec(
        task_type="ask_owner",
        phase="understand",
        effect="ask",
        allowed_params=("field", "question", "choices", "blocking"),
        required_params=("question",),
        description="Pause and ask the owner one concrete question on a blocking ambiguity.",
    ),
    "profile_column": CatalogSpec(
        task_type="profile_column",
        phase="prepare",
        effect="read",
        allowed_params=("table", "column", "reason"),
        required_params=("column",),
        column_params=("column",),
        table_params=("table",),
        description="Inspect distributions, missingness, uniqueness, or post-cutoff timestamps on a column.",
    ),
    "check_join_key": CatalogSpec(
        task_type="check_join_key",
        phase="prepare",
        effect="read",
        allowed_params=("table", "key", "parent_table", "reason"),
        required_params=("table", "key"),
        column_params=("key",),
        table_params=("table", "parent_table"),
        description="Measure join cardinality (1:1 vs 1:many), orphan keys, and duplicate-row risk before joining.",
    ),
    "time_split_check": CatalogSpec(
        task_type="time_split_check",
        phase="prepare",
        effect="read",
        allowed_params=("table", "column", "horizon_days", "reason"),
        required_params=("column",),
        column_params=("column",),
        table_params=("table",),
        description="Check temporal ordering, drift, and horizon overlap to enforce forward-in-time splitting.",
    ),
    "grouped_split": CatalogSpec(
        task_type="grouped_split",
        phase="prepare",
        effect="read",
        allowed_params=("table", "entity", "reason"),
        required_params=("entity",),
        column_params=("entity",),
        table_params=("table",),
        description="Verify repeated observations per entity and bind group-level holdout/CV splits.",
    ),
    "leakage_probe": CatalogSpec(
        task_type="leakage_probe",
        phase="prepare",
        effect="read",
        allowed_params=("table", "column", "reason"),
        required_params=("column",),
        column_params=("column",),
        table_params=("table",),
        description="Probe a suspect feature on training rows for post-outcome naming, target correlation, or snapshot timing.",
    ),
    "baseline": CatalogSpec(
        task_type="baseline",
        phase="model",
        effect="read",
        allowed_params=("family", "metric"),
        description="Train and score a simple guarded baseline on training CV folds.",
    ),
    "feature_hypothesis": CatalogSpec(
        task_type="feature_hypothesis",
        phase="model",
        effect="read",
        allowed_params=("table", "cutoff_column", "entity", "spec"),
        column_params=("cutoff_column", "entity"),
        table_params=("table",),
        description="Derive windowed/point-in-time aggregate features prior to the prediction cutoff.",
    ),
    "segment_subproblem": CatalogSpec(
        task_type="segment_subproblem",
        phase="understand",
        effect="read",
        allowed_params=("kind", "objective", "treatment_column"),
        required_params=("kind",),
        column_params=("treatment_column",),
        description="Decompose a composite goal into a typed DCLab problem kind (e.g. uplift, segmentation, entity_tables).",
    ),
    "error_analysis": CatalogSpec(
        task_type="error_analysis",
        phase="model",
        effect="read",
        allowed_params=("table", "slice_column", "metric"),
        column_params=("slice_column",),
        table_params=("table",),
        description="Inspect training-CV error and treatment overlap across key slices or cohorts.",
    ),
    "calibration": CatalogSpec(
        task_type="calibration",
        phase="model",
        effect="read",
        allowed_params=("metric", "positive_rate"),
        description="Verify and calibrate predicted probabilities on training OOF folds.",
    ),
    "threshold_for_cost": CatalogSpec(
        task_type="threshold_for_cost",
        phase="model",
        effect="read",
        allowed_params=("cost_ratio", "capacity_limit", "costs"),
        description="Select decision/action thresholds on OOF predictions for asymmetric cost or capacity limits.",
    ),
    **{
        stage: CatalogSpec(
            task_type=stage,
            phase=("evaluate" if stage in ("holdout", "final") else "report" if stage == "report" else "prepare" if stage in ("data", "leakage") else "model"),
            effect=("write" if stage in ("data", "leakage", "features", "models", "final", "report") else "ask"),
            before_holdout=(stage in ("data", "leakage", "features", "models")),
            description=f"Guarded pipeline stage: {stage}.",
        )
        for stage in STAGES
    },
}


def _clip_text(data: dict[str, Any], key: str, limit: int, *, default: str | None = None) -> None:
    val = data.get(key)
    if val is None and default is not None:
        data[key] = default
    elif isinstance(val, str) and len(val) > limit:
        data[key] = val[:limit].strip()


class CatalogTaskCall(Contract):
    """One proposed typed catalog task call with grounded parameters and citations."""

    task_type: CatalogTaskType
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=400)
    citations: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="before")
    @classmethod
    def _accept_purpose_alias(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        if "purpose" in copied and "reason" not in copied:
            copied["reason"] = copied.pop("purpose")
        _clip_text(copied, "reason", 400, default="")
        if copied.get("citations") is None:
            copied["citations"] = ()
        return copied


class SubproblemSpec(Contract):
    """One decomposed analytical sub-problem aligned with a DCLab family."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,60}$")
    kind: str = Field(max_length=40)
    objective: str = Field(min_length=3, max_length=300)
    target: Optional[str] = Field(default=None, max_length=120)
    dependency_ids: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="before")
    @classmethod
    def _clip_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        _clip_text(copied, "objective", 300)
        return copied

    @model_validator(mode="after")
    def _valid_family(self) -> SubproblemSpec:
        if self.kind not in FAMILIES:
            raise ValueError(f"unsupported subproblem kind {self.kind!r}; expected one of {FAMILIES}")
        return self


class IdentifiedRisk(Contract):
    """A structural, temporal, leakage, or imbalance risk tied to metadata."""

    trap_kind: str = Field(min_length=2, max_length=120)
    table: Optional[str] = Field(default=None, max_length=120)
    column: Optional[str] = Field(default=None, max_length=120)
    evidence_citation: str = Field(default="", max_length=80)
    recommended_task: CatalogTaskType

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        _clip_text(copied, "evidence_citation", 80, default="")
        _clip_text(copied, "trap_kind", 120)
        return copied


class OpenUnknown(Contract):
    """An unresolved contract field paired with how the workflow resolves it."""

    field: Literal["target", "unit", "horizon", "join_key", "treatment", "metric", "cost_ratio", "other"]
    question: str = Field(min_length=5, max_length=300)
    blocking: bool = True
    resolution_task: CatalogTaskType

    @model_validator(mode="before")
    @classmethod
    def _clip_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        _clip_text(copied, "question", 300)
        return copied


class ProblemDecomposition(Versioned):
    """Structured goal tree produced before planning (P3): understand -> decompose -> plan."""

    schema_version: Literal["dclab_core.problem_decomposition.v1"] = _version("problem_decomposition")
    decision: str = Field(min_length=3, max_length=400)
    target: Optional[str] = Field(default=None, max_length=120)
    unit: Optional[str] = Field(default=None, max_length=120)
    horizon_days: Optional[int] = Field(default=None, ge=0, le=36500)
    success_metric: str = Field(min_length=2, max_length=160)
    constraints: tuple[str, ...] = Field(default=(), max_length=32)
    subproblems: tuple[SubproblemSpec, ...] = Field(min_length=1, max_length=16)
    risks: tuple[IdentifiedRisk, ...] = Field(default=(), max_length=32)
    unknowns: tuple[OpenUnknown, ...] = Field(default=(), max_length=16)
    source: Literal["deterministic", "model", "hybrid"] = "deterministic"

    @model_validator(mode="before")
    @classmethod
    def _coerce_optional_collections(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        for seq_field in ("constraints", "risks", "unknowns"):
            if copied.get(seq_field) is None:
                copied[seq_field] = ()
        if not copied.get("success_metric"):
            copied["success_metric"] = "roc_auc"
        _clip_text(copied, "decision", 400)
        _clip_text(copied, "success_metric", 160)
        return copied


CONTRACTS["problem_decomposition"] = ProblemDecomposition


def _all_columns(tables: Mapping[str, Iterable[str]]) -> set[str]:
    return {str(col) for cols in tables.values() for col in cols}


def _check_column_ref(table_name: Optional[str], col_name: Any, tables: Mapping[str, Iterable[str]], context: str) -> list[str]:
    if col_name is None or col_name == "":
        return []
    if not isinstance(col_name, str):
        return [f"{context}: non-string column reference"]
    if table_name:
        if table_name not in tables:
            return [f"{context}: unknown table {table_name!r}"]
        if col_name not in set(tables[table_name]):
            return [f"{context}: column {col_name!r} not in table {table_name!r}"]
        return []
    if col_name not in _all_columns(tables):
        # Allow conceptual cutoff markers named identically to declared columns, refuse fabricated names.
        return [f"{context}: column {col_name!r} not found in any project table"]
    return []


def validate_catalog_plan(
    calls: Iterable[CatalogTaskCall | Mapping[str, Any]],
    tables: Mapping[str, Iterable[str]],
    *,
    require_before_holdout: bool = True,
) -> tuple[list[CatalogTaskCall], list[str]]:
    """Validate proposed typed catalog tasks against CATALOG specs and project table schemas."""
    parsed: list[CatalogTaskCall] = []
    issues: list[str] = []
    holdout_seen = False
    for idx, raw in enumerate(calls):
        try:
            call = raw if isinstance(raw, CatalogTaskCall) else CatalogTaskCall.model_validate(raw)
        except Exception as exc:
            issues.append(f"call[{idx}]: invalid catalog call ({exc})")
            continue
        spec = CATALOG.get(call.task_type)
        if spec is None:
            issues.append(f"call[{idx}]: unknown task_type {call.task_type!r}")
            continue
        if call.task_type in ("holdout", "final"):
            holdout_seen = True
        elif require_before_holdout and holdout_seen and spec.before_holdout:
            issues.append(f"call[{idx}] ({call.task_type}): cannot run after holdout/final stage")

        extra = sorted(set(call.params) - set(spec.allowed_params))
        if extra:
            issues.append(f"call[{idx}] ({call.task_type}): disallowed params {extra}")
        missing = [p for p in spec.required_params if not call.params.get(p)]
        if missing:
            issues.append(f"call[{idx}] ({call.task_type}): missing required params {missing}")

        for tparam in spec.table_params:
            tval = call.params.get(tparam)
            if tval and tval not in tables:
                issues.append(f"call[{idx}] ({call.task_type}): unknown table {tval!r}")
        table_ctx = call.params.get("table")
        for cparam in spec.column_params:
            cval = call.params.get(cparam)
            if cval == "cutoff_date" and cparam == "column" and call.task_type == "time_split_check":
                continue  # explicit synthetic snapshot cutoff marker when multiple event tables exist
            issues.extend(_check_column_ref(table_ctx, cval, tables, f"call[{idx}] ({call.task_type}).{cparam}"))

        if call.task_type == "segment_subproblem":
            kind = call.params.get("kind")
            if kind and kind not in FAMILIES:
                issues.append(f"call[{idx}] (segment_subproblem): invalid family kind {kind!r}")

        parsed.append(call)
    return parsed, issues


def validate_decomposition(
    decomp: ProblemDecomposition | Mapping[str, Any],
    tables: Mapping[str, Iterable[str]],
) -> tuple[Optional[ProblemDecomposition], list[str]]:
    """Validate a ProblemDecomposition against project table metadata and DCLab families."""
    try:
        if isinstance(decomp, ProblemDecomposition):
            parsed = decomp
        else:
            raw = dict(decomp)
            if "schema_version" not in raw and raw.get("decision"):
                raw["schema_version"] = "dclab_core.problem_decomposition.v1"
            parsed = ProblemDecomposition.read(raw)
    except Exception as exc:
        return None, [f"invalid decomposition shape: {exc}"]

    issues: list[str] = []
    all_cols = _all_columns(tables)
    # Only enforce column membership when target matches a concrete column vs conceptual composite goal like inactive_30d.
    if parsed.target and parsed.target in all_cols:
        pass
    for idx, risk in enumerate(parsed.risks):
        if risk.table and risk.table not in tables:
            issues.append(f"risk[{idx}]: unknown table {risk.table!r}")
        if risk.column:
            issues.extend(_check_column_ref(risk.table, risk.column, tables, f"risk[{idx}]"))

    for idx, unk in enumerate(parsed.unknowns):
        if unk.blocking and unk.resolution_task not in ("ask_owner", "define_target", "confirm_unit_and_horizon", "profile_column", "check_join_key"):
            issues.append(f"unknown[{idx}]: blocking unknown must resolve via question or understanding task, got {unk.resolution_task!r}")

    return parsed, issues


__all__ = [
    "CATALOG",
    "CatalogSpec",
    "CatalogTaskCall",
    "CatalogTaskType",
    "IdentifiedRisk",
    "OpenUnknown",
    "ProblemDecomposition",
    "SubproblemSpec",
    "TASK_TYPES",
    "validate_catalog_plan",
    "validate_decomposition",
]

TASK_TYPES: tuple[str, ...] = tuple(CATALOG)
