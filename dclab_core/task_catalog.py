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
    subproblem: Optional[str] = Field(default=None, max_length=61)  # the sub-problem this step serves: its place on the path

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
        before = len(issues)
        if call.task_type in ("holdout", "final"):
            holdout_seen = True
        elif require_before_holdout and holdout_seen and spec.before_holdout:
            issues.append(f"call[{idx}] ({call.task_type}): cannot run after holdout/final stage")

        extra = sorted(set(call.params) - set(spec.allowed_params))
        if extra:
            # a key the catalog does not know is dropped and named; the step itself is kept (before, a question for the
            # person with one stray key was refused whole, and the question was lost)
            issues.append(f"call[{idx}] ({call.task_type}): disallowed params {extra}, dropped")
            call = call.model_copy(update={"params": {k: v for k, v in call.params.items() if k not in extra}})
            before += 1
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

        if len(issues) == before:
            # a call with an issue (a column or a table the project does not have, a key the catalog does not know) is
            # refused alone, with its reason; it is never run as if it were valid, and the rest of the plan is kept
            parsed.append(call)
    return parsed, issues


def validate_decomposition(
    decomp: ProblemDecomposition | Mapping[str, Any],
    tables: Mapping[str, Iterable[str]],
    *,
    keep_valid: bool = False,
) -> tuple[Optional[ProblemDecomposition], list[str]]:
    """Validate a ProblemDecomposition against project table metadata and DCLab families.

    With ``keep_valid`` the breakdown comes back without the parts that failed (a risk on a column the tables do not
    have, a blocking unknown no question can settle), each named in the issues: one wrong part never throws the whole
    breakdown away."""
    try:
        if isinstance(decomp, ProblemDecomposition):
            parsed = decomp
        else:
            raw = dict(decomp)
            if "schema_version" not in raw and raw.get("decision"):
                raw["schema_version"] = "dclab_core.problem_decomposition.v1"
            parsed = ProblemDecomposition.read(raw)
    except Exception as exc:
        if not keep_valid or not isinstance(decomp, Mapping):
            return None, [f"invalid decomposition shape: {exc}"]
        # one malformed risk, unknown or sub-problem is dropped with its reason; the rest of the breakdown is read again
        raw, dropped = dict(raw), []
        for name, model in (("risks", IdentifiedRisk), ("unknowns", OpenUnknown), ("subproblems", SubproblemSpec)):
            kept = []
            items = raw.get(name)
            if not isinstance(items, (list, tuple)):  # a number or a word where a list belongs: the part is dropped whole
                if items not in (None, "", ()):
                    dropped.append(f"{name}: not a list, dropped")
                items = ()
            seen_ids: set[str] = set()
            for idx, item in enumerate(items):
                if name == "subproblems" and isinstance(item, Mapping):
                    if item.get("id") in seen_ids:  # two sub-problems under one id: the first is kept
                        dropped.append(f"subproblems[{idx}]: id {item.get('id')!r} is used twice, dropped")
                        continue
                    seen_ids.add(item.get("id"))
                try:
                    model.model_validate(item)
                    kept.append(item)
                except Exception:
                    dropped.append(f"{name}[{idx}]: malformed, dropped")
            raw[name] = kept
        try:
            parsed = ProblemDecomposition.read(raw)
        except Exception as again:
            return None, [f"invalid decomposition shape: {again}"]
        _, rest = validate_decomposition(parsed, tables, keep_valid=True)
        parsed, _ = validate_decomposition(parsed, tables, keep_valid=True)
        return parsed, [*dropped, *rest]

    issues: list[str] = []
    risks, unknowns = [], []
    # A target may be a composite goal (inactive_30d) and need not be a column; risks and unknowns are checked.
    for idx, risk in enumerate(parsed.risks):
        found = [f"risk[{idx}]: unknown table {risk.table!r}"] if risk.table and risk.table not in tables else []
        if risk.column:
            found.extend(_check_column_ref(risk.table, risk.column, tables, f"risk[{idx}]"))
        issues.extend(found)
        if not found:
            risks.append(risk)

    for idx, unk in enumerate(parsed.unknowns):
        if unk.blocking and unk.resolution_task not in ("ask_owner", "define_target", "confirm_unit_and_horizon", "profile_column", "check_join_key"):
            issues.append(f"unknown[{idx}]: blocking unknown must resolve via question or understanding task, got {unk.resolution_task!r}")
        else:
            unknowns.append(unk)

    known = {s.id for s in parsed.subproblems}
    if len(known) != len(parsed.subproblems):
        issues.append("two sub-problems share one id: the first is kept")
        if keep_valid:
            parsed = parsed.model_copy(update={"subproblems": tuple({s.id: s for s in reversed(parsed.subproblems)}.values())[::-1]})
    for sub in parsed.subproblems:
        for dep in sub.dependency_ids:
            if dep not in known or dep == sub.id:
                issues.append(f"subproblem {sub.id!r}: depends on {dep!r}, which is not another sub-problem")
    if keep_valid and issues:
        subs = tuple(s.model_copy(update={"dependency_ids": tuple(d for d in s.dependency_ids if d in known and d != s.id)}) for s in parsed.subproblems)
        parsed = parsed.model_copy(update={"risks": tuple(risks), "unknowns": tuple(unknowns), "subproblems": subs})
    return parsed, issues


PERSON_STEPS = ("ask_owner", "define_target", "confirm_unit_and_horizon")  # steps only the person can finish
# each stage of the build with the reason it stands where it does: the reason a person reads before the stage runs
BUILD_STEPS = (("data", "Prepare the data and fix the split", "first, so nothing chosen later can have seen the rows kept for the final check"),
               ("features", "Choose the features, inside the training folds", "after the split and the leak checks, so a column the plan keeps out cannot come back in"),
               ("models", "Compare models on validation", "after the features are fixed, so models are compared on the same honest ground"),
               ("holdout", "Your approval to read the holdout", "the holdout can be read once: a person decides the work is ready for it"),
               ("final", "Read the holdout once", "last, after your approval and only when every catch of the plan is enforced or decided by you"),
               ("report", "Write the report", "after the final number exists, so every figure in it is traced to its record"))


def _order_reason(sub: SubproblemSpec, by_id: Mapping[str, SubproblemSpec]) -> str:
    """Why a sub-problem stands where it does, written by code from its dependencies (never asked of a model)."""
    first = [by_id[d].objective for d in sub.dependency_ids if d in by_id and d != sub.id]
    return ("Comes after: " + "; ".join(first) + ". That must be settled first.") if first else "Nothing has to be settled before it."


def solution_path(decomposition: ProblemDecomposition | Mapping[str, Any] | None,
                  calls: Iterable[CatalogTaskCall | Mapping[str, Any]],
                  evidence: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The path from a problem to its solution, as one numbered list the engine and the person both follow.

    The problem is the breakdown's decision; each sub-problem is a leg, in the order its dependencies allow; a leg's
    steps are the checks the plan chose for it, in the plan's order, each with what it found; the build's stages close
    the path. Every part carries its reason. A step's reason is the plan's own purpose for it; the reason of a
    sub-problem's place and of each build stage is written by code, from the dependencies and the build's fixed order.
    No model is asked to write its reasoning: the model's thinking is the API thinking summary of the planning request,
    kept by the gateway. A step with no sub-problem named belongs to the first leg. Names and reasons only, never a
    table's value."""
    if decomposition is not None and not isinstance(decomposition, ProblemDecomposition):
        try:
            decomposition = ProblemDecomposition.model_validate(dict(decomposition))
        except Exception:
            decomposition = None
    subs = list({s.id: s for s in reversed(decomposition.subproblems)}.values())[::-1] if decomposition else []  # one per id: the first
    order, placed = [], set()
    while len(order) < len(subs):  # dependencies first; a cycle or an unknown name falls back to the written order
        ready = [s for s in subs if s.id not in placed and all(d in placed or d not in {x.id for x in subs} for d in s.dependency_ids)]
        nxt = ready[0] if ready else next(s for s in subs if s.id not in placed)
        order.append(nxt)
        placed.add(nxt.id)
    legs = [{"id": s.id, "kind": s.kind, "objective": s.objective, "target": s.target, "after": [d for d in s.dependency_ids if d in placed],
             "why": _order_reason(s, {x.id: x for x in subs}), "steps": []} for s in order] or [{"id": "main", "kind": None, "objective": decomposition.decision if decomposition else "", "target": None,
                                              "after": [], "why": "", "steps": []}]
    by_id = {leg["id"]: leg for leg in legs}
    rows = list(evidence)
    for index, raw in enumerate(calls):
        call = raw if isinstance(raw, CatalogTaskCall) else CatalogTaskCall.model_validate(raw)
        row = rows[index] if index < len(rows) and isinstance(rows[index], Mapping) and rows[index].get("task_type") == call.task_type else {}
        where = ".".join(str(call.params[k]) for k in ("table", "column") if call.params.get(k)) or next(
            (str(call.params[k]) for k in ("entity", "key", "field", "kind", "metric") if call.params.get(k)), None)
        by_id.get(call.subproblem or "", legs[0])["steps"].append({
            "task": call.task_type, "on": where, "why": call.reason, "for_person": call.task_type in PERSON_STEPS,
            "question": str(call.params.get("question") or "") or None if call.task_type == "ask_owner" else None,
            "found": [str(t) for t in row.get("traps") or ()], "findings": [str(f) for f in row.get("findings") or ()]})
    n = 0
    for leg in legs:
        for step in leg["steps"]:
            n += 1
            step["n"] = n
    return {"version": "solution_path.v1", "goal": decomposition.decision if decomposition else None, "legs": legs,
            "build": [{"n": n + i + 1, "task": task, "what": what, "why": why, "for_person": task == "holdout"}
                      for i, (task, what, why) in enumerate(BUILD_STEPS)]}


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
    "solution_path",
    "validate_catalog_plan",
    "validate_decomposition",
]

TASK_TYPES: tuple[str, ...] = tuple(CATALOG)
