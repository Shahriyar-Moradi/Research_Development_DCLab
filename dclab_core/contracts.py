"""The contracts: what dclab_core produces, as frozen and versioned pydantic models.

A host stores them as JSON (each has a published JSON Schema under ``schemas/``) and maps them onto its own records.
:class:`ProblemBrief` covers the main product's ``ProblemSpec`` (business objective, task type, target column,
prediction unit, prediction time column, prediction horizon, primary metric, constraints, success criteria);
:class:`SolutionPlan` holds one :class:`Decision` per decision point (split strategy, primary metric, model families,
...), each with its reason and the alternatives set aside, so a host adopts it through its own governed decisions.

Versioning. ``schema_version`` names a contract and its version. *Any* change to a contract's shape (a field added,
removed, renamed or retyped, in it or in a model it contains) moves the version: readers refuse unknown fields
(``extra="forbid"``), so even an added optional field would break a reader pinned to an older release. The schema
under each version is frozen in ``schemas/baseline/`` and a test fails when the shape changes under the same version.

Reading. Hosts read documents with :meth:`Versioned.read`, which requires the ``schema_version`` to be present and
to match: a document is never silently stamped as the current version.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from dclab_core import evidence as ev

TaskType = Literal["binary", "multiclass", "regression", "unknown"]
Status = Literal["stated", "inferred", "unknown"]
Source = Literal["text", "data", "person", "advisor", "rule"]
Severity = Literal["info", "warning", "critical"]
Scope = Literal["training_rows", "all_rows", "label_file", "schema"]
MAX_TEXT = 4000

_FACT_RULES = {"allOf": [
    {"if": {"properties": {"status": {"const": "stated"}}, "required": ["status"]},
     "then": {"required": ["quote", "value"], "properties": {"quote": {"type": "string", "minLength": 1},
                                                             "source": {"enum": ["text", "person"]}}}},
    {"if": {"properties": {"status": {"const": "inferred"}}, "required": ["status"]},
     "then": {"required": ["reason", "value"], "properties": {"reason": {"type": "string", "minLength": 1}}}},
    {"if": {"properties": {"status": {"const": "unknown"}}},
     "then": {"properties": {"value": {"type": "null"}}}},
]}


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _require_version(schema: dict[str, Any]) -> None:
    schema.setdefault("required", [])
    if "schema_version" not in schema["required"]:
        schema["required"].insert(0, "schema_version")


class Versioned(Contract):
    """A contract a host stores on its own: it carries its version, and reading requires it."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_require_version)

    @classmethod
    def read(cls, document: Union[str, bytes, Mapping[str, Any]]):
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        expected = cls.model_fields["schema_version"].default
        if data.get("schema_version") != expected:
            raise ValueError(f"expected schema_version {expected!r}, found {data.get('schema_version')!r}")
        return cls.model_validate(data)


def _version(name: str, major: int = 1) -> Any:
    return Field(default=f"dclab_core.{name}.v{major}", frozen=True)


# ---------------------------------------------------------------------------------------------------- facts


class Fact(Contract):
    """One thing known about the problem, and how it is known.

    *stated*: in the person's own words (``quote`` holds them; ``source`` is ``text`` or ``person``); *inferred*:
    concluded from data, rules or an advisor (``reason`` says how); *unknown*: not known yet, so no value. The same
    rules are in the JSON Schema (``allOf``), so a host validating JSON gets them too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_FACT_RULES)

    value: JsonValue = None
    status: Status = "unknown"
    quote: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    reason: str = Field(default="", max_length=MAX_TEXT)
    source: Source = "rule"

    @field_validator("value")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        return ev.finite(value)

    @model_validator(mode="after")
    def _known_how(self) -> "Fact":
        if self.status == "stated":
            if not (self.quote and self.quote.strip()):
                raise ValueError("a stated fact needs the quote it came from")
            if self.source not in ("text", "person"):
                raise ValueError("only a person's words (the text, or what a person entered) are stated")
        if self.status == "inferred" and not self.reason.strip():
            raise ValueError("an inferred fact needs its reason")
        if self.status != "unknown" and self.value is None:
            raise ValueError("a known fact has a value")
        if self.status == "unknown" and self.value is not None:
            raise ValueError("an unknown fact has no value")
        return self

    @classmethod
    def unknown(cls):
        return cls()

    @property
    def known(self) -> bool:
        return self.status != "unknown"


class TextFact(Fact):
    value: Optional[str] = Field(default=None, max_length=MAX_TEXT)


class TaskFact(Fact):
    value: Optional[TaskType] = None


class LabelSourceFact(Fact):
    value: Optional[Literal["column", "file", "none"]] = None


class CoverageFact(Fact):
    value: Optional[Literal["complete", "partial"]] = None


class NamesFact(Fact):
    value: Optional[list[str]] = Field(default=None, max_length=20)


class Constraint(Contract):
    kind: Literal["confidential", "latency", "review_capacity", "budget", "fairness", "regulation", "other"]
    text: str = Field(max_length=MAX_TEXT)
    status: Literal["stated", "inferred"] = "stated"
    quote: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    reason: str = Field(default="", max_length=MAX_TEXT)
    source: Source = "text"

    @model_validator(mode="after")
    def _grounded(self) -> "Constraint":
        if self.status == "stated" and not (self.quote and self.quote.strip()):
            raise ValueError("a stated constraint needs its quote")
        if self.status == "inferred" and not self.reason.strip():
            raise ValueError("an inferred constraint needs its reason")
        return self


class Deliverable(Contract):
    name: str = Field(max_length=300)
    points: Optional[float] = Field(default=None, ge=0)
    status: Literal["stated", "inferred"] = "stated"
    quote: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    reason: str = Field(default="", max_length=MAX_TEXT)
    source: Source = "text"

    @model_validator(mode="after")
    def _grounded(self) -> "Deliverable":
        if self.status == "stated" and not (self.quote and self.quote.strip()):
            raise ValueError("a stated deliverable needs its quote")
        if self.status == "inferred" and not self.reason.strip():
            raise ValueError("an inferred deliverable needs its reason")
        return self


class Question(Contract):
    """An open question, ranked by how much its answer would change the plan (1 = most)."""

    field: str
    text: str = Field(max_length=1000)
    impact: int = Field(ge=1, le=5)
    options: list[str] = Field(default_factory=list, max_length=8)


def _fact(cls: type, description: str) -> Any:
    return Field(default_factory=cls, description=description)


class ProblemBrief(Versioned):
    """The problem as understood from its description, before any data is analysed."""

    schema_version: Literal["dclab_core.problem_brief.v1"] = _version("problem_brief")
    source_digest: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$",
                                         description="SHA-256 of the description and file headers it was built from")
    goal: TextFact = _fact(TextFact, "The business objective (ProblemSpec.business_objective).")
    action: TextFact = _fact(TextFact, "What is done with each prediction.")
    unit: TextFact = _fact(TextFact, "What one scored row is: a ride, a customer-month, a ticket (ProblemSpec.prediction_unit).")
    task: TaskFact = _fact(TaskFact, "binary, multiclass or regression (ProblemSpec.task_type).")
    target_column: TextFact = _fact(TextFact, "The column holding the label, when there is one (ProblemSpec.target_column).")
    target_definition: TextFact = _fact(TextFact, "How the label is defined when no column holds it, e.g. 'listed in known_frauds.csv'.")
    label_source: LabelSourceFact = _fact(LabelSourceFact, "Where labels come from: a column, another file, or none.")
    label_coverage: CoverageFact = _fact(CoverageFact, "complete, or partial (a sample: unlabelled is not negative).")
    prediction_moment: TextFact = _fact(TextFact, "When the prediction is made and what is known then.")
    time_column: TextFact = _fact(TextFact, "The column saying when each row is predicted (ProblemSpec.prediction_time_column).")
    horizon: TextFact = _fact(TextFact, "How far ahead the prediction looks (ProblemSpec.prediction_horizon).")
    entities: NamesFact = _fact(NamesFact, "Columns of the entities rows repeat for (a biker, a subscriber): kept on one side of a split.")
    primary_metric: TextFact = _fact(TextFact, "The metric the description names, if any (ProblemSpec.primary_metric).")
    constraints: list[Constraint] = Field(default_factory=list, max_length=30)
    deliverables: list[Deliverable] = Field(default_factory=list, max_length=50)
    success_criteria: list[TextFact] = Field(default_factory=list, max_length=20)
    open_questions: list[Question] = Field(default_factory=list, max_length=20)


# ---------------------------------------------------------------------------------------------------- data


class Finding(Contract):
    """One thing the data shows, with its evidence (numbers and names, never row values) and what it implies."""

    id: str = Field(pattern=r"^F-[a-z0-9_.-]+$")
    topic: str = Field(max_length=80)
    statement: str = Field(max_length=MAX_TEXT)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    implication: str = Field(max_length=MAX_TEXT)
    affects: list[str] = Field(default_factory=list, max_length=12)   # decision keys it bears on
    severity: Severity = "info"
    scope: Scope = "training_rows"

    @field_validator("evidence", mode="before")
    @classmethod
    def _summary_only(cls, evidence: Any) -> Any:
        try:
            ev.check(evidence, ev.FINDING)
        except ev.NotASummary as exc:
            raise ValueError(str(exc)) from None
        return ev.plain(evidence)


class TableSummary(Contract):
    name: str = Field(max_length=300)
    role: Literal["main", "labels", "related"] = "main"
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)


class DataFindings(Versioned):
    schema_version: Literal["dclab_core.data_findings.v1"] = _version("data_findings")
    task: TaskType = "unknown"
    target: Optional[str] = None
    tables: list[TableSummary] = Field(default_factory=list, max_length=20)
    training_rows: Optional[int] = Field(default=None, ge=0,
                                         description="The rows statistics that steer choices were computed on.")
    findings: list[Finding] = Field(default_factory=list, max_length=200)

    @field_validator("findings")
    @classmethod
    def _unique_ids(cls, findings: list[Finding]) -> list[Finding]:
        ids = [f.id for f in findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding ids must be unique")
        return findings


# ---------------------------------------------------------------------------------------------------- plan


class Alternative(Contract):
    choice: JsonValue
    why_not: str = Field(min_length=3, max_length=MAX_TEXT)

    @field_validator("choice")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        return ev.finite(value)


class Decision(Versioned):
    """One decision of the plan: what was chosen, why (citing findings or evidence records), and what was set aside.

    Keys are decision points, for example ``task.framing``, ``labels.coverage``, ``split.strategy``,
    ``metric.primary``, ``baseline``, ``features.direction``, ``features.forbidden``, ``model.families``,
    ``threshold.policy``.
    """

    schema_version: Literal["dclab_core.decision.v1"] = _version("decision")
    key: str = Field(pattern=r"^[a-z_]+(\.[a-z_]+)*$", max_length=80)
    choice: JsonValue
    reason: str = Field(min_length=10, max_length=MAX_TEXT)
    cites: list[str] = Field(default_factory=list, max_length=20)
    alternatives: list[Alternative] = Field(default_factory=list, max_length=8)
    source: Source = "rule"
    status: Literal["proposed", "accepted", "overridden"] = "proposed"

    @field_validator("choice")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        return ev.finite(value)


class Risk(Contract):
    statement: str = Field(max_length=MAX_TEXT)
    check: str = Field(max_length=MAX_TEXT)
    cites: list[str] = Field(default_factory=list, max_length=20)


class SolutionPlan(Versioned):
    schema_version: Literal["dclab_core.solution_plan.v1"] = _version("solution_plan")
    task: TaskType = "unknown"
    decisions: list[Decision] = Field(default_factory=list, max_length=60)
    risks: list[Risk] = Field(default_factory=list, max_length=30)
    would_change: list[str] = Field(default_factory=list, max_length=20)   # what would change this plan

    @field_validator("decisions")
    @classmethod
    def _one_per_key(cls, decisions: list[Decision]) -> list[Decision]:
        keys = [d.key for d in decisions]
        if len(keys) != len(set(keys)):
            raise ValueError("one decision per key")
        return decisions

    def decision(self, key: str) -> Optional[Decision]:
        return next((d for d in self.decisions if d.key == key), None)


# ---------------------------------------------------------------------------------------------------- review, report


class Hypothesis(Contract):
    statement: str = Field(max_length=MAX_TEXT)
    cites: list[str] = Field(min_length=1, max_length=20)   # the measured weakness it follows from
    test: str = Field(max_length=MAX_TEXT)


class Experiment(Contract):
    name: str = Field(max_length=200)
    hypothesis: int = Field(ge=0)                           # index into Review.hypotheses
    expected_gain: str = Field(max_length=500)
    cost: Literal["low", "medium", "high"] = "medium"


class Review(Versioned):
    schema_version: Literal["dclab_core.review.v1"] = _version("review")
    weaknesses: list[Finding] = Field(default_factory=list, max_length=100)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=20)
    next_experiments: list[Experiment] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _experiments_test_hypotheses(self) -> "Review":
        for experiment in self.next_experiments:
            if experiment.hypothesis >= len(self.hypotheses):
                raise ValueError(f"experiment {experiment.name!r} tests hypothesis {experiment.hypothesis}, which does not exist")
        return self


class ReportSection(Contract):
    title: str = Field(max_length=200)
    body: str = Field(max_length=20_000)
    cites: list[str] = Field(default_factory=list, max_length=50)


class SolutionReport(Versioned):
    schema_version: Literal["dclab_core.solution_report.v1"] = _version("solution_report")
    title: str = Field(max_length=300)
    sections: list[ReportSection] = Field(default_factory=list, max_length=40)
    limits: list[str] = Field(default_factory=list, max_length=30)
    numbers: dict[str, float] = Field(default_factory=dict)  # every number the prose uses, by the record key it came from


CONTRACTS: dict[str, type[Versioned]] = {
    "problem_brief": ProblemBrief,
    "data_findings": DataFindings,
    "solution_plan": SolutionPlan,
    "decision": Decision,
    "review": Review,
    "solution_report": SolutionReport,
}

import dclab_core.experience  # noqa: E402,F401  (registers its contracts in CONTRACTS: every reader sees all of them)
import dclab_core.card  # noqa: E402,F401  (the problem card, 20.1)
