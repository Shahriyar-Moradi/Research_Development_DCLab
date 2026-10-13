"""The decision ledger (package 19.6): one record per decision DCLab makes, and the labels that say what was right.

A :class:`DecisionRecord` is one decision at one **point** (a shared vocabulary, :data:`POINTS`: the main product's
governed points and Jev's purposes, the solution plan's keys, and this studio's own, each mapped onto the others), about
one **subject** (a column, a project, a draft, a conversation, the workflow), seen in one **state**, with the
**options** it had, every **answer** it got (a rule, a model role, a classifier, a person), the value used, and the
level at which a model's answer may act (the main product's L0 to L3). Two kinds of point: ``technical`` (what a data
scientist decides: an identifier, a leak, a split, a metric) and ``workflow`` (what steers solving: the family, the
next question, the next move, the tool, the next experiment, whether to stop).

The **state** is what a decision model is trained on, so it holds no table value: only fields from
:data:`STATE_FIELDS`, each of a declared type: a band from a closed set (Jev's own bands where Jev has the field:
dtype, uniqueness, nulls, cardinality, value pattern, availability, task), a name (a column's or the target's, as
stored), name tokens, a flag, a count, a ratio, a measure (an aggregate such as a lift), or a code (a key). A field
outside the vocabulary, a band value outside its band, or a name too long to be one is refused; every answer, option
and label is checked against its point's answer shape (a flag, one of its choices, a number, a short name). What
types cannot rule out: a name field holds the name as stored (Jev's own rule: a column's name travels as untrusted
text), a code field any lower-case key, a count or a measure one number: a caller writes aggregates there, never a
row's value.

A :class:`DecisionLabel` says what the right value was, from one **source**: ``benchmark`` (a planted truth: the
replica or the benchmark case planted it), ``blind`` (a reviewer who saw only the state, with the weight of how it was
sampled), ``measured`` (a number decided it: a leak's apparent lift, a split's optimism gap, a tuning gain) or
``user_decision_exposed`` (a person who saw the answers: may demote, never promote, as in the main product). Labels
are new records superseding older ones, never edits.

Adapters: ``dclab_core.adapters.main_product_ledger`` reads the main product's ``semantic_decision_answers`` and
``project_decision_records`` rows as records and labels, field by field, and writes a record back as a
``semantic_decision_answers`` row.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

# ------------------------------------------------------------------------------------------------ the state's vocabulary
# Jev's bands, as its release defines them (main product apps/api/app/agents/semantic/releases.py at the recorded commit)
DTYPE_KINDS = ("integer", "float", "boolean", "datetime", "string", "categorical", "other")
RATIO_BANDS = ("none", "under_1pct", "from_1_to_10pct", "from_10_to_50pct", "from_50_to_90pct", "from_90_to_99pct", "over_99pct", "all")
CARDINALITY_BANDS = ("constant", "binary", "low", "medium", "high", "very_high")
VALUE_PATTERNS = ("digits", "mixed", "date_like", "long_text", "short_text", "not_text")
TASKS = ("binary_classification", "multiclass_classification", "regression")
ROLES = ("numeric", "categorical_code", "identifier", "datetime", "free_text", "other")
AVAILABILITY = ("before_prediction", "after_outcome", "unknown")
COUNT_BANDS = ("none", "one", "several")
# this studio's own bands
SIZE_BANDS = ("tiny", "small", "medium", "large", "very_large")            # rows: <200, <2k, <50k, <1M, more
FAMILIES = ("classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking",
            "segmentation", "uplift", "entity_tables", "streams", "text_or_images")
LABEL_SITUATIONS = ("column", "file", "none", "unknown")

BAND, NAME, TOKENS, FLAG, COUNT, RATIO, MEASURE, CODE = "band", "name", "tokens", "flag", "count", "ratio", "measure", "code"
NAME_CHARS = 200
CODE_PATTERN = re.compile(r"^[a-z0-9_]+([.:-][a-z0-9_]+)*$")
TOKEN_PATTERN = re.compile(r"^[a-z0-9]{1,40}$")

STATE_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    # a column, as Jev sees it
    "column": (NAME, ()), "name_tokens": (TOKENS, ()), "dtype": (BAND, DTYPE_KINDS), "uniqueness": (BAND, RATIO_BANDS),
    "nulls": (BAND, RATIO_BANDS), "cardinality": (BAND, CARDINALITY_BANDS), "value_pattern": (BAND, VALUE_PATTERNS),
    "availability": (BAND, AVAILABILITY), "role": (BAND, ROLES),
    # the problem
    "target": (NAME, ()), "task": (BAND, TASKS), "family": (BAND, FAMILIES), "labels": (BAND, LABEL_SITUATIONS),
    "rows": (BAND, SIZE_BANDS), "columns": (COUNT, ()), "classes": (COUNT, ()), "positive_rate": (BAND, RATIO_BANDS),
    "has_time": (FLAG, ()), "has_groups": (FLAG, ()), "has_text": (FLAG, ()), "tables": (COUNT, ()),
    "moment_known": (FLAG, ()), "outcome_known": (FLAG, ()), "open_questions": (COUNT, ()), "blocking": (COUNT, ()),
    # measurements a decision can be made on (aggregates, never a row)
    "apparent_lift": (MEASURE, ()), "optimism_gap": (MEASURE, ()), "tuning_gain": (MEASURE, ()), "cv_mean": (MEASURE, ()),
    "cv_std": (MEASURE, ()), "holdout_value": (MEASURE, ()), "baseline_value": (MEASURE, ()), "share": (RATIO, ()),
    "correlation": (MEASURE, ()),
    # the workflow
    "stage": (CODE, ()), "move": (CODE, ()), "from_state": (CODE, ()), "to_state": (CODE, ()), "agent": (CODE, ()),
    "pack": (CODE, ()), "kind": (CODE, ()), "metric": (CODE, ()), "strategy": (CODE, ()), "step": (COUNT, ()),
    "steps_left": (COUNT, ()), "tools_offered": (COUNT, ()), "revisits": (COUNT, ()), "budget_left": (RATIO, ()),
    "gate_required": (FLAG, ()), "approved": (FLAG, ()), "holdout_used": (COUNT, ()), "reason_given": (FLAG, ()),
}
Scalar = Union[str, int, float, bool, None]


def check_state(state: dict[str, Any]) -> list[str]:
    """Why a state cannot be kept (empty when it can): every field declared, every value of its declared type."""
    problems = []
    for name, value in state.items():
        kind, band = STATE_FIELDS.get(name, (None, ()))
        if kind is None:
            problems.append(f"{name}: not a state field (dclab_core.ledger.STATE_FIELDS): a table's values never enter a state")
        elif kind == BAND and value not in band:
            problems.append(f"{name}: {value!r} is not one of its bands ({', '.join(band)})")
        elif kind == NAME and not (isinstance(value, str) and 0 < len(value) <= NAME_CHARS and "\n" not in value):
            problems.append(f"{name}: a name of at most {NAME_CHARS} characters on one line")
        elif kind == TOKENS and not (isinstance(value, list) and len(value) <= 16 and all(isinstance(t, str) and TOKEN_PATTERN.match(t) for t in value)):
            problems.append(f"{name}: at most 16 lower-case tokens")
        elif kind == FLAG and not isinstance(value, bool):
            problems.append(f"{name}: true or false")
        elif kind == COUNT and not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            problems.append(f"{name}: a count")
        elif kind == RATIO and not (isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1):
            problems.append(f"{name}: a ratio between 0 and 1")
        elif kind == MEASURE and not (isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) < 1e12):
            problems.append(f"{name}: a finite number")
        elif kind == CODE and not (isinstance(value, str) and len(value) <= 80 and CODE_PATTERN.match(value)):
            problems.append(f"{name}: a code (lower-case key)")
    return problems


# ------------------------------------------------------------------------------------------------ the decision points
class Point(Contract):
    """One decision point: what kind of decision, what it decides about, the shape of its answer, and its names in the
    main product (its governed decision point, its Jev purpose) and in the solution plan, where it has them."""

    key: str
    kind: Literal["technical", "workflow"]
    subjects: tuple[Literal["column", "project", "draft", "conversation", "workflow", "proposal"], ...]
    answer: Literal["noul", "choice", "score", "value"]  # yes/no/unknown, one of options, a level, a value
    about: str
    main_product: Optional[str] = None
    jev: Optional[str] = None
    plan_key: Optional[str] = None
    choices: tuple[str, ...] = ()


def _p(key, kind, subjects, answer, about, main_product=None, jev=None, plan_key=None, choices=()):
    subjects = (subjects,) if isinstance(subjects, str) else tuple(subjects)
    return Point(key=key, kind=kind, subjects=subjects, answer=answer, about=about, main_product=main_product, jev=jev,
                 plan_key=plan_key, choices=tuple(choices))


DP = ("draft", "project")  # a problem is understood on Home (a draft) and in the main product on its project


T, W = "technical", "workflow"
POINTS: dict[str, Point] = {p.key: p for p in (
    # technical: what a data scientist decides (the first use)
    _p("column.is_identifier", T, "column", "noul", "Is the column an identifier (never a feature)?", "column.is_identifier", "column.is_identifier"),
    _p("column.semantic_role", T, "column", "choice", "Which role does the column play?", "column.semantic_role", "column.semantic_role", choices=ROLES),
    _p("column.missing_value_action", T, "column", "choice", "What is done with the column's missing values?", "column.missing_value_action",
       choices=("keep", "impute", "flag", "drop")),
    _p("column.category_code", T, "column", "noul", "Is the numeric column a code for categories?"),
    _p("column.time", T, "column", "noul", "Is the column the time each row happened?"),
    _p("column.group", T, "column", "noul", "Does the column name the entity rows must be grouped by?"),
    _p("feature.leakage_suspect", T, "column", "noul", "Would the column be unknown at the prediction moment, or encode the target?",
       "feature.leakage_suspect", "feature.leakage_suspect", plan_key="features.forbidden"),
    _p("target.column", T, DP, "value", "Which column is the outcome?", "target.column"),
    _p("task.framing", T, DP, "choice", "Which task is it?", plan_key="task.framing",
       choices=("binary", "multiclass", "regression")),
    _p("labels.situation", T, DP, "choice", "How are the outcomes known?", plan_key="labels.coverage", choices=LABEL_SITUATIONS),
    _p("labels.join_key", T, DP, "value", "Which key joins the label file to the table?"),
    _p("data.cases_table", T, "draft", "noul", "Is this table a list of known cases rather than the rows to score?"),
    _p("data.cleaning_step", T, "column", "choice", "Which cleaning step applies?"),
    # a code, not a closed choice: each host names its strategies (here random, group, time, time_group; the main
    # product group_disjoint, stratified_kfold, …)
    _p("split.strategy", T, DP, "value", "How are rows split for validation and the holdout?", "split.strategy", plan_key="split.strategy"),
    _p("split.folds", T, "project", "value", "How many folds?"),
    _p("data.sampling", T, "project", "choice", "Which rows does a run read?"),
    _p("metric.primary", T, DP, "value", "Which metric decides, with its constraints?", "spec.objective", plan_key="metric.primary"),
    _p("features.recipe", T, "project", "value", "Which feature recipe is kept?", plan_key="features.direction"),
    _p("model.family", T, "project", "value", "Which model family is kept?", "training.families_budget", plan_key="model.families"),
    _p("model.review", T, "project", "choice", "Does the critic's review change the model?", "experiment.review", choices=("keep", "change")),
    _p("tuning.accept", T, "project", "noul", "Is the tuned configuration kept over the default?"),
    _p("threshold.operating_point", T, "project", "value", "At which threshold or budget are cases flagged?", plan_key="threshold.policy"),
    _p("holdout.reuse", T, "project", "noul", "May the holdout be used again?"),
    _p("report.claim", T, "project", "noul", "Is the report's line kept?"),
    _p("lesson.accept", T, "project", "noul", "Is the lesson kept for the workspace?"),
    _p("improve.next_action", T, "project", "value", "What is improved next?", "improve.next_action"),
    _p("experiment.accept", T, "project", "noul", "Is the experiment's result kept?"),
    _p("model.promote", T, "project", "noul", "Is the model promoted to the project's champion?"),
    # workflow: what steers solving (the second use)
    _p("problem.family", W, DP, "choice", "Which family of problem is it?", choices=FAMILIES),
    _p("problem.ready", W, DP, "noul", "Is the problem understood enough to build?"),
    _p("problem.pack", W, DP, "value", "Which domain pack does it start from?"),
    _p("workflow.next_question", W, "conversation", "value", "Which question is asked next, if any?"),
    _p("workflow.template", W, DP, "value", "Which solution workflow does it follow?"),
    _p("workflow.move", W, "workflow", "noul", "Is this move allowed now?"),
    _p("workflow.revisit", W, "workflow", "value", "Which earlier step is revisited, if any?"),
    _p("workflow.stage_approval", W, "project", "noul", "Is the stage's choice approved?"),
    _p("workflow.stop", W, "workflow", "noul", "Is the work finished, or does it go on?"),
    _p("agent.tool_choice", W, "conversation", "value", "Which tool does the agent call next?"),
    _p("experiment.next", W, "project", "value", "Which experiment is run next?"),
    _p("command.intent_route", W, "conversation", "value", "Which quick action does the person's message ask for?",
       "command.intent_route", "command.intent_route"),
    # operations, after a model is in use (the main product's governed points; this studio decides none of them yet)
    _p("ops.diagnose", W, "project", "value", "What does the monitoring say is wrong?", "ops.diagnose"),
    _p("ops.retrain", W, "project", "noul", "Is the model retrained now?", "ops.retrain"),
    _p("ops.release", W, "project", "noul", "Is the new model released?", "ops.release"),
    _p("ops.rollback", W, "project", "noul", "Is the release rolled back?", "ops.rollback"),
    _p("proposal.completeness", W, "proposal", "score", "How complete is the proposal for a reviewer?",
       "proposal.completeness", "proposal.completeness"),
)}
ANSWERERS = ("rule", "model_role", "classifier", "person")
LEVELS = (0, 1, 2, 3)  # L0 shadow (never used) … L3 (acts alone): the main product's levels
LABEL_SOURCES = ("benchmark", "blind", "measured", "user_decision_exposed")


def answer_problem(point: "Point", value: Any) -> Optional[str]:
    """Why ``value`` is not an answer at ``point`` (None when it is, or when it is None: an abstention). A yes/no point
    answers a flag; a choice one of its choices; a score a number; a value point a name, a code or a number, short.
    So no answer, option or label carries a row's text."""
    if value is None:
        return None
    if point.answer == "noul":
        return None if isinstance(value, bool) else "a yes/no point answers true or false"
    if point.answer == "score":
        return None if isinstance(value, (int, float)) and not isinstance(value, bool) else "a score is a number"
    if point.answer == "choice" and point.choices:
        return None if value in point.choices else f"not one of the choices ({', '.join(point.choices)})"
    if isinstance(value, str):
        return None if len(value) <= NAME_CHARS and "\n" not in value else f"a name or code of at most {NAME_CHARS} characters"
    return None if isinstance(value, (int, float)) else "a name, a code or a number"


# ------------------------------------------------------------------------------------------------ records and labels
class Subject(Contract):
    kind: Literal["column", "project", "draft", "conversation", "workflow", "proposal"]
    key: str = Field(min_length=1, max_length=NAME_CHARS, description="What it is about: a column's name, a project's id, a move.")


class Answer(Contract):
    """One answerer's answer: who (a rule's id, a model role and its model, a classifier, a person's role), what, how sure."""

    by: Literal["rule", "model_role", "classifier", "person"]
    answerer: str = Field(min_length=1, max_length=120, description="The rule's id, the model's id, the person's role.")
    value: Scalar = Field(description="The answer in the point's vocabulary; None when a model abstained (outside its acting band).")
    raw: Scalar = Field(default=None, description="A model's answer as it gave it, before its acting band: a probability for a "
                        "yes/no point, a label for a choice (kept so a row read from the main product goes back unchanged).")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    reason: Optional[str] = Field(default=None, max_length=600, description="Its stated reason (a rule's words or a model's): advisory.")
    source: Optional[str] = Field(default=None, max_length=120, description="Where it came from: a prompt's hash, a release, a request id.")


class DecisionRecord(Versioned):
    """One decision: its point, its subject, the state it was made in, its options, every answer, and what was used."""

    schema_version: Literal["dclab_core.decision_record.v1"] = _version("decision_record")
    id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{6,80}$")
    point: str
    subject: Subject
    state: dict[str, Any] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list, max_length=64)
    answers: list[Answer] = Field(default_factory=list, max_length=20)
    value_used: Scalar = None
    used: Literal["rule", "model_role", "classifier", "person", "none"] = "none"
    review: bool = Field(default=False, description="Held for a person: the answers disagreed at a level that may not act alone.")
    level: int = Field(default=0, ge=0, le=3, description="How far a model may act here: 0 shadow, 1 flag, 2 act with a check, 3 act.")
    evidence_partition: Optional[Literal["metadata", "train"]] = Field(default=None, description="What the state was read from.")
    workspace: Optional[str] = Field(default=None, max_length=64)
    project: Optional[str] = Field(default=None, max_length=64)
    experiment: Optional[str] = Field(default=None, max_length=80)
    dataset: Optional[str] = Field(default=None, max_length=80)
    draft: Optional[str] = Field(default=None, max_length=64)
    run: Optional[str] = Field(default=None, max_length=80)
    conversation: Optional[str] = Field(default=None, max_length=80)
    at: str

    @field_validator("point")
    @classmethod
    def _known(cls, value: str) -> str:
        if value not in POINTS:
            raise ValueError(f"unknown decision point {value!r}: the vocabulary is dclab_core.ledger.POINTS")
        return value

    @model_validator(mode="after")
    def _checked(self) -> "DecisionRecord":
        point = POINTS[self.point]
        if self.subject.kind not in point.subjects:
            raise ValueError(f"{self.point} decides about a {' or a '.join(point.subjects)}, not a {self.subject.kind}")
        problems = check_state(self.state)
        if problems:
            raise ValueError("state refused: " + "; ".join(problems))
        if self.used != "none" and not any(a.by == self.used for a in self.answers):
            raise ValueError(f"the value used came from {self.used}, which gave no answer")
        if self.used in ("model_role", "classifier") and self.level < 2:
            raise ValueError(f"a model's answer is used only from level 2 (this record is at level {self.level})")
        for what, value in [("value_used", self.value_used), *((f"{a.by}'s answer", a.value) for a in self.answers)]:
            problem = answer_problem(point, value)
            if problem:
                raise ValueError(f"{what}: {problem}")
        for option in self.options:
            if len(option) > NAME_CHARS or "\n" in option or (point.choices and option not in point.choices):
                raise ValueError(f"option {option[:40]!r}: an option is one of the point's choices, or a short name or code")
        return self

    @property
    def policy_outcome(self) -> str:
        """As the main product says where the value came from: rule, ai, review or human."""
        if self.review:
            return "review"
        return {"rule": "rule", "model_role": "ai", "classifier": "ai", "person": "human", "none": "rule"}[self.used]

    @property
    def kind(self) -> str:
        return POINTS[self.point].kind

    def answer(self, by: str) -> Optional[Answer]:
        return next((a for a in self.answers if a.by == by), None)

    def agreement(self) -> str:
        """The rule's answer against a model's (role or classifier), as the main product says it: agree, disagree,
        abstain (the model gave no value), unavailable (no model answered)."""
        rule = self.answer("rule")
        model = self.answer("classifier") or self.answer("model_role")
        if model is None or rule is None:
            return "unavailable"
        if model.value is None:
            return "abstain"
        return "agree" if model.value == rule.value else "disagree"


class Evidence(Contract):
    kind: Literal["planted_truth", "apparent_lift", "optimism_gap", "tuning_gain", "ladder", "holdout_vs_cv", "review", "override"]
    value: Optional[float] = None
    ref: Optional[str] = Field(default=None, max_length=200, description="What it cites: a planted case's id, a record path.")


class DecisionLabel(Versioned):
    """What the right value of a decision was, from one source; a new label supersedes an older one, never edits it."""

    schema_version: Literal["dclab_core.decision_label.v1"] = _version("decision_label")
    id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{6,80}$")
    record: str = Field(min_length=1, max_length=80, description="The decision record it labels.")
    point: str
    value: Scalar
    source: Literal["benchmark", "blind", "measured", "user_decision_exposed"]
    by: str = Field(min_length=1, max_length=120, description="Who or what: a replica's name, a reviewer's role, a measurement.")
    at: str
    supersedes: Optional[str] = Field(default=None, max_length=80)
    evidence: Optional[Evidence] = None
    weight: Optional[float] = Field(default=None, gt=0, description="A blind label's sampling weight (1 / its probability of being drawn).")
    weight_note: Optional[str] = Field(default=None, max_length=300, description="Why a blind label has no weight here: its "
                                       "source keeps the sampling probability elsewhere. An estimate must not use it unweighted.")

    @model_validator(mode="after")
    def _checked(self) -> "DecisionLabel":
        if self.point not in POINTS:
            raise ValueError(f"unknown decision point {self.point!r}")
        if self.source == "blind" and self.weight is None and not self.weight_note:
            raise ValueError("a blind label carries its sampling weight, or says where its source keeps it (weight_note)")
        if self.source != "blind" and (self.weight is not None or self.weight_note):
            raise ValueError("only a blind label is weighted")
        problem = answer_problem(POINTS[self.point], self.value)
        if problem:
            raise ValueError(f"value: {problem}")
        if self.supersedes == self.id:
            raise ValueError("a label cannot supersede itself")
        return self


# ------------------------------------------------------------------------------------------------ which label counts (19.8)
PRECEDENCE = ("blind", "benchmark", "measured", "user_decision_exposed")  # first is strongest


def current_labels(labels: list[DecisionLabel]) -> list[DecisionLabel]:
    """The labels no later label supersedes."""
    superseded = {x.supersedes for x in labels if x.supersedes}
    return [x for x in labels if x.id not in superseded]


def effective_label(record: DecisionRecord, labels: list[DecisionLabel]) -> Optional[DecisionLabel]:
    """The one label that says what was right for ``record``: among the labels of this record no later one supersedes,
    the strongest source wins (blind, benchmark, measured, then exposed), the latest within a source. An exposed label
    (a person who saw the answers) may demote, never promote: it counts only when it says the value used was wrong,
    since a person who saw the answer and agreed is not evidence that it was right: it counts when it contradicts an
    answer a rule or a model gave. A label without a value says nothing and never counts."""
    mine = [x for x in current_labels(labels) if x.record == record.id and x.value is not None]
    shown = [a.value for a in record.answers if a.by != "person" and a.value is not None]
    for source in PRECEDENCE:
        found = sorted((x for x in mine if x.source == source), key=lambda x: x.at)
        if source == "user_decision_exposed":
            found = [x for x in found if any(v != x.value for v in shown)]
        if found:
            return found[-1]
    return None


# ------------------------------------------------------------------------------------------------ blind labels (19.8)
BLIND_RATES = {"agree": 0.10, "disagree": 0.60, "single": 0.10, "abstain": 0.30}  # disagreements over-sampled
# points a blind reviewer cannot answer from the state alone, or whose state names the answer
NOT_BLIND = frozenset({"workflow.revisit", "workflow.template", "problem.pack", "workflow.next_question", "agent.tool_choice",
                       "workflow.stop", "report.claim", "lesson.accept", "workflow.stage_approval"})


def stratum(record: DecisionRecord) -> str:
    """How its answers relate: they disagree (any two answers with a value differ: a rule against a person, a model
    against a rule), agree, a model abstained, or there was a single answer."""
    values = [a.value for a in record.answers if a.value is not None]
    if len({repr(v) for v in values}) > 1:
        return "disagree"
    if any(a.by in ("model_role", "classifier") and a.value is None for a in record.answers):
        return "abstain"
    return "agree" if len(values) > 1 else "single"


def blind_probability(record: DecisionRecord, rates: Optional[dict[str, float]] = None) -> float:
    """The probability with which ``record`` is drawn for a blind label: fixed per stratum, so every blind label
    carries its weight, 1 / this, and estimates can be corrected for the over-sampling."""
    return float((rates or BLIND_RATES).get(stratum(record), 0.10))


def blind_options(record: DecisionRecord) -> list[str]:
    """What a blind reviewer chooses among: yes or no, the point's choices, or the options the decision had."""
    point = POINTS[record.point]
    if point.answer == "noul":
        return ["yes", "no"]
    return list(point.choices) or list(record.options)


def blind_eligible(record: DecisionRecord) -> bool:
    """A decision a reviewer can answer from what it shows: a point whose state does not name the answer, with options."""
    return record.point not in NOT_BLIND and bool(blind_options(record))


def blind_draw(record: DecisionRecord, seed: str = "blind-v1", rates: Optional[dict[str, float]] = None) -> bool:
    """Whether ``record`` is in the blind sample: a hash of its id and the seed against its stratum's probability, so
    the draw is reproducible and its probability known before anyone labels."""
    import hashlib

    if not blind_eligible(record):
        return False
    u = int(hashlib.sha256(f"{seed}:{record.id}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return u < blind_probability(record, rates)


def blind_view(record: DecisionRecord) -> dict[str, Any]:
    """What a blind reviewer sees: the question, the subject, the state and the options; never an answer, the value
    used, or who decided."""
    point = POINTS[record.point]
    state = {k: v for k, v in record.state.items() if v != record.value_used or isinstance(v, bool)}  # never the answer itself
    return {"record": record.id, "point": record.point, "question": point.about, "answer": point.answer, "subject": record.subject.model_dump(),
            "state": state, "options": blind_options(record)}


CONTRACTS.update({"decision_record": DecisionRecord, "decision_label": DecisionLabel})
