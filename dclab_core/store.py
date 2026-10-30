"""The store's contracts (the storing phase, 32.1): what is kept, how it is shaped, and what each of the eight model types
learns from. ``docs/guides/STORING_PHASE_PLAN.md`` is the plan.

* **Three tiers.** ``raw`` keeps a source whole (a :class:`StoreEntry` says what was kept, with its hash); ``records`` holds
  the same data normalised (:class:`StoreRecord`, one shape per kind); ``sets`` holds training examples per model type
  (:class:`DecisionExample`, :class:`ChatExample`, :class:`PreferencePair`, :class:`SolverTrajectory`, :class:`Episode`,
  :class:`JudgedExample`, :class:`RetrievalExample`), each set with its :class:`SetManifest`.
* **Every record and example carries its** :class:`Provenance`: the raw entry and its hash, the source, the commit, the
  workspace, the project or draft (deleted or not), who made it, the consent, and when.
* **Nine model types** (:data:`MODEL_TYPES`), each with its folder: the eight of :data:`ModelType`, whose examples share
  the contracts below, and ``m7_routing`` (2026-10-29), the routing model's data, whose examples are their own contract
  (:class:`dclab_core.routing_example.RoutingExample`). **Raw kinds** (:data:`KINDS`), each with its folder.
* **Curation is a decision** (:class:`CurationDecision`): a curator's choice for one record, kept with its reason.
* **Reasoning is labelled** (:data:`REASONING_SOURCES`): recorded where it was made, drawn from the record by code,
  written by a teacher afterwards, or visible: the agent's own words before its tool call, kept when the model's
  reasoning was not returned (never counted as recorded reasoning).

Pure pydantic. The host owns the files, the agents and the models; the core never reaches a disk or a database.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field, field_validator, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

# ----------------------------------------------------------------------------------------------- the vocabulary
ModelType = Literal["m1_students", "m2_workflow_decisions", "m3_ml_decisions", "m4_workflow_llm", "m5_solver_llm",
                    "m6_stepwise_decisions", "m7_judges", "m8_retrieval"]
MODEL_TYPES: dict[str, str] = {  # folder -> what it is (the plan's table)
    "m1_students": "Student models: small, many, one or more per decision of every other model",
    "m2_workflow_decisions": "Workflow decision models for DCLab, the R&D system and internal automation",
    "m3_ml_decisions": "Data science and machine-learning decision models (classification and the other families)",
    "m4_workflow_llm": "An LLM that knows the machine-learning workflow, from the problem and the solution to the code",
    "m5_solver_llm": "An LLM that knows many machine-learning problems, solutions and code, to solve MLE-bench",
    "m6_stepwise_decisions": "A step-by-step decision model",
    "m7_judges": "A judge or reward model: scores a plan, a solution, a report, a step or code",
    "m8_retrieval": "A retriever and reranker: finds the evidence, the example, the pitfall or the case that fits",
    # 2026-10-29 (the owner, P8): the 7th model, routing; not in ModelType (adding it there would move every example contract's
    # version); its examples are dclab_core.routing_example.RoutingExample
    "m7_routing": "A routing model: which kind of help a request needs and where Home sends it, from the person's words",
}
Kind = Literal["sessions", "actions", "decisions", "conversations", "tasks", "workflows", "fixes", "tests", "model_requests",
               "agents", "teachers", "students", "audit", "deletions", "curation", "other"]
KINDS: dict[str, str] = {  # raw folder -> what it keeps
    "sessions": "the experience log's events",
    "actions": "agent steps and tool calls",
    "decisions": "the decision ledger: records, labels and outcomes",
    "conversations": "Home, the intern, the workspace chat, the twin, MCP",
    "tasks": "every job's state changes; the development sessions' task lists",
    "workflows": "each walk from a draft to a report; team and orchestra boards, walks and interrupts",
    "fixes": "a failing test, the passing run after it, and the commits between",
    "tests": "test runs and gates",
    "model_requests": "every model request with its shadows and checks",
    "agents": "other agents' sessions: Claude Code (the main checkout and its worktrees), testers'",
    "teachers": "every teacher's answer, whole, with its prompt, thinking, cost and version",
    "students": "every student version, whole: the fitted model, its training snapshot, card, measurements, shadow answers",
    "audit": "the audit trail",
    "deletions": "each deletion in the product, recorded beside what it removed",
    "curation": "every curator's choice: where each record went, its quality, why, and its failed answers",
    "other": "anything else a workspace holds (logs, results, designs, transcripts, settings), so nothing is left out",
}
Split = Literal["train", "dev", "sealed"]
Consent = Literal["operate", "improve", "train"]
LabelSource = Literal["person", "measured", "benchmark", "blind", "teacher", "rule"]  # who knows: a teacher's or a rule's label is silver
CONFIRMED = ("person", "measured", "benchmark", "blind")  # the sources that confirm a label (blind: weighted by its draw)
LEDGER_SOURCES: dict[str, str] = {  # the ledger's label sources (dclab_core.ledger.LABEL_SOURCES) in the store's words
    "user_decision_exposed": "person", "measured": "measured", "benchmark": "benchmark", "blind": "blind", "model": "teacher"}
ReasoningSource = Literal["recorded", "from_record", "teacher", "visible", "replayed"]  # replayed: the same inputs asked again with thinking on
REASONING_SOURCES = ("recorded", "from_record", "teacher", "visible")
Quality = Literal["keep", "silver", "out"]  # out: kept out of the set (quarantined with why), never deleted
SLUG = re.compile(r"^[a-z0-9][a-z0-9_.:\-]{0,119}$")


def _slug(value: str) -> str:
    if not SLUG.match(value or ""):
        raise ValueError(f"{value!r}: a lower-case name (letters, digits, _ . : -), at most 120 characters")
    return value


# ------------------------------------------------------------------------------------------- tier 1: kept whole
class StoreEntry(Versioned):
    """One kept source (v1): what it was, where it came from, its version and hash; never a value of its data."""

    schema_version: Literal["dclab_core.store_entry.v1"] = _version("store_entry")
    id: str = Field(max_length=200)
    kind: Kind
    source: str = Field(max_length=4000)         # what was copied, relative to its workspace (a log file, a job, a transcript), whole
    workspace: str = Field(max_length=120)
    version: int = Field(ge=1)                   # the source's n-th kept version: a changed source is a new version beside the old
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    captured_at: str
    consent: Consent = "operate"
    run: Optional[str] = Field(default=None, max_length=120)  # the keeper's run
    path: str = Field(max_length=400)            # where it is, relative to the store's root
    note: Optional[str] = Field(default=None, max_length=600)  # e.g. a source that could not be read, kept as bytes
    offset: int = Field(default=0, ge=0)         # a log's new tail starts here (0: the whole source)
    whole_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")  # the hash of the source up to this version's end

    @field_validator("workspace")
    @classmethod
    def _workspace(cls, value: str) -> str:
        return _slug(value)


class Provenance(Contract):
    """Where a record or an example came from (carried by every one)."""

    entry: str = Field(max_length=200)           # the raw entry's id
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(max_length=4000)
    workspace: str = Field(max_length=120)
    commit: Optional[str] = Field(default=None, max_length=64)
    project: Optional[str] = Field(default=None, max_length=120)
    draft: Optional[str] = Field(default=None, max_length=120)
    deleted: bool = False                        # the project or draft was deleted in the product; the store kept it
    by: Optional[str] = Field(default=None, max_length=120)  # the person or agent
    consent: Consent = "operate"
    at: str
    rule: Optional[str] = Field(default=None, max_length=80)  # the capture rule (and its version) it was checked by


# --------------------------------------------------------------------------------------------- tier 2: records
class StoreRecord(Versioned):
    """One normalised record of a kind (v1): its body is the kind's own shape, scrubbed (no secret, no table row)."""

    schema_version: Literal["dclab_core.store_record.v1"] = _version("store_record")
    id: str = Field(max_length=200)
    kind: Kind
    type: str = Field(max_length=80)             # the record type within its kind (event, step, decision, label, turn, job_state ...)
    at: str                                      # when it happened (not when it was kept): the state is what was known then
    body: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    split: Optional[Split] = None                # from its capture rule's sealed key (None: its key is missing, flagged)
    flags: list[str] = Field(default_factory=list)  # what its capture rule found missing, late or unconfirmed


# ------------------------------------------------------------------------------------------ tier 3: examples
class _Example(Versioned):
    id: str = Field(max_length=200)
    model_type: ModelType
    split: Split
    provenance: Provenance
    more: list[Provenance] = Field(default_factory=list)  # the other raw entries it spans (an episode, a trajectory)
    weight: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _shape_fits(self):
        shape = self.schema_version.split(".")[1]
        if shape not in EXAMPLE_SHAPES.get(self.model_type, ()):
            raise ValueError(f"a {shape} is not an example of {self.model_type} ({', '.join(EXAMPLE_SHAPES.get(self.model_type, ()))})")
        return self


def _state_ok(state: dict[str, Any]) -> dict[str, Any]:
    from dclab_core.ledger import check_state

    problems = check_state(state)
    if problems:
        raise ValueError("; ".join(problems))
    return state


class Answer(Contract):
    by: str = Field(max_length=120)              # rule:..., model:..., person, student:...
    value: Any = None
    reason: Optional[str] = Field(default=None, max_length=1200)


class DecisionExample(_Example):
    """M1, M2, M3: one decision as a classifier learns it (v2: ``split_group``, the table or project it belongs to, so folds and
    splits keep one table's rows together; v1 reads, its split group unknown)."""

    schema_version: Literal["dclab_core.decision_example.v4"] = _version("decision_example", 4)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.decision_example.v1", "dclab_core.decision_example.v2",
                                                 "dclab_core.decision_example.v3"})
    point: str = Field(max_length=120)
    group: str = Field(max_length=80)            # M2: the system (dclab, rnd, internal); M3: the family; M1: the student's model
    split_group: Optional[str] = Field(default=None, max_length=200)  # the record's sealed group (a project, a draft's project, a table)
    state: dict[str, Any] = Field(default_factory=dict)   # bands, codes, counts, flags: the ledger's state, never a value
    options: list[str] = Field(default_factory=list)
    answers: list[Answer] = Field(default_factory=list)
    label: Any = None
    label_source: Optional[LabelSource] = None
    # v3 (2026-10-28, the owner: every example carries its reasoning): why the decision, from what was known when it was made
    # (never the label or a person's answer), with its source; None on rows whose builder has none, read from v1 and v2
    reasoning: Optional[str] = Field(default=None, max_length=2000)
    reasoning_source: Optional[ReasoningSource] = None
    # v4 (2026-10-30): why a row has no reasoning when a model made it (its thinking not kept, or not linkable), said like a
    # chat turn's ``no_reasoning``; a row no model made needs none
    no_reasoning: Optional[str] = Field(default=None, max_length=300)

    _state = field_validator("state")(classmethod(lambda cls, v: _state_ok(v)))

    @model_validator(mode="before")
    @classmethod
    def _read_v1(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("schema_version") in ("dclab_core.decision_example.v1", "dclab_core.decision_example.v2",
                                                                     "dclab_core.decision_example.v3"):
            data = {**data, "schema_version": "dclab_core.decision_example.v4"}
        return data

    @model_validator(mode="after")
    def _reasoning_labelled(self):
        if (self.reasoning is None) != (self.reasoning_source is None):
            raise ValueError("a reasoning and its source come together")
        if self.reasoning is not None and self.no_reasoning:
            raise ValueError("a reasoning, or why there is none, never both")
        return self

    @model_validator(mode="after")
    def _label_says_who(self):
        if self.label is not None and self.label_source is None:
            raise ValueError("a label says who knows it (label_source)")
        return self


def _reads(old: str | tuple[str, ...], new: str) -> Any:
    """A validator that reads earlier versions' rows as this one (every field a version added has a default)."""
    olds = (old,) if isinstance(old, str) else old

    @model_validator(mode="before")
    @classmethod
    def _read(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("schema_version") in olds:
            data = {**data, "schema_version": new}
        return data
    return _read


class Message(Contract):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    reasoning: Optional[str] = None              # an assistant turn's thinking before its answer
    reasoning_source: Optional[ReasoningSource] = None
    no_reasoning: Optional[str] = Field(default=None, max_length=300)  # why an assistant turn has none (never silently absent)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: Optional[str] = Field(default=None, max_length=120)  # a tool turn: which of the assistant's calls it answers
    name: Optional[str] = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def _reasoning_labelled(self):
        if (self.reasoning is None) != (self.reasoning_source is None):
            raise ValueError("a reasoning and its source come together")
        if self.role == "assistant" and self.reasoning is None and not self.no_reasoning:
            raise ValueError("an assistant turn has its reasoning, or says why not (no_reasoning)")
        return self


class ChatExample(_Example):
    """M4, M5: messages as the model is trained, every assistant turn with its labelled reasoning, and the tools the model
    was offered (v2: ``tools``, ``Message.tool_call_id`` and the ``visible`` reasoning source; v1 reads, its tools unknown;
    v3: the ``replayed`` reasoning source; v1 and v2 read)."""

    schema_version: Literal["dclab_core.chat_example.v3"] = _version("chat_example", 3)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.chat_example.v1", "dclab_core.chat_example.v2"})
    _read_v1 = _reads(("dclab_core.chat_example.v1", "dclab_core.chat_example.v2"), "dclab_core.chat_example.v3")
    task: str = Field(max_length=120)
    stage: str = Field(max_length=60)            # M4: understand_problem, understand_data, design_solution, plan, code, report
    messages: list[Message] = Field(min_length=2)
    tools: list[dict[str, Any]] = Field(default_factory=list)  # the tool schemas offered, as sent ([]: none, or not kept)
    check: dict[str, Any] = Field(default_factory=dict)  # cited, grounded, passed, the gate


class PreferencePair(_Example):
    """M4, M5, M7: one prompt, the chosen and the rejected answers, and who decided (v2: its messages may carry a
    ``tool_call_id`` and the ``visible`` reasoning source; v1 reads; v3: the ``replayed`` reasoning source; v1 and v2 read)."""

    schema_version: Literal["dclab_core.preference_pair.v3"] = _version("preference_pair", 3)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.preference_pair.v1", "dclab_core.preference_pair.v2"})
    _read_v1 = _reads(("dclab_core.preference_pair.v1", "dclab_core.preference_pair.v2"), "dclab_core.preference_pair.v3")
    task: str = Field(max_length=120)
    prompt: list[Message] = Field(min_length=1)
    chosen: Message
    rejected: Message
    decided_by: LabelSource


class TrajectoryStep(Contract):
    n: int = Field(ge=0)
    read: Optional[str] = None                   # what the agent read before the step
    thought: Optional[str] = None
    thought_source: Optional[ReasoningSource] = None
    no_thought: Optional[str] = Field(default=None, max_length=300)  # why a step has no thought
    tool: Optional[str] = Field(default=None, max_length=120)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True

    @model_validator(mode="after")
    def _thought_labelled(self):
        if (self.thought is None) != (self.thought_source is None):
            raise ValueError("a thought and its source come together")
        if self.thought is None and not self.no_thought:
            raise ValueError("a step has its thought, or says why not (no_thought)")
        return self


class SolverTrajectory(_Example):
    """M5: a problem solved as an agent, every step, the submission and its score on the problem's own metric (v2: ``tools``,
    the schemas the agent was offered, and the ``visible`` thought source; v1 reads, its tools unknown; v3: the ``replayed``
    thought source; v1 and v2 read)."""

    schema_version: Literal["dclab_core.solver_trajectory.v3"] = _version("solver_trajectory", 3)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.solver_trajectory.v1", "dclab_core.solver_trajectory.v2"})
    _read_v1 = _reads(("dclab_core.solver_trajectory.v1", "dclab_core.solver_trajectory.v2"), "dclab_core.solver_trajectory.v3")
    problem_kind: str = Field(max_length=80)
    problem: dict[str, Any] = Field(default_factory=dict)  # its description and data card (never a row)
    tools: list[dict[str, Any]] = Field(default_factory=list)  # the tool schemas offered, as sent ([]: none, or not kept)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    submission: Optional[dict[str, Any]] = None
    metric: Optional[str] = Field(default=None, max_length=80)
    score: Optional[float] = None


class EpisodeStep(Contract):
    n: int = Field(ge=0)
    state: dict[str, Any] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list)
    action: str = Field(max_length=200)
    outcome: Optional[str] = Field(default=None, max_length=200)
    at: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, max_length=2000)  # v2: why this action, from what was known at the step, by its source
    reasoning_source: Optional[ReasoningSource] = None
    no_reasoning: Optional[str] = Field(default=None, max_length=300)  # v3 (2026-10-30): why a step has none

    _state = field_validator("state")(classmethod(lambda cls, v: _state_ok(v)))

    @model_validator(mode="after")
    def _reasoning_labelled(self):
        if (self.reasoning is None) != (self.reasoning_source is None):
            raise ValueError("a reasoning and its source come together")
        if self.reasoning is not None and self.no_reasoning:
            raise ValueError("a reasoning, or why there is none, never both")
        return self


class Episode(_Example):
    """M6: a process decided step by step, each step's (state, options, action, outcome, v2: its reasoning), and its final
    outcome."""

    schema_version: Literal["dclab_core.episode.v3"] = _version("episode", 3)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.episode.v1", "dclab_core.episode.v2"})
    _read_v1 = _reads(("dclab_core.episode.v1", "dclab_core.episode.v2"), "dclab_core.episode.v3")
    process: str = Field(max_length=80)          # the walk, a stage, the intern's session, a team's walk, a conductor run
    steps: list[EpisodeStep] = Field(min_length=1)
    gaps: list[int] = Field(default_factory=list)  # step numbers missing from the record, said
    outcome: Optional[str] = Field(default=None, max_length=200)  # from the record, never written by hand

    @model_validator(mode="after")
    def _in_order(self):
        numbers = [s.n for s in self.steps]
        if numbers != sorted(set(numbers)):
            raise ValueError("an episode's steps are in order, each once")
        missing = sorted(set(range(numbers[-1] + 1)) - set(numbers))
        if missing != sorted(self.gaps):
            raise ValueError(f"steps {missing} are missing and must be said in gaps")
        return self


class JudgedExample(_Example):
    """M7: a candidate with its score, or a pair with the preferred one, its judge and the kind of evidence (v1)."""

    schema_version: Literal["dclab_core.judged_example.v1"] = _version("judged_example")
    judged: str = Field(max_length=60)           # plan, solution, report, step, code, answer
    candidate: Any = None
    candidate_by: str = Field(max_length=120)    # who made the candidate (a judge of its own answer with no other evidence is kept out)
    other: Any = None                            # the second candidate of a pair
    score: Optional[float] = None
    preferred: Optional[Literal["candidate", "other"]] = None
    judge: str = Field(max_length=120)
    evidence: Literal["measured", "test", "verdict", "person", "critic", "reviewer", "model"]


class RetrievalExample(_Example):
    """M8: a query, the passages offered, the ones used and cited, and whether the answer passed (v1)."""

    schema_version: Literal["dclab_core.retrieval_example.v1"] = _version("retrieval_example")
    corpus: str = Field(max_length=80)
    query: str = Field(max_length=2000)
    offered: list[str] = Field(default_factory=list)
    used: list[str] = Field(default_factory=list)
    cited: list[str] = Field(default_factory=list)
    passed: Optional[bool] = None


class ExperimentRecord(_Example):
    """32.20: one model fitted on one table, for learning which approach wins on which kind of data (v1): the table's shape in
    bands and counts (never a row or a value), the task, the model and its settings, every fold's scores, the holdout's when it
    was the one chosen, the time; keyed to the table by its hash."""

    schema_version: Literal["dclab_core.experiment_record.v1"] = _version("experiment_record")
    task: str = Field(max_length=40)
    table: dict[str, Any] = Field(default_factory=dict)   # rows band, columns, column kinds' counts, positive rate band, the table's sha256
    model: str = Field(max_length=120)
    params: dict[str, Any] = Field(default_factory=dict)
    recipe: Optional[str] = Field(default=None, max_length=120)
    cv: Optional[str] = Field(default=None, max_length=300)
    folds: Optional[int] = None
    metrics: dict[str, Any] = Field(default_factory=dict)  # metric -> mean, std, min, max over the folds
    primary_metric: Optional[str] = Field(default=None, max_length=60)
    selected: bool = False
    holdout: dict[str, Any] = Field(default_factory=dict)  # the holdout's metrics, only for the model chosen
    seconds: Optional[float] = None


class CurationDecision(Versioned):
    """One curator's choice for one record (v1): which model it teaches, where it goes, its quality and why."""

    schema_version: Literal["dclab_core.curation_decision.v1"] = _version("curation_decision")
    id: str = Field(max_length=200)
    record: str = Field(max_length=200)
    model_type: ModelType
    folder: Optional[str] = Field(default=None, max_length=300)  # where it went (the point, system, family, stage, kind)
    quality: Quality
    weight: float = Field(default=1.0, ge=0, le=1)
    reason: str = Field(default="", max_length=1200)
    curator: str = Field(max_length=120)         # rule:<curator> or model:<model>
    used: bool = True                            # False: an answer in shadow, or refused by the write tool's checks
    refused: Optional[str] = Field(default=None, max_length=600)  # why the write tool refused it
    at: str


class SetManifest(Versioned):
    """One training set (v2): its model type, shape, sources with their hashes, split rule, counts, labels, consent and checks
    (v2, 2026-10-29: the model type is any of :data:`MODEL_TYPES`, ``m7_routing`` included; v1 reads)."""

    schema_version: Literal["dclab_core.set_manifest.v2"] = _version("set_manifest", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.set_manifest.v1"})
    _read_v1 = _reads("dclab_core.set_manifest.v1", "dclab_core.set_manifest.v2")
    model_type: str = Field(max_length=40)
    name: str = Field(max_length=200)            # the folder below the model type (point, system/point, stage, kind ...)
    version: str = Field(max_length=40)
    shape: str = Field(max_length=80)            # the example contract and its schema version
    sources: list[dict[str, Any]] = Field(default_factory=list)  # the records' entries with their hashes
    split_rule: str = Field(max_length=200)
    counts: dict[str, int] = Field(default_factory=dict)        # per split
    label_sources: dict[str, int] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    consent: Consent = "train"
    decontaminated: list[str] = Field(default_factory=list)      # the sealed sets and benchmarks it was checked against
    quarantined: int = 0
    commit: Optional[str] = Field(default=None, max_length=64)
    built_at: str

    @field_validator("model_type")
    @classmethod
    def _model_type(cls, value: str) -> str:
        if value not in MODEL_TYPES:
            raise ValueError(f"{value!r} is not a model type ({', '.join(MODEL_TYPES)})")
        return value


class WalkStep(Contract):
    at: str
    kind: Literal["draft_event", "move", "refused", "stage", "gate", "decision", "agent_run"]
    ref: str = Field(max_length=200)             # the event's seq, the move, the stage, the record's id, the run's id
    detail: dict[str, Any] = Field(default_factory=dict)  # names, codes, statuses and counts; never a row or a person's words


class WorkflowRecord(Versioned):
    """One walk from a draft to a report as a whole (v1, 32.5): every step in time order, the Workflow Run's twelve steps and its
    gates, and the ids of the decisions and agent runs inside it, so each links back to its own record."""

    schema_version: Literal["dclab_core.workflow_record.v1"] = _version("workflow_record")
    id: str = Field(max_length=200)
    draft: Optional[str] = Field(default=None, max_length=120)
    project: Optional[str] = Field(default=None, max_length=120)
    started: Optional[str] = None
    ended: Optional[str] = None
    steps: list[WalkStep] = Field(default_factory=list)
    workflow_run: Optional[dict[str, Any]] = None
    gates: dict[str, Any] = Field(default_factory=dict)
    decisions: list[str] = Field(default_factory=list)
    agent_runs: list[str] = Field(default_factory=list)
    outcome: Literal["reported", "built", "stopped_in_draft", "deleted"] = "stopped_in_draft"

    @model_validator(mode="after")
    def _in_time(self):
        times = [s.at for s in self.steps]
        if times != sorted(times):
            raise ValueError("a walk's steps are in time order")
        return self


for _name, _cls in (("experiment_record", ExperimentRecord), ("workflow_record", WorkflowRecord), ("store_entry", StoreEntry), ("store_record", StoreRecord), ("decision_example", DecisionExample),
                    ("chat_example", ChatExample), ("preference_pair", PreferencePair), ("solver_trajectory", SolverTrajectory),
                    ("episode", Episode), ("judged_example", JudgedExample), ("retrieval_example", RetrievalExample),
                    ("curation_decision", CurationDecision), ("set_manifest", SetManifest)):
    CONTRACTS[_name] = _cls

EXAMPLE_SHAPES: dict[str, tuple[str, ...]] = {  # model type -> the example contracts its sets hold
    "m1_students": ("decision_example",), "m2_workflow_decisions": ("decision_example",), "m3_ml_decisions": ("decision_example", "experiment_record"),
    "m4_workflow_llm": ("chat_example", "preference_pair"), "m5_solver_llm": ("solver_trajectory", "chat_example", "preference_pair", "experiment_record"),
    "m6_stepwise_decisions": ("episode",), "m7_judges": ("judged_example", "preference_pair"), "m8_retrieval": ("retrieval_example",),
    "m7_routing": ("routing_example",),
}

__all__ = ["ModelType", "MODEL_TYPES", "Kind", "KINDS", "Split", "LabelSource", "ReasoningSource", "REASONING_SOURCES", "Quality",
           "StoreEntry", "Provenance", "StoreRecord", "Answer", "DecisionExample", "Message", "ChatExample", "PreferencePair",
           "TrajectoryStep", "SolverTrajectory", "EpisodeStep", "Episode", "JudgedExample", "RetrievalExample", "CurationDecision",
           "SetManifest", "EXAMPLE_SHAPES", "WalkStep", "WorkflowRecord", "ExperimentRecord"]

import dclab_core.store_rules  # noqa: E402,F401  (the capture rules register their contract; imported here so every reader sees it)
import dclab_core.routing_example  # noqa: E402,F401  (m7_routing's example contract, registered with the others)
