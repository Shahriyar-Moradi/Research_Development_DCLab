"""Team work: the task, its actions, the report that comes back, the decision (package W1, 2026-11-20).

The owner's collaborative workspace (``Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md``): each team talks to
its own agent, and the agents carry questions and answers between teams. This module holds what travels and the rules
that keep it honest. Pure pydantic and Python: the host stores the records, delivers them and runs the agents.

- :class:`TeamTask`: a question with one accountable owner and a definition of done. A clarification keeps it open; a
  delegation creates a child task (:func:`delegate`).
- :class:`TeamAction`: one recorded attempt to move a task forward, bound to a tool or to a recorded human response.
  A planned action, a denial, a timeout and a success are different states; only a terminal one can be claimed.
- :class:`WorkReport`: the answer and what actually happened, built only from the action log and the child reports
  (:func:`build_report`). It never lists an unfinished action as done, never fills a missing child result, never
  carries a private note, and says when anything in it was simulated.
- :class:`TeamDecision`: the decision owner's choice, with the evidence and constraint revisions it rested on. Only that
  person approves it (:func:`may_approve`), and every write of it passes :func:`decision_write` (0.169.0): once approved,
  its option and citations never change; a correction is a new decision naming the one it replaces.
- :class:`AgentConfig`: an agent's job description, read by code and shown to people: who it represents, what it owns,
  the actions it may take (each bound to ``tool:<name>`` or ``human:<who>``), what its report must cover, what it may
  read, what it may never do. :data:`DIRECTORY` has one per row of the owner's action directory.
- :class:`TeamWork`: one problem's work in one document (tasks, actions, reports, decisions), for export and replay.

Contract ``dclab_core.team_work.v3`` (:data:`CONTRACT`; v3 carries ``team_task.v2``, whose ``target`` is the outcome a
person confirmed (``dclab_core.target_contract``, 0.170.0); v2 carried ``team_decision.v2``; it reads v1 and v2), with each record
also versioned on its own so a host can store it as a row. Names, references and short words only: never a table's value.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, ClassVar, Iterable, Literal, Mapping, Optional, Sequence, Union

from pydantic import Field, field_validator, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.target_contract import TargetContract

CONTRACT = "dclab_core.team_work.v3"

ID = r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,119}$"
NAME = r"^[a-z][a-z0-9_]*(:[A-Za-z0-9_.-]+)?$"  # a team, agent or action name, with an optional qualifier (model:eta)
SOURCE = r"^(tool|human):[A-Za-z0-9_.:@-]+$"
TEXT = 2000
SHORT = 300

TaskStatus = Literal["open", "accepted", "waiting_for_person", "waiting_for_team", "partial", "complete", "blocked",
                     "failed", "cancelled", "superseded"]
ActionStatus = Literal["planned", "running", "denied", "timed_out", "succeeded", "failed", "skipped", "simulated"]
ReportStatus = Literal["complete", "partial", "blocked", "failed", "superseded"]
ActorKind = Literal["agent", "team", "person"]
AgentKind = Literal["team_agent", "specialist", "coordinator", "entry"]  # 0.167.0: the Lead coordinates; Home is the way in
MemoryScope = Literal["person", "team", "workspace", "project", "problem", "model_version"]

TERMINAL_ACTIONS = frozenset({"denied", "timed_out", "succeeded", "failed", "skipped", "simulated"})
DONE_ACTIONS = frozenset({"succeeded", "simulated"})
FAILED_ACTIONS = frozenset({"denied", "timed_out", "failed"})
CLOSED_TASKS = frozenset({"complete", "failed", "cancelled", "superseded"})
RETURNED_REPORTS = frozenset({"complete", "blocked", "failed"})  # a child that sent one of these has answered

# Which status may follow which. A clarification (waiting_for_person) keeps the task open; ``complete`` and ``failed``
# move only to ``superseded`` (a correction makes a new task or report, never reopens the old one).
MOVES: dict[str, frozenset[str]] = {
    "open": frozenset({"accepted", "waiting_for_person", "blocked", "cancelled", "superseded"}),
    "accepted": frozenset({"waiting_for_person", "waiting_for_team", "partial", "complete", "blocked", "failed",
                           "cancelled", "superseded"}),
    "waiting_for_person": frozenset({"accepted", "blocked", "failed", "cancelled", "superseded"}),
    "waiting_for_team": frozenset({"accepted", "waiting_for_person", "partial", "complete", "blocked", "failed",
                                   "cancelled", "superseded"}),
    "partial": frozenset({"accepted", "waiting_for_person", "waiting_for_team", "complete", "blocked", "failed",
                          "cancelled", "superseded"}),
    "blocked": frozenset({"accepted", "failed", "cancelled", "superseded"}),
    "complete": frozenset({"superseded"}),
    "failed": frozenset({"superseded"}),
    "cancelled": frozenset(),
    "superseded": frozenset(),
}


def _utc(value: Optional[str]) -> Optional[str]:
    """An ISO time with its zone, as UTC (so a later time always sorts later)."""
    if value is None:
        return None
    moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("a time names its zone")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def report_id(task_id: str, revision: int) -> str:
    """A report's id: its task and revision (``T-7@r2``), so a reply is deduplicated by exactly those two."""
    return f"{task_id}@r{revision}"


def retry_key(task_id: str, action: str, input_revision: str) -> str:
    """The key that makes a retried action the same attempt: the task, the directory action and its input revision."""
    return f"{task_id}:{action}:{input_revision}"


# ------------------------------------------------------------------------------------------------------ who


class Actor(Contract):
    """Who did something: an agent (by its configuration's name), a team, or a person (by their id)."""

    kind: ActorKind
    id: str = Field(min_length=1, max_length=120)
    team: Optional[str] = Field(default=None, pattern=NAME, description="The team the actor belongs to or represents.")


class Requester(Contract):
    """Who asked: the team, the member who started it, and the conversation the answer goes back to."""

    team: str = Field(pattern=NAME)
    member: Optional[str] = Field(default=None, max_length=120)
    thread_id: str = Field(pattern=ID)


# ------------------------------------------------------------------------------------------------------ task


class TeamTask(Versioned):
    """One assignment: a question, one accountable owner, what counts as done, what it may use.

    v2 (0.170.0) adds ``target``: the outcome a person confirmed for this task, as a contract (the event as a code, with
    their words), so the owner checks the data against what was asked and the next team reads it as a field, never out
    of a sentence. A v1 task reads as one with no confirmed target."""

    schema_version: Literal["dclab_core.team_task.v2"] = _version("team_task", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.team_task.v1"})  # the same fields, without target
    id: str = Field(pattern=ID)
    workspace_id: str = Field(pattern=ID)
    project_id: str = Field(pattern=ID)
    problem_id: str = Field(pattern=ID)
    brief_revision: int = Field(ge=1, description="The problem brief's revision the task was asked on.")
    parent_task_id: Optional[str] = Field(default=None, pattern=ID)
    requester: Requester
    owner_team: str = Field(pattern=NAME, description="The one accountable owner: a team or a specialist.")
    question: str = Field(min_length=1, max_length=TEXT)
    expected_output: str = Field(min_length=1, max_length=SHORT)
    done_criteria: list[str] = Field(min_length=1, max_length=20)
    allowed_inputs: list[str] = Field(default_factory=list, max_length=50)
    allowed_tools: list[str] = Field(default_factory=list, max_length=50)
    budget_eur: float = Field(default=0.0, ge=0)
    status: TaskStatus = "open"
    superseded_by: Optional[str] = Field(default=None, pattern=ID)
    created_at: str
    target: Optional[TargetContract] = Field(default=None, description="v2: the outcome a person confirmed for this task.")

    @model_validator(mode="before")
    @classmethod
    def _v1(cls, data: Any) -> Any:
        """A v1 task (alone, or inside a stored bundle or row) is the same task with no confirmed target."""
        if isinstance(data, Mapping) and data.get("schema_version") in cls.READS:
            return {**data, "schema_version": cls.model_fields["schema_version"].default}
        return data

    @field_validator("created_at")
    @classmethod
    def _time(cls, value: str) -> str:
        return _utc(value)

    @model_validator(mode="after")
    def _rules(self):
        if self.target is not None and self.target.brief_revision > self.brief_revision:
            raise ValueError("a confirmed target is never on a later brief revision than its task")
        if self.parent_task_id == self.id:
            raise ValueError("a task cannot be its own parent")
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("a superseded task names its replacement, and only a superseded task does")
        if self.superseded_by == self.id:
            raise ValueError("a task cannot replace itself")
        return self


# ------------------------------------------------------------------------------------------------------ action


class TeamAction(Versioned):
    """One attempt: who, which directory action, from which tool or human response, on which input, and how it ended."""

    schema_version: Literal["dclab_core.team_action.v1"] = _version("team_action")
    task_id: str = Field(pattern=ID)
    action_id: str = Field(pattern=ID)
    actor: Actor
    action: str = Field(pattern=NAME, description="A name from the actor's action directory.")
    source: str = Field(pattern=SOURCE, description="tool:<name>, or human:<response id> for a recorded human answer.")
    input_revision: str = Field(min_length=1, max_length=120)
    status: ActionStatus = "planned"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    output_ref: Optional[str] = Field(default=None, max_length=SHORT)
    failure_ref: Optional[str] = Field(default=None, max_length=SHORT)
    reason: Optional[str] = Field(default=None, max_length=SHORT, description="Why it was skipped or failed, in words.")
    retry_key: str = Field(min_length=1, max_length=SHORT)
    responded_by: Optional[Actor] = Field(default=None, description="The person whose recorded answer a human-bound "
                                          "action is. The host (W5) checks the person holds the bound role.")

    @field_validator("started_at", "ended_at")
    @classmethod
    def _times(cls, value: Optional[str]) -> Optional[str]:
        return _utc(value)

    @model_validator(mode="after")
    def _rules(self):
        if self.status == "planned" and (self.started_at or self.ended_at or self.output_ref or self.failure_ref):
            raise ValueError("a planned action has not started: no time, output or failure")
        if self.status == "running" and (self.ended_at or self.output_ref or self.failure_ref):
            raise ValueError("a running action has not ended: no end, output or failure")
        if self.status in TERMINAL_ACTIONS and not self.ended_at:
            raise ValueError("an ended action records when it ended")
        if self.status in DONE_ACTIONS and (not self.output_ref or self.failure_ref):
            raise ValueError("a succeeded or simulated action names its output and no failure")
        if self.status in FAILED_ACTIONS and (not self.failure_ref or self.output_ref):
            raise ValueError("a denied, timed-out or failed action names its failure and no output")
        if self.status == "skipped" and not self.reason:
            raise ValueError("a skipped action says why")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("an action cannot end before it starts")
        if self.responded_by is not None and (self.responded_by.kind != "person" or not self.source.startswith("human:")):
            raise ValueError("responded_by is a person, on an action sourced from a human response only")
        if self.source.startswith("human:") and self.status in DONE_ACTIONS and self.responded_by is None:
            raise ValueError("a completed human-sourced action names the person who responded")
        return self


# ------------------------------------------------------------------------------------------------------ report


class Assignment(Contract):
    """The question the report answers and what counted as done."""

    question: str = Field(min_length=1, max_length=TEXT)
    done_criteria: list[str] = Field(min_length=1, max_length=20)


class ActionLine(Contract):
    """One terminal action as the report shows it."""

    action_id: str = Field(pattern=ID)
    action: str = Field(pattern=NAME)
    actor: str = Field(min_length=1, max_length=120)
    status: ActionStatus
    ref: Optional[str] = Field(default=None, max_length=SHORT, description="Its output, or its failure.")
    reason: Optional[str] = Field(default=None, max_length=SHORT)


class ReportActions(Contract):
    """Completed actions, failed attempts and skipped work, each a terminal action record."""

    completed: list[ActionLine] = Field(default_factory=list, max_length=200)
    failed: list[ActionLine] = Field(default_factory=list, max_length=200)
    skipped: list[ActionLine] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _buckets(self):
        if any(line.status not in DONE_ACTIONS for line in self.completed):
            raise ValueError("only a succeeded or simulated action is listed as completed")
        if any(line.status not in FAILED_ACTIONS for line in self.failed):
            raise ValueError("only a denied, timed-out or failed action is listed as a failed attempt")
        if any(line.status != "skipped" or not line.reason for line in self.skipped):
            raise ValueError("only a skipped action, with its reason, is listed as skipped")
        return self


class EvidenceRef(Contract):
    """A reference the reader may open: where it came from, the run or model, the input revision, its limits."""

    source: str = Field(min_length=1, max_length=SHORT)
    ref: str = Field(min_length=1, max_length=SHORT, description="A run, model version, report or fixture reference.")
    input_revision: str = Field(min_length=1, max_length=120)
    uncertainty: Optional[str] = Field(default=None, max_length=SHORT)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    private: bool = Field(default=False, description="A team's own note: dropped by build_report, refused in a report.")


class NextStep(Contract):
    """Who owes the next response and what unblocks it, or that the task is closed."""

    owner: Optional[str] = Field(default=None, min_length=1, max_length=120)
    unblocks: Optional[str] = Field(default=None, min_length=1, max_length=SHORT)
    closed: bool = False

    @model_validator(mode="after")
    def _one(self):
        if self.closed and (self.owner or self.unblocks):
            raise ValueError("a closed task has no next owner")
        if not self.closed and not (self.owner and self.unblocks):
            raise ValueError("an open next step names its owner and what unblocks it")
        return self


class WorkReport(Versioned):
    """The answer and what actually happened, at one revision."""

    schema_version: Literal["dclab_core.work_report.v1"] = _version("work_report")
    task_id: str = Field(pattern=ID)
    revision: int = Field(ge=1)
    parent_task_id: Optional[str] = Field(default=None, pattern=ID)
    brief_revision: int = Field(ge=1, description="The problem brief's revision this report answers.")
    answer: str = Field(min_length=1, max_length=TEXT)
    status: ReportStatus
    assignment: Assignment
    owner: str = Field(pattern=NAME, description="Who performed the work: the task's owner.")
    actions: ReportActions = Field(default_factory=ReportActions)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=100)
    limits: list[str] = Field(default_factory=list, max_length=50)
    next_task: NextStep
    outstanding: list[str] = Field(default_factory=list, max_length=50, description="Child tasks not yet returned.")
    supersedes: Optional[str] = Field(default=None, pattern=ID)
    superseded_by: Optional[str] = Field(default=None, pattern=ID)
    simulated: bool = False
    child_reports: list[str] = Field(default_factory=list, max_length=50)
    created_at: Optional[str] = None

    @field_validator("created_at")
    @classmethod
    def _time(cls, value: Optional[str]) -> Optional[str]:
        return _utc(value)

    @property
    def id(self) -> str:
        return report_id(self.task_id, self.revision)

    @model_validator(mode="after")
    def _rules(self):
        if any(e.private for e in self.evidence):
            raise ValueError("a report is shared: it never carries a private note")
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("a superseded report names its replacement, and only a superseded report does")
        if self.status == "complete" and self.outstanding:
            raise ValueError("a report with an outstanding child is partial, not complete")
        if self.status in ("partial", "blocked", "failed") and self.next_task.closed:
            raise ValueError("a partial, blocked or failed report names who acts next and what unblocks it")
        if self.supersedes == self.id or self.superseded_by == self.id:
            raise ValueError("a report cannot replace itself")
        if len(set(self.child_reports)) != len(self.child_reports):
            raise ValueError("a child report is joined once")
        if any(line.status == "simulated" for line in self.actions.completed) and not self.simulated:
            raise ValueError("a report with a simulated action is marked simulated")
        if self.status == "complete" and not self.actions.completed and not self.child_reports:
            raise ValueError("a complete report needs a completed action or a returned child")
        return self


# ------------------------------------------------------------------------------------------------------ decision


class Approval(Contract):
    who: Actor = Field(description="The person who approved: always the decision owner.")
    when: str

    @field_validator("when")
    @classmethod
    def _time(cls, value: str) -> str:
        return _utc(value)


class TeamDecision(Versioned):
    """The decision owner's choice, the evidence and constraints it rested on, and what happened after.

    v2 (0.169.0) adds what a host needs to keep one honest record of it: ``revision`` (each accepted write is the next
    one, so a person who read an older one is told), ``request_id`` (the same request sent twice is one write),
    ``supersedes`` (the earlier decision this one replaces: an approved decision is never edited), who measured the
    outcome, when and on which measure, and the policy it was approved under. A v1 decision reads as revision 1."""

    schema_version: Literal["dclab_core.team_decision.v2"] = _version("team_decision", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.team_decision.v1"})  # the same fields, without the v2 ones
    id: str = Field(pattern=ID)
    problem_id: str = Field(pattern=ID)
    decision_owner: Actor = Field(description="The person who decides.")
    option: str = Field(min_length=1, max_length=TEXT)
    evidence_revisions: list[str] = Field(default_factory=list, max_length=50, description="Report ids it rested on.")
    constraint_revisions: list[str] = Field(default_factory=list, max_length=50)
    approval: Optional[Approval] = None
    follow_up_date: Optional[str] = None
    outcome: Optional[str] = Field(default=None, max_length=TEXT)
    simulated_follow_up: bool = False
    revision: int = Field(default=1, ge=1, description="v2: which accepted write of this decision this is, from 1.")
    request_id: Optional[str] = Field(default=None, pattern=ID, description="v2: the request that made this write; the "
                                      "same request again is the same write.")
    supersedes: Optional[str] = Field(default=None, pattern=ID, description="v2: the earlier decision this one replaces.")
    outcome_at: Optional[str] = Field(default=None, description="v2: when the outcome was observed.")
    outcome_by: Optional[Actor] = Field(default=None, description="v2: who recorded the outcome.")
    outcome_measure: Optional[str] = Field(default=None, pattern=ID, description="v2: the measure the outcome was read on, "
                                           "by its name (renewed_within_30_days): a name, never a sentence.")
    policy_ref: Optional[str] = Field(default=None, pattern=ID, description="v2: the policy the decision was approved "
                                      "under, by reference (policy:renewal-outreach@v2): a reference, never its words.")

    @model_validator(mode="before")
    @classmethod
    def _v1(cls, data: Any) -> Any:
        """A v1 decision (alone, or inside a stored bundle or ledger row) is the same decision at revision 1."""
        if isinstance(data, Mapping) and data.get("schema_version") in cls.READS:
            return {**data, "schema_version": cls.model_fields["schema_version"].default}
        return data

    @field_validator("follow_up_date")
    @classmethod
    def _date(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("outcome_at")
    @classmethod
    def _when(cls, value: Optional[str]) -> Optional[str]:
        return _utc(value)

    @model_validator(mode="after")
    def _rules(self):
        if self.decision_owner.kind != "person":
            raise ValueError("a decision owner is a person, never an agent or a team")
        if self.approval is not None and (self.approval.who.kind != "person" or self.approval.who.id != self.decision_owner.id):
            raise ValueError("only the decision owner approves the decision")
        if self.outcome is not None and self.approval is None:
            raise ValueError("an outcome follows an approved decision")
        if self.simulated_follow_up and self.outcome is None:
            raise ValueError("a simulated follow-up records its (simulated) outcome")
        if self.outcome is None and (self.outcome_at or self.outcome_by or self.outcome_measure):
            raise ValueError("when, by whom and on which measure belong to a recorded outcome")
        if self.supersedes == self.id:
            raise ValueError("a decision cannot replace itself")
        return self


class DecisionVerdict(Contract):
    """Whether a write of a decision is kept, what it is, and in plain words why not (ids and states, never the option)."""

    ok: bool
    kind: Literal["recorded", "approved", "outcome", "unchanged",                                   # kept
                  "other_decision", "owner_changed", "agent", "not_owner", "no_permission",        # refused: who
                  "immutable", "no_approval", "not_actor"]                                         # refused: what
    reason: str = Field(min_length=1, max_length=SHORT)
    revision: int = Field(ge=1, description="The revision the write gets when kept; the stored one otherwise.")


_WRITE_STAMPS = frozenset({"revision", "request_id"})   # set by the host on each accepted write: never a change by themselves
_FROZEN = ("option", "evidence_revisions", "constraint_revisions", "approval", "policy_ref")  # what an approval covers
_OUTCOME = ("outcome", "outcome_at", "outcome_by", "outcome_measure", "simulated_follow_up")


def decision_write(stored: Optional[TeamDecision], new: TeamDecision, actor: Actor, may_approve_role: bool) -> DecisionVerdict:
    """Whether ``actor`` may replace the stored decision (None: not recorded yet) with ``new``.

    ``may_approve_role``: whether the host's role check lets this actor approve a team's decision (the permission; that
    the actor is who they say is the host's check too). Why each rule:

    - A record keeps its decision: the same id and problem, and the same decision owner for good (another person's
      record never takes a decision over).
    - An approval is the decision owner's own act: the actor is that person (:func:`may_approve`: never an agent or a
      team, however good its recommendation), and holds the permission. A person who may write but does not own the
      decision cannot write the owner's approval down for them.
    - What was approved stays as it was approved: the option, the evidence and constraint revisions, the policy and
      the approval itself. A change is a new decision that names this one in ``supersedes``, so whoever acted on the
      first can still read what they acted on. What it replaces is said when a decision is first recorded.
    - An outcome follows an approval already recorded (never the same write: what happened comes after the choice), and
      names the actor as the one who recorded it (a person, or an agent logging a follow-up under its own name). Once
      kept it is never rewritten or erased in place: what was observed stays what was recorded.

    Until it is approved, a proposal may be recorded and changed by anyone the host lets write. The same content again is
    ``unchanged`` (nothing to write). ``revision`` and ``request_id`` are the host's stamps and are not compared."""
    current = stored.revision if stored is not None else 0
    following = current + 1

    def no(kind: str, reason: str) -> DecisionVerdict:
        return DecisionVerdict(ok=False, kind=kind, reason=reason[:SHORT], revision=max(current, 1))

    if stored is not None:
        if (stored.id, stored.problem_id) != (new.id, new.problem_id):
            return no("other_decision", f"{new.id} is not the stored decision {stored.id} of problem {stored.problem_id}")
        if stored.decision_owner.id != new.decision_owner.id:
            return no("owner_changed", f"{stored.id} is {stored.decision_owner.id}'s decision: its owner never changes")
        if stored.model_dump(exclude=_WRITE_STAMPS) == new.model_dump(exclude=_WRITE_STAMPS):
            return DecisionVerdict(ok=True, kind="unchanged", reason=f"{new.id} is already recorded so", revision=current)
        if stored.supersedes != new.supersedes:
            return no("immutable", f"what {stored.id} supersedes is said once, when it is first recorded")
    approved = stored is not None and stored.approval is not None
    if approved:
        changed = [f for f in _FROZEN if getattr(stored, f) != getattr(new, f)]
        if changed:
            return no("immutable", f"{stored.id} is approved: its {', '.join(changed)} cannot change; record a new "
                                   f"decision that supersedes {stored.id}")
    approving = new.approval is not None and not approved
    if approving:
        if actor.kind != "person":
            return no("agent", f"a {actor.kind} never approves for the decision owner ({new.decision_owner.id})")
        if not may_approve(actor, new):
            return no("not_owner", f"only {new.decision_owner.id}, the decision owner, approves {new.id}; {actor.id} does not")
        if not may_approve_role:
            return no("no_permission", f"{actor.id} owns {new.id} but does not hold the permission to approve a team's decision")
    outcome_changed = any(getattr(new, f) != (getattr(stored, f) if stored is not None else TeamDecision.model_fields[f].default)
                          for f in _OUTCOME)
    if outcome_changed:
        if not approved:
            return no("no_approval", f"an outcome of {new.id} is recorded after its approval, in a write of its own")
        if stored.outcome is not None:
            return no("immutable", f"the outcome of {stored.id} is recorded: it is not rewritten or erased; record a new "
                                   f"decision that supersedes {stored.id}")
        if new.outcome_by is None or (new.outcome_by.kind, new.outcome_by.id) != (actor.kind, actor.id):
            named = new.outcome_by.id if new.outcome_by is not None else "nobody"
            return no("not_actor", f"the outcome of {new.id} names {named} as its recorder, but {actor.id} records it")
    if approving:
        return DecisionVerdict(ok=True, kind="approved", reason=f"{actor.id} approves {new.id}", revision=following)
    if outcome_changed:
        return DecisionVerdict(ok=True, kind="outcome", reason=f"the outcome of {new.id} is recorded", revision=following)
    return DecisionVerdict(ok=True, kind="recorded", reason=f"{new.id} is recorded", revision=following)


# ------------------------------------------------------------------------------------------------------ agents


class DirectoryAction(Contract):
    """One action an agent may take, bound to the tool that does it or the person whose recorded answer it is."""

    action: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    bound_to: str = Field(pattern=r"^(tool|human):[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=SHORT)


class AgentConfig(Versioned):
    """An agent's job description: who it represents, what it owns, what it may do, report, read, and never do."""

    schema_version: Literal["dclab_core.agent_config.v2"] = _version("agent_config", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.agent_config.v1"})  # v2 adds the coordinator and entry kinds
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: AgentKind
    identity: str = Field(min_length=1, max_length=SHORT, description="The team (or role) it represents.")
    responsibility: str = Field(min_length=1, max_length=SHORT, description="The task kinds it owns, in words.")
    action_directory: list[DirectoryAction] = Field(min_length=1, max_length=50)
    required_report: list[str] = Field(min_length=1, max_length=20)
    memory_scope: list[MemoryScope] = Field(min_length=1, max_length=6)
    forbidden: list[str] = Field(default_factory=list, max_length=20, description="Permissions it never has.")

    @model_validator(mode="after")
    def _rules(self):
        names = [a.action for a in self.action_directory]
        if len(set(names)) != len(names):
            raise ValueError("an action is listed once in a directory")
        if "approve" not in self.forbidden:
            raise ValueError("no agent approves on a person's behalf: every configuration forbids approve")
        if set(names) & set(self.forbidden):
            raise ValueError("an action in the directory cannot also be forbidden")
        return self

    @classmethod
    def read(cls, document: Any):
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        if data.get("schema_version") in cls.READS:  # a v1 configuration: the same fields, a narrower set of kinds
            data = {**data, "schema_version": cls.model_fields["schema_version"].default}
        return super().read(data)

    def binding(self, action: str) -> Optional[str]:
        return next((a.bound_to for a in self.action_directory if a.action == action), None)


# ------------------------------------------------------------------------------------------------------ bundle


class TeamWork(Versioned):
    """One problem's team work in one document: its tasks, actions, reports and decisions. v2 (0.169.0): its decisions are
    ``team_decision.v2``. v3 (0.170.0): its tasks are ``team_task.v2`` (a confirmed target). A v1 or v2 document reads
    unchanged, its decisions at revision 1 and its tasks with no confirmed target."""

    schema_version: Literal["dclab_core.team_work.v3"] = _version("team_work", 3)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.team_work.v1", "dclab_core.team_work.v2"})
    problem_id: str = Field(pattern=ID)
    tasks: list[TeamTask] = Field(default_factory=list, max_length=2000)
    actions: list[TeamAction] = Field(default_factory=list, max_length=20000)
    reports: list[WorkReport] = Field(default_factory=list, max_length=5000)
    decisions: list[TeamDecision] = Field(default_factory=list, max_length=500)

    @classmethod
    def read(cls, document: Any):
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        if data.get("schema_version") in cls.READS:  # an older document: the same fields; its decisions and tasks upgrade themselves
            data = {**data, "schema_version": cls.model_fields["schema_version"].default}
        return super().read(data)

    @model_validator(mode="after")
    def _one_problem(self):
        if any(t.problem_id != self.problem_id for t in self.tasks):
            raise ValueError("every task belongs to the document's problem")
        if any(d.problem_id != self.problem_id for d in self.decisions):
            raise ValueError("every decision belongs to the document's problem")
        return self


for _model in (TeamTask, TeamAction, WorkReport, TeamDecision, AgentConfig, TeamWork):
    CONTRACTS[_model.model_fields["schema_version"].default.split(".")[1]] = _model


# ------------------------------------------------------------------------------------------------------ the directory


def _cfg(name: str, kind: str, identity: str, responsibility: str, actions: Sequence[tuple[str, str, str]],
         required: Sequence[str], scope: Sequence[str], forbidden: Sequence[str]) -> AgentConfig:
    return AgentConfig(name=name, kind=kind, identity=identity, responsibility=responsibility,
                       action_directory=[DirectoryAction(action=a, bound_to=b, description=d) for a, b, d in actions],
                       required_report=list(required), memory_scope=list(scope), forbidden=list(forbidden))


_NEVER = ("approve", "train", "deploy", "contact_customers")

# One configuration per row of the owner's action directory (2026-11-20). Every action is bound to a tool or to a
# person's recorded answer; recording the owner's decision writes down what the person decided, it never decides.
DIRECTORY: dict[str, AgentConfig] = {c.name: c for c in (
    _cfg("business", "team_agent", "Business team",
         "Turn a business concern into a decision and own the return conversation",
         [("open_problem", "tool:open_problem", "Open a problem in the project."),
          ("confirm_terms", "human:decision_owner", "Confirm the decision's terms with its owner."),
          ("delegate_analysis", "tool:delegate", "Ask Data Science what the data can answer."),
          ("request_sales_input", "tool:delegate", "Ask Sales for capacity and exclusions."),
          ("request_marketing_input", "tool:delegate", "Ask Marketing for offer terms."),
          ("reconcile_replies", "tool:join_replies", "Join the teams' replies into one brief."),
          ("present_options", "tool:present_options", "Show the owner the feasible options and their evidence."),
          ("record_decision", "human:decision_owner", "Write down the decision the owner made.")],
         ["question_answered", "contributing_teams", "evidence", "unresolved_constraints", "next_owner"],
         ["team", "project", "problem"], _NEVER),
    _cfg("data_science", "team_agent", "Data Science team", "Decide what the data can answer",
         [("inspect_schema", "tool:inspect_schema", "Read the permitted tables' columns and source dates."),
          ("confirm_prediction_moment", "human:requester", "Confirm when the prediction is made."),
          ("confirm_target", "human:requester", "Confirm what the outcome means."),
          ("check_leakage", "tool:leakage_check", "Find columns known only after the prediction moment."),
          ("define_baseline", "tool:define_baseline", "Name the simple rule a model must beat."),
          ("delegate_specialist", "tool:delegate", "Open a child task for a specialist or ML Engineering."),
          ("review_evidence", "tool:review_evidence", "Check a returned result against its question."),
          ("explain_uncertainty", "tool:explain_uncertainty", "Say how sure the finding is and why.")],
         ["feasibility_or_finding", "checks_performed", "exclusions", "limits", "remaining_inputs"],
         ["team", "project", "problem"], _NEVER),
    _cfg("ml_engineering", "team_agent", "ML Engineering team", "Run an agreed experiment within its budget",
         [("validate_contract", "tool:validate_contract", "Check the experiment's contract before running."),
          ("run_training", "tool:run_stage", "Run the permitted training stage."),
          ("run_evaluation", "tool:run_stage", "Run the permitted evaluation stage."),
          ("compare_baseline", "tool:compare_baseline", "Compare the candidate with the agreed baseline."),
          ("register_version", "tool:register_version", "Register the evaluated model version."),
          ("return_run_evidence", "tool:lineage", "Return the run and version references.")],
         ["model_version", "run_ref", "baseline_comparison", "data_boundary", "failed_or_skipped_checks",
          "intended_use"],
         ["team", "project", "problem", "model_version"], ("approve", "deploy", "contact_customers")),
    _cfg("sales", "team_agent", "Sales team", "Produce an outreach plan within confirmed capacity",
         [("ask_capacity", "human:sales_team", "Ask a member of the Sales team how many contacts it can make (0 to 30)."),
          ("check_exclusions", "tool:check_exclusions", "Remove accounts that must not be contacted."),
          ("draft_contact_list", "tool:draft_list", "Draft the list within capacity."),
          ("return_restricted_ref", "tool:share_reference", "Return a restricted reference to the draft."),
          ("revise_on_correction", "tool:draft_list", "Redraft when capacity is corrected.")],
         ["capacity_source", "eligible_count_or_blocker", "draft_revision", "next_approver"],
         ["team", "project", "problem"], _NEVER),
    _cfg("marketing", "team_agent", "Marketing team", "Determine whether a proposed offer can be used",
         [("check_eligibility", "tool:offer_rules", "Check who the offer applies to."),
          ("check_campaign_dates", "tool:campaign_calendar", "Check the campaign's dates."),
          ("check_cost", "tool:offer_cost", "Check what the offer costs."),
          ("confirm_terms", "human:marketing_team", "Ask a member of the Marketing team for the offer's dated terms and cost."),
          ("return_conditions", "tool:share_reference", "Return the offer's conditions.")],
         ["applicable_offer_or_reason", "source", "constraints", "next_reviewer"],
         ["team", "project", "problem"], _NEVER),
    _cfg("project", "specialist", "The project", "Keep the shared project coordinated",
         [("resolve_membership", "tool:membership", "Read who is in the project and its teams."),
          ("list_open_tasks", "tool:list_tasks", "List the problems' open tasks."),
          ("identify_owners", "tool:list_tasks", "Name who owes each next response."),
          ("refresh_summary", "tool:project_summary", "Rebuild the summary from authorized records.")],
         ["project_summary", "outstanding_dependencies", "task_refs"], ["workspace", "project"], _NEVER),
    _cfg("problem", "specialist", "The problem", "Keep the problem definition consistent across handoffs",
         [("retrieve_brief", "tool:problem_brief", "Read the latest brief."),
          ("record_correction", "human:fact_owner", "Record a correction its owner confirmed."),
          ("identify_affected_tasks", "tool:list_tasks", "Find tasks that rested on the corrected fact."),
          ("flag_stale", "tool:mark_stale", "Mark conclusions that rested on it as stale.")],
         ["revised_brief", "what_changed", "tasks_to_reopen", "unaffected_evidence"], ["project", "problem"], _NEVER),
    _cfg("model", "specialist", "One model version", "Answer a bounded question for one model version",
         [("verify_cutoff", "tool:model_contract", "Check the inputs exist at the model's cutoff."),
          ("check_input_contract", "tool:model_contract", "Check the inputs match the version's contract."),
          ("score", "tool:model_score", "Score with an existing version."),
          ("retrieve_evaluation", "tool:model_evaluation", "Read the version's evaluation."),
          ("report_unsupported_inputs", "tool:model_contract", "Name inputs the version cannot take.")],
         ["versioned_estimate_or_refusal", "snapshot", "evidence", "limitations", "caller"],
         ["problem", "model_version"], _NEVER),
    _cfg("home", "entry", "The person's own conversation on Home",
         "Hand the person's request to the Lead and bring the Lead's questions and report back to them",
         [("open_case", "tool:delegate", "Hand the person's request, in their words, to the Lead."),
          ("confirm_with_person", "human:requester", "Ask the person a team's question and record their answer."),
          ("present_report", "tool:present_options", "Show the person the Lead's joined report.")],
         ["case_link", "attribution"], ["person", "project", "problem"], _NEVER),
    _cfg("lead", "coordinator", "The coordinating role", "Coordinate the teams' work within one problem",
         [("check_prerequisites", "tool:check_prerequisites", "Check what a task needs before it starts."),
          ("delegate", "tool:delegate", "Open a child task for a team agent."),
          ("confirm_with_requester", "human:requester", "Pass a team's question to the person who asked and carry their answer back."),
          ("review_evidence", "tool:review_evidence", "Check each team's reply against its question.")],
         ["coordinated_reply", "contributing_teams", "next_owner", "attribution"], ["person", "project", "problem"], _NEVER),
    _cfg("advisor", "specialist", "The reviewing role", "Review evidence and suggest a next step",
         [("review_evidence", "tool:review_evidence", "Read the evidence behind an answer."),
          ("propose_revision", "tool:propose_revision", "Suggest a revision, never apply it.")],
         ["advisory_finding", "attribution"], ["person", "project", "problem"], _NEVER),
)}


# ------------------------------------------------------------------------------------------------------ rules


# 0.167.0 (2026-11-21): the kinds of help the Lead hands to teams, and to which team agents, in order. Only teams whose agent
# owns tasks in this release are named; business decisions and advice stay with the person and the Lead does not take them.
# 0.168.0: a model also goes to ML Engineering, after Data Science has answered, to check how it would be trained and held
# out. This release binds no training tool (``run_stage``), so the check is all it does; a host that binds one keeps the
# Lead's tasks check-only.
LEAD_ROUTES: dict[str, tuple[str, ...]] = {
    "model": ("data_science", "ml_engineering"),
    "analyze": ("data_science",),
    "report": ("data_science",),
    "prepare": ("data_science",),
}

# 0.168.0: a request about calling or contacting accounts also goes to Sales, which plans the calls within its capacity
# and never contacts anyone (``contact_customers`` stays forbidden to every agent). Words, not an understanding: a request
# that names outreach another way goes without Sales, and the Lead's report says which teams answered.
_PEOPLE = r"(?:accounts?|customers?|clients?|leads?|users?|members?|subscribers?|them)"
OUTREACH = re.compile(  # review F2: a verb aimed at people, or "who/which … should … call", never a bare word ("phone service")
    # the review of 2026-11-22: "phone" and "calls" are nouns too ("phone service customers", "how many calls customers make"),
    # so neither is a verb here; between the verb and the people only the/our/these/those (and one word after them: "the
    # at-risk customers"); "who/which … call" needs should/shall/must ("which users will call support" is a forecast)
    r"\b(?:call|calling|ring|contact|contacting|e-?mail|reach out to|follow[- ]?up with)\s+(?:(?:the|our|these|those)\s+(?:[\w-]+\s+)?)?"
    + _PEOPLE + r"\b"
    r"|\b(?:who|which\s+(?:[\w-]+\s+)?" + _PEOPLE + r")\s+(?:should|shall|must)\b[^.?!]{0,60}?"
    r"\b(?:call|contact|e-?mail|reach out)\w*"
    r"|\boutreach\b", re.I)
OUTREACH_TRACKS = ("model", "analyze")
# 0.173.2 (B4.1, the team-arms dev run): a request that asks for an offer, a credit, a discount, a budget or a campaign also
# goes to Marketing (the offer's terms and cost), and to Sales, which works the accounts it is for. A sentence that leaves
# the offer out ("leave the offer out", "no offer", "without a discount", "not ... the budget") does not.
# a word that names data, not an offer asked for ("the campaign column", "the budget for the model's compute": the review
# of B4.1, F8)
_DATA_AFTER = (r"(?!\s+(?:column|columns|feature|features|field|fields|data|variable|variables|flag|flags|history|"
               r"for\s+(?:the\s+)?(?:[\w-]+\s+)?(?:model|compute|training|run|cloud|gpus?|servers?))\b)")
OFFER = re.compile(r"\b(?:offers?|credits?(?!\s+(?:cards?|scores?|scoring|risk|ratings?|limits?|lines?|history|checks?|bureau)\b)|"
                   r"discounts?|incentives?|vouchers?|coupons?|budgets?|campaigns?|promotions?|promo)\b" + _DATA_AFTER, re.I)
_NO_OFFER = re.compile(r"(?:\b(?:no|not|without|leave|leaving|left|skip|skipping|exclude|excluding|except|drop|ignore)|\w+n't)\b"
                       r"(?:\s+[\w'-]+){0,3}?\s+(?:offers?|credits?|discounts?|incentives?|vouchers?|coupons?|budgets?|campaigns?|"
                       r"promotions?|promo)\b", re.I)
_SENTENCES = re.compile(r"(?<=[.!?;])\s+")
# a dispatch's offer to a courier is no customer offer ("will a courier accept the first offer?")
_COURIER_OFFER = re.compile(r"\b(?:first|courier|rider|driver|delivery|dispatch|job)(?:'s)?\s+offers?\b|"
                            r"\boffers?\s+(?:to|for)\s+(?:a\s+|the\s+|each\s+)?(?:courier|rider|driver)s?\b|"
                            r"\baccept\w*\s+(?:the\s+|an?\s+|its\s+)?(?:[\w-]+\s+)?offers?\b", re.I)


# what asks for one: "with which offer", "what credit Marketing gives", "an offer budget", "plan the campaign"; never a word
# that only names one ("predict the contact rate of the campaign")
_ASKS = re.compile(r"\b(?:which|what|give|gives|giving|gets|get|receive|receives|offer|offering|propose|proposing|set|size|sizing|choose|"
                   r"pick|design|launch|run|plan|fund|spend|put\s+together|recommend|suggest|decide)\b(?:\s+[\w'%-]+){0,3}?\s+$", re.I)


# an offer of the past to analyse, not one asked for ("Which campaign drove the most renewals?", "What discount did churned
# customers receive?", "customers who got a discount last year": the re-review of B4.1)
_PAST_BEFORE = re.compile(r"\b(?:got|received|used|had|took|was\s+given|were\s+given)\s+(?:[\w'%-]+\s+){0,2}$", re.I)
_PAST_AFTER = re.compile(r"\s+(?:did|drove|worked|was|were|performed|led|had|brought|ran|cost)\b", re.I)


def asks_for_offer(question: str) -> bool:
    """Whether a request asks for an offer, a credit, a budget or a campaign (one of ``OFFER`` after a word that asks for it,
    within three words) in a sentence that does not leave it out."""
    for part in _SENTENCES.split(_COURIER_OFFER.sub(" ", str(question or ""))):
        if _NO_OFFER.search(part):
            continue
        if any(_ASKS.search(part[:m.start()]) and not _PAST_BEFORE.search(part[:m.start()]) and not _PAST_AFTER.match(part[m.end():])
               for m in OFFER.finditer(part)):
            return True
    return False


def lead_route(track: str, question: str = "") -> tuple[str, ...]:
    """The team agents the Lead asks for one kind of help (``LEAD_ROUTES``), in order, and Sales after them when the
    request is about calling or contacting accounts (``OUTREACH``); 0.173.2: Sales and Marketing after them when it asks
    for an offer, a credit, a budget or a campaign (``asks_for_offer``). Empty when the Lead does not take the kind of help."""
    teams = LEAD_ROUTES.get(str(track or ""), ())
    if teams and str(track) in OUTREACH_TRACKS and (OUTREACH.search(str(question or "")) or asks_for_offer(question)):
        teams = teams + ("sales",)
    if teams and str(track) in OUTREACH_TRACKS and asks_for_offer(question):
        teams = teams + ("marketing",)
    return teams


def opening_refusal(config: AgentConfig, owner_team: str) -> Optional[str]:
    """Why ``config``'s agent may not open a new top-level task for ``owner_team``, or None.

    Why: a top-level task is a new case, so only a way in opens one. A team agent opens it for another team (the Business
    thread asking Data Science); Home (the entry) opens it only for a coordinator, so a request from Home always passes
    through the Lead; a coordinator or a specialist never opens a case of its own."""
    owner = DIRECTORY.get(str(owner_team).split(":")[0])
    if config.kind == "team_agent":
        if owner is None or owner.kind != "team_agent":
            return f"a team opens a case for another team, not for {owner_team}"
        return None
    if config.kind == "entry":
        if owner is None or owner.kind != "coordinator":
            return f"{config.name} hands a request only to a coordinator (the Lead), not to {owner_team}"
        return None
    return f"only a team agent or the way in opens a new top-level task; {config.name} is a {config.kind}"


def delegation_refusal(config: AgentConfig, owner_team: str) -> Optional[str]:
    """Why ``config``'s agent may not open a child task for ``owner_team``, or None.

    Why: the Lead coordinates teams; it hands work to team agents only, never to another coordinator, the way in or a
    specialist (a team agent asks its own specialists). The way in never delegates: it hands one case to the Lead."""
    owner = DIRECTORY.get(str(owner_team).split(":")[0])
    if config.kind == "coordinator" and (owner is None or owner.kind != "team_agent"):
        return f"the Lead hands work only to team agents, not to {owner_team}"
    if config.kind == "entry":
        return f"{config.name} opens a case for the Lead; it does not delegate inside one"
    if owner is not None and owner.kind in ("entry", "coordinator"):
        return f"work is handed to a team or a specialist, not to {owner_team}"
    return None



class Verdict(Contract):
    """Whether a reply is joined, and in plain words why not."""

    ok: bool
    kind: Literal["accepted", "other_branch", "obsolete", "duplicate", "closed"]
    reason: str = Field(min_length=1, max_length=SHORT)


class Join(Contract):
    """Where a parent stands with its children: which returned, which are still out."""

    status: Literal["complete", "partial"]
    returned: list[str] = Field(default_factory=list, description="Child task ids with a returned report.")
    outstanding: list[str] = Field(default_factory=list, description="Child task ids still out.")
    unresolved: list[str] = Field(default_factory=list, description="Children that returned blocked or failed.")
    reports: list[str] = Field(default_factory=list, description="The one report id joined per returned child.")


def is_open(task: TeamTask) -> bool:
    """Whether the task still waits on someone. A clarification (waiting_for_person) keeps it open."""
    return task.status not in CLOSED_TASKS


def validate_transition(task: TeamTask, new_status: str, by: Actor, *, superseded_by: Optional[str] = None,
                        children: Sequence[TeamTask] = (), child_reports: Iterable[WorkReport] = ()) -> TeamTask:
    """The task in its new status, or ValueError.

    Why: one accountable owner moves its own task (only the owner team accepts, asks, delegates, completes or blocks
    it); the requester alone withdraws its question (cancel); a correction (supersede) may come from either and must
    name the replacement. A closed task never reopens: a change after completion is a new task or report revision, so
    what was already delivered keeps its own snapshot. A parent completes only when every child it delegated has
    returned (pass its children and their reports): otherwise it is partial, naming the one still out."""
    if new_status not in MOVES:
        raise ValueError(f"unknown task status {new_status!r}")
    if new_status == task.status:
        raise ValueError(f"the task is already {task.status}")
    if new_status not in MOVES[task.status]:
        raise ValueError(f"a task cannot move from {task.status} to {new_status}")
    team = by.team or (by.id if by.kind == "team" else None)
    if new_status == "cancelled":
        if team != task.requester.team:
            raise ValueError("only the requesting team withdraws its question")
    elif new_status == "superseded":
        if team not in (task.owner_team, task.requester.team):
            raise ValueError("only the owner or the requester supersedes a task")
        if not superseded_by:
            raise ValueError("a superseded task names its replacement")
    elif team != task.owner_team:
        raise ValueError(f"only the owner ({task.owner_team}) moves its task")
    if new_status == "complete" and children:
        join = join_status(task, child_reports, children=children)
        if join.outstanding:
            raise ValueError(f"a parent with a child still out is partial, not complete: {join.outstanding}")
    return TeamTask.model_validate({**task.model_dump(), "status": new_status,
                                    "superseded_by": superseded_by if new_status == "superseded" else None})


REVISABLE = frozenset({"complete", "partial", "blocked"})  # 0.170.0: an answered task works again on a revised brief


def target_revision(task: TeamTask, event: str) -> int:
    """The brief revision a confirmed outcome of ``event`` stands on for this task (:func:`confirm_target`'s rule): the
    task's own when it is the same event, or a first one while it works; the next one otherwise."""
    held = task.target
    if held is not None and held.event == event:
        return held.brief_revision
    if held is None and task.status not in REVISABLE:
        return task.brief_revision
    return task.brief_revision + 1


def confirm_target(task: TeamTask, target: TargetContract, by: Actor) -> tuple[TeamTask, bool]:
    """The task with the outcome a person confirmed (0.170.0, T1), and whether that revised its brief.

    Why: what "feasible" means rests on the outcome. A first confirmed target given while the task works is set on the
    task's own brief revision. The same event in other words later changes nothing (the work that rested on it still
    stands). Another event, or a first one given after the task answered without one (its answer rested on an outcome
    nobody confirmed, or it blocked because the words did not settle it), is a new question: the task's brief revision
    rises by one, and ``target.brief_revision`` must be that new revision. A
    task that had answered (complete, partial or blocked) works again (accepted) on the new revision: its delivered
    report stays exactly as it was until the next revision supersedes it, and a parent refuses replies on the old
    revision (``accept_reply``). This is the one way an answered task works again; every other move still goes through
    :func:`validate_transition`. Only the task's owner or its requester records the target, always for a person
    (``target.confirmed_by``); a cancelled, superseded or failed task takes none."""
    team = by.team or (by.id if by.kind == "team" else None)
    if team not in (task.owner_team, task.requester.team):
        raise ValueError("only the task's owner or its requester records the outcome a person confirmed")
    if task.status in ("cancelled", "superseded", "failed"):
        raise ValueError(f"a {task.status} task takes no confirmed outcome")
    held = task.target
    if held is not None and held.event == target.event:
        return task, False
    if held is None and task.status not in REVISABLE:  # asked while it works: its answer will rest on this outcome
        if target.brief_revision != task.brief_revision:
            raise ValueError(f"a first confirmed outcome is on the task's brief revision ({task.brief_revision})")
        return TeamTask.model_validate({**task.model_dump(), "target": target.model_dump()}), False
    revision = task.brief_revision + 1
    if target.brief_revision != revision:
        raise ValueError(f"a changed outcome is on the next brief revision ({revision})")
    status = "accepted" if task.status in REVISABLE else task.status
    return TeamTask.model_validate({**task.model_dump(), "target": target.model_dump(), "brief_revision": revision,
                                    "status": status}), True


def budget_fits(parent: TeamTask, children: Sequence[TeamTask]) -> bool:
    """Whether the parent's live children (not cancelled or superseded) spend no more than the parent's budget.

    Why: each delegation carries part of the parent's budget; two children each given the whole of it would spend it
    twice."""
    live = [c for c in children if c.parent_task_id == parent.id and c.status not in ("cancelled", "superseded")]
    return sum(c.budget_eur for c in live) <= parent.budget_eur + 1e-9


def delegate(parent: TeamTask, *, id: str, owner_team: str, question: str, expected_output: str,
             done_criteria: Sequence[str], by: Actor, thread_id: str, created_at: str,
             allowed_inputs: Sequence[str] = (), allowed_tools: Sequence[str] = (), budget_eur: float = 0.0,
             siblings: Sequence[TeamTask] = (), target: Optional[TargetContract] = None) -> TeamTask:
    """A child task asked by the parent's owner.

    Why: a delegation never hands the parent over. The parent keeps its owner and stays open; the child has its own
    owner and points back, so its reply returns to the delegating agent and nowhere else. Its budget, with its
    siblings' (the parent's children so far), fits inside the parent's. ``target`` (0.170.0): the outcome a person
    confirmed on another task, carried as a field (on the child's own brief revision), never re-read from a sentence."""
    if target is not None:
        target = target.model_copy(update={"brief_revision": parent.brief_revision})
    if parent.status not in ("accepted", "waiting_for_team", "partial"):
        raise ValueError(f"a {parent.status} task does not delegate: only an accepted task still at work does")
    if (by.team or "") != parent.owner_team:
        raise ValueError("only the parent's owner delegates")
    child = TeamTask(id=id, workspace_id=parent.workspace_id, project_id=parent.project_id,
                     problem_id=parent.problem_id, brief_revision=parent.brief_revision, parent_task_id=parent.id,
                     requester=Requester(team=parent.owner_team, member=by.id if by.kind == "person" else None,
                                         thread_id=thread_id),
                     owner_team=owner_team, question=question, expected_output=expected_output,
                     done_criteria=list(done_criteria), allowed_inputs=list(allowed_inputs),
                     allowed_tools=list(allowed_tools), budget_eur=budget_eur, created_at=created_at, target=target)
    if any(s.id == id for s in siblings):
        raise ValueError(f"a child task {id} already exists")
    if not budget_fits(parent, [*siblings, child]):
        raise ValueError("the children's budgets together fit inside their parent's")
    return child


def may_take(config: AgentConfig, action: Union[str, TeamAction]) -> bool:
    """Whether the actor may take this action.

    Why: an agent can only start what its directory lists. Given a recorded action, it must also be this agent's
    (an actor of kind agent with the configuration's name: a person or a team named like an agent is not it), and its
    source must match the binding: a tool-bound action comes from that tool; a human-bound one from a recorded human
    response, and once done it names the person who responded (``responded_by``; that the person holds the bound role
    is the host's check, W5). A suggested tool call that is not in the directory is never an action."""
    name = action if isinstance(action, str) else action.action
    binding = config.binding(name)
    if binding is None:
        return False
    if isinstance(action, str):
        return True
    if action.actor.kind != "agent" or action.actor.id.split(":")[0] != config.name:
        return False
    if binding.startswith("tool:"):
        return action.source == binding and action.responded_by is None
    if not action.source.startswith("human:"):
        return False
    if action.status in DONE_ACTIONS:
        return action.responded_by is not None and action.responded_by.kind == "person"
    return True


def can_claim(action: TeamAction) -> bool:
    """Whether a report may list the action. Why: a planned or running action has not happened; only a terminal record
    (succeeded, simulated, denied, timed out, failed, skipped) says what did."""
    return action.status in TERMINAL_ACTIONS


def _latest(reports: Iterable[WorkReport]) -> dict[str, WorkReport]:
    """The current report per task: the highest revision not superseded (a duplicate delivery is the same report)."""
    out: dict[str, WorkReport] = {}
    for r in reports:
        if r.status == "superseded":
            continue
        if r.task_id not in out or r.revision > out[r.task_id].revision:
            out[r.task_id] = r
    return out


def _parse_id(value: str) -> tuple[str, int]:
    task, _, rev = value.rpartition("@r")
    if not task or not rev.isdigit():
        raise ValueError(f"not a report id: {value!r}")
    return task, int(rev)


def accept_reply(parent: TeamTask, child_report: WorkReport, current_brief_revision: int,
                 accepted: Iterable[Union[WorkReport, str]] = ()) -> Verdict:
    """Whether a child's reply joins the parent.

    Why: each branch keeps its own status. A reply from another branch (a child of a different parent) cannot close
    this one; a superseded report, a reply on an older brief revision, or a revision older than one already joined
    for the same task answers a question that has moved on; the same task and revision delivered twice (a retried
    delivery) is one reply; a cancelled or superseded parent takes no more replies."""
    if child_report.parent_task_id != parent.id:
        return Verdict(ok=False, kind="other_branch", reason=f"{child_report.id} answers another task, not {parent.id}")
    if parent.status in ("cancelled", "superseded"):
        return Verdict(ok=False, kind="closed", reason=f"{parent.id} is {parent.status} and takes no more replies")
    if child_report.status == "superseded":
        return Verdict(ok=False, kind="obsolete", reason=f"{child_report.id} was superseded by {child_report.superseded_by}")
    if child_report.brief_revision < current_brief_revision:
        return Verdict(ok=False, kind="obsolete", reason=f"{child_report.id} answers brief revision "
                       f"{child_report.brief_revision}; the brief is now at {current_brief_revision}")
    seen = [_parse_id(a if isinstance(a, str) else a.id) for a in accepted]
    if (child_report.task_id, child_report.revision) in seen:
        return Verdict(ok=False, kind="duplicate", reason=f"{child_report.id} was already joined")
    newer = [rev for task, rev in seen if task == child_report.task_id and rev >= child_report.revision]
    if newer:
        return Verdict(ok=False, kind="obsolete", reason=f"{child_report.id} is older than revision {max(newer)}, "
                       "already joined")
    return Verdict(ok=True, kind="accepted", reason=f"{child_report.id} joins {parent.id}")


def join_status(parent: TeamTask, children_reports: Iterable[WorkReport], *, children: Sequence[TeamTask]) -> Join:
    """Where the parent stands: complete only when every child returned, otherwise partial, naming the ones still out.

    Why: a parent closes only when all its children have returned, or it delivers an explicit partial result naming
    the one outstanding. Each child is joined once, by its current report (the latest revision not superseded); a
    partial child report is not a result; a cancelled or superseded child is no longer expected; a report from another
    branch is ignored; a current report on an older brief revision than the parent's is refused (it answers a question
    the parent no longer asks); a child revised since its current report (0.170.0) is still out."""
    expected = [c for c in children if c.parent_task_id == parent.id and c.status not in ("cancelled", "superseded")]
    ids = {c.id for c in expected}
    current = _latest(r for r in children_reports if r.task_id in ids and r.parent_task_id == parent.id)
    stale = [r.id for r in current.values() if r.brief_revision < parent.brief_revision]
    if stale:
        raise ValueError(f"child reports on an older brief revision than the parent's: {stale}")
    # 0.170.0 (T1): a child whose brief was revised (a changed confirmed outcome) is out again until it answers the new
    # revision: its earlier report rests on an outcome nobody asks any more, and is never joined as its result
    returned = [c.id for c in expected if c.id in current and current[c.id].status in RETURNED_REPORTS
                and current[c.id].brief_revision >= c.brief_revision]
    outstanding = [c.id for c in expected if c.id not in returned]
    unresolved = [t for t in returned if current[t].status != "complete"]
    return Join(status="partial" if outstanding else "complete", returned=returned, outstanding=outstanding,
                unresolved=unresolved, reports=[current[t].id for t in returned])


def _line(a: TeamAction) -> ActionLine:
    return ActionLine(action_id=a.action_id, action=a.action, actor=a.actor.id, status=a.status,
                      ref=a.output_ref or a.failure_ref, reason=a.reason)


def _quotes(text: Optional[str], notes: Sequence[str]) -> bool:
    return bool(text) and any(n and n.lower() in text.lower() for n in notes)


def _one_record_per_action(actions: Sequence[TeamAction]) -> list[TeamAction]:
    """One record per action id, in first-seen order. The same record twice is one; a planned or running record
    followed by its terminal one is the terminal one; two different terminal records for one id contradict."""
    by_id: dict[str, list[TeamAction]] = {}
    for a in actions:
        by_id.setdefault(a.action_id, [])
        if a not in by_id[a.action_id]:
            by_id[a.action_id].append(a)
    out = []
    for aid, records in by_id.items():
        terminal = [a for a in records if can_claim(a)]
        if len(terminal) > 1 or (not terminal and len(records) > 1):
            raise ValueError(f"action {aid} has conflicting records")
        out.append(terminal[0] if terminal else records[0])
    done_keys = [a.retry_key for a in out if a.status in DONE_ACTIONS]
    repeated = sorted({k for k in done_keys if done_keys.count(k) > 1})
    if repeated:
        raise ValueError(f"two completed attempts share a retry key (one side effect done twice): {repeated}")
    return out


def _tool(name: str) -> str:
    return name[5:] if name.startswith("tool:") else name


def build_report(task: TeamTask, actions: Iterable[TeamAction], child_reports: Iterable[WorkReport], *, answer: str,
                 evidence: Sequence[EvidenceRef] = (), limits: Sequence[str] = (), next_task: NextStep,
                 private_notes: Sequence[str] = (), children: Sequence[TeamTask] = (), revision: int = 1,
                 brief_revision: Optional[int] = None, status: Optional[str] = None,
                 created_at: Optional[str] = None) -> WorkReport:
    """The report, built only from the action log and the child reports.

    Why each rule:
    - Only terminal actions are listed (completed, failed, skipped), so a planned or running action is never shown as
      done; an unfinished one is named in the limits and keeps the report partial. An action is listed once (by its
      id); two completed attempts with one retry key would claim one side effect twice and are refused.
    - A child result is never filled in: child reports are joined only against the children they answer (pass
      ``children``), once each; a partial or outstanding child is named and keeps the report partial.
    - A private note never reaches a shared report. What is marked private is dropped: evidence with ``private`` and a
      limit that is itself one of ``private_notes``. Anything else that quotes a note (the answer, an action's
      reason, output or failure, the next step, any evidence field, another limit) is refused: an action record or an
      answer is never silently edited.
    - Only permitted evidence: when the task lists its allowed inputs or tools, evidence from another source and an
      action through another tool are refused.
    - Anything simulated (an action or a child report) makes the whole report simulated, so a demo figure is never
      read as a measured one.
    - The report is complete only when its task is complete, with a completed action or a returned child."""
    if task.status in ("cancelled", "superseded", "open"):
        raise ValueError(f"a {task.status} task has no report")
    actions = list(actions)
    foreign = [a.action_id for a in actions if a.task_id != task.id]
    if foreign:
        raise ValueError(f"actions of another task cannot be reported here: {foreign}")
    actions = _one_record_per_action(actions)
    if task.allowed_tools:
        allowed = {_tool(t) for t in task.allowed_tools}
        outside = [a.action_id for a in actions if a.source.startswith("tool:") and _tool(a.source) not in allowed]
        if outside:
            raise ValueError(f"actions through a tool the task does not allow: {outside}")
    if task.allowed_inputs:
        outside = [e.ref for e in evidence if not e.private and e.source not in task.allowed_inputs]
        if outside:
            raise ValueError(f"evidence from a source the task does not allow: {outside}")

    notes = [n for n in private_notes if n]
    kept_evidence = [e for e in evidence if not e.private]
    kept_limits = [x for x in limits if x.strip() not in {n.strip() for n in notes}]
    quoted = [("the answer", answer), ("the next owner", next_task.owner), ("what unblocks it", next_task.unblocks)]
    quoted += [(f"action {a.action_id}", v) for a in actions for v in (a.reason, a.output_ref, a.failure_ref)]
    quoted += [(f"evidence {e.ref}", v) for e in kept_evidence
               for v in (e.source, e.ref, e.input_revision, e.uncertainty, *e.assumptions)]
    quoted += [("a limit", x) for x in kept_limits]
    leaks = sorted({where for where, text in quoted if _quotes(text, notes)})
    if leaks:
        raise ValueError(f"a private note is quoted in {', '.join(leaks)}: a shared report never carries it")

    terminal = [a for a in actions if can_claim(a)]
    unfinished = [a for a in actions if not can_claim(a)]
    lines = ReportActions(completed=[_line(a) for a in terminal if a.status in DONE_ACTIONS],
                          failed=[_line(a) for a in terminal if a.status in FAILED_ACTIONS],
                          skipped=[_line(a) for a in terminal if a.status == "skipped"])

    child_reports = list(child_reports)
    other = [r.id for r in child_reports if r.parent_task_id != task.id]
    if other:
        raise ValueError(f"reports of another branch cannot be joined here: {other}")
    if child_reports and not children:
        raise ValueError("child reports are joined against the children they answer: pass children")
    joined: list[WorkReport] = []
    outstanding: list[str] = []
    if children:
        join = join_status(task, child_reports, children=children)
        joined = [r for r in _latest(child_reports).values() if r.id in join.reports]
        outstanding = join.outstanding

    if unfinished:
        kept_limits.append("Not finished yet: " + ", ".join(f"{a.action} ({a.status})" for a in unfinished) + ".")
    if outstanding:
        kept_limits.append("Still waiting on: " + ", ".join(outstanding) + ".")

    if task.status in ("blocked", "failed"):
        derived = task.status
    elif task.status == "complete" and not (outstanding or unfinished):
        derived = "complete"
    else:
        derived = "partial"
    status = status or derived
    if status == "complete" and task.status != "complete":
        raise ValueError(f"a report is complete only when its task is: this task is {task.status}")
    if status == "complete" and (outstanding or unfinished):
        raise ValueError("a report with an outstanding child or an unfinished action is partial, not complete")
    if status == "complete" and not lines.completed and not joined:
        raise ValueError("a complete report needs a completed action or a returned child")
    if status == "superseded":
        raise ValueError("a report becomes superseded only through supersede()")

    simulated = any(a.status == "simulated" for a in terminal) or any(r.simulated for r in joined)
    return WorkReport(task_id=task.id, revision=revision, parent_task_id=task.parent_task_id,
                      brief_revision=brief_revision or task.brief_revision, answer=answer, status=status,
                      assignment=Assignment(question=task.question, done_criteria=task.done_criteria),
                      owner=task.owner_team, actions=lines, evidence=kept_evidence, limits=kept_limits,
                      next_task=next_task, outstanding=outstanding, simulated=simulated,
                      child_reports=[r.id for r in joined], created_at=created_at)


def supersede(old_report: WorkReport, new_report: WorkReport) -> tuple[WorkReport, WorkReport]:
    """The old report marked superseded and pointing to the new one, and the new one pointing back.

    Why: a correction never rewrites what was delivered. The reader of the old report finds its replacement, the
    reader of the new one finds what it replaced, and reports that did not depend on the change stay as they were.
    Each link is set once: a replacement that already replaces another report, or is itself superseded, is refused."""
    if old_report.task_id != new_report.task_id:
        raise ValueError("a report is superseded by a later revision of the same task")
    if new_report.revision <= old_report.revision:
        raise ValueError("the replacement is a later revision")
    if old_report.superseded_by not in (None, new_report.id):
        raise ValueError(f"{old_report.id} is already superseded by {old_report.superseded_by}")
    if new_report.supersedes not in (None, old_report.id):
        raise ValueError(f"{new_report.id} already replaces {new_report.supersedes}")
    if new_report.status == "superseded":
        raise ValueError(f"{new_report.id} is itself superseded and cannot replace another report")
    old = old_report.model_copy(update={"status": "superseded", "superseded_by": new_report.id})
    new = new_report.model_copy(update={"supersedes": old_report.id})
    return WorkReport.model_validate(old.model_dump()), WorkReport.model_validate(new.model_dump())


def next_owner(task: TeamTask, reports: Iterable[WorkReport], *, children: Sequence[TeamTask] = ()) -> Optional[str]:
    """Who owes the next response, or None when nobody does.

    Why: the reader of a task must always see who acts next. An unaccepted or active task waits on its owner; a
    clarification waits on the person who asked; a delegation waits on the first outstanding child's owner; a returned
    report names its own next owner; a closed task with a closed report waits on nobody."""
    if task.status in ("cancelled", "superseded"):
        return None
    if task.status == "waiting_for_person":
        return task.requester.member or task.requester.team
    if task.status in ("waiting_for_team", "partial") and children:
        join = join_status(task, reports, children=children)
        if join.outstanding:
            return next(c.owner_team for c in children if c.id == join.outstanding[0])
    own = _latest(r for r in reports if r.task_id == task.id).get(task.id)
    if own is not None:
        return None if own.next_task.closed else own.next_task.owner
    if task.status in ("complete", "failed", "blocked"):
        return task.requester.team
    return task.owner_team


def may_approve(actor: Actor, decision: TeamDecision) -> bool:
    """Whether the actor may approve. Why: permission to answer is not permission to decide; an agent (or a team acting
    as one) never approves on behalf of the decision owner, however good its recommendation is."""
    return actor.kind == "person" and decision.decision_owner.kind == "person" and actor.id == decision.decision_owner.id


# --- T2: corrections with a base revision (2026-11-23) ---

CORRECTION_VERDICTS = ("apply", "duplicate", "stale_base")
_FIELD = re.compile(r"^[a-z][a-z0-9_]*$")
_VALUE = re.compile(r"^-?[A-Za-z0-9_.:+-]+$")
_CORRECTION = re.compile(r";([a-z][a-z0-9_]*)=(-?[A-Za-z0-9_.:+-]+)@r([1-9][0-9]*)$")


def correction_revision(snapshot: str, field: str, value: Any, base_revision: int) -> str:
    """The input revision of an estimate corrected on one field: the snapshot it reads, the field and its value, and the
    report revision the correction is based on (``dispatch:1042@18:00;kitchen_minutes=8@r1``).

    Why: a correction is an attempt of its own (its retry key is this revision), so an explicit zero (a delay withdrawn) is
    never the base snapshot again (which would return the first estimate as a duplicate), and the same value on another
    base is another correction (a late one on an older base is judged by :func:`accept_correction`, never replayed)."""
    snapshot, field = str(snapshot or ""), str(field or "")
    if not snapshot or ";" in snapshot:
        raise ValueError("a correction names the snapshot it corrects (with no ';' in it)")
    if not _FIELD.match(field):
        raise ValueError(f"a corrected field is a lower-case name, not {field!r}")
    if value is None or isinstance(value, bool) or not _VALUE.match(str(value)):
        raise ValueError(f"a correction's value is a number or a short code, not {value!r}")
    if isinstance(base_revision, bool) or int(base_revision) < 1:
        raise ValueError("a correction is based on a report revision (1 or later)")
    out = f"{snapshot};{field}={value}@r{int(base_revision)}"
    if len(out) > 120:
        raise ValueError("a correction's revision is at most 120 characters")
    return out


def read_correction(revision: str) -> Optional[tuple[str, str, int]]:
    """The field, value and base revision of a corrected input revision, or None for an uncorrected one."""
    found = _CORRECTION.search(str(revision or ""))
    return (found.group(1), found.group(2), int(found.group(3))) if found else None


def accept_correction(current_revision: int, base_revision: int, *, already_applied: bool = False) -> str:
    """Whether a correction is applied: ``apply`` (based on the current revision: it becomes the next one), ``duplicate``
    (already applied: nothing changes), or ``stale_base`` (based on an older revision than the current one).

    Why: the last correction to arrive must not win because it arrived last. A correction says which answer it corrects;
    when that answer has been replaced since, applying it would overwrite a newer answer it never saw, so it is refused
    and the sender reconciles. A base later than the current revision, or before the first, names no answer: refused."""
    current, base = int(current_revision), int(base_revision)
    if current < 1 or base < 1 or base > current:
        raise ValueError(f"a correction based on revision {base} names no answer: the current revision is {current}")
    if already_applied:
        return "duplicate"
    return "apply" if base == current else "stale_base"
