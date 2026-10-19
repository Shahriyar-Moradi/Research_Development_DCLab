"""A capture rule for every task (the storing phase, 32.1): what must be captured, when, for which models, what makes it
correct, and what decides its sealed split.

``check(record, rule)`` reads one normalised record (a :class:`dclab_core.store.StoreRecord`'s fields as a mapping) and
returns what is **missing**, **late** (its state written after its decision) or **unconfirmed** (a label nobody confirmed), as
findings. It never returns a bare pass or fail and never removes anything: the host keeps every record whole in ``raw``
and only routes a record with findings to quarantine for the sets that need what it lacks (the owner, 2026-10-20: the
guardrails must not prevent capturing the data).

The rules are versioned (:data:`VERSION`): a record says which version checked it, so a stricter rule can re-check old
records without losing them.

Pure Python and pydantic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.store import CONFIRMED, LEDGER_SOURCES, Kind, ModelType

VERSION = "capture_rules_v1"


class CaptureRule(Versioned):
    """One task's rule (v1)."""

    schema_version: Literal["dclab_core.capture_rule.v1"] = _version("capture_rule")
    task: str = Field(max_length=80)
    kind: Kind
    types: list[str] = Field(default_factory=list)      # the record types this rule reads (empty: every type of the kind)
    required: dict[str, str] = Field(default_factory=dict)  # body field -> where it comes from
    at_decision: bool = True                            # the state is captured when the decision is made
    feeds: list[ModelType] = Field(default_factory=list)
    correct_when: list[str] = Field(default_factory=list)  # what makes it right, said
    confirmed_by: list[str] = Field(default_factory=list)  # label sources that confirm it (the store's words; empty: no label)
    sealed_key: Optional[str] = Field(default=None, max_length=80)  # the body field whose hash decides the split
    none_when: dict[str, list[str]] = Field(default_factory=dict)   # a required field may be None when another says so: field -> [other, value]
    keeper: str = Field(default="", max_length=80)  # the keeper that should have caught it (named in a finding)


class Finding(Contract):
    rule: str
    what: Literal["missing", "late", "untimed", "unconfirmed", "unsealed", "unknown_rule"]
    field: Optional[str] = None
    keeper: str = ""
    detail: str = ""


def _t(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _r(task, kind, types, required, feeds, correct_when, confirmed_by=(), sealed_key=None, at_decision=True, none_when=None, keeper=None):
    return CaptureRule(task=task, kind=kind, types=list(types), required=dict(required), feeds=list(feeds), correct_when=list(correct_when),
                       confirmed_by=list(confirmed_by), sealed_key=sealed_key, at_decision=at_decision, none_when=dict(none_when or {}),
                       keeper=keeper or kind)


RULES: dict[str, CaptureRule] = {r.task: r for r in (
    _r("decision", "decisions", ("decision",), {"point": "the ledger", "state": "the decision site, at its moment", "answers": "every answerer",
       "used": "which answer the studio used (none: in shadow)", "value_used": "what the studio used", "group": "its project, draft or table (datasets.group_of)"},
       ("m1_students", "m2_workflow_decisions", "m3_ml_decisions"),
       ["its state holds what was known when it was made, in bands", "every answerer's answer is kept, the used one marked"],
       sealed_key="group", none_when={"value_used": ["used", "none"]}, keeper="decisions"),
    _r("label", "decisions", ("label",), {"record": "the decision it labels", "value": "the label", "source": "who knows (the store's words)",
       "group": "its decision's group"}, ("m1_students", "m2_workflow_decisions", "m3_ml_decisions"),
       ["its source says who knows: a person, a measurement, a benchmark, a weighted blind draw"],
       confirmed_by=CONFIRMED, sealed_key="group", at_decision=False, keeper="decisions"),
    _r("stage_run", "workflows", ("stage_run",), {"stage": "the engine", "card": "the input card", "choices": "each choice with its reason",
       "folds": "every fold's score", "holdout_reads": "the holdout's read count", "seconds": "its time", "cost_eur": "its cost",
       "project": "its project"},
       ("m3_ml_decisions", "m4_workflow_llm", "m5_solver_llm", "m6_stepwise_decisions", "m7_judges"),
       ["the holdout was read at most once", "every choice has its reason"], sealed_key="project", keeper="projects"),
    _r("graph_move", "workflows", ("graph_move",), {"move": "the validator", "from_state": "before", "to_state": "after", "allowed": "the validator's verdict",
       "project": "its project"},
       ("m2_workflow_decisions", "m6_stepwise_decisions"), ["a refused move keeps its refusal"], sealed_key="project", keeper="projects"),
    _r("conversation_turn", "conversations", ("turn",), {"surface": "home, intern, workspace_chat, twin, mcp", "role": "person or agent",
       "text": "the words, redacted", "conversation": "its conversation"}, ("m4_workflow_llm", "m2_workflow_decisions"), ["the person's words are redacted"],
       sealed_key="conversation", keeper="conversations"),
    _r("agent_step", "actions", ("step",), {"agent": "the harness", "tool": "the tool or none", "arguments": "its arguments",
       "result": "its result", "ok": "whether it worked", "run": "its run"}, ("m5_solver_llm", "m6_stepwise_decisions", "m7_judges"),
       ["every tool call has its step in the same run"], sealed_key="run", keeper="actions"),
    _r("job", "tasks", ("job_state",), {"job": "the job table", "kind": "the handler", "state": "the new state"},
       ("m2_workflow_decisions", "m6_stepwise_decisions"), ["every state change is a version"], at_decision=False, keeper="tasks", sealed_key="job"),
    _r("test_run", "tests", ("test_run",), {"files": "the units run", "ok": "the verdict", "commit": "the code it ran on"},
       ("m2_workflow_decisions", "m7_judges"), ["the whole output of a failing unit is kept"], at_decision=False, keeper="test_runs", sealed_key="run"),
    _r("model_request", "model_requests", ("request",), {"purpose": "the gateway purpose", "model": "the served model", "cost_eur": "its cost"},
       ("m4_workflow_llm", "m5_solver_llm", "m7_judges"), ["its thinking is kept whole beside its answer"], at_decision=False, keeper="model_requests", sealed_key="group"),
    _r("teacher_answer", "teachers", ("answer",), {"teacher": "Claude, GLiNER, the rules, a person, a model", "point": "the point or task",
       "answer": "the answer", "version": "the teacher's version",
       "group": "its decision's group (sealed with it)"}, ("m1_students", "m7_judges"),
       ["a teacher's answer is silver, apart from a confirmed label"], at_decision=False, none_when={"answer": ["outcome", "abstained"]}, keeper="judgments", sealed_key="group"),
    _r("student_version", "students", ("version",), {"point": "its point", "student": "its name", "version": "its version",
       "training_snapshot": "the record ids and hashes it was fitted on", "card": "its model card", "measurement": "its measurement"},
       ("m1_students",), ["a refit is a new version beside the old"], at_decision=False, keeper="students", sealed_key="point"),
    _r("correction", "decisions", ("correction",), {"what": "the card, solution, map, report, plan or cell", "before": "what DCLab proposed",
       "after": "what the person made it", "source": "who corrected it (a person)", "project": "its project or draft"}, ("m2_workflow_decisions", "m4_workflow_llm", "m6_stepwise_decisions", "m7_judges"),
       ["before and after are both kept"], confirmed_by=("person",), sealed_key="project", keeper="decisions"),
    _r("deletion", "deletions", ("deletion",), {"what": "project, draft or session", "id": "its id", "by": "who deleted it"}, (),
       ["the deleted data is still in the store"], at_decision=False, keeper="hooks"),
    _r("session_event", "sessions", (), {"kind": "the experience log's event kind", "actor": "who acted", "visit": "the visit"},
       ("m2_workflow_decisions", "m4_workflow_llm", "m6_stepwise_decisions"), ["its ids bind it to its visit, project and run"],
       sealed_key="visit", at_decision=False, keeper="sessions"),
    _r("agent_session", "agents", (), {"agent": "Claude Code, a tester's agent", "session": "its id", "where": "the checkout or worktree"},
       ("m4_workflow_llm", "m5_solver_llm", "m7_judges"), ["the transcript is whole, its reasoning included"], sealed_key="session",
       at_decision=False, keeper="agents"),
    _r("fix", "fixes", ("fix",), {"test": "the failing test", "failing_run": "its failing run", "passing_run": "the run that passed",
       "commits": "the commits between, with their files", "source": "who accepted it (a person)"}, ("m4_workflow_llm", "m5_solver_llm", "m7_judges"),
       ["a person accepted it before it became a development record"], confirmed_by=("person",), at_decision=False, keeper="fixes", sealed_key="test"),
    _r("audit_row", "audit", (), {"kind": "the audit kind", "status": "allowed, blocked, applied ...", "actor": "who"},
       ("m2_workflow_decisions",), ["every governed action is in it"], at_decision=False, keeper="audit", sealed_key="group"),
    _r("intern_session", "conversations", ("intern_session",), {"session": "its id", "version": "the write's number", "messages": "its turns",
       "steps": "its steps"}, ("m4_workflow_llm", "m5_solver_llm", "m6_stepwise_decisions"), ["each write of the session is a version"],
       sealed_key="session", at_decision=False, keeper="conversations"),
    _r("team_walk", "workflows", ("walk", "board", "interrupt"), {"team": "the team", "walk": "its id", "node": "the node or hand-off",
       "board": "the board at that moment"}, ("m2_workflow_decisions", "m5_solver_llm", "m6_stepwise_decisions"),
       ["every hand-off has its board before and after"], sealed_key="walk", keeper="projects"),
    _r("build", "workflows", ("build",), {"model_type": "M1 to M8", "context": "the business, in bands", "data": "its data card",
       "recipe": "every choice with its reason", "spent": "people's time, labels, compute, euros", "result": "against its bar",
       "verdict": "its BuildVerdict"}, ("m4_workflow_llm", "m5_solver_llm", "m7_judges"),
       ["every part filled, or unknown with why"], sealed_key="build", at_decision=False, keeper="walks"),
    _r("other_file", "other", (), {"source": "the file's path in its workspace"}, (), ["kept whole, so a later use can find it"],
       at_decision=False, keeper="other"),
    _r("curation", "curation", ("curation",), {"record": "the record", "model_type": "the model", "quality": "keep, silver or out",
       "curator": "rule or model"}, ("m1_students", "m2_workflow_decisions", "m7_judges"), ["a refused write keeps its reason"],
       at_decision=False, keeper="curators", sealed_key="group"),
)}


def rule_for(kind: str, type_: str) -> Optional[CaptureRule]:
    """The rule of a record's kind and type (None: no task has a rule for it yet, itself a finding)."""
    for rule in RULES.values():
        if rule.kind == kind and (not rule.types or type_ in rule.types):
            return rule
    return None


def check(record: Mapping[str, Any], rule: Optional[CaptureRule] = None) -> list[Finding]:
    """What the record lacks against its rule: missing fields, a state written after its decision, a label nobody confirmed.
    ``record``: a store record's fields (kind, type, at, body, provenance). Never removes anything."""
    rule = rule or rule_for(str(record.get("kind")), str(record.get("type")))
    if rule is None:
        return [Finding(rule="-", what="unknown_rule", detail=f"no capture rule for {record.get('kind')}/{record.get('type')} yet")]
    body = record.get("body") or {}
    k = rule.keeper

    def absent(field: str) -> bool:
        if body.get(field) not in (None, ""):
            return False
        other = rule.none_when.get(field)
        return not (other and body.get(other[0]) == other[1])  # e.g. no value used, because the decision was in shadow

    out = [Finding(rule=rule.task, what="missing", field=f, keeper=k, detail=f"from {src}") for f, src in rule.required.items() if absent(f)]
    if rule.at_decision:
        decided, written = _t(record.get("at")), _t(body.get("state_at"))
        if decided is None or written is None:
            out.append(Finding(rule=rule.task, what="untimed", field="state_at" if decided else "at", keeper=k,
                               detail="the decision's time and its state's time are both needed to say the state is what was known then"))
        elif written > decided:
            out.append(Finding(rule=rule.task, what="late", field="state", keeper=k,
                               detail=f"its state was written at {body.get('state_at')}, after the decision at {record.get('at')}"))
    source = LEDGER_SOURCES.get(str(body.get("source")), body.get("source"))  # the ledger's words in the store's
    if rule.confirmed_by and source not in rule.confirmed_by:
        out.append(Finding(rule=rule.task, what="unconfirmed", field="source", keeper=k,
                           detail=f"{body.get('source')!r} is not one of {', '.join(rule.confirmed_by)}"))
    if rule.sealed_key and body.get(rule.sealed_key) in (None, "") and rule.sealed_key not in rule.required:
        out.append(Finding(rule=rule.task, what="unsealed", field=rule.sealed_key, keeper=k, detail="no key to decide its split by"))
    return out


CONTRACTS["capture_rule"] = CaptureRule

__all__ = ["CaptureRule", "Finding", "RULES", "VERSION", "rule_for", "check"]
