"""The record of what a person and DCLab did (package 19.1): one event shape, a closed vocabulary of kinds, and the
functions every host runs on it.

Each host keeps its own stores. This studio writes one log; DCLab's main product already has append-only tables for
agent events, run events, model invocations and decisions, and maps their rows onto :class:`ExperienceEvent` with the
adapter in ``dclab_core.adapters.main_product``. Everything built on events (a session's timeline, its card, a
redacted export, later the failure detectors and the training sets) is a function here, so both hosts run the same
code on their own data.

Ids. Every event carries the same ids (:class:`Ids`): the workspace (the tenant), the person, the visit (one sitting,
across Home, a draft, a project and the intern), the conversation (a draft or an intern session), the project, the
run, the job, the request, the model request, and the event that caused it. ``seq`` orders events of one visit (or of
one conversation when there is no visit) where their times are equal.

Content classes. Every payload field has a declared class (:data:`KINDS`): ``meta`` (names, ids, kinds, states),
``count`` (numbers and aggregates), ``person_text`` (what a person wrote), ``agent_text`` (what an agent said),
``model_text`` (a prompt or a reply), ``cell_values`` (values from a person's table). An event says each field's class,
and it must be the declared one: a field cannot be recorded as a lower class than it is. Secrets are never recorded.
:func:`redact` keeps only the classes a use is allowed (an export for training never keeps cell values).

Versions. v2 (package 19.5) added one kind, ``run.recorded`` (a run's lineage, ``dclab_core.lineage``); a v1 event is
read as v2 by :meth:`ExperienceEvent.read`, since every v1 event is a valid v2 event.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

ContentClass = Literal["meta", "count", "person_text", "agent_text", "model_text", "cell_values"]
Actor = Literal["person", "agent", "system", "model"]
CLASS_ORDER = ("meta", "count", "agent_text", "person_text", "model_text", "cell_values")  # from least to most sensitive
META_CHARS = 300  # a meta field is a name, an id or a state: short, never prose
# the shapes of credentials: an event holding one is refused whatever its class (keys are never recorded)
SECRET = re.compile(
    r"(?<![A-Za-z0-9])sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"   # OpenAI, Anthropic
    r"|(?<![A-Za-z0-9])sk_live_\w{20,}"                      # Stripe
    r"|(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}"             # AWS key ids
    r"|(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{40,}"
    r"|(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"                   # Hugging Face
    r"|(?<![A-Za-z0-9])AIza[\w-]{35}"                         # Google
    r"|(?<![A-Za-z0-9])xox[abpr]-[A-Za-z0-9-]{10,}"           # Slack
    r"|eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}"                    # a JSON web token
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?i:\bbearer\s+[A-Za-z0-9._~+/-]{20,})"
    r"|[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@")                # a password in a connection string
SHAREABLE = frozenset({"meta", "count"})  # what any export may keep
READ_AS_CURRENT = frozenset({"dclab_core.experience_event.v1", "dclab_core.experience_event.v2", "dclab_core.experience_event.v3",
                             "dclab_core.experience_event.v4", "dclab_core.experience_event.v5",
                             "dclab_core.experience_event.v6", "dclab_core.experience_event.v7", "dclab_core.experience_event.v8",
                             "dclab_core.experience_event.v9", "dclab_core.experience_event.v10",
                             "dclab_core.experience_event.v11"})  # v12: model.request's prompt-cache tokens (cache_read_tokens, cache_write_tokens); v11 is read as v12  # v11 (28.4) added chat.agent's question_source (the understanding critic's questions); v10 is read as v11  # v10 (28.2) added team.handoff, team.interrupt and team.recorded; v9 is read as v10  # read as v9: v9 added the problem.*, solution.* and programme.* kinds of a problem of several parts; read as v8: each a subset (v3 added consent; v4 the ui.* and auth.* kinds; v5 the error fingerprint; v6 outcome.observed; v7 feedback kinds, column sensitivity and the pipeline steps; v8 trace, span and parent_span in the ids)
FINAL_STAGES = frozenset({"final", "terminal"})  # the last stage: this studio's, and the main product's

M, C, P, A, X, V = "meta", "count", "person_text", "agent_text", "model_text", "cell_values"

# The vocabulary: kind -> {payload field: content class}. Closed: a host maps its own rows onto these kinds, and a kind
# it cannot fill is listed as missing, never invented. A new kind or field moves the contract's version.
KINDS: dict[str, dict[str, str]] = {
    # a sitting: from the first request to the last
    "visit.started": {"surface": M, "client": M},
    "visit.ended": {"reason": M, "seconds": C},
    # 23.3: sign-in and sessions; a user agent as a hash, a network address without its last part, never a token or a password
    "auth.sign_in": {"via": M, "session": M, "agent": M, "net": M},
    "auth.sign_in_failed": {"reason": M, "via": M, "agent": M, "net": M},
    "auth.sign_out": {"session": M},
    "auth.session_expired": {"session": M},
    "auth.token_created": {"token": M, "name": P},
    "auth.token_revoked": {"token": M},
    "auth.workspace_switched": {"from": M, "to": M},
    "auth.role_changed": {"member": M, "was": M, "role": M, "by": M},
    # 23.2: the browser's own events: a page, an element, a file's kind, a code; never a form's contents (``ui_payload`` refuses them)
    "ui.page_viewed": {"view": M, "pane": M, "route": M},
    "ui.action": {"view": M, "name": M, "target": M},
    "ui.upload_started": {"view": M, "file_kind": M, "bytes": C},
    "ui.upload_finished": {"view": M, "status": M, "seconds": C},
    "ui.error_shown": {"view": M, "code": M, "status": M},
    # chat, on Home or with the intern
    "chat.person": {"text": P, "answers_question": M, "surface": M},
    "chat.agent": {"text": A, "kind": M, "question_field": M, "options": A, "surface": M, "message": M, "version": M,  # 23.6: the message's id and the release that wrote it
                   "question_source": M},  # 28.4: who sent the question ("critic": the understanding critic sent a field back)
    # a draft: Home before a project exists
    "draft.created": {"problem": P, "pack": M},
    "draft.edited": {"field": M, "before": P, "after": P},
    "draft.built": {"project": M, "target": M, "kind": M},
    "draft.refused": {"reason": A, "rule": M},
    # data
    "data.uploaded": {"name": M, "bytes": C, "rows": C, "columns": C, "source": M},
    "data.cleaned": {"steps": C, "rows_before": C, "rows_after": C},
    "data.analysed": {"rows": C, "columns": C, "column_names": M, "roles": M, "findings": C, "sensitivity": M},  # 23.7: column -> class, only the ones that are not none
    "data.labels_joined": {"key": M, "label_rows": C, "rows_marked": C, "not_found": C},
    "data.cases_found": {"keys": M, "case_kinds": M, "tables_matched": C},
    # the solution: what is predicted, at which moment, without which columns
    "solution.proposed": {"by": M, "target": M, "task": M, "kind": M, "forbidden": M, "moment": P},
    "solution.saved": {"target": M, "task": M, "kind": M, "metric": M, "forbidden": M, "moment": P, "changed": M},
    "solution.refused": {"target": M, "reason": A, "rule": M},
    # the engine's stages and the workflow graph
    "stage.started": {"stage": M, "machine": M},
    "stage.finished": {"stage": M, "metric": M, "value": C, "seconds": C, "chosen": M},
    "stage.failed": {"stage": M, "code": M, "error": A, "fingerprint": M, "frame": M},  # 23.4: grouped by fingerprint
    "stage.approved": {"stage": M, "choice": M, "overrode": M, "reason": P},
    "graph.move": {"move": M, "allowed": M, "from": M, "to": M, "failed_checks": M, "rules": M},
    # agents, their tools and the models behind them
    "agent.step": {"agent": M, "n": C, "state": M, "reply": A, "tool": M, "verdict": M, "seconds": C},
    # a tool's arguments and result can hold a table's values: they are cell values; their names and a code are meta
    "tool.call": {"tool": M, "scope": M, "argument_names": M, "arguments": V, "verdict": M, "code": M, "result": V},
    "model.checked": {"purpose": M, "tier": M, "model": M, "passed": M, "reason": A},  # 23.10: the code that checks a reply, and its verdict
    "model.request": {"purpose": M, "tier": M, "model": M, "input_tokens": C, "output_tokens": C, "cache_read_tokens": C, "cache_write_tokens": C,
                      "cost": C, "currency": M,
                      "seconds": C, "outcome": M, "prompt": X, "reply": X},
    # decisions: proposed by an agent or a rule, accepted, overridden or refused by a person or a check
    "decision.proposed": {"point": M, "by": M, "choice": M, "why": A},
    "decision.accepted": {"point": M, "choice": M, "by": M},
    "decision.overridden": {"point": M, "proposed": M, "chosen": M, "reason": P},
    "decision.refused": {"point": M, "choice": M, "rule": M, "reason": A},
    # quality signals
    "feedback.given": {"on": M, "rating": M, "reason": P, "kind": M, "version": M, "score": C, "edited": P, "role": M},  # 23.6: kinds, the judged message's version, an edit
    "outcome.observed": {"metric": M, "value": C, "reference": C, "observed": M, "verdict": M, "decisions": C, "note": P},  # 23.5
    "consent.changed": {"scope": M, "use": M, "on": M, "was": M, "level": M, "notice_version": M},  # 23.1: a switch moved
    "export.made": {"what": M, "format": M},
    "job.started": {"kind": M, "key": M},
    "job.stopped": {"kind": M, "by": M, "status": M},
    "error.raised": {"where": M, "code": M, "message": A, "fingerprint": M, "frame": M, "status": M},  # a message may quote a value: agent text, never meta; 23.4: the fingerprint groups
    # a run's lineage (v2, package 19.5): the flat parts of a dclab_core.lineage.RunRecord; the record stays with the host
    "pipeline.step_started": {"run": M, "step": M, "kind": M, "index": C, "fold": C},  # 23.8
    "pipeline.step_finished": {"run": M, "step": M, "kind": M, "index": C, "fold": C, "metric": M, "value": C, "seconds": C, "digest": M},
    "run.recorded": {"run": M, "run_kind": M, "name": M, "status": M, "fingerprint": M, "inputs": M, "params": M, "outputs": M, "code": M,
                     "upstream": M, "parent": M, "seconds": C, "numbers": C, "steps": C},
    # v9 (24.7): a problem of several parts. The map as drawn (the fingerprint, its size, the families of its sub-problems: names, never the person's words), a person's change to it, the
    # approach routed to a sub-problem (with the candidates and what blocked them), the composition chosen on validation, and the programme built (its numbers: aggregates only)
    "problem.mapped": {"map": M, "size": M, "template": M, "source": M, "by": M, "parts": C, "sub_problems": C, "families": M},
    "problem.part_changed": {"map": M, "action": M, "part": M, "by": M, "parts_before": C, "parts_after": C, "reason": P},
    "solution.routed": {"map": M, "sub_problem": M, "family": M, "chosen": M, "source": M, "candidates": M, "blocked": M},
    "solution.composed": {"programme": M, "solution": M, "chosen": M, "best_leaf": M, "leaves": C, "positives": C, "best_leaf_score": C, "chosen_score": C},
    "programme.built": {"programme": M, "state": M, "metric": M, "holdout_read": M, "leaves_built": C, "leaves_blocked": C, "composed": C, "best_leaf": C,
                        "difference": C, "low": C, "high": C, "beats_best_leaf": M},
    # v10 (28.2): a team of agents passing its board, and the pauses a person answers
    "team.handoff": {"thread": M, "team": M, "from_node": M, "to_node": M, "what": M, "board_before": C, "board_after": C, "state": M},
    "team.interrupt": {"thread": M, "team": M, "node": M, "interrupt": M, "field": M, "gate": M, "status": M, "text": A,
                       "board_before": C, "board_after": C},
    "team.recorded": {"thread": M, "team": M, "status": M, "nodes": C, "handoffs": C, "interrupts": C, "board_version": C, "cost": C,
                      "seconds": C, "started": M, "finished": M},
}


Kind = Literal[tuple(KINDS)]  # type: ignore[valid-type]


def _vocabulary(schema: dict[str, Any]) -> None:
    """The vocabulary in the JSON Schema too: each kind's payload fields, and the class each must carry."""
    schema.setdefault("required", [])
    if "schema_version" not in schema["required"]:
        schema["required"].insert(0, "schema_version")
    def both_ways(field: str) -> list[dict[str, Any]]:  # a payload field has its class, and a class has its field
        return [{"if": {"properties": {"payload": {"required": [field]}}},
                 "then": {"properties": {"classes": {"required": [field]}}}},
                {"if": {"properties": {"classes": {"required": [field]}}},
                 "then": {"properties": {"payload": {"required": [field]}}}}]

    schema["allOf"] = [{"if": {"properties": {"kind": {"const": kind}}, "required": ["kind"]},
                        "then": {"properties": {"payload": {"propertyNames": {"enum": sorted(fields)}},
                                                "classes": {"properties": {f: {"const": c} for f, c in sorted(fields.items())},
                                                            "propertyNames": {"enum": sorted(fields)}}},
                                 "allOf": [rule for f in sorted(fields) for rule in both_ways(f)]}}
                       for kind, fields in sorted(KINDS.items())]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Ids(Contract):
    """The ids that join an event to everything else. ``workspace`` is always set; the rest when they exist."""

    workspace: str = Field(min_length=1, max_length=80)
    user: Optional[str] = Field(default=None, max_length=80)
    visit: Optional[str] = Field(default=None, max_length=80)
    conversation: Optional[str] = Field(default=None, max_length=80)  # a draft or an intern session
    project: Optional[str] = Field(default=None, max_length=80)
    run: Optional[str] = Field(default=None, max_length=80)
    job: Optional[str] = Field(default=None, max_length=80)
    request: Optional[str] = Field(default=None, max_length=80)
    model_request: Optional[str] = Field(default=None, max_length=80)
    parent: Optional[str] = Field(default=None, max_length=80)
    trace: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$", description="v8 (23.11): the W3C trace id of the request, shared by its events, its jobs and its model requests.")
    span: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{16}$", description="v8: the span this event was recorded in (the request's, a job's, a step's, a model request's).")
    parent_span: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{16}$", description="v8: the span that span ran inside.")


class ExperienceEvent(Versioned):
    """One thing a person, an agent, the system or a model did, with its ids and the class of every field it holds."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_vocabulary)

    schema_version: Literal["dclab_core.experience_event.v12"] = _version("experience_event", 12)
    id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{6,80}$")
    at: str = Field(description="ISO 8601 with microseconds, UTC.")
    seq: int = Field(default=0, ge=0)
    ids: Ids
    actor: Actor
    kind: Kind
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    classes: dict[str, ContentClass] = Field(default_factory=dict)
    redacted: list[str] = Field(default_factory=list, description="Fields removed by redact(), with their class.")
    source: Optional[str] = Field(default=None, max_length=120, description="Where the host read it: table:type.")
    consent: Literal["operate", "improve", "train"] = Field(default="operate", description="v3 (23.1): the level the record was "
                                                           "written under; a later change of the switches never re-labels it.")

    @classmethod
    def read(cls, document):
        """An event as stored; a v1 event (before ``run.recorded``) is read as v2: v2 only added a kind."""
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        if data.get("schema_version") in READ_AS_CURRENT:
            data["schema_version"] = cls.model_fields["schema_version"].default
        return super().read(data)

    @field_validator("at")
    @classmethod
    def _iso(cls, value: str) -> str:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            raise ValueError("at needs its time zone: events from different hosts are ordered together")
        return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")

    @field_validator("redacted")
    @classmethod
    def _removed(cls, value: list[str]) -> list[str]:
        for item in value:
            field, _, kind = item.partition(":")
            if not field or kind not in CLASS_ORDER:
                raise ValueError(f"redacted names a field and its class (field:class), not {item!r}")
        return value

    @model_validator(mode="after")
    def _declared(self) -> "ExperienceEvent":
        fields = KINDS.get(self.kind)
        if fields is None:
            raise ValueError(f"unknown kind {self.kind!r}: the vocabulary is closed (dclab_core.experience.KINDS)")
        unknown = set(self.payload) - set(fields)
        if unknown:
            raise ValueError(f"{self.kind}: fields not in the vocabulary: {sorted(unknown)}")
        if set(self.classes) != set(self.payload):
            raise ValueError(f"{self.kind}: every payload field needs its class, and only those")
        if SECRET.search(" ".join([_text(self.payload), _text(self.ids.model_dump()), self.source or ""])):
            raise ValueError(f"{self.kind}: holds what looks like a credential; credentials are never recorded")
        for name, cls in self.classes.items():
            if cls != fields[name]:
                raise ValueError(f"{self.kind}.{name} is {fields[name]}, not {cls}: a field is never recorded as another class")
            if cls == "meta" and not _meta(self.payload[name]):
                raise ValueError(f"{self.kind}.{name} is meta: names, ids and states only (scalars, a flat list or a flat "
                                 f"mapping of them), at most {META_CHARS} characters each")
            if cls == "count" and not _number(self.payload[name]):
                raise ValueError(f"{self.kind}.{name} is a count: a number")
        return self


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} {_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return "" if value is None else str(value)


def _scalar(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) <= META_CHARS
    return value is None or isinstance(value, (bool, int, float))


def _meta(value: Any) -> bool:
    """A scalar, or a flat list or mapping of scalars: nothing nested, where values could hide."""
    if isinstance(value, (list, tuple)):
        return len(value) <= 200 and all(_scalar(v) for v in value)
    if isinstance(value, dict):
        return len(value) <= 200 and all(isinstance(k, str) and _scalar(k) and _scalar(v) for k, v in value.items())
    return _scalar(value)


def _number(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def hash_user_agent(user_agent: Any) -> Optional[str]:
    """A browser's user-agent string as sixteen hex characters (23.3): enough to tell two browsers apart, never the string."""
    import hashlib

    text = str(user_agent or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else None


def truncate_ip(address: Any) -> Optional[str]:
    """A network address with its last part removed (23.3): ``203.0.113.x`` for IPv4, the first three groups and ``::``
    for IPv6; None for anything else."""
    text = str(address or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", text):
        return ".".join(text.split(".")[:3]) + ".x"
    if ":" in text:  # the groups before a "::" are the prefix; those after it belong to the host and never enter
        head = [g for g in text.split("::")[0].split(":") if g != ""]
        if len(head) < 3 and "::" not in text:
            return None
        groups = (head + ["0", "0", "0"])[:3]
        return ":".join(groups) + "::"
    return None


UI_KINDS = frozenset(k for k in KINDS if k.startswith("ui."))
UI_NAME = re.compile(r"^[A-Za-z0-9_./#:@-]{1,80}$")  # a route pattern, an element's id or data name, a code: never words a person typed


def ui_payload(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """A browser event's payload as the record may keep it (23.2): only the kind's own fields; a name must look like a
    route, an element or a code (no spaces, no punctuation of prose), a count must be a non-negative number. Anything
    else is refused: a form's contents, a query string, a message, a file's name cannot pass."""
    if kind not in UI_KINDS:
        raise ValueError(f"not a browser event kind: {kind!r}")
    fields = KINDS[kind]
    out: dict[str, Any] = {}
    for name, cls in fields.items():
        value = payload.get(name)
        if value is None or value == "":
            continue
        if cls == C:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{kind}.{name} is a count")
            out[name] = value
        else:
            if not isinstance(value, str) or not UI_NAME.fullmatch(value):
                raise ValueError(f"{kind}.{name} is a route, an element's name or a code")
            out[name] = value
    return out


def make(kind: str, actor: str, ids: Mapping[str, Any] | Ids, payload: Mapping[str, Any] | None = None, *, id: str,
         at: str | None = None, seq: int = 0, source: str | None = None, consent: str = "operate") -> ExperienceEvent:
    """An event of ``kind`` with each field's class taken from the vocabulary. Fields left None are dropped. ``consent``:
    the level it is written under (``dclab_core.consent.level_of``), ``operate`` when the host knows none."""
    fields = KINDS.get(kind)
    if fields is None:
        raise ValueError(f"unknown kind {kind!r}")
    body = {k: v for k, v in (payload or {}).items() if v is not None}
    return ExperienceEvent(id=id, at=at or _now(), seq=seq, ids=ids if isinstance(ids, Ids) else Ids(**ids), actor=actor, kind=kind,
                           payload=body, classes={k: fields.get(k, "meta") for k in body}, source=source, consent=consent)


def timeline(events: Iterable[ExperienceEvent]) -> list[ExperienceEvent]:
    """The events in the order they happened: by time, then ``seq``, then id; an event seen twice is kept once."""
    seen: dict[str, ExperienceEvent] = {}
    for event in events:
        seen.setdefault(event.id, event)
    return sorted(seen.values(), key=lambda e: (datetime.fromisoformat(e.at), e.seq, e.id))


def redact(event: ExperienceEvent, allowed: Iterable[str]) -> ExperienceEvent:
    """The event with only the fields whose class is ``allowed``; the others are named in ``redacted``."""
    keep = set(allowed)
    unknown = keep - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown content classes: {sorted(unknown)}")
    declared = KINDS[event.kind]  # the vocabulary decides, not the event's own labels (a copy skips validation)
    payload = {k: v for k, v in event.payload.items() if declared.get(k) in keep}
    dropped = sorted(f"{k}:{declared.get(k, 'cell_values')}" for k in event.payload if k not in payload)
    return ExperienceEvent.model_validate({**event.model_dump(), "payload": payload, "classes": {k: declared[k] for k in payload},
                                           "redacted": sorted(set(event.redacted) | set(dropped))})


class ExperienceSession(Versioned):
    """A sitting, summed up from its events: who, when, on what, how it ended, what it cost."""

    schema_version: Literal["dclab_core.experience_session.v1"] = _version("experience_session")
    visit: str
    workspace: str
    user: Optional[str] = None
    started: str
    ended: str
    events: int = Field(ge=0)
    by_kind: dict[str, int] = Field(default_factory=dict)
    conversations: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    person_turns: int = Field(ge=0)
    agent_turns: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    cost: dict[str, float] = Field(default_factory=dict, description="Model spend by currency (ISO code).")
    problem: Optional[str] = Field(default=None, max_length=300, description="The problem as the person first wrote it (person text).")
    kind: Optional[str] = Field(default=None, max_length=80, description="The problem kind or pack, when one was set.")
    refused: int = Field(ge=0, description="Moves, solutions and decisions a check refused.")
    errors: int = Field(ge=0)
    outcome: Literal["built", "ran", "finished", "open"] = "open"


def session_card(events: Iterable[ExperienceEvent]) -> ExperienceSession:
    """The card of one visit's events (they must share a workspace and a visit)."""
    ordered = timeline(events)
    if not ordered:
        raise ValueError("no events")
    visits = {e.ids.visit for e in ordered}
    workspaces = {e.ids.workspace for e in ordered}
    if len(visits) != 1 or None in visits or len(workspaces) != 1:
        raise ValueError("a session card sums one visit of one workspace")
    kinds = Counter(e.kind for e in ordered)
    cost: dict[str, float] = {}
    for e in ordered:
        if e.kind == "model.request" and e.payload.get("cost") is not None:
            currency = str(e.payload.get("currency") or "unknown").upper()
            cost[currency] = round(cost.get(currency, 0.0) + float(e.payload["cost"]), 6)
    outcome = ("finished" if any(e.kind == "stage.finished" and e.payload.get("stage") in FINAL_STAGES for e in ordered)
               else "ran" if kinds["stage.finished"] else "built" if kinds["draft.built"] else "open")
    created = next((e for e in ordered if e.kind == "draft.created"), None)
    kind = next((e.payload.get("kind") for e in ordered if e.kind in ("draft.built", "solution.saved") and e.payload.get("kind")),
                (created.payload.get("pack") if created else None))
    return ExperienceSession(
        visit=next(iter(visits)), workspace=next(iter(workspaces)), user=next((e.ids.user for e in ordered if e.ids.user), None),
        started=ordered[0].at, ended=ordered[-1].at, events=len(ordered), by_kind=dict(sorted(kinds.items())),
        conversations=sorted({e.ids.conversation for e in ordered if e.ids.conversation}),
        projects=sorted({e.ids.project for e in ordered if e.ids.project}),
        person_turns=kinds["chat.person"], agent_turns=kinds["chat.agent"], model_requests=kinds["model.request"], cost=cost,
        problem=(str(created.payload.get("problem"))[:300] if created and created.payload.get("problem") else None), kind=kind,
        refused=kinds["solution.refused"] + kinds["decision.refused"] + kinds["draft.refused"]
        + sum(1 for e in ordered if e.kind == "graph.move" and e.payload.get("allowed") is False)
        + sum(1 for e in ordered if e.kind == "tool.call" and e.payload.get("verdict") == "denied"),
        errors=kinds["error.raised"] + kinds["stage.failed"], outcome=outcome)


CONTRACTS.update({"experience_event": ExperienceEvent, "experience_session": ExperienceSession})  # their schemas are published
