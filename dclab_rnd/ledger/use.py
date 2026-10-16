"""Decision models inside the workflow (package 22.9): where the studio decides, it asks the answerer routed for that
point (22.6) and acts on its answer only after the point's own validator accepts it.

``choose(point, state, options, rule_value, rule_name, check=...)`` returns the rule's value at once when nothing is
routed for the point (no model is loaded). When a model is routed: ``routing.answer`` gives the value to act on (level
1 flags beside the rule, 2 acts when it agrees or holds a disagreement for review, 3 acts), and a value from the model
goes through ``check`` first: a list of problems refuses it, the rule's value stands, and the refusal is recorded. A
model never moves the graph and never removes a caution the rules took: each caller's check says which way a model may
move (add a forbidden column, never free one; mark an identifier, never unmark one; choose among the runnable
experiments only; flag a tool, never replace the agent's).

``scope(doc)`` sets the routing for the code inside it (a measurement's arm, 22.9's suite); otherwise the workspace's
routing (``decide.workspace_routing``). ``choices()`` lists what was asked inside a ``recording()`` block, for a
measurement to count each model's contribution.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

_SCOPE: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar("decision_routing", default=None)
_SEEN: contextvars.ContextVar[Optional[list["Choice"]]] = contextvars.ContextVar("decision_choices", default=None)


@dataclass
class Choice:
    """What the studio acts on at one decision: the value, who it came from, the level, whether a person should look,
    and the validator's problems when it refused the model's value."""

    point: str
    subject: str
    value: Any
    used: str = "rule"
    level: int = 0
    review: bool = False
    refused: list[str] = field(default_factory=list)
    answerer: Optional[str] = None
    model_value: Any = None
    rule_value: Any = None
    state: Optional[dict[str, Any]] = None  # the state the model was shown, when the caller keeps it for the record

    @property
    def by_model(self) -> bool:
        return self.used not in ("rule", "none")

    @property
    def changed(self) -> bool:
        """The model's answer acted and differs from the rule's: what the model contributed, not an agreement."""
        from dclab_core.answerers import _key

        return self.by_model and _key(self.value) != _key(self.rule_value)


@contextlib.contextmanager
def scope(doc: Optional[dict[str, Any]]) -> Iterator[None]:
    token = _SCOPE.set(doc)
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextlib.contextmanager
def recording() -> Iterator[list[Choice]]:
    seen: list[Choice] = []
    token = _SEEN.set(seen)
    try:
        yield seen
    finally:
        _SEEN.reset(token)


def scoped() -> Optional[dict[str, Any]]:
    """The routing a scope set (a journey's arm), or None outside one."""
    return _SCOPE.get()


def routing_doc() -> dict[str, Any]:
    doc = _SCOPE.get()
    if doc is not None:
        return doc
    from ..decide import workspace_routing

    return workspace_routing()[0]


ENGINE_POINTS = ("column.semantic_role", "experiment.next")  # what a stage run decides with a routed model


def fingerprint(points: tuple[str, ...] = ENGINE_POINTS) -> Optional[str]:
    """The routing a stage run decides with, as a short hash (None when nothing is routed for those points): a run's
    input (19.5), so a change of routing makes the stages stale and a replay or an export refuse to pretend."""
    import hashlib
    import json

    from . import routing

    doc = routing_doc()
    entries = {p: {k: routing.entry(doc, p).get(k) for k in ("answerer", "level")} for p in points if routed(p, doc)}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16] if entries else None


def routed(point: str, doc: Optional[dict[str, Any]] = None) -> bool:
    from . import routing

    e = routing.entry(routing_doc() if doc is None else doc, point)
    return bool(e.get("answerer")) and int(e.get("level") or 0) > 0


def choose(point: str, state: dict[str, Any], options: list[Any], rule_value: Any, rule_name: str, *, subject: str = "",
           check: Optional[Callable[[Any], list[str]]] = None, doc: Optional[dict[str, Any]] = None, answerers: Any = None) -> Choice:
    """The value to act on at ``point``: the rule's unless a routed model's answer passes ``check`` (see the module)."""
    doc = routing_doc() if doc is None else doc
    if not routed(point, doc):
        return Choice(point, subject, rule_value, rule_value=rule_value)
    from dclab_core.ledger import Answer, check_state

    from . import routing

    state = {k: v for k, v in (state or {}).items() if v is not None}
    if check_state(state):  # a state the ledger would refuse is never shown to a model
        return Choice(point, subject, rule_value, rule_value=rule_value)
    rule = Answer(by="rule", answerer=rule_name, value=rule_value) if rule_value is not None else None
    value, used, level, answers, review = routing.answer(doc, point, state, options, rule, answerers)
    model = next((a for a in answers if a.by != "rule"), None)
    choice = Choice(point, subject, value, used, level, review, answerer=model.answerer if model else None,
                    model_value=model.value if model else None, rule_value=rule_value)
    if choice.by_model and check is not None:
        problems = list(check(value) or [])
        if problems:  # the validator first: the rule's value stands
            choice = Choice(point, subject, rule_value, "rule", level, review, problems, choice.answerer, choice.model_value, rule_value)
            from .. import experience

            experience.record("decision.refused", "system", {"point": point, "choice": str(value)[:120], "rule": "validator",
                                                             "reason": f"{choice.answerer}: " + "; ".join(problems)[:280]})
    seen = _SEEN.get()
    if seen is not None:
        seen.append(choice)
    return choice
