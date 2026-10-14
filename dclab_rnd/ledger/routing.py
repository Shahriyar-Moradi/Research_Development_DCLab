"""Choosing a model per decision (package 22.6): which answerer serves each decision point, at which level.

The routing lives in the workspace's routing document (``models/routing.py``: the same file or PostgreSQL row the model
purposes use), under ``decisions``: per point, the answerer (``name@version``), its level (``dclab_core.ledger.LEVELS``:
0 shadow, 1 flag, 2 act with a check, 3 act), its bar (the accuracy below which it is demoted) and since when it serves.
``switch`` is the one way a person changes it: a reviewer or the owner, a reason, and an audit row in the same call.
Above shadow an answerer must have passed its sealed gate (22.3); to act (levels 2 and 3) it must also have been
measured in shadow: at least ``MIN_LABELS`` labelled disagreements with the rule, and a weighted accuracy on them at
least the rule's (``shadow.measure``). ``answer`` is what a caller asks: level 3 uses the answerer's value; level 2
uses it when it agrees with the rule and holds the decision for review when they differ (the rule's value stands
until a person decides); below level 2, and whenever the answerer abstains, fails or gives a value the point refuses,
the rule's. ``demote`` (DCLab's own, not a person's) reads the labels given since an answerer was routed, blind ones
weighted by their draw, and moves one below its bar back to shadow, audited: a demoted model falls back to the rule.

Ranking (``candidates``) is by sealed accuracy, then the weighted shadow accuracy, then cost; speed is not measured.
Only classifiers can be routed above shadow today: a model answerer (22.5) has no sealed verdict yet. Nothing in the
studio asks ``answer`` yet: the workflow's own decisions are 22.9's.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

REVIEWERS = ("reviewer", "owner")
LEVELS = (0, 1, 2, 3)
BAR = 0.9  # a routed answerer's accuracy on new labels below this sends it back to shadow
MIN_LABELS = 20  # labelled decisions needed to promote above flagging, and to judge a demotion
SYSTEM = "system:demotion"
_LOCK = threading.Lock()  # one routing change at a time in this process (the document is read, changed and written whole)


class DecisionRoutingError(ValueError):
    """A routing change that is refused; the reason says why."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _when(value: str) -> datetime:
    at = datetime.fromisoformat(str(value))
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def entry(doc: dict[str, Any], point: str) -> dict[str, Any]:
    return dict(((doc.get("decisions") or {}).get(point)) or {})


def passed() -> dict[str, dict[str, Any]]:
    """The answerers whose sealed gate (22.3) passed, by ``name@version``, with the verdict's figures."""
    import json

    from .shadow import GATE

    out = {}
    for path in sorted(GATE.glob("DMG-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("result") == "pass":
            out[f"{doc['model']['name']}@{doc['model']['version']}"] = {"point": doc["point"], "verdict": doc.get("verdict") or {},
                                                                       "gate": doc["gate"]["version"], "family": doc["model"].get("family")}
    return out


def candidates(point: str, measured: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """The answerers that may serve ``point`` above shadow, best first: those that passed their sealed gate, by their
    sealed accuracy, then their weighted accuracy on labelled disagreements in shadow, then cost (a fitted classifier
    answers on this machine for nothing). Speed is not measured."""
    out = []
    for name, info in passed().items():
        if info["point"] != point:
            continue
        verdict, shadow = info["verdict"], (measured or {}).get(name) or {}
        out.append({"answerer": name, "sealed_accuracy": (verdict.get("model") or {}).get("accuracy"),
                    "sealed_lower": (((verdict.get("bounds") or {}).get("model") or {}).get("accuracy") or [None, None])[1],
                    "shadow_weighted_accuracy_on_disagreements": shadow.get("weighted_accuracy_on_disagreements"),
                    "shadow_labelled_disagreements": shadow.get("labelled_disagreements"), "cost": "free (a fitted classifier, on this machine)"})
    return sorted(out, key=lambda c: (-(c["sealed_accuracy"] or 0), -(c["shadow_weighted_accuracy_on_disagreements"] or 0)))


def ready_to_act(name: str, measured: Optional[dict[str, Any]]) -> Optional[str]:
    """Why ``name`` may not act yet (None when it may): its shadow measurement must hold enough labelled disagreements
    and a weighted accuracy on them at least the rule's."""
    m = (measured or {}).get(name) or {}
    n = int(m.get("labelled_disagreements") or 0)
    if n < MIN_LABELS:
        return f"{name} has {n} labelled disagreements with the rule in shadow; it acts only after {MIN_LABELS} (blind labels add them)"
    mine, rule = m.get("weighted_right") or 0.0, m.get("weighted_rule_right") or 0.0
    if mine < rule:
        return f"{name} is right on fewer labelled disagreements than the rule in shadow (weighted {mine} against {rule})"
    return None


def switch(store: Any, audit_log: Any, point: str, answerer: Optional[str], level: int, role: Optional[str], reason: str = "",
           bar: float = BAR, measured: Optional[dict[str, Any]] = None, by_system: bool = False) -> dict[str, Any]:
    """One audited change of a point's answerer (None and level 0: the rule alone). Above shadow the answerer must have
    passed its sealed gate, and to act it must have been measured in shadow (``measured``: ``shadow.measure``); a
    reviewer or the owner decides, with a reason kept in the audit trail. ``by_system``: DCLab's own demotion."""
    from dclab_core.ledger import POINTS

    if point not in POINTS:
        raise DecisionRoutingError(f"unknown decision point {point!r}")
    if level not in LEVELS:
        raise DecisionRoutingError(f"level must be one of {LEVELS}")
    if not by_system and role not in REVIEWERS:
        raise PermissionError("Only a reviewer or the owner routes a decision to a model")
    reason = " ".join(str(reason or "").split())[:300]
    if not reason:
        raise DecisionRoutingError("Say why: the reason is kept in the audit trail")
    if answerer is not None and not isinstance(answerer, str):
        raise DecisionRoutingError("the answerer is a name@version")
    if answerer is None and level:
        raise DecisionRoutingError("a level above shadow needs an answerer")
    if not isinstance(bar, (int, float)) or not 0.5 <= float(bar) <= 1.0:
        raise DecisionRoutingError("the bar is an accuracy between 0.5 and 1")
    if level and not by_system:
        gates = passed()
        if answerer not in gates:
            raise DecisionRoutingError(f"{answerer} has not passed its sealed gate: it may answer in shadow only")
        if gates[answerer]["point"] != point:
            raise DecisionRoutingError(f"{answerer} was judged at {gates[answerer]['point']}, not {point}")
        if level >= 2:
            why = ready_to_act(answerer, measured)
            if why:
                raise DecisionRoutingError(why)
    with _LOCK:
        doc = store.get()
        current = entry(doc, point)
        new = {"answerer": answerer, "level": level, "bar": float(bar), "since": _now()} if answerer else {}
        keys = ("answerer", "level", "bar")
        if {k: current.get(k) for k in keys} == {k: new.get(k) for k in keys}:
            return doc  # the same setting again changes nothing
        decisions = dict(doc.get("decisions") or {})
        if new:
            decisions[point] = new
        else:
            decisions.pop(point, None)
        made = {"at": _now(), "decision_point": point, "from": {k: current.get(k) for k in keys} if current else None,
                "to": {k: new.get(k) for k in keys} if new else None, "by": SYSTEM if by_system else role, "reason": reason}
        saved = store.save({**doc, "decisions": decisions, "history": [*(doc.get("history") or []), made]})
    if audit_log is not None:
        from .. import audit

        audit.record(audit_log, "routing", "agent" if by_system else "human", "DCLab (automatic demotion)" if by_system else role.capitalize(),
                     move="set_decision_routing", status="allowed", args={"decision_point": point, "from": made["from"], "to": made["to"]},
                     message=reason)
    return saved


def answer(doc: dict[str, Any], point: str, state: dict[str, Any], options: list[Any], rule: Any,
           answerers: Optional[Iterable[Any]] = None) -> tuple[Any, str, int, list[Any], bool]:
    """The value a caller acts on: (value, used, level, answers, review). Level 3: the routed answerer's value. Level 2:
    its value when it agrees with the rule; when they differ the rule's value stands and ``review`` is True (a person
    decides). Below level 2, and when the answerer abstains, fails or gives a value the point refuses: the rule's
    (``rule`` its ``Answer``; None and ``used`` "none" when there is no rule)."""
    from dclab_core.answerers import _key
    from dclab_core.ledger import POINTS, answer_problem

    from . import shadow

    e = entry(doc, point)
    level, name = int(e.get("level") or 0), e.get("answerer")
    found = None
    if name:
        found = next((a for a in (shadow.answerers() if answerers is None else answerers) if f"{a.name}@{a.version}" == name), None)
    got = None
    if found is not None and level:
        try:
            got = found.answer(point, state, options)
        except Exception:  # noqa: BLE001 - a model that fails leaves the rule's answer
            got = None
        if got is not None and (got.value is None or answer_problem(POINTS[point], got.value)):
            got = None  # a value the point refuses is no answer
    answers = [a for a in (rule, got) if a is not None]
    ruled = (rule.value if rule is not None else None), ("rule" if rule is not None else "none")
    if got is None or level < 2:
        return ruled[0], ruled[1], level, answers, False
    if level == 2 and rule is not None and rule.value is not None and _key(rule.value) != _key(got.value):
        return ruled[0], ruled[1], level, answers, True  # act with a check: they differ, a person decides
    return got.value, got.by, level, answers, False


def demote(store: Any, audit_log: Any, records: Iterable[Any], labels: Iterable[Any], min_labels: int = MIN_LABELS) -> list[dict[str, Any]]:
    """Each routed answerer whose accuracy on the decisions labelled since it was routed (blind labels weighted by their
    draw; an exposed label only where it says the routed answer was wrong, as the label rule reads it) is below its bar,
    moved back to shadow by DCLab itself, audited; the rule answers again."""
    from dclab_core.answerers import _key
    from dclab_core.ledger import current_labels, effective_label

    doc = store.get()
    routed = {p: e for p, e in (doc.get("decisions") or {}).items() if e.get("level")}
    if not routed:
        return []
    by_record: dict[str, list[Any]] = {}
    for x in current_labels(list(labels)):  # a label another one supersedes no longer counts
        by_record.setdefault(x.record, []).append(x)
    records = [r for r in records if r.point in routed]
    out = []
    for point, e in routed.items():
        name, since = e["answerer"], _when(e.get("since") or "1970-01-01T00:00:00+00:00")
        right = total = 0.0
        n = 0
        for r in records:
            if r.point != point or _when(r.at) < since:
                continue
            mine = next((a for a in r.answers if a.answerer == name and a.value is not None), None)
            truth = effective_label(r, by_record.get(r.id, []))
            if mine is None or truth is None:
                continue
            weight = float(truth.weight or 1.0) if truth.source == "blind" else 1.0
            n += 1
            total += weight
            right += weight * (_key(mine.value) == _key(truth.value))
        bar = float(e.get("bar") or BAR)
        if n >= min_labels and total and right / total < bar:
            accuracy = round(right / total, 4)
            reason = f"weighted accuracy {accuracy} on {n} decisions labelled since it was routed, below its bar {bar}: demoted to shadow"
            switch(store, audit_log, point, name, 0, None, reason, bar=bar, by_system=True)
            out.append({"decision_point": point, "answerer": name, "accuracy": accuracy, "labelled": n})
    return out


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m dclab_rnd.ledger.routing [demote]``: the workspace's decision routing, or its demotion check."""
    import argparse
    import json
    from pathlib import Path

    from .. import audit
    from ..models.routing import open_routing
    from ..settings import Settings
    from . import open_ledger

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.ledger.routing")
    parser.add_argument("command", nargs="?", choices=["show", "demote"], default="show")
    args = parser.parse_args(argv)
    home = Path(Settings().agent_home)
    store = open_routing(home)
    if args.command == "demote":
        book = open_ledger(home)
        print(json.dumps(demote(store, audit.open_audit(home), book.records(), book.labels()), indent=1))
    else:
        print(json.dumps({"decisions": store.get().get("decisions") or {}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
