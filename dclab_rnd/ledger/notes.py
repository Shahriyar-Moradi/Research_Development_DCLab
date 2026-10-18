"""Every user is a teacher (package 29.10): where a point's student whispered something other than the rule on a project's
decision, a note asks the person which is right.

``pending`` lists a project's decisions where the point's current student (the register's card) disagrees with the rule
in shadow and this person has not answered yet: the question, what it is about, the rule's answer and the student's,
the options. A decision of the blind draw is never a note (it belongs to the blind queue of 29.8, whose labels may count
toward acting: one a person saw both answers of must not come back there), nor one that already has a blind label (it
outranks any answer here). ``answer`` writes the person's choice as a label given having seen both (``user_decision_exposed``, with
``shown`` naming the two answers: dclab_core.ledger v5), superseding their own earlier answer on the decision. Nothing
decided changes: the record, the project, its solution and its stages are never touched. A person who saw the answers
may demote a student, never promote it: the measurement keeps these labels apart from what acting reads, and demotion
counts them only where they say the student was wrong.
"""

from __future__ import annotations

from typing import Any, Optional

LIMIT = 50  # notes a page shows at once


def _rule(record: Any) -> Any:
    return next((a for a in record.answers if a.by == "rule" and a.value is not None), None)


def _student(record: Any, name: Optional[str]) -> Any:
    from dclab_core.answerers import SHADOW, _key

    rule = _rule(record)
    if rule is None or not name:
        return None
    return next((a for a in record.answers if a.source == SHADOW and a.answerer == name and a.value is not None
                 and _key(a.value) != _key(rule.value)), None)


def _current(current: Optional[dict[str, str]]) -> dict[str, str]:
    if current is not None:
        return current
    from .labels import _students

    return _students()


def _mine(store: Any, person: str) -> dict[str, Any]:
    """This person's current exposed labels, by record."""
    from dclab_core.ledger import current_labels

    return {x.record: x for x in current_labels(list(store.labels())) if x.source == "user_decision_exposed" and x.by == person}


def _open(records: list[Any], labels: list[Any], person: str, current: dict[str, str]) -> dict[str, list[tuple[Any, Any]]]:
    """By project: the newest decision per point and subject where the point's student disagrees with the rule, not in
    the blind draw, without a blind label and not yet answered by ``person``, with the student's answer."""
    from dclab_core import ledger as contract

    blind = {x.record for x in labels if x.source == "blind"}
    answered = {x.record for x in contract.current_labels(labels) if x.source == "user_decision_exposed" and x.by == person}
    latest: dict[tuple[str, str, str], Any] = {}
    for r in records:
        if not r.project:
            continue
        key = (r.project, r.point, r.subject.key)
        if key not in latest or (r.at, r.id) > (latest[key].at, latest[key].id):
            latest[key] = r
    out: dict[str, list[tuple[Any, Any]]] = {}
    for r in sorted(latest.values(), key=lambda r: (r.point, r.subject.key)):
        if r.id in blind or r.id in answered or contract.blind_draw(r, "blind-v1"):
            continue
        student = _student(r, current.get(r.point))
        if student is not None:
            out.setdefault(r.project, []).append((r, student))
    return out


def counts(store: Any, person: str, current: Optional[dict[str, str]] = None) -> dict[str, int]:
    """Every project's number of open notes for ``person``, from one read of the ledger (Home's inbox)."""
    if store is None:
        return {}
    found = _open(list(store.records()), list(store.labels()), person, _current(current))
    return {project: len(items) for project, items in found.items()}


def pending(store: Any, project: str, person: str, current: Optional[dict[str, str]] = None, limit: int = LIMIT) -> dict[str, Any]:
    """The project's open notes for ``person``: the newest decision per point and subject where the point's student
    disagrees with the rule (see the module's note for the ones left out)."""
    from dclab_core.ledger import POINTS

    if store is None:
        return {"items": [], "total": 0}
    found = _open(list(store.records(project=project)), list(store.labels()), person, _current(current)).get(project, [])
    items = []
    for r, student in found:
        rule, point = _rule(r), POINTS[r.point]
        options = [str(o) for o in r.options] or (["yes", "no"] if point.answer == "noul" else [])
        items.append({"record": r.id, "point": r.point, "question": point.about, "subject": r.subject.model_dump(),
                      "rule": {"answerer": rule.answerer, "value": rule.value}, "student": {"answerer": student.answerer, "value": student.value},
                      "options": options, "answer": point.answer})
    return {"items": items[:max(0, int(limit))], "total": len(items)}


def answer(store: Any, project: str, record_id: str, value: Any, person: str, current: Optional[dict[str, str]] = None) -> Any:
    """The person's choice on a note, as an exposed label naming what they saw; KeyError when the decision is not this
    project's, ValueError when it is no longer a disagreement or the value is not one of its options."""
    from dclab_core.ledger import POINTS

    from . import label

    record = next((r for r in store.records(project=project) if r.id == record_id), None) if store is not None else None
    if record is None:
        raise KeyError("no such decision in this project")
    student = _student(record, _current(current).get(record.point))
    if student is None:
        raise ValueError("the student agrees with the rule here, or is no longer this point's: there is nothing to answer")
    from dclab_core.ledger import blind_draw

    if blind_draw(record, "blind-v1") or any(x.source == "blind" for x in store.labels(record=record.id)):
        raise ValueError("this decision is the blind queue's: it is labelled without seeing the answers")
    point = POINTS[record.point]
    if point.answer == "noul":
        said = {"yes": True, "no": False, True: True, False: False}.get(value if isinstance(value, bool) else str(value).strip().lower())
    else:
        said = next((o for o in record.options if str(o) == str(value)), None)
    if said is None:
        raise ValueError("the answer is one of the decision's options")
    earlier = _mine(store, person).get(record.id)
    out = label(record, said, "user_decision_exposed", person, ledger=store, shown=[_rule(record).answerer, student.answerer],
                supersedes=earlier.id if earlier is not None else None)
    if out is None:
        raise ValueError("the label was refused")
    return out


__all__ = ["LIMIT", "counts", "pending", "answer"]
