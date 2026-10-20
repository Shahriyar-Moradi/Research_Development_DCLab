"""Labels from every source (package 19.8): what was right, written as superseding records, never edits.

- **benchmark**: planted truth. A replica declares its traps (``facts``: the leaky columns, the entity, the time
  column, the identifiers); every decision about one of those columns on that replica gets the planted answer. Only
  what was planted is labelled: a column a replica does not name is unknown, never assumed safe.
- **measured**: a number decided it (the leakage stage's apparent lift for a forbidden column, a random split's
  optimism over a time split) — written by ``capture`` where the number is measured.
- **user_decision_exposed**: a person who saw the rule's answer and chose otherwise (a saved solution, an override);
  written by ``capture``. It may demote, never promote (``dclab_core.ledger.effective_label``).
- **blind**: a reviewer shown only the state, on a sample drawn with known probabilities (:func:`next_blind`).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from . import current_ledger, label

# a replica's facts → the point each planted trap is about, and the planted answer (the replicas name them differently)
TRUTH_POINTS = {"leaky_columns": ("feature.leakage_suspect", True), "identifiers": ("column.is_identifier", True),
                "entity": ("column.group", True), "time": ("column.time", True)}
FACT_NAMES = {"leaky_columns": ("leaky_columns", "leak", "leaks"), "identifiers": ("identifiers", "identifier"),
              "entity": ("entity", "group"), "time": ("time", "time_column")}


def truth_of(facts: dict[str, Any] | None) -> dict[str, list[str]]:
    """A replica's planted traps by kind of trap, each as a list of column names, whatever name the replica gives it."""
    out: dict[str, list[str]] = {}
    for key, names_of in FACT_NAMES.items():
        names: list[str] = []
        for name in names_of:
            value = (facts or {}).get(name)
            names += [value] if isinstance(value, str) else [v for v in value if isinstance(v, str)] if isinstance(value, (list, tuple)) else []
        if names:
            out[key] = list(dict.fromkeys(names))
    return out


def from_replica(project_id: str, facts: dict[str, Any] | None, replica: str, ledger: Any = None) -> int:
    """Benchmark labels for the project's decisions about the replica's planted columns. Returns how many."""
    store = ledger if ledger is not None else current_ledger()
    truth = truth_of(facts)
    if store is None or not truth:
        return 0
    written = 0
    have = {(x.record, x.source) for x in store.labels()}
    for key, names in truth.items():
        point, value = TRUTH_POINTS[key]
        for record in store.records(project=project_id, point=point):
            if record.subject.key in names and (record.id, "benchmark") not in have:
                written += bool(label(record, value, "benchmark", f"replica:{replica}", ledger=store,
                                      evidence={"kind": "planted_truth", "ref": f"{replica}:{key}:{record.subject.key}"[:200]}))
    return written


def next_blind(ledger: Any = None, seed: str = "blind-v1", point: Optional[str] = None, skip: Sequence[str] = ()) -> Optional[dict[str, Any]]:
    """The next decision for a blind label: drawn with its stratum's known probability, not yet labelled blind; what
    the reviewer sees (``dclab_core.ledger.blind_view``) and the weight its label will carry. ``skip``: decisions the
    reviewer could not tell from what they saw; they stay in the draw, unlabelled, and count as drawn and not labelled
    (a guess would be worse than a gap: the gap is reported, a guess is not)."""
    from dclab_core import ledger as contract

    store = ledger if ledger is not None else current_ledger()
    if store is None:
        return None
    done = {x.record for x in store.labels() if x.source == "blind"}
    skipped = set(skip)
    drawn = [r for r in store.records(point=point) if r.id not in done and r.id not in skipped and contract.blind_draw(r, seed)]
    # 22.4: the decisions where answers differ (a shadow classifier against the rule) are labelled first; the draw and
    # its weights are unchanged, only the order in which a reviewer meets them
    drawn.sort(key=lambda r: contract.stratum(r) != "disagree")
    if drawn:
        record = drawn[0]
        return {**contract.blind_view(record), "weight": round(1 / contract.blind_probability(record), 6), "seed": seed}
    return None


def _disagreement(record: Any, student: Optional[str]) -> bool:
    """29.8: whether the point's current student (``name@version``, the one the register counts) disagrees with the rule's
    answer on this decision, in shadow. Another version, or a workspace's own student, does not count here: a label of it
    would count toward nothing the register shows (the review of 29.8)."""
    from dclab_core.answerers import SHADOW, _key

    rule = next((a for a in record.answers if a.by == "rule" and a.value is not None), None)
    if rule is None or not student:
        return False
    return any(a.source == SHADOW and a.answerer == student and a.value is not None and _key(a.value) != _key(rule.value) for a in record.answers)


def _students() -> dict[str, str]:
    """Each point's current student, as the register names it (its card's ``name@version``)."""
    from ..extensions import optional

    register = optional("students")  # the students are the full edition's
    return {} if register is None else {p: c["answerer"] for p, c in register.students().items() if c.get("answerer")}


def queue(ledger: Any = None, point: Optional[str] = None, limit: int = 50, seed: str = "blind-v1", skip: Sequence[str] = (),
          current: Optional[dict[str, str]] = None, order: str = "fair") -> dict[str, Any]:
    """29.8: the disagreements worth a person's look: decisions where the point's current student (in shadow) disagrees
    with the rule, not yet labelled blind, each shown blind (``blind_view``: the question, the subject, the state and the
    options; never an answer, nor whose it was). Only decisions of the blind draw are queued, so a label keeps the weight
    of how its decision was drawn; and they come in an order drawn from the seed, never the student's surest first: any
    part of the queue a person labels is then a fair sample of the drawn disagreements, so the figures a student's
    promotion rests on are not its best cases (the review of 29.8). ``current``: each point's student (default: the
    register's cards); ``points``: how many each point has waiting. ``order="teach"`` (29.11): every decision a point's
    student answered outside the blind draw and without a truth yet, the student's least sure answer and a decision drawn
    at random in turn (see ``_teach`` for what its measurement showed: not passed); its labels are written without a
    weight (``blind_label(taught=True)``): they train, and never estimate."""
    if order == "teach":
        return _teach(ledger, point, limit, seed, skip, current)
    import hashlib

    from dclab_core import ledger as contract

    store = ledger if ledger is not None else current_ledger()
    if store is None:
        return {"items": [], "points": {}, "seed": seed}
    current = _students() if current is None else current
    done = {x.record for x in store.labels() if x.source == "blind"}
    skipped = set(skip)
    waiting: list[tuple[str, Any]] = []
    counts: dict[str, int] = {}
    for r in store.records():
        if r.id in done or not contract.blind_draw(r, seed) or not _disagreement(r, current.get(r.point)):
            continue
        counts[r.point] = counts.get(r.point, 0) + 1
        if (point is None or r.point == point) and r.id not in skipped:
            waiting.append((hashlib.sha256(f"{seed}:queue:{r.id}".encode()).hexdigest(), r))
    waiting.sort(key=lambda x: x[0])
    items = [{**contract.blind_view(r), "weight": round(1 / contract.blind_probability(r), 6), "seed": seed} for _, r in waiting[:max(0, int(limit))]]
    return {"items": items, "points": dict(sorted(counts.items())), "seed": seed}


def _teach(ledger: Any, point: Optional[str], limit: int, seed: str, skip: Sequence[str], current: Optional[dict[str, str]]) -> dict[str, Any]:
    """29.11: the teaching order (see ``queue``): the student's least sure answer and a decision drawn at random, in turn
    (``dclab_core.active.select_mixed``), chosen within each point and the points taken in turn (confidences of points with
    different numbers of options are not compared). Measured on invented tables (ALQ-002, a second rule written after the
    first failed, on the same dev rows): it did not pass its gate (split.strategy +0.062 on 5 of 5 seeds, data.cases_table
    +0.005, task.framing -0.004); the first rule, a disagreement first (ALQ-001), taught less than random labels. A decision
    of the blind draw is left to the fair queue (a taught label there would take it out of the sample a promotion reads),
    and one that already has a truth (a benchmark, a measurement, a blind label) is not asked again."""
    from dclab_core import ledger as contract
    from dclab_core.active import select_mixed
    from dclab_core.answerers import SHADOW

    store = ledger if ledger is not None else current_ledger()
    if store is None:
        return {"items": [], "points": {}, "seed": seed, "order": "teach"}
    current = _students() if current is None else current
    known = {x.record for x in store.labels() if x.source in ("blind", "benchmark", "measured")}  # a truth already there
    skipped = set(skip)
    by_point: dict[str, list[tuple[Any, Any]]] = {}
    counts: dict[str, int] = {}
    for r in store.records():
        name = current.get(r.point)
        student = next((a for a in r.answers if a.source == SHADOW and a.answerer == name and a.value is not None), None) if name else None
        if student is None or not contract.blind_eligible(r) or r.id in known or contract.blind_draw(r, seed):
            continue
        counts[r.point] = counts.get(r.point, 0) + 1
        if r.id not in skipped and (point is None or r.point == point):
            by_point.setdefault(r.point, []).append((r, student))
    lists = []
    for p in sorted(by_point):  # within each point: one least sure, one drawn at random, in turn
        pool = sorted(by_point[p], key=lambda x: x[0].id)  # a stable pool: the seed decides the random half
        picked = select_mixed([{"confidence": s.confidence} for _, s in pool], max(0, int(limit)), f"{seed}:teach:{p}")
        unsure, drawn = picked[: len(picked) // 2], picked[len(picked) // 2:]
        order = [i for pair in zip(unsure, drawn) for i in pair] + unsure[len(drawn):] + drawn[len(unsure):]
        lists.append([pool[i][0] for i in order])
    chosen: list[Any] = []  # the points in turn, each in its own order
    for i in range(max((len(x) for x in lists), default=0)):
        chosen += [x[i] for x in lists if i < len(x)]
    items = [{**contract.blind_view(r), "weight": None, "taught": True, "seed": seed} for r in chosen[:max(0, int(limit))]]
    return {"items": items, "points": dict(sorted(counts.items())), "seed": seed, "order": "teach"}


def blind_label(record_id: str, value: Any, by: str, seed: str = "blind-v1", ledger: Any = None, taught: bool = False) -> Any:
    """A blind reviewer's label for a drawn decision, with its weight; refused for a decision not in the draw. ``taught``
    (29.11): a label asked for by the teaching order, on any decision: no weight, a note saying it was chosen to teach
    (it trains a student, and never estimates one's accuracy)."""
    from dclab_core import ledger as contract

    store = ledger if ledger is not None else current_ledger()
    record = next((r for r in store.records() if r.id == record_id), None) if store is not None else None
    if record is None:
        raise KeyError("no such decision")
    if not taught and not contract.blind_draw(record, seed):
        raise ValueError("this decision is not in the blind draw: its weight would be unknown")
    if taught and not contract.blind_eligible(record):
        raise ValueError("this decision cannot be answered blind: its state names the answer, or it has no options")
    if taught and contract.blind_draw(record, seed):
        raise ValueError("this decision is in the blind draw: label it from the fair sample, where its label carries a weight")
    if value is None:
        raise ValueError("a blind label says what was right: a value")
    if any(x.source == "blind" and x.record == record_id and x.by == by for x in store.labels(record=record_id)):
        raise ValueError("this reviewer has labelled this decision already: one blind label each, or its weight counts twice")
    if taught:
        from dclab_core.active import TEACH_NOTE

        out = label(record, value, "blind", by, weight_note=TEACH_NOTE, ledger=store)
    else:
        out = label(record, value, "blind", by, weight=round(1 / contract.blind_probability(record), 6), ledger=store)  # as the page showed it
    if out is None:
        raise ValueError("the label was refused: the value is not an answer at this point")
    return out


def summary(ledger: Any = None) -> dict[str, Any]:
    """Counts for the page: decisions by point and kind, who decided, and labels by source."""
    from collections import Counter

    from dclab_core import ledger as contract

    store = ledger if ledger is not None else current_ledger()
    if store is None:
        return {"decisions": 0, "labels": 0, "by_point": {}, "outcomes": {}, "by_used": {}, "by_kind": {}, "by_source": {}, "drawn": 0, "shadow": {}}
    records = store.records()
    labels = store.labels()
    from . import shadow

    from dclab_core import outcomes as oc

    return {"decisions": len(records), "labels": len(labels), "by_point": dict(Counter(r.point for r in records).most_common()),
            "outcomes": oc.by_point(records, store.outcomes()) if hasattr(store, "outcomes") else {},  # 23.5: turned out well or badly, per point
            "by_used": dict(Counter(r.used for r in records)), "by_kind": dict(Counter(r.kind for r in records)),
            "by_source": dict(Counter(x.source for x in labels)), "drawn": sum(1 for r in records if contract.blind_draw(r)),
            "shadow": shadow.measure(records, labels) if shadow.on() else {}}  # 22.4: each classifier beside the rule
