"""Evaluation of every answerer (package 25.8): precision, recall, F1 and accuracy, by decision point or twin step, against
the truth, with the counts behind each number.

DCLab decides with several answerers: its rules, its roles served by a model, its fitted decision classifiers, the
AI-first agent (Claude), and, in the main product, Jev. Each is measured here the same way, from the cases where the
truth is known:

* **a flag** (a yes/no point such as ``feature.leakage_suspect``): the confusion counts with ``True`` as the positive
  class, accuracy, precision, recall and F1 of the positive class;
* **a choice** (one of several values: a split, a family, a pack): accuracy, and precision, recall and F1 per class with
  their macro means over every class seen in the truth or the answers (scikit-learn's macro);
* **a set** (several items: the leak columns, the identifiers): precision, recall and F1 over the items, summed over the
  cases (micro), and accuracy as the share of cases whose whole set was right;
* **text** and **numbers** (a moment in words, a threshold): accuracy only (a match by key terms or within a tolerance).

An answerer that gave no answer on a case is counted in ``abstained`` and, for accuracy, as wrong (``coverage`` says how
often it answered); abstentions never enter precision and count against recall. Every number carries ``n``; accuracy
carries a Wilson interval (on the effective n when cases are weighted). :func:`from_twin_steps` reads twin steps (both
arms, each judged field; a person's pick is a preference, not a truth to score against, and is left to the twin matrix),
:func:`from_ledger` reads decision records with their labels: blind labels weighted by how they were drawn, a person's
correction of what they were shown left out (it exists only where the shown answer was wrong). No model, no store.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Kind = Literal["flag", "choice", "set", "text", "numeric"]


def wilson(right: int, n: int, z: float = 1.96) -> list[Optional[float]]:
    if not n:
        return [None, None]
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _ratio(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b else None


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


class ClassScore(Contract):
    label: str = Field(max_length=120)
    support: int = Field(ge=0, description="Cases whose truth is this class (items, for a set).")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


class Score(Contract):
    """One answerer at one decision point (or one field of a twin step), against one source of truth."""

    answerer: str = Field(max_length=160)
    by: str = Field(default="", max_length=40, description="rule, model_role, classifier, person, ai_first, system.")
    point: str = Field(max_length=120, description="A ledger point, or a twin step and its field ('home.pack:pack').")
    kind: Kind
    truth: str = Field(max_length=60, description="The source of the truth: planted, benchmark, blind, measured, a person's pick, a label.")
    n: int = Field(ge=0, description="Cases with a truth.")
    answered: int = Field(ge=0)
    abstained: int = Field(ge=0)
    coverage: Optional[float] = None
    accuracy: Optional[float] = None
    accuracy_interval: list[Optional[float]] = Field(default_factory=lambda: [None, None])
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion: dict[str, int] = Field(default_factory=dict, description="tp, fp, fn, tn for a flag or a set (tn only for a flag).")
    classes: list[ClassScore] = Field(default_factory=list, max_length=50)


class EvaluationBoard(Versioned):
    """Every answerer's scores, as the dashboard shows them."""

    schema_version: Literal["dclab_core.evaluation_board.v1"] = _version("evaluation_board")
    source: str = Field(max_length=300, description="Where the cases came from.")
    scores: list[Score] = Field(default_factory=list, max_length=2000)
    totals: dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------------------------------------- the metrics


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().casefold()
    return v


def _effective_n(weights: Sequence[float]) -> int:
    """Kish's effective sample size: what a weighted count is worth for an interval."""
    total, squares = sum(weights), sum(w * w for w in weights)
    return int(round(total * total / squares)) if squares else 0


def score(answerer: str, point: str, kind: Kind, truth_source: str, pairs: Sequence[tuple[Any, ...]], by: str = "",
          same: Any = None) -> Score:
    """``pairs`` are (answer, truth) or (answer, truth, weight) for the cases with a truth; an answer of None is an
    abstention. A weight (a blind label's sampling weight) weighs each count; the interval then reads the effective n.
    ``same(answer, truth)`` decides a text or numeric match.

    One rule for undefined ratios, the same for every kind (scikit-learn's ``zero_division=0`` where the truth has
    positives): precision is 0 when nothing was predicted but something was there to find, recall 0 when nothing was
    found; a ratio with nothing on either side is undefined (None). Abstentions never enter precision; they count against
    recall (a positive not found) and accuracy (a case not answered)."""
    rows = [(p[0], p[1], float(p[2]) if len(p) > 2 and p[2] is not None else 1.0) for p in pairs]
    if kind == "flag":  # a flag's answer is a flag: anything else is no answer
        rows = [(a if isinstance(a, bool) else None, t, w) for a, t, w in rows]
    n = len(rows)
    weights = [w for _, _, w in rows]
    total = sum(weights)
    answered = [(a, t, w) for a, t, w in rows if a is not None]
    eff = _effective_n(weights) if any(w != 1.0 for w in weights) else n
    out: dict[str, Any] = dict(answerer=answerer[:160], by=by[:40], point=point[:120], kind=kind, truth=truth_source[:60], n=n,
                               answered=len(answered), abstained=n - len(answered), coverage=_ratio(len(answered), n))

    def acc(right: float) -> None:
        out.update(accuracy=_ratio(right, total), accuracy_interval=wilson(int(round(eff * right / total)) if total else 0, eff))

    def pr(tp: float, predicted: float, positives: float) -> tuple[Optional[float], Optional[float]]:
        p = _ratio(tp, predicted) if predicted else (0.0 if positives else None)
        r = _ratio(tp, positives) if positives else None
        return p, r

    if kind == "flag":
        tp = sum(w for a, t, w in answered if a and bool(t))
        fp = sum(w for a, t, w in answered if a and not bool(t))
        tn = sum(w for a, t, w in answered if not a and not bool(t))
        positives = sum(w for _, t, w in rows if bool(t))
        fn = positives - tp  # a missed positive, answered or not
        acc(tp + tn)
        p, r = pr(tp, tp + fp, positives)
        out.update(precision=p, recall=r, f1=_f1(p, r), confusion={k: int(round(v)) for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn))})
    elif kind == "choice":
        right = sum(w for a, t, w in answered if _norm(a) == _norm(t))
        labels = sorted({str(_norm(t)) for _, t, _ in rows} | {str(_norm(a)) for a, _, _ in answered})  # sklearn's macro: the union
        classes = []
        for label in labels:
            tp = sum(w for a, t, w in answered if str(_norm(a)) == label and str(_norm(t)) == label)
            predicted = sum(w for a, _, w in answered if str(_norm(a)) == label)
            support = sum(w for _, t, w in rows if str(_norm(t)) == label)
            p, r = pr(tp, predicted, support)
            if r is None:  # a class only answered, never true: its recall is 0 in the macro (sklearn)
                r = 0.0
            classes.append(ClassScore(label=label[:120], support=int(round(support)), precision=p if p is not None else 0.0, recall=r,
                                      f1=_f1(p if p is not None else 0.0, r)))
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        acc(right)
        out.update(precision=mean([c.precision for c in classes]), recall=mean([c.recall for c in classes]), f1=mean([c.f1 for c in classes]),
                   classes=classes[:50])  # the macro over every class; only the stored list is cut
    elif kind == "set":
        tp = fp = fn = whole = 0.0
        for a, t, w in rows:
            truth = {str(_norm(x)) for x in (t or [])}
            got = {str(_norm(x)) for x in (a or [])} if a is not None else set()
            tp, fp, fn = tp + w * len(got & truth), fp + w * len(got - truth), fn + w * len(truth - got)
            whole += w * int(a is not None and got == truth)
        acc(whole)
        if answered and tp + fp == 0 and tp + fn == 0:  # nothing to find and nothing claimed, on cases it answered: perfect
            p = r = 1.0
        elif not answered:
            p = r = None
        else:
            p, r = pr(tp, tp + fp, tp + fn)
        out.update(precision=p, recall=r, f1=_f1(p, r), confusion={k: int(round(v)) for k, v in (("tp", tp), ("fp", fp), ("fn", fn))})
    else:  # text, numeric: a match or not
        match = same or (lambda a, t: _norm(a) == _norm(t))
        acc(sum(w for a, t, w in answered if match(a, t)))
    return Score(**out)


# ------------------------------------------------------------------------------------------------- from the records

_COMPARE_KIND = {"equal": "choice", "set": "set", "ranked": "choice", "numeric": "numeric", "text": "text", "mapping": "choice"}


def _value_for(kind: str, v: Any) -> Any:
    if kind == "choice" and isinstance(v, (list, dict)):
        import json

        return json.dumps(v, sort_keys=True, default=str)
    return v


def from_twin_steps(steps: Iterable[Any], source: str = "twin steps") -> EvaluationBoard:
    """Each arm of each judged twin step, field by field: the system and the AI-first agent scored on the same cases.
    A flag field (``home.ready:ready``) is scored as a flag; the answerer is the arm's name and who answered it. The
    system's answer to one step of one session counts once, however many AI answers were set beside it (a twin run's
    lane and its expert record). A person's pick is left out: it is the chosen side by construction."""
    from dclab_core import twin

    cases: dict[tuple[str, str, str, str, str], list[tuple[Any, Any]]] = defaultdict(list)
    kinds: dict[tuple[str, str, str, str, str], str] = {}
    tolerance: dict[str, float] = {}
    seen_system: set[tuple[str, ...]] = set()
    distinct: set[tuple[str, str]] = set()
    for st in steps:
        if st.truth is None or not st.truth.value or st.system.status == "skipped" or st.truth.source == "person_pick":
            continue
        spec = twin.spec(st.step)
        tolerance[st.step] = spec.tolerance
        for field, value in st.truth.value.items():
            compare = spec.fields[field]
            kind = "flag" if isinstance(value, bool) else _COMPARE_KIND[compare]
            distinct.add((st.session, f"{st.step}:{field}"))
            for arm in twin.ARMS:
                if arm == "system":  # the same answer to the same question in one session counts once
                    import json as _json

                    once = (st.session, st.step, field, _json.dumps(st.system.value.get(field), sort_keys=True, default=str), _json.dumps(value, sort_keys=True, default=str))
                    if once in seen_system:
                        continue
                    seen_system.add(once)
                answer = getattr(st, arm)
                got = answer.value.get(field) if answer.status == "answered" else None
                key = (arm, answer.answerer if arm == "ai_first" else "system", f"{st.step}:{field}", kind, st.truth.source)
                kinds[key] = compare
                cases[key].append((_value_for(kind, got), _value_for(kind, value)))
    scores = []
    for (arm, answerer, point, kind, truth_source), pairs in sorted(cases.items()):
        step = point.split(":", 1)[0]
        compare = kinds[(arm, answerer, point, kind, truth_source)]
        scores.append(score(answerer, point, kind, truth_source, pairs, by=arm,
                            same=lambda a, t, c=compare, s=step: twin.right(c, a, t, tolerance.get(s, 0.05))))
    return EvaluationBoard(source=source[:300], scores=scores, totals={"cases": len(distinct), "scores": len(scores)})


NUMERIC_POINTS = frozenset({"threshold.operating_point", "split.folds"})


def _point_kind(point: Any) -> Kind:
    if point.answer == "noul":
        return "flag"
    if point.answer == "score" or point.key in NUMERIC_POINTS:
        return "numeric"
    return "choice"


def _close(a: Any, t: Any) -> bool:
    try:
        return abs(float(a) - float(t)) <= 1e-6 + 0.01 * abs(float(t))
    except (TypeError, ValueError):
        return False


def from_ledger(records: Iterable[Any], labels: Sequence[Any], source: str = "the decision ledger") -> EvaluationBoard:
    """Every answer on every labelled decision record, by answerer and point, against the record's effective label
    (``ledger.effective_label``). A blind label weighs as it was drawn (``weight``); a person's correction of what they
    were shown is left out (it is accepted only where the shown answer was wrong, so it would score it wrong by
    construction). Shadow answers are scored too (that is what a shadow is for)."""
    from dclab_core.ledger import POINTS, effective_label

    by_record: dict[str, list[Any]] = defaultdict(list)
    for lab in labels:
        by_record[lab.record].append(lab)
    cases: dict[tuple[str, str, str, str], list[tuple[Any, Any, float]]] = defaultdict(list)
    distinct: set[str] = set()
    for rec in records:
        lab = effective_label(rec, by_record.get(rec.id, []))
        if lab is None or rec.point not in POINTS or lab.source == "user_decision_exposed":
            continue
        kind = _point_kind(POINTS[rec.point])
        weight = float(lab.weight) if lab.source == "blind" and getattr(lab, "weight", None) else 1.0
        distinct.add(rec.id)
        for answer in rec.answers:
            if answer.by == "person":
                continue
            got = answer.value if kind != "choice" or answer.value is None else str(answer.value)
            cases[(answer.answerer, answer.by, rec.point, lab.source)].append((got, lab.value if kind != "choice" else str(lab.value), weight))
    scores = [score(answerer, point, _point_kind(POINTS[point]), source_, pairs, by=by, same=_close)
              for (answerer, by, point, source_), pairs in sorted(cases.items())]
    return EvaluationBoard(source=source[:300], scores=scores, totals={"cases": len(distinct), "scores": len(scores)})


def merge(boards: Iterable[EvaluationBoard], source: str) -> EvaluationBoard:
    boards = list(boards)
    scores = [s for b in boards for s in b.scores]
    return EvaluationBoard(source=source[:300], scores=scores[:2000], totals={"cases": sum(int(b.totals.get("cases", 0)) for b in boards), "scores": len(scores)})


CONTRACTS["evaluation_board"] = EvaluationBoard

__all__ = ["Score", "ClassScore", "EvaluationBoard", "score", "from_twin_steps", "from_ledger", "merge", "wilson"]
