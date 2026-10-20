"""One interface for every decision model (package 22.0).

A :class:`DecisionAnswerer` answers a decision point from its state (the ledger's bands, never a table's value) and
its options: a value in the point's vocabulary with a confidence, or ``None`` (it abstains). Every kind of model
answers through it, each registered with its name, version and the points it answers:

* :class:`RecordedRule`: the rule that made the decision, as the ledger recorded it (today's code);
* :class:`ScriptedAnswerer`: answers from a table, for tests and replays;
* :class:`ClassifierAnswerer`: a classical classifier trained on decision rows (the state's bands and name tokens);
* a model through a host's gateway, or a hosted typed-judgment model: a host implements the protocol.

:func:`compare` runs answerers on the same recorded decisions and says, for each, how often it answered, how often it
agreed with the rule, and how often it was right where a label says what was right (``ledger.effective_label``).
:func:`with_answers` returns a record with every answerer's answer beside the rule's, the shape the ledger keeps
(``by`` says which kind answered; a shadow answer is recorded, never used).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Protocol, runtime_checkable

from dclab_core.ledger import POINTS, Answer, DecisionLabel, DecisionRecord, effective_label

Scalar = Any


@runtime_checkable
class DecisionAnswerer(Protocol):
    name: str
    version: str
    by: str  # rule | model_role | classifier | person (the ledger's vocabulary)
    points: frozenset[str]

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Optional[Answer]: ...


REGISTRY: dict[str, DecisionAnswerer] = {}
KINDS = ("rule", "model_role", "classifier")  # who may answer as a model of a decision (a person answers in the ledger)
SHADOW = "shadow"  # an answer recorded beside the rule's, never shown and never used


def register(answerer: DecisionAnswerer) -> DecisionAnswerer:
    """Register an answerer under ``name@version``: its points and kind are what a host routes by (22.6). A different
    answerer under a name and version already taken is refused: a refit is a new version."""
    if not isinstance(answerer, DecisionAnswerer):
        raise TypeError("an answerer has name, version, by, points and answer(point, state, options)")
    if answerer.by not in KINDS:
        raise ValueError(f"by is one of {', '.join(KINDS)}")
    unknown = set(answerer.points) - set(POINTS)
    if unknown:
        raise ValueError(f"not decision points: {', '.join(sorted(unknown))}")
    key = f"{answerer.name}@{answerer.version}"
    if key in REGISTRY and REGISTRY[key] is not answerer:
        raise ValueError(f"{key} is taken: register a refit under a new version")
    REGISTRY[key] = answerer
    return answerer


class RecordedRule:
    """Today's code: the rule's answer the ledger recorded with each decision (what a classifier is compared with)."""

    by = "rule"

    def __init__(self, points: Iterable[str] = (), name: str = "recorded_rule", version: str = "1"):
        self.name, self.version, self.points = name, version, frozenset(points or POINTS)

    def of(self, record: DecisionRecord) -> Optional[Answer]:
        """The rule's answer: the first with a value (a rule that abstained first and answered later answered)."""
        rules = [a for a in record.answers if a.by == "rule"]
        return next((a for a in rules if a.value is not None), rules[0] if rules else None)

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Optional[Answer]:
        return None  # a recorded rule answers through ``of(record)``: its state alone does not say what the code saw


class ScriptedAnswerer:
    """Answers from a function of (point, state, options), or a fixed value: for tests and replays."""

    def __init__(self, name: str, answer: Any, points: Iterable[str], version: str = "1", by: str = "model_role",
                 confidence: float = 1.0):
        self.name, self.version, self.by, self.points, self._answer, self._confidence = name, version, by, frozenset(points), answer, confidence

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Optional[Answer]:
        value = self._answer(point, state, options) if callable(self._answer) else self._answer
        if value is None:
            return None
        return Answer(by=self.by, answerer=f"{self.name}@{self.version}", value=value, confidence=self._confidence)


NAMES = ("column", "target")  # a name enters only through its tokens: never whole (a model would learn projects' names)


def features(state: Mapping[str, Any], exclude: Iterable[str] = ()) -> dict[str, float]:
    """A state as features: each band as a one-hot (``field=band``), each name token, each number as itself (scaled
    when fitted). ``exclude``: fields a label is derived from (a measured label's own measure never enters)."""
    out: dict[str, float] = {}
    skip = set(exclude)
    for field, value in state.items():
        if field in skip:
            continue
        if isinstance(value, bool):
            out[f"{field}={value}"] = 1.0
        elif isinstance(value, (int, float)):
            out[field] = float(value)
        elif isinstance(value, str):
            if field not in NAMES:
                out[f"{field}={value}"] = 1.0
        elif isinstance(value, list):
            for token in value[:16]:
                out[f"{field}:{token}"] = 1.0
    return out


def contributions(model: Any, state: Mapping[str, Any], cls: Any, exclude: Iterable[str] = ()) -> list[tuple[str, float]]:
    """29.13: how much each state field moved a classifier toward ``cls``: the probability of ``cls`` with the state as it
    is, less the probability with that field left out (occlusion; any model). Strongest first; a field whose absence
    raises the probability moved it away (a negative number)."""
    skip = set(exclude)
    classes = list(model.classes_)
    if cls not in classes:
        return []
    k = classes.index(cls)
    full = float(model.predict_proba([features(state, skip)])[0][k])
    fields = [f for f in state if f not in skip]
    if not fields:
        return []
    rows = [features({g: v for g, v in state.items() if g != f}, skip) for f in fields]
    probs = model.predict_proba(rows)
    out = [(f, round(full - float(p[k]), 4)) for f, p in zip(fields, probs)]
    return sorted(out, key=lambda x: -abs(x[1]))


def explain(answerer: Any, state: Mapping[str, Any], cls: Any, top: int = 3) -> Optional[str]:
    """The two or three state fields that moved an answer most, as its reason: ``"moved by: uniqueness +0.31, ..."``."""
    try:
        moved = [x for x in contributions(answerer.model, state, cls, getattr(answerer, "exclude", ())) if abs(x[1]) >= 0.01][:top]
    except Exception:  # noqa: BLE001 - an answer without its reason is still the answer
        return None
    return ("moved by: " + ", ".join(f"{f} {d:+.2f}" for f, d in moved))[:600] if moved else None


class ClassifierAnswerer:
    """A classical classifier for one decision point: logistic regression on the state's features, trained on decision
    rows with labels; it abstains below ``abstain_below`` confidence."""

    by = "classifier"

    def __init__(self, point: str, name: str = "classifier", version: str = "1", abstain_below: float = 0.0, C: float = 1.0,
                 exclude: Iterable[str] = ()):
        self.point, self.name, self.version, self.points = point, name, version, frozenset({point})
        self.abstain_below, self.C, self.exclude = abstain_below, C, tuple(exclude)
        self.model: Any = None
        self.trained_on = 0

    def fit(self, rows: Iterable[tuple[Mapping[str, Any], Any]]) -> "ClassifierAnswerer":
        """``rows``: (state, the label's value). The numbers are scaled on the training rows only (a lift of 0.02 and
        a count of 2,000 weigh alike); fitted only when two values or more are seen."""
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MaxAbsScaler

        rows = [(features(state, self.exclude), value) for state, value in rows if value is not None]
        if len({_key(v) for _, v in rows}) < 2:
            raise ValueError("a classifier needs at least two labelled values")
        self.model = make_pipeline(DictVectorizer(sparse=True), MaxAbsScaler(), LogisticRegression(C=self.C, max_iter=1000))
        self.model.fit([f for f, _ in rows], [_key(v) for _, v in rows])
        self._values = {_key(v): v for _, v in rows}
        self.trained_on = len(rows)
        return self

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Optional[Answer]:
        """The likeliest value among the decision's own options (renormalised), or None: untrained, another point, no
        option it knows, or below its confidence."""
        if self.model is None or point != self.point:
            return None
        proba = self.model.predict_proba([features(state, self.exclude)])[0]
        classes = list(self.model.classes_)
        allowed = {_key(_option(o)) for o in options} if options else set(classes)
        keep = [i for i, c in enumerate(classes) if c in allowed]
        if not keep:
            return None
        total = float(sum(proba[i] for i in keep)) or 1.0
        best = max(keep, key=lambda i: proba[i])
        confidence = float(proba[best]) / total
        if confidence < self.abstain_below:
            return None
        why = explain(self, state, classes[best]) if getattr(self, "explain", False) else None
        return Answer(by="classifier", answerer=f"{self.name}@{self.version}", value=self._values.get(classes[best], classes[best]),
                      raw=round(confidence, 6), confidence=round(confidence, 4), reason=why)


class ModelAnswerer:
    """A language model as an answerer (22.5): it is asked through an injected ``Advisor`` (the host's gateway) with the
    decision's question, state (bands and name tokens, never a value) and options, and answers one option. An answer
    outside the options, or no answer, is an abstention. ``name`` and ``version`` say which model and prompt."""

    by = "model_role"

    def __init__(self, advisor: Any, points: Iterable[str], name: str = "model", version: str = "1", purpose: str = "decision_answer"):
        points = frozenset(points)
        unknown = sorted(p for p in points if p not in POINTS)
        if unknown:
            raise ValueError(f"not decision points: {', '.join(unknown)}")
        # 29.9: every point's state is bands, counts and codes (dclab_core.ledger.STATE_FIELDS), never a table's value: the
        # purpose may see any point's, not only a column's
        self.advisor, self.points, self.name, self.version, self.purpose = advisor, points, name, version, purpose
        self.errors = 0  # requests that failed (a cut or broken reply): each an abstention, counted

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Optional[Answer]:
        from dclab_core.ledger import POINTS

        try:  # the decision sets' export: the full edition's (the open-source edition has no tuned decision model to ask)
            from dclab_core.decision_export import request
        except ModuleNotFoundError as exc:
            if exc.name != "dclab_core.decision_export":
                raise
            return None

        if point not in self.points:
            return None
        from dclab_core.ledger import check_state

        if check_state(dict(state)):  # 29.9: only a state the ledger would keep (bands, counts, codes) is ever sent: else no answer
            return None
        evidence = request(point, state, options)  # what a tuned model was trained on, word for word
        try:
            reply = self.advisor.ask(self.purpose, evidence)
        except Exception:  # noqa: BLE001 - a failed request is an abstention, and counted
            self.errors += 1
            return None
        if not isinstance(reply, Mapping):
            return None
        raw = reply.get("answer", "")
        said = ("yes" if raw else "no") if isinstance(raw, bool) else str(raw).strip().lower()  # a JSON true is a yes
        chosen = next((o for o in options if str(o).strip().lower() == said), None)
        if chosen is None:
            return None
        reason = str(reply.get("reason") or "")[:300] or None
        return Answer(by="model_role", answerer=f"{self.name}@{self.version}", value=_option(chosen), reason=reason)


def _key(value: Any) -> str:
    """One spelling per value: True and "True" are the same answer; a number by its shortest form."""
    return str(value) if not isinstance(value, float) else f"{value:.12g}"


def _option(option: Any) -> Any:
    """An option as the ledger records it, as its value: a yes/no point's "yes" and "no" are True and False."""
    return {"yes": True, "no": False}.get(option, option) if isinstance(option, str) else option


def with_answers(record: DecisionRecord, answerers: Iterable[DecisionAnswerer]) -> DecisionRecord:
    """The record with each answerer's answer beside the ones it has: a shadow (``source="shadow"``), recorded, never
    used, and never counted against a label (``compare`` sets shadows aside). An answer the point's vocabulary refuses
    is an abstention; the record is validated again, as the ledger would."""
    from dclab_core.ledger import answer_problem

    extra = []
    for a in answerers:
        if record.point in a.points and not isinstance(a, RecordedRule) and getattr(a, "by", None) in KINDS:
            got = a.answer(record.point, record.state, list(record.options))
            if got is not None and got.by != "person" and not answer_problem(POINTS[record.point], got.value):
                extra.append(got.model_copy(update={"source": SHADOW}))
    if not extra:
        return record
    return type(record).model_validate({**record.model_dump(), "answers": [a.model_dump() for a in [*record.answers, *extra][:20]]})


def _unshadowed(record: DecisionRecord) -> DecisionRecord:
    shown = [a for a in record.answers if a.source != SHADOW]
    return record if len(shown) == len(record.answers) else record.model_copy(update={"answers": shown})


def compare(records: Iterable[DecisionRecord], labels: Iterable[DecisionLabel], answerers: Iterable[DecisionAnswerer],
            accepted: Optional[Mapping[str, Iterable[Any]]] = None) -> dict[str, Any]:
    """Each answerer on the same recorded decisions: how many it answered (coverage), its agreement with the rule
    (where both answered), its accuracy where it answered and a label says what was right, and its accuracy over every
    labelled decision with an abstention counted as wrong (``accuracy_all``: a selective answerer is not flattered);
    overall and by point. Shadow answers already in a record never move what counts as right. ``accepted`` (29.9): per
    record, every answer that counts as right (a workflow decision's journey accepts several); otherwise the label's."""
    accepted_keys = {r: {_key(v) for v in values} for r, values in (accepted or {}).items()}
    answerers = list(answerers)
    from dclab_core.ledger import current_labels

    by_record: dict[str, list[DecisionLabel]] = defaultdict(list)
    for label in current_labels(list(labels)):  # a label another one supersedes, on any record, no longer counts
        by_record[label.record].append(label)
    rule = RecordedRule()
    stats: dict[str, dict[str, Counter]] = {}
    for record in records:
        truth = effective_label(_unshadowed(record), by_record.get(record.id, []))  # what was right never moves with a shadow
        ruled = rule.of(record)
        for a in answerers:
            if record.point not in a.points:
                continue
            key = f"{a.name}@{a.version}"
            got = a.of(record) if isinstance(a, RecordedRule) else a.answer(record.point, record.state, list(record.options))
            single = len(record.options) < 2  # 29.9: a decision with one option: any answer to it is right, so it is counted apart
            for scope in ("all", record.point):
                c = stats.setdefault(key, {}).setdefault(scope, Counter())
                c["decisions"] += 1
                c["truths"] += int(truth is not None)
                c["single_option"] += int(single)
                if got is None or got.value is None:
                    continue
                c["answered"] += 1
                if ruled is not None and ruled.value is not None:
                    c["compared_with_rule"] += 1
                    c["agree_with_rule"] += int(_key(got.value) == _key(ruled.value))
                if truth is not None:
                    c["labelled"] += 1
                    right = accepted_keys.get(record.id) or {_key(truth.value)}
                    c["right"] += int(_key(got.value) in right)
                    if not single:
                        c["labelled_choices"] += 1
                        c["right_choices"] += int(_key(got.value) in right)

    def rates(c: Counter) -> dict[str, Any]:
        return {"decisions": c["decisions"], "answered": c["answered"],
                "coverage": round(c["answered"] / c["decisions"], 4) if c["decisions"] else None,
                "agreement_with_rule": round(c["agree_with_rule"] / c["compared_with_rule"], 4) if c["compared_with_rule"] else None,
                "labelled": c["labelled"], "accuracy": round(c["right"] / c["labelled"], 4) if c["labelled"] else None,
                "accuracy_all": round(c["right"] / c["truths"], 4) if c["truths"] else None,  # an abstention counted as wrong
                "single_option": c["single_option"],  # 29.9: decisions with one option (any answer is right): left out of the next
                "accuracy_choices": round(c["right_choices"] / c["labelled_choices"], 4) if c["labelled_choices"] else None}

    return {key: {scope: rates(c) for scope, c in scopes.items()} for key, scopes in stats.items()}


__all__ = ["contributions", "explain", "DecisionAnswerer", "REGISTRY", "register", "RecordedRule", "ScriptedAnswerer", "ClassifierAnswerer", "features",
           "with_answers", "compare"]
