"""Classical decision classifiers (package 22.2): one per decision point, trained on decision rows, with a card.

:func:`train` fits a classifier for one point on a decision set's training rows (``dclab_core.datasets.decisions``:
the state, the label that counts, the group): the family and its strength are chosen by out-of-fold accuracy over
folds grouped by table (a table's rows never sit in two folds), then refitted on all training rows. The dev rows
measure it beside the rule; the sealed rows are never read here (22.3 reads them once, under a gate fixed before).
:func:`metrics` gives accuracy, and precision and recall of the positive answer for a yes/no point (macro over the
classes for a choice), each with its 95% Wilson interval. The :class:`ModelCard` says what was trained on what, and
how it compares with the rule on the same rows. The chosen model is calibrated (Platt scaling on the same table-grouped
folds of the training rows) and its calibration error measured on dev.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import Field



class _Answerers:
    """``dclab_core.answerers``, imported when first used: contracts imports this module to register the card, and
    answerers imports the ledger, which imports contracts; at import time the cycle would break whichever came first."""

    def __getattr__(self, name: str) -> Any:
        from dclab_core import answerers

        return getattr(answerers, name)


_answerers = _Answerers()
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

FAMILIES = ("logistic", "boosting")
STRENGTHS = (0.1, 1.0, 10.0)


def wilson(right: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if not n:
        return None, None
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


class Score(Contract):
    n: int = Field(ge=0)
    accuracy: Optional[float] = None
    accuracy_ci: list[Optional[float]] = Field(default_factory=list, max_length=2)
    precision: Optional[float] = None
    precision_ci: list[Optional[float]] = Field(default_factory=list, max_length=2)
    recall: Optional[float] = None
    recall_ci: list[Optional[float]] = Field(default_factory=list, max_length=2)
    intervals: str = Field("", max_length=200, description="How the precision and recall intervals were drawn.")
    answered: int = Field(0, ge=0, description="Rows it answered (an abstention is counted wrong in accuracy).")
    per_class: dict[str, dict[str, Any]] = Field(default_factory=dict, description=(
        "For a choice: each class's rows, precision and recall with their Wilson intervals (macro figures are their means)."))


def _macro(t: Any, a: Any, w: Any, k: int) -> tuple[float, float]:
    """Macro precision and recall over ``k`` classes (codes 0..k-1; -1 an abstention), rows weighted by ``w``; a class
    never answered has precision 0, a class never true has recall 0 (as scikit-learn's ``zero_division=0``)."""
    import numpy as np

    tp = np.bincount(t[t == a], weights=w[t == a], minlength=k)
    said = np.bincount(a[a >= 0], weights=w[a >= 0], minlength=k)
    true = np.bincount(t, weights=w, minlength=k)
    p = np.divide(tp, said, out=np.zeros(k), where=said > 0)
    r = np.divide(tp, true, out=np.zeros(k), where=true > 0)
    return float(p.mean()), float(r.mean())


def metrics(truth: list[Any], answers: list[Any], positive: Any = True, groups: Optional[list[Any]] = None,
            draws: int = 1000, seed: int = 0) -> Score:
    """Accuracy over every row (an abstention, None, is wrong). For a yes/no point, the precision and recall of
    ``positive`` with their Wilson intervals. For a choice, macro precision and recall over every class true or
    answered (a class never answered counts precision 0, one never true recall 0), with intervals from a bootstrap
    clustered by ``groups`` (the table a row came from; each row its own cluster when none), and each class's figures
    with Wilson intervals."""
    import numpy as np

    pairs = [(_answerers._key(t), None if a is None else _answerers._key(a)) for t, a in zip(truth, answers)]
    n = len(pairs)
    right = sum(t == a for t, a in pairs)

    def counts(c: str) -> tuple[int, int, int]:
        return (sum(t == c and a == c for t, a in pairs), sum(t != c and a == c for t, a in pairs),
                sum(t == c and a != c for t, a in pairs))

    def ratio(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    base = {"n": n, "accuracy": ratio(right, n), "accuracy_ci": list(wilson(right, n)),
            "answered": sum(a is not None for _, a in pairs)}
    seen = sorted({t for t, _ in pairs} | {a for _, a in pairs if a is not None})
    if set(seen) <= {"True", "False"}:
        tp, fp, fn = counts(_answerers._key(positive))
        return Score(**base, precision=ratio(tp, tp + fp), precision_ci=list(wilson(tp, tp + fp)),
                     recall=ratio(tp, tp + fn), recall_ci=list(wilson(tp, tp + fn)), intervals="Wilson, 95%")
    per = {}
    for c in seen:
        tp, fp, fn = counts(c)
        per[c] = {"n": tp + fn, "precision": ratio(tp, tp + fp) if tp + fp else 0.0, "precision_ci": list(wilson(tp, tp + fp)),
                  "recall": ratio(tp, tp + fn) if tp + fn else 0.0, "recall_ci": list(wilson(tp, tp + fn))}
    if not n:
        return Score(**base, per_class=per)
    code = {c: i for i, c in enumerate(seen)}
    t = np.array([code[x] for x, _ in pairs])
    a = np.array([-1 if y is None else code[y] for _, y in pairs])
    precision, recall = _macro(t, a, np.ones(n), len(seen))
    cluster_of = {g: i for i, g in enumerate(dict.fromkeys(groups))} if groups is not None else None
    cl = np.array([cluster_of[g] for g in groups]) if cluster_of else np.arange(n)
    rng = np.random.default_rng(seed)
    k = int(cl.max()) + 1
    drawn = [_macro(t, a, np.bincount(rng.integers(0, k, k), minlength=k)[cl].astype(float), len(seen)) for _ in range(draws)]
    ps, rs = np.array([d[0] for d in drawn]), np.array([d[1] for d in drawn])

    def ci(x: Any) -> list[Optional[float]]:
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]

    return Score(**base, precision=round(precision, 4), precision_ci=ci(ps), recall=round(recall, 4), recall_ci=ci(rs),
                 intervals=f"bootstrap of {draws} draws, clustered by {'table group' if cluster_of else 'row'}, 95%", per_class=per)


class ModelCard(Versioned):
    """A decision classifier: its point, how it was chosen, what it was trained on, and how it compares with the rule."""

    schema_version: Literal["dclab_core.decision_model_card.v1"] = _version("decision_model_card")
    point: str = Field(max_length=80)
    name: str = Field(max_length=120)
    version: str = Field(max_length=40, description="The fitted model's identity: a hash of its candidate and its training rows.")
    family: Literal["logistic", "boosting"]
    strength: float = Field(description="Logistic: the inverse regularisation C. Boosting: the trees' depth is 2 plus its whole part.")
    features: int = Field(ge=0, description="How many features the state gave (bands one-hot, name tokens, numbers).")
    fields: list[str] = Field(default_factory=list, max_length=200, description="The state fields the features came from.")
    excluded: list[str] = Field(default_factory=list, max_length=20, description="State fields left out (a label's own measure).")
    trained_on: int = Field(ge=0)
    trained_set: str = Field("", max_length=64, description="SHA-256 of the training rows' records and labels.")
    groups: int = Field(ge=0, description="Tables (groups) in the training rows.")
    label_sources: dict[str, int] = Field(default_factory=dict, description="Training labels by source.")
    chosen_by: str = Field(max_length=300)
    oof_accuracy: dict[str, float] = Field(default_factory=dict, description="Out-of-fold accuracy of each candidate, by table folds.")
    calibration: str = Field("", max_length=200, description="How the chosen model's confidences were calibrated.")
    dev: Optional[Score] = None
    rule_dev: Optional[Score] = None
    dev_by_source: dict[str, Score] = Field(default_factory=dict, description="The model's dev score by label source.")
    dev_calibration_error: Optional[float] = Field(None, description=(
        "Expected calibration error on dev, after calibration: the gap between the answer's confidence and how often "
        "it is right, over ten bins of confidence."))
    limits: list[str] = Field(default_factory=list, max_length=10)


CONTRACTS["decision_model_card"] = ModelCard


def calibration_error(confidences: list[float], right: list[bool], bins: int = 10) -> Optional[float]:
    """Expected calibration error: each bin's gap between mean confidence and accuracy, weighted by its rows."""
    if not confidences:
        return None
    total = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confidences) if min(int(c * bins), bins - 1) == b]
        if idx:
            total += len(idx) * abs(sum(confidences[i] for i in idx) / len(idx) - sum(right[i] for i in idx) / len(idx))
    return round(total / len(confidences), 4)


def _labelled(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("label") and r["label"].get("value") is not None]


def _splits(rows: list[Mapping[str, Any]], folds: int) -> list[tuple[Any, Any]]:
    from sklearn.model_selection import GroupKFold

    groups = [r["group"] for r in rows]
    return list(GroupKFold(n_splits=min(folds, len(set(groups)))).split(rows, groups=groups))


def _fit(point: str, family: str, strength: float, rows: list[Mapping[str, Any]], exclude: tuple[str, ...],
         calibrate: Optional[list[tuple[Any, Any]]] = None) -> "_answerers.ClassifierAnswerer":
    """A logistic model (numbers scaled on the training rows) or gradient-boosted trees (depth 2 plus the strength's
    whole part) on the state's features; with ``calibrate`` (table-grouped folds of ``rows``), Platt-scaled on them."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MaxAbsScaler

    data = [(_answerers.features(r["state"], exclude), r["label"]["value"]) for r in rows]
    if len({_answerers._key(v) for _, v in data}) < 2:
        raise ValueError("a classifier needs at least two labelled values")
    if family == "boosting":
        steps: list[Any] = [DictVectorizer(sparse=False)]
        estimator: Any = HistGradientBoostingClassifier(max_depth=int(2 + strength), random_state=0)
    else:
        steps = [DictVectorizer(sparse=True), MaxAbsScaler()]
        estimator = LogisticRegression(C=strength, max_iter=1000)
    if calibrate:
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=calibrate)
    clf = _answerers.ClassifierAnswerer(point, name=f"{point}.{family}", C=strength, exclude=exclude)
    clf.model = make_pipeline(*steps, estimator)
    clf.model.fit([f for f, _ in data], [_answerers._key(v) for _, v in data])
    clf._values = {_answerers._key(v): v for _, v in data}
    clf.trained_on = len(data)
    return clf


def _oof(point: str, family: str, strength: float, rows: list[Mapping[str, Any]], exclude: tuple[str, ...], folds: int) -> float:
    """Out-of-fold accuracy over table-grouped folds; a fold whose training rows hold one value is not fitted and its
    rows count wrong (as an abstention does)."""
    right = 0
    for train_idx, test_idx in _splits(rows, folds):
        train = [rows[i] for i in train_idx]
        if len({_answerers._key(r["label"]["value"]) for r in train}) < 2:
            continue
        clf = _fit(point, family, strength, train, exclude)
        for i in test_idx:
            got = clf.answer(point, rows[i]["state"], list(rows[i].get("options") or []))
            right += int(got is not None and _answerers._key(got.value) == _answerers._key(rows[i]["label"]["value"]))
    return right / len(rows) if rows else 0.0


def train(point: str, train_rows: Iterable[Mapping[str, Any]], dev_rows: Iterable[Mapping[str, Any]] = (),
          exclude: Iterable[str] = (), folds: int = 5, families: Iterable[str] = FAMILIES,
          strengths: Iterable[float] = STRENGTHS) -> tuple["_answerers.ClassifierAnswerer", ModelCard]:
    """The classifier for ``point``: chosen by out-of-fold accuracy over table-grouped folds of ``train_rows`` (a tie
    goes to the smaller strength, then to the logistic model), refitted on all of them and calibrated on the same
    folds, measured on ``dev_rows`` beside the rule. Sealed rows are refused."""
    rows, dev_rows = _labelled(train_rows), list(dev_rows)
    if any(r.get("split") == "sealed" for r in [*rows, *dev_rows]):
        raise ValueError("sealed rows are never trained, tuned or measured on here")
    if len({r["group"] for r in rows}) < 2:
        raise ValueError("table-grouped folds need training rows from at least two tables")
    exclude = tuple(exclude)
    scores = {f"{family}:{strength}": _oof(point, family, strength, rows, exclude, folds) for family in families for strength in strengths}
    order = list(scores)
    best = max(order, key=lambda k: (scores[k], -float(k.split(":")[1]), -order.index(k)))
    family, strength = best.split(":")[0], float(best.split(":")[1])
    splits = _splits(rows, folds)
    model = _fit(point, family, strength, rows, exclude, calibrate=splits)
    trained = sorted(f"{r.get('record')}={_answerers._key(r['label']['value'])}" for r in rows)
    trained_set = hashlib.sha256("\n".join(trained).encode()).hexdigest()
    model.version = hashlib.sha256(f"{best}|{trained_set}".encode()).hexdigest()[:12]
    dev = _labelled(dev_rows)
    dev_score = rule_score = ece = None
    by_source: dict[str, Score] = {}
    if dev:
        truth = [r["label"]["value"] for r in dev]
        groups = [r.get("group") for r in dev]
        got = [model.answer(point, r["state"], list(r.get("options") or [])) for r in dev]
        answers = [None if a is None else a.value for a in got]
        dev_score = metrics(truth, answers, groups=groups)
        rule_score = metrics(truth, [r.get("value_used") for r in dev], groups=groups)
        for source in sorted({r["label"].get("source", "unknown") for r in dev}):
            idx = [i for i, r in enumerate(dev) if r["label"].get("source", "unknown") == source]
            by_source[source] = metrics([truth[i] for i in idx], [answers[i] for i in idx], groups=[groups[i] for i in idx])
        answered = [(a.confidence, _answerers._key(a.value) == _answerers._key(t)) for a, t in zip(got, truth) if a is not None]
        ece = calibration_error([c for c, _ in answered], [r for _, r in answered])
    vectorizer = model.model.steps[0][1]
    card = ModelCard(point=point, name=model.name, version=model.version, family=family, strength=strength,
                     features=len(vectorizer.feature_names_),
                     fields=sorted({f for r in rows for f in r["state"] if f not in exclude}),
                     excluded=list(exclude), trained_on=len(rows), trained_set=trained_set, groups=len({r["group"] for r in rows}),
                     label_sources=dict(Counter(r["label"].get("source", "unknown") for r in rows)),
                     chosen_by=f"out-of-fold accuracy over {len(splits)} folds grouped by table; best {best}",
                     oof_accuracy={k: round(v, 4) for k, v in scores.items()},
                     calibration=f"Platt scaling on the same {len(splits)} table-grouped folds of the training rows",
                     dev=dev_score, rule_dev=rule_score, dev_by_source=by_source, dev_calibration_error=ece,
                     limits=["Measured on dev rows; the sealed partition is read once, under 22.3's gate.",
                             "Trained on generated tables (22.1): it learns these planted kinds, not real tables.",
                             "A group is a base and a seed: dev holds new seeds of the same bases as training, not new "
                             "kinds of table, so dev measures reading planted kinds in tables like those trained on.",
                             "Every generated table plants the same kinds, so intervals drawn by table group are narrow: "
                             "they say how stable the model is across these tables, not across other kinds of table."])
    return model, card


__all__ = ["train", "metrics", "wilson", "calibration_error", "ModelCard", "Score", "FAMILIES"]
