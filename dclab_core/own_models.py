"""A company's own decision models (package 26.7): DCLab's base classifier fine-tuned on the company's own decisions.

A workspace's labelled decisions (a person's corrections, blind labels, measured outcomes) train a model of its own for
one decision point. DCLab's base classifier for that point (generation 1, 22.2 and 26.3) is never changed: the own
model reads the base model's answer as features (its probability for each value, ``base:<value>``) beside the
decision's state, and learns where the company's decisions differ from it (stacking). With no base model for the point,
the own model learns from the state alone. It is chosen and calibrated by ``decision_models.train`` (folds grouped by
the company's projects) and judged like every other decision model, by ``decision_gate.verdict`` on the workspace's own
sealed decisions, read once.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Optional

from . import answerers as _answerers
from .decision_models import ModelCard, train

BASE = "base:"
LIMITS = ["Trained and measured on this workspace's own labelled decisions (a person's corrections, blind labels, measured "
          "outcomes): it says how the model reads this company's decisions, not anyone else's.",
          "DCLab's base model is read, never changed: its answer is one of the own model's inputs (stacking).",
          "Its sealed gate is the workspace's own sealed decisions (whole projects), read once; a small workspace makes wide "
          "intervals, and a model that does not pass stays in shadow."]


def base_features(base: Any, point: str, state: Mapping[str, Any]) -> dict[str, float]:
    """The base model's probability for each value it knows, as ``base:<value>`` features (none without a base)."""
    if base is None or getattr(base, "model", None) is None or point not in getattr(base, "points", ()):
        return {}
    proba = base.model.predict_proba([_answerers.features(state, base.exclude)])[0]
    return {f"{BASE}{c}": round(float(p), 6) for c, p in zip(base.model.classes_, proba)}


def augment(state: Mapping[str, Any], base: Any, point: str) -> dict[str, Any]:
    return {**{k: v for k, v in state.items() if not str(k).startswith(BASE)}, **base_features(base, point, state)}


class TunedAnswerer:
    """An own model: the base model's answer and the state in, one value out (``answer`` as every answerer's)."""

    by = "classifier"

    def __init__(self, point: str, inner: Any, base: Any = None, base_version: Optional[str] = None):
        self.point, self.points, self.inner, self.base = point, frozenset({point}), inner, base
        self.base_version = base_version or (f"{getattr(base, 'name', 'base')}@{getattr(base, 'version', '')}" if base is not None else None)
        self.name, self.version = "own", inner.version
        self.exclude, self.abstain_below, self.trained_on = inner.exclude, inner.abstain_below, inner.trained_on

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Any:
        if point != self.point:
            return None
        got = self.inner.answer(point, augment(state, self.base, point), options)
        return None if got is None else got.model_copy(update={"answerer": f"{self.name}@{self.version}"})

    def answer_many(self, point: str, states: Iterable[Mapping[str, Any]], options: Iterable[list[Any]]) -> list[Any]:
        return [self.answer(point, s, o) for s, o in zip(states, options)]


def tune(point: str, train_rows: Iterable[Mapping[str, Any]], dev_rows: Iterable[Mapping[str, Any]] = (), base: Any = None,
         name: str = "own", **options: Any) -> tuple[TunedAnswerer, ModelCard]:
    """The own model of ``point``: the rows' states with the base model's answer added, chosen and calibrated by
    ``decision_models.train`` on folds grouped by the rows' groups (sealed rows refused there)."""
    rows = [{**r, "state": augment(r["state"], base, point)} for r in train_rows]
    dev = [{**r, "state": augment(r["state"], base, point)} for r in dev_rows]
    inner, card = train(point, rows, dev, limits=LIMITS, **options)
    base_name = f"{getattr(base, 'name', 'base')}@{getattr(base, 'version', '')}" if base is not None else None
    version = hashlib.sha256(f"{inner.version}|{base_name}".encode()).hexdigest()[:12]
    inner.version = version
    model = TunedAnswerer(point, inner, base, base_name)
    model.name = name
    card = card.model_copy(update={"name": name, "version": version,
                                   "chosen_by": card.chosen_by + (f"; stacked on DCLab's base model {base_name}" if base_name else "; no base model for this point")})
    return model, card


__all__ = ["BASE", "LIMITS", "TunedAnswerer", "augment", "base_features", "tune"]
