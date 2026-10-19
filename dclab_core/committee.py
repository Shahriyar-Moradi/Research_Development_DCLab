"""A committee: one decision answered by a vote of several answerers (package 29.16).

The members are the rule, the point's student, other models (GLiNER, a model role): each votes with a weight from its
measured accuracy on the point (``weight``: the log-odds of being right over the chance of a guess among the options, so
a member no better than chance never votes and one always right outweighs every other), and the option with the most
weight wins (``vote``). An abstaining member gives no vote. A tie goes to the tied option of the heaviest member that
named one, then to the first member in the committee's order (the rule first). The committee is one more answerer
(``Committee``): its version is the fingerprint of its members' versions and weights, so another weighting is another
answerer, examined again.

The weights are measured before a committee is examined, on rows the exam never reads; a member whose accuracy was
measured on a sample chosen by difficulty is weighted low, never high, by it (the weight is a floor, not an estimate).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

from .ledger import Answer

VERSION = "committee_v1"
CEILING = 0.995  # an accuracy of 1.0 measured on finitely many rows is read as this: one member never outvotes all forever


def weight(accuracy: Optional[float], options: int = 2) -> float:
    """A member's vote: ``log((k - 1) a / (1 - a))`` for accuracy ``a`` among ``k`` options (the weighted majority's
    weight when every wrong option is equally likely); 0 at chance or below, or with no measurement."""
    if accuracy is None:
        return 0.0
    k = max(2, int(options))
    a = min(max(float(accuracy), 0.0), CEILING)
    if a <= 1.0 / k:
        return 0.0
    return round(math.log((k - 1) * a / (1 - a)), 6)


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def vote(answers: Mapping[str, Any], weights: Mapping[str, float], order: Sequence[str]) -> tuple[Any, Optional[float], dict[str, float]]:
    """The committee's answer from its members' (``None``: an abstention), the share of the weight behind it, and the
    weight each option got. No member with a weight answered: no answer."""
    tally: dict[str, float] = {}
    values: dict[str, Any] = {}
    for member in order:
        value = answers.get(member)
        w = float(weights.get(member) or 0.0)
        if value is None or w <= 0:
            continue
        k = _key(value)
        values.setdefault(k, value)
        tally[k] = tally.get(k, 0.0) + w
    if not tally:
        return None, None, {}
    best = max(tally.values())
    tied = [k for k, v in tally.items() if abs(v - best) < 1e-9]
    if len(tied) > 1:  # the heaviest member's choice among the tied options, then the committee's order
        ranked = sorted((m for m in order if answers.get(m) is not None and float(weights.get(m) or 0) > 0 and _key(answers[m]) in tied),
                        key=lambda m: (-float(weights[m]), list(order).index(m)))
        winner = _key(answers[ranked[0]])
    else:
        winner = tied[0]
    total = sum(tally.values())
    return values[winner], round(tally[winner] / total, 4), {k: round(v, 6) for k, v in tally.items()}


def fingerprint(members: Iterable[tuple[str, str]], weights: Mapping[str, float]) -> str:
    """The committee's version: its members (name and version) and their weights."""
    text = json.dumps({"members": sorted(members), "weights": {k: round(float(v), 6) for k, v in sorted(weights.items())}}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class Committee:
    """One point's committee as an answerer. ``members``: (name, answerer) pairs in the committee's order, each with
    ``answer(point, state, options)``; ``given``: answers already known for this decision (the rule's, recorded where it
    decided; another model's, captured) passed to :meth:`answer` by name."""

    by = "classifier"  # a vote of answerers, kept as a classifier's answer is (a whisper until it is routed)

    def __init__(self, point: str, order: Sequence[str], weights: Mapping[str, float], members: Mapping[str, Any] | None = None,
                 versions: Mapping[str, str] | None = None, name: str = "committee"):
        self.point, self.order, self.weights = point, list(order), dict(weights)
        self.members = dict(members or {})
        self.name = name
        self.version = fingerprint([(m, str((versions or {}).get(m, ""))) for m in self.order], self.weights)

    def answers(self, state: Mapping[str, Any], options: list[Any], given: Mapping[str, Any] | None = None) -> dict[str, Any]:
        out = {m: v for m, v in (given or {}).items() if m in self.order}
        for m, answerer in self.members.items():
            if m in out or m not in self.order:
                continue
            try:
                got = answerer.answer(self.point, state, options)
            except Exception:  # noqa: BLE001 - a member that fails abstains
                got = None
            out[m] = None if got is None else got.value
        return out

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any], given: Mapping[str, Any] | None = None) -> Optional[Answer]:
        if point != self.point:
            return None
        value, share, _ = vote(self.answers(state, options, given), self.weights, self.order)
        if value is None:
            return None
        return Answer(by=self.by, answerer=f"{self.name}@{self.version}", value=value, confidence=share)


__all__ = ["VERSION", "CEILING", "weight", "vote", "fingerprint", "Committee"]
