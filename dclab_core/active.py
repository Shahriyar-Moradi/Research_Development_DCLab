"""Students ask for the labels they need most (package 29.11): the order in which a label teaches most.

Measured on the generated sets (evidence/campaigns/active_learning_v1): the first rule, :func:`priority` and
:func:`select` (a disagreement with the rule first, then the least sure answer), taught less than labels drawn at random
(ALQ-001: it skewed the labels toward one kind of case); the second, :func:`select_mixed` (half the least sure answers,
half drawn at random, written after the first failed and judged on the same dev rows) did not pass its gate either
(ALQ-002: split.strategy +0.062 on 5 of 5 seeds, data.cases_table +0.005, task.framing -0.004 and worse on 4 of 5 seeds;
the gate asked for two points of three better by 0.01). One quick logistic student, invented tables only. A host may offer
the second as an order a person chooses, never as the default, and says what it measured. A label chosen this way is not a sample: the figures a promotion reads
(a student's accuracy where it differs from the rule) must never be estimated from it, only from labels drawn with a
known probability; a host keeps the two apart.

Pure: the standard library and this package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

TEACH_NOTE = "chosen to teach (29.11), not drawn: never an estimate of accuracy, only a training label"


def priority(student: Any, rule: Any, confidence: Optional[float], need: int = 0) -> tuple[int, int, float]:
    """The sort key of one decision (smaller first): a disagreement with the rule before an agreement, a point needing
    more labels before one needing fewer, the least sure answer first. ``student``/``rule``: the two answers' values;
    ``confidence``: the student's, in [0, 1] (unknown: treated as sure); ``need``: the labels the point still needs."""
    from dclab_core.answerers import _key

    disagrees = student is not None and rule is not None and _key(student) != _key(rule)
    sure = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 1.0
    return (0 if disagrees else 1, -int(max(0, need)), round(sure, 6))


def select(pool: Sequence[Mapping[str, Any]], k: int, needs: Optional[Mapping[str, int]] = None) -> list[int]:
    """The indices of the ``k`` decisions of ``pool`` that teach most, each item ``{"point", "student", "rule",
    "confidence"}``; ties keep the pool's order (the caller's seeded order), so the choice is reproducible."""
    needs = needs or {}
    order = sorted(range(len(pool)), key=lambda i: (*priority(pool[i].get("student"), pool[i].get("rule"), pool[i].get("confidence"),
                                                              needs.get(str(pool[i].get("point")), 0)), i))
    return order[:max(0, int(k))]


def select_mixed(pool: Sequence[Mapping[str, Any]], k: int, seed: str) -> list[int]:
    """The second rule (ALQ-002, fixed after ALQ-001 failed): half of ``k`` the student's least sure answers, whatever the
    rule said, half drawn at random from the rest (seeded): the uncertain cases teach, the random ones keep the labelled set
    like the decisions it will meet. A disagreement-first choice (``select``) skewed the labels toward one kind of case."""
    import random

    k = max(0, min(int(k), len(pool)))
    sure = lambda i: float(pool[i].get("confidence")) if isinstance(pool[i].get("confidence"), (int, float)) else 1.0  # noqa: E731
    unsure = sorted(range(len(pool)), key=lambda i: (sure(i), i))[: k // 2]
    rest = [i for i in range(len(pool)) if i not in set(unsure)]
    random.Random(seed).shuffle(rest)
    return unsure + rest[: k - len(unsure)]


def split_labels(labels: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    """(labels an estimate may read, labels chosen to teach): a blind label without a sampling weight was not drawn
    with a known probability (taught, or imported without its weight), so it trains but never estimates."""
    estimate, taught = [], []
    for x in labels:
        (taught if getattr(x, "source", None) == "blind" and getattr(x, "weight", None) is None else estimate).append(x)
    return estimate, taught


__all__ = ["TEACH_NOTE", "priority", "select", "select_mixed", "split_labels"]
