"""Outcomes linked to decisions (package 23.5): what a decision led to, read from the number that followed it.

The verdict compares the number with its reference as points: it does not consult the interval (a holdout below its
estimate with the reference inside the interval is still ``badly``); a set that learns from it carries both.

A decision record says what was chosen; an ``Outcome`` says how it turned out: the metric, its value and interval,
what it was expected to be (the reference: the chosen model's cross-validation mean, a target a person typed), when and
how it was observed (``holdout``: the final stage's one holdout read; ``promotion``: the final stage approved;
``business``: a number a person typed after the model ran) and the verdict ``judge`` gives against the reference.
Every decision of the project that shaped the number gets its own row, so a decision set can carry "turned out
well/badly" beside its labels; an outcome is never a label (it does not say the decision was right, only what followed).
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Versioned, _version

OBSERVED: tuple[str, ...] = ("holdout", "promotion", "business")
VERDICTS: tuple[str, ...] = ("well", "badly", "unknown")


def judge(value: Any, reference: Any, higher_is_better: Any, tolerance: Optional[float] = None) -> str:
    """``well`` when the value is not worse than the reference beyond the tolerance (an honest model holds its
    cross-validation estimate), ``badly`` when it is, ``unknown`` without a reference or a direction. The tolerance is
    0.02 for a bounded metric (|reference| <= 1) and 5% of the reference otherwise, unless given."""
    if value is None or reference is None or higher_is_better is None:
        return "unknown"
    try:
        v, r = float(value), float(reference)
    except (TypeError, ValueError):
        return "unknown"
    if v != v or r != r:
        return "unknown"
    tol = tolerance if tolerance is not None else (0.02 if abs(r) <= 1 else 0.05 * abs(r))
    worse = (r - v) if higher_is_better else (v - r)
    return "badly" if worse > tol else "well"


class Outcome(Versioned):
    """How a decision turned out: one row per decision and observation; the newest per record is the current one."""

    schema_version: Literal["dclab_core.outcome.v1"] = _version("outcome", 1)
    READS: ClassVar[frozenset[str]] = frozenset()
    id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{6,80}$")
    record: str = Field(min_length=1, max_length=80, description="The decision record it follows.")
    point: str
    metric: str = Field(pattern=r"^[A-Za-z0-9_.:%/-]{1,80}$", description="A metric's name as a slug (roc_auc, retained_customers): "
                        "a person's words never travel in it.")
    value: float
    interval: Optional[tuple[float, float]] = None
    reference: Optional[float] = Field(default=None, description="What it was expected to be: the CV mean, a target.")
    higher_is_better: Optional[bool] = None
    observed: Literal["holdout", "promotion", "business"]
    verdict: Literal["well", "badly", "unknown"]
    by: str = Field(min_length=1, max_length=120, description="What observed it: final_stage, promotion, a person's role.")
    note: Optional[str] = Field(default=None, max_length=300, description="A person's words with a business number (redacted when it leaves).")
    workspace: Optional[str] = Field(default=None, max_length=64)
    project: Optional[str] = Field(default=None, max_length=64)
    run: Optional[str] = Field(default=None, max_length=80)
    at: str
    consent: Literal["operate", "improve", "train"] = "operate"


CONTRACTS["outcome"] = Outcome


def preference(observed: Any, verdict: Any, at: Any, ident: Any) -> tuple:
    """Which of a decision's outcomes is the current one: a judged verdict before an unjudged one (a number typed with no
    reference never erases what the holdout said), a business number before the model's own, then the newest."""
    return (verdict != "unknown", observed == "business", str(at or ""), str(ident or ""))


def join(records: Iterable[Any], outcomes: Iterable[Outcome]) -> dict[str, Outcome]:
    """The current outcome of each decision record (see ``preference``), for the records given."""
    wanted = {getattr(r, "id", r) for r in records}
    out: dict[str, Outcome] = {}
    for o in outcomes:
        if o.record in wanted and (o.record not in out or preference(o.observed, o.verdict, o.at, o.id)
                                   > preference(out[o.record].observed, out[o.record].verdict, out[o.record].at, out[o.record].id)):
            out[o.record] = o
    return out


def by_point(records: Iterable[Any], outcomes: Iterable[Outcome]) -> dict[str, dict[str, int]]:
    """Per decision point: how many decisions turned out well, badly, unknown, and how many have no outcome yet."""
    records = list(records)
    current = join(records, outcomes)
    out: dict[str, dict[str, int]] = {}
    for r in records:
        row = out.setdefault(r.point, {"well": 0, "badly": 0, "unknown": 0, "none": 0})
        o = current.get(r.id)
        row[o.verdict if o is not None else "none"] += 1
    return out


__all__ = ["OBSERVED", "VERDICTS", "Outcome", "judge", "join", "by_point"]
