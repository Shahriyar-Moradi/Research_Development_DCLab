"""Delivering many projects (package 30.24, the owner's pattern 11): fifteen requests and capacity for five.

* **Each request scored** (:func:`score`) on four things, each from 0 to 5 with its reason: value (what it is worth a month), data
  readiness (the tables exist, the outcome is labelled), effort (tables, a kind DCLab has not run, the data to join) and risk (personal
  data, decisions acted on with no person). The total favours value and readiness and weighs effort and risk against them.
* **K chosen of N** (:func:`choose`) by the total, a request with no success number never chosen (it is agreed before building, the
  brief's bar), ties to the one that reuses most; what the chosen share (the same tables, the same kind) is said as reuse.
* **Projects nobody uses retired** (:func:`retire`): a project with no use for ``days`` is flagged, with when it was last used.

Pure Python and pydantic; the host fills the requests (Home's answers) and the use (the outcomes).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

WEIGHTS = {"value": 2.0, "readiness": 1.5, "effort": -1.0, "risk": -1.0}


class Request(Versioned):
    """One requested project, as the person described it (v1)."""

    schema_version: Literal["dclab_core.project_request.v1"] = _version("project_request")
    id: str = Field(max_length=60)
    title: str = Field(max_length=200)
    kind: Optional[str] = Field(default=None, max_length=40)
    value_eur_month: Optional[float] = Field(default=None, ge=0)
    tables: list[str] = Field(default_factory=list, max_length=30)
    tables_exist: bool = False
    labelled: Optional[bool] = None       # None: the request needs no outcome (a forecast, a segmentation) or does not say
    new_kind: bool = False                # DCLab has not run this kind of problem
    personal_data: bool = False
    acts_on_its_own: bool = False         # its decisions are applied with no person
    success: Optional[str] = Field(default=None, max_length=300)  # the success number agreed before building


class Score(Contract):
    request: str
    value: int
    readiness: int
    effort: int
    risk: int
    total: float
    reasons: list[str] = Field(default_factory=list)


def score(r: Request) -> Score:
    reasons = []
    v = r.value_eur_month or 0
    value = 5 if v >= 50_000 else 4 if v >= 20_000 else 3 if v >= 5_000 else 2 if v >= 1_000 else 1 if v > 0 else 0
    reasons.append(f"value {value}: " + (f"about EUR {v:,.0f} a month" if v else "no value said"))
    readiness = (3 if r.tables_exist else 0) + (2 if r.labelled or r.labelled is None and r.tables_exist else 0)
    reasons.append(f"readiness {readiness}: " + ("the tables exist" if r.tables_exist else "the tables are not there yet")
                   + ("" if r.labelled is None else ", the outcome is labelled" if r.labelled else ", nothing is labelled"))
    effort = min(5, len(r.tables) + (2 if r.new_kind else 0) + (1 if not r.tables_exist else 0))
    reasons.append(f"effort {effort}: {len(r.tables)} table(s)" + (", a kind DCLab has not run" if r.new_kind else ""))
    risk = min(5, (2 if r.personal_data else 0) + (3 if r.acts_on_its_own else 0))
    said = ", ".join(x for x, on in (("personal data", r.personal_data), ("decisions applied with no person", r.acts_on_its_own)) if on)
    reasons.append(f"risk {risk}: {said or 'none said'}")
    total = sum(WEIGHTS[k] * x for k, x in (("value", value), ("readiness", readiness), ("effort", effort), ("risk", risk)))
    return Score(request=r.id, value=value, readiness=readiness, effort=effort, risk=risk, total=round(total, 2), reasons=reasons)


class Portfolio(Versioned):
    """K chosen of N, why, and what they reuse (v1)."""

    schema_version: Literal["dclab_core.portfolio.v1"] = _version("portfolio")
    capacity: int
    chosen: list[str]
    waiting: list[str]
    not_ready: list[str] = Field(default_factory=list)  # no success number agreed: never chosen
    scores: list[Score]
    reuse: list[dict[str, Any]] = Field(default_factory=list)


def choose(requests: list[Request], capacity: int) -> Portfolio:
    scores = {r.id: score(r) for r in requests}
    ready = [r for r in requests if r.success]
    shares = {r.id: sum(1 for o in ready if o.id != r.id and (set(o.tables) & set(r.tables) or (o.kind and o.kind == r.kind))) for r in ready}
    ranked = sorted(ready, key=lambda r: (-scores[r.id].total, -shares[r.id], r.id))
    chosen = [r for r in ranked[:capacity]]
    reuse = []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            tables = sorted(set(a.tables) & set(b.tables))
            if tables or (a.kind and a.kind == b.kind):
                reuse.append({"requests": [a.id, b.id], "tables": tables, "kind": a.kind if a.kind == b.kind else None})
    return Portfolio(capacity=capacity, chosen=[r.id for r in chosen], waiting=[r.id for r in ranked[capacity:]],
                     not_ready=[r.id for r in requests if not r.success], scores=list(scores.values()), reuse=reuse)


def _when(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    when = datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def retire(used: dict[str, Optional[str]], *, now: str, days: int = 90, created: Optional[dict[str, Optional[str]]] = None) -> list[dict[str, Any]]:
    """Projects with no use for ``days`` (``used``: project -> its last use, an ISO time, or None for never), each with when it was last
    used; one in use is never flagged, and one never used only once it is older than ``days`` (``created``: project -> when it was made;
    with no date it is not flagged, since it may be new)."""
    cut = _when(now) - timedelta(days=days)
    out = []
    for project, last in sorted(used.items()):
        when = _when(last)
        if when is not None:
            if when < cut:
                out.append({"project": project, "last_used": last, "why": f"no use in {days} days"})
            continue
        made = _when((created or {}).get(project))
        if made is not None and made < cut:
            out.append({"project": project, "last_used": None, "why": f"never used since it was made, {days} days or more ago"})
    return out


CONTRACTS["project_request"] = Request
CONTRACTS["portfolio"] = Portfolio

__all__ = ["Request", "Score", "Portfolio", "WEIGHTS", "score", "choose", "retire"]
