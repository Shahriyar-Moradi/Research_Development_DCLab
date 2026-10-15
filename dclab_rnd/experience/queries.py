"""The standard questions of the record (package 23.9), asked of this studio's log and ledger.

The questions themselves are ``dclab_core.queries`` (pure functions over events, decisions, labels and outcomes, so the main
product asks them of its adapted rows); this module reads the log and the ledger and calls them. Reads only.
``python -m dclab_rnd.experience query NAME`` and ``GET /api/experience/queries/NAME`` answer the same:

    request        what happened in one request: every event it left, in order
    project        a project's story: its moves, runs, decisions and outcomes, to replay it
    rejections     where agents and checks are turned down most (by kind and point)
    failed_calls   which model calls failed (by purpose and outcome)
    overrides      which decisions a person overrode, and what they chose instead
    outcomes       whether approved decisions turned out well (23.5), by point
    cost           what the workspace spends per week and currency
    training       what is ready for training: by consent level and label source
    deletion       what deleting a project, draft or session would remove
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from dclab_core import queries as q

NAMES = ("request", "project", "rejections", "failed_calls", "overrides", "outcomes", "cost", "training", "deletion")
LIMIT = 100_000


def _since(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


def _outcomes(ledger: Any, **filters: Any) -> list[Any]:
    return ledger.outcomes(**filters) if ledger is not None and hasattr(ledger, "outcomes") else []


def run(name: str, log: Any, ledger: Any = None, *, request: str | None = None, project: str | None = None, draft: str | None = None,
        visit: str | None = None, days: int | None = None, usage: Any = None) -> dict[str, Any]:
    """The answer to one standard question; ``ValueError`` names what is missing or unknown."""
    if name not in NAMES:
        raise ValueError(f"the questions are: {', '.join(NAMES)}")
    since = _since(days or 30)
    if name == "request":
        if not request:
            raise ValueError("name the request id")
        return q.request(log.read(request=request, limit=LIMIT), request)
    if name == "project":
        if not project:
            raise ValueError("name the project")
        return q.project_story(log.read(project=project, limit=LIMIT), ledger.records(project=project) if ledger else [], _outcomes(ledger, project=project),
                               project_id=project)
    if name == "rejections":
        got = q.rejections(log.read(refused_only=True, since=since, limit=LIMIT), since=since)
        return {**got, "title": f"Refusals in the last {days or 30} days"}
    if name == "failed_calls":
        checks = usage.failing(since) if usage is not None else []  # replies that came back and failed their check are not events
        got = q.failed_calls(log.read(kinds=["model.request"], since=since, limit=LIMIT), since=since, checks=checks)
        return {**got, "title": f"Model calls that failed in the last {days or 30} days"}
    if name == "cost":
        weeks = max(1, (days or 84) // 7)
        got = q.cost(log.read(kinds=["model.request"], since=_since(weeks * 7), limit=LIMIT), since=_since(weeks * 7))
        return {**got, "title": f"Model spend per week, last {weeks} weeks"}
    if ledger is None:
        raise ValueError("this question reads the ledger")
    if name == "overrides":
        got = q.overrides(ledger.records(), since=_since(days or 90))
        return {**got, "title": f"Decisions a person overrode in the last {days or 90} days"}
    if name == "outcomes":
        return q.outcomes_by_point(ledger.records(project=project) if project else ledger.records(), _outcomes(ledger, project=project) if project else _outcomes(ledger))
    if name == "training":
        from ..consent import stamped_counts

        return q.training((), ledger.records(), ledger.labels(), _outcomes(ledger), event_levels=stamped_counts(log))  # counted in the store, whole
    # a deletion removes everything its ids reach, not the newest slice of it: one filtered read per id, no cap
    seen: dict[str, Any] = {}
    for filters in ({"project": project}, {"conversation": draft}, {"visit": visit}):
        if next(iter(filters.values())):
            for e in log.read(limit=10**9, **filters):
                seen[e.id] = e
    prompts = usage.count_prompts(project_id=project or None, draft_id=draft or None) if usage is not None and (project or draft) else None
    return q.deletion(list(seen.values()), ledger.records(), ledger.labels(), _outcomes(ledger),
                      project_id=project or None, draft_id=draft or None, visit_id=visit or None, prompts=prompts)


__all__ = ["NAMES", "run"]
