"""The standard questions of the record, as pure functions over events and decisions (package 23.9).

A host reads its events (``ExperienceEvent``), decision records, labels and outcomes however it stores them (the studio's
log and ledger; the main product through ``adapters.main_product_experience``) and asks the same questions of them. Every
function returns ``{"name", "title", "columns", "rows"}``; none changes anything or reads the clock (the caller passes the
window as ``since``, an ISO time).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable, Optional

from dclab_core import outcomes as oc


def _result(name: str, title: str, columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"name": name, "title": title, "columns": columns, "rows": rows}


def what(event: Any) -> str:
    """The most telling meta field of an event, as ``key=value`` (never text a person wrote)."""
    p = event.payload
    for key in ("point", "move", "tool", "stage", "purpose", "where", "kind", "on", "metric", "step", "name", "code"):
        if p.get(key) and event.classes.get(key, "meta") == "meta":  # a field a person typed (a token's label) is not meta: never shown here
            return f"{key}={p[key]}"
    return ""


def refused(event: Any) -> bool:
    if event.kind in ("draft.refused", "solution.refused", "decision.refused"):
        return True
    if event.kind == "graph.move":
        return event.payload.get("allowed") is False
    return event.kind == "tool.call" and event.payload.get("verdict") == "denied"


def request(events: Iterable[Any], request_id: str) -> dict[str, Any]:
    """What happened in one request: every event it left (a job's included), oldest first."""
    if not request_id:
        raise ValueError("name the request id")
    rows = [[e.at, e.kind, e.actor, e.ids.visit, e.ids.job, what(e)] for e in events if e.ids.request == request_id]
    return _result("request", f"What happened in request {request_id}", ["at", "kind", "actor", "visit", "job", "what"], rows)


def project_story(events: Iterable[Any], records: Iterable[Any] = (), outcomes: Iterable[Any] = (), *, project_id: str = "") -> dict[str, Any]:
    """A project's story in order: its moves, stage runs (with the run id), decisions and feedback, then the ledger's counts."""
    if not project_id:
        raise ValueError("name the project")
    rows = [[e.at, e.kind, e.ids.run, what(e)] for e in events
            if e.ids.project == project_id and e.kind.startswith(("stage.", "graph.", "run.", "solution.", "decision.", "outcome.", "feedback.", "export."))]
    mine = [r for r in records if r.project == project_id]
    counts = Counter(r.point for r in mine)
    rows.append(["", "ledger", None, f"{len(mine)} decisions at {len(counts)} points; {sum(1 for o in outcomes if o.project == project_id)} outcome rows"])
    return _result("project", f"The story of project {project_id}", ["at", "kind", "run", "what"], rows)


def rejections(events: Iterable[Any], *, since: str = "") -> dict[str, Any]:
    """Where a check, a rule or a person turned something down, most often first: by kind, and by the point, move, tool or rule."""
    counts: Counter = Counter()
    last: dict[tuple, str] = {}
    for e in events:
        if e.at < since or not refused(e):
            continue
        p = e.payload
        where = p.get("point") or p.get("move") or p.get("tool") or p.get("target") or "—"
        why = p.get("rule") or p.get("code") or ("; ".join(map(str, p["failed_checks"])) if p.get("failed_checks") else None) or "—"
        key = (e.kind, str(where), str(why)[:80])
        counts[key] += 1
        last[key] = max(last.get(key, ""), e.at)
    return _result("rejections", "Refusals", ["kind", "where", "rule", "times", "last"], [[k, w, r, n, last[(k, w, r)]] for (k, w, r), n in counts.most_common()])


def failed_calls(events: Iterable[Any], *, since: str = "", checks: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """Model calls that failed: the request did not return ok (a transport or provider error, from the events) and, beyond
    those, the replies that came back but failed the code that checks them (``checks``: the host's per purpose, tier and model
    failure counts, as ``usage.failing`` gives them)."""
    counts: Counter = Counter()
    total: Counter = Counter()
    for e in events:
        if e.kind != "model.request" or e.at < since:
            continue
        purpose = str(e.payload.get("purpose") or "?")
        total[purpose] += 1
        outcome = str(e.payload.get("outcome") or "ok")
        if outcome != "ok":
            counts[(purpose, outcome[:80])] += 1
    rows = [[p, o, n, total[p]] for (p, o), n in counts.most_common()]
    rows += [[c.get("purpose"), f"failed its check: {str(c.get('last_reason') or 'no reason kept')[:60]}", c.get("failed"), c.get("checked")] for c in checks]
    return _result("failed_calls", "Model calls that failed", ["purpose", "outcome", "times", "of calls"], rows)


def overrides(records: Iterable[Any], *, since: str = "") -> dict[str, Any]:
    """Decisions where a person chose another value than the rule's (or a model's), by point: proposed, chosen, how often."""
    counts: Counter = Counter()
    for r in records:
        if r.used != "person" or r.at < since:
            continue
        proposed = next((a.value for a in r.answers if a.by in ("rule", "classifier", "model_role") and a.value is not None and a.source != "shadow"), None)
        if proposed is not None and proposed != r.value_used:
            counts[(r.point, str(proposed)[:60], str(r.value_used)[:60])] += 1
    return _result("overrides", "Decisions a person overrode", ["point", "the rule said", "the person chose", "times"], [[p, a, b, n] for (p, a, b), n in counts.most_common()])


def outcomes_by_point(records: Iterable[Any], outcomes: Iterable[Any]) -> dict[str, Any]:
    """Whether decisions turned out well, per point (23.5): well, badly, unknown and no outcome yet."""
    per = oc.by_point(records, outcomes)
    return _result("outcomes", "Did the decisions turn out well?", ["point", "well", "badly", "unknown", "no outcome yet"],
                   [[p, v["well"], v["badly"], v["unknown"], v["none"]] for p, v in sorted(per.items(), key=lambda kv: -(kv[1]["well"] + kv[1]["badly"]))])


def cost(events: Iterable[Any], *, since: str = "") -> dict[str, Any]:
    """Model spend per ISO week and currency, from the model requests."""
    per: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.kind != "model.request" or e.at < since:
            continue
        year, week, _ = datetime.fromisoformat(e.at).isocalendar()
        cell = per[(f"{year}-W{week:02d}", str(e.payload.get("currency") or "?"))]
        cell[0] += float(e.payload.get("cost") or 0.0)
        cell[1] += 1
    return _result("cost", "Model spend per week", ["week", "currency", "spend", "requests"], [[w, c, round(v[0], 4), v[1]] for (w, c), v in sorted(per.items())])


def training(events: Iterable[Any], records: Iterable[Any] = (), labels: Iterable[Any] = (), outcomes: Iterable[Any] = (), *,
             event_levels: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """The records by the level they were written under (only `improve` and `train` rows can ever be exported, and then only if
    the switch is on and the project allows it) and the labels by source. A host with many events passes ``event_levels``
    (its own count by level) instead of the events."""
    rows = [["events", level, n] for level, n in sorted((event_levels if event_levels is not None else Counter(e.consent for e in events)).items())]
    rows += [["decisions", level, n] for level, n in sorted(Counter(r.consent for r in records).items())]
    rows += [["labels", source, n] for source, n in sorted(Counter(x.source for x in labels).items())]
    rows += [["outcomes", level, n] for level, n in sorted(Counter(o.consent for o in outcomes).items())]
    return _result("training", "Records by the level they were written under, and labels by source", ["what", "level or source", "rows"], rows)


def deletion(events: Iterable[Any], records: Iterable[Any] = (), labels: Iterable[Any] = (), outcomes: Iterable[Any] = (), *,
             project_id: Optional[str] = None, draft_id: Optional[str] = None, visit_id: Optional[str] = None, prompts: Optional[int] = None) -> dict[str, Any]:
    """What deleting a project, a draft or a session would remove from the record and the ledger (counts only)."""
    if not (project_id or draft_id or visit_id):
        raise ValueError("name a project, a draft or a visit")
    gone = [e for e in events if (project_id and e.ids.project == project_id) or (draft_id and e.ids.conversation == draft_id) or (visit_id and e.ids.visit == visit_id)]
    rows: list[list[Any]] = [["events", len({e.id for e in gone})]]
    if project_id or draft_id:
        mine = [r for r in records if (project_id and r.project == project_id) or (draft_id and r.draft == draft_id)]
        ids = {r.id for r in mine}
        rows += [["decisions", len(mine)], ["labels", sum(1 for x in labels if x.record in ids)],
                 ["outcomes", sum(1 for o in outcomes if o.record in ids or (project_id and o.project == project_id))]]
    if prompts is not None:
        rows.append(["prompts kept in the usage log", prompts])
    return _result("deletion", "What a deletion would remove", ["what", "rows"], rows)


__all__ = ["what", "refused", "request", "project_story", "rejections", "failed_calls", "overrides", "outcomes_by_point", "cost", "training", "deletion"]
