"""The usage log: one entry per model request, whatever happened (package A1.1).

An entry holds the purpose, the tier, the model, the endpoint host, the project or draft, the tokens, the seconds,
the number of attempts and the outcome. It never holds a key, and the prompt only when DCLAB_LOG_PROMPTS=1.
With DCLAB_DATABASE_URL set the entries are rows of ``model_request``; otherwise lines of a JSON file in the workspace.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

FIELDS = ("at", "purpose", "tier", "model", "endpoint", "project_id", "draft_id", "input_tokens", "output_tokens",
          "seconds", "attempts", "outcome", "prompt", "cost_eur", "cost_basis")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


CHECK_FIELDS = ("at", "purpose", "tier", "model", "passed", "reason")


def _failing(entries: list[dict[str, Any]], at_least: int = 2) -> list[dict[str, Any]]:
    """Purpose and model pairs whose output failed its check at least ``at_least`` times, newest reason first."""
    groups: dict[tuple, dict[str, Any]] = {}
    for e in entries:
        key = (e.get("purpose"), e.get("tier"), e.get("model"))
        g = groups.setdefault(key, {"purpose": key[0], "tier": key[1], "model": key[2], "checked": 0, "failed": 0, "last_reason": None, "last_at": None})
        g["checked"] += 1
        if not e.get("passed"):
            g["failed"] += 1
            if not g["last_at"] or str(e.get("at")) >= g["last_at"]:
                g["last_reason"], g["last_at"] = e.get("reason"), str(e.get("at"))
    return sorted((g for g in groups.values() if g["failed"] >= at_least), key=lambda g: -g["failed"])


class UsageLog(Protocol):
    def record(self, entry: dict[str, Any]) -> None: ...
    def recent(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def totals(self, since: str | None = None, project_id: str | None = None) -> dict[str, Any]: ...
    def spent(self, since: str, project_id: str | None = None) -> float: ...
    def record_check(self, entry: dict[str, Any]) -> None: ...
    def failing(self, since: str) -> list[dict[str, Any]]: ...


def _totals(entries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"requests": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "by_purpose": {}, "by_model": {},
                           "eur": 0.0, "priced": 0, "local": 0, "unpriced": 0, "by_project_eur": {}}
    for e in entries:
        ok = e.get("outcome") == "ok"
        out["requests"] += 1
        out["failed"] += 0 if ok else 1
        basis = e.get("cost_basis")
        out[{"price": "priced", "local": "local"}.get(basis, "unpriced")] += 1
        if e.get("cost_eur") is not None:
            out["eur"] += float(e["cost_eur"])
            key = str(e.get("project_id") or "(no project)")
            out["by_project_eur"][key] = round(out["by_project_eur"].get(key, 0.0) + float(e["cost_eur"]), 6)
        for key in ("input_tokens", "output_tokens"):
            out[key] += int(e.get(key) or 0)
        out["seconds"] += float(e.get("seconds") or 0)
        for group, name in (("by_purpose", e.get("purpose")), ("by_model", e.get("model"))):
            g = out[group].setdefault(str(name), {"requests": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0})
            g["requests"] += 1
            g["failed"] += 0 if ok else 1
            g["input_tokens"] += int(e.get("input_tokens") or 0)
            g["output_tokens"] += int(e.get("output_tokens") or 0)
    out["seconds"] = round(out["seconds"], 1)
    out["eur"] = round(out["eur"], 6)
    return out


def month_start(now: datetime | None = None) -> str:
    """The first moment of the current month (UTC), in the format the usage log's ``at`` uses."""
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")


class FileUsage:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.checks = self.path.with_name(self.path.stem + "_checks.jsonl")
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps({k: entry.get(k) for k in FIELDS}, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self._all()[-limit:]))

    def record_check(self, entry: dict[str, Any]) -> None:
        line = json.dumps({k: entry.get(k) for k in CHECK_FIELDS}, ensure_ascii=False, default=str)
        with self._lock, self.checks.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def failing(self, since: str) -> list[dict[str, Any]]:
        if not self.checks.is_file():
            return []
        rows = [json.loads(line) for line in self.checks.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _failing([r for r in rows if str(r.get("at")) >= since])

    def spent(self, since: str, project_id: str | None = None) -> float:
        return round(sum(float(e.get("cost_eur") or 0) for e in self._all()
                         if str(e.get("at")) >= since and (project_id is None or e.get("project_id") == project_id)), 6)

    def totals(self, since: str | None = None, project_id: str | None = None) -> dict[str, Any]:
        return _totals([e for e in self._all() if (not since or str(e.get("at")) >= since) and (not project_id or e.get("project_id") == project_id)])


class PgUsage:
    def __init__(self, workspace_id: str, url: str | None = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def record(self, entry: dict[str, Any]) -> None:
        import sqlalchemy as sa

        from ..storage.models import model_request

        with self.engine.begin() as c:
            c.execute(sa.insert(model_request).values(workspace_id=self.workspace_id, **{k: entry.get(k) for k in FIELDS}))

    def _query(self, since: str | None = None, project_id: str | None = None, limit: int | None = None):
        import sqlalchemy as sa

        from ..storage.models import model_request as t

        q = sa.select(*[t.c[k] for k in FIELDS]).where(t.c.workspace_id == self.workspace_id)
        if since:
            q = q.where(t.c.at >= since)
        if project_id:
            q = q.where(t.c.project_id == project_id)
        q = q.order_by(t.c.id.desc())
        if limit is not None:
            q = q.limit(limit)
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(q)]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._query(limit=limit)

    def record_check(self, entry: dict[str, Any]) -> None:
        import sqlalchemy as sa

        from ..storage.models import model_output_check

        with self.engine.begin() as c:
            c.execute(sa.insert(model_output_check).values(workspace_id=self.workspace_id, **{k: entry.get(k) for k in CHECK_FIELDS}))

    def failing(self, since: str) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from ..storage.models import model_output_check as t

        with self.engine.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(sa.select(*[t.c[k] for k in CHECK_FIELDS]).where(t.c.workspace_id == self.workspace_id, t.c.at >= since))]
        return _failing(rows)

    def spent(self, since: str, project_id: str | None = None) -> float:
        import sqlalchemy as sa

        from ..storage.models import model_request as t

        q = sa.select(sa.func.coalesce(sa.func.sum(t.c.cost_eur), 0.0)).where(t.c.workspace_id == self.workspace_id, t.c.at >= since)
        if project_id is not None:
            q = q.where(t.c.project_id == project_id)
        with self.engine.connect() as c:
            return round(float(c.execute(q).scalar() or 0.0), 6)

    def totals(self, since: str | None = None, project_id: str | None = None) -> dict[str, Any]:
        return _totals(self._query(since, project_id))


def open_usage(home: Path) -> UsageLog:
    from ..storage import db

    if db.database_url(required=False):
        return PgUsage(db.workspace_for(Path(home)))
    return FileUsage(Path(home) / "model_requests.jsonl")
