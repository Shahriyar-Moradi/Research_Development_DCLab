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
          "seconds", "attempts", "outcome", "prompt")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class UsageLog(Protocol):
    def record(self, entry: dict[str, Any]) -> None: ...
    def recent(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def totals(self, since: str | None = None, project_id: str | None = None) -> dict[str, Any]: ...


def _totals(entries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"requests": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "by_purpose": {}, "by_model": {}}
    for e in entries:
        ok = e.get("outcome") == "ok"
        out["requests"] += 1
        out["failed"] += 0 if ok else 1
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
    return out


class FileUsage:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def totals(self, since: str | None = None, project_id: str | None = None) -> dict[str, Any]:
        return _totals(self._query(since, project_id))


def open_usage(home: Path) -> UsageLog:
    from ..storage import db

    if db.database_url(required=False):
        return PgUsage(db.workspace(str(Path(home).resolve()), Path(home).name))
    return FileUsage(Path(home) / "model_requests.jsonl")
