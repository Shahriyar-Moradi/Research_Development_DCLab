"""The shadow log (package A6.4): how a shadow model's answers compare with the served model's.

One row per shadowed request: which tool each model chose (a name, or ``text`` for an answer without a tool call),
whether the tools and their arguments were the same, and whether the shadow failed. Never a prompt, an argument or an
answer: the row says that they differed, not what they said. ``agreement()`` gives the rates per purpose and per day.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

FIELDS = ("at", "purpose", "primary_tier", "primary_model", "shadow_tier", "shadow_model", "project_id", "draft_id",
          "primary_tool", "shadow_tool", "same_tool", "same_arguments", "outcome", "seconds", "input_tokens", "output_tokens")


def first_tool(reply: dict[str, Any] | None, offered: set[str] | None = None) -> tuple[str, Any]:
    """The first tool call's name and arguments. A name is a model's text: it is kept only when it is one of the tools
    the request offered (anything else is "other"), so a row never holds a value the model saw."""
    calls = (reply or {}).get("tool_calls") or []
    if not calls:
        return "text", None
    name = str(calls[0].get("name"))
    return (name if offered is not None and name in offered else "other"), calls[0].get("arguments")


def compare(primary: dict[str, Any], shadow: dict[str, Any] | None, offered: set[str] | None = None) -> dict[str, Any]:
    p_tool, p_args = first_tool(primary, offered)
    if shadow is None:
        return {"primary_tool": p_tool, "shadow_tool": None, "same_tool": None, "same_arguments": None}
    s_tool, s_args = first_tool(shadow, offered)
    same_tool = p_tool == s_tool
    same_args = (json.dumps(p_args, sort_keys=True, default=str) == json.dumps(s_args, sort_keys=True, default=str)) if same_tool and p_tool != "text" else None
    return {"primary_tool": p_tool, "shadow_tool": s_tool, "same_tool": same_tool, "same_arguments": same_args}


class ShadowLog(Protocol):
    def record(self, row: dict[str, Any]) -> None: ...
    def rows(self, since: str | None = None) -> list[dict[str, Any]]: ...


class FileShadows:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, row: dict[str, Any]) -> None:
        line = json.dumps({k: row.get(k) for k in FIELDS}, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def rows(self, since: str | None = None) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is None or (row.get("at") or "") >= since:
                out.append(row)
        return out


class PgShadows:
    def __init__(self, workspace_id: str, url: str | None = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def record(self, row: dict[str, Any]) -> None:
        import sqlalchemy as sa

        from ..storage.models import model_shadow as t

        with self.engine.begin() as c:
            c.execute(sa.insert(t).values(workspace_id=self.workspace_id, **{k: row.get(k) for k in FIELDS}))

    def rows(self, since: str | None = None) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from ..storage.models import model_shadow as t

        q = sa.select(*[t.c[k] for k in FIELDS]).where(t.c.workspace_id == self.workspace_id)
        if since is not None:
            q = q.where(t.c.at >= since)
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(q.order_by(t.c.at, t.c.id))]


def open_shadows(home: Path) -> ShadowLog:
    from ..storage.url import database_url

    if database_url(required=False):
        from ..storage import db

        return PgShadows(db.workspace_for(Path(home)))
    return FileShadows(Path(home) / "model_shadow.jsonl")


def agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per purpose and shadow model: requests, how often the same tool was chosen, the same arguments, the shadow's
    failures, and the same per day (the disagreement log a person reads before switching)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("purpose") or "?", r.get("shadow_model") or "?")].append(r)
    out = []
    for (purpose, model), rs in sorted(groups.items()):
        answered = [r for r in rs if r.get("same_tool") is not None]
        with_args = [r for r in answered if r.get("same_arguments") is not None]
        days: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            days[(r.get("at") or "")[:10]].append(r)
        out.append({"purpose": purpose, "shadow_model": model, "shadow_tier": rs[-1].get("shadow_tier"), "requests": len(rs),
                    "errors": sum(1 for r in rs if (r.get("outcome") or "ok") != "ok"),
                    "same_tool": round(sum(1 for r in answered if r["same_tool"]) / len(answered), 3) if answered else None,
                    "same_arguments": round(sum(1 for r in with_args if r["same_arguments"]) / len(with_args), 3) if with_args else None,
                    "disagreements": sum(1 for r in answered if not r["same_tool"] or r.get("same_arguments") is False),
                    "by_day": [{"day": d, "requests": len(v), "disagreements": sum(1 for r in v if r.get("same_tool") is False or r.get("same_arguments") is False)}
                               for d, v in sorted(days.items())],
                    "first": rs[0].get("at"), "last": rs[-1].get("at")})
    return out
