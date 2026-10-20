"""The database keeper (32.3): with ``DCLAB_DATABASE_URL`` set, a workspace's records live in PostgreSQL tables, not files, so
the file keepers would miss them. This keeper exports each workspace's rows of each table as JSON Lines and keeps them:

- an append-only table (events, decisions, labels, outcomes, the audit, steps, model requests ...) ordered by its time, so
  the store keeps only the rows added since the last run (``Store.keep_log``); a row order that changed is kept whole again;
- a table whose rows change (projects, drafts, intern sessions, jobs ...) as a whole snapshot, a new version when it changed.

Sign-ins, tokens, users and counters are never exported as rows (credentials, contacts): the people keeper (``people.py``, 32.32) keeps
each person and session by pseudonymous id and hashes, named columns only; the counters hold nothing to learn from. The workspace's
name in the store is its folder's (the same one the file keepers use) when its folder is known, else a hash of its id.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterator

from . import Claim, Keeper, Source

APPEND_ONLY = {  # table -> its order (time first, then a key that breaks ties)
    "experience_event": ("at", "seq"), "decision_record": ("at", "id"), "decision_label": ("at", "id"), "decision_outcome": ("at", "id"),
    "audit_event": ("at", "id"), "agent_step": ("at", "id"), "model_request": ("at", "id"), "model_output_check": ("at", "id"),
    "model_shadow": ("at", "id"), "draft_event": ("seq",), "activity": ("at",), "transition": ("at",),
}
SNAPSHOT = {"project": ("id",), "draft": ("id",), "intern_session": ("id",), "job": ("id",), "stage_record": ("project_id", "stage"),
            "workspace_lesson": ("id",), "model_routing": ("workspace_id",), "stored_file": ("id",)}
NEVER = ("user_session", "api_token", "app_user", "membership", "usage_counter", "model_pause")


def _engine() -> Any:
    from ...storage import db

    return db.engine() if db.database_url(required=False) else None


def _rows(conn: Any, table: Any, workspace_id: str, order: tuple[str, ...]) -> list[dict[str, Any]]:
    import sqlalchemy as sa

    q = sa.select(table).where(table.c.workspace_id == workspace_id)
    for col in order:
        if col in table.c:
            q = q.order_by(table.c[col])
    return [dict(r._mapping) for r in conn.execute(q)]


def _name_and_consent(key: str, base: Path) -> tuple[str, str]:
    from ..layout import Layout
    from .workspaces import _consent, discover

    folder = Path(key)
    if folder.exists():
        for ws in discover(base):
            if ws.root.resolve() == folder.resolve():
                return ws.name, ws.consent
        return Layout.workspace(key), _consent(folder, owner=False)
    return Layout.workspace(key), "operate"


def sources(env: Any) -> Iterator[Source]:
    """Every workspace's rows, table by table; nothing when no database is configured."""
    engine = _engine()
    if engine is None:
        return
    from ...storage import models
    from .workspaces import AGENT_RUNS

    base = Path(env["agent_runs"]) if isinstance(env, dict) and env.get("agent_runs") else AGENT_RUNS
    only = Path(env["only"]).resolve() if isinstance(env, dict) and env.get("only") else None
    tmp = Path(tempfile.mkdtemp(prefix="dclab-db-keep-"))
    try:
        yield from _export(engine, models, base, only, tmp)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)  # each export was kept (read whole) when its source was yielded


def _export(engine: Any, models: Any, base: Path, only: Any, tmp: Path) -> Iterator[Source]:
    with engine.connect() as conn:
        for wid, key in conn.execute(models.workspace.select().with_only_columns(models.workspace.c.id, models.workspace.c.key)):
            if only is not None and Path(key).resolve() != only:
                continue
            name, consent = _name_and_consent(key, base)
            for table_name, order in {**APPEND_ONLY, **SNAPSHOT}.items():
                table = models.metadata.tables.get(table_name)
                if table is None or "workspace_id" not in table.c:
                    continue
                rows = _rows(conn, table, wid, order)
                if not rows:
                    continue
                if consent == "operate":
                    yield Source(name, f"db/{table_name}~counts.json", json.dumps({"rows": len(rows)}).encode(), consent=consent,
                                 note="counts only: this workspace's consent does not allow keeping its content")
                    continue
                path = tmp / f"{name}-{table_name}.jsonl"
                with open(path, "w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n")
                from .agents import redact

                yield Source(name, f"db/{table_name}.jsonl", path, consent=consent, log=table_name in APPEND_ONLY, transform=redact)


KEEPER = Keeper("database", (Claim("sessions", "db/experience_event"), Claim("decisions", "db/decision_"), Claim("audit", "db/audit_event"),
                             Claim("actions", "db/agent_step"), Claim("model_requests", "db/model_"), Claim("conversations", "db/draft"),
                             Claim("conversations", "db/intern_session"), Claim("workflows", "db/project"), Claim("workflows", "db/stage_record"),
                             Claim("workflows", "db/activity"), Claim("workflows", "db/transition"), Claim("tasks", "db/job"),
                             Claim("students", "db/workspace_lesson"), Claim("audit", "db/stored_file")),
                sources, ("m1_students", "m2_workflow_decisions", "m3_ml_decisions", "m4_workflow_llm", "m6_stepwise_decisions"),
                "every PostgreSQL table of every workspace (with DCLAB_DATABASE_URL)")

__all__ = ["KEEPER", "sources", "APPEND_ONLY", "SNAPSHOT", "NEVER"]
