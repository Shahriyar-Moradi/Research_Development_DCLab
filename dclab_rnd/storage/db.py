"""The database connection, its pool, and the migrations."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import sqlalchemy as sa

POOL_SIZE = int(os.environ.get("DCLAB_DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.environ.get("DCLAB_DB_MAX_OVERFLOW", "20"))
_ENGINES: dict[str, sa.Engine] = {}
_GUARD = threading.Lock()


def database_url(required: bool = True) -> str | None:
    url = os.environ.get("DCLAB_DATABASE_URL", "").strip()
    if not url and required:
        raise RuntimeError("DCLAB_DATABASE_URL is not set. Example: postgresql+psycopg://USER@/dclab_dev (see .env.example)")
    return url or None


def _finite(value: Any) -> Any:
    """JSONB has no NaN or Infinity; a score that was not a number is stored as null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(value), ensure_ascii=False, default=str, allow_nan=False)


def engine(url: str | None = None) -> sa.Engine:
    """One pooled engine per URL for the whole process. Sized for several app instances behind PgBouncer in production."""
    url = url or database_url()
    with _GUARD:
        if url not in _ENGINES:
            _ENGINES[url] = sa.create_engine(url, pool_size=POOL_SIZE, max_overflow=MAX_OVERFLOW, pool_pre_ping=True, json_serializer=dumps)
        return _ENGINES[url]


def dispose() -> None:
    with _GUARD:
        for item in _ENGINES.values():
            item.dispose()
        _ENGINES.clear()


def _config(url: str):
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    config.attributes["url"] = url
    return config


def upgrade(url: str | None = None, revision: str = "head") -> None:
    from alembic import command

    command.upgrade(_config(url or database_url()), revision)


def current(url: str | None = None) -> str | None:
    from alembic.runtime.migration import MigrationContext

    with engine(url).connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def reachable(url: str | None = None) -> str | None:
    """None when the database answers, otherwise a plain-words reason (never the URL: it can hold a password)."""
    try:
        with engine(url).connect() as connection:
            connection.execute(sa.text("select 1"))
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: the database did not answer"


def workspace(key: str, name: str | None = None, url: str | None = None) -> str:
    """The id of the workspace for ``key`` (a folder today), created on first use."""
    import hashlib

    from .models import workspace as table

    wid = "w" + hashlib.sha1(key.encode()).hexdigest()[:15]
    with engine(url).begin() as connection:
        connection.execute(sa.text("insert into workspace (id, key, name) values (:id, :key, :name) on conflict (key) do nothing"),
                           {"id": wid, "key": key, "name": name or Path(key).name or "workspace"})
        found = connection.execute(sa.select(table.c.id).where(table.c.key == key)).scalar_one()
    return found
