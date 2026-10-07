"""The database connection, its pool, and the migrations."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import sqlalchemy as sa

def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:  # a bad value: the server refuses to start and says why (Settings.problems); a script keeps the default
        return default


POOL_SIZE = _int("DCLAB_DB_POOL_SIZE", 10)
MAX_OVERFLOW = _int("DCLAB_DB_MAX_OVERFLOW", 20)
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


MIGRATION_LOCK = 0x44434C4142  # "DCLAB": the advisory lock every instance takes before migrating


def upgrade(url: str | None = None, revision: str = "head") -> None:
    """Migrate to ``revision``, one process at a time: several app instances start together in a cloud (package 12.6),
    and each runs the upgrade first; the second waits for the first, then finds nothing left to do."""
    from alembic import command

    url = url or database_url()
    with engine(url).connect().execution_options(isolation_level="AUTOCOMMIT") as connection:  # not idle in a transaction meanwhile
        connection.execute(sa.text("select pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK})
        try:
            command.upgrade(_config(url), revision)
        finally:
            connection.execute(sa.text("select pg_advisory_unlock(:key)"), {"key": MIGRATION_LOCK})


def current(url: str | None = None) -> str | None:
    from alembic.runtime.migration import MigrationContext

    with engine(url).connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def head() -> str:
    """The newest migration this code knows."""
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(_config("postgresql://unused")).get_current_head()


def reachable(url: str | None = None) -> str | None:
    """None when the database answers, otherwise a plain-words reason (never the URL: it can hold a password)."""
    try:
        with engine(url).connect() as connection:
            connection.execute(sa.text("select 1"))
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: the database did not answer"


MARKER = ".dclab_workspace"  # in the workspace folder: which workspace it is, wherever the folder goes (package 9.3)


def workspace_key(home: Path) -> str:
    """The key of the workspace that lives in ``home``. It was the folder's path; it is now written into the folder
    the first time (the same path, so an existing workspace keeps its id), so a moved folder is still the same workspace."""
    home = Path(home)
    marker = home / MARKER
    try:
        found = marker.read_text(encoding="utf-8").strip()
        if found:
            return found
    except OSError:
        pass
    key = str(home.resolve())
    try:
        home.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(key + "\n", encoding="utf-8")
        os.replace(temporary, marker)
    except OSError:  # a read-only folder keeps the old behaviour: its path is its key
        pass
    return key


def workspace_for(home: Path, url: str | None = None) -> str:
    """The id of the workspace in ``home`` (its marker, else its path), created on first use."""
    return workspace(workspace_key(home), Path(home).name, url)


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
