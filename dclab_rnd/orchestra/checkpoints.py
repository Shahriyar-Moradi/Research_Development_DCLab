"""Checkpoints bound to the workspace store (28.1): a walk pauses for a person or dies with the process, and resumes from
its last node by its thread id.

PostgreSQL when the workspace runs on it (``DCLAB_DATABASE_URL``: the checkpoint tables live in the same database, made
by ``setup()`` once), else SQLite in ``<home>/orchestra/checkpoints.sqlite3``. ``open_checkpointer`` yields the saver;
``where`` says which one serves, for the Admin page and the tests.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import FOLDER


def where(home: Path, database_url: Optional[str] = None) -> dict[str, Any]:
    """Which store the checkpoints use for this home (without opening it)."""
    url = database_url if database_url is not None else os.environ.get("DCLAB_DATABASE_URL") or None
    if url:
        return {"kind": "postgres"}
    return {"kind": "sqlite", "path": str(Path(home) / FOLDER / "checkpoints.sqlite3")}


@contextmanager
def open_checkpointer(home: Path, database_url: Optional[str] = None) -> Iterator[Any]:
    """The checkpoint saver of this workspace, open for the block."""
    place = where(home, database_url)
    if place["kind"] == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver

        from ..storage.notify import dsn  # the repository's SQLAlchemy URL as psycopg's DSN

        url = database_url if database_url is not None else os.environ["DCLAB_DATABASE_URL"]
        with PostgresSaver.from_conn_string(dsn(url)) as saver:
            saver.setup()  # idempotent: the tables and their migrations
            yield saver
        return
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(place["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


__all__ = ["where", "open_checkpointer"]
