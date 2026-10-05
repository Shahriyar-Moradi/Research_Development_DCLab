"""Helpers for tests that need the product's PostgreSQL database.

They use DCLAB_TEST_DATABASE_URL, or the local ``dclab_test`` database when the server on this machine has it, and are
skipped when neither answers. The database is emptied before each use: point it only at a database that exists for tests.
"""
import getpass
import os
import unittest

import sqlalchemy as sa

_READY: dict[str, str | None] = {}


def url() -> str | None:
    candidate = os.environ.get("DCLAB_TEST_DATABASE_URL") or f"postgresql+psycopg://{getpass.getuser()}@/dclab_test"
    if candidate not in _READY:
        try:
            from dclab_rnd.storage import db

            problem = db.reachable(candidate)
            if problem is None:
                db.upgrade(candidate)
            _READY[candidate] = None if problem else candidate
        except Exception:  # noqa: BLE001 — no driver, no server, no database: the tests are skipped
            _READY[candidate] = None
    return _READY[candidate]


def require() -> str:
    found = url()
    if found is None:
        raise unittest.SkipTest("no PostgreSQL test database (createdb dclab_test, or set DCLAB_TEST_DATABASE_URL)")
    return found


def empty(found: str) -> None:
    from dclab_rnd.storage import db
    from dclab_rnd.storage.models import TABLES

    with db.engine(found).begin() as connection:
        connection.execute(sa.text("truncate " + ", ".join(TABLES) + " restart identity cascade"))
