"""Whether the workspace lives in PostgreSQL, read without SQLAlchemy: the agent environment (.venv-agent) has no
SQLAlchemy and keeps its records in files, so the stores ask here before they import the database layer."""

from __future__ import annotations

import os


def database_url(required: bool = True) -> str | None:
    url = os.environ.get("DCLAB_DATABASE_URL", "").strip()
    if not url and required:
        raise RuntimeError("DCLAB_DATABASE_URL is not set. Example: postgresql+psycopg://USER@/dclab_dev (see .env.example)")
    return url or None
