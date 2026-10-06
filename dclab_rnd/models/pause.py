"""The model pause, shared by every app instance and worker (package 10.5).

After a refusal that will not go away by itself (no credit, a refused key, an unknown model) requests to that endpoint
and model stop for a while (``client.PAUSE_SECONDS``): each attempt would make a person wait for the same refusal.
The pause was a variable of one process; with several instances each found out on its own. With
``DCLAB_DATABASE_URL`` set it is a row of ``model_pause`` with an expiry, read before each request; without it, a
variable as before. The key names the endpoint, the model and a hash of the API key (never the key): a new key is
not paused by the old one's refusal.
"""

from __future__ import annotations

import hashlib
import sys
import time

_LOCAL: dict[str, tuple[float, str]] = {}


def key(base_url: str, model: str, api_key: str = "") -> str:
    digest = hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else "nokey"
    return f"{base_url} {model} {digest}"


def _url() -> str | None:
    from ..storage import db

    return db.database_url(required=False)


def get(name: str) -> tuple[float, str]:
    """(until, reason) of the pause on ``name``; (0, "") when there is none. A database that cannot be read pauses nothing."""
    local = _LOCAL.get(name, (0.0, ""))
    if local[0] <= time.time():
        _LOCAL.pop(name, None)
        local = (0.0, "")
    url = _url()
    if not url:
        return local
    try:
        import sqlalchemy as sa

        from ..storage import db
        from ..storage.models import model_pause as t

        with db.engine(url).connect() as c:
            row = c.execute(sa.select(t.c.until, t.c.reason).where(t.c.key == name, t.c.until > time.time())).first()
        shared = (float(row.until), row.reason) if row else (0.0, "")
        return max(shared, local)  # whichever ends later: a pause this process set stands even if sharing it failed
    except Exception as exc:  # noqa: BLE001 — never block a model request because the pause could not be read
        print(f"model pause: not read ({type(exc).__name__})", file=sys.stderr, flush=True)
        return local


def put(name: str, until: float, reason: str) -> None:
    _LOCAL[name] = (until, reason)  # this process pauses at once, whatever happens to the write below
    url = _url()
    if not url:
        return
    try:
        from sqlalchemy.dialects.postgresql import insert

        from ..storage import db
        from ..storage.models import model_pause as t

        with db.engine(url).begin() as c:
            c.execute(insert(t).values(key=name, until=until, reason=reason)
                      .on_conflict_do_update(index_elements=[t.c.key], set_={"until": until, "reason": reason}))
    except Exception as exc:  # noqa: BLE001
        print(f"model pause: not shared ({type(exc).__name__})", file=sys.stderr, flush=True)


def clear() -> None:
    """Forget every pause (a new key or endpoint was configured, or a test starts)."""
    _LOCAL.clear()
    url = _url()
    if not url:
        return
    try:
        import sqlalchemy as sa

        from ..storage import db
        from ..storage.models import model_pause as t

        with db.engine(url).begin() as c:
            c.execute(sa.delete(t))
    except Exception:  # noqa: BLE001 — the table may not exist yet (a database before 0011)
        pass
