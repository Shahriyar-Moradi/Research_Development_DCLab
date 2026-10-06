"""Rate limits and quotas per user and per workspace (package 10.6).

With about 1,000 users one person must not starve the rest. Four limits, each per user and per workspace:

    requests       API requests a minute                  DCLAB_LIMIT_REQUESTS_PER_MINUTE (600) / DCLAB_LIMIT_WORKSPACE_REQUESTS_PER_MINUTE (3000)
    upload_bytes   bytes brought in a day (uploads, imports)  DCLAB_LIMIT_UPLOAD_MB_PER_DAY (2000) / DCLAB_LIMIT_WORKSPACE_UPLOAD_MB_PER_DAY (10000)
    jobs           jobs queued or running at once         DCLAB_LIMIT_JOBS_PER_USER (4) / DCLAB_LIMIT_JOBS_PER_WORKSPACE (16)
    model_eur      model spend a month (priced models)    DCLAB_LIMIT_USER_MONTHLY_EUR (25) / DCLAB_WORKSPACE_MONTHLY_EUR (the gateway's cap)

Counters live in PostgreSQL (``usage_counter``, one row per scope, kind and window, incremented in one statement), so
every app instance counts the same. A request over a limit gets 429 with what was exceeded, by whom, and when it
resets (``LimitExceeded.body``). Limits apply when accounts are on (``DCLAB_AUTH``): one owner on their own machine is
limited by nothing but the machine. A value of 0 switches that limit off.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

WINDOWS = {"requests": 60, "upload_bytes": 86400}
DEFAULTS = {  # (per user, per workspace)
    "requests": (("DCLAB_LIMIT_REQUESTS_PER_MINUTE", 600), ("DCLAB_LIMIT_WORKSPACE_REQUESTS_PER_MINUTE", 3000)),
    "upload_bytes": (("DCLAB_LIMIT_UPLOAD_MB_PER_DAY", 2000), ("DCLAB_LIMIT_WORKSPACE_UPLOAD_MB_PER_DAY", 10000)),
    "jobs": (("DCLAB_LIMIT_JOBS_PER_USER", 4), ("DCLAB_LIMIT_JOBS_PER_WORKSPACE", 16)),
    "model_eur": (("DCLAB_LIMIT_USER_MONTHLY_EUR", 25), (None, None)),  # the workspace's monthly cap is the gateway's (models/gateway.py)
}
WORDS = {"requests": "requests a minute", "upload_bytes": "MB brought in a day", "jobs": "jobs running at once", "model_eur": "euros of model use this month"}


def _number(name: str | None, default: float | None) -> float | None:
    if name is None:
        return None
    raw = (os.environ.get(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default  # a bad value never stops the server: the safe default holds
    return None if value is None or value <= 0 else value


def limits() -> dict[str, dict[str, float | None]]:
    """Every limit, per user and per workspace, in its unit (bytes for uploads); None is no limit."""
    out = {}
    for kind, ((user_env, user_default), (ws_env, ws_default)) in DEFAULTS.items():
        user, workspace = _number(user_env, user_default), _number(ws_env, ws_default)
        if kind == "upload_bytes":
            user, workspace = (v * 1024 * 1024 if v else None for v in (user, workspace))
        if kind == "model_eur":
            workspace = _number("DCLAB_WORKSPACE_MONTHLY_EUR", None)  # the gateway's cap, read leniently: a bad value is no limit here
        out[kind] = {"user": user, "workspace": workspace}
    return out


@dataclass(eq=False)
class LimitExceeded(Exception):
    kind: str
    scope: str            # user or workspace
    limit: float
    used: float
    resets_at: float      # seconds since the epoch; for jobs, when one ends (unknown: now + a minute)

    def __str__(self) -> str:
        return self.body["message"]

    @property
    def body(self) -> dict[str, Any]:
        unit = WORDS[self.kind]
        shown = (lambda v: f"{v / 1024 / 1024:,.0f}") if self.kind == "upload_bytes" else (lambda v: f"{v:,.2f}" if self.kind == "model_eur" else f"{v:,.0f}")
        who = "you have" if self.scope == "user" else "this workspace has"
        when = datetime.fromtimestamp(self.resets_at, timezone.utc).isoformat(timespec="seconds")
        message = (f"{who.capitalize()} reached the limit of {shown(self.limit)} {unit}; "
                   + ("it frees when one of them ends" if self.kind == "jobs" else f"it resets at {when}"))
        return {"error": "limit", "limit": self.kind, "scope": self.scope, "allowed": self.limit, "used": round(self.used, 2),
                "resets_at": when, "retry_after": max(1, math.ceil(self.resets_at - time.time())), "message": message}


def on() -> bool:
    from .settings import Settings

    return Settings.load().auth != "none"


def _window(kind: str, now: float) -> float:
    size = WINDOWS[kind]
    return math.floor(now / size) * size


def charge(kind: str, amount: float, user_id: str | None, workspace_id: str | None, now: float | None = None) -> None:
    """Count ``amount`` against the user's and the workspace's limit of ``kind``; raise LimitExceeded (and count
    nothing more) when either would pass it. Counted in one statement each, so instances racing still agree."""
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import insert

    from .storage import db
    from .storage.models import usage_counter as t

    now = time.time() if now is None else now
    window = _window(kind, now)
    caps = limits()[kind]
    scopes = [("user", f"user:{user_id}", caps["user"]), ("workspace", f"workspace:{workspace_id}", caps["workspace"])]
    with db.engine().begin() as c:
        for scope, key, cap in scopes:
            if cap is None or key.endswith(":None"):
                continue
            count = c.execute(insert(t).values(scope=key, kind=kind, window=window, amount=amount)
                              .on_conflict_do_update(index_elements=[t.c.scope, t.c.kind, t.c.window], set_={"amount": t.c.amount + amount})
                              .returning(t.c.amount)).scalar()
            if count > cap:  # raising inside the transaction rolls it back: a refused request does not eat the allowance
                raise LimitExceeded(kind, scope, cap, count - amount, window + WINDOWS[kind])
        if random.random() < 0.01:  # now and then, the windows that ended more than two days ago go
            c.execute(sa.delete(t).where(t.c.window < now - 2 * 86400))


def used(kind: str, user_id: str | None, workspace_id: str | None, now: float | None = None) -> dict[str, float]:
    import sqlalchemy as sa

    from .storage import db
    from .storage.models import usage_counter as t

    window = _window(kind, time.time() if now is None else now)
    with db.engine().connect() as c:
        rows = dict(c.execute(sa.select(t.c.scope, t.c.amount).where(t.c.kind == kind, t.c.window == window,
                                                                      t.c.scope.in_([f"user:{user_id}", f"workspace:{workspace_id}"]))).all())
    return {"user": float(rows.get(f"user:{user_id}", 0)), "workspace": float(rows.get(f"workspace:{workspace_id}", 0))}


@contextlib.contextmanager
def job_slot(workspace_id: str | None):
    """Held while a job is counted and queued, so two requests at once cannot both take the last slot (an advisory
    lock per workspace, on its own connection; the queued row is committed before it is let go)."""
    import sqlalchemy as sa

    from .storage import db

    key = int.from_bytes(hashlib.sha1(f"dclab-job-slot:{workspace_id}".encode()).digest()[:8], "big", signed=True)
    connection = db.engine().connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        connection.execute(sa.text("select pg_advisory_lock(:k)"), {"k": key})
        yield
    finally:
        try:
            connection.execute(sa.text("select pg_advisory_unlock(:k)"), {"k": key})
        finally:
            connection.close()


def check_jobs(job_store: Any, user_id: str | None, workspace_id: str | None) -> None:
    """Before a job is queued: the user's and the workspace's jobs queued or running stay under their limits."""
    caps = limits()["jobs"]
    active = job_store.list(active=True, limit=10_000)
    mine = sum(1 for j in active if (j.get("user") or {}).get("user_id") == user_id) if user_id else 0
    for scope, cap, count in (("user", caps["user"], mine), ("workspace", caps["workspace"], len(active))):
        if cap is not None and count >= cap and (scope == "workspace" or user_id):
            raise LimitExceeded("jobs", scope, cap, count, time.time() + 60)


def summary(user_id: str | None, workspace_id: str | None, spent_user: float | None, spent_workspace: float | None,
            jobs_mine: int, jobs_all: int) -> dict[str, Any]:
    """What the Admin page shows: each limit, per user and per workspace, with what is used now."""
    caps = limits()
    req, up = (used(k, user_id, workspace_id) for k in ("requests", "upload_bytes"))
    rows = {"requests": (req["user"], req["workspace"]), "upload_bytes": (up["user"], up["workspace"]), "jobs": (jobs_mine, jobs_all),
            "model_eur": (spent_user, spent_workspace)}
    return {"on": True, "limits": [{"kind": k, "words": WORDS[k], "user": {"used": rows[k][0], "limit": caps[k]["user"]},
                                    "workspace": {"used": rows[k][1], "limit": caps[k]["workspace"]}} for k in DEFAULTS]}
