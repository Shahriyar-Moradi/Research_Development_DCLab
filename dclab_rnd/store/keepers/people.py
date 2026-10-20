"""The people keeper (32.32): with accounts on (``DCLAB_AUTH``, PostgreSQL), each workspace's people and their sessions, kept
so every record can be joined to whose it is (``ids.user``), for training and fine-tuning.

What is kept, per workspace (the same names the other keepers use), from the account tables by named columns only:

- ``people/person.jsonl``: a log of what happened to each person (joined, changed, removed): the user id (already a
  pseudonym), a hash of the email (to recognise the same person, never the email), how they sign in (password or single
  sign-on, never the secret), their role, whether they are active, when they joined, and how many API tokens they hold;
- ``people/session.jsonl``: the workspace's sessions as of this run: a hash of the session id, the user, and the start,
  last-seen and expiry times.

Never kept, and never read into memory beyond the hashing: a password or its hash, a session cookie or its digest, a request
token, an API token, its digest or prefix or name, a name, an email, the identity provider's subject. A removed person's
last record stays in the log as ``removed`` (nothing is rewritten: only the owner's audited forget removes from the store).
A workspace whose consent is not ``improve`` or ``train`` keeps counts only, as the database keeper does.

The email hash is salted with a salt that stays on this machine (``<store>.people/state.json``, beside the store, never inside
it, so it is never pushed): a dictionary of emails cannot be run against the pushed hashes. The same file holds the last
state seen, from which the log of changes is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import Claim, Keeper, Source


def _state_dir(env: Any = None) -> Path:
    from ..layout import Layout

    root = Path(env["store_root"]) if isinstance(env, dict) and env.get("store_root") else Layout().root  # the run's own store
    return root.with_name(root.name + ".people")  # beside the store, not in it: the store is pushed


def _load(folder: Path) -> dict[str, Any]:
    path = folder / "state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(state, dict) and state.get("salt"):
            return state
    except (OSError, ValueError):
        pass
    return {"salt": secrets.token_hex(32), "people": {}}


def _save(folder: Path, state: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"state.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(tmp, folder / "state.json")
    try:
        os.chmod(folder / "state.json", 0o600)
    except OSError:
        pass


def mask(salt: str, kind: str, value: str) -> str:
    """A short, salted, one-way mark of a value (an email, a session id): the same value always gives the same mark here."""
    return hashlib.sha256(f"{salt}:{kind}:{value}".encode()).hexdigest()[:16]


def _iso(epoch: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _people(conn: Any, models: Any, workspace_id: str, salt: str) -> dict[str, dict[str, Any]]:
    """The workspace's people by user id, from named columns only. The email is hashed here; the password, the subject and the
    name are never selected as values (only whether a password or a subject exists)."""
    import sqlalchemy as sa

    user, member, token = models.app_user, models.membership, models.api_token
    tokens = {r.user_id: r.n for r in conn.execute(sa.select(token.c.user_id, sa.func.count().label("n")).where(token.c.workspace_id == workspace_id,
                                                                                                                 token.c.revoked.is_(False)).group_by(token.c.user_id))}
    rows = conn.execute(sa.select(user.c.id, user.c.email, user.c.active, user.c.created, member.c.role,
                                  user.c.password_hash.isnot(None).label("has_password"), user.c.subject.isnot(None).label("has_subject"))
                        .join(member, member.c.user_id == user.c.id).where(member.c.workspace_id == workspace_id).order_by(user.c.id))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r.id] = {"user": r.id, "email_hash": mask(salt, "email", str(r.email or "").strip().lower()), "role": r.role, "active": bool(r.active),
                     "signs_in": "password" if r.has_password else "sso" if r.has_subject else "none", "joined": str(r.created) if r.created else None,
                     "api_tokens": int(tokens.get(r.id, 0))}
    return out


def _sessions(conn: Any, models: Any, workspace_id: str, salt: str) -> list[dict[str, Any]]:
    import sqlalchemy as sa

    s = models.user_session
    return [{"session": "s" + mask(salt, "session", r.id), "user": r.user_id, "started": _iso(r.started), "last_seen": _iso(r.last_seen), "expires": _iso(r.expires)}
            for r in conn.execute(sa.select(s.c.id, s.c.user_id, s.c.started, s.c.last_seen, s.c.expires).where(s.c.workspace_id == workspace_id)
                                  .order_by(s.c.started, s.c.id))]


def sources(env: Any) -> Iterator[Source]:
    """Every workspace's people and sessions; nothing when no database is configured."""
    from .database import _engine, _name_and_consent
    from .workspaces import AGENT_RUNS

    engine = _engine()
    if engine is None:
        return
    from ...storage import models

    base = Path(env["agent_runs"]) if isinstance(env, dict) and env.get("agent_runs") else AGENT_RUNS
    only = Path(env["only"]).resolve() if isinstance(env, dict) and env.get("only") else None
    folder = _state_dir(env)
    state = _load(folder)
    tmp = Path(tempfile.mkdtemp(prefix="dclab-people-"))
    try:
        with engine.connect() as conn:
            for wid, key in conn.execute(models.workspace.select().with_only_columns(models.workspace.c.id, models.workspace.c.key)):
                if only is not None and Path(key).resolve() != only:
                    continue
                name, consent = _name_and_consent(key, base)
                current = _people(conn, models, wid, state["salt"])
                sessions = _sessions(conn, models, wid, state["salt"])
                if consent not in ("improve", "train"):
                    yield Source(name, "people/people~counts.json", json.dumps({"people": len(current), "sessions": len(sessions)}).encode(), consent=consent,
                                 note="counts only: this workspace's consent does not allow keeping its people")
                    continue
                known = state["people"].setdefault(name, {})
                events: list[dict[str, Any]] = []
                at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for uid, row in current.items():
                    before = known.get(uid)
                    if before is None:
                        events.append({"at": at, "event": "joined", **row})
                    elif {k: v for k, v in before.items() if k != "removed"} != row or before.get("removed"):
                        events.append({"at": at, "event": "changed", **row})
                    known[uid] = row
                for uid, row in list(known.items()):
                    if uid not in current and not row.get("removed"):
                        events.append({"at": at, "event": "removed", **{k: v for k, v in row.items() if k != "removed"}, "active": False})
                        known[uid] = {**row, "removed": True}  # kept as seen last: the log says it, the person stays in it
                log = state.setdefault("log", {}).setdefault(name, [])
                log.extend(events)  # the whole history, so the store keeps only its new tail
                path = tmp / f"{name}-person.jsonl"
                path.write_text("".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in log), encoding="utf-8")
                yield Source(name, "people/person.jsonl", path, consent=consent, log=True,
                             note="each person: a pseudonymous id, a hash of the email, how they sign in, their role; never a password, token, email or name")
                if sessions:
                    spath = tmp / f"{name}-session.jsonl"
                    spath.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in sessions), encoding="utf-8")
                    yield Source(name, "people/session.jsonl", spath, consent=consent, note="the workspace's sessions: hashed id, user, start, last seen, expiry")
        _save(folder, state)  # after every workspace was read: a run that failed halfway leaves the last state to be read again
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


KEEPER = Keeper("people", (Claim("sessions", "people/"),), sources, ("m2_workflow_decisions", "m4_workflow_llm", "m6_stepwise_decisions"),
                "each workspace's people and sessions: pseudonymous ids and hashes, never a password, token, email or name")

__all__ = ["KEEPER", "sources", "mask"]
