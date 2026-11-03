"""The Admin area's own sign-in on a machine without accounts (A1 of the 2026-11-03 plan): one username and password, the owner's.

Without ``DCLAB_AUTH`` every request is the local owner. Once an admin login exists in the workspace folder
(``admin_login.json``, written only by ``python -m dclab_rnd.accounts admin-set``), the owner-only permissions (``admin``,
``timelines``: the Admin page, the session records, members, budgets, routing switches, the benchmark) need an admin session
as well; everything else stays as it is. With no file nothing changes, so a clean checkout, the tests and the command line
work as before.

- The password is an argon2id hash (``accounts.passwords``); the file never holds it.
- A session is a random token in the ``dclab_admin`` cookie (HttpOnly, SameSite=Strict, Secure behind TLS); only its SHA-256 is
  kept, in ``admin_sessions.json``. It ends after ``IDLE_MINUTES`` without use or ``MAX_HOURS`` in all, at sign-out, and for
  every session at a password change.
- ``FAILURES`` wrong passwords within ``WINDOW_MINUTES`` lock the login for ``LOCK_MINUTES``, doubled at each lock in a row
  (at most a day); a right password while locked is refused too. The address is always this machine's, so the count is the
  login's.

It is a protection of the Admin area from other people and other browser tabs on this machine, not from someone who can
read the workspace folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from . import passwords

FILE = "admin_login.json"
SESSIONS = "admin_sessions.json"
COOKIE = "dclab_admin"
IDLE_MINUTES = 30
MAX_HOURS = 8
FAILURES = 5
WINDOW_MINUTES = 15
LOCK_MINUTES = 15
MAX_LOCK_MINUTES = 24 * 60
TOUCH_SECONDS = 60  # a session's last use is written at most once a minute
USERNAME = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
_LOCK = threading.Lock()


class Locked(Exception):
    """The login is locked after too many wrong passwords; ``until`` says when it opens."""

    def __init__(self, until: str):
        super().__init__(until)
        self.until = until


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    return when.isoformat(timespec="seconds")


def _read(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def exists(home: Path | str) -> bool:
    return (Path(home) / FILE).is_file()


def username(home: Path | str) -> Optional[str]:
    return _read(Path(home) / FILE).get("username") or None


def set_login(home: Path | str, name: str, password: str) -> None:
    """Create or replace the admin login (the command line's ``admin-set``); every admin session ends."""
    if not USERNAME.match(name or ""):
        raise ValueError("a username is 3 to 32 letters, digits, dot, dash or underscore")
    home = Path(home)
    with _LOCK:
        old = _read(home / FILE)
        _write(home / FILE, {"username": name, "hash": passwords.hash_password(password), "created": old.get("created") or _iso(_now()),
                             "updated": _iso(_now()), "failures": [], "locked_until": None, "locks": 0})
        _write(home / SESSIONS, {})


def remove(home: Path | str) -> bool:
    """Remove the admin login: the Admin area is open again on this machine, as before. Returns whether one was there."""
    home = Path(home)
    with _LOCK:
        found = (home / FILE).is_file()
        for name in (FILE, SESSIONS):
            (home / name).unlink(missing_ok=True)
    return found


def sign_in(home: Path | str, name: str, password: str) -> str:
    """A new session token for a right username and password; ValueError when wrong, Locked when locked."""
    home = Path(home)
    with _LOCK:
        doc = _read(home / FILE)
        if not doc:
            raise ValueError("no admin login is set on this machine")
        now = _now()
        until = doc.get("locked_until")
        if until and datetime.fromisoformat(until) > now:
            raise Locked(until)
        ok = passwords.verify(doc.get("hash"), password or "") and secrets.compare_digest((name or "").encode(), doc["username"].encode())
        if not ok:
            window = now - timedelta(minutes=WINDOW_MINUTES)
            failures = [f for f in doc.get("failures", []) if datetime.fromisoformat(f) > window] + [_iso(now)]
            doc["failures"] = failures
            if len(failures) >= FAILURES:
                minutes = min(LOCK_MINUTES * 2 ** int(doc.get("locks") or 0), MAX_LOCK_MINUTES)
                doc["locked_until"], doc["locks"], doc["failures"] = _iso(now + timedelta(minutes=minutes)), int(doc.get("locks") or 0) + 1, []
                _write(home / FILE, doc)
                raise Locked(doc["locked_until"])
            _write(home / FILE, doc)
            raise ValueError("wrong username or password")
        doc.update(failures=[], locked_until=None, locks=0)
        _write(home / FILE, doc)
        token = secrets.token_urlsafe(32)
        sessions = _live(_read(home / SESSIONS), now)
        sessions[_digest(token)] = {"created": _iso(now), "seen": _iso(now)}
        _write(home / SESSIONS, sessions)
        return token


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _live(sessions: dict[str, Any], now: datetime) -> dict[str, Any]:
    out = {}
    for key, s in sessions.items():
        try:  # a hand-edited or broken entry is no session, never an error on every request
            created, seen = datetime.fromisoformat(s["created"]), datetime.fromisoformat(s["seen"])
            alive = now - seen <= timedelta(minutes=IDLE_MINUTES) and now - created <= timedelta(hours=MAX_HOURS)
        except (KeyError, TypeError, ValueError):
            continue
        if alive:
            out[key] = s
    return out


def session(home: Path | str, token: Optional[str]) -> bool:
    """Whether ``token`` is a live admin session; a use keeps it alive (written at most once a minute)."""
    if not token:
        return False
    home = Path(home)
    now = _now()
    sessions = _read(home / SESSIONS)
    found = sessions.get(_digest(token))
    if not found or _digest(token) not in _live({_digest(token): found}, now):
        return False
    if (now - datetime.fromisoformat(found["seen"])).total_seconds() >= TOUCH_SECONDS:  # _live parsed it: a valid time
        with _LOCK:
            current = _live(_read(home / SESSIONS), now)
            if _digest(token) in current:
                current[_digest(token)]["seen"] = _iso(now)
                _write(home / SESSIONS, current)
    return True


def sign_out(home: Path | str, token: Optional[str]) -> None:
    if not token:
        return
    home = Path(home)
    with _LOCK:
        sessions = _read(home / SESSIONS)
        if sessions.pop(_digest(token), None) is not None:
            _write(home / SESSIONS, sessions)


def change_password(home: Path | str, current: str, new: str) -> None:
    """A new password after the current one; every admin session ends (the caller signs in again)."""
    home = Path(home)
    name = username(home)
    if name is None:
        raise ValueError("no admin login is set on this machine")
    with _LOCK:
        doc = _read(home / FILE)
        if not passwords.verify(doc.get("hash"), current or ""):
            raise ValueError("the current password is wrong")
    set_login(home, name, new)


def state(home: Path | str, token: Optional[str]) -> dict[str, Any]:
    """What the page needs: whether a login is set, whether this browser is signed in, and until when it is locked."""
    home = Path(home)
    doc = _read(home / FILE)
    until = doc.get("locked_until")
    try:
        locked = bool(until) and datetime.fromisoformat(until) > _now()
    except (TypeError, ValueError):
        locked = False
    signed = bool(doc) and session(home, token)
    return {"required": bool(doc), "signed_in": signed, "username": doc.get("username") if signed else None,  # never before sign-in
            "locked_until": until if locked else None, "idle_minutes": IDLE_MINUTES, "max_hours": MAX_HOURS}


__all__ = ["COOKIE", "Locked", "exists", "username", "set_login", "remove", "sign_in", "session", "sign_out", "change_password", "state",
           "IDLE_MINUTES", "MAX_HOURS", "FAILURES", "WINDOW_MINUTES", "LOCK_MINUTES"]
