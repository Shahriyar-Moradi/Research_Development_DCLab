"""Users, workspaces and their members, sessions and API tokens, in PostgreSQL (package 10.2).

Accounts need the database: a session must outlive the process that signed the person in and be seen by every app
instance. A session token and an API token are never stored: the table keeps their SHA-256, so a copy of the database
cannot sign anyone in. Passwords are hashed with argon2 (``passwords``).
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

import sqlalchemy as sa

from .roles import ROLES

SESSION_HOURS = 12        # a session not used for this long ends
SESSION_MAX_DAYS = 7      # and none lasts longer than this
TOKEN_PREFIX = "dclab_"


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AccountError(ValueError):
    pass


class Accounts:
    def __init__(self, url: str | None = None):
        from ..storage import db

        self.engine = db.engine(url)

    # ------------------------------------------------------------------ users
    def create_user(self, email: str, name: str = "", password: str | None = None, subject: str | None = None) -> dict[str, Any]:
        from ..storage.models import app_user
        from . import passwords

        email = email.strip().lower()
        if "@" not in email or len(email) > 320:
            raise AccountError("an email address is needed")
        uid = "u" + secrets.token_hex(7)
        values = {"id": uid, "email": email, "name": name.strip()[:200], "subject": subject,
                  "password_hash": passwords.hash_password(password) if password else None}
        try:
            with self.engine.begin() as c:
                c.execute(sa.insert(app_user).values(**values))
        except sa.exc.IntegrityError:
            raise AccountError("a user with this email exists") from None
        return self.user(uid)

    def user(self, user_id: str) -> dict[str, Any]:
        from ..storage.models import app_user as t

        with self.engine.connect() as c:
            row = c.execute(sa.select(t.c.id, t.c.email, t.c.name, t.c.subject, t.c.active).where(t.c.id == user_id)).first()
        if row is None:
            raise KeyError(user_id)
        return dict(row._mapping)

    def by_email(self, email: str) -> dict[str, Any] | None:
        from ..storage.models import app_user as t

        with self.engine.connect() as c:
            row = c.execute(sa.select(t.c.id, t.c.email, t.c.name, t.c.subject, t.c.active, t.c.password_hash)
                            .where(t.c.email == email.strip().lower())).first()
        return dict(row._mapping) if row else None

    def by_subject(self, subject: str) -> dict[str, Any] | None:
        from ..storage.models import app_user as t

        with self.engine.connect() as c:
            row = c.execute(sa.select(t.c.id, t.c.email, t.c.name, t.c.subject, t.c.active).where(t.c.subject == subject)).first()
        return dict(row._mapping) if row else None

    def update_user(self, user_id: str, **values: Any) -> None:
        from ..storage.models import app_user as t
        from . import passwords

        if "password" in values:
            values["password_hash"] = passwords.hash_password(values.pop("password"))
        with self.engine.begin() as c:
            c.execute(sa.update(t).where(t.c.id == user_id).values(**values))

    def check_password(self, email: str, password: str) -> dict[str, Any] | None:
        """The user when the password is theirs and the account is active; None otherwise (the same answer for a
        wrong password and an unknown email, after the same work)."""
        from . import passwords

        found = self.by_email(email)
        ok = passwords.verify(found["password_hash"] if found and found.get("password_hash") else None, password)
        if not (found and ok and found["active"]):
            return None
        return {k: found[k] for k in ("id", "email", "name", "subject", "active")}

    # ------------------------------------------------------------------ workspaces and members
    def create_workspace(self, name: str) -> str:
        from ..storage import db

        return db.workspace("dclab-workspace:" + secrets.token_hex(8), name.strip()[:120] or "workspace")

    def workspace(self, workspace_id: str) -> dict[str, Any]:
        from ..storage.models import workspace as t

        with self.engine.connect() as c:
            row = c.execute(sa.select(t.c.id, t.c.key, t.c.name).where(t.c.id == workspace_id)).first()
        if row is None:
            raise KeyError(workspace_id)
        return dict(row._mapping)

    def set_member(self, workspace_id: str, user_id: str, role: str) -> None:
        from sqlalchemy.dialects.postgresql import insert

        from ..storage.models import membership as t

        if role not in ROLES:
            raise AccountError(f"the role is one of {', '.join(ROLES)}")
        with self.engine.begin() as c:
            c.execute(insert(t).values(workspace_id=workspace_id, user_id=user_id, role=role)
                      .on_conflict_do_update(index_elements=[t.c.workspace_id, t.c.user_id], set_={"role": role}))

    def remove_member(self, workspace_id: str, user_id: str) -> None:
        from ..storage.models import api_token as k, membership as t, user_session as s

        with self.engine.begin() as c:
            c.execute(sa.delete(t).where(t.c.workspace_id == workspace_id, t.c.user_id == user_id))
            c.execute(sa.delete(s).where(s.c.workspace_id == workspace_id, s.c.user_id == user_id))  # signed out of it now
            c.execute(sa.update(k).where(k.c.workspace_id == workspace_id, k.c.user_id == user_id).values(revoked=True))  # and its tokens end

    def role_of(self, user_id: str, workspace_id: str) -> str | None:
        from ..storage.models import membership as t

        with self.engine.connect() as c:
            return c.execute(sa.select(t.c.role).where(t.c.workspace_id == workspace_id, t.c.user_id == user_id)).scalar()

    def members(self, workspace_id: str) -> list[dict[str, Any]]:
        from ..storage.models import app_user as u, membership as m

        with self.engine.connect() as c:
            rows = c.execute(sa.select(u.c.id, u.c.email, u.c.name, u.c.active, m.c.role, (u.c.password_hash.isnot(None)).label("password"),
                                       (u.c.subject.isnot(None)).label("single_sign_on"))
                             .join(m, m.c.user_id == u.c.id).where(m.c.workspace_id == workspace_id).order_by(u.c.email))
            return [dict(r._mapping) for r in rows]

    def workspaces_of(self, user_id: str) -> list[tuple[str, str, str]]:
        from ..storage.models import membership as m, workspace as w

        with self.engine.connect() as c:
            rows = c.execute(sa.select(w.c.id, w.c.name, m.c.role).join(m, m.c.workspace_id == w.c.id)
                             .where(m.c.user_id == user_id).order_by(w.c.name, w.c.id))
            return [(r.id, r.name, r.role) for r in rows]

    # ------------------------------------------------------------------ sessions
    def start_session(self, user_id: str, workspace_id: str) -> tuple[str, str]:
        """A new session: (the cookie's token, the session's request token). Only the token's hash is kept."""
        from ..storage.models import user_session as t

        token, csrf, now = secrets.token_urlsafe(32), secrets.token_urlsafe(32), time.time()
        with self.engine.begin() as c:
            c.execute(sa.insert(t).values(id=digest(token), user_id=user_id, workspace_id=workspace_id, csrf=csrf,
                                          started=now, last_seen=now, expires=now + SESSION_HOURS * 3600))
        return token, csrf

    def session(self, token: str) -> dict[str, Any] | None:
        """The live session for a cookie's token, its expiry moved on; None when unknown or expired."""
        from ..storage.models import user_session as t

        now = time.time()
        with self.engine.begin() as c:
            row = c.execute(sa.select(t).where(t.c.id == digest(token))).first()
            if row is None:
                return None
            if row.expires < now or row.started + SESSION_MAX_DAYS * 86400 < now:
                c.execute(sa.delete(t).where(t.c.id == row.id))
                return None
            if now - row.last_seen > 60:  # one write a minute at most
                c.execute(sa.update(t).where(t.c.id == row.id).values(last_seen=now, expires=min(now + SESSION_HOURS * 3600,
                                                                                                 row.started + SESSION_MAX_DAYS * 86400)))
        return dict(row._mapping)

    def switch_session(self, token: str, workspace_id: str) -> None:
        from ..storage.models import user_session as t

        with self.engine.begin() as c:
            c.execute(sa.update(t).where(t.c.id == digest(token)).values(workspace_id=workspace_id))

    def end_session(self, token: str) -> None:
        from ..storage.models import user_session as t

        with self.engine.begin() as c:
            c.execute(sa.delete(t).where(t.c.id == digest(token)))

    # ------------------------------------------------------------------ API tokens (for /mcp clients and scripts)
    def create_token(self, user_id: str, workspace_id: str, name: str) -> tuple[str, dict[str, Any]]:
        """A new API token: (the token, shown once; its record). The token acts as its user in that workspace."""
        from ..storage.models import api_token as t

        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        record = {"id": "t" + secrets.token_hex(6), "user_id": user_id, "workspace_id": workspace_id, "name": name.strip()[:80] or "token",
                  "token_hash": digest(token), "prefix": token[:12], "created": time.time()}
        with self.engine.begin() as c:
            c.execute(sa.insert(t).values(**record))
        return token, {k: v for k, v in record.items() if k != "token_hash"}

    def tokens(self, user_id: str) -> list[dict[str, Any]]:
        from ..storage.models import api_token as t

        with self.engine.connect() as c:
            rows = c.execute(sa.select(t.c.id, t.c.workspace_id, t.c.name, t.c.prefix, t.c.created, t.c.last_used)
                             .where(t.c.user_id == user_id, t.c.revoked.is_(False)).order_by(t.c.created))
            return [dict(r._mapping) for r in rows]

    def token(self, token: str) -> dict[str, Any] | None:
        from ..storage.models import api_token as t

        if not token.startswith(TOKEN_PREFIX):
            return None
        with self.engine.begin() as c:
            row = c.execute(sa.select(t).where(t.c.token_hash == digest(token), t.c.revoked.is_(False))).first()
            if row is None:
                return None
            if not row.last_used or time.time() - row.last_used > 60:
                c.execute(sa.update(t).where(t.c.id == row.id).values(last_used=time.time()))
        return dict(row._mapping)

    def revoke_token(self, user_id: str, token_id: str) -> bool:
        from ..storage.models import api_token as t

        with self.engine.begin() as c:
            return bool(c.execute(sa.update(t).where(t.c.id == token_id, t.c.user_id == user_id, t.c.revoked.is_(False))
                                  .values(revoked=True)).rowcount)
