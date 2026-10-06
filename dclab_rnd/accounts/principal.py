"""Who is asking, for the length of one request or one job (package 10.2).

``Principal`` is set by the server's middleware from the session cookie or an API token, or is the local owner when
no sign-in is configured (today's single person on this machine). It lives in a context variable: the audit takes the
user from it, and the role check reads it, without passing it through every call.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field

from .roles import label


@dataclass(frozen=True)
class Principal:
    role: str | None                 # owner, data_scientist, reviewer, viewer; None when signed out
    workspace_id: str | None = None  # the workspace this request acts in
    user_id: str | None = None       # None for the local owner (no accounts) and when signed out
    email: str = ""
    name: str = ""
    via: str = "local"               # local, session, token, or anonymous
    session_id: str | None = None
    csrf: str | None = None          # the session's request token (writes from a browser carry it)
    workspaces: tuple = field(default_factory=tuple)  # (id, name, role) of every workspace the user belongs to

    @property
    def signed_in(self) -> bool:
        return self.role is not None

    @property
    def who(self) -> str:
        """How the audit and the approvals name this person."""
        if self.user_id is None:
            return "Owner" if self.role == "owner" else "Person"
        return self.name or self.email or label(self.role)

    def public(self) -> dict:
        return {"signed_in": self.signed_in, "user_id": self.user_id, "email": self.email, "name": self.name, "role": self.role,
                "role_label": label(self.role) if self.role else None, "workspace_id": self.workspace_id, "via": self.via,
                "workspaces": [{"id": w, "name": n, "role": r, "role_label": label(r)} for w, n, r in self.workspaces]}


LOCAL_OWNER = Principal(role="owner", via="local")
ANONYMOUS = Principal(role=None, via="anonymous")

_CURRENT: contextvars.ContextVar[Principal | None] = contextvars.ContextVar("dclab_principal", default=None)


def current() -> Principal | None:
    return _CURRENT.get()


@contextlib.contextmanager
def acting(principal: Principal | None):
    token = _CURRENT.set(principal)
    try:
        yield principal
    finally:
        _CURRENT.reset(token)


def set_current(principal: Principal | None) -> contextvars.Token:
    return _CURRENT.set(principal)


def reset_current(token: contextvars.Token) -> None:
    _CURRENT.reset(token)
