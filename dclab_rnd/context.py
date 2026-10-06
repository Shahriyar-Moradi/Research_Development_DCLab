"""Which workspace a request or a job acts in (package 10.2).

One server can serve several workspaces; each has its own ``Services`` (its stores, model gateway, lessons, jobs and
audit). The server's middleware sets the request's workspace here, and a worker sets the job's; code that reaches a
workspace object without being handed one (the evidence search's lessons, a stage note's model client) reads it from
here, and falls back to what was installed for the process when nothing is set (a script, a test).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any

_SERVICES: contextvars.ContextVar[Any] = contextvars.ContextVar("dclab_services", default=None)


def services() -> Any:
    return _SERVICES.get()


@contextlib.contextmanager
def using(services: Any):
    token = _SERVICES.set(services)
    try:
        yield services
    finally:
        _SERVICES.reset(token)


def set_services(services: Any) -> contextvars.Token:
    return _SERVICES.set(services)


def reset_services(token: contextvars.Token) -> None:
    _SERVICES.reset(token)
