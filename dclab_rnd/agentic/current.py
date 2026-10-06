"""The current workspace's objects, for code that was handed them once at startup (package 10.2).

The draft routes and the pages were registered with one workspace's stores. ``Current("drafts")`` stands in for
them: each use reaches the ``drafts`` of the workspace the request acts in (``dclab_rnd.context``, set by the
server's middleware), and the server's own workspace when nothing is set. ``Current()`` stands for the Services.
"""

from __future__ import annotations

from typing import Any


class Current:
    def __init__(self, attribute: str | None = None, default: Any = None):
        object.__setattr__(self, "_attribute", attribute)
        object.__setattr__(self, "_default", default)  # the server's own Services: used outside a request

    def _target(self) -> Any:
        from .. import context

        services = context.services() or self._default
        return services if self._attribute is None else getattr(services, self._attribute)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target(), name, value)

    def __contains__(self, key: Any) -> bool:
        return key in self._target()

    def __getitem__(self, key: Any) -> Any:
        return self._target()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._target()[key] = value

    def __iter__(self):
        return iter(self._target())

    def __len__(self) -> int:
        return len(self._target())

    def __bool__(self) -> bool:
        return self._target() is not None

    def __repr__(self) -> str:
        return f"Current({self._attribute!r})"
