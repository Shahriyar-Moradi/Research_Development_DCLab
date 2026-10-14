"""One lock per folder of append-only files, across threads and processes (package 19.10).

The session record and the decision ledger on files are appended to by the server and the worker, and rewritten only
by a deletion or a retention sweep (``experience.retention``). Every writer of a folder holds :func:`held` on it: a
thread lock for the process, and ``fcntl.flock`` on ``<folder>/.lock`` for the others, so a rewrite never drops a line
appended meanwhile. On a system without ``fcntl`` only the thread lock holds.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

_LOCKS: dict[str, threading.RLock] = {}
_GUARD = threading.Lock()
NAME = ".lock"


@contextlib.contextmanager
def held(root: Path) -> Iterator[None]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = str(root.resolve())
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock, open(root / NAME, "a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows: the thread lock alone
            pass
        yield
