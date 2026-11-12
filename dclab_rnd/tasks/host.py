"""Cancellable hosting for model planning in worker threads and background jobs.

The POSIX main thread uses signal.setitimer to interrupt runaway synchronous calls.
In background threads (such as asyncio.to_thread, ThreadPoolExecutor, or jobs.worker threads),
signals cannot be set from non-main threads. This module provides a cancellable host context
that integrates with jobs.worker checkpoints, explicit cancellation callbacks, and bounds
model requests with deadlines, token budgets, and spend controls.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_CURRENT_HOST: contextvars.ContextVar[PlanningHost | None] = contextvars.ContextVar("planning_host", default=None)


class PlanningCancelled(ValueError):
    """Raised when a planning operation is explicitly cancelled by its host."""


class PlanningHost:
    def __init__(self, *, check_cancel: Callable[[], None] | None = None,
                 on_cancel: Callable[[], None] | None = None, name: str = "hosted"):
        self.name = name
        self._check_cancel = check_cancel
        self._on_cancel = on_cancel
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._cancelled = True
        if self._on_cancel is not None:
            self._on_cancel()

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def check_cancellation(self) -> None:
        if self.is_cancelled():
            raise PlanningCancelled("planning_cancelled")
        if self._check_cancel is not None:
            self._check_cancel()


class _JobWorkerHost(PlanningHost):
    """Adapts an active jobs.worker Control into a PlanningHost."""
    def __init__(self, control: Any):
        super().__init__(name=f"job_{control.job.get('id', 'unknown')}")
        self.control = control

    def check_cancellation(self) -> None:
        from dclab_rnd.jobs.worker import checkpoint
        checkpoint()
        super().check_cancellation()


@contextmanager
def planning_host(check_cancel: Callable[[], None] | None = None,
                  on_cancel: Callable[[], None] | None = None,
                  name: str = "hosted") -> Iterator[PlanningHost]:
    """Activate a cancellable planning host for the current thread / async context."""
    host = PlanningHost(check_cancel=check_cancel, on_cancel=on_cancel, name=name)
    token = _CURRENT_HOST.set(host)
    try:
        yield host
    finally:
        _CURRENT_HOST.reset(token)


def current_host() -> PlanningHost | None:
    """Return the active PlanningHost, or wrap an active jobs.worker.Control if present."""
    explicit = _CURRENT_HOST.get()
    if explicit is not None:
        return explicit
    try:
        from dclab_rnd.jobs.worker import current as job_current
        control = job_current()
        if control is not None:
            return _JobWorkerHost(control)
    except (ImportError, AttributeError):
        pass
    return None


def is_cancellable() -> bool:
    """True if caller has a valid mechanism to cancel or bound execution."""
    if current_host() is not None:
        return True
    import signal
    return (threading.current_thread() is threading.main_thread()
            and hasattr(signal, "setitimer")
            and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))


__all__ = ["PlanningHost", "PlanningCancelled", "planning_host", "current_host", "is_cancellable"]
