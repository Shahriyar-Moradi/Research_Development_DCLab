"""Steps inside a stage (package 23.8): what a run did, one entry at a time, with a digest of what each step produced.

A stage's run record says what it read and what it made; ``Step`` says how it got there: a cleaning action, an encoding, a
feature transform, a fold, a model fit, a prediction, a metric — each with its name, parameters, the shapes in and out, the
fold, the metric and its value, the time and a digest of the numbers it produced. A decision can then be traced to the
exact numbers it was made on, and a replay that diverges is pinned to the first step whose digest differs.

The recording is a context: ``with collect() as log:`` around a run, ``with step("model.fit", "fit", ...) as s:`` around
each action (a no-op when nothing collects, so library code stays free of cost and of a host). Nothing here keeps a table's
value: a digest is a hash, a shape is two numbers.
"""

from __future__ import annotations

import contextvars
import hashlib
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, Optional, Union

from dclab_core.lineage import MAX_STEPS, SAMPLE_COLS, SAMPLE_ROWS, Step, StepKind  # the contract lives with the run record it belongs to


def digest_of(*objects: Any) -> str:
    """Sixteen hex characters of what ``objects`` hold: arrays and frames by their shape and the first ``SAMPLE_ROWS`` rows and
    ``SAMPLE_COLS`` columns rounded to six decimals, anything else by its text. Deterministic for equal numbers; reads no more
    than a bounded sample, and keeps nothing of it."""
    h = hashlib.sha256()
    for obj in objects:
        h.update(_bytes(obj))
    return h.hexdigest()[:16]


def _bytes(obj: Any) -> bytes:
    try:
        import numpy as np

        if hasattr(obj, "toarray") and hasattr(obj, "shape") and not hasattr(obj, "iloc"):  # a scipy sparse matrix
            rows = min(obj.shape[0], SAMPLE_ROWS)
            cols = min(obj.shape[1], SAMPLE_COLS)
            total = float(round(float(obj.sum()), 6)) + 0.0 if obj.nnz > rows * cols else None
            return f"sparse{obj.shape}{total}".encode() + _bytes(obj[:rows, :cols].toarray())
        if hasattr(obj, "iloc"):  # a pandas frame or series
            part = obj.iloc[:SAMPLE_ROWS] if obj.ndim == 1 else obj.iloc[:SAMPLE_ROWS, :SAMPLE_COLS]
            return f"frame{obj.shape}".encode() + _bytes(part.to_numpy())
        if isinstance(obj, np.ndarray):
            part = obj[:SAMPLE_ROWS] if obj.ndim == 1 else obj[:SAMPLE_ROWS, :SAMPLE_COLS]
            if part.dtype.kind in "fiub":
                whole = obj.astype("float64", copy=False)
                total = float(np.round(np.nansum(whole), 6)) + 0.0 if obj.size > part.size else None  # a change past the sample moves the sum
                return (f"array{obj.shape}{total}".encode()
                        + np.ascontiguousarray(np.round(part.astype("float64"), 6) + 0.0).tobytes())  # + 0.0: -0.0 and 0.0 are one number
            return f"array{obj.shape}".encode() + repr(part.tolist()).encode()
    except ImportError:  # numpy is a dependency; a host without it digests by text
        pass
    return repr(obj).encode()


class StepLog:
    """The steps one run collected, in the order they began (an outer step before the ones inside it, though it ends after
    them), and how many were dropped past ``MAX_STEPS``."""

    def __init__(self, on_start: Optional[Callable[[Step], None]] = None, on_finish: Optional[Callable[[Step], None]] = None):
        self._slots: list[Optional[Step]] = []
        self.dropped = 0
        self.on_start, self.on_finish = on_start, on_finish

    @property
    def steps(self) -> list[Step]:
        return [s for s in self._slots if s is not None]

    def reserve(self) -> Optional[int]:
        """A place for the step that begins now (its index), or None when the log is full: it is counted, not kept."""
        if len(self._slots) >= MAX_STEPS:
            self.dropped += 1
            return None
        self._slots.append(None)
        return len(self._slots) - 1


_ACTIVE: contextvars.ContextVar[Optional[StepLog]] = contextvars.ContextVar("dclab_core_steps", default=None)


@contextmanager
def collect(on_start: Optional[Callable[[Step], None]] = None, on_finish: Optional[Callable[[Step], None]] = None) -> Iterator[StepLog]:
    """Collect the steps taken inside the block. ``on_start`` and ``on_finish`` (a host's events) are called with each step;
    a callback that raises never stops the run."""
    log = StepLog(on_start, on_finish)
    token = _ACTIVE.set(log)
    try:
        yield log
    finally:
        _ACTIVE.reset(token)


def active() -> bool:
    return _ACTIVE.get() is not None


def current() -> Optional[StepLog]:
    """The log collecting now, or None."""
    return _ACTIVE.get()


class _Handle:
    """What a step reports on the way out; every call is optional."""

    def __init__(self) -> None:
        self.rows_out: Optional[int] = None
        self.cols_out: Optional[int] = None
        self.metric: Optional[str] = None
        self.value: Optional[float] = None
        self.digest: Optional[str] = None

    def out(self, *, shape: Any = None, rows: Optional[int] = None, cols: Optional[int] = None, metric: Optional[str] = None,
            value: Optional[float] = None, produced: Any = None) -> None:
        """Say what the step made: its output's shape (or rows and columns), a metric and its value, and the object whose digest
        stands for it (an array, a frame)."""
        if shape is not None and len(shape) >= 1:
            self.rows_out, self.cols_out = int(shape[0]), (int(shape[1]) if len(shape) > 1 else None)
        if rows is not None:
            self.rows_out = int(rows)
        if cols is not None:
            self.cols_out = int(cols)
        if metric is not None:
            self.metric = str(metric)[:80]
        if value is not None:
            try:
                number = float(value)
                self.value = number if number == number and abs(number) != float("inf") else None
            except (TypeError, ValueError):
                self.value = None
        if produced is not None:
            try:
                self.digest = digest_of(produced)
            except Exception:  # noqa: BLE001 - a digest that cannot be made never changes a run
                self.digest = None
        elif self.value is not None and self.digest is None:
            self.digest = digest_of(round(self.value, 8) + 0.0)  # the number itself is what the step produced


@contextmanager
def step(name: str, kind: StepKind, *, params: Optional[dict[str, Any]] = None, shape_in: Any = None, fold: Optional[int] = None) -> Iterator[_Handle]:
    """One action of a run. A no-op (and free) when nothing collects. A step that raises is kept with what it knew, then the
    error goes on."""
    log = _ACTIVE.get()
    handle = _Handle()
    if log is None:
        yield handle
        return
    index = log.reserve()
    clean = {str(k)[:40]: v for k, v in (params or {}).items() if isinstance(v, (str, int, float, bool)) or v is None}
    rows_in = int(shape_in[0]) if shape_in is not None and len(shape_in) >= 1 else None
    cols_in = int(shape_in[1]) if shape_in is not None and len(shape_in) > 1 else None
    if index is not None and log.on_start is not None:
        try:
            _call(log.on_start, Step(index=index, name=name[:120] or "step", kind=kind, params=_finite(clean), rows_in=rows_in, cols_in=cols_in, fold=fold))
        except Exception:  # noqa: BLE001 - a step's own record never changes a run
            pass
    started = time.perf_counter()
    try:
        yield handle
    finally:
        if index is not None:
            try:
                done = Step(index=index, name=name[:120] or "step", kind=kind, params=_finite(clean), rows_in=rows_in, cols_in=cols_in,
                            rows_out=handle.rows_out, cols_out=handle.cols_out, fold=fold, metric=handle.metric, value=handle.value,
                            seconds=round(time.perf_counter() - started, 4), digest=handle.digest)
                log._slots[index] = done
                if log.on_finish is not None:
                    _call(log.on_finish, done)
            except Exception:  # noqa: BLE001 - a step that cannot be recorded leaves a gap, never an error in the run it watched
                pass


def _finite(params: dict[str, Any]) -> dict[str, Any]:
    """The first sixteen parameters, a float that is not a number left out (the contract refuses it)."""
    return {k: v for k, v in list(params.items())[:16] if not (isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))))}


def _call(callback: Callable[[Step], None], item: Step) -> None:
    try:
        callback(item)
    except Exception:  # noqa: BLE001 - a host's event never stops a run
        pass


def compare(recorded: list[Step], replayed: list[Step]) -> dict[str, Any]:
    """Where a replay first differs from the run it repeats: the first step (by position) whose name, kind, fold or digest
    differs, or the first step one run has and the other lacks; ``identical`` when none does. Times are not compared."""
    for position in range(max(len(recorded), len(replayed))):
        a = recorded[position] if position < len(recorded) else None
        b = replayed[position] if position < len(replayed) else None
        if a is None or b is None or (a.name, a.kind, a.fold, a.digest) != (b.name, b.kind, b.fold, b.digest):
            return {"identical": False, "recorded_steps": len(recorded), "replayed_steps": len(replayed),
                    "first_difference": {"position": position,
                                         "recorded": a.model_dump(exclude_defaults=True) if a is not None else None,
                                         "replayed": b.model_dump(exclude_defaults=True) if b is not None else None}}
    return {"identical": True, "recorded_steps": len(recorded), "replayed_steps": len(replayed), "first_difference": None}


__all__ = ["MAX_STEPS", "Step", "StepLog", "collect", "step", "active", "current", "digest_of", "compare"]
