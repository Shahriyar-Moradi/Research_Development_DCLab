"""Retrain and champion/challenger (34.11): when a model in use is retrained, and whether the new fit replaces it.

A :class:`RetrainPolicy` says what starts a retrain: a drift alert or a performance alert from the live report (34.10),
or a schedule, and never before enough newly labelled rows exist. :func:`should_retrain` answers with its reasons.

The challenger is fitted by the host (the same recipe and settings as the champion, on the newer labelled rows, with or
without the champion's own: the host's choice, said in the record's window); both are then scored on the **latest labelled window that neither trained on**, and
:func:`compare` reads the difference on those same rows with a paired bootstrap. The challenger is promoted only when it is
better beyond its interval (the difference's lower bound above the margin); otherwise the champion stays, and the record
says why. Every comparison is a :class:`ChallengerRecord`, kept whether it promoted or not. With a margin of zero an equally
good challenger is promoted about once in twenty (the bound is one-sided at 95%; measured on simulated rows): a needless
redeploy, never a worse model; a margin above zero, in the measure's own units, makes it rarer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field

from .contracts import CONTRACTS, Versioned, _version

BOOTSTRAP = 1000
METRICS = {"binary": ("roc_auc", True), "multiclass": ("accuracy", True), "regression": ("mae", False)}


class RetrainPolicy(Versioned):
    """What starts a retrain (34.11, v1)."""

    schema_version: Literal["dclab_core.retrain_policy.v1"] = _version("retrain_policy")
    on_drift: bool = True
    on_performance: bool = True
    every_days: Optional[int] = Field(default=None, ge=1, le=3650)
    min_new_labelled: int = Field(default=500, ge=1)
    window_share: float = Field(default=0.2, gt=0, lt=0.5, description="The latest labelled rows held out to compare the two.")
    margin: float = Field(default=0.0, ge=0, description="How much better the challenger must be, beyond its interval.")


CONTRACTS["retrain_policy"] = RetrainPolicy


def should_retrain(policy: RetrainPolicy, *, live_report: Any = None, last_trained: Optional[str] = None, now: Optional[str] = None,
                   new_labelled: int = 0) -> dict[str, Any]:
    """Whether to retrain now, and why: a reason from the live report or the schedule, and enough new labels."""
    because = []
    report = live_report.model_dump(mode="json") if hasattr(live_report, "model_dump") else (live_report or {})
    windows = report.get("windows") or []
    drift = [w["window"] for w in windows if any("shifted" in a or "flags" in a for a in w.get("alerts") or [])]
    worse = [w["window"] for w in windows if any("worse than the holdout" in a for a in w.get("alerts") or [])]
    if policy.on_drift and drift:
        because.append(f"the inputs or the score shifted (from {drift[0]})")
    if policy.on_performance and worse:
        because.append(f"live performance fell below the holdout's (from {worse[0]})")
    if policy.every_days and last_trained:
        moment = datetime.fromisoformat((now or datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
        age = (moment - datetime.fromisoformat(last_trained.replace("Z", "+00:00"))).days
        if age >= policy.every_days:
            because.append(f"the schedule: {age} days since the last training, every {policy.every_days}")
    if not because:
        return {"retrain": False, "because": [], "wait": "nothing calls for a retrain"}
    if new_labelled < policy.min_new_labelled:
        return {"retrain": False, "because": because,
                "wait": f"{new_labelled} newly labelled rows; a retrain waits for {policy.min_new_labelled}"}
    return {"retrain": True, "because": because, "wait": None}


class ChallengerRecord(Versioned):
    """One comparison of a challenger with the champion (34.11, v1), kept whether it promoted or not."""

    schema_version: Literal["dclab_core.challenger_record.v1"] = _version("challenger_record")
    metric: str
    higher_is_better: bool
    window_rows: int = Field(ge=1)
    window: str = Field(default="", max_length=200, description="Which rows: the latest labelled window neither model trained on.")
    champion: float
    challenger: float
    difference: float = Field(description="Challenger minus champion, oriented so that positive is better.")
    interval: list[Optional[float]] = Field(min_length=2, max_length=2)
    margin: float = 0.0
    promoted: bool
    because: list[str] = Field(default_factory=list)
    reason: str = Field(max_length=1000)
    champion_version: str = ""
    challenger_version: str = ""


CONTRACTS["challenger_record"] = ChallengerRecord


def _auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    from .live import _auc as auc

    return auc(y, s)


def _value(task: str, y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if task == "binary":
        return _auc(y.astype(int), p)
    if task == "multiclass":
        return float(np.mean((p.argmax(axis=1) if p.ndim == 2 else p) == y))
    return float(np.mean(np.abs(p - y)))


MIN_WINDOW = 30  # rows: below it no interval is trusted and nothing is promoted (the review of 34.11: 1 to 4 rows promoted)
MIN_PER_CLASS = 5  # a yes/no window needs this many of each class


def compare(task: str, y_window: Sequence[Any], champion: Any, challenger: Any, *, policy: Optional[RetrainPolicy] = None,
            because: Sequence[str] = (), window: str = "", champion_version: str = "", challenger_version: str = "",
            metric: Optional[str] = None, metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
            higher_is_better: Optional[bool] = None, groups: Optional[Sequence[Any]] = None,
            bootstrap: int = BOOTSTRAP, seed: int = 0) -> ChallengerRecord:
    """The two models' predictions on the same window rows, read with a paired bootstrap over those rows (over whole groups
    when ``groups`` are given: an entity's rows move together); the challenger promoted only when the difference's 90% lower
    bound is above the policy's margin, and never on a window of fewer than ``MIN_WINDOW`` rows (or, yes/no, fewer than
    ``MIN_PER_CLASS`` of a class). ``metric_fn(y, predictions)``: the project's own measure (the default is ROC AUC, accuracy
    or mean absolute error by task), with ``metric`` its name and ``higher_is_better`` its direction."""
    if task not in METRICS:
        raise ValueError(f"A task is one of {', '.join(METRICS)}")
    name, higher = METRICS[task]
    if metric_fn is not None:
        if metric is None or higher_is_better is None:
            raise ValueError("A measure of the host's own needs its name and its direction")
        name, higher = metric, bool(higher_is_better)
    y = np.asarray(y_window, dtype=float)
    a, b = np.asarray(champion, dtype=float), np.asarray(challenger, dtype=float)
    if len(y) == 0 or len(a) != len(y) or len(b) != len(y):
        raise ValueError("The champion's and the challenger's predictions need one per window row")
    if not (np.isfinite(y).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("The window's outcomes and both models' predictions must be finite")

    def value(rows_y: np.ndarray, rows_p: np.ndarray) -> Optional[float]:
        if metric_fn is None:
            return _value(task, rows_y, rows_p)
        try:
            v = float(metric_fn(rows_y, rows_p))
        except (ValueError, ZeroDivisionError):
            return None
        return v if np.isfinite(v) else None

    sign = 1.0 if higher else -1.0
    va, vb = value(y, a), value(y, b)
    if va is None or vb is None:
        raise ValueError("The window cannot be read with this measure (for ROC AUC it needs both classes)")
    members = None
    if groups is not None:
        codes = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)[1]
        order = np.argsort(codes, kind="stable")
        members = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        rows = (np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) if members is not None
                else rng.integers(0, len(y), len(y)))
        da, db = value(y[rows], a[rows]), value(y[rows], b[rows])
        if da is not None and db is not None:
            draws.append(sign * (db - da))
    interval = [round(float(np.quantile(draws, 0.05)), 6), round(float(np.quantile(draws, 0.95)), 6)] if len(draws) >= 50 else [None, None]
    margin = (policy or RetrainPolicy()).margin
    difference = sign * (vb - va)
    small = len(y) < MIN_WINDOW or (task == "binary" and min(int((y == 1).sum()), int((y == 0).sum())) < MIN_PER_CLASS)
    promoted = not small and interval[0] is not None and interval[0] > margin
    shown = (f"its 90% interval {interval[0]} to {interval[1]}" + (" over whole groups" if members is not None else "")
             if interval[0] is not None else "no interval (too few draws could be read)")
    reason = (f"The challenger's {name} {vb:.4f} against the champion's {va:.4f} on {len(y)} rows neither trained on: "
              + (f"too few rows to compare (at least {MIN_WINDOW}, and {MIN_PER_CLASS} of each class for a yes/no outcome): the champion stays."
                 if small else
                 f"better by {difference:.4f}, {shown} above {margin:g}: promoted." if promoted else
                 f"a difference of {difference:+.4f}, {shown}, not above {margin:g}: the champion stays."))
    return ChallengerRecord(metric=name, higher_is_better=higher, window_rows=len(y), window=window[:200], champion=round(va, 6),
                            challenger=round(vb, 6), difference=round(difference, 6), interval=interval, margin=margin, promoted=promoted,
                            because=list(because)[:10], reason=reason, champion_version=champion_version, challenger_version=challenger_version)


__all__ = ["RetrainPolicy", "should_retrain", "ChallengerRecord", "compare", "MIN_WINDOW", "MIN_PER_CLASS"]
