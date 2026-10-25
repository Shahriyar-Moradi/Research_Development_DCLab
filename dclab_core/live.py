"""Monitoring that runs (34.10): a model in use read window by window from its own log.

The prediction log (34.9) holds what was scored and decided, never an input value; the outcome log holds what really
happened, once known. ``monitor_log`` reads them against the model's monitoring plan (30.15) and its holdout, a window at
a time (a day, a week or a batch):

- the score's distribution against the training rows' (its PSI) and the share the rule flags against its band, as
  ``monitoring.monitor`` reads a batch;
- the inputs' drift from profiles (:class:`BatchProfile`): the share of at least ``PROFILE_MIN_ROWS`` scored rows in each of
  the plan's bins, gathered by the service as it scores and written with no link to a request, so the inputs are watched
  without a value of a row ever being written (a profile of one row would be that row binned);
- live performance once a window is mature: its outcomes known by ``as_of`` and at least ``min_label_share`` of its rows
  labelled, or ``label_horizon_days`` past its end. A window read earlier would be judged on the rows labelled first,
  which are often the ones the model flagged (alerts are reviewed first): measured, a false alert on an undrifted day.

A window is alerted when an input or the score shifted past the alert line, when the rule flags outside its band, or when
live performance is worse than the holdout beyond both intervals in two read windows running: one such window is a
warning (measured 2026-10-25 on twenty simulated runs: one window alone was worse by chance in about 2 of 120 undrifted
windows, so a daily alert on one window would ring several times a year for nothing). ``as_of`` reads the log as it stood
then: nothing scored, profiled or recorded after it. Pure numpy and pandas: the host reads its logs and records the alerts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field, field_validator

# monitoring and serving are read at call time: contracts imports this module at its end, while either may still be
# loading (a host that imports one of them first would otherwise meet a half-made module)
from . import monitoring as _monitoring
from . import serving as _serving
from .contracts import CONTRACTS, Contract, Versioned, _version

KEY = r"^[0-9a-f]{32}$"  # as serving.KEY: a salted hash or a made-up key
WINDOWS = ("day", "week", "batch")
MIN_LABELLED = 100
MIN_LABEL_SHARE = 0.9
PROFILE_MIN_ROWS = 20  # a profile covers at least this many scored rows: never one row binned
BOOTSTRAP = 200
METRICS = {"binary": ("roc_auc", True), "multiclass": ("accuracy", True), "regression": ("mae", False)}


def _iso(value: str) -> str:
    return _serving._iso(value)


def _moment(value: str) -> datetime:
    return _serving._moment(value)


class BatchProfile(Versioned):
    """Scored rows' inputs as the plan's bins see them (34.10): each column's share of the rows in each bin, over at least
    ``PROFILE_MIN_ROWS`` rows and with no link to a request. Never a value: a bin's edges are the plan's, a share a count."""

    schema_version: Literal["dclab_core.batch_profile.v1"] = _version("batch_profile")
    model_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.:-]+$")
    model_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    at: str = Field(description="When the last row it covers was scored (it is placed in that moment's window).")
    rows: int = Field(ge=PROFILE_MIN_ROWS)
    shares: dict[str, list[float]] = Field(default_factory=dict, description="Each monitored column's bin shares, in the plan's bins.")

    @field_validator("at")
    @classmethod
    def _time(cls, value: str) -> str:
        return _iso(value)

    @field_validator("shares")
    @classmethod
    def _shares(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        for column, shares in value.items():
            if not shares or any(not 0 <= s <= 1 for s in shares) or abs(sum(shares) - 1) > 0.01:
                raise ValueError(f"{column}: a column's bin shares are between 0 and 1 and add up to one")
        return value


CONTRACTS["batch_profile"] = BatchProfile


def counts(plan: Any, frame: Any) -> dict[str, np.ndarray]:
    """Each plan column's count of the rows in each of its bins; a column the rows lack counts as missing on every row (a
    caller that stops sending a column is seen, as the model sees it: missing)."""
    import pandas as pd

    n = len(frame)
    out = {}
    for base in plan.columns:
        values = frame[base.column] if base.column in getattr(frame, "columns", []) else pd.Series([np.nan] * n)
        shares = _monitoring.shares_of(base, values)
        if shares:
            out[base.column] = np.round(np.asarray(shares, dtype=float) * n)
    return out


class Profiler:
    """Gathers bin counts as rows are scored and gives a profile only once ``PROFILE_MIN_ROWS`` rows are in it."""

    def __init__(self, plan: Any, model_id: str, model_version: str, min_rows: int = PROFILE_MIN_ROWS):
        self.plan, self.model_id, self.model_version, self.min_rows = plan, model_id, model_version, max(int(min_rows), PROFILE_MIN_ROWS)
        self.rows, self.totals = 0, {}

    def add(self, frame: Any, at: str) -> Optional[BatchProfile]:
        for column, c in counts(self.plan, frame).items():
            self.totals[column] = self.totals.get(column, 0) + c
        self.rows += len(frame)
        if self.rows < self.min_rows:
            return None
        shares = {k: [round(float(x), 4) for x in v / self.rows] for k, v in self.totals.items()}
        shares = {k: v for k, v in shares.items() if abs(sum(v) - 1) <= 0.01}
        made = BatchProfile(model_id=self.model_id, model_version=self.model_version, at=at, rows=self.rows, shares=shares)
        self.rows, self.totals = 0, {}
        return made


def profile(plan: Any, frame: Any, *, model_id: str, model_version: str, at: str) -> BatchProfile:
    """One profile of rows already gathered (at least ``PROFILE_MIN_ROWS``): the library's way for a host that batches itself."""
    if len(frame) < PROFILE_MIN_ROWS:
        raise ValueError(f"A profile covers at least {PROFILE_MIN_ROWS} rows: fewer would show a row's own bins")
    made = Profiler(plan, model_id, model_version).add(frame, at)
    if made is None:  # pragma: no cover - the row count was checked above
        raise ValueError("No profile was made")
    return made


class WindowReading(Contract):
    window: str
    start: str
    end: str
    rows: int
    score_psi: Optional[float] = None
    flagged_share: Optional[float] = None
    flagged_band: Optional[tuple[float, float]] = None
    columns_shifted: list[str] = Field(default_factory=list)
    labelled: int = 0
    mature: bool = False
    metric: Optional[str] = None
    value: Optional[float] = None
    interval: list[Optional[float]] = Field(default_factory=lambda: [None, None])
    vs_holdout: Literal["within", "worse", "better", "unread"] = "unread"
    status: Literal["stable", "warn", "alert", "too_few_rows"]
    alerts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LiveReport(Versioned):
    """A model in use, window by window (34.10, v1): what moved, how it performs once outcomes are known, and the first
    window that raised an alert."""

    schema_version: Literal["dclab_core.live_report.v1"] = _version("live_report")
    model_id: str
    model_versions: list[str] = Field(default_factory=list)
    window: Literal["day", "week", "batch"]
    as_of: str
    holdout: Optional[dict[str, Any]] = None
    windows: list[WindowReading] = Field(default_factory=list)
    first_alert: Optional[str] = None
    alerts: list[str] = Field(default_factory=list)
    status: Literal["stable", "warn", "alert", "no_rows"]
    note: str = ""


CONTRACTS["live_report"] = LiveReport


def _window_of(at: str, batch: Optional[str], window: str) -> str:
    moment = _moment(at).astimezone(timezone.utc)
    if window == "day":
        return moment.date().isoformat()
    if window == "week":
        year, week, _ = moment.isocalendar()
        return f"{year}-W{week:02d}"
    return str(batch)


def _auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    """ROC AUC by the rank sum, ties at their average rank (vectorised: 34.10's review measured the loop at 14 s a report)."""
    y = np.asarray(y).astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return None
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    avg = (starts + ends + 1) / 2.0  # the 1-based average rank of each run of ties
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _metric(task: str, y: np.ndarray, s: np.ndarray, c: np.ndarray) -> Optional[float]:
    if task == "binary":
        return _auc(y.astype(int), s)
    if task == "multiclass":
        return float(np.mean(c == y)) if len(y) else None
    return float(np.mean(np.abs(s - y))) if len(y) else None


def _interval(task: str, y: np.ndarray, s: np.ndarray, c: np.ndarray, draws: int, rng: np.random.Generator,
              groups: Optional[np.ndarray] = None) -> list[Optional[float]]:
    """A 90% bootstrap interval, over rows, or over whole groups when ``groups`` are given (rows of one entity move together)."""
    members = None
    if groups is not None:
        codes = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)[1]
        order = np.argsort(codes, kind="stable")
        members = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    values = []
    for _ in range(draws):
        if members is not None:
            rows = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        else:
            rows = rng.integers(0, len(y), len(y))
        v = _metric(task, y[rows], s[rows], c[rows])
        if v is not None:
            values.append(v)
    if len(values) < max(20, draws // 2):
        return [None, None]
    return [round(float(np.quantile(values, 0.05)), 4), round(float(np.quantile(values, 0.95)), 4)]


def reference(task: str, y_true: Sequence[Any], predictions: Any, *, groups: Optional[Sequence[Any]] = None,
              bootstrap: int = BOOTSTRAP, seed: int = 0) -> dict[str, Any]:
    """The holdout's value of the measure the log is read with (ROC AUC, accuracy or mean absolute error) and its 90%
    interval (over whole groups when the project's split keeps them together): what a model in use is compared with.
    ``predictions``: the positive class's score, the class probabilities (multiclass) or the predicted values."""
    if task not in METRICS:
        raise ValueError(f"A task is one of {', '.join(METRICS)}")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if task == "multiclass":
        s, c = np.full(len(y), np.nan), (p.argmax(axis=1) if p.ndim == 2 else p).astype(float)
    else:
        s, c = (p[:, -1] if p.ndim == 2 else p), np.full(len(y), -1.0)
    value = _metric(task, y, s, c)
    if value is None:
        return {"metric": METRICS[task][0], "value": None, "interval": [None, None]}
    g = None if groups is None else np.asarray(groups, dtype=object)
    return {"metric": METRICS[task][0], "value": round(value, 4),
            "interval": _interval(task, y, s, c, bootstrap, np.random.default_rng(seed), groups=g),
            "over": "groups" if g is not None else "rows"}


def monitor_log(plan: Any, predictions: Iterable[Any], outcomes: Iterable[Any] = (), *, profiles: Iterable[BatchProfile] = (),
                window: str = "day", holdout: Optional[Mapping[str, Any]] = None, as_of: Optional[str] = None,
                min_labelled: int = MIN_LABELLED, min_label_share: float = MIN_LABEL_SHARE, label_horizon_days: Optional[float] = None,
                bootstrap: int = BOOTSTRAP, seed: int = 0) -> LiveReport:
    """The model's log read window by window (see the module). ``holdout``: ``live.reference`` from the final stage;
    ``as_of``: the moment the log is read as it stood then (default: the latest moment in it). A window's performance is
    read once it is mature: ``label_horizon_days`` past its end when the host knows its label delay, else with at least
    ``min_label_share`` of its rows labelled; and with at least ``min_labelled`` outcomes."""
    import pandas as pd

    if window not in WINDOWS:
        raise ValueError(f"A window is one of {', '.join(WINDOWS)}")
    records = list(predictions)
    known = list(outcomes)
    moments = [_moment(r.at) for r in records] + [_moment(o.recorded_at) for o in known]
    cut = _moment(_iso(as_of)) if as_of else (max(moments) if moments else datetime.now(timezone.utc))
    records = [r for r in records if _moment(r.at) <= cut]  # the log as it stood at the cut
    if not records:
        return LiveReport(model_id="-", window=window, as_of=cut.isoformat(), status="no_rows", note="Nothing was scored by then.")
    tasks = {r.task for r in records}
    if len(tasks) != 1:
        raise ValueError("A log is read for one task at a time")
    task = tasks.pop()
    latest: dict[str, Any] = {}
    for o in known:  # the outcome recorded last wins, among those known and recorded by the cut
        if _moment(o.known_at) <= cut and _moment(o.recorded_at) <= cut and (
                o.key not in latest or _moment(o.recorded_at) >= _moment(latest[o.key].recorded_at)):
            latest[o.key] = o
    by_window: dict[str, list[Any]] = {}
    for r in records:
        by_window.setdefault(_window_of(r.at, r.batch, window), []).append(r)
    pooled: dict[str, dict[str, np.ndarray]] = {}
    pooled_rows: dict[str, int] = {}
    for p in profiles:  # each window's inputs: the profiles' shares weighted by their rows, placed by when they were made
        if _moment(p.at) > cut or window == "batch":
            continue  # a profile is linked to no request, so it has no batch window
        w = _window_of(p.at, None, window)
        if w not in by_window:
            continue
        pooled_rows[w] = pooled_rows.get(w, 0) + p.rows
        for column, shares in p.shares.items():
            acc = pooled.setdefault(w, {})
            acc[column] = acc.get(column, 0) + np.asarray(shares, dtype=float) * p.rows
    metric, higher = METRICS[task]
    reference_ = dict(holdout) if holdout else None
    if reference_ and reference_.get("metric") != metric:
        reference_ = None  # another measure than the one read live: not compared
    scores_only = plan.model_copy(update={"columns": []})
    baselines = {b.column: b for b in plan.columns}
    rng = np.random.default_rng(seed)
    readings, alerts = [], []
    previous_worse = False  # the last window whose performance was read
    for name in sorted(by_window, key=lambda k: min(_moment(r.at) for r in by_window[k])):
        rows = by_window[name]
        n = len(rows)
        scores = [r.score for r in rows if r.score is not None]
        report = _monitoring.monitor(scores_only, pd.DataFrame(index=range(n)), scores if len(scores) == n else None)
        w_alerts, w_warn = list(report.alerts), list(report.warnings)
        shifted = []
        for column, total in pooled.get(name, {}).items():
            base = baselines.get(column)
            if base is None or len(total) != len(base.shares):
                continue
            index = _monitoring.psi(base.shares, total / max(pooled_rows[name], 1))
            if pooled_rows[name] >= plan.min_rows and index >= plan.psi_alert:
                shifted.append(column)
                w_alerts.append(f"{column} shifted (PSI {index:.3f}, the alert line {plan.psi_alert})")
            elif pooled_rows[name] >= plan.min_rows and index >= plan.psi_warn:
                w_warn.append(f"{column} moved (PSI {index:.3f})")
        labelled = [(r, latest[r.key]) for r in rows if r.key in latest]
        end = max(_moment(r.at) for r in rows)
        mature = (cut - end >= timedelta(days=label_horizon_days)) if label_horizon_days is not None else len(labelled) >= min_label_share * n
        value, interval, versus = None, [None, None], "unread"
        if mature and len(labelled) >= min_labelled:
            y = np.array([o.outcome for _, o in labelled], dtype=float)
            s = np.array([r.score if r.score is not None else np.nan for r, _ in labelled], dtype=float)
            c = np.array([r.predicted_class if r.predicted_class is not None else -1 for r, _ in labelled], dtype=float)
            value = _metric(task, y, s, c)
            interval = _interval(task, y, s, c, bootstrap, rng) if value is not None else [None, None]
            ref_iv = (reference_ or {}).get("interval") or [None, None]
            if value is not None and reference_ and None not in interval and None not in ref_iv:
                lo, hi = interval
                worse = hi < ref_iv[0] if higher else lo > ref_iv[1]
                better = lo > ref_iv[1] if higher else hi < ref_iv[0]
                versus = "worse" if worse else "better" if better else "within"
                if worse:
                    said = (f"live {metric} {value:.3f} (90% {lo:.3f} to {hi:.3f}) is worse than the holdout's "
                            f"{reference_.get('value')} (its interval {ref_iv[0]} to {ref_iv[1]})")
                    (w_alerts if previous_worse else w_warn).append(said + ("; the second window running" if previous_worse else
                                                                            "; one window: an alert if the next one is worse too"))
                previous_worse = worse
        status = "alert" if w_alerts else "warn" if w_warn else "too_few_rows" if n < plan.min_rows else "stable"
        times = sorted(r.at for r in rows)
        readings.append(WindowReading(window=name, start=times[0], end=times[-1], rows=n, score_psi=report.score.psi if report.score else None,
                                      flagged_share=report.flagged_share, flagged_band=report.flagged_band, columns_shifted=shifted,
                                      labelled=len(labelled), mature=bool(mature), metric=metric if value is not None else None,
                                      value=None if value is None else round(value, 4), interval=interval, vs_holdout=versus,
                                      status=status, alerts=w_alerts, warnings=w_warn))
        alerts += [f"{name}: {a}" for a in w_alerts]
    first = next((r.window for r in readings if r.status == "alert"), None)
    overall = "alert" if first else "warn" if any(r.status == "warn" for r in readings) else "stable"
    unread = sum(1 for r in readings if r.vs_holdout == "unread")
    note = (f"{len(records)} predictions in {len(readings)} {window} windows; {sum(r.labelled for r in readings)} with an outcome known by "
            "the reading" + (f"; {unread} windows not read for performance (not mature yet: the label delay; or too few rows)" if unread else "")
            + f"; inputs read from profiles of at least {PROFILE_MIN_ROWS} rows, never a value.")
    return LiveReport(model_id=records[0].model_id, model_versions=sorted({r.model_version for r in records}), window=window,
                      as_of=cut.isoformat(), holdout=reference_, windows=readings, first_alert=first, alerts=alerts, status=overall, note=note)


__all__ = ["BatchProfile", "Profiler", "profile", "counts", "WindowReading", "LiveReport", "monitor_log", "reference", "WINDOWS",
           "MIN_LABELLED", "PROFILE_MIN_ROWS"]
