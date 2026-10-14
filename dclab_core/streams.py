"""Streams of events and sensor readings (package 16.11): alerts before incidents, judged as incidents.

Readings arrive per unit (a machine, a session) over time; incidents (failures) are rare and their labels sparse.
The question is asked at a moment: "should this machine raise an alert now?". A careful answer:

- :func:`windows`: one row per (unit, window end) at a regular step; features from each unit's readings in the window
  before the end only (mean, spread, last value, slope, and the change from the unit's own earlier level), never a
  reading after it. Each row records the last reading it read.
- :func:`label_windows`: a window is positive when an incident of its unit happens within the warning horizon after its
  end; windows inside an incident's own downtime are left out.
- :func:`evaluate`: units split by time; every method fitted on the first part of the training period, its alert
  threshold set out of sample on the second part for a budget of alerts per unit-day, the method chosen there by
  incident recall, and all of them scored once on the later holdout period. A training window counts only when its
  horizon and the delay before its label is known end by the cut. Methods: an isolation forest and a change score of
  each unit against its own past (no labels), any detectors a host injects (the studio passes its percentile rules and
  ECOD), and a classifier. Machines never seen are scored on the same periods. A future check recomputes the windows
  with every reading after a cut changed: no window ending before it may move.
- Scores are about incidents, not rows: the share of incidents with an alert in the warning horizon before them
  (incident recall; the horizon starts after the previous incident's downtime), the hours of warning the first alert
  gave (lead time), and alerts and false alerts per unit-day. An incident is scored only where its whole horizon lies in
  the scored period. How long after an incident starts it is detected is not measured: the windows in its downtime are
  dropped.

Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _when(values: Any) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True, errors="coerce").dt.tz_localize(None)


def windows(readings: pd.DataFrame, *, unit: str, time: str, sensors: Sequence[str], window_hours: float = 24, step_hours: float = 6,
            baseline_hours: float = 168) -> pd.DataFrame:
    """Rows (unit, end) every ``step_hours``; features from the unit's readings in the ``window_hours`` up to its last
    reading at or before the end (mean, spread, last value, slope per hour), and the change of the window's mean against
    the unit's own level in the ``baseline_hours`` before the window. Each row records the last reading it read."""
    data = readings[[unit, time, *sensors]].assign(_t=_when(readings[time]).to_numpy()).dropna(subset=["_t"]).sort_values([unit, "_t"])
    w, b = f"{int(window_hours * 3600)}s", f"{int(baseline_hours * 3600)}s"
    out = []
    for key, g in data.groupby(unit, sort=False):
        g = g.set_index("_t")
        hours = pd.Series((g.index - g.index[0]) / pd.Timedelta(hours=1), index=g.index)
        feats = pd.DataFrame(index=g.index)
        roll_h = hours.rolling(w)
        mh, mhh = roll_h.mean(), (hours * hours).rolling(w).mean()
        count = hours.rolling(w).count()
        for s in sensors:
            x = g[s].astype(float)
            r = x.rolling(w)
            mean, std = r.mean(), r.std(ddof=0)
            feats[f"{s}_mean"], feats[f"{s}_std"], feats[f"{s}_last"] = mean, std, x
            cov = (x * hours).rolling(w).mean() - mean * mh
            var = mhh - mh * mh
            feats[f"{s}_slope"] = (cov / var.where(var > 1e-12)).fillna(0.0)
            need = max(3, int(baseline_hours / max(window_hours, 1e-9)))  # a baseline of a few windows' readings, or none
            base_mean, base_std = x.rolling(b, min_periods=need).mean(), x.rolling(b, min_periods=need).std(ddof=0)
            base_std = base_std.where(base_std > 1e-6 * (base_mean.abs() + 1.0))  # a flat baseline gives no score, not 1e9
            # the baseline as it stood when the window began: read at (t - window), never inside the window
            at = pd.DataFrame({"_q": g.index - pd.Timedelta(seconds=int(window_hours * 3600))})
            known = pd.DataFrame({"_q": g.index, "bm": base_mean.to_numpy(), "bs": base_std.to_numpy()})
            joined = pd.merge_asof(at, known, on="_q", direction="backward")
            feats[f"{s}_vs_own"] = (mean.to_numpy() - joined["bm"].to_numpy()) / joined["bs"].to_numpy()
        feats["_count"] = count
        feats["_last_read"] = g.index
        ends = pd.date_range(g.index[0] + pd.Timedelta(hours=window_hours), g.index[-1], freq=pd.Timedelta(hours=step_hours))
        picked = pd.merge_asof(pd.DataFrame({"end": ends}), feats.reset_index(drop=True).assign(_q=g.index).rename(columns={"_q": "_at"}),
                               left_on="end", right_on="_at", direction="backward", tolerance=pd.Timedelta(hours=window_hours))
        picked = picked[picked["_count"].fillna(0) >= 3].drop(columns=["_count", "_at"])  # a gap longer than a window gives no window
        picked.insert(0, unit, key)
        out.append(picked)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def label_windows(rows: pd.DataFrame, incidents: pd.DataFrame, *, unit: str, time: str, horizon_hours: float = 48,
                  downtime_hours: float = 24) -> pd.DataFrame:
    """``target`` 1 when an incident of the unit happens in (end, end + horizon]; rows within ``downtime_hours`` after an
    incident are dropped (the machine is being repaired, not warning)."""
    inc = incidents.assign(_t=_when(incidents[time]).to_numpy())
    rows = rows.copy()
    rows["target"] = 0
    keep = np.ones(len(rows), dtype=bool)
    h, d = pd.Timedelta(hours=horizon_hours), pd.Timedelta(hours=downtime_hours)
    for key, g in inc.groupby(unit):
        mine = (rows[unit] == key).to_numpy()
        for t in g["_t"]:
            ends = rows["end"]
            rows.loc[mine & ((ends < t) & (ends >= t - h)).to_numpy(), "target"] = 1
            keep &= ~(mine & ((ends >= t) & (ends < t + d)).to_numpy())
    return rows[keep].reset_index(drop=True)


def _threshold(scores: np.ndarray, unit_days: float, budget: float) -> float:
    """The score above which alerts stay within ``budget`` alerts per unit-day on these rows."""
    allowed = max(int(budget * unit_days), 1)
    ordered = np.sort(scores[np.isfinite(scores)])[::-1]
    return float(ordered[min(allowed, len(ordered)) - 1]) if len(ordered) else np.inf


def incident_metrics(rows: pd.DataFrame, alerts: np.ndarray, incidents: pd.DataFrame, *, unit: str, time: str, horizon_hours: float,
                     start: pd.Timestamp, stop: pd.Timestamp, downtime_hours: float = 24) -> dict[str, Any]:
    """Incidents whose time is in (start, stop] (callers pass bounds that leave each one its whole warning horizon in the
    scored windows): caught when an alert came in [t - horizon, t), never before the previous incident's downtime ended;
    the hours of warning the first such alert gave (lead time before the incident); alerts per unit-day; and false alerts
    (not followed by an incident within the horizon) per unit-day."""
    inc = incidents.assign(_t=_when(incidents[time]).to_numpy()).sort_values([unit, "_t"])
    inc["_prev"] = inc.groupby(unit)["_t"].shift(1)
    inc = inc[(inc["_t"] > start) & (inc["_t"] <= stop) & inc[unit].isin(rows[unit].unique())]
    fired = rows[alerts]
    h, d = pd.Timedelta(hours=horizon_hours), pd.Timedelta(hours=downtime_hours)
    caught, leads = 0, []
    for _, i in inc.iterrows():
        opens = i["_t"] - h if pd.isna(i["_prev"]) else max(i["_t"] - h, i["_prev"] + d)
        before = fired[(fired[unit] == i[unit]) & (fired["end"] < i["_t"]) & (fired["end"] >= opens)]
        if len(before):
            caught += 1
            leads.append((i["_t"] - before["end"].min()).total_seconds() / 3600)
    unit_days = max(rows.groupby(unit)["end"].agg(lambda e: (e.max() - e.min()).total_seconds() / 86400).sum(), 1e-9)
    false = int(len(fired) - fired["target"].sum()) if "target" in fired else None
    return {"incidents": int(len(inc)), "caught": caught, "incident_recall": _r(caught / len(inc), 3) if len(inc) else None,
            "lead_time_hours_median": _r(np.median(leads), 1) if leads else None, "alerts": int(len(fired)),
            "alerts_per_unit_day": _r(len(fired) / unit_days, 3), "false_alerts_per_unit_day": _r(false / unit_days, 3) if false is not None else None,
            "unit_days": _r(unit_days, 1)}


class _Forest:
    """An isolation forest as a detector: fitted on windows without labels; its score is how unusual a window is."""

    def __init__(self, seed: int = 0):
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(n_estimators=200, random_state=seed)

    def fit(self, X: Any, y: Any = None) -> "_Forest":
        self.model.fit(X)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        s = -self.model.score_samples(X)
        return np.column_stack([1 - s, s])


class _OwnChange:
    """The largest change of a window against its unit's own past level (no labels, nothing fitted)."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def fit(self, X: Any, y: Any = None) -> "_OwnChange":
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        cols = [i for i, c in enumerate(self.columns) if c.endswith("_vs_own")]
        v = np.abs(np.asarray(X, dtype=float)[:, cols]) if cols else np.zeros((len(X), 1))
        s = np.where(np.isfinite(v).any(axis=1), np.nanmax(np.where(np.isfinite(v), v, -np.inf), axis=1), 0.0)
        return np.column_stack([1 - s, s])


class _Classifier:
    def __init__(self, seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=seed)

    def fit(self, X: Any, y: Any = None) -> "_Classifier":
        if y is None or len(np.unique(y)) < 2:
            raise ValueError("the classifier needs windows of both kinds (before an incident and not) to learn from")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)


def _unit_days(r: pd.DataFrame, unit: str) -> float:
    return max(r.groupby(unit)["end"].agg(lambda e: (e.max() - e.min()).total_seconds() / 86400).sum(), 1e-9)


def _future_check(readings: pd.DataFrame, rows: pd.DataFrame, *, unit: str, time: str, sensors: Sequence[str], window_hours: float,
                  step_hours: float) -> dict[str, Any]:
    """Recompute the windows with every reading after a cut changed: a window ending at or before the cut that changes
    read the future (whatever its own bookkeeping says)."""
    cut = rows["end"].quantile(0.5)
    changed = readings.copy()
    after = _when(changed[time]).to_numpy() > np.datetime64(cut)
    for s in sensors:
        changed.loc[after, s] = pd.to_numeric(changed.loc[after, s], errors="coerce") * 3 + 7
    one = windows(readings, unit=unit, time=time, sensors=sensors, window_hours=window_hours, step_hours=step_hours)
    two = windows(changed, unit=unit, time=time, sensors=sensors, window_hours=window_hours, step_hours=step_hours)
    cols = [c for c in one.columns if c not in ("_last_read",)]
    a = one[one["end"] <= cut][cols].reset_index(drop=True)
    b = two[two["end"] <= cut][cols].reset_index(drop=True)
    moved = int((~np.isclose(a.drop(columns=[unit, "end"]).to_numpy(float), b.drop(columns=[unit, "end"]).to_numpy(float), equal_nan=True)).any(axis=1).sum()) \
        if a.shape == b.shape else len(a)
    return {"cut": str(cut), "windows_before_the_cut": int(len(a)), "changed_by_the_future": moved}


def evaluate(readings: pd.DataFrame, incidents: pd.DataFrame, *, unit: str, time: str, sensors: Sequence[str], incident_time: str,
             window_hours: float = 24, step_hours: float = 6, horizon_hours: float = 48, downtime_hours: float = 24,
             label_delay_hours: float = 0, budget_per_unit_day: float = 0.2, holdout_share: float = 0.3,
             detectors: Optional[dict[str, Any]] = None, seed: int = 0) -> dict[str, Any]:
    """Windows and labels; every method (detectors with no labels, and a classifier) fitted on the first part of the
    training period, its alert threshold set on the second part (out of sample) for the alert budget, the method chosen
    there by incident recall, and all of them scored once on the later holdout period by incident. A training window is
    used only when its horizon, and the delay before its label is known, end by the cut. Units never seen are scored on
    the same periods: methods fitted on the other units' training period, read on these units' holdout period.
    ``detectors``: name -> a factory of an estimator with ``fit(X)`` and ``predict_proba(X)[:, 1]`` as its score."""
    every = windows(readings, unit=unit, time=time, sensors=sensors, window_hours=window_hours, step_hours=step_hours)
    if every.empty:
        raise ValueError("no window: the readings are shorter than one window, or too sparse")
    cut = every["end"].quantile(1 - holdout_share)  # from the windows alone: the incidents never move the line
    rows = label_windows(every, incidents, unit=unit, time=incident_time, horizon_hours=horizon_hours, downtime_hours=downtime_hours)
    features = [c for c in rows.columns if c not in (unit, "end", "target", "_last_read")]
    h, delay = pd.Timedelta(hours=horizon_hours), pd.Timedelta(hours=label_delay_hours)
    train = rows[rows["end"] + h + delay <= cut]  # its label is complete and known by the cut
    test = rows[rows["end"] > cut]
    vcut = train["end"].quantile(0.6)
    fit = train[train["end"] + h + delay <= vcut]
    val = train[train["end"] > vcut]
    inc_t = _when(incidents[incident_time])
    matched = int((incidents[unit].isin(rows[unit].unique()).to_numpy() & (inc_t <= vcut).to_numpy()).sum())
    if matched < 3 or fit["target"].nunique() < 2 or val.empty or test.empty:
        raise ValueError(f"{matched} incidents of these units in the first part of the training period: too few to learn from (at least 3), "
                         "or no period left to set thresholds and to test")
    end_all = rows["end"].max()
    makers = {"isolation_forest": lambda: _Forest(seed), "own_change": lambda: _OwnChange(features), **(detectors or {}),
              "supervised": lambda: _Classifier(seed)}

    def run(fit_rows, val_rows, test_rows, start, stop):
        out, chosen = {}, {}
        for name, make in makers.items():
            model = make().fit(fit_rows[features].fillna(0.0).to_numpy(), fit_rows["target"].to_numpy())
            v = model.predict_proba(val_rows[features].fillna(0.0).to_numpy())[:, 1]
            threshold = _threshold(v, _unit_days(val_rows, unit), budget_per_unit_day)  # out of sample: the rows the model never saw
            chosen[name] = incident_metrics(val_rows, v >= threshold, incidents, unit=unit, time=incident_time, horizon_hours=horizon_hours,
                                            start=vcut + h, stop=cut - h - delay, downtime_hours=downtime_hours)
            t = model.predict_proba(test_rows[features].fillna(0.0).to_numpy())[:, 1]
            out[name] = {"threshold": _r(threshold, 6), **incident_metrics(test_rows, t >= threshold, incidents, unit=unit, time=incident_time,
                                                                           horizon_hours=horizon_hours, start=start, stop=stop,
                                                                           downtime_hours=downtime_hours)}
        return out, chosen

    results, validation = run(fit, val, test, cut + h, end_all)
    ranked = {n: m for n, m in validation.items() if m["incidents"]}
    best = max(ranked or validation, key=lambda n: ((ranked or validation)[n]["incident_recall"] or 0,
                                                    -((ranked or validation)[n]["false_alerts_per_unit_day"] or 0)))
    units = np.array(sorted(rows[unit].unique()))
    held = set(np.random.default_rng(seed).choice(units, max(1, len(units) // 4), replace=False))
    unseen = None
    others = lambda frame: frame[~frame[unit].isin(held)]  # noqa: E731
    if int(others(fit)["target"].sum()) >= 3:
        try:
            got, _ = run(others(fit), others(val), test[test[unit].isin(held)], cut + h, end_all)
            unseen = got.get(best)
        except ValueError:
            unseen = None
    return {"windows": int(len(rows)), "window_hours": window_hours, "step_hours": step_hours, "horizon_hours": horizon_hours,
            "label_delay_hours": label_delay_hours, "budget_per_unit_day": budget_per_unit_day,
            "split": {"by": "time", "cut": str(cut), "validation_from": str(vcut), "fit_windows": int(len(fit)), "validation_windows": int(len(val)),
                      "holdout_windows": int(len(test)), "train_labels_until": str(train["end"].max() + h + delay),
                      "thresholds_set_on": "the validation period, out of sample"},
            "future_check": _future_check(readings, rows, unit=unit, time=time, sensors=sensors, window_hours=window_hours, step_hours=step_hours),
            "methods": results, "validation": validation, "best": best, "chosen_by": "incident recall on the validation period",
            "unseen_units": unseen, "held_out_units": sorted(map(str, held)),
            "not_measured": ["detection delay after an incident starts (alerts after an incident are not counted: the windows in its downtime are dropped)"]}


# ---------------------------------------------------------------------------- 16.15 C: in the engine, split / choose / score

def _makers(features: Sequence[str], detectors: Optional[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {"isolation_forest": lambda: _Forest(seed), "own_change": lambda: _OwnChange(features), **(detectors or {}),
            "supervised": lambda: _Classifier(seed)}


def split(readings: pd.DataFrame, *, unit: str, time: str, sensors: Sequence[str], window_hours: float = 24, step_hours: float = 6,
          horizon_hours: float = 48, label_delay_hours: float = 0, holdout_share: float = 0.3) -> dict[str, Any]:
    """Where the periods end, read from the windows' dates only (never an incident): the holdout is the windows after
    ``cut`` (the latest ``holdout_share`` of window ends); a training window counts when its horizon and the delay before
    its label end by the cut; models are fitted on the training windows whose label is known by ``validation_from`` and
    their thresholds set on the windows after it."""
    every = windows(readings, unit=unit, time=time, sensors=sensors, window_hours=window_hours, step_hours=step_hours)
    if every.empty:
        raise ValueError("no window: the readings are shorter than one window, or too sparse")
    h, delay = pd.Timedelta(hours=horizon_hours), pd.Timedelta(hours=label_delay_hours)
    first, last = every["end"].min(), every["end"].max()
    # on the time axis, not by counting windows: a machine down after a failure writes fewer windows, and must not move the line
    cut = (first + (1 - holdout_share) * (last - first)).floor("h")
    usable = cut - h - delay
    if usable <= first:
        raise ValueError("no training window's label is known by the holdout's start: longer readings, or a shorter horizon")
    vcut = (first + 0.6 * (usable - first)).floor("h")
    known = every[every["end"] + h + delay <= cut]
    return {"by": "time", "cut": str(cut), "validation_from": str(vcut), "data_end": str(every["end"].max()),
            "train_labels_until": str(known["end"].max() + h + delay), "window_hours": window_hours, "step_hours": step_hours,
            "horizon_hours": horizon_hours, "label_delay_hours": label_delay_hours,
            "thresholds_set_on": "the validation period, out of sample"}


def known_by(readings: pd.DataFrame, incidents: pd.DataFrame, parts: dict[str, Any], *, time: str, incident_time: str) -> tuple:
    """The readings before the holdout's start, and the incidents whose label is known by then: what training may read."""
    cut = pd.Timestamp(parts["cut"])
    delay = pd.Timedelta(hours=parts["label_delay_hours"])
    r = readings[(_when(readings[time]) <= cut).to_numpy()]
    i = incidents[((_when(incidents[incident_time]) + delay) <= cut).to_numpy()]
    return r, i


def training_view(readings: pd.DataFrame, incidents: pd.DataFrame, parts: dict[str, Any], *, unit: str, time: str, sensors: Sequence[str],
                  incident_time: str, downtime_hours: float = 24) -> dict[str, Any]:
    """The fit and validation windows, labelled, from the readings and incidents known by the holdout's start only."""
    r, i = known_by(readings, incidents, parts, time=time, incident_time=incident_time)
    cut, vcut = pd.Timestamp(parts["cut"]), pd.Timestamp(parts["validation_from"])
    h, delay = pd.Timedelta(hours=parts["horizon_hours"]), pd.Timedelta(hours=parts["label_delay_hours"])
    every = windows(r, unit=unit, time=time, sensors=sensors, window_hours=parts["window_hours"], step_hours=parts["step_hours"])
    rows = label_windows(every, i, unit=unit, time=incident_time, horizon_hours=parts["horizon_hours"], downtime_hours=downtime_hours)
    train = rows[rows["end"] + h + delay <= cut]
    fit, val = train[train["end"] + h + delay <= vcut], train[train["end"] > vcut]
    features = [c for c in rows.columns if c not in (unit, "end", "target", "_last_read")]
    return {"fit": fit.reset_index(drop=True), "validation": val.reset_index(drop=True), "features": features,
            "incidents_known": int(len(i)), "readings_read": int(len(r)),
            "readings_after_the_cut_read": int((pd.to_datetime(rows["_last_read"]) > cut).sum()) if len(rows) else 0}


def _fit_and_threshold(view: dict[str, Any], name: str, make: Any, budget: float, unit: str) -> tuple:
    features = view["features"]
    model = make().fit(view["fit"][features].fillna(0.0).to_numpy(), view["fit"]["target"].to_numpy())
    v = model.predict_proba(view["validation"][features].fillna(0.0).to_numpy())[:, 1]
    return model, v, _threshold(v, _unit_days(view["validation"], unit), budget)


def choose(view: dict[str, Any], incidents: pd.DataFrame, parts: dict[str, Any], *, unit: str, incident_time: str, budget_per_unit_day: float = 0.2,
           downtime_hours: float = 24, detectors: Optional[dict[str, Any]] = None, seed: int = 0) -> dict[str, Any]:
    """Every method fitted on the fit windows, its threshold set on the validation windows for the alert budget, and
    scored there by incident; the best incident recall is chosen (fewer false alerts break a tie). The holdout's
    windows are never built here; ``incidents`` are only those known by the holdout's start."""
    fit, val = view["fit"], view["validation"]
    h, delay = pd.Timedelta(hours=parts["horizon_hours"]), pd.Timedelta(hours=parts["label_delay_hours"])
    matched = int((incidents[unit].isin(fit[unit].unique()).to_numpy() & (_when(incidents[incident_time]) <= pd.Timestamp(parts["validation_from"])).to_numpy()).sum())
    if matched < 3 or fit["target"].nunique() < 2 or val.empty:
        raise ValueError(f"{matched} incidents in the first part of the training period: too few to learn from (at least 3), or no period to set thresholds")
    out = {}
    for name, make in _makers(view["features"], detectors, seed).items():
        _, v, threshold = _fit_and_threshold(view, name, make, budget_per_unit_day, unit)
        out[name] = {"threshold": _r(threshold, 6), **incident_metrics(val, v >= threshold, incidents, unit=unit, time=incident_time, horizon_hours=parts["horizon_hours"],
                                                                       start=pd.Timestamp(parts["validation_from"]) + h, stop=pd.Timestamp(parts["cut"]) - h - delay,
                                                                       downtime_hours=downtime_hours)}
    if not any(m["incidents"] for m in out.values()):
        raise ValueError("no incident has its whole warning horizon in the validation period: no method can be chosen by incident recall "
                         "(longer readings, or a shorter horizon)")
    ranked = out
    best = max(ranked, key=lambda n: (ranked[n]["incident_recall"] or 0, -(ranked[n]["false_alerts_per_unit_day"] or 0)))
    return {"validation": out, "chosen": best, "chosen_by": "incident recall on the validation period (fewer false alerts break a tie)",
            "fit_windows": int(len(fit)), "validation_windows": int(len(val))}


def score(readings: pd.DataFrame, incidents: pd.DataFrame, view: dict[str, Any], parts: dict[str, Any], chosen: str, *, unit: str, time: str,
          sensors: Sequence[str], incident_time: str, budget_per_unit_day: float = 0.2, downtime_hours: float = 24,
          detectors: Optional[dict[str, Any]] = None, seed: int = 0) -> dict[str, Any]:
    """The chosen method and the two that need no label, as fitted and thresholded in ``choose``, scored once on the
    holdout's windows (built from every reading, labelled with every incident); and the chosen one again fitted
    without a quarter of the units, read on those units' holdout windows (machines never seen)."""
    cut, h = pd.Timestamp(parts["cut"]), pd.Timedelta(hours=parts["horizon_hours"])
    every = windows(readings, unit=unit, time=time, sensors=sensors, window_hours=parts["window_hours"], step_hours=parts["step_hours"])
    rows = label_windows(every, incidents, unit=unit, time=incident_time, horizon_hours=parts["horizon_hours"], downtime_hours=downtime_hours)
    test = rows[rows["end"] > cut]
    lost = [c for c in view["features"] if c not in test.columns]
    if lost or test.empty:
        raise ValueError(f"the holdout has no window, or lacks training features: {', '.join(lost)}")
    makers = _makers(view["features"], detectors, seed)
    stop = rows["end"].max()
    out = {}
    for name in dict.fromkeys([chosen, "own_change", "isolation_forest"]):
        model, _, threshold = _fit_and_threshold(view, name, makers[name], budget_per_unit_day, unit)
        t = model.predict_proba(test[view["features"]].fillna(0.0).to_numpy())[:, 1]
        out[name] = {"threshold": _r(threshold, 6), **incident_metrics(test, t >= threshold, incidents, unit=unit, time=incident_time,
                                                                       horizon_hours=parts["horizon_hours"], start=cut + h, stop=stop, downtime_hours=downtime_hours)}
    units = np.array(sorted(view["fit"][unit].unique()))
    held = set(np.random.default_rng(seed).choice(units, max(1, len(units) // 4), replace=False))
    unseen, fitted_on = None, []
    others = {k: (v[~v[unit].isin(held)] if k in ("fit", "validation") else v) for k, v in view.items()}
    mine = test[test[unit].isin(held)]
    if int(others["fit"]["target"].sum()) >= 3 and others["fit"]["target"].nunique() == 2 and len(others["validation"]) and len(mine):
        model, _, threshold = _fit_and_threshold(others, chosen, makers[chosen], budget_per_unit_day, unit)
        fitted_on = sorted(map(str, set(others["fit"][unit]) | set(others["validation"][unit])))
        t = model.predict_proba(mine[view["features"]].fillna(0.0).to_numpy())[:, 1]
        unseen = incident_metrics(mine, t >= threshold, incidents, unit=unit, time=incident_time, horizon_hours=parts["horizon_hours"],
                                  start=cut + h, stop=stop, downtime_hours=downtime_hours)
    return {"holdout": out, "holdout_windows": int(len(test)), "unseen_units": unseen, "held_out_units": sorted(map(str, held)),
            "unseen_fitted_on_units": fitted_on}


def written_ahead(view: dict[str, Any], incidents: pd.DataFrame, *, unit: str, incident_time: str, sensors: Sequence[str],
                  horizon_hours: float, limit: float = 0.5) -> dict[str, Any]:
    """Sensors whose value follows the hours to the unit's next known incident on windows far before it (more than twice
    the warning horizon): a physical warning sign cannot see that far, a column filled in afterwards (hours to the next
    failure, a service counter reset at repair) can. Read on the training windows and the incidents known by the cut;
    each such sensor is to be confirmed by a person (|Spearman| at least ``limit``)."""
    rows = pd.concat([view["fit"], view["validation"]], ignore_index=True)
    inc = incidents.assign(_t=_when(incidents[incident_time]).to_numpy()).dropna(subset=["_t"]).sort_values("_t")
    nxt = pd.merge_asof(rows[[unit, "end"]].reset_index().sort_values("end"), inc[[unit, "_t"]].rename(columns={"_t": "_next"}),
                        left_on="end", right_on="_next", by=unit, direction="forward").set_index("index").sort_index()
    hours = (nxt["_next"] - rows["end"]).dt.total_seconds() / 3600
    far = hours > 2 * horizon_hours
    out, why = [], {}
    for s in sensors:
        col = f"{s}_last"
        if col not in rows or far.sum() < 30:
            continue
        rho = rows.loc[far, col].astype(float).rank().corr(hours[far].rank())
        if pd.notna(rho) and abs(rho) >= limit:
            out.append(s)
            why[s] = (f"on {int(far.sum())} windows more than {2 * horizon_hours:g} hours before the next incident, it follows the hours left "
                      f"(Spearman {rho:.2f}): written knowing when the failure came")
    return {"to_confirm": out, "why": why, "windows_read": int(far.sum())}


def without(view: dict[str, Any], sensors: Sequence[str]) -> dict[str, Any]:
    """The view with every feature of ``sensors`` left out."""
    gone = {c for s in sensors for c in view["features"] if c.startswith(f"{s}_")}
    return {**view, "features": [c for c in view["features"] if c not in gone]}


def wilson(caught: int, total: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    """The Wilson interval of a share (caught of total): exact enough for few incidents, never outside [0, 1]."""
    if not total:
        return None, None
    p = caught / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    w = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return _r(max(0.0, c - w), 4), _r(min(1.0, c + w), 4)
