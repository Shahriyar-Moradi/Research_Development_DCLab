"""Forecasting (package 16.3): many series, a horizon, rolling-origin backtests that never see the future.

A forecasting problem is a table of rows (series id, time, target, covariates). At a forecast moment (an origin) the
model may use, for each series, only the target's past and the covariates known at the origin for the dates ahead (a
promotion planned ahead is known; the day's weather is not). :func:`backtest` cuts the history at several origins
(rolling origin, with a gap between the last row used and the first date forecast), fits each model on what was known at
the origin only, forecasts ``horizon`` steps ahead, and scores against what happened:

- baselines: naive (the last value), seasonal naive (the value one season back) and a moving average;
- a global gradient-boosted model on lags and rolling windows of each series' past (computed from the past of the
  origin only), the calendar, and the covariates allowed at the moment, one model for all series;
- a classical model per series (simple exponential smoothing with a seasonal profile) where the series is long enough;
- MASE (against the seasonal naive's in-sample error) and sMAPE;
- prediction intervals by horizon step from residuals scaled by each series' own error scale; their coverage and pinball
  (quantile) loss measured on the last origin with intervals built from the earlier origins only.

The target is read only up to the origin: each origin records the last date of the target it used, and a forecast is
returned with its dates and values, so a test can change the future and see that no forecast moves.

:func:`covariate_check` says which covariates may be used at the forecast moment: the columns the person declares known
ahead, and the columns that describe a series (the same on every row of it); every other column is kept out. A declared
column whose day-to-day changes move with the target's is kept but marked "to confirm": it may be measured on the day.
Text covariates are coded from the training rows of each origin. Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

LAGS = (1, 7, 14)
WINDOWS = (7, 28)
MIN_CLASSICAL = 56  # rows a series needs for its own model
TRACKS = 0.3  # |correlation| of day-over-day changes above which a column moves with the target
SUMMARISES = 0.9  # |correlation| across series above which a series attribute looks like a summary of the target


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _tracking(frame: pd.DataFrame, series: str, time: str, target: str, column: str) -> float:
    """How much a numeric column's day-over-day changes move with the target's, within each series."""
    if not pd.api.types.is_numeric_dtype(frame[column]):
        return 0.0
    diffs = frame.sort_values([series, time]).groupby(series)[[column, target]].diff().dropna()
    if len(diffs) <= 10 or not diffs[column].std() > 0 or not diffs[target].std() > 0:
        return 0.0
    return float(diffs[column].corr(diffs[target]))


def _level_tracking(frame: pd.DataFrame, series: str, target: str, column: str) -> float:
    """How closely a column that is constant within each series follows each series' mean of the target, across series."""
    if not pd.api.types.is_numeric_dtype(frame[column]):
        return 0.0
    per = frame.groupby(series).agg(value=(column, "first"), level=(target, "mean")).dropna()
    if len(per) < 5 or not per["value"].std() > 0 or not per["level"].std() > 0:
        return 0.0
    return float(per["value"].corr(per["level"]))


def covariate_check(frame: pd.DataFrame, *, series: str, time: str, target: str, known_ahead: Sequence[str] = ()) -> dict[str, Any]:
    """Which covariates the forecast may use: ``kept`` (declared known ahead, or describing the series), ``kept_out``
    (everything else: a value for a future date is not known at the moment), and ``to_confirm`` (declared, but its
    day-to-day changes move with the target's, as a value measured on the day does; or constant in each series but
    following the series' mean of the target, kept out until declared). Each with its reason. Pass the training rows
    only: the check reads the target."""
    out: dict[str, Any] = {"kept": [], "kept_out": [], "to_confirm": [], "why": {}}
    for column in [c for c in frame.columns if c not in (series, time, target)]:
        if column in known_ahead:
            corr = _tracking(frame, series, time, target, column)
            out["kept"].append(column)
            if abs(corr) >= TRACKS:
                out["to_confirm"].append(column)
                out["why"][column] = (f"declared known ahead, but its day-to-day changes move with the target's (correlation {corr:.2f}), "
                                      "as a value measured on the day does: confirm it is set before the forecast moment")
            else:
                out["why"][column] = "declared known ahead of the forecast moment"
            continue
        if frame.groupby(series)[column].nunique(dropna=False).max() <= 1:
            level = _level_tracking(frame, series, target, column)
            if abs(level) >= SUMMARISES:  # a series attribute this close to the series' own level may be computed from the target
                out["kept_out"].append(column)
                out["to_confirm"].append(column)
                out["why"][column] = (f"the same on every row of its series, but it follows each series' mean of the target (correlation "
                                      f"{level:.2f} across series), as a summary computed from the target, perhaps over future dates, does: "
                                      "kept out until it is declared known ahead")
                continue
            out["kept"].append(column)
            out["why"][column] = "the same on every row of its series: it describes the series, so it is known ahead"
            continue
        out["kept_out"].append(column)
        out["why"][column] = "not declared known ahead of the forecast moment: a value for a future date is not known at the moment"
    return out


def _codes(past: pd.DataFrame, known: Sequence[str]) -> dict[str, dict[Any, int]]:
    """Codes for the text covariates, fixed on the rows known at the origin; a value first seen later is -1."""
    return {c: {v: i for i, v in enumerate(sorted(past[c].dropna().astype(str).unique()))}
            for c in known if not pd.api.types.is_numeric_dtype(past[c])}


def _value(plan: pd.DataFrame, date: pd.Timestamp, column: str, codes: dict[str, dict[Any, int]]) -> float:
    if date not in plan.index:
        return np.nan
    v = plan.at[date, column]
    if column in codes:
        return float(codes[column].get(str(v), -1)) if pd.notna(v) else np.nan
    return float(v) if pd.notna(v) else np.nan


def _features(history: pd.DataFrame, dates: pd.DatetimeIndex, *, target: str, known: Sequence[str], season: int, gap: int = 0,
              plan: Optional[pd.DataFrame] = None, codes: Optional[dict[str, dict[Any, int]]] = None) -> pd.DataFrame:
    """One series' features for ``dates`` (the ``horizon`` dates after the ``gap``): lags and windows of the target
    computed from ``history`` only (the past of the origin), the calendar, and the known-ahead covariates of those dates
    from ``plan`` (covariates only, never the target)."""
    y = history[target].to_numpy(dtype=float)
    rows = []
    for step, date in enumerate(dates, 1):
        feats: dict[str, float] = {"step": step, "dow": date.dayofweek, "month": date.month}
        for lag in LAGS:  # the value `lag` days before the forecast date, if that day is in the past of the origin
            back = lag - step - gap
            feats[f"lag_{lag}"] = y[len(y) - 1 - back] if 0 <= back < len(y) else np.nan
        feats["last"] = y[-1] if len(y) else np.nan
        for w in WINDOWS:
            feats[f"mean_{w}"] = float(np.mean(y[-w:])) if len(y) else np.nan
        feats["season_back"] = y[len(y) - season + (gap + step - 1) % season] if len(y) >= season else np.nan
        if plan is not None:
            for c in known:
                feats[c] = _value(plan, date, c, codes or {})
        rows.append(feats)
    return pd.DataFrame(rows, index=dates)


def _training_rows(history: pd.DataFrame, *, target: str, known: Sequence[str], season: int, horizon: int, gap: int,
                   codes: Optional[dict[str, dict[Any, int]]] = None) -> pd.DataFrame:
    """Training examples from one series' history, each built as an origin inside the history would have seen it: the
    past up to a cut, and the targets ``gap`` days after it."""
    out = []
    times = history.index
    plan = history[list(known)]
    for cut in range(max(WINDOWS[0], season) + 1, len(history) - gap - horizon + 1, max(1, horizon // 2)):
        past = history.iloc[:cut]
        dates = times[cut + gap: cut + gap + horizon]
        feats = _features(past, dates, target=target, known=known, season=season, gap=gap, plan=plan, codes=codes)
        feats[target] = history.loc[dates, target].to_numpy(dtype=float)
        out.append(feats)
    return pd.concat(out) if out else pd.DataFrame()


def _ses_seasonal(y: np.ndarray, season: int, horizon: int, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing of the level with a seasonal profile (the mean of each season position, as a ratio)."""
    profile = np.ones(season)
    if len(y) >= 2 * season:
        tail = y[-(len(y) // season) * season:].reshape(-1, season)  # column j: the days at position j counted from the end
        mean = tail.mean()
        profile = np.where(tail.mean(axis=0) > 0, tail.mean(axis=0) / mean, 1.0) if mean else profile
    position = (np.arange(len(y)) - len(y)) % season  # each past day's position in the same counting
    deseason = y / profile[position] if len(y) else y
    level = deseason[0] if len(deseason) else 0.0
    for value in deseason[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.array([level * profile[h % season] for h in range(horizon)])  # the day after the last is position 0 again


def _pinball(errs: np.ndarray, lo: np.ndarray, hi: np.ndarray, lo_q: float, hi_q: float) -> float:
    """The mean quantile loss of the two interval ends (errors and ends in the same scaled units)."""
    def one(q, e):
        return np.maximum(q * e, (q - 1) * e)
    return float(np.mean(one(lo_q, errs - lo) + one(hi_q, errs - hi)))


def _interval_check(scaled: dict[str, dict[str, dict[int, list[float]]]], last: str, lo_q: float, hi_q: float) -> dict[str, Any]:
    """For each model: intervals from the origins before ``last``, scored on ``last``: coverage and pinball loss."""
    out = {}
    for model, by_origin in scaled.items():
        earlier = [o for o in by_origin if o != last]
        if last not in by_origin or not earlier:
            continue
        cov, pin, n = [], [], 0
        for h, errs in by_origin[last].items():
            fit = np.array([e for o in earlier for e in by_origin[o].get(h, [])])
            if len(fit) < 10:
                continue
            lo, hi = np.quantile(fit, lo_q), np.quantile(fit, hi_q)
            e = np.array(errs)
            cov.append(np.mean((e >= lo) & (e <= hi)))
            pin.append(_pinball(e, np.full_like(e, lo), np.full_like(e, hi), lo_q, hi_q))
            n += len(e)
        if cov:
            out[model] = {"origin": last, "coverage": _r(np.mean(cov), 3), "pinball": _r(np.mean(pin)), "forecasts": n}
    return out


def backtest(frame: pd.DataFrame, *, series: str, time: str, target: str, horizon: int, known_ahead: Sequence[str] = (),
             season: int = 7, origins: int = 4, gap: int = 0, coverage: float = 0.8) -> dict[str, Any]:
    """Rolling-origin backtests: ``origins`` cuts in the last part of the history, each forecasting ``horizon`` steps
    after a ``gap``; every model fitted on what was known at the origin only. Returns the scores per model (MASE, sMAPE,
    and the measured interval coverage and pinball loss), each forecast with its dates and values, the last target date
    each origin used, the intervals, and which covariates were used."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    known = [c for c in known_ahead if c in frame.columns]
    data = frame[[series, time, target, *known]].copy()
    data[time] = pd.to_datetime(data[time])
    data = data.sort_values([series, time])
    groups = {k: g.set_index(time) for k, g in data.groupby(series)}
    dates = sorted(data[time].unique())
    last = len(dates) - horizon - gap
    cuts = [last - i * horizon for i in range(origins)][::-1]
    names = ("naive", "seasonal_naive", "moving_average", "global_gbm", "per_series")
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in names}
    scaled: dict[str, dict[str, dict[int, list[float]]]] = {"global_gbm": {}, "seasonal_naive": {}}  # model → origin → step → errors
    seen_until: dict[str, str] = {}
    for cut in cuts:
        origin = pd.Timestamp(dates[cut - 1])
        key_o = str(origin.date())
        codes = _codes(data[data[time] <= origin], known)
        train_parts, forecasts, read = [], [], []
        for key, g in groups.items():
            past = g[g.index <= origin]  # the target is read from here only
            if len(past) < max(WINDOWS[0], season) + horizon + 2:
                continue
            read.append(past.index.max())
            train_parts.append(_training_rows(past, target=target, known=known, season=season, horizon=horizon, gap=gap, codes=codes))
            future_dates = g.index[g.index > origin][gap: gap + horizon]
            if len(future_dates) < horizon:
                continue
            feats = _features(past, pd.DatetimeIndex(future_dates), target=target, known=known, season=season, gap=gap,
                              plan=g[known], codes=codes)
            forecasts.append((key, past, feats, g.loc[future_dates, target].to_numpy(dtype=float)))
        seen_until[key_o] = str(max(read).date()) if read else key_o
        train = pd.concat([p for p in train_parts if len(p)]) if any(len(p) for p in train_parts) else pd.DataFrame()
        model = None
        if len(train) > 50:
            X = train.drop(columns=[target])
            X = X.loc[:, X.notna().any()]  # a lag the gap hides is empty in every row: scikit-learn 1.9 cannot bin a column with no value
            model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0).fit(X, train[target])
            columns = list(X.columns)
        for key, past, feats, actual in forecasts:
            y = past[target].to_numpy(dtype=float)
            scale = float(np.mean(np.abs(y[season:] - y[:-season]))) if len(y) > season else float("nan")
            preds = {"naive": np.repeat(y[-1], horizon),
                     "seasonal_naive": np.array([y[len(y) - season + (gap + h) % season] for h in range(horizon)]),
                     "moving_average": np.repeat(np.mean(y[-WINDOWS[0]:]), horizon)}
            if model is not None:
                preds["global_gbm"] = model.predict(feats[columns])
            if len(y) >= MIN_CLASSICAL:
                preds["per_series"] = _ses_seasonal(y, season, gap + horizon)[gap:]
            for name, p in preds.items():
                err = actual - p
                results[name].append({"origin": key_o, "series": str(key), "dates": [str(feats.index[0].date()), str(feats.index[-1].date())],
                                      "pred": [round(float(v), 6) for v in p], "mae": float(np.mean(np.abs(err))), "scale": scale,
                                      "smape": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(actual) + np.abs(p), 1e-9)))})
                if name in scaled and scale and np.isfinite(scale):
                    for h, e in enumerate(err, 1):  # in units of the series' own error scale: series of any size pool together
                        scaled[name].setdefault(key_o, {}).setdefault(h, []).append(float(e / scale))
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    pooled = {h: [e for o in scaled["global_gbm"].values() for e in o.get(h, [])] for h in range(1, horizon + 1)}
    intervals = {h: [_r(np.quantile(r, lo_q)), _r(np.quantile(r, hi_q))] for h, r in pooled.items() if len(r) >= 10}
    measured = _interval_check(scaled, str(pd.Timestamp(dates[cuts[-1] - 1]).date()), lo_q, hi_q) if len(cuts) > 1 else {}
    scores = {}
    for name, rows in results.items():
        if not rows:
            continue
        mase = np.nanmean([r["mae"] / r["scale"] for r in rows if r["scale"] and np.isfinite(r["scale"])])
        scores[name] = {"mase": _r(mase), "smape": _r(np.mean([r["smape"] for r in rows])), "mae": _r(np.mean([r["mae"] for r in rows])),
                        "forecasts": len(rows)}
        if name in measured:
            scores[name].update({"pinball": measured[name]["pinball"], "interval_coverage": measured[name]["coverage"]})
    every = scores.get("seasonal_naive", {}).get("forecasts")
    for name in scores:  # a model that skips series (too short for its own model) is not compared on the same forecasts
        scores[name]["covers_every_series"] = every is None or scores[name]["forecasts"] >= every
    best = min((n for n in scores if n != "seasonal_naive" and scores[n]["covers_every_series"]), key=lambda n: scores[n]["mase"] if scores[n]["mase"] is not None else np.inf, default=None)
    return {"horizon": horizon, "gap": gap, "season": season, "origins": [str(pd.Timestamp(dates[c - 1]).date()) for c in cuts],
            "known_ahead": known, "scores": scores, "best": best,
            "beats_seasonal_naive": {n: (s["mase"] is not None and scores.get("seasonal_naive", {}).get("mase") is not None
                                         and s["mase"] < scores["seasonal_naive"]["mase"]) for n, s in scores.items() if n != "seasonal_naive"},
            "intervals": {"coverage": coverage, "by_step": intervals,
                          "units": "the series' own error scale (its seasonal naive's mean absolute error): multiply by a series' scale",
                          "from": "the global model's backtest residuals, scaled per series", "measured": measured},
            "target_seen_until": seen_until, "future_seen": [o for o, d in seen_until.items() if d > o],
            "per_origin": {n: r for n, r in results.items()}}


def holdout_origin(frame: pd.DataFrame, *, time: str, horizon: int, gap: int = 0) -> pd.Timestamp:
    """The forecast moment of the holdout: the last origin, whose ``horizon`` dates after the ``gap`` end the data."""
    dates = sorted(pd.to_datetime(frame[time]).unique())
    if len(dates) < horizon + gap + 2:
        raise ValueError(f"{len(dates)} dates: too few for a horizon of {horizon} after a gap of {gap}")
    return pd.Timestamp(dates[len(dates) - horizon - gap - 1])


def origin_scores(result: dict[str, Any], origin: str, series: Optional[Sequence[str]] = None) -> dict[str, dict[str, Any]]:
    """Each model's scores at one origin of a backtest (MASE, sMAPE, MAE over its series), and its MASE per series;
    with ``series``, over those series only, so models are compared on the same forecasts."""
    out = {}
    for name, rows in result["per_origin"].items():
        mine = [r for r in rows if r["origin"] == origin and (series is None or r["series"] in series)]
        if not mine:
            continue
        ratios = [r["mae"] / r["scale"] for r in mine if r["scale"] and np.isfinite(r["scale"])]
        out[name] = {"mase": _r(np.mean(ratios)) if ratios else None, "smape": _r(np.mean([r["smape"] for r in mine])),
                     "mae": _r(np.mean([r["mae"] for r in mine])), "forecasts": len(mine), "mase_by_series": [_r(x) for x in ratios],
                     "series": sorted(r["series"] for r in mine)}
    return out
