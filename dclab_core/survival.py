"""Time to event (package 16.4): how long until something happens, when for some rows it has not happened yet.

A row has a start moment, a duration (until the event, or until the data ends when it has not happened: right-censored)
and an event flag. A censored row is not a row without an outcome: it says the event did not happen during its
follow-up. Dropping censored rows keeps the short durations and forgets the long ones; treating them as events says
they all ended. Both bias every estimate towards shorter lives; :func:`censoring_bias` measures by how much on the
curve, and :func:`evaluate` on the chosen model.

- :func:`kaplan_meier`: the survival curve without covariates (also per group); a value past the longest duration
  followed is not known, and is returned as None.
- :func:`evaluate`: rows split by start day (the latest starts held out, read once). Training rows are censored at the
  holdout's first day, and the rows a model is fitted on at the validation cut, so no outcome dated after a cut is read
  before it. A model is chosen on the validation rows: Kaplan-Meier (the baseline), Cox proportional hazards
  (ridge-penalised, Breslow ties and baseline hazard), and discrete-time hazard models on person-period rows (logistic
  and gradient-boosted), whose survival inside an interval assumes a constant hazard there.
- Scores: Harrell's C-index, the integrated Brier score with inverse-probability-of-censoring weights taken from the
  scored rows' own censoring (no model reads it), and Brier and time-dependent AUC (cumulative cases, dynamic
  controls) at named horizons the scored rows' follow-up reaches.
- :func:`duration_check`: columns measured past the start moment. A column counted during the follow-up moves with the
  duration both among rows still going (where the duration is only the time the data has followed them) and among rows
  that ended; a column known at the start that merely trends with the calendar does not do both. A column filled, or
  given a value, only when the event happens is written at the end.

Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

TRACKS = 0.5  # |Spearman| with the duration, among rows still going and among rows that ended, of a column counted during the follow-up
BINS = 12  # discrete-time intervals, at quantiles of the training rows' event times
MAX_COX_CELLS = 50_000_000  # rows x features x features the Cox fit holds at once
PAIR_BLOCK = 2000  # rows per block of the C-index's pairs


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


# ---------------------------------------------------------------------------------------------------------- curves
def kaplan_meier(duration: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, survival just after each time): the product-limit estimate, censored rows at risk until they leave."""
    d, e = np.asarray(duration, dtype=float), np.asarray(event, dtype=int)
    times = np.unique(d[e == 1])
    surv, s = [], 1.0
    for t in times:
        at_risk = np.sum(d >= t)
        events = np.sum((d == t) & (e == 1))
        s *= 1 - events / at_risk if at_risk else 1.0
        surv.append(s)
    return times, np.array(surv)


def step(times: np.ndarray, values: np.ndarray, at: Any, before: bool = False) -> np.ndarray:
    """A right-continuous step function (1 before the first time) read at ``at``; ``before``: its left limit."""
    at = np.atleast_1d(np.asarray(at, dtype=float))
    if not len(values):
        return np.ones(len(at))
    idx = np.searchsorted(times, at, side="left" if before else "right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], 1.0)


def _km_at(duration: np.ndarray, event: np.ndarray, horizons: Sequence[float]) -> dict[str, Optional[float]]:
    """Kaplan-Meier at each horizon; None past the longest duration followed (nothing is known there)."""
    d = np.asarray(duration, dtype=float)
    t, s = kaplan_meier(d, event)
    longest = d.max() if len(d) else 0.0
    return {str(int(h)): (_r(step(t, s, [h])[0], 3) if h <= longest else None) for h in horizons}


def curves_by_group(frame: pd.DataFrame, duration: str, event: str, group: str, at: Sequence[float]) -> dict[str, Any]:
    """Kaplan-Meier survival per group at the times ``at`` (the curves the final record carries)."""
    return {str(key): {"rows": int(len(g)), "events": int(g[event].sum()), "survival": _km_at(g[duration].to_numpy(), g[event].to_numpy(), at)}
            for key, g in frame.groupby(group)}


# ---------------------------------------------------------------------------------------------------------- checks
def censoring_bias(duration: np.ndarray, event: np.ndarray, horizons: Sequence[float]) -> dict[str, Any]:
    """Survival at each horizon three ways: honest (Kaplan-Meier), censored rows dropped, censored rows as events."""
    d, e = np.asarray(duration, dtype=float), np.asarray(event, dtype=int)
    honest, dropped, as_events = _km_at(d, e, horizons), _km_at(d[e == 1], e[e == 1], horizons), _km_at(d, np.ones_like(e), horizons)
    return {"censored_share": _r(1 - e.mean(), 3) if len(e) else None,
            "survival": {h: {"kaplan_meier": honest[h], "censored_dropped": dropped[h], "censored_as_events": as_events[h]} for h in honest}}


def censoring_check(event: pd.Series) -> dict[str, Any]:
    """Whether the event flag can carry censoring: two values, and some rows not (yet) ended."""
    values = pd.Series(event).dropna().unique()
    if len(values) < 2:
        return {"ok": False, "why": "the event flag has one value: either every row ended, or rows still going were marked as ended "
                                    "or left out of the export; either way the durations cannot be read honestly"}
    share = float(pd.Series(event).astype(float).mean())
    return {"ok": True, "censored_share": _r(1 - share, 3), "why": f"{1 - share:.0%} of rows have not ended: they are censored, not missing"}


def duration_check(frame: pd.DataFrame, *, duration: str, event: str, exclude: Sequence[str] = ()) -> dict[str, Any]:
    """Columns measured past the start moment. Numbers: they move with the duration (|Spearman| >= 0.5, same sign) both
    among rows still going and among rows that ended. Any column: filled, or given a value, only when the event happens."""
    ended = frame[event].astype(int) == 1
    going, done = frame[~ended], frame[ended]
    out: dict[str, Any] = {"past_the_start": [], "why": {}}

    def flag(column: str, why: str) -> None:
        out["past_the_start"].append(column)
        out["why"][column] = why

    for column in [c for c in frame.columns if c not in (duration, event, *exclude)]:
        filled_going, filled_done = going[column].notna().mean() if len(going) else None, done[column].notna().mean() if len(done) else None
        if filled_going is not None and filled_done is not None and filled_done >= 0.5 and filled_going <= 0.05:
            flag(column, f"filled for {filled_done:.0%} of rows that ended and {filled_going:.0%} of rows still going: it is written when the event happens")
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            if len(going) >= 30 and len(done) >= 30:
                only = set(done[column].dropna().astype(str)) - set(going[column].dropna().astype(str))
                share = done[column].astype(str).isin(only).mean() if only else 0.0
                if share >= 0.5:
                    flag(column, f"{share:.0%} of rows that ended hold a value no row still going has: it is written when the event happens")
            continue
        rhos = {}
        for name, part in (("still going", going), ("ended", done)):
            if len(part) >= 30 and part[column].nunique() > 1:
                rhos[name] = float(part[column].rank().corr(part[duration].rank()))
        both = len(rhos) == 2 and all(abs(r) >= TRACKS for r in rhos.values()) and np.sign(rhos["still going"]) == np.sign(rhos["ended"])
        alone = len(rhos) == 1 and abs(next(iter(rhos.values()))) >= 0.8  # one kind of row only: only a near copy is flagged
        if both or alone:
            flag(column, "it moves with the duration (Spearman " + ", ".join(f"{v:.2f} among rows {k}" for k, v in rhos.items())
                 + "): it is counted during the follow-up, not known at the start")
    return out


# ---------------------------------------------------------------------------------------------------------- models
def _design(train: pd.DataFrame, frames: Sequence[pd.DataFrame], features: Sequence[str]) -> list[np.ndarray]:
    """Numbers for the models: categories one-hot and numbers standardised, both fitted on ``train`` only."""
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    nums = [c for c in features if c not in cats]
    levels = {c: sorted(train[c].astype(str).unique())[1:] for c in cats}  # the first level is the reference
    mean, std = train[nums].mean(), train[nums].std().replace(0, 1).fillna(1)

    def one(f):
        parts = [((f[nums] - mean) / std).fillna(0).to_numpy(dtype=float)] if nums else []
        for c in cats:
            parts.append(np.column_stack([(f[c].astype(str) == v).to_numpy(dtype=float) for v in levels[c]]) if levels[c] else np.zeros((len(f), 0)))
        return np.hstack(parts) if parts else np.zeros((len(f), 0))
    return [one(f) for f in frames]


class Cox:
    """Cox proportional hazards, Breslow ties, a small ridge penalty, Newton-Raphson."""

    def __init__(self, ridge: float = 0.1, steps: int = 30):
        self.ridge, self.steps = ridge, steps

    def fit(self, X: np.ndarray, d: np.ndarray, e: np.ndarray) -> "Cox":
        if X.shape[0] * X.shape[1] ** 2 > MAX_COX_CELLS:
            raise ValueError(f"{X.shape[0]} rows by {X.shape[1]} features is too large for this Cox fit: use fewer features")
        order = np.argsort(-d, kind="stable")  # longest first: the risk set of a row is every row before it, ties included
        X, d, e = X[order], d[order], e[order]
        beta = np.zeros(X.shape[1])
        last = np.searchsorted(-d, -d, side="right") - 1  # ties share the risk set up to the last row with the same time
        ev = e == 1
        for _ in range(self.steps):
            w = np.exp(np.clip(X @ beta, -30, 30))
            cw = np.cumsum(w)
            cwx = np.cumsum(w[:, None] * X, axis=0)
            cwxx = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
            S0, S1, S2 = cw[last][ev], cwx[last][ev], cwxx[last][ev]
            mean = S1 / S0[:, None]
            grad = (X[ev] - mean).sum(axis=0) - self.ridge * beta
            hess = -(S2 / S0[:, None, None] - mean[:, :, None] * mean[:, None, :]).sum(axis=0) - self.ridge * np.eye(len(beta))
            delta = np.linalg.solve(hess, grad)
            beta = beta - delta
            if np.max(np.abs(delta)) < 1e-6:
                break
        self.beta = beta
        w = np.exp(X @ beta)
        times = np.unique(d[ev])
        self.times = times
        self.cumhaz = np.cumsum(np.array([np.sum((d == t) & ev) / w[d >= t].sum() for t in times]))
        return self

    def risk(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def survival(self, X: np.ndarray, at: np.ndarray) -> np.ndarray:
        at = np.asarray(at, dtype=float)
        H = np.where(at < self.times[0], 0.0, step(self.times, self.cumhaz, at)) if len(self.times) else np.zeros(len(at))
        return np.exp(-np.exp(self.risk(X))[:, None] * H[None, :])


class DiscreteTime:
    """The hazard in each interval as a classification on person-period rows (one row per interval a person was at
    risk in): logistic regression or gradient boosting. Inside an interval the hazard is taken as constant."""

    def __init__(self, boosted: bool = False, bins: int = BINS):
        self.boosted, self.bins = boosted, bins

    def _periods(self, X, d, e):
        k = np.searchsorted(self.edges, d, side="left")  # the interval each row ends in
        rows, ys = [], []
        for j in range(len(self.edges)):
            alive = k >= j
            if not alive.any():
                continue
            rows.append(np.column_stack([X[alive], np.full(alive.sum(), j)]))
            ys.append(((k[alive] == j) & (e[alive] == 1)).astype(int))
        return np.vstack(rows), np.concatenate(ys)

    def fit(self, X: np.ndarray, d: np.ndarray, e: np.ndarray) -> "DiscreteTime":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        if not (e == 1).any():
            raise ValueError("no row ended in the rows this model is fitted on: nothing to learn a hazard from")
        self.edges = np.unique(np.quantile(d[e == 1], np.linspace(1 / self.bins, 1, self.bins)))
        Z, y = self._periods(X, d, e)
        if len(np.unique(y)) < 2:
            raise ValueError("every interval ended the same way: a hazard model cannot be fitted")
        if self.boosted:
            self.model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_leaf_nodes=15, categorical_features=[Z.shape[1] - 1],
                                                        random_state=0).fit(Z, y)
        else:
            self.model = LogisticRegression(max_iter=2000, C=1.0).fit(self._onehot(Z), y)
        return self

    def _onehot(self, Z: np.ndarray) -> np.ndarray:
        return np.hstack([Z[:, :-1], np.eye(len(self.edges))[Z[:, -1].astype(int)]])

    def hazards(self, X: np.ndarray) -> np.ndarray:
        n, J = len(X), len(self.edges)
        Z = np.column_stack([np.repeat(X, J, axis=0), np.tile(np.arange(J), n)])
        return self.model.predict_proba(Z if self.boosted else self._onehot(Z))[:, 1].reshape(n, J)

    def survival(self, X: np.ndarray, at: np.ndarray) -> np.ndarray:
        return read_discrete(self.hazards(X), self.edges, at)

    def risk(self, X: np.ndarray) -> np.ndarray:
        return -np.log(np.clip(np.cumprod(1 - self.hazards(X), axis=1)[:, -1], 1e-12, 1))  # cumulative hazard over the span


def read_discrete(h: np.ndarray, edges: np.ndarray, at: Any) -> np.ndarray:
    """Survival at ``at`` from per-interval hazards ``h`` (rows x intervals ending at ``edges``), the hazard constant
    inside an interval: S(t) = S(edge before t) * (1 - h_j) ** (share of interval j elapsed at t). Past the last edge the
    last interval's hazard goes on."""
    at = np.atleast_1d(np.asarray(at, dtype=float))
    S = np.cumprod(1 - h, axis=1)
    lower = np.r_[0.0, edges[:-1]]
    width = np.maximum(edges - lower, 1e-12)
    out = np.empty((h.shape[0], len(at)))
    for n, t in enumerate(at):
        j = min(int(np.searchsorted(edges, t, side="left")), len(edges) - 1)  # the interval t falls in (the last one past the end)
        before = S[:, j - 1] if j > 0 else np.ones(h.shape[0])
        out[:, n] = before * (1 - h[:, j]) ** (max(t - lower[j], 0.0) / width[j])
    return out


class KaplanMeier:
    """The baseline: the same curve for everyone."""

    def fit(self, X, d, e):
        self.times, self.surv = kaplan_meier(d, e)
        return self

    def survival(self, X, at):
        return np.tile(step(self.times, self.surv, at), (len(X), 1))

    def risk(self, X):
        return np.zeros(len(X))


MODELS: dict[str, Callable[[], Any]] = {"kaplan_meier": KaplanMeier, "cox": Cox, "discrete_logistic": lambda: DiscreteTime(False),
                                        "discrete_boosted": lambda: DiscreteTime(True)}


# ---------------------------------------------------------------------------------------------------------- scores
def c_index(d: np.ndarray, e: np.ndarray, risk: np.ndarray) -> Optional[float]:
    """Harrell's: of the pairs where the shorter duration is an event, the share where it has the higher risk (ties in
    risk count a half). Pairs are counted in blocks, so memory stays bounded."""
    d, e, risk = np.asarray(d, float), np.asarray(e, int), np.asarray(risk, float)
    usable = conc = ties = 0.0
    for lo in range(0, len(d), PAIR_BLOCK):
        di, ei, ri = d[lo:lo + PAIR_BLOCK, None], e[lo:lo + PAIR_BLOCK, None], risk[lo:lo + PAIR_BLOCK, None]
        pair = (di < d[None, :]) & (ei == 1)
        usable += pair.sum()
        conc += (pair & (ri > risk[None, :])).sum()
        ties += (pair & (ri == risk[None, :])).sum()
    return float((conc + 0.5 * ties) / usable) if usable else None


def censoring_weights(d: np.ndarray, e: np.ndarray) -> Callable[..., np.ndarray]:
    """G, the probability of still being followed: Kaplan-Meier of the censoring, from the scored rows themselves
    (their follow-up is what decides who is censored when). Capped at 0.05 so no row counts more than 20 times."""
    t, s = kaplan_meier(d, 1 - np.asarray(e, int))
    return lambda at, before=False: np.clip(step(t, s, at, before=before), 0.05, 1)


def brier(d, e, surv_at_t: np.ndarray, t: float, G) -> float:
    """The weighted Brier score at ``t``: events by ``t`` weighted by 1/G(just before their time), rows still followed
    after ``t`` by 1/G(t); rows censored before ``t`` count through the weights only."""
    died = (d <= t) & (e == 1)
    alive = d > t
    return float(np.mean(np.where(died, surv_at_t ** 2 / G(d, before=True), 0) + np.where(alive, (1 - surv_at_t) ** 2 / G([t])[0], 0)))


def integrated_brier(d, e, S: np.ndarray, grid: np.ndarray, G) -> float:
    """The Brier score averaged over ``grid`` (trapezoids, divided by the span)."""
    scores = np.array([brier(d, e, S[:, i], t, G) for i, t in enumerate(grid)])
    return float(_trapz(scores, grid) / (grid[-1] - grid[0])) if len(grid) > 1 else float(scores[0])


def auc_at(d, e, risk, t: float, G) -> Optional[float]:
    """Cumulative/dynamic AUC at ``t``: rows whose event came by ``t`` (weighted by 1/G) against rows still followed after it."""
    cases = (d <= t) & (e == 1)
    controls = d > t
    if not cases.any() or not controls.any():
        return None
    w = 1 / G(d[cases], before=True)
    rc, rn = risk[cases], risk[controls]
    hit = (rc[:, None] > rn[None, :]).mean(axis=1) + 0.5 * (rc[:, None] == rn[None, :]).mean(axis=1)
    return float(np.sum(w * hit) / np.sum(w))


# ---------------------------------------------------------------------------------------------------------- the run
def censor_at(start_day: np.ndarray, d: np.ndarray, e: np.ndarray, cut_day: float) -> tuple[np.ndarray, np.ndarray]:
    """The rows as they were known on ``cut_day``: a duration running past it stops there, and an event after it has not happened."""
    known = np.maximum(cut_day - start_day, 0.0)
    return np.minimum(d, known), np.where(d <= known, e, 0)


def split(frame: pd.DataFrame, *, duration: str, event: str, start: str, holdout: float = 0.25, validation: float = 0.2,
          days_per_unit: float = 1.0) -> dict[str, Any]:
    """Rows ordered by their start day: the latest ``holdout`` share held out, training rows censored at its first day, and
    a validation cut inside them (the rows a model is fitted on censored at it). No outcome dated after a cut is read
    before it. Returns the parts (frames with ``_day`` and ``_start``) and their description."""
    data = frame.copy()
    stamps = pd.to_datetime(data[start], errors="coerce")
    if stamps.isna().any():
        raise ValueError(f"{int(stamps.isna().sum())} values of {start!r} are not dates: the rows cannot be ordered in time")
    data["_day"] = stamps.dt.normalize()
    data[event] = data[event].astype(int)
    data = data.sort_values("_day", kind="stable").reset_index(drop=True)
    origin = data["_day"].min()
    data["_start"] = (data["_day"] - origin).dt.days.to_numpy(float) / days_per_unit  # in the duration's unit

    def boundary(rows: pd.DataFrame, share: float) -> int:  # never cut inside one start day: it is all before, or all after
        at = int(len(rows) * share)
        while 0 < at < len(rows) and rows["_day"].iloc[at] == rows["_day"].iloc[at - 1]:
            at += 1
        return at

    cut = boundary(data, 1 - holdout)
    train, test = data.iloc[:cut].copy(), data.iloc[cut:].copy()
    if not len(train) or not len(test):
        raise ValueError("every row starts on the same day, or too few days: there is no later start to hold out")
    holdout_day = float(test["_start"].iloc[0])
    train[duration], train[event] = censor_at(train["_start"].to_numpy(), train[duration].to_numpy(float), train[event].to_numpy(int), holdout_day)
    inner = boundary(train, 1 - validation)
    fit_part, val_part = train.iloc[:inner].copy(), train.iloc[inner:].copy()
    if not len(fit_part) or not len(val_part):
        raise ValueError("too few start days to keep a validation cut inside the training rows")
    val_day = float(val_part["_start"].iloc[0])
    fit_part[duration], fit_part[event] = censor_at(fit_part["_start"].to_numpy(), fit_part[duration].to_numpy(float), fit_part[event].to_numpy(int), val_day)
    for name, rows in (("training", train), ("fitting", fit_part)):
        if not rows[event].any():
            raise ValueError(f"no row of the {name} rows ended before its cut: there is nothing to learn when it happens")
    return {"train": train, "test": test, "fit": fit_part, "val": val_part,
            "split": {"by": start, "train": int(len(train)), "holdout": int(len(test)), "holdout_from": str(test["_day"].iloc[0].date()),
                      "train_until": str(train["_day"].iloc[-1].date()), "validation_from": str(val_part["_day"].iloc[0].date()),
                      "validation_rows": int(len(val_part)), "censored_at_cut": True}}


def _score(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame, *, duration: str, event: str, features: Sequence[str], horizons: Sequence[float]):
    Xf, Xe = _design(fit_rows, [fit_rows, eval_rows], features)
    df, ef = fit_rows[duration].to_numpy(float), fit_rows[event].to_numpy(int)
    de, ee = eval_rows[duration].to_numpy(float), eval_rows[event].to_numpy(int)
    G = censoring_weights(de, ee)
    reach = [h for h in horizons if h < np.quantile(de, 0.9)]  # a horizon the follow-up of the scored rows reaches
    grid = np.linspace(max(np.quantile(de[ee == 1], 0.05), 1.0), np.quantile(de, 0.8), 20) if (ee == 1).sum() > 5 else np.array([])
    out = {}
    for name, make in MODELS.items():
        model = make().fit(Xf, df, ef)
        risk = model.risk(Xe)
        row = {"c_index": _r(c_index(de, ee, risk) if name != "kaplan_meier" else 0.5, 4)}
        if len(grid) > 1:
            row["integrated_brier"] = _r(integrated_brier(de, ee, model.survival(Xe, grid), grid, G))
        if reach:
            S = model.survival(Xe, np.array(reach, dtype=float))
            row["auc"] = {str(int(h)): _r(auc_at(de, ee, risk, h, G) if name != "kaplan_meier" else 0.5, 4) for h in reach}
            row["brier"] = {str(int(h)): _r(brier(de, ee, S[:, i], h, G)) for i, h in enumerate(reach)}
            row["mean_survival"] = {str(int(h)): _r(S[:, i].mean(), 3) for i, h in enumerate(reach)}
        out[name] = row
    return out, reach, (Xf, df, ef, Xe, de, ee)


def choose(parts: dict[str, Any], *, duration: str, event: str, features: Sequence[str], horizons: Sequence[float]) -> dict[str, Any]:
    """Every model fitted on the first training rows and scored on the validation rows (never the holdout); the chosen
    one has the lowest integrated Brier score, Kaplan-Meier only when nothing else can be scored."""
    on_validation, _, _ = _score(parts["fit"], parts["val"], duration=duration, event=event, features=features, horizons=horizons)
    best = min((n for n in on_validation if n != "kaplan_meier" and on_validation[n].get("integrated_brier") is not None),
               key=lambda n: on_validation[n]["integrated_brier"], default="kaplan_meier")
    return {"validation": on_validation, "chosen": best,
            "chosen_by": "the integrated Brier score on the latest training rows (validation), before the holdout"}


def score(parts: dict[str, Any], chosen: str, *, duration: str, event: str, features: Sequence[str], horizons: Sequence[float],
          group: Optional[str] = None) -> dict[str, Any]:
    """The holdout, read once: every model fitted on the training rows and scored on the latest starts; the chosen model
    also fitted with censored rows dropped and counted as events, against what the held-out rows show."""
    final, reach, (Xf, df, ef, Xe, de, ee) = _score(parts["train"], parts["test"], duration=duration, event=event, features=features, horizons=horizons)
    shortcut_horizons = [h for h in horizons if h <= np.quantile(de, 0.95)]
    observed = _km_at(de, ee, shortcut_horizons)  # what the held-out rows show, honestly
    shortcuts: dict[str, Any] = {}
    for label, (X_, d_, e_) in (("honest", (Xf, df, ef)), ("censored_dropped", (Xf[ef == 1], df[ef == 1], ef[ef == 1])),
                                ("censored_as_events", (Xf, df, np.ones_like(ef)))):
        S = MODELS[chosen]().fit(X_, d_, e_).survival(Xe, np.array(shortcut_horizons, dtype=float)) if shortcut_horizons else np.zeros((len(Xe), 0))
        shortcuts[label] = {str(int(h)): _r(S[:, i].mean(), 3) for i, h in enumerate(shortcut_horizons)}
    km = final["kaplan_meier"]
    return {"holdout": final, "horizons_reached": [int(h) for h in reach],
            "beats_kaplan_meier": {n: (s.get("integrated_brier") is not None and km.get("integrated_brier") is not None
                                       and s["integrated_brier"] < km["integrated_brier"]) for n, s in final.items() if n != "kaplan_meier"},
            "shortcuts": {"model": chosen, "observed_on_holdout": observed, "mean_survival": shortcuts},
            "censoring": censoring_bias(df, ef, horizons),
            "curves": curves_by_group(parts["train"].assign(**{duration: df, event: ef}), duration, event, group, horizons) if group else {}}


def evaluate(frame: pd.DataFrame, *, duration: str, event: str, start: str, features: Sequence[str], horizons: Sequence[float],
             holdout: float = 0.25, validation: float = 0.2, group: Optional[str] = None, days_per_unit: float = 1.0) -> dict[str, Any]:
    """:func:`split`, :func:`choose` on validation, then :func:`score` on the holdout once."""
    parts = split(frame, duration=duration, event=event, start=start, holdout=holdout, validation=validation, days_per_unit=days_per_unit)
    picked = choose(parts, duration=duration, event=event, features=features, horizons=horizons)
    scored = score(parts, picked["chosen"], duration=duration, event=event, features=features, horizons=horizons, group=group)
    return {"split": parts["split"], **picked, **scored}


_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed it
