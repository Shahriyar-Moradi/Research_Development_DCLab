"""Decisions on top of a model (package 16.9): a review budget from the value and cost of a review, and a price from a
model of acceptance by price.

A model ranks or predicts; a team acts. The person states the action, what it is worth and what it costs, and DCLab
chooses the operating point that maximises expected value, on out-of-fold predictions of the training rows, with an
interval and the assumption written down. The holdout is read once, after the choice.

* :func:`review_budget`: review the riskiest share of rows. Each case found is worth ``value_per_case``, each review
  costs ``cost_per_review``; the share with the highest expected value on the out-of-fold scores is chosen, and a
  bootstrap over rows (or over entities, when ``groups`` are given) gives an interval for the value and the share.
  Assumption: a review finds the case when it is one, and the review does not change the outcome. With partial labels
  an unlabelled row may be a case nobody found, so the value is a lower bound.
* :func:`price_curve`: offer a price and the customer accepts or not. A monotone model of acceptance by price (logistic
  in the log of the price relative to a reference, plus the row's numeric context) is fitted inside folds; each held-out
  row is scored at every price on a grid within the prices seen; the expected margin per offer, ``(price - unit_cost) *
  P(accept)``, gives the curve and the best price, with a bootstrap interval from refits. Assumption: the prices in the
  data were set independently of anything that also moves acceptance and is not in the data; acceptance responds to
  price as it did then. :func:`price_variation` says when the data cannot teach the response: too few distinct prices,
  too little spread, or prices set by a formula of the other columns. Then the curve is not drawn, and the warning says
  why.

Pure: arrays in, plain data out (numpy, pandas and scikit-learn only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

MIN_PRICES = 5           # distinct prices needed to see a response
MIN_SPREAD = 0.05        # the coefficient of variation of price (relative to the reference) needed
FORMULA_R2 = 0.97        # price explained this well by the other columns: set by a formula, not varied
GRID = 25
BOOTSTRAP = 200


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if np.isfinite(value) else None


def _interval(values: Sequence[float], level: float = 0.9) -> list[Optional[float]]:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return [None, None]
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [_r(lo), _r(hi)]


# ------------------------------------------------------------------------------------------------ review budget


def _budget_curve(y: np.ndarray, score: np.ndarray, value: float, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Expected value of reviewing the top k rows, for every k (k = 0 .. n)."""
    order = np.argsort(-score, kind="mergesort")
    found = np.r_[0, np.cumsum(y[order])]
    reviews = np.arange(len(y) + 1)
    return reviews, value * found - cost * reviews


def review_budget(y_oof: Sequence[int], score_oof: Sequence[float], *, value_per_case: float, cost_per_review: float,
                  groups: Optional[Sequence[Any]] = None, partial: bool = False, rows_per_day: Optional[float] = None,
                  y_test: Optional[Sequence[int]] = None, score_test: Optional[Sequence[float]] = None,
                  bootstrap: int = BOOTSTRAP, seed: int = 0) -> dict[str, Any]:
    """The share of rows to review that maximises expected value, chosen on out-of-fold scores, then read on the holdout."""
    if value_per_case <= 0 or cost_per_review <= 0:
        raise ValueError("A review budget needs a positive value per case found and a positive cost per review")
    y, s = np.asarray(y_oof).astype(int), np.asarray(score_oof, dtype=float)
    n = len(y)
    reviews, ev = _budget_curve(y, s, value_per_case, cost_per_review)
    best = int(np.argmax(ev))
    share = best / n if n else 0.0
    rng = np.random.default_rng(seed)
    if groups is not None:  # entities as codes; a row without one is its own unit (a missing id is no entity)
        codes, _ = pd.factorize(pd.Series(np.asarray(groups, dtype=object)))
        codes = np.where(codes < 0, codes.max(initial=-1) + 1 + np.arange(n), codes)
        order = np.argsort(codes, kind="stable")
        keys, starts = np.unique(codes[order], return_index=True)
        index = dict(zip(keys, np.split(order, starts[1:])))
    else:
        keys, index = np.arange(n), None
    shares, values = [], []
    for _ in range(bootstrap):
        draw = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([index[k] for k in draw]) if index is not None else draw
        r_reviews, r_ev = _budget_curve(y[rows], s[rows], value_per_case, cost_per_review)
        k = int(np.argmax(r_ev))
        shares.append(k / len(rows))
        values.append(r_ev[k] / len(rows))
    out: dict[str, Any] = {
        "decision": "review_budget", "value_per_case": value_per_case, "cost_per_review": cost_per_review,
        "share": _r(share, 6), "reviews_training": best, "threshold": _r(np.sort(s)[::-1][best - 1], 6) if best else None,
        "expected_value_per_row": _r(ev[best] / n if n else None), "expected_value_per_row_interval": _interval(values),
        "share_interval": _interval(shares), "break_even_precision": _r(cost_per_review / value_per_case),
        "chosen_on": "out-of-fold training predictions",
        "interval": f"90% bootstrap over {'entities' if groups is not None else 'rows'}, {bootstrap} draws",
        "curve": [{"share": _r(k / n, 4), "expected_value_per_row": _r(ev[k] / n)} for k in np.unique(np.linspace(0, n, 21).astype(int))],
        "assumption": ("A review finds the case when the row is one, and reviewing does not change the outcome; the value of a "
                       "case and the cost of a review are the person's stated numbers."),
    }
    if best == 0:
        out["note"] = (f"No review pays: at a cost of {cost_per_review:g} per review and {value_per_case:g} per case found, even the "
                       "riskiest rows are not cases often enough (the break-even precision is above what the model reaches).")
    if rows_per_day:
        out["reviews_per_day"] = _r(share * rows_per_day, 1)
    if partial:
        out["partial_note"] = ("The labels are partial: an unlabelled row may be a case nobody found, so the value is a lower bound "
                               "and the best share may be larger.")
    if y_test is not None and score_test is not None and best:
        yt, st = np.asarray(y_test).astype(int), np.asarray(score_test, dtype=float)
        k = max(1, int(round(share * len(yt))))
        top = np.argsort(-st, kind="mergesort")[:k]
        found = int(yt[top].sum())
        out["holdout"] = {"reviews": k, "cases_found": found, "precision": _r(found / k),
                          "expected_value_per_row": _r((value_per_case * found - cost_per_review * k) / len(yt))}
    return out


# ------------------------------------------------------------------------------------------------ prices

MAX_LEVELS = 30          # a text column with at most this many levels is a category of the context
MAX_MISSING_VALUE = 0.05  # unit values missing on more rows than this: the margin cannot be computed


def _context(context: Optional[pd.DataFrame], rows: int) -> pd.DataFrame:
    """The context the price may depend on: numeric columns (not entirely missing) and low-cardinality categories."""
    if context is None or not len(context.columns):
        return pd.DataFrame(index=range(rows))
    keep = {}
    for column in context.columns:
        series = context[column].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(float)
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().any():
                keep[str(column)] = series.astype(float)
        elif 1 < series.nunique(dropna=True) <= MAX_LEVELS:
            keep[str(column)] = series.astype("string").fillna("(missing)")
    return pd.DataFrame(keep, index=range(rows))


def _codes(ctx: pd.DataFrame) -> np.ndarray:
    """Every context column as numbers for the formula check: categories as codes (the check's trees split on them)."""
    columns = [ctx[c].to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(ctx[c]) else pd.factorize(ctx[c])[0].astype(float)
               for c in ctx.columns]
    X = np.column_stack(columns) if columns else np.zeros((len(ctx), 0))
    return np.where(np.isnan(X), np.nanmedian(X, axis=0), X) if X.size else X


def price_variation(price: Sequence[float], context: Optional[pd.DataFrame] = None, reference: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """Whether the data can teach how acceptance responds to price: enough distinct prices, enough spread relative to
    the reference, and prices not set by a formula of the context (numeric columns and categories alike). A logged test
    arm belongs out of ``context``: it explains the price because the test set it."""
    p = np.asarray(price, dtype=float)
    ref = np.asarray(reference, dtype=float) if reference is not None else np.full(len(p), np.nanmedian(p))
    relative = p / np.where(ref > 0, ref, np.nan)
    ok = np.isfinite(relative) & (relative > 0)
    relative = relative[ok]
    distinct = int(pd.Series(np.round(relative, 3)).nunique())
    spread = float(np.std(relative) / np.mean(relative)) if len(relative) and np.mean(relative) else 0.0
    out: dict[str, Any] = {"distinct_relative_prices": distinct, "relative_spread": _r(spread), "learnable": True, "why": []}
    if distinct < MIN_PRICES:
        out["why"].append(f"only {distinct} distinct price levels (relative to the reference): at least {MIN_PRICES} are needed")
    if spread < MIN_SPREAD:
        out["why"].append(f"prices vary by {spread:.1%} around the reference: at least {MIN_SPREAD:.0%} is needed to see a response")
    ctx = _context(context, len(p)).iloc[np.flatnonzero(ok)].reset_index(drop=True)
    if len(ctx.columns) and len(relative) >= 30:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict

        X = _codes(ctx)
        target = np.log(relative)  # the price against its reference: what was varied, if anything
        if np.var(target) > 1e-12:
            fitted = cross_val_predict(HistGradientBoostingRegressor(max_iter=150, random_state=0), X, target, cv=3)
            r2 = 1 - float(np.var(target - fitted) / np.var(target))
            out["explained_by_context_r2"] = _r(r2)
            if r2 >= FORMULA_R2:
                strength = {c: abs(float(pd.Series(X[:, i]).corr(pd.Series(target), method="spearman") or 0)) for i, c in enumerate(ctx.columns)}
                named = ", ".join(sorted(strength, key=lambda c: -strength[c])[:3])
                out["explains_the_price"] = sorted(strength, key=lambda c: -strength[c])[:3]
                out["why"].append(f"the other columns explain {r2:.1%} of the price against its reference (most: {named}): it was set by "
                                  "a formula of them, never varied on its own, so its effect cannot be told apart from theirs; if one of "
                                  "them is the arm of a price test, name it as such and it is left out of this check")
    out["learnable"] = not out["why"]
    return out


def _frame(price: np.ndarray, reference: np.ndarray, ctx: pd.DataFrame) -> pd.DataFrame:
    return ctx.assign(__log_relative_price=np.log(price / reference))


def _model(ctx: pd.DataFrame):
    """Acceptance by price: logistic in the log relative price and the context; the scaler, the imputer and the
    categories are all fitted inside the rows it is fitted on."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = ["__log_relative_price"] + [c for c in ctx.columns if pd.api.types.is_numeric_dtype(ctx[c])]
    categories = [c for c in ctx.columns if c not in numeric]
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric)]
    if categories:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categories))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=1.0, max_iter=2000))


def price_curve(price: Sequence[float], accepted: Sequence[int], *, reference: Optional[Sequence[float]] = None,
                context: Optional[pd.DataFrame] = None, unit_cost: float | Sequence[float] = 0.0,
                unit_value: Optional[float | Sequence[float]] = None, groups: Optional[Sequence[Any]] = None, folds: int = 5,
                grid: int = GRID, bootstrap: int = BOOTSTRAP, seed: int = 0) -> dict[str, Any]:
    """The expected margin per offer at each price multiple of the reference, from out-of-fold acceptance predictions;
    the best multiple with a bootstrap interval; refused with the reasons when prices were not varied enough.

    Selling (the default): the margin of an accepted offer is ``price - unit_cost``. Paying (``unit_value`` given, as a
    fee offered to a courier or a supplier): it is ``unit_value - price``, and acceptance rises with the price.
    ``groups``: the entity of each row (a courier who sees many offers): folds and the bootstrap keep its rows together.
    Rows without a usable price, reference or unit value are left out, and counted."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    p = np.asarray(price, dtype=float)
    y = np.asarray(accepted).astype(int)
    n = len(p)
    ref = np.asarray(reference, dtype=float) if reference is not None else np.full(n, float(np.nanmedian(p)))
    paying = unit_value is not None
    cost = np.broadcast_to(np.asarray(unit_value if paying else unit_cost, dtype=float), p.shape).astype(float)
    assumption = ("Acceptance responds to price as it did in the data, and the prices in the data were set independently of anything "
                  "that also moves acceptance and is not in the data (no hidden confounder); the curve is read only within the prices seen.")
    usable = np.isfinite(p) & (p > 0) & np.isfinite(ref) & (ref > 0)
    missing_value = float(np.mean(~np.isfinite(cost[usable]))) if usable.any() else 1.0
    if missing_value > MAX_MISSING_VALUE:
        return {"decision": "price", "learnable": False, "assumption": assumption,
                "warning": f"The value of an accepted offer is missing on {missing_value:.0%} of the rows: the margin cannot be computed "
                           "(a value filled only for accepted offers would bias it). Give a value for every offer."}
    usable &= np.isfinite(cost)
    dropped = int(n - usable.sum())
    keep = np.flatnonzero(usable)
    p, y, ref, cost = p[keep], y[keep], ref[keep], cost[keep]
    ctx = _context(context, n).iloc[keep].reset_index(drop=True)
    units = None
    if groups is not None:
        codes, _ = pd.factorize(pd.Series(np.asarray(groups, dtype=object)[keep]))
        units = np.where(codes < 0, codes.max(initial=-1) + 1 + np.arange(len(codes)), codes)
    variation = price_variation(p, ctx, ref)
    if not variation["learnable"]:
        return {"decision": "price", "learnable": False, "variation": variation, "assumption": assumption, "rows_left_out": dropped,
                "warning": "Cannot learn how acceptance responds to price: " + "; ".join(variation["why"]) + ". Vary prices on purpose "
                           "(a randomised test of a few price levels) before choosing one from data."}
    if len(np.unique(y)) < 2:
        return {"decision": "price", "learnable": False, "variation": variation, "assumption": assumption,
                "warning": "Every offer in the training rows was accepted, or none was: there is no response to learn."}
    sign = -1.0 if paying else 1.0  # the margin is sign * (price - cost)
    relative = p / ref
    lo, hi = np.quantile(relative, [0.05, 0.95])
    multiples = np.linspace(lo, hi, grid)
    accept_at = np.zeros((len(p), grid))
    splitter = (StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(p, y, units) if units is not None
                else StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(p, y))
    for train, test in splitter:
        model = _model(ctx).fit(_frame(p[train], ref[train], ctx.iloc[train]), y[train])
        for j, m in enumerate(multiples):
            accept_at[test, j] = model.predict_proba(_frame(ref[test] * m, ref[test], ctx.iloc[test]))[:, 1]
    margin = sign * (ref[:, None] * multiples[None, :] - cost[:, None]) * accept_at
    curve = margin.mean(axis=0)
    best = int(np.argmax(curve))
    rng = np.random.default_rng(seed)
    if units is not None:
        order = np.argsort(units, kind="stable")
        keys, starts = np.unique(units[order], return_index=True)
        members = np.split(order, starts[1:])
    best_multiples, best_margins = [], []
    for _ in range(bootstrap):
        if units is not None:
            rows = np.concatenate([members[k] for k in rng.integers(0, len(keys), len(keys))])
        else:
            rows = rng.integers(0, len(p), len(p))
        if len(np.unique(y[rows])) < 2:
            continue
        model = _model(ctx).fit(_frame(p[rows], ref[rows], ctx.iloc[rows]), y[rows])
        values = [float(np.mean(sign * (ref[rows] * m - cost[rows]) * model.predict_proba(_frame(ref[rows] * m, ref[rows], ctx.iloc[rows]))[:, 1]))
                  for m in multiples]
        k = int(np.argmax(values))
        best_multiples.append(multiples[k])
        best_margins.append(values[k])
    current = float(np.mean(sign * (p - cost) * y))
    out = {"decision": "price", "side": "pay" if paying else "sell", "learnable": True, "variation": variation, "assumption": assumption,
           "best_multiple": _r(multiples[best]), "best_multiple_interval": _interval(best_multiples),
           "expected_margin_per_offer": _r(curve[best]), "expected_margin_interval": _interval(best_margins),
           "margin_per_offer_as_priced": _r(current), "acceptance_at_best": _r(accept_at[:, best].mean()), "rows_left_out": dropped,
           "elasticity_note": "the multiple is applied to the reference price of each row (its usual price), within the 5th-95th "
                              "percentile of the multiples seen",
           "chosen_on": "out-of-fold acceptance predictions of the training rows",
           "interval": f"90% bootstrap over {'entities' if units is not None else 'rows'}, {len(best_multiples)} refits",
           "curve": [{"multiple": _r(m), "acceptance": _r(accept_at[:, j].mean()), "expected_margin_per_offer": _r(curve[j])}
                     for j, m in enumerate(multiples)]}
    if best in (0, grid - 1):
        out["note"] = ("The best price is at the edge of the prices seen: the true best may lie outside them, where the data says "
                       "nothing; test prices beyond it before moving there.")
    return out
