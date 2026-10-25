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

import re
from collections.abc import Sequence
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from .contracts import CONTRACTS, Versioned, _version

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


# ------------------------------------------------------------------------------------------------ one decision core (34.1)


class DecisionSpec(Versioned):
    """What acting on a row is worth and costs (34.1, v1): the one statement every operating point is chosen from.

    Acting on a row that is a case earns ``value_per_case`` (or the row's own value, given beside the scores) and avoids
    ``cost_false_negative``, the cost of missing it; acting on a row that is not a case costs ``cost_false_positive``;
    every action costs ``cost_per_action`` (a review, a call, a coupon). ``capacity_share`` or ``capacity_per_day``
    caps how many rows may be acted on. The review budget (16.9) is value and an action's cost; the stated-cost
    threshold (17.5) is a false alarm's and a miss's cost: both are special cases, with the same answers."""

    schema_version: Literal["dclab_core.decision_spec.v1"] = _version("decision_spec")
    action: str = Field(min_length=1, max_length=80)
    value_per_case: float = Field(default=0.0, ge=0)
    cost_per_action: float = Field(default=0.0, ge=0)
    cost_false_positive: float = Field(default=0.0, ge=0)
    cost_false_negative: float = Field(default=0.0, ge=0)
    value_column: Optional[str] = Field(default=None, max_length=200)  # a row's own value of a case (an order's amount)
    capacity_share: Optional[float] = Field(default=None, gt=0, lt=1)
    capacity_per_day: Optional[float] = Field(default=None, gt=0)
    notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _worth_deciding(self):
        if not (self.value_per_case or self.cost_false_negative or self.value_column):
            raise ValueError("A decision needs what acting on a case is worth: a value per case, a row's value column, or a miss's cost")
        return self


CONTRACTS["decision_spec"] = DecisionSpec


def _gain(y: np.ndarray, value: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    """What acting on each row earns against not acting on it, given whether it is a case."""
    return np.where(y == 1, value + spec.cost_false_negative, -spec.cost_false_positive) - spec.cost_per_action


def _prediction(values: Sequence[float], n_train: int, n_test: int, level: float = 0.9) -> list[Optional[float]]:
    """Where one new sample of ``n_test`` rows' value per row should fall: the expected value's bootstrap spread widened
    by the new sample's own noise (normal approximation, variance scaled by 1 + n_train / n_test)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 2 or not n_test:
        return [None, None]
    z = {0.9: 1.6449, 0.95: 1.96}.get(level, 1.6449)
    half = z * float(v.std(ddof=1)) * np.sqrt(1 + n_train / n_test)
    return [_r(v.mean() - half), _r(v.mean() + half)]


def choose_operating_point(y_oof: Sequence[int], score_oof: Sequence[float], spec: DecisionSpec, *,
                           value_per_case: Optional[Sequence[float]] = None, groups: Optional[Sequence[Any]] = None,
                           rows_per_day: Optional[float] = None, y_test: Optional[Sequence[int]] = None,
                           score_test: Optional[Sequence[float]] = None, value_test: Optional[Sequence[float]] = None,
                           weights: Optional[Sequence[float]] = None, bootstrap: int = BOOTSTRAP, seed: int = 0,
                           folds: Optional[Sequence[Any]] = None) -> dict[str, Any]:
    """The number of rows to act on with the highest expected value, chosen on out-of-fold scores under the capacity,
    with its interval (a bootstrap over rows, or entities when ``groups`` are given), then read once on the holdout.

    The holdout's prediction interval (0.114.0, fraud_decision_v2 measured the normal approximation too narrow by about a
    quarter on an entity split): in each bootstrap draw, a sample of the holdout's number of rows is drawn from the draw
    (whole entities when ``groups`` are given, until the rows are reached) and the decision applied to it as it runs; the
    interval is those values' 5th and 95th percentiles: the training rows' own uncertainty, the holdout's sampling, its
    entities, the quota's few slots and the values' skew, with no normal curve. With ``folds`` (the fold whose model
    scored each out-of-fold row), each sample comes from one fold's rows: one model's ranking, as the holdout is scored by
    one model, so a single model's quirks count (the review of 0.114.0: with a quota of a dozen rows they dominate). A
    fold's rows hold a fifth of the data, so without real model differences this widens the interval somewhat (more with
    more folds); pass ``folds`` only when each entity's rows sit in one fold (grouped cross-validation) and the folds'
    models are alike (not a time split's expanding windows, whose first model is the weakest).

    Rows are ranked by score; with a value per row (``value_per_case``, from ``spec.value_column``) and scores that are
    probabilities, by expected gain (score × value, less the costs), so a likely small case can come after a less likely
    large one. The expected value is against acting on no row."""
    y, s = np.asarray(y_oof).astype(int), np.asarray(score_oof, dtype=float)
    n = len(y)
    per_row = value_per_case is not None
    if spec.value_column and not per_row:
        raise ValueError(f"The decision values each case by `{spec.value_column}`: pass that column's values for the rows scored")
    value = np.asarray(value_per_case, dtype=float) if per_row else np.full(n, spec.value_per_case)
    if per_row and (len(value) != n or not np.isfinite(value).all()):
        raise ValueError("A value per row needs one finite value for every row scored")
    if spec.capacity_per_day and not rows_per_day:
        raise ValueError("A capacity per day needs how many rows arrive a day (rows_per_day): without it the cap cannot hold")
    if not n:
        raise ValueError("No rows were scored: there is no operating point to choose")
    probabilities = s.min() >= 0 and s.max() <= 1
    ranked_by = "expected gain" if per_row and probabilities else "score"

    def rank_of(score: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (score * (v + spec.cost_false_negative) - (1 - score) * spec.cost_false_positive - spec.cost_per_action) if ranked_by == "expected gain" else score

    rank = rank_of(s, value)
    rw = np.ones(n) if weights is None else np.asarray(weights, dtype=float)  # 34.3: a row's weight (a changed base rate)
    if len(rw) != n or not np.isfinite(rw).all() or (rw < 0).any() or rw.sum() <= 0:
        raise ValueError("Weights need one finite, non-negative value per row, not all zero")
    if weights is not None and (bootstrap or groups is not None or y_test is not None):
        # the review of 34.3: the bootstrap, the entities and the holdout read rows, not weights; a weighted choice is a what-if
        raise ValueError("Weighted rows choose a point only (bootstrap=0, no groups, no holdout): the intervals and the holdout read rows")
    caps = [c for c in (spec.capacity_share, (spec.capacity_per_day / rows_per_day) if spec.capacity_per_day and rows_per_day else None) if c]
    cap_share = min(min(caps), 1.0) if caps else None  # the tighter of a share and a daily count

    def curve(rows: np.ndarray) -> tuple[int, np.ndarray]:
        """The best count to act on and the value at every count; a count may end only where the rank changes, so a
        threshold (act on every row ranked at least so high) takes rows that tie with it together (the review of 34.1)."""
        r = rank[rows]
        order = np.argsort(-r, kind="mergesort")
        ev = np.r_[0.0, np.cumsum((rw[rows] * _gain(y[rows], value[rows], spec))[order])]
        ranked_r = r[order]
        valid = np.r_[True, np.r_[ranked_r[:-1] != ranked_r[1:], True]] if len(rows) else np.array([True])
        if cap_share is not None and weights is None:  # the capacity itself may end the list inside a tie (the review of
            valid = valid.copy()                        # 0.113.0: else a tie across it left only "act on none")
            valid[min(int(np.floor(cap_share * len(rows))), len(rows))] = True
        if weights is None:
            within = np.arange(len(ev)) <= (len(rows) if cap_share is None else int(np.floor(cap_share * len(rows))))
        else:  # a capacity is a share of the weighted rows
            cum = np.r_[0.0, np.cumsum(rw[rows][order])]
            within = cum <= (np.inf if cap_share is None else cap_share * cum[-1] + 1e-9)
        allowed = np.where(valid & within, ev, -np.inf)
        return int(np.argmax(allowed)), ev

    best, ev = curve(np.arange(n))
    free = int(np.argmax(np.where(np.r_[True, np.r_[np.sort(rank)[::-1][:-1] != np.sort(rank)[::-1][1:], True]], ev, -np.inf)))
    weighted_share = np.r_[0.0, np.cumsum(rw[np.argsort(-rank, kind="mergesort")])] / rw.sum()  # the share acted on at each count
    threshold = float(np.sort(rank)[::-1][best - 1]) if best else None
    binds = cap_share is not None and float(weighted_share[free]) > cap_share + 1e-12
    quota = binds and best > 0  # the capacity, not the value, ends the list: the decision is a quota

    def acted_rows(r: np.ndarray, sc: np.ndarray, v: np.ndarray) -> np.ndarray:
        """The positions acted on among rows with ranks ``r``, scores ``sc`` and values ``v``, as the decision runs (the
        measurement of 34.1 on the fraud replica): under a binding capacity, the highest-ranked rows the capacity allows,
        each still worth acting on alone (a threshold carried to rows scored on another scale would act on too few or too
        many); otherwise every row ranked at least as high as the threshold, cut to the capacity."""
        size = len(r)
        if quota:
            if ranked_by == "expected gain":
                worth = r > 0
            elif probabilities:
                worth = sc * (v + spec.cost_false_negative) - (1 - sc) * spec.cost_false_positive - spec.cost_per_action > 0
            else:  # scores that are not probabilities say nothing of one row's worth: the quota alone
                worth = np.ones(size, dtype=bool)
            worth = worth | (r >= threshold)  # never stricter than the training's own choice, which took every row this high
            candidates = np.flatnonzero(worth)
            return candidates[np.argsort(-r[candidates], kind="mergesort")[: int(np.floor(cap_share * size))]]
        top = np.flatnonzero(r >= threshold) if threshold is not None else np.array([], dtype=int)
        if cap_share is not None and len(top) > int(np.floor(cap_share * size)):
            top = top[np.argsort(-r[top], kind="mergesort")[: int(np.floor(cap_share * size))]]
        return top

    rng = np.random.default_rng(seed)
    if groups is not None:  # entities as codes; a row without one is its own unit
        codes, _ = pd.factorize(pd.Series(np.asarray(groups, dtype=object)))
        codes = np.where(codes < 0, codes.max(initial=-1) + 1 + np.arange(n), codes)
        order = np.argsort(codes, kind="stable")
        keys, starts = np.unique(codes[order], return_index=True)
        index = dict(zip(keys, np.split(order, starts[1:])))
    else:
        keys, index = np.arange(n), None
    n_test = len(y_test) if y_test is not None else 0
    fold_of = None
    if folds is not None:
        fold_of = np.asarray(folds)
        if len(fold_of) != n:
            raise ValueError("One fold per out-of-fold row: the folds must be as many as the scores")
    sizes = None
    if index is not None:  # an entity's rows by its code (a row with no entity has a code past the others: look up by code)
        sizes = np.zeros(int(keys.max()) + 1, dtype=int)
        sizes[keys] = [len(index[k]) for k in keys]
    key_fold = ({k: fold_of[index[k][0]] for k in keys} if index is not None else None) if fold_of is not None else None
    inner = np.random.default_rng([seed, 1])  # its own stream: the training intervals do not move when a holdout is given

    def holdout_sized(draw: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """A sample of the holdout's number of rows from this draw: whole entities (with replacement) until the rows are
        reached, else rows; from one fold's rows when the folds are known."""
        if index is not None:
            pool = draw
            if key_fold is not None:  # a fold by a random row (the review: by entity, a fold of many single rows came up too often)
                fold = fold_of[rows[inner.integers(0, len(rows))]]
                pool = pool[np.fromiter((key_fold[k] == fold for k in pool), bool, len(pool))]
            need = max(4 * len(pool), int(np.ceil(2 * n_test / max(float(sizes[pool].mean()), 1.0))))
            picks = pool[inner.integers(0, len(pool), size=need)]
            total = np.cumsum(sizes[picks])
            while total[-1] < n_test:  # a small fold: draw on until the holdout's rows are reached
                more = pool[inner.integers(0, len(pool), size=need)]
                picks, total = np.r_[picks, more], np.r_[total, total[-1] + np.cumsum(sizes[more])]
            stop = int(np.searchsorted(total, n_test)) + 1
            return np.concatenate([index[k] for k in picks[:stop]])
        pool = rows
        if fold_of is not None:
            pool = pool[fold_of[pool] == fold_of[pool[inner.integers(0, len(pool))]]]
        return pool[inner.integers(0, len(pool), size=n_test)]

    shares, values, held, held_rows = [], [], [], []
    for _ in range(bootstrap):
        draw = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([index[k] for k in draw]) if index is not None else draw
        k, _ = curve(rows)
        shares.append(k / len(rows))  # the best share again: how stable the choice is
        acted = rows[acted_rows(rank[rows], s[rows], value[rows])]  # the decision taken, as it runs
        values.append(float(_gain(y[acted], value[acted], spec).sum()) / len(rows))
        if n_test and best:  # a holdout-sized sample of this draw, the decision applied to it as it runs
            sample = holdout_sized(draw, rows)
            held_rows.append(len(sample))
            taken = sample[acted_rows(rank[sample], s[sample], value[sample])]
            held.append(float(_gain(y[taken], value[taken], spec).sum()) / len(sample))
    ranked = np.sort(rank)[::-1]
    out: dict[str, Any] = {
        "decision": "operating_point", "spec": spec.model_dump(), "ranked_by": ranked_by,
        "share": _r(float(weighted_share[best]), 6),
        "actions_training": best, "threshold": _r(threshold, 6),
        "threshold_of": "the score" if ranked_by == "score" else "the expected gain of acting on the row",
        "expected_value_per_row": _r(ev[best] / rw.sum()), "expected_value_per_row_interval": _interval(values),
        "share_interval": _interval(shares), "capacity_share": _r(cap_share, 6),
        "capacity_binds": binds,
        "applied_as": (("a quota: the highest-ranked rows the capacity allows, each worth acting on alone or ranked as high as the "
                        "rows chosen" + ("; a capacity per day is applied as one pool over the rows read, not day by day"
                                        if spec.capacity_per_day else "")) if quota else
                       "a threshold: every row ranked at least as high, cut to the capacity" if best else "no action"),
        "chosen_on": "out-of-fold training predictions",
        "interval": (f"90% bootstrap over {'entities' if groups is not None else 'rows'}, {bootstrap} draws, of the expected value per row "
                     "at the share chosen (the share's own interval: the best share in each draw)"),
        "curve": [{"share": _r(float(weighted_share[k]), 4), "expected_value_per_row": _r(ev[k] / rw.sum())}
                  for k in np.unique(np.linspace(0, n, 21).astype(int))] if n else [],
        "assumption": ("Acting on a row does not change whether it is a case; the values and costs are the person's stated numbers"
                       + ("; the scores are read as probabilities to rank rows by expected gain" if ranked_by == "expected gain" else "")
                       + ". Chosen against acting on no row."),
    }
    if spec.cost_per_action and (spec.value_per_case or spec.cost_false_negative) and not per_row:
        out["break_even_precision"] = _r((spec.cost_per_action + spec.cost_false_positive)
                                         / (spec.value_per_case + spec.cost_false_negative + spec.cost_false_positive))
    if per_row and ranked_by == "score":
        out["rank_note"] = "the scores are not probabilities, so rows are ranked by score, not by expected gain: calibrate to weigh the values"
    if best == 0:
        out["note"] = ("No action pays: on the training rows' out-of-fold scores, acting on even the most likely rows costs more than it "
                       "earns at these values and costs.")
    if rows_per_day:
        out["actions_per_day"] = _r(float(weighted_share[best]) * rows_per_day, 1)
    if y_test is not None and score_test is not None and best:
        yt, st = np.asarray(y_test).astype(int), np.asarray(score_test, dtype=float)
        if per_row and (value_test is None or len(value_test) != len(yt)):
            raise ValueError("The holdout's rows need their own values per case (value_test), one for each row, to be read")
        vt = np.asarray(value_test, dtype=float) if per_row else np.full(len(yt), spec.value_per_case)
        rank_t = rank_of(st, vt)
        top = acted_rows(rank_t, st, vt)  # the decision as it would run (a quota under a binding capacity, else the threshold)
        k = len(top)
        realised = float(_gain(yt[top], vt[top], spec).sum())
        interval = _interval(held) if len(held) >= 20 else _prediction(values, n, len(yt))
        out["holdout"] = {"actions": k, "cases_found": int(yt[top].sum()), "precision": _r(yt[top].mean()) if k else None,
                          "expected_value_per_row": _r(realised / len(yt)),
                          "prediction_interval": interval,
                          "prediction_interval_method": ((f"holdout-sized samples of each bootstrap draw ({len(yt)} rows, about "
                                                          f"{int(np.mean(held_rows))} drawn"
                                                          + (", whole entities" if index is not None else "")
                                                          + (", one fold's rows each" if fold_of is not None else "") + ")")
                                                         if len(held) >= 20 else "normal approximation of the bootstrap spread"),
                          "reads": "the expected value's interval is for the value per row on average; a holdout is one more sample with its "
                                   "own noise, so its value is read against the prediction interval"}
    return out


# ------------------------------------------------------------------------------------------------ a decision in words (34.4)

_AMOUNT = re.compile(r"(?:[€$£]\s*)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)\s*(k\b|thousand\b)?(?!\s*%|\d)")
_CLAUSES = re.compile(r"[;\n]|\.(?!\d)|,(?!\d)|\band\b|\bbut\b|\bwhile\b|\bwhereas\b")
# the review of 34.4: a number followed by one of these is a count, a time, a share or a total, never a cost per row
_NOT_MONEY_AFTER = re.compile(r"\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|seconds?|secs?|customers?|clients?|cases?|"
                              r"calls?|reviews?|rows?|orders?|people|persons?|users?|items?|tickets?|times?|alerts?|percent|per\s*cent|"
                              r"million|mn|m|bn|billion)\b")
_NOT_MONEY_BEFORE = re.compile(r"\b(top|first|last|after|within|every|each)\s*$")  # "the top 100", "after 12": a rank or a time
_MONEY = re.compile(r"[€$£]|\b(euros?|eur|dollars?|usd|pounds?|gbp|cost|costs|costing|worth|lose|loses|lost|losing|save|saves|"
                    r"saving|earn|earns|pay|pays|spend|spends|price|fee|fees|revenue|profit|margin)\b")
_UNREADABLE = re.compile(r"(?<![\d,])\.\d|-\s*[€$£]?\s*\d|\b\d{1,3}\.\d{3}\b")  # ".5", "-300", "10.000": not read at all
_FALSE_ALARM = ("false alarm", "false positive", "wrongly", "needless", "for nothing", "wasted", "wrong flag", "annoy")
_MISS = ("lost", "lose", "loses", "losing", "miss", "missed", "churn", "leaves", "leaving", "undetected", "not caught",
         "unnoticed", "default", "false negative")
_VALUE = ("worth", "saves", "save ", "earns", "earn ", "brings", "gain", "revenue", "profit")
_ACTIONS = ("call", "review", "check", "visit", "offer", "email", "message", "discount", "voucher", "inspection", "audit",
            "contact", "text", "sms", "coupon", "investigat", "follow-up", "follow up", "intervention", "outreach")


def read_spec(text: str) -> Optional[DecisionSpec]:
    """A decision read from a person's words (34.4), or None when they do not say enough plainly. Each clause must hold
    exactly one amount, a word that makes it money (a currency, costs, worth, lose, save, pay...) and a word that says
    what it is: a false alarm's cost, a miss's cost (a lost customer, a missed fraud), a case's value, or what acting on
    a row costs (a call, a review). A clause whose number is a count, a time, a share or a total ("the top 100", "200 a
    day" without money, "5 minutes", "3 customers", "15 percent", "2 million") is left out, never guessed; a text with an
    amount that cannot be read surely (".5", "-300", "10.000") gives nothing. Needs a cost of acting (or of a false
    alarm) and a value (or a miss's cost)."""
    raw = str(text or "")
    if _UNREADABLE.search(raw):
        return None
    found: dict[str, float] = {}
    what = ""
    for clause in _CLAUSES.split(raw):
        low = clause.lower().strip()
        matches = list(_AMOUNT.finditer(low))
        if len(matches) != 1 or not _MONEY.search(low):
            continue
        match = matches[0]
        if _NOT_MONEY_AFTER.match(low, match.end()) or _NOT_MONEY_BEFORE.search(low[: match.start()]):
            continue
        number, scale = match.group(1), match.group(2)
        thousands = re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", number)  # 1,200 is twelve hundred; 2,5 is two and a half
        amount = float(number.replace(",", "") if thousands else number.replace(",", ".")) * (1000.0 if scale else 1.0)
        if any(w in low for w in _FALSE_ALARM):
            key = "cost_false_positive"
        elif any(w in low for w in _MISS):
            key = "cost_false_negative"
        elif any(w in low for w in _VALUE):
            key = "value_per_case"
        elif any(a in low for a in _ACTIONS) or "cost" in low:
            key = "cost_per_action"
            what = what or next((a.strip() for a in _ACTIONS if a in low), "")
        else:
            continue
        if key in found and found[key] != amount:  # said twice with two amounts: not read
            return None
        found[key] = amount
    acting = found.get("cost_per_action") or found.get("cost_false_positive")
    worth = found.get("value_per_case") or found.get("cost_false_negative")
    if not (acting and worth):
        return None
    words = " ".join(str(text).split())[:300]
    return DecisionSpec(action=(what or "act")[:80], notes=f'read from the words: "{words}"', **found)


# ------------------------------------------------------------------------------------------------ what-if (34.3)

WHAT_IF_TERMS = ("value_per_case", "cost_per_action", "cost_false_positive", "cost_false_negative", "capacity_share")
NOT_REFITTED = ("not refitted: the model and its scores stay as they are; a base rate change reweights the out-of-fold rows, "
                "assuming the scores of cases and of non-cases keep their shape")


def _rate_weights(y: np.ndarray, rate: float) -> np.ndarray:
    old = float(y.mean())
    if not 0 < old < 1:
        raise ValueError("A base rate can only change when the rows hold both cases and non-cases")
    return np.where(y == 1, rate / old, (1 - rate) / (1 - old))


def what_if(y_oof: Sequence[int], score_oof: Sequence[float], spec: DecisionSpec, changes: dict[str, float], *,
            value_per_case: Optional[Sequence[float]] = None, rows_per_day: Optional[float] = None) -> dict[str, Any]:
    """A chosen decision re-scored under changed numbers (34.3), with no refit: the operating point chosen before and
    after, and what keeping the old decision (the same rows) would be worth under the new numbers. ``changes`` names any
    of the spec's values, costs or capacity share, ``base_rate`` (cases more or less common, by reweighting the rows),
    or, when each row has its own value, ``value_scale`` (every row's value times it; ``value_per_case`` is not read then)."""
    allowed = set(WHAT_IF_TERMS) | {"base_rate"} | ({"value_scale"} if value_per_case is not None else set())
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"A what-if changes the decision's own numbers or the base rate, not {', '.join(sorted(unknown))}")
    if value_per_case is not None and "value_per_case" in changes:
        raise ValueError("Each row has its own value: change them all with value_scale, not value_per_case")
    y, s = np.asarray(y_oof).astype(int), np.asarray(score_oof, dtype=float)
    rate = changes.get("base_rate")
    if rate is not None and not 0 < rate < 1:
        raise ValueError("A base rate is a share strictly between 0 and 1")
    scale = changes.get("value_scale")
    if scale is not None and not scale > 0:
        raise ValueError("A value scale multiplies every row's value: it must be above 0")
    new = DecisionSpec(**{**spec.model_dump(), **{k: v for k, v in changes.items() if k not in ("base_rate", "value_scale")}})
    w = _rate_weights(y, rate) if rate is not None else None
    old_values = np.asarray(value_per_case, dtype=float) if value_per_case is not None else None
    new_values = old_values * scale if (old_values is not None and scale is not None) else old_values
    before = choose_operating_point(y, s, spec, value_per_case=old_values, rows_per_day=rows_per_day, bootstrap=0)
    after = choose_operating_point(y, s, new, value_per_case=new_values, rows_per_day=rows_per_day, weights=w, bootstrap=0)
    # the old decision kept: the very rows it acted on (ranked with the old numbers, the same count; the review of 34.3:
    # never re-thresholded on a rounded score nor re-ranked with new costs), cut to the new capacity, valued anew
    old_vals = old_values if old_values is not None else np.full(len(y), spec.value_per_case)
    vals = new_values if new_values is not None else np.full(len(y), new.value_per_case)
    old_rank = ((s * (old_vals + spec.cost_false_negative) - (1 - s) * spec.cost_false_positive - spec.cost_per_action)
                if before["ranked_by"] == "expected gain" else s)
    order = np.argsort(-old_rank, kind="mergesort")[: before["actions_training"]]
    ww = w if w is not None else np.ones(len(y))
    cap = after["capacity_share"]
    over = False
    if cap is not None and ww[order].sum() > cap * ww.sum() + 1e-9:
        over = True
        order = order[np.cumsum(ww[order]) <= cap * ww.sum() + 1e-9]  # the old decision's best rows the new capacity still allows
    acted = np.zeros(len(y), dtype=bool)
    acted[order] = True
    kept = float((ww * _gain(y, vals, new))[acted].sum() / ww.sum())

    def brief(r: dict[str, Any]) -> dict[str, Any]:
        return {"share": r["share"], "threshold": r["threshold"], "value_per_row": r["expected_value_per_row"]}

    keep = {"share": _r(float((ww * acted).sum() / ww.sum()), 6), "value_per_row": _r(kept)}
    if over:
        keep["cut_to_capacity"] = True
    out = {"decision": "what_if", "changes": dict(changes), "before": brief(before), "after": brief(after),
           "keep_the_old_threshold": keep, "regret_if_unchanged": _r((after["expected_value_per_row"] or 0.0) - kept), "note": NOT_REFITTED}
    return out


def sensitivity(y_oof: Sequence[int], score_oof: Sequence[float], spec: DecisionSpec, *,
                multipliers: Sequence[float] = (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 4.0),
                value_per_case: Optional[Sequence[float]] = None, rows_per_day: Optional[float] = None) -> dict[str, Any]:
    """How far each stated number can move before the decision changes (34.3): for every number the spec states (and the
    base rate), the share acted on and the value per row at each multiplier, and the nearest multiplier at which the
    share moves by more than a fifth (or to or from acting on no row)."""
    y = np.asarray(y_oof).astype(int)
    base = choose_operating_point(y, np.asarray(score_oof, dtype=float), spec, value_per_case=value_per_case,
                                  rows_per_day=rows_per_day, bootstrap=0)
    per_row = value_per_case is not None
    terms = ([t for t in WHAT_IF_TERMS if getattr(spec, t) and not (per_row and t == "value_per_case")] + (["value_scale"] if per_row else [])
             + (["base_rate"] if 0 < y.mean() < 1 else []))
    rows = []
    for term in terms:
        at = []
        for m in multipliers:
            current = float(y.mean()) if term == "base_rate" else 1.0 if term == "value_scale" else float(getattr(spec, term))
            changed = current * m
            if (term == "base_rate" and not 0 < changed < 1) or (term == "capacity_share" and not 0 < changed < 1):
                at.append({"multiplier": m, "share": None, "value_per_row": None, "skipped": "outside 0 to 1"})
                continue
            r = what_if(y, score_oof, spec, {term: changed}, value_per_case=value_per_case, rows_per_day=rows_per_day)
            at.append({"multiplier": m, "share": r["after"]["share"], "value_per_row": r["after"]["value_per_row"]})
        moved = [a["multiplier"] for a in at if a["share"] is not None and base["share"] is not None
                 and ((base["share"] == 0) != (a["share"] == 0) or abs(a["share"] - base["share"]) > 0.2 * max(base["share"], 1e-9))]
        rows.append({"term": term, "at": at, "moves_at": min(moved, key=lambda m: abs(np.log(m))) if moved else None})
    return {"decision": "sensitivity", "base": {"share": base["share"], "value_per_row": base["expected_value_per_row"]},
            "multipliers": list(multipliers), "rows": rows, "note": NOT_REFITTED}


# ------------------------------------------------------------------------------------------------ targeting and several actions (34.2)

MIN_ARM = 30  # rows of each arm a group of top rows needs before its uplift is read


def _cut_points(score: np.ndarray) -> np.ndarray:
    """Counts k (rows ranked by score, highest first) a threshold can take: those ending where the score changes."""
    ranked = np.sort(score)[::-1]
    return np.r_[True, np.r_[ranked[:-1] != ranked[1:], True]] if len(score) else np.array([True])


def _incremental(score: np.ndarray, t: np.ndarray, y: np.ndarray, w: np.ndarray, spec: DecisionSpec) -> np.ndarray:
    """The measured value of treating the top k rows, for every k: the treated and untreated rows among them compared
    (weighted when the arms were not randomised), their difference in outcome times k times the value of a conversion,
    less the cost of k treatments. NaN where either arm has fewer than ``MIN_ARM`` rows."""
    order = np.argsort(-score, kind="mergesort")
    tt, yy, ww = t[order], y[order], w[order]
    wt, wc = np.cumsum(ww * tt), np.cumsum(ww * (1 - tt))
    yt, yc = np.cumsum(ww * tt * yy), np.cumsum(ww * (1 - tt) * yy)
    nt, nc = np.cumsum(tt), np.cumsum(1 - tt)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where((nt >= MIN_ARM) & (nc >= MIN_ARM), yt / wt - yc / wc, np.nan)
    k = np.arange(1, len(score) + 1)
    return np.r_[0.0, (lift * spec.value_per_case - spec.cost_per_action) * k]


def target_by_uplift(score: Sequence[float], treated: Sequence[int], outcome: Sequence[int], spec: DecisionSpec, *,
                     weights: Optional[Sequence[float]] = None, score_test: Optional[Sequence[float]] = None,
                     t_test: Optional[Sequence[int]] = None, y_test: Optional[Sequence[int]] = None,
                     w_test: Optional[Sequence[float]] = None, bootstrap: int = BOOTSTRAP, seed: int = 0) -> dict[str, Any]:
    """Whom to treat (34.2): the share of rows, ranked by an uplift model's score, whose treatment is worth the most, its
    value measured from the arms of rows the model never trained on (the validation rows), then read once on the holdout.

    A conversion is worth ``spec.value_per_case`` and a treatment costs ``spec.cost_per_action``; ``spec.capacity_share``
    caps the share. ``weights`` (inverse propensities) when the treatment was not randomised. The value of treating
    everyone, no one and a random share of the same size are given beside it."""
    unsupported = [f for f in ("capacity_per_day", "value_column", "cost_false_positive", "cost_false_negative") if getattr(spec, f)]
    if unsupported:  # the review of 34.2: a field read by nobody would be silently ignored
        raise ValueError(f"Targeting reads a conversion's value, a treatment's cost and a share cap; not {', '.join(unsupported)}")
    s, t, y = np.asarray(score, dtype=float), np.asarray(treated).astype(int), np.asarray(outcome).astype(int)
    n = len(s)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if not (len(t) == len(y) == len(w) == n) or not np.isfinite(s).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Scores, arms, outcomes and weights need one finite value per row (weights above zero)")
    if score_test is not None:
        lengths = {len(x) for x in (score_test, t_test, y_test) if x is not None}
        if len(lengths) != 1 or t_test is None or y_test is None or not np.isfinite(np.asarray(score_test, dtype=float)).all():
            raise ValueError("The holdout needs a finite score, an arm and an outcome for every row")
        if weights is not None and (w_test is None or len(w_test) != len(score_test) or not np.isfinite(np.asarray(w_test, dtype=float)).all()):
            raise ValueError("The arms were weighted: the holdout's rows need their weights too (w_test) to be read")
    if n == 0 or min(int(t.sum()), int((1 - t).sum())) < MIN_ARM:
        raise ValueError(f"Targeting needs at least {MIN_ARM} treated and {MIN_ARM} untreated rows to compare")
    if not spec.value_per_case:
        raise ValueError("Targeting needs what a conversion is worth (value_per_case)")
    valid_k = _cut_points(s)

    def best_of(rows: np.ndarray) -> tuple[int, np.ndarray]:
        ev = _incremental(s[rows], t[rows], y[rows], w[rows], spec)
        ok = _cut_points(s[rows]) & np.isfinite(ev)
        if spec.capacity_share is not None:
            ok &= np.arange(len(ev)) <= int(np.floor(spec.capacity_share * len(rows)))
        return int(np.argmax(np.where(ok, ev, -np.inf))), ev

    best, ev = best_of(np.arange(n))
    threshold = float(np.sort(s)[::-1][best - 1]) if best else None
    everyone = _incremental(s, t, y, w, spec)[n]
    rng = np.random.default_rng(seed)
    values, shares = [], []
    for _ in range(bootstrap):
        rows = rng.choice(n, size=n, replace=True)
        k, _ = best_of(rows)
        shares.append(k / n)
        top = rows[s[rows] >= threshold] if threshold is not None else rows[:0]
        values.append(_value_of(top, rows, t, y, w, spec) if len(top) else 0.0)
    out: dict[str, Any] = {
        "decision": "targeting", "spec": spec.model_dump(), "share": _r(best / n, 6), "treated_training": best, "threshold": _r(threshold, 6),
        "value_per_row": _r(ev[best] / n), "value_per_row_interval": _interval(values), "share_interval": _interval(shares),
        "compared": {"treat no one": 0.0, "treat everyone": _r(everyone / n) if np.isfinite(everyone) else None,
                     "treat a random share of the same size": _r(best / n * everyone / n) if np.isfinite(everyone) else None},
        "chosen_on": "the validation rows' arms", "weighted": weights is not None,
        "interval": f"90% bootstrap over rows, {bootstrap} draws, of the value per row at the threshold chosen",
        "assumption": ("The treated and untreated rows among the top rows are comparable (randomised, or weighted by the propensity); "
                       "treating a row does not change another's outcome; a conversion's value and a treatment's cost are the person's numbers."),
    }
    if best == 0:
        out["note"] = "No share pays: on the validation rows, treating even the highest-scored rows does not earn its cost."
        out["pays"] = "no"
    elif out["value_per_row_interval"][0] is None or out["value_per_row_interval"][0] <= 0:
        out["pays"] = "cannot tell"  # the review of 34.2: on pure noise a small, noisy top share is often the best by chance
        out["note"] = ("Cannot tell whether it pays: the value's 90% interval reaches zero, so the share chosen may be noise "
                       "(the smallest groups of top rows are the noisiest); treat no one, or test the targeting first.")
    else:
        out["pays"] = "yes"
    if spec.capacity_share is not None:
        out["capacity_share"] = spec.capacity_share
    if score_test is not None and t_test is not None and y_test is not None and best:
        st, tt, yt = np.asarray(score_test, dtype=float), np.asarray(t_test).astype(int), np.asarray(y_test).astype(int)
        top = np.flatnonzero(st >= threshold)
        if spec.capacity_share is not None and len(top) > int(np.floor(spec.capacity_share * len(st))):
            top = top[np.argsort(-st[top], kind="mergesort")[: int(np.floor(spec.capacity_share * len(st)))]]
        everyone_rows = np.arange(len(st))
        wt = np.asarray(w_test, dtype=float) if w_test is not None else np.ones(len(st))
        measured = _value_of(top, everyone_rows, tt, yt, wt, spec) if len(top) else 0.0
        out["holdout"] = {"treated": int(len(top)), "value_per_row": _r(measured), "prediction_interval": _prediction(values, n, len(st)),
                          "reads": "measured from the holdout's own arms among the rows treated; a new sample, so read against the prediction interval"}
    return out


def _value_of(top: np.ndarray, rows: np.ndarray, t: np.ndarray, y: np.ndarray, w: np.ndarray, spec: DecisionSpec) -> float:
    """The measured value per row of ``rows`` when ``top`` are treated: the arms' difference among ``top`` times its size."""
    tt, yy, ww = t[top], y[top], w[top]
    if min(int(tt.sum()), int((1 - tt).sum())) < 1:
        return float("nan")
    lift = (ww * tt * yy).sum() / (ww * tt).sum() - (ww * (1 - tt) * yy).sum() / (ww * (1 - tt)).sum()
    return float((lift * spec.value_per_case - spec.cost_per_action) * len(top) / len(rows))


def choose_actions(values: Any, capacities: Sequence[Optional[int]], actions: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Which of several actions each row gets (34.2): ``values[i, a]`` is what action ``a`` on row ``i`` is worth against no
    action (-1 when none is worth it); each action may be taken at most ``capacities[a]`` times (None: no limit).

    Exact: rows are placed one at a time, each along the best chain of moves (it takes an action; if that action is full,
    one of its rows moves to another action or to none, and so on), as a min-cost flow is solved; checked against an
    exhaustive search (600 random cases with ties, no difference)."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError("Values must be a table: one row per row scored, one column per action")
    n, a = v.shape
    if len(capacities) != a:
        raise ValueError(f"One capacity per action: {a} actions, {len(capacities)} capacities")
    if any(c is not None and c < 0 for c in capacities):
        raise ValueError("A capacity is a count of rows, never negative")
    if np.isnan(v).any() or np.isposinf(v).any():
        raise ValueError("Every value must be a number (minus infinity marks an action a row may not take)")
    names = list(actions) if actions is not None else [f"action_{j}" for j in range(a)]
    best_assign = _flow_assign(v, [n if c is None else int(c) for c in capacities])
    best_total = float(sum(v[i, j] for i, j in enumerate(best_assign) if j >= 0))
    counts = {names[j]: int(sum(1 for x in best_assign if x == j)) for j in range(a)}
    counts["none"] = int(sum(1 for x in best_assign if x < 0))
    return {"decision": "actions", "assignment": [int(x) for x in best_assign], "labels": [names[x] if x >= 0 else None for x in best_assign],
            "counts": counts, "total_value": _r(best_total, 6), "capacities": list(capacities), "method": "min-cost flow (exact)"}


TARGETING_WHAT_IF_TERMS = ("value_per_case", "cost_per_action", "capacity_share")


def targeting_what_if(score: Sequence[float], treated: Sequence[int], outcome: Sequence[int], spec: DecisionSpec, changes: dict[str, float], *,
                      weights: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """Whom to treat (34.2) re-chosen under changed numbers (34.3), with no refit: the share chosen before and after, and
    what treating the same rows (the old top rows by score, cut to a smaller capacity) is worth under the new numbers.
    ``changes`` names a conversion's value, a treatment's cost or the capacity share. A base rate is refused: an uplift is
    read as the difference between the arms among the top rows, and reweighting the cases would not move it honestly."""
    unknown = set(changes) - set(TARGETING_WHAT_IF_TERMS)
    if unknown:
        raise ValueError(f"A targeting what-if changes {', '.join(TARGETING_WHAT_IF_TERMS)}; not {', '.join(sorted(unknown))}")
    new = DecisionSpec(**{**spec.model_dump(), **changes})
    before = target_by_uplift(score, treated, outcome, spec, weights=weights, bootstrap=0)
    after = target_by_uplift(score, treated, outcome, new, weights=weights, bootstrap=0)
    s, t, y = np.asarray(score, dtype=float), np.asarray(treated).astype(int), np.asarray(outcome).astype(int)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    n = len(s)
    k = before["treated_training"]  # the very rows treated before: the top k by score
    limit = None if new.capacity_share is None else int(np.floor(new.capacity_share * n))
    over = limit is not None and k > limit
    if over:  # cut where the score changes, as target_by_uplift only stops there (the review of 0.113.0: ties)
        k = int(np.flatnonzero(_cut_points(s)[: limit + 1])[-1])
    curve = _incremental(s, t, y, w, new)  # read as target_by_uplift reads it: NaN when an arm has too few rows
    kept = float(curve[k]) / n if k else 0.0
    kept_value = _r(kept) if np.isfinite(kept) else None

    def brief(r: dict[str, Any]) -> dict[str, Any]:  # no "pays": these choices are made without a bootstrap, so it cannot say yes
        return {"share": r["share"], "threshold": r["threshold"], "value_per_row": r["value_per_row"]}

    keep: dict[str, Any] = {"share": _r(k / n, 6), "value_per_row": kept_value}
    if over:
        keep["cut_to_capacity"] = True
    regret = _r((after["value_per_row"] or 0.0) - kept) if kept_value is not None else None
    return {"decision": "targeting_what_if", "changes": dict(changes), "before": brief(before), "after": brief(after),
            "keep_the_old_rows": keep, "regret_if_unchanged": regret,
            "note": "not refitted: the uplift model's scores and the arms stay as they are; the share is chosen again on the same rows"}


ACTIONS_WHAT_IF_TERMS = ("capacities", "value_scale", "added_cost")


def actions_what_if(values: Any, capacities: Sequence[Optional[int]], changes: dict[str, dict[str, float]], *,
                    actions: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Which action each row gets (34.2) re-chosen under changed numbers (34.3), with no refit: ``changes`` maps
    ``capacities`` (an action's new count, None for no limit), ``value_scale`` (every row's value of that action times a
    number above 0) or ``added_cost`` (subtracted from every row's value of that action), each by action name. The old
    assignment kept is valued with the new values, each action cut to its new capacity (its rows worth least go to no
    action)."""
    unknown = set(changes) - set(ACTIONS_WHAT_IF_TERMS)
    if unknown:
        raise ValueError(f"An actions what-if changes {', '.join(ACTIONS_WHAT_IF_TERMS)}; not {', '.join(sorted(unknown))}")
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError("Values must be a table: one row per row scored, one column per action")
    names = list(actions) if actions is not None else [f"action_{j}" for j in range(v.shape[1])]
    if len(capacities) != v.shape[1] or len(names) != v.shape[1]:
        raise ValueError(f"One capacity and one name per action: {v.shape[1]} actions")
    index = {name: j for j, name in enumerate(names)}
    for term, by_action in changes.items():
        if not isinstance(by_action, dict) or set(by_action) - set(index):
            raise ValueError(f"{term} names actions among {', '.join(names)}")
    caps = list(capacities)
    for name, c in (changes.get("capacities") or {}).items():
        if c is not None and c < 0:
            raise ValueError("A capacity is a count of rows, never negative")
        caps[index[name]] = c
    new_v = v.copy()
    for name, k in (changes.get("value_scale") or {}).items():
        if not k > 0:
            raise ValueError("A value scale multiplies an action's values: it must be above 0")
        new_v[:, index[name]] *= k
    for name, cost in (changes.get("added_cost") or {}).items():
        new_v[:, index[name]] -= cost
    before = choose_actions(v, capacities, names)
    after = choose_actions(new_v, caps, names)
    kept = list(before["assignment"])
    over = False
    for j, c in enumerate(caps):
        rows = [i for i, a in enumerate(kept) if a == j]
        if c is not None and len(rows) > c:
            over = True
            for i in sorted(rows, key=lambda i: -new_v[i, j])[int(c):]:  # the rows of that action worth least under the new values
                kept[i] = -1
    total = float(sum(new_v[i, a] for i, a in enumerate(kept) if a >= 0))
    counts = {names[j]: int(sum(1 for a in kept if a == j)) for j in range(len(names))}
    counts["none"] = int(sum(1 for a in kept if a < 0))
    keep: dict[str, Any] = {"counts": counts, "total_value": _r(total, 6)}
    if over:
        keep["cut_to_capacity"] = True
    return {"decision": "actions_what_if", "changes": {k: dict(v_) for k, v_ in changes.items()},
            "before": {"counts": before["counts"], "total_value": before["total_value"]},
            "after": {"counts": after["counts"], "total_value": after["total_value"]},
            "keep_the_old_assignment": keep, "regret_if_unchanged": _r((after["total_value"] or 0.0) - total, 6),
            "note": "not refitted: each row's values come from the same models; the assignment is solved again exactly"}


def _flow_assign(v: np.ndarray, caps: list[int]) -> list[int]:
    """Maximum total value assignment of rows to actions (or none) under the actions' capacities, exactly, as successive
    shortest paths of a min-cost flow: rows are placed one at a time (highest value first) along the best chain of moves.
    A chain may pass a full action only by moving one of its rows to another action or to none; since a row belongs to
    one action, the best row for each move (from action j to k) is found with one vectorised step, and the best chain by
    a Bellman-Ford sweep over the actions alone."""
    n, a = v.shape
    assign = np.full(n, -1)
    load = np.zeros(a, dtype=int)
    for i in np.argsort(-v.max(axis=1), kind="mergesort"):
        moves: dict[tuple[int, int], tuple[float, int]] = {}  # (from j, to k or -1): (value change, the row moved)
        for j in range(a):
            if load[j] < caps[j]:
                continue  # a free slot: nobody needs to move out of j
            members = np.flatnonzero(assign == j)
            if not len(members):
                continue
            out = -v[members, j]
            r = int(np.argmax(out))
            moves[(j, -1)] = (float(out[r]), int(members[r]))
            for k in range(a):
                if k != j:
                    gain = v[members, k] - v[members, j]
                    r = int(np.argmax(gain))
                    moves[(j, k)] = (float(gain[r]), int(members[r]))
        free = np.array([0.0 if load[j] < caps[j] else -np.inf for j in range(a)])  # the best value of freeing a slot at j
        parent: list[Optional[tuple[int, int]]] = [None] * a
        for _ in range(a + 1):
            changed = False
            for (j, k), (gain, row) in moves.items():
                value = gain if k < 0 else (gain + free[k] if np.isfinite(free[k]) else -np.inf)
                if value > free[j] + 1e-12:
                    free[j], parent[j], changed = value, (row, k), True
            if not changed:
                break
        gains = v[i] + free
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]) or gains[j] <= 0:
            continue
        row, target = int(i), j
        for _ in range(a + 1):  # walk the chain: each row enters an action; a full one sends its chosen row on
            assign[row] = target
            if load[target] < caps[target]:
                load[target] += 1
                break
            moved, nxt = parent[target]
            if nxt < 0:  # the chain ends with a row leaving for no action
                assign[moved] = -1
                break
            row, target = moved, nxt
    return [int(x) for x in assign]


def choose_by_cost_matrix(proba: Any, cost_matrix: Any, classes: Optional[Sequence[Any]] = None) -> dict[str, Any]:
    """The class to act on for each row (34.2): the one with the lowest expected cost, ``cost_matrix[true, chosen]`` weighed
    by the row's class probabilities. Costs are never negative; the diagonal is usually zero."""
    p, c = np.asarray(proba, dtype=float), np.asarray(cost_matrix, dtype=float)
    if p.ndim != 2 or c.shape != (p.shape[1], p.shape[1]):
        raise ValueError("The cost matrix must be square, one row and one column per class of the probabilities")
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("A cost is a number, never negative")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each row's class probabilities must be numbers from 0 to 1 that add up to 1")
    expected = p @ c  # expected[i, j]: the cost of choosing j for row i
    choice = expected.argmin(axis=1)
    out = {"decision": "cost_matrix", "choice": [int(x) for x in choice],
           "expected_cost": [_r(expected[i, j], 6) for i, j in enumerate(choice)],
           "assumption": "The probabilities are calibrated; the costs are the person's numbers."}
    if classes is not None:
        out["labels"] = [classes[int(x)] for x in choice]
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
