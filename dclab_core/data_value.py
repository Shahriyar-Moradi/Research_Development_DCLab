"""Would more rows help? (package 34.13) A learning curve on the training folds only, and a plain answer a person can act
on.

The question a team asks before paying to collect or label more rows: would the model get better with them? The curve
answers it with what the training rows already show. Each training fold's fit rows are cut to 25, 50, 75 and 100 %,
the model is fitted on each cut and scored on the same validation rows. The cut follows the split: whole groups when
the split keeps groups apart, the latest rows when it is ordered by time (more rows then means more history), a sample
stratified by class otherwise. The smaller cuts are drawn ``repeats`` times, so one lucky sample does not decide.

* **The score at each size**, with a 90 % interval: the validation rows (or their groups, or blocks of time) are
  resampled within each fold, and so are the repeats; every size is scored on the same draw, so the sizes are paired.
* **The last step's gain** (75 % to 100 % of the rows, a third more rows), with its interval from the same paired draws:
  positive means better (a fall, for an error metric). ``per_doubling`` says what doubling the rows would add if the
  curve stayed as steep as its last step; curves flatten, so it is an upper guess.
* **The answer** (:data:`RULE`, fixed before the replicas were run): ``unlikely_help`` when the interval's top is below
  the practical gain (even the optimistic reading adds less than counts); ``likely_help`` when its bottom is above zero
  and the gain itself reaches the practical gain; ``cannot_tell`` otherwise. The practical gain is 0.005 for a score
  between 0 and 1 (AUC, accuracy, F1, R²) and 1 % of the score for an error in the outcome's units, unless the caller
  states its own.
* **Per segment where errors concentrate**: the levels whose share of the out-of-fold error (every row, every fold)
  is well above their share of the rows get the same curve, scored on their own rows. The models are the same fits, so
  it says whether the segment's score still climbs as all rows grow, not what rows of that segment alone would add.

The holdout is never read: the function receives the training rows' labels and folds over them, and refuses a fold
that names a row past them. The model is injected (``fit_predict``), so a host fits its own pipeline inside each fold.

Not :class:`dclab_core.service.LearningCurve` (32.25): that one is the store's knowledge across many builds (a decision
point's or a problem kind's score by the count of labels, never a business's names or figures); this one is one project's
measurement inside its own training folds, with the answer for that project and segments named by its table's levels.

Pure: arrays in, a contract out (numpy, pandas and scikit-learn only).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
LIKELY, UNLIKELY, CANNOT = "likely_help", "unlikely_help", "cannot_tell"
ANSWERS = (LIKELY, UNLIKELY, CANNOT)
RULE = "data_value_rule_v1"
LEVEL = 0.9
BOUNDED = frozenset({"roc_auc", "average_precision", "accuracy", "balanced_accuracy", "macro_f1", "f1", "weighted_f1", "precision",
                     "recall", "recall_at_budget", "precision_at_k", "r2", "pr_auc", "ndcg", "map", "concordance", "c_index"})
SEGMENT_LIFT = 1.2  # a level's share of the error at least this many times its share of the rows
Answer = Literal["likely_help", "unlikely_help", "cannot_tell"]
How = Literal["random", "stratified", "group", "time"]


class SizeScore(Contract):
    """The score with a share of the training rows: the mean over folds (the pooled rows, for a segment)."""

    fraction: float = Field(gt=0, le=1)
    fit_rows: int = Field(ge=0)  # the rows a model was fitted on, the mean over folds
    score: float
    low: float
    high: float


class Slope(Contract):
    """The gain of the curve's last step, oriented so that positive is better, with its interval."""

    gain: float
    low: float
    high: float
    per_doubling: float  # the gain from doubling the rows if the curve stayed as steep as its last step (an upper guess)


class SegmentValue(Contract):
    """A segment where the errors concentrate (a level of a column), and its own curve."""

    column: str = Field(max_length=200)
    level: str = Field(max_length=200)
    rows: int = Field(ge=0)  # its validation rows, every fold
    row_share: float = Field(ge=0, le=1)
    error_share: float = Field(ge=0, le=1)
    points: tuple[SizeScore, ...] = ()
    slope: Optional[Slope] = None
    answer: Answer
    reason: str = Field(max_length=600)


class DataValue(Versioned):
    """Would more rows help (34.13, v1): the learning curve on the training folds, its last step's gain, the answer and
    the assumption it rests on, and the same per segment where errors concentrate. The holdout is never read."""

    schema_version: Literal["dclab_core.data_value.v1"] = _version("data_value")
    metric: str = Field(min_length=1, max_length=80)
    higher_is_better: bool
    subsample: How
    subsample_rule: str = Field(max_length=400)
    folds: int = Field(ge=1)
    repeats: int = Field(ge=1)
    bootstrap: int = Field(ge=0)
    level: float = Field(gt=0, lt=1)
    training_rows: int = Field(ge=0)
    points: tuple[SizeScore, ...]
    slope: Optional[Slope] = None
    practical: float = Field(ge=0)
    answer: Answer
    plain: str = Field(max_length=1200)
    assumption: str = Field(max_length=1200)
    rule: str = RULE
    segments: tuple[SegmentValue, ...] = ()
    holdout_read: Literal[False] = False
    notes: tuple[str, ...] = ()


CONTRACTS["data_value"] = DataValue


# ------------------------------------------------------------------------------------------------ subsampling


def subsample(fit_rows: Sequence[int], fraction: float, *, how: How, rng: np.random.Generator, groups: Any = None, order: Any = None,
              strata: Any = None) -> np.ndarray:
    """``fraction`` of ``fit_rows``, cut the way the split is made: the latest rows by ``order`` (``time``), whole
    groups drawn at random until the rows reach the share (``group``), a random sample within each class (``stratified``,
    at least two of a class that has them), or plain random rows. ``groups``, ``order`` and ``strata`` are indexed by
    the row positions ``fit_rows`` holds."""
    rows = np.asarray(fit_rows, dtype=int)
    if fraction >= 1 or len(rows) == 0:
        return np.sort(rows)
    target = max(1, int(round(fraction * len(rows))))
    if how == "time":
        keys = np.asarray(order)[rows]
        latest = np.argsort(keys, kind="stable")[-target:]
        return np.sort(rows[latest])
    if how == "group":
        labels = _labels(np.asarray(groups, dtype=object)[rows])
        unique, counts = np.unique(labels, return_counts=True)
        shuffled = rng.permutation(len(unique))
        cumulative = np.cumsum(counts[shuffled])
        take = int(np.argmin(np.abs(cumulative - target))) + 1  # the number of groups whose rows come closest to the share
        chosen = unique[shuffled[:take]]
        return np.sort(rows[np.isin(labels, chosen)])
    if how == "stratified":
        classes = _labels(np.asarray(strata, dtype=object)[rows])
        picked = []
        for c in np.unique(classes):
            members = rows[classes == c]
            k = min(len(members), max(min(2, len(members)), int(round(fraction * len(members)))))
            picked.append(rng.choice(members, k, replace=False))
        return np.sort(np.concatenate(picked))
    return np.sort(rng.choice(rows, target, replace=False))


def _labels(values: np.ndarray) -> np.ndarray:
    """Text labels; a missing value is its own label per row (NaN never equals itself)."""
    missing = pd.isna(pd.Series(values, dtype=object)).to_numpy()
    return np.array([f"__row_{i}" if missing[i] else str(v) for i, v in enumerate(values)], dtype=str)


def _resample(n: int, rng: np.random.Generator, clusters: Optional[np.ndarray] = None, block: Optional[int] = None) -> np.ndarray:
    """One bootstrap draw of positions 0..n-1: whole clusters, moving blocks (time-ordered rows), or rows."""
    if n == 0:
        return np.zeros(0, dtype=int)
    if clusters is not None:
        unique, inverse = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
        drawn = rng.integers(0, len(unique), len(unique))
        return np.concatenate([members[i] for i in drawn])
    if block and block > 1 and n > block:
        starts = rng.integers(0, n - block + 1, int(math.ceil(n / block)))
        return np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return rng.integers(0, n, n)


# ------------------------------------------------------------------------------------------------ the answer


def practical_gain(metric: str, score_full: float) -> float:
    """The smallest gain of the last step that counts: 0.005 on a score between 0 and 1, else 1 % of the score."""
    return 0.005 if metric in BOUNDED else 0.01 * abs(float(score_full))


def answer_of(slope: Optional[Slope], practical: float) -> tuple[str, str]:
    """The answer and why, by :data:`RULE`."""
    if slope is None:
        return CANNOT, "the curve's last step could not be measured"
    if slope.high < practical:
        return UNLIKELY, f"even the top of the interval ({slope.high:+.4g}) is below the practical gain ({practical:.4g})"
    if slope.low > 0 and slope.gain >= practical:
        return LIKELY, f"the interval ({slope.low:+.4g} to {slope.high:+.4g}) is above zero and the gain reaches {practical:.4g}"
    return CANNOT, f"the interval ({slope.low:+.4g} to {slope.high:+.4g}) holds both no gain and a gain of {practical:.4g}"


def default_row_loss(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """A row's error: one minus the probability of its class (a matrix of probabilities), else the absolute gap."""
    p = np.asarray(pred, dtype=float)
    if p.ndim == 2:
        return 1.0 - p[np.arange(len(p)), np.asarray(y).astype(int)]
    return np.abs(np.asarray(y, dtype=float) - p)


# ------------------------------------------------------------------------------------------------ the curve


def learning_curve(y: Sequence[Any], folds: Sequence[tuple[Sequence[int], Sequence[int]]], fit_predict: Callable[[np.ndarray, np.ndarray], Any],
                   score: Callable[[Any, Any], float], *, metric: str, higher_is_better: bool = True, groups: Any = None, order: Any = None,
                   strata: Any = None, segments: Optional[Mapping[str, Sequence[Any]]] = None,
                   row_loss: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None, fractions: Sequence[float] = FRACTIONS,
                   repeats: int = 2, bootstrap: int = 200, level: float = LEVEL, practical: Optional[float] = None, seed: int = 0,
                   min_fit_rows: int = 20, min_segment_rows: int = 60, max_segments: int = 3) -> DataValue:
    """The learning curve of ``fit_predict`` on the training rows ``y`` (positions 0..n-1) and their ``folds``.

    ``fit_predict(fit_rows, valid_rows)`` fits on the first positions and returns predictions for the second, in order;
    ``score(y_true, predictions)`` is the metric. ``groups`` (whole groups are cut) or ``order`` (the latest rows are
    kept; it wins over ``groups``, whose folds already keep them apart) or ``strata`` (classes kept) say how a fit's rows
    are cut; each is indexed by position. ``segments``: candidate columns (a value per training row) where errors may
    concentrate. See the module's note for the rule."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    fractions = tuple(float(f) for f in fractions)
    if len(fractions) < 2 or any(b <= a for a, b in zip(fractions, fractions[1:])) or fractions[0] <= 0 or fractions[-1] != 1.0:
        raise ValueError("fractions rise strictly and end at 1.0")
    if not folds:
        raise ValueError("the curve needs at least one training fold")
    folds = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in folds]
    for a, b in folds:
        if len(a) == 0 or len(b) == 0 or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= n:
            raise ValueError(f"a fold names a row outside the {n} training rows given: the curve reads training rows only")
        if np.intersect1d(a, b).size:
            raise ValueError("a fold's fit rows and validation rows overlap")
    how: How = "time" if order is not None else "group" if groups is not None else "stratified" if strata is not None else "random"
    reps = [1 if (f >= 1 or how == "time") else max(1, int(repeats)) for f in fractions]
    rng = np.random.default_rng(seed)
    notes: list[str] = []

    preds: list[list[list[np.ndarray]]] = []  # fold -> fraction -> repeat
    fit_rows: list[list[float]] = []
    failed = False
    for fold, (fit, valid) in enumerate(folds):
        per_fraction, counts = [], []
        for j, fraction in enumerate(fractions):
            runs, sizes = [], []
            for _ in range(reps[j]):
                cut = subsample(fit, fraction, how=how, rng=rng, groups=groups, order=order, strata=strata)
                sizes.append(len(cut))
                if len(cut) < min_fit_rows:
                    failed = True
                    notes.append(f"fold {fold + 1}: {fraction:.0%} of the fit rows is {len(cut)} rows, below {min_fit_rows}")
                    break
                try:
                    out = np.asarray(fit_predict(cut, valid))
                except Exception as exc:  # noqa: BLE001 - a cut that cannot be fitted (one class) makes the curve unmeasured, never a crash
                    failed = True
                    notes.append(f"fold {fold + 1}: {fraction:.0%} of the fit rows could not be fitted ({type(exc).__name__})")
                    break
                if len(out) != len(valid):
                    raise ValueError("fit_predict returned a prediction count unlike the validation rows'")
                runs.append(out)
            per_fraction.append(runs)
            counts.append(float(np.mean(sizes)) if sizes else 0.0)
            if failed:
                break
        preds.append(per_fraction)
        fit_rows.append(counts)
        if failed:
            break

    sign = 1.0 if higher_is_better else -1.0
    rule_text = {"time": "the latest rows of each fold's fit rows (more rows means more history)",
                 "group": "whole groups of each fold's fit rows, drawn at random",
                 "stratified": "a random sample of each fold's fit rows within each class",
                 "random": "a random sample of each fold's fit rows"}[how]
    common = dict(metric=metric, higher_is_better=higher_is_better, subsample=how, subsample_rule=rule_text, folds=len(folds),
                  repeats=max(reps), bootstrap=int(bootstrap), level=level, training_rows=int(n))
    if failed:
        return DataValue(points=(), slope=None, practical=float(practical or 0.0), answer=CANNOT,
                         plain="Cannot tell whether more rows help: the curve could not be drawn on these training rows (" + notes[-1] + ").",
                         assumption=_assumption(how, None), notes=tuple(notes[:20]), **common)

    clusters = None if groups is None else _labels(np.asarray(groups, dtype=object))
    block = None
    if how == "time":
        block = max(2, int(round(np.mean([len(v) for _, v in folds]) ** (1 / 3))))

    def evaluate(selected: list[np.ndarray], pooled: bool, draw_seed: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Scores per fraction and the bootstrap draws (draws x fractions); ``selected[f]``: positions within fold f's validation rows."""
        def scores_of(take: list[np.ndarray], rep_pick: Callable[[int, int], list[int]]) -> np.ndarray:
            out = np.empty(len(fractions))
            for j in range(len(fractions)):
                if pooled:
                    values = []
                    for r in rep_pick(-1, j):
                        yy = np.concatenate([y_arr[folds[f][1]][take[f]] for f in range(len(folds))])
                        pp = np.concatenate([preds[f][j][r % len(preds[f][j])][take[f]] for f in range(len(folds))])
                        values.append(_safe(score, yy, pp))
                    out[j] = np.mean(values)
                else:
                    per_fold = []
                    for f in range(len(folds)):
                        yy = y_arr[folds[f][1]][take[f]]
                        per_fold.append(np.mean([_safe(score, yy, preds[f][j][r][take[f]]) for r in rep_pick(f, j)]))
                    out[j] = np.mean(per_fold)
            return out

        point = scores_of(selected, lambda f, j: list(range(reps[j])))
        if not bootstrap:
            return point, None
        r = np.random.default_rng(draw_seed)
        draws = np.empty((int(bootstrap), len(fractions)))
        for b in range(int(bootstrap)):
            take = []
            for f in range(len(folds)):
                sel = selected[f]
                cl = None if clusters is None else clusters[folds[f][1]][sel]
                take.append(sel[_resample(len(sel), r, cl, block)])
            picks = {(f, j): list(r.integers(0, reps[j], reps[j])) for f in range(-1, len(folds)) for j in range(len(fractions))}
            draws[b] = scores_of(take, lambda f, j: picks[(f, j)])
        return point, draws

    every = [np.arange(len(v)) for _, v in folds]
    point, draws = evaluate(every, pooled=False, draw_seed=seed + 1)
    points, slope = _summarise(point, draws, fractions, fit_rows, sign, level)
    if points is None:
        return DataValue(points=(), slope=None, practical=float(practical or 0.0), answer=CANNOT,
                         plain="Cannot tell whether more rows help: the metric could not be computed on these validation rows.",
                         assumption=_assumption(how, None), notes=("the metric was not finite on the validation rows",), **common)
    gain_floor = float(practical) if practical is not None else practical_gain(metric, points[-1].score)
    answer, why = answer_of(slope, gain_floor)
    found = _segments(y_arr, folds, preds, segments, row_loss or default_row_loss, min_segment_rows, max_segments) if segments else []
    out_segments = []
    for k, (column, value, share_rows, share_error, rows, selected) in enumerate(found):
        s_point, s_draws = evaluate(selected, pooled=True, draw_seed=seed + 101 + k)
        s_points, s_slope = _summarise(s_point, s_draws, fractions, fit_rows, sign, level)
        s_answer, s_why = answer_of(s_slope, gain_floor) if s_points is not None else (CANNOT, "the metric could not be computed on this segment's rows")
        out_segments.append(SegmentValue(column=str(column)[:200], level=str(value)[:200], rows=int(rows), row_share=share_rows, error_share=share_error,
                                         points=tuple(s_points or ()), slope=s_slope, answer=s_answer,
                                         reason=(f"{share_error:.0%} of the out-of-fold error on {share_rows:.0%} of the rows; scored on its own rows, pooled over folds, "
                                                 f"with the same fits: {s_why}")[:600]))
    return DataValue(points=tuple(points), slope=slope, practical=gain_floor, answer=answer, plain=_plain(answer, why, metric, slope, fractions),
                     assumption=_assumption(how, slope), segments=tuple(out_segments), notes=tuple(notes[:20]), **common)


def _safe(score: Callable[[Any, Any], float], y: np.ndarray, p: np.ndarray) -> float:
    try:
        value = float(score(y, p))
    except (ValueError, ZeroDivisionError):  # a draw with one class has no AUC: it says nothing and is dropped
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _summarise(point: np.ndarray, draws: Optional[np.ndarray], fractions: tuple[float, ...], fit_rows: list[list[float]], sign: float,
               level: float) -> tuple[Optional[list[SizeScore]], Optional[Slope]]:
    if not np.all(np.isfinite(point)):
        return None, None
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    usable = None
    if draws is not None:
        usable = draws[np.all(np.isfinite(draws), axis=1)]
        if len(usable) < max(10, len(draws) // 2):  # most draws said nothing (a class missing): no interval is honest
            usable = None
    sizes = np.mean(np.asarray(fit_rows, dtype=float), axis=0)
    points = []
    for j, fraction in enumerate(fractions):
        if usable is not None:
            lo, hi = float(np.quantile(usable[:, j], lo_q)), float(np.quantile(usable[:, j], hi_q))
            lo, hi = min(lo, float(point[j])), max(hi, float(point[j]))
        else:
            lo = hi = float(point[j])
        points.append(SizeScore(fraction=fraction, fit_rows=int(round(sizes[j])), score=float(point[j]), low=lo, high=hi))
    if usable is None:
        return points, None
    gain = sign * (float(point[-1]) - float(point[-2]))
    gains = sign * (usable[:, -1] - usable[:, -2])
    lo, hi = float(np.quantile(gains, lo_q)), float(np.quantile(gains, hi_q))
    lo, hi = min(lo, gain), max(hi, gain)
    doubling = math.log2(1.0 / fractions[-2])
    return points, Slope(gain=gain, low=lo, high=hi, per_doubling=gain / doubling)


def _segments(y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], preds: list, segments: Mapping[str, Sequence[Any]],
              row_loss: Callable[[np.ndarray, np.ndarray], np.ndarray], min_rows: int, limit: int) -> list[tuple[Any, ...]]:
    """The levels whose share of the out-of-fold error (every row, every fold) is at least :data:`SEGMENT_LIFT` times their
    share of the rows, the largest excess first: (column, level, row share, error share, rows, positions per fold)."""
    valid_all = np.concatenate([v for _, v in folds])
    full = np.concatenate([np.asarray(preds[f][-1][0]) for f in range(len(folds))])
    try:
        loss = np.asarray(row_loss(y[valid_all], full), dtype=float)
    except (TypeError, ValueError):
        return []
    total = float(np.nansum(loss))
    if not math.isfinite(total) or total <= 0:
        return []
    found = []
    for column, values in segments.items():
        values = pd.Series(np.asarray(values, dtype=object))
        if len(values) != len(y):
            raise ValueError(f"segment column {column!r} has {len(values)} values for {len(y)} training rows")
        at = values.iloc[valid_all].reset_index(drop=True)
        for level, count in at.dropna().astype(str).value_counts().items():
            if count < min_rows or count > len(at) - min_rows:
                continue
            mask = (at.astype(str) == level).to_numpy() & at.notna().to_numpy()
            row_share, error_share = count / len(at), float(np.nansum(loss[mask])) / total
            if error_share >= SEGMENT_LIFT * row_share:
                found.append((error_share - row_share, column, level, row_share, error_share, int(count)))
    found.sort(key=lambda t: -t[0])
    out = []
    for _, column, level, row_share, error_share, count in found[:limit]:
        selected = [np.flatnonzero(pd.Series(np.asarray(segments[column], dtype=object)[v]).astype(str).to_numpy() == level) for _, v in folds]
        out.append((column, level, round(row_share, 6), round(min(1.0, error_share), 6), count, selected))
    return out


def _plain(answer: str, why: str, metric: str, slope: Optional[Slope], fractions: tuple[float, ...]) -> str:
    step = f"from {fractions[-2]:.0%} to 100% of the training rows"
    if slope is None:
        return f"Cannot tell whether more rows help: {why}."
    moved = f"{metric} moved by {slope.gain:+.4f} {step} (90% interval {slope.low:+.4f} to {slope.high:+.4f}; positive is better)"
    if answer == LIKELY:
        return (f"More rows likely help: {moved}, and {why}. If the curve kept that steepness, doubling the rows would add at most about "
                f"{slope.per_doubling:+.4f}.")
    if answer == UNLIKELY:
        return f"More rows of the same kind are unlikely to help much: {moved}; {why}. Better columns or cleaner labels are the likelier lever."
    return f"Cannot tell whether more rows help: {moved}; {why}. More folds or more rows would narrow it."


def _assumption(how: str, slope: Optional[Slope]) -> str:
    kind = {"time": "the next period like the latest ones", "group": "new groups like the ones already here",
            "stratified": "rows with the same class balance", "random": "rows like these"}[how]
    return ("New rows would come from the same process as the training rows (" + kind + "); the model family and its features stay as "
            "chosen; the curve does not get steeper with more rows (curves usually flatten, so the gain per doubling is an upper guess). "
            "The interval counts the chance in the validation rows and in which rows were cut, not a change in the world; the folds are "
            "fixed, and the holdout was never read.")
