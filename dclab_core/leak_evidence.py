"""Evidence about columns suspected of leakage by their name (2026-11-19).

A column whose name sounds late (``courier_rating``, ``resolved_by``) may be filled in after the prediction moment, or
may be an honest feature with an unlucky name. When nobody is there to ask, the name alone decided nothing and the column
stayed in the model. The held-out comparison (E2E-004) showed the cost: rules alone claimed macro F1 0.94 on tickets and
got 0.28 on new cases.

This module weighs each suspect on the rows a build may read before its holdout, with three checks:

1. Timing. A time column whose values come after the row's own time is late. A time column whose values are all before
   it existed in time. A column blank far more often on the most recent rows is filled in later.
2. Too good. A model on the column alone scores at least as well as a model on every other column together.
3. With and without. The same model with every clean column, with and without the suspect, on the same folds, so the
   person sees what keeping it out costs.

The decision: late or too good keeps it out; an earlier time keeps it in; no evidence either way follows ``unsure``
(the safe side, out, by default). Every column gets a sentence in plain words for the person. Names, scores and counts
only: never a table's value.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Basis = Literal["late", "too_good", "early", "unsure"]
Decision = Literal["keep_out", "keep_in"]

LATE_SHARE = 0.95  # a time after the row's own time on this share of rows, or more, is late
EARLY_SHARE = 0.99  # a time at or before the row's own time on this share of rows, or more, existed in time
FILLED_LATER = 0.25  # blank on the latest fifth of the rows read this much more often than on the rest
MIN_ROWS = 60  # fewer rows read than this: no model is fitted, only the timing is read
MAX_ROWS = 20000
FOLDS = 3
PARSE_SHARE = 0.9  # a text column whose values read as times on this share or more is a time (a CSV keeps dates as text)
MARGIN_LOWER = 0.10  # too good: better than the rest by 10 % of its error (lower-is-better metrics) ...
MARGIN_HIGHER = 0.15  # ... or by 0.15 of the score (noise reached AUC 0.61 against 0.48 on one seed of 60) (higher-is-better); a tie with weak columns is chance, not evidence


class ColumnEvidence(Contract):
    """One suspect column: what each check found, the decision and why, in plain words."""

    column: str = Field(max_length=1000)
    decision: Decision
    basis: Basis
    why: str = Field(max_length=600, description="For the person: what was found and what it means.")
    late_share: Optional[float] = Field(None, description="Share of rows where its time is after the row's own time.")
    blank_recent: Optional[float] = Field(None, description="How much more often it is blank on the latest rows read.")
    alone: Optional[float] = Field(None, description="Cross-validated score of a model on this column alone.")
    rest: Optional[float] = Field(None, description="The same, on every clean column without it.")
    with_it: Optional[float] = Field(None, description="The same, on every clean column with it.")
    gain: Optional[float] = Field(None, description="How much better the score is with it (positive: it helps).")


class LeakEvidence(Versioned):
    """The suspects weighed on the rows a build reads before its holdout."""

    schema_version: Literal["dclab_core.leak_evidence.v1"] = _version("leak_evidence")
    metric: str
    lower_is_better: bool
    rows_read: int
    by_time: bool
    unsure: Decision
    columns: list[ColumnEvidence] = Field(default_factory=list, max_length=200)


CONTRACTS["leak_evidence"] = LeakEvidence


def _metric(task: str) -> tuple[str, bool]:
    return {"regression": ("mae", True), "binary": ("roc_auc", False)}.get(task, ("macro_f1", False))


def _as_time(s: pd.Series) -> Optional[pd.Series]:
    """The column as times without a zone (UTC), or None when it is not a time. A text column is a time when nearly all
    of its values read as one: a CSV keeps dates as text. A number is never read as a time (an elapsed counter is not a date)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        filled = s.dropna()
        if not len(filled):
            return None
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(s, errors="coerce", utc=True)
        if out[filled.index].notna().mean() < PARSE_SHARE:
            return None
    else:
        return None
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def _margin(reference: Optional[float], lower: bool) -> float:
    return MARGIN_LOWER * abs(reference or 0.0) if lower else MARGIN_HIGHER


def _encode(frame: pd.DataFrame, time_column: Optional[str]) -> pd.DataFrame:
    """Numbers a tree can split: a time as seconds after the row's own time (its lateness is what matters), a category
    as its code. Codes carry no outcome, so reading them on every row read is not fitting on a fold's test rows."""
    out = {}
    base = frame[time_column] if time_column and time_column in frame and pd.api.types.is_datetime64_any_dtype(frame[time_column]) else None
    for name in frame.columns:
        s = frame[name]
        if pd.api.types.is_datetime64_any_dtype(s):
            if base is None or name == time_column:
                continue
            out[name] = (s - base).dt.total_seconds()
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            out[name] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            out[name] = s.astype("string").astype("category").cat.codes.replace(-1, np.nan).astype(float)
    return pd.DataFrame(out, index=frame.index)


def _folds(n: int, by_time: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold, TimeSeriesSplit

    splitter = TimeSeriesSplit(FOLDS) if by_time else KFold(FOLDS, shuffle=True, random_state=0)
    return list(splitter.split(np.arange(n)))


def _score(X: pd.DataFrame, y: pd.Series, task: str, folds: list) -> Optional[float]:
    """The mean score over the folds of a small boosted tree; None when no fold can be scored."""
    each = _scores(X, y, task, folds)
    return float(np.mean([v for v in each if v is not None])) if any(v is not None for v in each) else None


def _scores(X: pd.DataFrame, y: pd.Series, task: str, folds: list) -> list[Optional[float]]:
    """The score on each fold (None where a fold cannot be scored); [] without columns."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

    if X.shape[1] == 0:
        return []
    scores: list[Optional[float]] = []
    for train, test in folds:
        ytr, yte = y.iloc[train], y.iloc[test]
        if task != "regression" and ytr.nunique() < 2:
            scores.append(None)
            continue
        model = (HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier)(max_iter=100, random_state=0)
        model.fit(X.iloc[train], ytr)
        if task == "regression":
            scores.append(mean_absolute_error(yte, model.predict(X.iloc[test])))
        elif task == "binary":
            if yte.nunique() < 2:
                scores.append(None)
                continue
            positive = list(model.classes_).index(sorted(model.classes_)[-1])
            scores.append(roc_auc_score(yte == model.classes_[positive], model.predict_proba(X.iloc[test])[:, positive]))
        else:
            scores.append(f1_score(yte, model.predict(X.iloc[test]), average="macro"))
    return scores


def _every_fold(a: list, b: list, lower: bool) -> bool:
    """``a`` better than ``b`` on every fold both scored: one lucky fold is chance, not evidence."""
    pairs = [(x, z) for x, z in zip(a, b) if x is not None and z is not None]
    return bool(pairs) and all((x < z) if lower else (x > z) for x, z in pairs)


def _better(a: Optional[float], b: Optional[float], lower: bool) -> Optional[float]:
    """How much better ``a`` is than ``b`` (positive: better)."""
    if a is None or b is None:
        return None
    return round((b - a) if lower else (a - b), 4)


def _fmt(v: Optional[float]) -> str:
    return "unknown" if v is None else f"{v:.3f}"


def weigh(frame: pd.DataFrame, target: str, task: str, suspects: Sequence[str], features: Sequence[str],
          time_column: Optional[str] = None, unsure: Decision = "keep_out",
          suspected_because: Optional[Mapping[str, str]] = None) -> LeakEvidence:
    """Weigh each suspect. ``frame``: the table; ``features``: the clean columns the model would read (no suspect, no
    identifier, no proven leak); ``time_column``: the column that orders the rows, if any. Only the rows a build may
    read before its holdout are read (the earliest 75 % by time, else a seeded 75 %, which a random holdout drawn later
    may overlap: ``by_time`` says which). ``suspected_because``: why each column is a suspect, in words, for its sentence."""
    from dclab_core.modifiers import provisional_reading

    metric, lower = _metric(task)
    has_target = frame[target].notna()
    if task == "regression":
        has_target &= pd.to_numeric(frame[target], errors="coerce").notna()
    present = frame.index[has_target]
    times = frame.loc[present, time_column] if time_column and time_column in frame else None
    read, by_time = provisional_reading(present, times)
    rows = frame.loc[read]
    if by_time:
        rows = rows.assign(_order=pd.to_datetime(rows[time_column], errors="coerce") if not pd.api.types.is_numeric_dtype(rows[time_column])
                           else rows[time_column]).sort_values("_order", kind="stable").drop(columns="_order")
    if len(rows) > MAX_ROWS:
        rows = rows.iloc[np.sort(np.random.default_rng(0).choice(len(rows), MAX_ROWS, replace=False))]
    rows = rows.copy()
    for name in dict.fromkeys([*suspects, *([time_column] if time_column else [])]):
        if name in rows.columns and name != target:
            parsed = _as_time(rows[name])
            if parsed is not None:
                rows[name] = parsed
    y = pd.to_numeric(rows[target], errors="coerce") if task == "regression" else rows[target].astype(str)
    suspects = [c for c in dict.fromkeys(suspects) if c in rows.columns and c != target]
    clean = [c for c in dict.fromkeys(features) if c in rows.columns and c not in suspects and c != target]
    encoded = _encode(rows[[*clean, *suspects, *([time_column] if time_column and time_column in rows else [])]], time_column)
    clean = [c for c in clean if c in encoded]
    modelled = len(rows) >= MIN_ROWS
    folds = _folds(len(rows), by_time) if modelled else []
    rest_each = _scores(encoded[clean], y, task, folds) if modelled else []
    rest = float(np.mean([v for v in rest_each if v is not None])) if any(v is not None for v in rest_each) else None
    # a time is compared only with a time: an elapsed counter as the time column would make every date look late
    base_time = rows[time_column] if time_column and time_column in rows and pd.api.types.is_datetime64_any_dtype(rows[time_column]) else None

    out = []
    for name in suspects:
        s = rows[name]
        late_share = early = None
        if base_time is not None and pd.api.types.is_datetime64_any_dtype(s):
            both = s.notna() & base_time.notna()
            if both.sum():
                late_share = round(float((s[both] > base_time[both]).mean()), 4)
                early = float((s[both] <= base_time[both]).mean()) >= EARLY_SHARE
        blank_recent = None
        if by_time and s.isna().any() and len(s) >= 10:
            cut = int(len(s) * 0.8)
            blank_recent = round(float(s.iloc[cut:].isna().mean() - s.iloc[:cut].isna().mean()), 4)
        alone = with_it = shuffled = None
        alone_each: list = []
        shuffled_each: list = []
        if modelled and name in encoded:
            alone_each = _scores(encoded[[name]], y, task, folds)
            alone = float(np.mean([v for v in alone_each if v is not None])) if any(v is not None for v in alone_each) else None
            with_it = _score(encoded[[*clean, name]], y, task, folds)
            # the same column with its values shuffled: what it scores by chance
            mixed = encoded[[name]].assign(**{name: np.random.default_rng(0).permutation(encoded[name].to_numpy())})
            shuffled_each = _scores(mixed, y, task, folds)
            shuffled = float(np.mean([v for v in shuffled_each if v is not None])) if any(v is not None for v in shuffled_each) else None
        gain = _better(with_it, rest, lower)
        alone_vs_rest = _better(alone, rest, lower)
        over_chance = _better(alone, shuffled, lower)
        too_good = (alone_vs_rest is not None and alone_vs_rest > _margin(rest, lower)
                    and over_chance is not None and over_chance > _margin(shuffled, lower)
                    and _every_fold(alone_each, rest_each, lower) and _every_fold(alone_each, shuffled_each, lower))
        cost = "" if gain is None else (f" Keeping it out changes the score from {_fmt(with_it)} to {_fmt(rest)} ({metric}, "
                                        f"{'lower' if lower else 'higher'} is better).")
        if late_share is not None and late_share >= LATE_SHARE:
            basis, why = "late", f"Its time is after the row's own time on {late_share:.0%} of the rows, so it does not exist yet when you predict."
        elif blank_recent is not None and blank_recent >= FILLED_LATER:
            basis, why = "late", f"It is blank much more often on the most recent rows ({blank_recent:+.0%}), as a value filled in later is."
        elif too_good:
            basis, why = "too_good", (f"Alone it predicts better than every other column together ({_fmt(alone)} against {_fmt(rest)} "
                                      f"{metric}), which an honest column rarely does.")
        elif early:
            basis, why = "early", "Its time is at or before the row's own time on every row, so it exists when you predict."
        else:
            because = (suspected_because or {}).get(name) or "its name sounds like it is filled in after the event"
            basis, why = "unsure", f"It was suspected because {because[:200]}, and the data neither proves nor rules that out."
        decision: Decision = "keep_in" if basis == "early" else "keep_out" if basis in ("late", "too_good") else unsure
        action = (" Kept out of the model." if decision == "keep_out" else " Kept in the model.")
        if basis == "unsure":
            action += (" If it is known when you predict, put it back." if decision == "keep_out" else " If it is filled in later, take it out.")
        out.append(ColumnEvidence(column=name, decision=decision, basis=basis, why=(why + action + cost)[:600], late_share=late_share,
                                  blank_recent=blank_recent, alone=None if alone is None else round(alone, 4), rest=None if rest is None else round(rest, 4),
                                  with_it=None if with_it is None else round(with_it, 4), gain=gain))
    return LeakEvidence(metric=metric, lower_is_better=lower, rows_read=int(len(rows)), by_time=bool(by_time), unsure=unsure, columns=out)


def summary(evidence: LeakEvidence) -> list[str]:
    """One line a column, for a report."""
    return [f"`{c.column}`: {c.why}" for c in evidence.columns]


__all__ = ["ColumnEvidence", "LeakEvidence", "weigh", "summary"]
