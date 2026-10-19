"""Review capacity and noisy labels, measured (package 30.12): the statistics a screening problem is judged by when a team
can check K cases a period and its labels come from several sources of unequal worth.

- ``queue`` and ``kth_threshold``: the K highest scores of each period are the cases a team checks; the threshold is
  the K-th highest score of an average period (the median over periods), the number an artifact records.
- ``precision_at_k``: of the queue's cases with a confirmed label, the share that are cases, with its Wilson interval
  (a queued case without a confirmed label is not counted, and said); with the review's design (which rows were
  reviewed, in which stratum), each reviewed row weighed as in ``audit_recall`` (the queue's reviewed rows were chosen by
  the old rule and the audit, not at random within the queue), the raw counts beside it.
- ``audit_recall``: the share of all cases the queue catches, estimated only from rows reviewed at random within their
  stratum and period (a team's checked rule hits and its random audit of the rest): each reviewed row stands for the
  rows of its stratum and period it was drawn from (``N / n``); a stratum and period with rows but none reviewed cannot
  be estimated and its share of rows is said. The interval is the wider of a bootstrap that draws whole periods and a
  Wilson interval on the weighted cases' effective count (a few cases all caught never read as certain).
- ``rule_columns``: a column holding one value on every row a noisy source labelled, and other values elsewhere: the
  rule that built those labels, which a model would learn back.
- ``agreement``: how often a noisy source's label (a rule's value) is right on the rows a person confirmed, the weight
  its labels deserve, with its Wilson interval.

Only arrays and counts: no table enters here, and nothing here reads a holdout twice (the host passes the rows once).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import wilson

VERSION = "1"


def _arrays(*values: Any) -> list[np.ndarray]:
    return [np.asarray(v) for v in values]


def queue(scores: Sequence[float], periods: Sequence[Any], k: int) -> np.ndarray:
    """The K highest scores of each period (ties broken by position): the cases a team checks."""
    scores, periods = _arrays(scores, periods)
    out = np.zeros(len(scores), dtype=bool)
    frame = pd.DataFrame({"s": scores, "p": periods, "i": np.arange(len(scores))})
    for _, g in frame.groupby("p", sort=False):
        top = g.sort_values(["s", "i"], ascending=[False, True]).head(max(0, int(k)))
        out[top["i"].to_numpy()] = True
    return out


def kth_threshold(scores: Sequence[float], periods: Sequence[Any], k: int) -> Optional[float]:
    """The score of the K-th case on an average period: each period's K-th highest score (a period of fewer rows, its
    lowest), the median over periods."""
    scores, periods = _arrays(scores, periods)
    if not len(scores) or k < 1:
        return None
    frame = pd.DataFrame({"s": scores, "p": periods})
    kth = frame.groupby("p")["s"].apply(lambda s: float(np.sort(s.to_numpy())[::-1][min(int(k), len(s)) - 1]))
    return round(float(np.median(kth.to_numpy())), 6)


def _weights(periods: np.ndarray, stratum: np.ndarray, reviewed: np.ndarray) -> tuple[np.ndarray, float]:
    """Each row's weight (``N / n`` of its period and stratum when reviewed, 0 otherwise) and the share of rows in a cell
    with none reviewed."""
    frame = pd.DataFrame({"p": periods, "s": stratum, "r": reviewed})
    cells = frame.groupby(["p", "s"])["r"].agg(["size", "sum"])
    ratio = {cell: row["size"] / row["sum"] for cell, row in cells.iterrows() if row["sum"] > 0}
    w = np.array([ratio.get((p_, s_), 0.0) if r_ else 0.0 for p_, s_, r_ in zip(periods, stratum, reviewed)], dtype=float)
    return w, float(cells[cells["sum"] == 0]["size"].sum()) / len(periods) if len(periods) else 0.0


def precision_at_k(scores: Sequence[float], periods: Sequence[Any], k: int, y: Sequence[float], reviewed: Optional[Sequence[bool]] = None,
                   stratum: Optional[Sequence[Any]] = None) -> dict[str, Any]:
    """Of the queue's cases with a confirmed label (``y`` 1 or 0; NaN: no confirmed label), the share that are cases;
    weighed by the review's design when ``reviewed`` and ``stratum`` are given."""
    scores, periods, y = _arrays(scores, periods, y)
    yf = y.astype(float)
    q = queue(scores, periods, k)
    known = q & ~np.isnan(yf)
    hits = int(np.nansum(yf[known]))
    n = int(known.sum())
    out = {"k": int(k), "periods": int(len(set(periods.tolist()))), "queued": int(q.sum()), "queued_confirmed": n, "hits": hits,
           "precision": round(hits / n, 4) if n else None, "precision_ci": wilson(hits, n),
           "unconfirmed_share": round(1 - n / q.sum(), 4) if q.sum() else None}
    if reviewed is not None and stratum is not None and n:
        w, _ = _weights(periods, np.asarray(stratum), np.asarray(reviewed).astype(bool) & ~np.isnan(yf))
        wq = w * q
        total = float(wq.sum())
        if total > 0:
            p = float((wq * np.nan_to_num(yf)).sum()) / total
            effective = total ** 2 / float((wq ** 2).sum())
            low, high = _wilson(p, effective)
            out.update({"raw_precision": out["precision"], "raw_precision_ci": out["precision_ci"], "precision": round(p, 4),
                        "precision_ci": [round(low, 4), round(high, 4)], "effective_queued": round(effective, 1),
                        "weighed": "each reviewed row by N/n of its period and stratum"})
    return out


def _recall(caught: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[float]:
    total = float((w * y).sum())
    return float((w * y * caught).sum()) / total if total > 0 else None


def audit_recall(caught: Sequence[bool], y: Sequence[float], reviewed: Sequence[bool], stratum: Sequence[Any], periods: Sequence[Any],
                 draws: int = 1000, seed: int = 0) -> dict[str, Any]:
    """The share of all cases ``caught`` (a queue, as booleans over every row) catches, from the rows ``reviewed`` at
    random within their stratum and period (``y`` their confirmed label), with a bootstrap over periods."""
    caught, y, reviewed, stratum, periods = _arrays(caught, y, reviewed, stratum, periods)
    caught, reviewed = caught.astype(bool), reviewed.astype(bool)
    frame = pd.DataFrame({"p": periods, "s": stratum, "r": reviewed})
    cells = frame.groupby(["p", "s"])["r"].agg(["size", "sum"])
    weight_of = {cell: (row["size"] / row["sum"]) for cell, row in cells.iterrows() if row["sum"] > 0}
    blind = cells[cells["sum"] == 0]["size"].sum()
    idx = np.flatnonzero(reviewed & ~np.isnan(y.astype(float)))
    w = np.array([weight_of[(periods[i], stratum[i])] for i in idx], dtype=float)
    yy, cc, pp = y.astype(float)[idx], caught[idx].astype(float), periods[idx]
    point = _recall(cc, yy, w)
    out: dict[str, Any] = {"recall": round(point, 4) if point is not None else None, "reviewed": int(len(idx)), "reviewed_cases": int(yy.sum()),
                           "estimated_cases": round(float((w * yy).sum()), 1), "unsampled_share": round(float(blind) / len(periods), 4) if len(periods) else None,
                           "interval": "a bootstrap drawing whole periods, 95%"}
    rng = np.random.default_rng(seed)
    days = np.unique(pp)
    by_day = {d: np.flatnonzero(pp == d) for d in days}
    values = []
    for _ in range(draws if len(days) > 1 else 0):
        pick = np.concatenate([by_day[d] for d in rng.choice(days, len(days), replace=True)])
        r = _recall(cc[pick], yy[pick], w[pick])
        if r is not None:
            values.append(r)
    boot = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else None
    wy = w * yy
    effective = float(wy.sum() ** 2 / (wy ** 2).sum()) if (wy ** 2).sum() > 0 else 0.0  # Kish's effective count of the weighted cases
    out["effective_cases"] = round(effective, 1)
    band = _wilson(point, effective) if point is not None and effective > 0 else None
    missed = _missed_bound(float((w * yy * cc).sum()), caught, y, reviewed)  # 30.17: what the uncaught rows' audit allows, beside the bootstrap
    if boot is None and band is None and missed is None:
        out["recall_ci"] = [None, None]
    else:
        lows, highs = [x[0] for x in (boot, band, missed) if x], [x[1] for x in (boot, band, missed) if x]
        out["recall_ci"] = [round(min(lows), 4), round(max(highs), 4)]
    return out


def _missed_bound(found: float, caught: np.ndarray, y: np.ndarray, reviewed: np.ndarray) -> Optional[tuple[float, float]]:
    """Recall's interval from the cases the uncaught rows may hold (the review of 30.17: an audit that finds no case among
    the rows a queue left does not say they hold none). ``found``: the caught cases, weighted as the point estimate weighs
    them; the uncaught rows' share of cases is bounded by Wilson's interval on the audited uncaught rows (a random draw
    among them), times how many uncaught rows there are."""
    yy = np.nan_to_num(y.astype(float))
    left = ~caught
    audited_left = left & reviewed
    if found <= 0 or not audited_left.any():
        return None
    n = int(audited_left.sum())
    low, high = _wilson(float(yy[audited_left].sum()) / n, n)
    rows_left = float(left.sum())
    return (float(found / (found + rows_left * high)), float(found / (found + rows_left * low)))


def _wilson(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson's interval for a share ``p`` of ``n`` cases (``n`` may be an effective, fractional count)."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rule_columns(frame: pd.DataFrame, source: Sequence[Any], noisy: Iterable[str], exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    """The columns that hold one value on every row a noisy source labelled and other values on the other rows: the rule
    that built those labels (each with the value and its share of the rest)."""
    source = pd.Series(list(source), index=frame.index).astype(str)
    out = []
    skip = set(exclude)
    for name in noisy:
        rows = source == name
        if rows.sum() < 5 or rows.all():
            continue
        for column in frame.columns:
            if column in skip:
                continue
            values = frame.loc[rows, column].dropna()
            if len(values) < rows.sum() * 0.99 or values.nunique() != 1:
                continue
            value = values.iloc[0]
            rest = frame.loc[~rows, column]
            share = float((rest == value).mean()) if len(rest) else 1.0
            if share < 0.5:  # the value marks the source's rows, not every row
                out.append({"source": name, "column": str(column), "value": value.item() if hasattr(value, "item") else value,
                            "share_elsewhere": round(share, 4)})
    return out


def agreement(rule_says: Sequence[bool], y: Sequence[float]) -> dict[str, Any]:
    """On the rows a person confirmed (``y`` 1 or 0), how often a noisy source's positive (``rule_says``) is right."""
    rule_says, y = _arrays(rule_says, y)
    known = rule_says.astype(bool) & ~np.isnan(y.astype(float))
    right, n = int(np.nansum(y.astype(float)[known])), int(known.sum())
    return {"checked": n, "right": right, "agreement": round(right / n, 4) if n else None, "agreement_ci": wilson(right, n)}


def summary(scores: Sequence[float], periods: Sequence[Any], k: int, y: Sequence[float], reviewed: Sequence[bool],
            stratum: Sequence[Any], draws: int = 1000, seed: int = 0) -> Mapping[str, Any]:
    """A queue of K a period judged both ways: its precision on confirmed labels and its recall from the audit."""
    q = queue(scores, periods, k)
    return {"k": int(k), "kth_score_holdout": kth_threshold(scores, periods, k),
            "precision_at_k": precision_at_k(scores, periods, k, y, reviewed, stratum),
            "audit_recall": audit_recall(q, y, reviewed, stratum, periods, draws=draws, seed=seed),
            "note": "Recall is estimated only from the rows reviewed at random within their stratum and period. The K-th score of "
                    "the holdout's average period is a check, never the operating threshold (that is set on training rows)."}


__all__ = ["VERSION", "queue", "kth_threshold", "precision_at_k", "audit_recall", "rule_columns", "agreement", "summary"]
