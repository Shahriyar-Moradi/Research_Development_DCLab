"""The screening cascade (package 30.17, the owner's pattern 3): how a team that can review K cases a day finds most of
the fraud among many rows, stage by stage, and how what it confirms teaches the next stage.

The stages, run day by day as the team would (each day's rows are screened when the day is known, never later):

1. **The wide net:** a row enters when a rule flags it, when a link check ties it to another record (the matching kind's
   exact links: a return by the cashier who sold the bill, the same bill returned before, a near-identical bill minutes
   earlier), or when it is among the day's most unusual ``NET_SHARE`` of rows (a robust z-score whose medians and spreads
   are fitted on the days before, refitted every round) or, once a model is learned, among its highest ``NET_SHARE``.
2. **The queue:** the net ranked by one fused risk; the day's K reviews are the K minus ``audit`` highest, plus a random
   audit sample of the day's other rows (net or not), so what the queue and the net miss can be estimated.
3. **The labels back:** each reviewed row is confirmed by the reviewers (``review``, asked about that row only); every
   ``retrain_days`` a supervised model is fitted on the confirmed rows so far and joins the fused risk once it has seen
   ``MIN_POSITIVES`` of each class.
4. **The case summary:** each queued case gets a short summary for its reviewer: the facts behind its rank, in the
   template's words or a model's (``summarize``), shown only the columns screening reads, never for a ``private``
   project. It is advisory: it never changes a rank or a label, and it is kept whole with why it was written.

Recall is measured per stage on the latest period (the holdout days): the share of its cases the net holds and the
queue reviews when a replica's truth is given (``measure``, read after the run), and always the audit's estimate of both
(``capacity.audit_recall``), beside the cascade with the learning off. Nothing here reads a row's outcome before the
reviewers confirm it, and a column the caller names in ``keep_out`` (the label, a verdict written after the review, a
rule computed with the future) is refused as a feature, a rule, a link, an anomaly column or a summary's input.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

STAGES = ("net", "queue", "audit")
MIN_POSITIVES = 15   # confirmed cases (and as many confirmed honest rows) before the supervised model joins the risk
NET_SHARE = 0.15     # the anomaly detector's share of a day's rows in the net, and the learned model's (the replica: half the rows, nine cases in ten)


def _rank(x: np.ndarray) -> np.ndarray:
    """A score as its percentile within the day (0 to 1): scores of different stages fused on one scale."""
    if len(x) == 0:
        return x
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (order + 1) / len(x)


def _fit_anomaly(rows: pd.DataFrame, columns: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Each column's median and robust spread, on the rows given (the days before the one screened)."""
    out = {}
    for c in columns:
        x = pd.to_numeric(rows[c], errors="coerce")
        med = float(x.median()) if x.notna().any() else 0.0
        mad = float((x - med).abs().median()) if x.notna().any() else 0.0
        out[c] = (med, mad or 1.0)
    return out


def _anomaly(fitted: Mapping[str, tuple[float, float]], rows: pd.DataFrame) -> np.ndarray:
    """The largest robust z-score of a row's columns."""
    out = np.zeros(len(rows))
    for c, (med, mad) in fitted.items():
        z = np.abs((pd.to_numeric(rows[c], errors="coerce").fillna(med).to_numpy() - med) / (1.4826 * mad))
        out = np.maximum(out, z)
    return out


def _top(scores: np.ndarray, share: float) -> np.ndarray:
    keep = np.zeros(len(scores), dtype=bool)
    if len(scores):
        keep[np.argsort(-scores, kind="stable")[: max(1, int(np.ceil(share * len(scores))))]] = True
    return keep


def _summary(why: Sequence[str]) -> str:
    return "In the queue because " + ("; ".join(why) if why else "of its rank") + "."


def run(frame: pd.DataFrame, *, day: str, key: str, review: Mapping[Any, int], measure: Optional[Mapping[Any, int]] = None,
        rules: Sequence[str] = (), anomaly: Sequence[str] = (), links: Sequence[str] = (), features: Sequence[str] = (),
        keep_out: Sequence[str] = (), k: int = 5, audit: int = 1, holdout_days: int = 90, retrain_days: int = 30, net_share: float = NET_SHARE,
        learn: bool = True, summarize: Optional[Callable[[Mapping[str, Any], Sequence[str]], str]] = None, private: bool = False,
        seed: int = 0) -> dict[str, Any]:
    """The cascade over every day of ``frame`` (one row a case to screen; ``day`` its date column, ``key`` its id).
    ``review``: what a reviewer confirms for a reviewed row (1 a case, 0 not), asked only for the rows reviewed;
    ``measure``: every row's truth, for a replica only (the true recall per stage, after the run); ``rules``, ``links``:
    0/1 columns; ``anomaly``: numeric columns for the detector; ``features``: what the supervised model reads (with the
    rules', the links' and the detector's own scores); ``keep_out``: columns that must never be read (refused if named
    above); ``summarize``: words for a reviewer (a model's), advisory, never asked for a ``private`` project."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    read = [*rules, *anomaly, *links, *features]
    refused = sorted(set(read) & set(keep_out))
    if refused:
        raise ValueError(f"{', '.join(refused)} must never be read (keep_out): not as a rule, a link, an anomaly column or a feature")
    shown = list(dict.fromkeys([key, day, *read]))  # what screening reads: all a summary may be shown
    rng = np.random.default_rng(seed)
    f = frame.reset_index(drop=True).copy()
    days = pd.to_datetime(f[day], errors="coerce").dt.normalize()
    f["__day"] = days
    last = days.max()
    start = last - pd.Timedelta(days=holdout_days - 1)
    f["__anomaly"] = 0.0
    f["__rule"] = f[list(rules)].fillna(0).astype(int).max(axis=1) if rules else 0
    f["__link"] = f[list(links)].fillna(0).astype(int).max(axis=1) if links else 0
    cols = [*features, "__anomaly", "__rule", "__link"]
    reviewed, in_net, queued, audited = (np.zeros(len(f), dtype=bool) for _ in range(4))
    label = np.full(len(f), -1)
    model, fitted, trained_on, rounds, cases = None, None, 0, [], {}
    next_round = days.min() + pd.Timedelta(days=retrain_days)
    for d in sorted(days.dropna().unique()):
        d = pd.Timestamp(d)
        if d >= next_round or fitted is None:  # a round: the detector refitted on the days before, the model on the labels so far
            before = f[days < d]
            fitted = _fit_anomaly(before if len(before) else f[days == d], anomaly) if anomaly else {}  # the first day: its own batch, known at screening
            if learn and d >= next_round:
                seen = reviewed & (label >= 0)
                positives = int((label[seen] == 1).sum())
                if positives >= MIN_POSITIVES and (label[seen] == 0).sum() >= MIN_POSITIVES:
                    X = f.loc[seen, cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
                    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed).fit(X, label[seen])
                    trained_on = int(seen.sum())
                rounds.append({"day": str(d.date()), "confirmed": int(seen.sum()), "positives": positives, "model": model is not None})
            if d >= next_round:
                next_round = d + pd.Timedelta(days=retrain_days)
        idx = np.flatnonzero((days == d).to_numpy())
        rows = f.iloc[idx]
        score = _anomaly(fitted, rows) if anomaly else np.zeros(len(idx))
        f.loc[f.index[idx], "__anomaly"] = score
        rows = f.iloc[idx]
        top_anomaly = _top(score, net_share) if anomaly else np.zeros(len(idx), dtype=bool)
        net = (rows["__rule"].to_numpy() == 1) | (rows["__link"].to_numpy() == 1) | top_anomaly
        risk = (_rank(score) + 2 * rows["__rule"].to_numpy() + 3 * rows["__link"].to_numpy()) / 6
        learned, top_learned = None, np.zeros(len(idx), dtype=bool)
        if model is not None:
            learned = model.predict_proba(rows[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy())[:, 1]
            risk = 0.5 * risk + 0.5 * _rank(learned)
            top_learned = _top(learned, net_share)
            net |= top_learned
        order = [i for i in np.argsort(-risk, kind="stable") if net[i]]
        top = order[: max(k - audit, 0)]
        rest = [i for i in range(len(idx)) if i not in set(top)]
        sample = list(rng.choice(rest, min(audit, len(rest)), replace=False)) if rest and audit else []
        for i in top + sample:
            row = idx[i]
            reviewed[row], label[row] = True, int(review.get(f.at[row, key], 0))  # the reviewer confirms: the only read of an outcome
        for i in top:
            row = idx[i]
            why = [w for w, on in (("a rule flags it", f.at[row, "__rule"] == 1), ("a link ties it to another record", f.at[row, "__link"] == 1),
                                   ("it is among the day's most unusual", top_anomaly[i]),
                                   ("the model learned from confirmed cases ranks it among the day's highest", top_learned[i])) if on]
            text, error, source = _summary(why), None, "template"
            if summarize is not None and not private:
                try:
                    worded = str(summarize({c: f.at[row, c] for c in shown}, why))
                    if worded:
                        text, source = worded, "model"
                except Exception as exc:  # noqa: BLE001 - a summary that fails leaves the template's words: it never changes the queue
                    error = f"{type(exc).__name__}: {exc}"
            cases[f.at[row, key]] = {"day": str(d.date()), "summary": text, "source": source, "why": why, "error": error,
                                     "risk": round(float(risk[i]), 6), "learned": None if learned is None else round(float(learned[i]), 6),
                                     "anomaly": round(float(score[i]), 6)}
        in_net[idx] = net
        queued[idx[top]] = True
        audited[idx[sample]] = True
    hold = (days >= start).to_numpy()
    y = np.asarray([int(measure.get(v, 0)) for v in f[key]]) if measure is not None else None  # the replica's truth, after the run
    cases_in_hold = int(y[hold].sum()) if y is not None else None

    def share(mask: np.ndarray) -> Optional[float]:
        return round(float((mask & hold & (y == 1)).sum()) / cases_in_hold, 4) if cases_in_hold else None

    from .capacity import audit_recall

    seen_label = np.where(label[hold] >= 0, label[hold], 0)
    stratum = np.where(queued, "queue", "rest")[hold]
    by_day = f.loc[hold, "__day"].astype(str).to_numpy()
    looked = (queued | audited)[hold]
    estimate = audit_recall(queued[hold], seen_label, looked, stratum, by_day)
    net_estimate = audit_recall(in_net[hold], seen_label, looked, stratum, by_day)  # the same audit: what the net misses
    reviewed_queue = queued & hold
    precision = float((label[reviewed_queue] == 1).mean()) if reviewed_queue.any() else None  # every queued row is reviewed: its label is known
    rows_hold = int(hold.sum())
    return {"stages": {"net": {"recall": share(in_net), "rows": int((in_net & hold).sum()),
                               "share_of_rows": round(float((in_net & hold).sum()) / rows_hold, 4) if rows_hold else None, "estimated_recall": net_estimate},
                       "queue": {"recall": share(queued), "rows": int((queued & hold).sum()), "precision": round(precision, 4) if precision is not None else None},
                       "audit": {"rows": int((audited & hold).sum()), "estimated_recall": estimate}},
            "holdout": {"from": str(start.date()), "to": str(last.date()), "rows": rows_hold, "cases": cases_in_hold,
                        "reviewed_share": round(float(((queued | audited) & hold).sum()) / rows_hold, 4) if rows_hold else None},
            "rounds": rounds, "learned": model is not None, "trained_on": trained_on, "k": k, "audit": audit, "net_share": net_share,
            "labels": {"confirmed": int((reviewed & (label >= 0)).sum()), "cases_confirmed": int((label == 1).sum())},
            "summaries": {c: v["summary"] for c, v in cases.items()}, "cases": cases, "queued": f.loc[queued, key].tolist(),
            "audited": f.loc[audited, key].tolist(), "label_of": {f.at[i, key]: int(label[i]) for i in np.flatnonzero(reviewed)}, "shown": shown}


__all__ = ["STAGES", "MIN_POSITIVES", "NET_SHARE", "run"]
