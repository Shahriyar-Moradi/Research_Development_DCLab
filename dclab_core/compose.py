"""Composition (package 24.5): the leaves' scores combined, chosen on validation, scored once on the holdout against the best single leaf.

A complex problem (24.1) is solved leaf by leaf (24.3, 24.4); each leaf hands a score per row for the validation period and for the holdout, and none of
them scores the holdout itself. This module is what puts them together, with the rules a composition must keep:

* **Candidates.** Every leaf alone, and the ways to combine them: ``rank_mean`` and ``rank_max`` (each leaf's score turned into its rank among the rows of
  the period, so that scores on different scales mix), ``rank_weighted`` (the same, each leaf weighted by what it did alone on validation) and, when the map has
  a gate, ``gate:<a>><b>@<share>`` (a row counts for leaf ``b`` only when leaf ``a`` puts it in its top share).
* **Chosen on validation.** The highest recall at the review budget (or average precision when there is no budget) on the validation rows; a tie goes to the
  simpler candidate (a leaf alone before a fusion). So a composition that does not help is not chosen: the best leaf alone is, and the result says so.
* **Scored once.** ``read_holdout`` scores the chosen composition and every leaf on the holdout, once, with an interval from a bootstrap over clusters of rows
  (rows of one account or one biker move together), and the *paired* difference of the composition against the best leaf **chosen on validation** (never the
  best on the holdout). Nothing about the holdout reaches ``choose``.

The labels are the cases someone confirmed: partial and biased towards what is easy to see, so a recall here is a recall of what was confirmed, never of all fraud.
Pure functions on arrays and series; no model, no store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd

Metric = Literal["recall_at_budget", "average_precision"]
FUSIONS = ("rank_mean", "rank_max", "rank_weighted")
GATE_SHARES = (0.1, 0.25, 0.5)
MIN_POSITIVES = 5          # fewer confirmed cases in a period: a choice on it is a coin toss, and the result says so
BOOTSTRAPS = 500
MIN_DRAWS = 50             # fewer resamples than this and the interval says nothing: no gain is claimed

if TYPE_CHECKING:  # the contracts these functions return live in programme (24.5) and are imported where they are built
    from dclab_core.programme import Choice, Estimate, HoldoutResult


# ------------------------------------------------------------------------------------------------------------------------ building blocks

def budget_rows(rows: int, share: Optional[float]) -> int:
    """The review budget as a number of rows: a share of the rows (0 < share < 1), a count, or 1% by default."""
    if share is None or share <= 0:
        return max(1, int(round(0.01 * rows)))
    return max(1, int(round(share * rows))) if share < 1 else max(1, min(rows, int(share)))


def percentile(score: pd.Series) -> pd.Series:
    """A score as its rank among the rows of its own period, in (0, 1]: the same scale for a detector's distance and a classifier's probability."""
    return score.rank(method="average", pct=True)


def fuse(scores: Mapping[str, pd.Series], how: str, weights: Optional[Mapping[str, float]] = None) -> pd.Series:
    """One score per row from the leaves' scores (each a series on the same index)."""
    if how not in FUSIONS:
        raise ValueError(f"unknown fusion {how}")
    if not scores:
        raise ValueError("a fusion needs at least one leaf")
    ranks = pd.concat({k: percentile(v) for k, v in scores.items()}, axis=1)
    if how == "rank_max":
        return ranks.max(axis=1)
    if how == "rank_mean":
        return ranks.mean(axis=1)
    w = pd.Series({k: float((weights or {}).get(k, 0.0)) for k in ranks.columns})
    if w.sum() <= 0:
        w[:] = 1.0
    return (ranks * (w / w.sum())).sum(axis=1)


def recall_at(score: np.ndarray, y: np.ndarray, k: int) -> float:
    """The share of the confirmed cases in the top ``k`` rows. Rows that tie at the cut share the places that are left: the result is the expected share over every
    order of the tied rows, never the order the rows happen to be in (a leaf with few non-zero scores does not win or lose by where its zeros sit)."""
    positives = float(y.sum())
    n = len(score)
    k = max(0, min(int(k), n))
    if positives == 0 or k == 0:
        return 0.0
    if k >= n:
        return 1.0
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
    cut = np.partition(score, n - k)[n - k]
    above, tied = score > cut, score == cut
    gain = float(y[above].sum()) + (k - int(above.sum())) * float(y[tied].sum()) / int(tied.sum())
    return gain / positives


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.0
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, score))


def metric_value(score: np.ndarray, y: np.ndarray, metric: Metric, share: Optional[float]) -> float:
    if metric == "average_precision":
        return average_precision(score, y)
    return recall_at(score, y, budget_rows(len(score), share))


# ------------------------------------------------------------------------------------------------------------------------ candidates

def _name_gate(by: str, leaf: str, share: float) -> str:
    return f"gate:{by}>{leaf}@{int(round(share * 100))}"      # in percent, a whole number: a name has no dot to break a record path


def candidate_names(leaves: Sequence[str], gates: Sequence[tuple[str, str]] = (), fused: Optional[Sequence[str]] = None) -> list[str]:
    """The candidates in the order of their simplicity: a leaf alone, then the fusions (of the ``fused`` leaves, by default all), then the gates."""
    names = list(leaves)
    if len([l for l in (fused if fused is not None else leaves) if l in leaves]) > 1:
        names += list(FUSIONS)
    for by, leaf in gates:
        if by in leaves and leaf in leaves and by != leaf:
            names += [_name_gate(by, leaf, s) for s in GATE_SHARES]
    return names


def build(name: str, scores: Mapping[str, pd.Series], weights: Optional[Mapping[str, float]] = None, fused: Optional[Sequence[str]] = None) -> pd.Series:
    """One candidate's score on the given rows (validation or holdout), the same way on both. A fusion combines the ``fused`` leaves only (default: all)."""
    if name in scores:
        return scores[name]
    if name in FUSIONS:
        use = {k: v for k, v in scores.items() if fused is None or k in fused}
        return fuse(use, name, weights)
    if name.startswith("gate:"):
        head, share = name[5:].rsplit("@", 1)
        by, leaf = head.split(">", 1)
        if by not in scores or leaf not in scores:
            raise ValueError(f"{name} names a leaf that is not here")
        return gated(scores[leaf], scores[by], float(share) / 100.0)
    raise ValueError(f"unknown candidate {name}")


def gated(score: pd.Series, by: pd.Series, share: float) -> pd.Series:
    """``score``, ranked, for the rows ``by`` puts in its top ``share``; every other row ranks below all of them, in its own order (a budget larger than the
    gate still fills from the rest)."""
    inside = percentile(by) > 1.0 - share
    r = percentile(score)
    return r.where(inside, r - 2.0)


def choose(validation: Mapping[str, pd.Series], y: pd.Series, *, metric: Metric = "recall_at_budget", share: Optional[float] = None,
           gates: Sequence[tuple[str, str]] = (), leaf_order: Optional[Sequence[str]] = None, fused: Optional[Sequence[str]] = None) -> "Choice":
    """Judge every candidate on the validation rows and pick the best; a tie goes to the simpler one (a leaf alone, then a fusion, then a gate). The best leaf alone, the
    bar to beat, is taken among every leaf given; the fusions combine only the ``fused`` ones (default: all)."""
    from dclab_core.programme import Choice

    leaves = [l for l in (leaf_order or list(validation)) if l in validation]
    if not leaves:
        raise ValueError("no leaf has a validation score")
    fused = [l for l in (fused if fused is not None else leaves) if l in leaves]
    index = validation[leaves[0]].index
    yv = y.reindex(index).fillna(0).astype(int).to_numpy()
    alone = {l: metric_value(validation[l].to_numpy(dtype=float), yv, metric, share) for l in leaves}
    weights = {l: max(alone[l], 1e-6) for l in leaves}
    warnings = []
    positives = int(yv.sum())
    if positives < MIN_POSITIVES:
        warnings.append(f"Only {positives} confirmed case(s) among the validation rows: the choice below is close to a coin toss.")
    scored: dict[str, float] = {}
    for name in candidate_names(leaves, gates, fused):
        scored[name] = alone[name] if name in alone else metric_value(build(name, validation, weights, fused).to_numpy(dtype=float), yv, metric, share)
    order = list(scored)
    best_leaf = max(leaves, key=lambda l: (alone[l], -order.index(l)))
    chosen = max(order, key=lambda n: (round(scored[n], 9), -order.index(n)))
    rule = ("the highest " + ("recall at the review budget" if metric == "recall_at_budget" else "average precision") +
            " on the validation rows; a tie to the simpler candidate (a leaf alone, then a fusion, then a gate)")
    if chosen in alone and len(leaves) > 1:
        warnings.append(f"No combination beat {chosen} alone on validation: the composition is that leaf.")
    return Choice(chosen=chosen, candidates={k: round(v, 6) for k, v in scored.items()}, best_leaf=best_leaf, best_leaf_score=round(alone[best_leaf], 6),
                  weights={k: round(v, 6) for k, v in weights.items()}, fused=fused, metric=metric, budget_share=share,
                  budget_rows=budget_rows(len(yv), share) if metric == "recall_at_budget" else 0, positives=positives, rule=rule, warnings=warnings)


# ------------------------------------------------------------------------------------------------------------------------ the one holdout read

def _clusters(clusters: Optional[pd.Series], index: pd.Index) -> tuple[list[np.ndarray], Optional[str]]:
    if clusters is None:
        return [np.array([i]) for i in range(len(index))], None
    codes = pd.factorize(clusters.reindex(index).astype(object).where(clusters.reindex(index).notna(), None).astype(str))[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds), str(clusters.name) if clusters.name else "cluster"


def read_holdout(holdout: Mapping[str, pd.Series], y: pd.Series, choice: "Choice", *, clusters: Optional[pd.Series] = None, bootstraps: int = BOOTSTRAPS,
                 seed: int = 0) -> "HoldoutResult":
    """Score the chosen composition and every leaf on the holdout, once. The comparison is with the best leaf *chosen on validation*; the interval is a
    bootstrap over clusters of rows (resampled together, the same resamples for every score, so the difference is paired)."""
    from dclab_core.programme import Estimate, HoldoutResult

    leaves = list(holdout)
    index = holdout[leaves[0]].index
    yh = y.reindex(index).fillna(0).astype(int).to_numpy()
    scores = {l: holdout[l].to_numpy(dtype=float) for l in leaves}
    composed = build(choice.chosen, holdout, choice.weights, choice.fused or None).to_numpy(dtype=float)
    metric, share = choice.metric, choice.budget_share
    best = scores[choice.best_leaf]
    groups, cluster_key = _clusters(clusters, index)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"composed": [], "best": [], **{l: [] for l in leaves}}
    skipped = 0
    for _ in range(max(bootstraps, 0)):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[g] for g in pick])
        yb = yh[rows]
        if yb.sum() == 0:
            skipped += 1
            continue
        draws["composed"].append(metric_value(composed[rows], yb, metric, share))
        draws["best"].append(metric_value(best[rows], yb, metric, share))
        for l in leaves:
            draws[l].append(metric_value(scores[l][rows], yb, metric, share))

    def estimate(value: float, key: str) -> "Estimate":
        d = np.asarray(draws[key])
        if not len(d):
            return Estimate(value=round(value, 4), low=round(value, 4), high=round(value, 4))
        low, high = np.percentile(d, [2.5, 97.5])
        return Estimate(value=round(value, 4), low=round(float(low), 4), high=round(float(high), 4))

    point_c = metric_value(composed, yh, metric, share)
    point_b = metric_value(best, yh, metric, share)
    diff_draws = np.asarray(draws["composed"]) - np.asarray(draws["best"])
    if len(diff_draws):
        lo, hi = np.percentile(diff_draws, [2.5, 97.5])
    else:
        lo = hi = point_c - point_b
    difference = Estimate(value=round(point_c - point_b, 4), low=round(float(lo), 4), high=round(float(hi), 4))
    limits = ["The cases are those someone confirmed, partial and biased towards what is easy to see: this is a recall of the confirmed cases, not of all fraud.",
              "The interval is a bootstrap over " + (f"clusters of rows sharing a {cluster_key}" if cluster_key else "rows") +
              "; rows that move together through another key are not resampled together."]
    if int(yh.sum()) < MIN_POSITIVES:
        limits.append(f"Only {int(yh.sum())} confirmed case(s) among the holdout rows: the interval is wide and means little.")
    if skipped:
        limits.append(f"{skipped} of {max(bootstraps, 0)} resamples held no confirmed case and were left out of the interval.")
    if len(diff_draws) < MIN_DRAWS:
        limits.append(f"Only {len(diff_draws)} resamples were drawn (at least {MIN_DRAWS} are needed): the interval is the point itself, and no gain is claimed.")
    if choice.chosen == choice.best_leaf:
        limits.append("The composition chosen on validation is the best leaf alone: the difference is zero by construction.")
    return HoldoutResult(reads=1, metric=metric, chosen=choice.chosen, best_leaf=choice.best_leaf, budget_share=share,
                         budget_rows=budget_rows(len(yh), share) if metric == "recall_at_budget" else 0, rows=int(len(yh)), positives=int(yh.sum()),
                         composed=estimate(point_c, "composed"), leaves={l: estimate(metric_value(scores[l], yh, metric, share), l) for l in leaves},
                         best_leaf_estimate=estimate(point_b, "best"), difference=difference, beats_best_leaf=bool(len(diff_draws) >= MIN_DRAWS and difference.low > 0),
                         cluster_key=cluster_key, bootstraps=len(draws["composed"]), limits=limits)


__all__ = ["Metric", "FUSIONS", "GATE_SHARES", "budget_rows", "percentile", "fuse", "gated", "build", "candidate_names", "choose",
           "read_holdout", "recall_at", "MIN_POSITIVES", "MIN_DRAWS", "average_precision", "metric_value"]
