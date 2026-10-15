"""Segmentation (package 16.6): groups in a table that has no outcome to predict.

There is no ground truth: a segmentation is judged by whether it comes back when the data is resampled (stability),
how separated its groups are (silhouette), and whether each group can be described in the data's own words. Nothing
here says a segment is "right"; :func:`segment` says so in its record.

- :func:`prepare`: the recorded preprocessing: numbers log-scaled when skewed and positive, then standardised;
  categories one-hot, each scaled so a column weighs the same however many levels it has; identifiers and constant
  columns left out. Every choice is in the record.
- :func:`segment`: for each number of segments k, k-means and a Gaussian mixture refitted on bootstrap resamples; the
  stability of k is the adjusted Rand agreement between each resample's labels and the full fit's, read on every
  row, with a 90% interval over resamples; the silhouette on a fixed sample, which must rise above the silhouette the
  same method reaches on data with no groups (one Gaussian with the data's spread, a flat box over its ranges). k is
  chosen among the stable, separated ones by silhouette; the others are rejected with their numbers. Columns that no
  stable, separated grouping of a first pass uses are left out, and the search runs again.
  HDBSCAN is run as a density-based view: its number of groups and its share of rows called noise.
- Profiles: each segment's size and the columns that set it apart (standardised mean differences, the share of each
  category), in the data's own columns and units.
- An optional outcome named by the person: its rate per segment with Wilson intervals; it describes the segments, it
  does not validate them.

Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

DENSITY_ROWS = 5000
SEPARATION = 0.02  # how far a silhouette must rise above the best one on data with no groups (a blob, a flat box)
PLAUSIBLE = 0.05  # silhouettes this close to the best stable, separated one are plausible groupings
UNUSED = 0.02  # a column whose spread lies less than this much between the groups of every stable, separated first-pass grouping is left out
NO_GROUND_TRUTH = ("There is no ground truth for these segments: they are groups the data supports when resampled, described in its own "
                   "columns. Whether they are useful is a business judgement, not a score.")


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def prepare(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> tuple[np.ndarray, dict[str, Any]]:
    """The matrix the methods read, and the record of every choice made to build it."""
    cols = list(columns) if columns is not None else list(frame.columns)
    record: dict[str, Any] = {"numbers": {}, "categories": {}, "left_out": {}, "spans": {}, "fitted": {}}
    parts = []
    width = 0
    n = len(frame)
    for c in cols:
        s = frame[c]
        if s.nunique(dropna=True) <= 1:
            record["left_out"][c] = "the same value on every row"
            continue
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_integer_dtype(s) and s.nunique() == n:
            record["left_out"][c] = "a whole number different on every row: an identifier"
            continue
        if pd.api.types.is_numeric_dtype(s):
            x = s.astype(float).fillna(s.median())
            how = [f"{int(s.isna().sum())} missing filled with the median {s.median():.4g}"] if s.isna().any() else []
            if x.min() >= 0 and x.skew() > 1:
                x = np.log1p(x)
                how.append("log(1 + x), because it is skewed and never negative")
            mu, sd = x.mean(), x.std() or 1.0
            record["fitted"][c] = {"median": float(s.median()), "log": bool(how and how[-1].startswith("log")), "mean": float(mu), "sd": float(sd)}
            parts.append(((x - mu) / sd).to_numpy()[:, None])
            record["spans"][c] = [width, width + 1]
            width += 1
            how.append(f"standardised (mean {mu:.4g}, sd {sd:.4g})")
            record["numbers"][c] = "; ".join(how)
        else:
            if s.nunique() > 0.5 * n:
                record["left_out"][c] = "a different value on most rows: an identifier or free text"
                continue
            s = s.astype(object).where(s.notna(), "(missing)")  # a missing value is its own level, said so
            levels = s.astype(str).value_counts().index[:20]
            record["fitted"][c] = {"levels": [str(v) for v in levels]}
            onehot = np.column_stack([(s.astype(str) == v).to_numpy(float) for v in levels])
            parts.append(onehot / np.sqrt(len(levels)))  # each column weighs about one number, whatever its levels
            record["spans"][c] = [width, width + len(levels)]
            width += len(levels)
            record["categories"][c] = (f"one-hot on {len(levels)} levels, scaled by 1/sqrt(levels)"
                                       + ("; missing values are the level (missing)" if "(missing)" in set(levels) else "")
                                       + ("; levels past the 20 most common are no level" if s.nunique() > 20 else ""))
    if not parts:
        raise ValueError("no column to segment on: every column is constant, an identifier or free text")
    return np.hstack(parts), record


def transform(frame: pd.DataFrame, record: dict[str, Any]) -> np.ndarray:
    """Other rows in the matrix ``prepare`` built (16.15 B: held-out rows): its columns in its order, each with the
    median, log, mean and sd, or the levels, it fitted on its own rows; nothing is learned from these rows."""
    parts = []
    for c in sorted(record["spans"], key=lambda name: record["spans"][name][0]):
        f, s = record["fitted"][c], frame[c]
        if "levels" in f:
            s = s.astype(object).where(s.notna(), "(missing)")
            parts.append(np.column_stack([(s.astype(str) == v).to_numpy(float) for v in f["levels"]]) / np.sqrt(len(f["levels"])))
        else:
            x = pd.to_numeric(s, errors="coerce").astype(float).fillna(f["median"])
            if f["log"]:
                x = np.log1p(x.clip(lower=0))
            parts.append(((x - f["mean"]) / f["sd"]).to_numpy()[:, None])
    return np.hstack(parts)


def wilson(hits: int, n: int, z: float = 1.959964) -> Optional[list[float]]:
    if not n:
        return None
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [_r(max(0.0, centre - half), 3), _r(min(1.0, centre + half), 3)]


def _fit(method: str, k: int, X: np.ndarray, seed: int):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
    # reg_covar keeps a component's variance away from zero on one-hot columns (standardised data: 0.05 of a unit)
    return GaussianMixture(n_components=k, covariance_type="diag", reg_covar=0.05, n_init=2, random_state=seed).fit(X)


def stability(method: str, k: int, X: np.ndarray, *, resamples: int = 20, seed: int = 0) -> dict[str, Any]:
    """Adjusted Rand agreement, on every row, between the full fit's labels and each bootstrap refit's labels."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    full = _fit(method, k, X, seed).predict(X)
    scores = []
    for b in range(resamples):
        rows = rng.integers(0, len(X), len(X))
        scores.append(adjusted_rand_score(full, _fit(method, k, X[rows], seed + b + 1).predict(X)))
    return {"mean": _r(np.mean(scores), 3), "ci90": [_r(np.quantile(scores, 0.05), 3), _r(np.quantile(scores, 0.95), 3)], "labels": full}


def profiles(frame: pd.DataFrame, labels: np.ndarray, columns: Sequence[str], top: int = 4) -> list[dict[str, Any]]:
    """Each segment in the data's own columns: its size, the numbers that set it apart (median, against the whole table's,
    by standardised difference) and its most over-represented categories."""
    out = []
    for g in sorted(set(labels) - {-1}):
        inside = labels == g
        diffs = []
        for c in columns:
            s = frame[c]
            if pd.api.types.is_numeric_dtype(s) and s.std() > 0:
                d = (s[inside].mean() - s.mean()) / s.std()
                diffs.append((abs(d), {"column": c, "segment_median": _r(s[inside].median(), 4), "overall_median": _r(s.median(), 4),
                                       "difference_sd": _r(d, 2)}))
            elif not pd.api.types.is_numeric_dtype(s) and s.nunique() <= 20:
                share_in = s[inside].astype(str).value_counts(normalize=True)
                share_all = s.astype(str).value_counts(normalize=True)
                lift = (share_in - share_all.reindex(share_in.index)).sort_values(ascending=False)
                if len(lift):
                    v = lift.index[0]
                    diffs.append((abs(lift.iloc[0]) * 3, {"column": c, "value": v, "segment_share": _r(share_in[v], 3), "overall_share": _r(share_all[v], 3)}))
        diffs.sort(key=lambda t: -t[0])
        out.append({"segment": int(g), "rows": int(inside.sum()), "share": _r(inside.mean(), 3), "sets_it_apart": [d for _, d in diffs[:top]]})
    return out


def segment(frame: pd.DataFrame, *, columns: Optional[Sequence[str]] = None, ks: Sequence[int] = range(2, 9), min_stability: float = 0.75,
            resamples: int = 20, outcome: Optional[str] = None, seed: int = 0, sample: int = 3000) -> dict[str, Any]:
    """Segments with their stability, separation and profiles, the choice of k explained, and the no-ground-truth limit."""
    cols = [c for c in (columns if columns is not None else frame.columns) if c != outcome]
    ks = [k for k in ks if 2 <= k < len(frame)]
    if not ks:
        raise ValueError(f"{len(frame)} rows cannot be split into the numbers of segments asked for")
    found = search(frame, cols, ks, min_stability=min_stability, resamples=resamples, seed=seed, sample=sample)
    out = _report(frame, found["final"], found["kept"], ks, min_stability, outcome)
    out["preprocessing"]["left_out"].update(found["unused"])
    out["passes"] = {"first": {"columns": cols, "chosen": found["first"]["chosen"]}, "unused_columns": found["unused"]}
    return out


def search(frame: pd.DataFrame, cols: Sequence[str], ks: Sequence[int], *, min_stability: float = 0.75, resamples: int = 20, seed: int = 0,
           sample: int = 3000) -> dict[str, Any]:
    """The choice of segments, never reading an outcome: a first pass over ``cols``, the columns no plausible grouping
    uses left out, and a second pass without them (16.15 B: the engine's models stage). ``final`` is the pass chosen
    from, ``kept`` its columns."""
    cols = list(cols)
    first = _search(frame, cols, ks, min_stability, resamples, seed, sample)
    unused = {}
    good = [v for v in first["table"].values() if v["stable"] and v["separated"]]
    top = max((v["silhouette"] for v in good), default=None)
    # the plausible groupings: stable, separated, and nearly as well separated as the best (a finer one that splits a
    # flat column into slices is far below it)
    candidates = [v for v in good if v["silhouette"] >= top - PLAUSIBLE] if good else []
    if candidates:  # a column that no plausible grouping uses only blurs the distances
        X0, rec0 = prepare(frame, cols)
        for c, (a, b) in rec0["spans"].items():
            shares = [_between_share(X0[:, a:b], first["labels_of"][(v["method"], v["k"])]) for v in candidates]
            if max(shares) < UNUSED:
                unused[c] = (f"none of the {len(candidates)} plausible groupings of the first pass explains more than {max(shares):.1%} "
                             "of its spread: no group uses it")
    kept = [c for c in cols if c not in unused]
    final = _search(frame, kept, ks, min_stability, resamples, seed, sample) if unused and len(kept) >= 2 else first
    return {"first": first, "final": final, "kept": kept if final is not first else cols, "unused": unused}


def _null_silhouette(method: str, k: int, X: np.ndarray, seed: int, draws: int = 3) -> Optional[float]:
    """The best silhouette the same method reaches with k groups on data that has none: one Gaussian with the data's
    mean and spread, and a uniform box over each column's range (1st to 99th percentile). A single column with two
    humps is groups; a blob or a flat square is not."""
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed + 7)
    cov = np.atleast_2d(np.cov(X.T)) + 1e-9 * np.eye(X.shape[1])
    low, high = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)
    best = None
    for draw in range(2 * draws):
        Z = rng.multivariate_normal(X.mean(axis=0), cov, len(X)) if draw % 2 == 0 else rng.uniform(low, high, X.shape)
        lab = _fit(method, k, Z, seed).predict(Z)
        if len(set(lab)) > 1:
            v = silhouette_score(Z, lab)
            best = v if best is None else max(best, v)
    return best


def _between_share(block: np.ndarray, labels: np.ndarray) -> float:
    """The share of a column's spread (its block of the matrix) that lies between the groups."""
    total = ((block - block.mean(axis=0)) ** 2).sum()
    if total <= 0:
        return 0.0
    between = sum((labels == g).sum() * ((block[labels == g].mean(axis=0) - block.mean(axis=0)) ** 2).sum() for g in set(labels))
    return float(between / total)


def _search(frame, cols, ks, min_stability, resamples, seed, sample) -> dict[str, Any]:
    from sklearn.metrics import silhouette_score

    X, record = prepare(frame, cols)
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(X), min(sample, len(X)), replace=False)
    table: dict[str, dict[str, Any]] = {}
    labels_of: dict[tuple[str, int], np.ndarray] = {}
    for method in ("kmeans", "mixture"):
        for k in ks:
            st = stability(method, k, X, resamples=resamples, seed=seed)
            labels_of[(method, k)] = st.pop("labels")
            sil = silhouette_score(X[probe], labels_of[(method, k)][probe]) if len(set(labels_of[(method, k)][probe])) > 1 else None
            null = _null_silhouette(method, k, X[probe], seed)
            table[f"{method}:{k}"] = {"method": method, "k": k, "stability": st, "silhouette": _r(sil, 3), "null_silhouette": _r(null, 3),
                                      "stable": st["ci90"][0] is not None and st["ci90"][0] >= min_stability,
                                      "separated": sil is not None and null is not None and sil >= null + SEPARATION}
    stable = [v for v in table.values() if v["stable"] and v["separated"]]
    chosen = max(stable, key=lambda v: (v["silhouette"] or -1, -v["k"])) if stable else None
    return {"X": X, "record": record, "table": table, "labels_of": labels_of, "chosen": chosen}


def _report(frame, found, cols, ks, min_stability, outcome) -> dict[str, Any]:
    from sklearn.cluster import HDBSCAN

    X, record, table, labels_of, chosen = found["X"], found["record"], found["table"], found["labels_of"], found["chosen"]
    rejected = [{"method": v["method"], "k": v["k"], "why": f"the lower end of its 90% stability interval, {v['stability']['ci90'][0]} "
                 f"(mean {v['stability']['mean']}), is below {min_stability}: resampled data does not give the same {v['k']} groups"}
                for v in table.values() if not v["stable"]]
    not_separated = [{"method": v["method"], "k": v["k"], "why": f"silhouette {v['silhouette']} is not {SEPARATION} above the "
                      f"{v['null_silhouette']} the same method reaches on data with no groups (one Gaussian blob, a flat box)"}
                     for v in table.values() if v["stable"] and not v["separated"]]
    rows = np.random.default_rng(0).choice(len(X), min(len(X), DENSITY_ROWS), replace=False)  # density on a sample: it grows fast
    dl = HDBSCAN(min_cluster_size=max(15, len(rows) // 50)).fit(X[rows]).labels_ if len(rows) >= 30 else np.full(len(rows), -1)
    out: dict[str, Any] = {"preprocessing": record, "candidates": table, "rejected_as_unstable": rejected, "rejected_as_not_separated": not_separated,
                           "density_view": {"method": "hdbscan", "groups": int(len(set(dl) - {-1})), "noise_share": _r(np.mean(dl == -1), 3),
                                            "rows": int(len(rows))},
                           "limit": NO_GROUND_TRUTH, "min_stability": min_stability}
    if chosen is None:
        out.update({"chosen": None, "why": f"no number of segments in {list(ks)} is both stable (lower bound {min_stability}) and better "
                                           "separated than on data with no groups: the data shows none"})
        return out
    labels = labels_of[(chosen["method"], chosen["k"])]
    out.update({"chosen": {"method": chosen["method"], "k": chosen["k"], "stability": chosen["stability"], "silhouette": chosen["silhouette"],
                           "null_silhouette": chosen["null_silhouette"]},
                "why": f"the best separated (silhouette {chosen['silhouette']}, against {chosen['null_silhouette']} on data with no groups) of the stable choices", "labels": labels.tolist(),
                "profiles": profiles(frame, labels, cols)})
    described = describe_outcome(frame, labels, outcome)
    if described is not None:
        out["outcome"] = described
    return out


def describe_outcome(frame: pd.DataFrame, labels: Any, outcome: Optional[str]) -> Optional[dict[str, Any]]:
    """A later outcome per segment, with its interval: it describes the segments, it never makes them right (None
    without an outcome)."""
    labels = np.asarray(labels)
    out: dict[str, Any] = {}
    if outcome is not None and outcome in frame.columns:
        raw = frame[outcome]
        y, said = pd.to_numeric(raw, errors="coerce"), None
        values = sorted(raw.dropna().astype(str).unique())
        if y.notna().sum() == 0 and len(values) == 2:  # a yes/no outcome in words: the later level in order counts as 1, said so
            y, said = (raw.astype(str) == values[1]).astype(float).where(raw.notna()), f"{values[1]!r} counted as 1, {values[0]!r} as 0"
        if y.notna().sum() == 0:
            return {"column": outcome, "note": f"not a number and not two levels ({len(values)} levels): it cannot describe the segments"}
        binary = set(y.dropna().unique()) <= {0, 1}
        by = {}
        for g in sorted(set(labels)):
            v = y[labels == g].dropna()
            if binary:
                by[str(g)] = {"rate": _r(v.mean(), 3), "ci95": wilson(int(v.sum()), int(len(v))), "rows": int(len(v))}
            else:
                half = 1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                by[str(g)] = {"mean": _r(v.mean(), 4), "ci95": [_r(v.mean() - half, 4), _r(v.mean() + half, 4)], "rows": int(len(v))}
        return {"column": outcome, "note": "describes the segments; it does not make them right", "by_segment": by, **({"levels": said} if said else {})}
    if outcome is not None:
        return {"column": outcome, "note": f"{outcome!r} is not a column of the table: nothing describes the segments"}
    return None
