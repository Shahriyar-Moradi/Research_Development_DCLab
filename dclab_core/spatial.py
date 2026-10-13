"""Places (package 16.10): coordinates as features and as a reason to split by area.

Rows that sit near each other often have similar targets (house prices in a district, demand in a neighbourhood). A
random split then puts a row's neighbours in training and the row in the test, and the score measures how well the
model remembers an area, not how it does in a new one. This module gives any kind with coordinates:

- :func:`project` and :func:`blocks`: coordinates in kilometres (equirectangular around the data's middle) and square
  grid cells of a chosen size, as a group for the split;
- :func:`morans_i`: spatial autocorrelation of a value over each row's nearest neighbours, with a permutation p-value;
- :func:`correlation_range`: the distance within which values stay alike, from pairs of rows by distance band; the
  block size follows it, so a block is wider than the range;
- :func:`neighbourhood_counts`: how many rows of the fitted set lie within a radius of each row (a row never counts
  itself); to be fitted on a fold's training rows (not yet a recipe of the studio's engine: a host calls it per fold);
- :func:`split_optimism`: the same model scored with random folds and with folds of whole blocks: the gap is how much a
  random split flatters it here.

Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

KM_PER_DEGREE = 111.32


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def project(lat: Sequence[float], lon: Sequence[float], centre: Optional[tuple[float, float]] = None) -> np.ndarray:
    """(x, y) in km around ``centre`` (the data's middle by default); fine for a city or a country, not a continent."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if np.isfinite(lon).any():
        wrapped = lon % 360
        if np.nanmax(wrapped) - np.nanmin(wrapped) < np.nanmax(lon) - np.nanmin(lon):
            lon = wrapped  # places across the antimeridian (179.9 and -179.9) are near, not 360 degrees apart
            if centre is not None:
                centre = (centre[0], centre[1] % 360)
    c_lat, c_lon = centre if centre is not None else (float(np.nanmedian(lat)), float(np.nanmedian(lon)))
    return np.column_stack([(lon - c_lon) * KM_PER_DEGREE * np.cos(np.radians(c_lat)), (lat - c_lat) * KM_PER_DEGREE])


def blocks(lat: Sequence[float], lon: Sequence[float], size_km: float, centre: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Square cells of ``size_km`` as labels ("x_y"); a row without coordinates gets "none"."""
    xy = project(lat, lon, centre)
    ok = np.isfinite(xy).all(axis=1)
    cells = np.floor(np.where(ok[:, None], xy, 0) / size_km).astype(int)
    return np.where(ok, np.char.add(np.char.add(cells[:, 0].astype(str), "_"), cells[:, 1].astype(str)), "none")


def _neighbours(xy: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(xy) - 1)
    return NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]  # each row's k others


def morans_i(values: Sequence[float], lat: Sequence[float], lon: Sequence[float], *, k: int = 8, permutations: int = 199,
             seed: int = 0) -> dict[str, Any]:
    """Moran's I over each row's ``k`` nearest neighbours (equal weights): near 0 when nearby values are unrelated, up to 1
    when they are alike; the p-value from shuffling the values over the places."""
    v = np.asarray(values, dtype=float)
    xy = project(lat, lon)
    ok = np.isfinite(v) & np.isfinite(xy).all(axis=1)
    v, xy = v[ok], xy[ok]
    if len(v) < 20 or np.std(v) == 0:
        return {"i": None, "p_value": None, "rows": int(len(v))}
    nb = _neighbours(xy, k)
    z = v - v.mean()

    def stat(zz):
        return float(np.sum(zz * zz[nb].mean(axis=1)) / np.sum(zz * zz))
    observed = stat(z)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(z)) for _ in range(permutations)])
    return {"i": _r(observed, 3), "p_value": _r((1 + np.sum(null >= observed)) / (permutations + 1), 4), "k": int(nb.shape[1]), "rows": int(len(v))}


def correlation_range(values: Sequence[float], lat: Sequence[float], lon: Sequence[float], *, bands: int = 12, pairs: int = 200_000,
                      seed: int = 0, below: float = 0.1) -> dict[str, Any]:
    """The distance within which values stay alike: the correlation of standardised values over random pairs of rows, by
    distance band; the range is the first band whose correlation falls below ``below``."""
    v = np.asarray(values, dtype=float)
    xy = project(lat, lon)
    ok = np.isfinite(v) & np.isfinite(xy).all(axis=1)
    v, xy = v[ok], xy[ok]
    z = (v - v.mean()) / (v.std() or 1.0)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(v), pairs), rng.integers(0, len(v), pairs)
    keep = a != b
    d = np.linalg.norm(xy[a[keep]] - xy[b[keep]], axis=1)
    prod = z[a[keep]] * z[b[keep]]
    edges = np.quantile(d, np.linspace(0, 1, bands + 1))
    table, range_km = [], None
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (d >= lo) & (d < hi)
        corr = float(prod[inside].mean()) if inside.any() else float("nan")
        table.append({"from_km": _r(lo, 2), "to_km": _r(hi, 2), "correlation": _r(corr, 3)})
        if range_km is None and corr < below:
            range_km = float(hi)
    span = float(np.ptp(xy, axis=0).max()) if len(xy) else 0.0
    return {"range_km": _r(range_km if range_km is not None else span, 2), "span_km": _r(span, 2), "bands": table}


def neighbourhood_counts(fit_lat: Sequence[float], fit_lon: Sequence[float], lat: Sequence[float], lon: Sequence[float], radius_km: float,
                         *, same_rows: bool = False, centre: Optional[tuple[float, float]] = None) -> np.ndarray:
    """For each row, how many fitted rows lie within ``radius_km`` (fitted rows only: call it with the training rows as the
    fitted set). ``same_rows``: the rows are the fitted rows themselves, and a row never counts itself."""
    from sklearn.neighbors import NearestNeighbors

    centre = centre or (float(np.nanmedian(fit_lat)), float(np.nanmedian(fit_lon)))
    fit, query = project(fit_lat, fit_lon, centre), project(lat, lon, centre)
    found = NearestNeighbors(radius=radius_km).fit(fit).radius_neighbors(query, return_distance=False)
    return np.array([len(f) - (1 if same_rows else 0) for f in found])


def split_optimism(X: pd.DataFrame, y: Sequence[float], lat: Sequence[float], lon: Sequence[float], *, block_km: float, task: str,
                   folds: int = 5, seed: int = 0) -> dict[str, Any]:
    """The same gradient-boosted model scored with random folds and with folds of whole blocks: random minus blocked is
    how much a random split flatters the model on these places (ROC-AUC for binary, R2 for regression)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score

    y = np.asarray(y)
    groups = blocks(lat, lon, block_km)
    model = (HistGradientBoostingClassifier if task == "binary" else HistGradientBoostingRegressor)(max_iter=150, random_state=seed)
    scoring = "roc_auc" if task == "binary" else "r2"
    random_cv = (StratifiedKFold if task == "binary" else KFold)(folds, shuffle=True, random_state=seed)
    n_groups = len(set(groups))
    if n_groups < folds:
        raise ValueError(f"{n_groups} blocks of {block_km} km: too few for {folds} folds; use smaller blocks")
    random_score = cross_val_score(model, X, y, cv=random_cv, scoring=scoring)
    blocked = cross_val_score(model, X, y, cv=GroupKFold(folds), groups=groups, scoring=scoring)
    return {"metric": scoring, "random": _r(random_score.mean()), "blocked": _r(blocked.mean()), "optimism": _r(random_score.mean() - blocked.mean()),
            "blocks": int(n_groups), "block_km": _r(block_km, 2), "folds": folds}
