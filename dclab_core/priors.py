"""Invented classification tables from priors, the pretraining data of an in-context tabular model (26.6b).

A TabPFN-style model learns to predict a new table's labels in one pass by being trained on very many invented tables
whose features and labels come from random generating processes (its "prior"). This module writes those tables, never
trains anything. Three priors, each a family of processes a real table could plausibly come from:

- ``scm``: a random structural causal model: a random directed acyclic network of nodes, each a random nonlinear
  function of its parents plus noise; features and the outcome are nodes picked from it (some features are causes of
  the outcome, some its effects, some unrelated), as in the published prior;
- ``trees``: the outcome a random sum of shallow axis-aligned splits on independent features (the shape boosted trees win on);
- ``linear``: the outcome a noisy logistic function of a random linear combination (the shape a linear model wins on).

Every table then gets what real tables have: some features cut into categories (stored as their integer code, marked in
``categorical``), missing cells in some columns, classes from 2 to ``max_classes`` with uneven balance, and a few
pure-noise columns. A table is (``X`` float32, ``y`` int64, ``categorical`` bool per column) and its ``meta`` (the
prior, its parameters, sizes, balance; never anything from outside). The same seed always makes the same table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

VERSION = "priors_v1"
PRIORS = ("scm", "trees", "linear")
ACTIVATIONS = ("tanh", "relu", "sin", "square", "abs", "identity")


@dataclass
class Table:
    X: np.ndarray
    y: np.ndarray
    categorical: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0)
    if name == "sin":
        return np.sin(x)
    if name == "square":
        return np.clip(x, -3, 3) ** 2
    if name == "abs":
        return np.abs(x)
    return x


def _standard(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else x * 0


def _scm(rng: np.random.Generator, rows: int, features: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    nodes = features + 1 + int(rng.integers(2, 2 + features))  # more nodes than are shown: hidden causes exist
    noise = float(rng.uniform(0.05, 0.6))
    values = np.zeros((rows, nodes))
    roots = int(rng.integers(1, max(2, nodes // 3) + 1))
    edges = 0
    parents_of: dict[int, np.ndarray] = {}
    for j in range(nodes):
        if j < roots:
            values[:, j] = rng.normal(size=rows) if rng.random() < 0.7 else rng.uniform(-2, 2, size=rows)
            continue
        parents = rng.choice(j, size=int(rng.integers(1, min(j, 4) + 1)), replace=False)
        parents_of[j] = parents
        edges += len(parents)
        weights = rng.normal(size=len(parents))
        act = ACTIVATIONS[int(rng.integers(len(ACTIVATIONS)))]
        values[:, j] = _standard(_act(act, values[:, parents] @ weights)) + rng.normal(scale=noise, size=rows)
    outcome = int(rng.integers(roots, nodes))  # the outcome is never a root: it has causes among the nodes
    others = [j for j in range(nodes) if j != outcome]
    shown = rng.choice(others, size=features, replace=False)
    # which shown features descend from the outcome (its effects: information after it, as the published prior holds them),
    # so a corpus can be filtered for them later
    below = {outcome}
    for j in range(outcome + 1, nodes):
        if set(int(x) for x in parents_of[j]) & below:
            below.add(j)
    effects = [i for i, j in enumerate(shown) if int(j) in below]
    return values[:, shown], values[:, outcome], {"nodes": nodes, "edges": edges, "roots": roots, "noise": round(noise, 3),
                                                  "effects_shown": effects}


def _trees(rng: np.random.Generator, rows: int, features: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X = rng.normal(size=(rows, features))
    trees, depth = int(rng.integers(3, 30)), int(rng.integers(1, 4))
    score = np.zeros(rows)
    used = set()
    for _ in range(trees):
        leaf = np.zeros(rows, dtype=np.int64)
        for d in range(depth):
            col = int(rng.integers(features))
            used.add(col)
            leaf = leaf * 2 + (X[:, col] > rng.normal(scale=0.8)).astype(np.int64)
        score += rng.normal(size=2 ** depth)[leaf]
    noise = float(rng.uniform(0.0, 1.0))
    return X, _standard(score) + rng.normal(scale=noise, size=rows), {"trees": trees, "depth": depth, "used": len(used), "noise": round(noise, 3)}


def _linear(rng: np.random.Generator, rows: int, features: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cov = rng.normal(size=(features, features)) * rng.uniform(0, 0.5)
    X = rng.normal(size=(rows, features)) @ (np.eye(features) + cov)
    w = rng.normal(size=features) * (rng.random(features) < rng.uniform(0.3, 1.0))
    noise = float(rng.uniform(0.1, 1.5))
    return X, _standard(X @ w) + rng.normal(scale=noise, size=rows), {"active": int((w != 0).sum()), "noise": round(noise, 3)}


def sample(seed: int, prior: str | None = None, rows: tuple[int, int] = (64, 1024), features: tuple[int, int] = (2, 30),
           max_classes: int = 5) -> Table:
    """One invented table from ``prior`` (or one drawn by the seed)."""
    rng = np.random.default_rng(seed)
    prior = prior or PRIORS[int(rng.integers(len(PRIORS)))]
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {', '.join(PRIORS)}")
    n = int(np.exp(rng.uniform(np.log(rows[0]), np.log(rows[1]))))
    d = int(rng.integers(features[0], features[1] + 1))
    noise_cols = int(rng.integers(0, max(1, d // 4) + 1))
    real = max(1, d - noise_cols)
    X, latent, params = {"scm": _scm, "trees": _trees, "linear": _linear}[prior](rng, n, real)
    X = np.concatenate([X, rng.normal(size=(n, d - real))], axis=1) if d > real else X
    order = rng.permutation(d)  # the noise columns are not at the end
    X = X[:, order]
    if "effects_shown" in params:  # the outcome's effects, as positions in the shown table
        where = {int(old): new for new, old in enumerate(order)}
        params["effects_shown"] = sorted(where[i] for i in params["effects_shown"])
    k = int(rng.integers(2, max_classes + 1))
    cuts = np.sort(rng.dirichlet(np.ones(k) * rng.uniform(0.6, 3.0)).cumsum()[:-1])  # uneven classes
    y = np.searchsorted(np.quantile(latent, cuts), latent).astype(np.int64)
    if rng.random() < 0.5:  # the class codes in no particular order: an ordinal meaning is not given away
        y = rng.permutation(k)[y]
    categorical = np.zeros(d, dtype=bool)
    for j in np.where(rng.random(d) < rng.uniform(0, 0.4))[0]:
        levels = int(rng.integers(2, 12))
        edges = np.quantile(X[:, j], np.sort(rng.uniform(0, 1, levels - 1)))
        X[:, j] = rng.permutation(levels)[np.searchsorted(edges, X[:, j])]  # codes, not ranks
        categorical[j] = True
    missing_share = float(rng.choice([0.0, 0.0, 0.02, 0.1, 0.3]))
    if missing_share:
        cols = np.where(rng.random(d) < 0.4)[0]
        mask = np.zeros_like(X, dtype=bool)
        mask[:, cols] = rng.random((n, len(cols))) < missing_share
        X[mask] = np.nan
    counts = np.bincount(y, minlength=k)
    meta = {"version": VERSION, "seed": int(seed), "prior": prior, "rows": n, "features": d, "noise_features": int(d - real),
            "classes": k, "class_shares": [round(float(c) / n, 4) for c in counts], "present_classes": int((counts > 0).sum()),
            "categorical": int(categorical.sum()), "missing_share": missing_share,
            "missing_cells": int(np.isnan(X).sum()), "params": params}
    return Table(X.astype(np.float32), y, categorical, meta)


def learnable(table: Table, seed: int = 0) -> dict[str, Any]:
    """How far a quick model gets above the majority class on held-out rows of the table (a check the table carries some
    signal; tables with none are kept too, as a prior holds them, and marked). ``signal`` None: the table could not be checked."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    y = table.y
    counts = np.bincount(y)
    if (counts > 0).sum() < 2 or counts[counts > 0].min() < 2:  # an empty class code is not a class; a class of one row cannot be split
        return {"majority": None, "model": None, "lift": None, "signal": None, "reason": "a class with fewer than 2 rows: not checked"}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    model = HistGradientBoostingClassifier(max_iter=60, categorical_features=table.categorical if table.categorical.any() else None,
                                           random_state=seed)
    model.fit(table.X[tr], y[tr])
    accuracy = float((model.predict(table.X[te]) == y[te]).mean())
    majority = float(np.bincount(y[te]).max() / len(te))
    return {"majority": round(majority, 4), "model": round(accuracy, 4), "lift": round(accuracy - majority, 4),
            "signal": accuracy - majority > 0.02}


__all__ = ["PRIORS", "Table", "VERSION", "learnable", "sample"]
