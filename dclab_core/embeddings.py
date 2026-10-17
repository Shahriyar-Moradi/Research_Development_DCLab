"""Images and long text as features (package 16.12): an encoder turns each item into a vector, cached by the item's
content, then any tabular model reads the vectors.

The encoder is injected (a host's pretrained model run locally, or the simple local encoders here); dclab_core never
trains a large network and never downloads one. What this module owns:

- :func:`content_key`: the SHA-256 of an item's bytes (an image file's bytes, a text's UTF-8): the cache key, so the
  same content is encoded once whatever its name or row.
- :class:`Cache`: vectors on disk by content key and encoder name (one ``.npy`` per item); a changed encoder never
  reads another encoder's vectors.
- :func:`encode`: every item through the cache; the record says how many were encoded and how many came from the cache.
- :func:`near_duplicates`: pairs of items closer than a cosine similarity across two sides of a split (the same scene
  photographed twice, a text copied with a new title): each such pair lets a model recognise rather than predict.
- :func:`calibrate`: the near-copy threshold for an encoder, from the similarity of items and slightly changed copies
  of them against the similarity of different items (or the finding that this encoder cannot tell them apart).
- :func:`evaluate`: a logistic (ridge) model on the vectors with folds grouped by source (scene, document source,
  author), the near duplicates across folds counted, and a holdout of whole sources read once. Other tabular models on
  the vectors are a host's choice; this one is the measured baseline.
- Two local encoders that need no download, for runs without a pretrained one: :class:`HashingText` (word and
  word-pair counts hashed into a fixed width) and :class:`RandomFeatures` (fixed random projections of small image
  patches, pooled): honest baselines, not pretrained models, and named as such in every record.

Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

MAX_PAIRS = 2000
THRESHOLD = 0.98  # the default cosine similarity of a near copy; calibrate() gives one for an encoder


class Encoder(Protocol):
    name: str
    pretrained: bool

    def encode(self, items: Sequence[Any]) -> np.ndarray: ...


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _bytes(item: Any) -> bytes:
    """An item's content as bytes, tagged by type: an array (or anything with ``__array__``, a PIL image) by its values,
    shape and dtype; a file path by the file's bytes; bytes as they are; a text as UTF-8. Anything else is refused,
    never keyed by its printed form."""
    if isinstance(item, Path):
        return b"file:" + item.read_bytes()
    if isinstance(item, (bytes, bytearray)):
        return b"bytes:" + bytes(item)
    if isinstance(item, str):
        return b"text:" + item.encode("utf-8")
    if isinstance(item, np.ndarray) or hasattr(item, "__array__") or hasattr(item, "__array_interface__"):
        a = np.ascontiguousarray(np.asarray(item))
        return b"array:" + str((a.shape, a.dtype.str)).encode() + a.tobytes()
    raise TypeError(f"an item to encode must be an array, an image, a path, bytes or a text, not {type(item).__name__}")


def content_key(item: Any) -> str:
    return hashlib.sha256(_bytes(item)).hexdigest()


class Cache:
    """Vectors on disk: ``root/<encoder name>/<content key>.npy``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, encoder: str, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", encoder)[:80]
        return self.root / safe / f"{key}.npy"

    def get(self, encoder: str, key: str) -> Optional[np.ndarray]:
        path = self._path(encoder, key)
        return np.load(path) if path.exists() else None

    def put(self, encoder: str, key: str, vector: np.ndarray) -> None:
        import os
        import tempfile

        path = self._path(encoder, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")  # one temp file per writer: two workers never share one
        with os.fdopen(handle, "wb") as f:
            np.save(f, np.asarray(vector, dtype=np.float32))
        os.replace(tmp, path)


def encode(items: Sequence[Any], encoder: Encoder, cache: Optional[Cache] = None, batch: int = 64) -> tuple[np.ndarray, dict[str, Any]]:
    """Every item's vector, from the cache when its content was encoded before by this encoder."""
    keys = [content_key(i) for i in items]
    vectors: dict[str, np.ndarray] = {}
    for k in set(keys):
        hit = cache.get(encoder.name, k) if cache is not None else None
        if hit is not None:
            vectors[k] = hit
    todo = [(k, i) for k, i in dict(zip(keys, items)).items() if k not in vectors]
    for start in range(0, len(todo), batch):
        part = todo[start:start + batch]
        out = encoder.encode([i for _, i in part])
        for (k, _), v in zip(part, out):
            vectors[k] = np.asarray(v, dtype=np.float32)
            if cache is not None:
                cache.put(encoder.name, k, vectors[k])
    record = {"encoder": encoder.name, "pretrained": bool(getattr(encoder, "pretrained", False)), "items": len(items),
              "distinct_contents": len(set(keys)), "encoded_now": len(todo), "from_cache": len(set(keys)) - len(todo)}
    widths = {vectors[k].shape[-1] for k in keys}
    if len(widths) > 1:
        raise ValueError(f"vectors of different widths {sorted(widths)} under one encoder name: the cache holds another encoder's vectors")
    return np.vstack([vectors[k] for k in keys]) if keys else np.zeros((0, 0)), record


def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


EXACT_UP_TO = 20_000  # items compared pair by pair; above, candidate pairs come from random-hyperplane hashing
BLOCK_CELLS = 40_000_000  # similarities held at once (160 MB of float32), whatever the number of items
LSH_BITS, LSH_TABLES = 16, 24  # bits per signature and signatures per item (``hashed_recall``: 0.9999 at cosine 0.98, 0.87 at 0.9)


def hashed_recall(threshold: float, bits: int = LSH_BITS, tables: int = LSH_TABLES) -> float:
    """The probability that a pair exactly at ``threshold`` shares a signature in at least one table (a pair above it
    is found more often): one bit agrees with probability 1 - angle / pi."""
    agree = 1.0 - float(np.arccos(np.clip(threshold, -1.0, 1.0))) / np.pi
    return 1.0 - (1.0 - agree ** bits) ** tables


def near_duplicates(X: np.ndarray, folds: Sequence[Any], *, threshold: float = THRESHOLD, block: Optional[int] = None,
                    method: str = "auto", seed: int = 0) -> dict[str, Any]:
    """Pairs of items on different sides (different fold labels) whose vectors' cosine similarity is at least
    ``threshold``: counted, and listed (row positions and similarity, at most ``MAX_PAIRS``).

    ``exact`` compares every pair, in blocks that hold at most ``BLOCK_CELLS`` similarities; ``hashed`` (``auto`` above
    ``EXACT_UP_TO`` items) compares only the pairs that share a random-hyperplane signature in one of ``LSH_TABLES``
    tables, then their exact cosine: every pair it counts is a real one, and a pair near the threshold can be missed
    (the result says which method ran)."""
    U = _unit(np.asarray(X, dtype=np.float32))
    f = np.asarray(folds)
    n = len(U)
    if method == "auto":
        method = "exact" if n <= EXACT_UP_TO else "hashed"
    if method == "hashed":
        return _hashed_pairs(U, f, threshold, seed)
    block = block or max(1, min(n, BLOCK_CELLS // max(n, 1)))
    pairs, examples = 0, []
    for lo in range(0, n, block):
        sim = U[lo:lo + block] @ U.T
        a, b = np.nonzero(sim >= threshold)
        a = a + lo
        keep = (a < b) & (f[a] != f[b])
        pairs += int(keep.sum())
        if len(examples) < MAX_PAIRS:
            examples += [(int(i), int(j), _r(sim[i - lo, j], 4)) for i, j in zip(a[keep][:MAX_PAIRS], b[keep][:MAX_PAIRS])]
    return {"threshold": threshold, "pairs_across_sides": pairs, "pairs": examples[:MAX_PAIRS], "method": "exact"}


BIG_BUCKET = 500  # a bucket this large (copies, boilerplate, empty texts) is compared exactly, in blocks, not pair by pair in Python


def _hashed_pairs(U: np.ndarray, f: np.ndarray, threshold: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    weights = 1 << np.arange(LSH_BITS, dtype=np.int64)
    candidates: set[tuple[int, int]] = set()
    crowded = np.zeros(len(U), dtype=bool)  # items of a bucket too large to list: all their pairs are compared exactly below
    for _ in range(LSH_TABLES):
        planes = rng.standard_normal((U.shape[1], LSH_BITS)).astype(np.float32)
        codes = ((U @ planes) > 0).astype(np.int64) @ weights
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        cuts = np.flatnonzero(np.diff(sorted_codes)) + 1
        for bucket in np.split(order, cuts):
            if len(bucket) < 2:
                continue
            if len(bucket) > BIG_BUCKET:
                crowded[bucket] = True
                continue
            bucket = np.sort(bucket)
            sides = f[bucket]
            for k, i in enumerate(bucket[:-1]):
                others = bucket[k + 1:][sides[k + 1:] != sides[k]]
                candidates.update((int(i), int(j)) for j in others if not (crowded[i] and crowded[j]))
    pairs, examples = 0, []
    if crowded.any():  # every pair inside the crowded items, exactly, counted and never listed twice
        inside = np.flatnonzero(crowded)
        exact = near_duplicates(U[inside], f[inside], threshold=threshold, method="exact")
        pairs += exact["pairs_across_sides"]
        examples += [(int(inside[a]), int(inside[b]), sim) for a, b, sim in exact["pairs"]]
        candidates = {(i, j) for i, j in candidates if not (crowded[i] and crowded[j])}
    for i, j in sorted(candidates):
        sim = float(U[i] @ U[j])
        if sim >= threshold:
            pairs += 1
            if len(examples) < MAX_PAIRS:
                examples.append((i, j, _r(sim, 4)))
    return {"threshold": threshold, "pairs_across_sides": pairs, "pairs": examples[:MAX_PAIRS], "method": "hashed",
            "candidates_compared": len(candidates), "compared_exactly": int(crowded.sum()), "recall_at_threshold": _r(hashed_recall(threshold), 4)}


def perturb_image(item: Any, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(item, dtype=np.float32)
    return a + rng.normal(0, 0.03 * (float(np.ptp(a)) or 1.0), a.shape).astype(np.float32)  # the same picture taken again


def perturb_text(item: Any, rng: np.random.Generator) -> str:
    return "Sent again. " + str(item)  # the same text with a new first line


def calibrate(items: Sequence[Any], encoder: Encoder, perturb: Any, *, sample: int = 60, seed: int = 0,
              sources: Optional[Sequence[Any]] = None) -> dict[str, Any]:
    """The similarity this encoder gives a near copy (an item and a slightly changed copy of it) against the similarity
    of two different items: the threshold lies between them (the 5th percentile of copies and the 99.9th of different
    pairs), three quarters of the way towards the copies. When they overlap the encoder cannot tell near copies apart,
    and the record says so. ``sources`` (26.8): two items of one source (two shots of a scene) are not counted as
    different items: they may be the very copies to find."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), min(sample, len(items)), replace=False)
    base = _unit(encoder.encode([items[i] for i in pick]))
    copy = _unit(encoder.encode([perturb(items[i], rng) for i in pick]))
    near = np.sum(base * copy, axis=1)
    others = base @ base.T
    upper = np.triu_indices(len(pick), 1)
    apart = others[upper]
    if sources is not None:
        src = np.asarray(sources, dtype=object)[pick]
        apart = apart[src[upper[0]] != src[upper[1]]]
    low, high = float(np.quantile(near, 0.05)), float(np.quantile(apart, 0.999)) if len(apart) else 0.0
    return {"encoder": encoder.name, "copies_5th_percentile": _r(low, 4), "different_999th_percentile": _r(high, 4),
            # three quarters of the way to the copies: a pair must be nearly as alike as a copy, not merely closer than strangers
            "separable": low > high, "threshold": _r(high + 0.75 * (low - high), 4) if low > high else None}


def evaluate(X: np.ndarray, y: Sequence[Any], sources: Sequence[Any], *, task: str = "binary", folds: int = 5, holdout: float = 0.2,
             seed: int = 0, threshold: float = THRESHOLD) -> dict[str, Any]:
    """A logistic (or ridge) model on the vectors: a holdout of whole sources read once, folds grouped by source on the
    rest; near duplicates counted across the holdout line and across folds; the same model with random folds for
    comparison (the gap is what recognising near copies is worth)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score

    y = np.asarray(y)
    groups = np.asarray(sources).astype(str)
    if len(set(groups)) < folds + 2:
        raise ValueError(f"{len(set(groups))} sources: too few to keep whole sources apart in {folds} folds and a holdout")
    train, test = next(GroupShuffleSplit(1, test_size=holdout, random_state=seed).split(X, y, groups))
    model = LogisticRegression(max_iter=3000, C=1.0) if task != "regression" else Ridge(alpha=1.0)
    multi = task == "multiclass"
    scoring = "accuracy" if multi else ("roc_auc" if task == "binary" else "r2")
    grouped = cross_val_score(model, X[train], y[train], cv=GroupKFold(folds), groups=groups[train], scoring=scoring)
    shuffled = cross_val_score(model, X[train], y[train], cv=KFold(folds, shuffle=True, random_state=seed), scoring=scoring)
    fold_of = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(folds).split(X[train], y[train], groups[train])):
        fold_of[va] = k
    random_fold = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(X[train])):
        random_fold[va] = k
    model.fit(X[train], y[train])  # the holdout's rows are never fitted
    scores = model.predict_proba(X[test])[:, 1] if task == "binary" else model.predict(X[test])
    if task == "binary":
        held = roc_auc_score(y[test], scores)
    elif multi:
        held = accuracy_score(y[test], scores)
    else:
        held = r2_score(y[test], scores)
    side = np.where(np.isin(np.arange(len(y)), test), "holdout", "train")
    across = near_duplicates(X, side, threshold=threshold)
    return {"model": "logistic regression" if task != "regression" else "ridge regression", "fitted_rows": int(len(train)),
            "train_rows": train.tolist(), "holdout_rows": test.tolist(), "holdout_predictions": [_r(v, 6) for v in np.ravel(scores)],
            "caveat": (f"{across['pairs_across_sides']} near copies straddle the holdout line: its score may be helped by them"
                       if across["pairs_across_sides"] else None),
            "metric": scoring, "grouped_folds": _r(grouped.mean()), "random_folds": _r(shuffled.mean()), "holdout": _r(held),
            "sources": int(len(set(groups))), "holdout_sources": int(len(set(groups[test]))),
            "sources_on_both_sides": int(len(set(groups[test]) & set(groups[train]))),
            "near_duplicates_across_holdout": across,
            "near_duplicates_across_folds": near_duplicates(X[train], fold_of, threshold=threshold),
            "near_duplicates_across_random_folds": near_duplicates(X[train], random_fold, threshold=threshold)}


# ---------------------------------------------------------------------------------------------- local encoders, no download
class HashingText:
    """Word and word-pair counts hashed into ``width`` columns, then L2-normalised: a baseline, not a pretrained model."""

    pretrained = False

    def __init__(self, width: int = 2048):
        self.width = width
        self.name = f"hashing-text-{width}"

    def encode(self, items: Sequence[Any]) -> np.ndarray:
        out = np.zeros((len(items), self.width), dtype=np.float32)
        for n, text in enumerate(items):
            words = re.findall(r"[a-z0-9']+", str(text).lower())
            for token in words + [a + " " + b for a, b in zip(words, words[1:])]:
                out[n, int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % self.width] += 1.0
        return _unit(np.log1p(out))


def pixel_columns(columns: Sequence[Any], prefix: str) -> list[str]:
    """26.8: an image table's pixel columns in pixel order, by the number after the prefix (``pixel2`` before ``pixel10``:
    names sorted as text would scramble an unpadded image); a column with the prefix and no number after it is refused."""
    found = [str(c) for c in columns if str(c).startswith(prefix)]
    bad = [c for c in found if not c[len(prefix):].isdigit()]
    if bad:
        raise ValueError(f"{bad[:3]} start with {prefix!r} but no pixel number follows: a pixel column is the prefix and its number")
    return sorted(found, key=lambda c: int(c[len(prefix):]))


def images_of(table: pd.DataFrame, pixels: Sequence[str]) -> list[np.ndarray]:
    """Each row's pixels as a square grey image (the side is the square root of the pixel count)."""
    side = int(round(len(pixels) ** 0.5))
    if side * side != len(pixels):
        raise ValueError(f"{len(pixels)} pixel columns do not make a square image")
    return list(table[list(pixels)].to_numpy(dtype=np.float32).reshape(-1, side, side))


class Pixels:
    """26.8: an image as its grey pixels, averaged down to a grid and standardised: what finding a near copy (the same
    scene photographed again) compares, as a perceptual hash does. Not a model's features: two shots of one scene stay
    nearly identical here, while features built to classify pull every image of a class together."""

    pretrained = False

    def __init__(self, grid: int = 16):
        self.grid, self.name = grid, f"pixels-{grid}"

    def encode(self, items: Sequence[Any]) -> np.ndarray:
        out = []
        for image in items:
            a = np.asarray(image, dtype=np.float32)
            a = a.mean(axis=2) if a.ndim == 3 else a
            step = max(a.shape[0] // self.grid, 1)
            g = a[:step * self.grid, :step * self.grid].reshape(self.grid, step, self.grid, step).mean(axis=(1, 3)).reshape(-1)
            out.append((g - g.mean()) / (g.std() + 1e-6))
        return _unit(np.asarray(out, dtype=np.float32))


class Centred:
    """An encoder's vectors centred on a mean fitted on training items (then unit length): a baseline whose vectors are
    all alike (pooled rectified features, every cosine near 1) becomes able to tell a near copy from another item. The
    mean is the training items' only, never a held-out item's (26.8)."""

    def __init__(self, encoder: Any, mean: np.ndarray):
        self.encoder, self.mean = encoder, np.asarray(mean, dtype=np.float32)
        self.name = f"{encoder.name}-centred-{hashlib.sha256(self.mean.tobytes()).hexdigest()[:8]}"  # a cache never mixes two means
        self.pretrained = bool(getattr(encoder, "pretrained", False))

    @classmethod
    def fit(cls, encoder: Any, training_items: Sequence[Any]) -> "Centred":
        return cls(encoder, encoder.encode(training_items).mean(axis=0))

    def encode(self, items: Sequence[Any]) -> np.ndarray:
        return _unit(self.encoder.encode(items) - self.mean)


class RandomFeatures:
    """Fixed random projections of 5x5 patches of a grey image, rectified, pooled over a 3x3 grid of regions (mean) and
    the whole image (max): a baseline that needs no weights, not a pretrained model."""

    pretrained = False

    def __init__(self, features: int = 64, seed: int = 0, size: int = 32):
        rng = np.random.default_rng(seed)
        self.filters = rng.normal(0, 1, (features, 25)).astype(np.float32)
        self.filters -= self.filters.mean(axis=1, keepdims=True)
        self.size = size
        self.name = f"random-features-{features}-seed{seed}"

    def encode(self, items: Sequence[Any]) -> np.ndarray:
        out = []
        for image in items:
            a = np.asarray(image, dtype=np.float32)
            a = a.mean(axis=2) if a.ndim == 3 else a
            a = (a - a.mean()) / (a.std() + 1e-6)
            view = np.lib.stride_tricks.sliding_window_view(a, (5, 5))[::2, ::2]
            gh, gw = view.shape[:2]
            r = np.maximum(view.reshape(gh, gw, 25) @ self.filters.T, 0)  # (rows, columns, features)
            cells = [r[i * gh // 3:(i + 1) * gh // 3, j * gw // 3:(j + 1) * gw // 3].mean(axis=(0, 1)) for i in range(3) for j in range(3)]
            out.append(np.concatenate([*cells, r.max(axis=(0, 1))]))  # what is where, and what at all
        return _unit(np.asarray(out, dtype=np.float32))


# ---------------------------------------------------------------------------- 16.15 C: in the engine, split / choose / score
CANDIDATE_C = (0.1, 1.0, 10.0)


def _model(task: str, c: float) -> Any:
    from sklearn.linear_model import LogisticRegression, Ridge

    return LogisticRegression(max_iter=3000, C=c) if task != "regression" else Ridge(alpha=1.0 / c)


def _scoring(task: str) -> str:
    return "accuracy" if task == "multiclass" else ("roc_auc" if task == "binary" else "r2")


def _names(sources: Sequence[Any]) -> np.ndarray:
    """Every source as the same text in every function (a date, a number or a string alike)."""
    return np.array([str(s) for s in sources], dtype=object)


def split(sources: Sequence[Any], *, holdout: float = 0.2, folds: int = 5, seed: int = 0) -> dict[str, Any]:
    """Which whole sources are held out, from the sources' names alone (never a label or an item): each source's place
    is the SHA-256 of the seed and its name, and the lowest ``holdout`` share is held out, so adding rows never moves
    a source. The record keeps a fingerprint of the held-out set, not the names."""
    names = sorted(set(_names(sources)))
    if len(names) < folds + 2:
        raise ValueError(f"{len(names)} sources: too few to keep whole sources apart in {folds} folds and a holdout")
    rank = sorted(names, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    held = sorted(rank[:max(1, int(round(holdout * len(names))))])
    return {"by": "whole sources", "sources": len(names), "holdout_sources": len(held), "holdout_share": holdout, "folds": folds, "seed": seed,
            "holdout_fingerprint": hashlib.sha256("\n".join(held).encode()).hexdigest()[:16], "_held": held}


def holdout_mask(sources: Sequence[Any], parts: dict[str, Any]) -> np.ndarray:
    return np.isin(_names(sources), parts["_held"])


def choose(X: np.ndarray, y: Sequence[Any], sources: Sequence[Any], *, task: str = "binary", folds: int = 5, seed: int = 0,
           threshold: float = THRESHOLD) -> dict[str, Any]:
    """Each regularisation strength scored with folds grouped by source on the training rows only; the best mean is
    chosen (a tie to the stronger regularisation). The chosen one is also scored with random folds: the gap is what
    recognising near copies of a source is worth. Near copies across the grouped and the random folds are counted."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, cross_val_score

    X, y, groups = np.asarray(X), np.asarray(y), _names(sources)
    scoring = _scoring(task)
    if task == "binary":
        positive = len(set(groups[y == 1]))
        if positive < folds or len(set(groups[y == 0])) < folds:
            raise ValueError(f"{positive} sources with a positive row: too few for {folds} folds grouped by source that each hold both classes")
    # classes are spread over the folds, sources kept whole, for a class task; a number's folds are by source only
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed) if task != "regression" else GroupKFold(folds)
    cuts = list(splitter.split(X, y, groups))
    scores = {c: cross_val_score(_model(task, c), X, y, cv=cuts, scoring=scoring) for c in CANDIDATE_C}
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("a fold grouped by source holds one class: no regularisation can be chosen by its score")
    best = max(CANDIDATE_C, key=lambda c: (round(float(scores[c].mean()), 4), -c))
    shuffled = cross_val_score(_model(task, best), X, y, cv=KFold(folds, shuffle=True, random_state=seed), scoring=scoring)
    fold_of, random_fold = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(cuts):
        fold_of[va] = k
    for k, (_, va) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(X)):
        random_fold[va] = k
    return {"metric": scoring, "by_c": {str(c): {"mean": _r(scores[c].mean()), "std": _r(scores[c].std())} for c in CANDIDATE_C},
            "chosen": str(best), "chosen_by": f"the best mean {scoring} over folds grouped by source, stratified by class (a tie to the stronger regularisation)",
            "grouped_folds": _r(scores[best].mean()), "random_folds": _r(shuffled.mean()),
            "near_duplicates_across_folds": (by_fold := near_duplicates(X, fold_of, threshold=threshold))["pairs_across_sides"],
            "near_duplicates_across_random_folds": near_duplicates(X, random_fold, threshold=threshold)["pairs_across_sides"],
            "near_duplicates_method": by_fold["method"], "near_duplicates_recall": by_fold.get("recall_at_threshold")}


def score(X: np.ndarray, y: Sequence[Any], sources: Sequence[Any], parts: dict[str, Any], chosen: str, *, task: str = "binary") -> dict[str, Any]:
    """The chosen model fitted on every training row and scored once on the held-out sources' rows."""
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    X, y = np.asarray(X), np.asarray(y)
    test = holdout_mask(sources, parts)
    train = ~test
    groups = _names(sources)
    model = _model(task, float(chosen)).fit(X[train], y[train])  # the holdout's rows are never fitted
    predicted = model.predict_proba(X[test])[:, 1] if task == "binary" else model.predict(X[test])
    why = None
    if task != "regression" and len(set(y[test].tolist())) < 2:
        held, why = None, "the held-out sources hold one class: the holdout score cannot be read"
    elif task == "binary":
        held = roc_auc_score(y[test], predicted)
    elif task == "multiclass":
        held = accuracy_score(y[test], predicted)
    else:
        held = r2_score(y[test], predicted)
    return {"metric": _scoring(task), "holdout": _r(held), "why_no_score": why, "fitted_rows": int(train.sum()), "train_rows": int(train.sum()),
            "holdout_rows": int(test.sum()), "sources_on_both_sides": int(len(set(groups[test]) & set(groups[train]))),
            "holdout_predictions": [_r(v, 6) if task != "multiclass" else v for v in np.ravel(predicted).tolist()]}
