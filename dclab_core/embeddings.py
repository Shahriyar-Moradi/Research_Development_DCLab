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


def near_duplicates(X: np.ndarray, folds: Sequence[Any], *, threshold: float = THRESHOLD, block: int = 2000) -> dict[str, Any]:
    """Pairs of items on different sides (different fold labels) whose vectors' cosine similarity is at least
    ``threshold``: counted, and listed (row positions and similarity, at most ``MAX_PAIRS``)."""
    U = _unit(np.asarray(X, dtype=np.float32))
    f = np.asarray(folds)
    pairs, examples = 0, []
    for lo in range(0, len(U), block):
        sim = U[lo:lo + block] @ U.T
        a, b = np.nonzero(sim >= threshold)
        a = a + lo
        keep = (a < b) & (f[a] != f[b])
        pairs += int(keep.sum())
        if len(examples) < MAX_PAIRS:
            examples += [(int(i), int(j), _r(sim[i - lo, j], 4)) for i, j in zip(a[keep][:MAX_PAIRS], b[keep][:MAX_PAIRS])]
    return {"threshold": threshold, "pairs_across_sides": pairs, "pairs": examples[:MAX_PAIRS]}


def perturb_image(item: Any, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(item, dtype=np.float32)
    return a + rng.normal(0, 0.03 * (float(np.ptp(a)) or 1.0), a.shape).astype(np.float32)  # the same picture taken again


def perturb_text(item: Any, rng: np.random.Generator) -> str:
    return "Sent again. " + str(item)  # the same text with a new first line


def calibrate(items: Sequence[Any], encoder: Encoder, perturb: Any, *, sample: int = 60, seed: int = 0) -> dict[str, Any]:
    """The similarity this encoder gives a near copy (an item and a slightly changed copy of it) against the similarity
    of two different items: the threshold lies between them (the 5th percentile of copies and the 99.9th of different
    pairs), three quarters of the way towards the copies. When they overlap the encoder cannot tell near copies apart,
    and the record says so."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), min(sample, len(items)), replace=False)
    base = _unit(encoder.encode([items[i] for i in pick]))
    copy = _unit(encoder.encode([perturb(items[i], rng) for i in pick]))
    near = np.sum(base * copy, axis=1)
    others = base @ base.T
    apart = others[np.triu_indices(len(pick), 1)]
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
