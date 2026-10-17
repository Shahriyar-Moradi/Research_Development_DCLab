"""Image decision points (package 26.8): what an image problem decides before a model is trusted, as ledger records.

- ``image.near_copy``: are two images of different sources the same subject photographed again? The scene column
  keeps a scene's shots on one side of the split; a scene filed again under another name is a near copy it cannot see.
  The rule: their pictures' cosine (:class:`dclab_core.embeddings.Pixels`) reaches the near-copy threshold.
- ``image.label_suspect``: is an image's label likely wrong? The rule: most of its near copies carry the other label
  (its own scene's shots are its copies), or, with no copy, most of its nearest images of other sources do.
- ``image.flip_safe``: does a mirror flip keep the label (may training images be augmented by flipping)? The rule
  measures it: a classifier fitted on half the sources, scored on the other half as they are and mirrored; a drop of
  more than ``FLIP_DROP`` says the label depends on the side (an arrow, a hand, text).

Every rule reads the images and labels it is given (the engine gives it its training rows only, never a held-out one).
A record's state holds measures and counts, never a pixel. :func:`decide` adds the replica's truths as benchmark labels
(``dclab_core.replicas.shapes_and_notes.shapes`` with its traps): true by construction, for the decision sets.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

import numpy as np

POINTS = ("image.near_copy", "image.label_suspect", "image.flip_safe")
THRESHOLD = 0.98  # a near copy's cosine on the pictures, unless the engine calibrated another
FLIP_DROP = 0.15  # accuracy lost on mirrored images beyond which a flip changes the label
NEIGHBOURS = 5
MAX_PAIRS = 120  # near-copy questions per table: at most half over the threshold, the rest the closest below it


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def _pictures(images: Sequence[Any]) -> np.ndarray:
    from .embeddings import Pixels

    return Pixels().encode(list(images))


def _round(x: float) -> float:
    return float(round(float(x), 4))


def _flip_drop(images: np.ndarray, sources: np.ndarray, labels: np.ndarray, seed: int = 0) -> Optional[dict[str, Any]]:
    """Accuracy on held-apart sources as they are and mirrored left to right, by a classifier fitted on the others
    (the engine's local encoder, centred on the fitting images); None when either half has a single class."""
    from sklearn.linear_model import LogisticRegression

    from .embeddings import Centred, RandomFeatures

    names = np.array(sorted({str(s) for s in sources}))
    if len(names) < 4:
        return None
    order = np.random.default_rng(seed).permutation(len(names))
    fit = np.isin(sources.astype(str), names[order[: len(names) // 2]])
    if len(set(labels[fit])) < 2 or len(set(labels[~fit])) < 2:
        return None
    size = images.shape[-1]
    encoder = Centred.fit(RandomFeatures(size=size), list(images[fit]))
    model = LogisticRegression(max_iter=3000).fit(encoder.encode(list(images[fit])), labels[fit])
    plain = float((model.predict(encoder.encode(list(images[~fit]))) == labels[~fit]).mean())
    mirrored = float((model.predict(encoder.encode(list(images[~fit][:, :, ::-1]))) == labels[~fit]).mean())
    return {"plain": plain, "mirrored": mirrored, "drop": plain - mirrored, "scored": int((~fit).sum())}


def rule_decisions(images: Sequence[Any], sources: Sequence[Any], labels: Sequence[Any], *, threshold: float = THRESHOLD,
                   seed: int = 0, task: str = "classification", totals: Optional[dict[str, int]] = None) -> list[dict[str, Any]]:
    """The rules' answers on these images: ``{point, key, state, value}``, the key a pair of positions (``"3-41"``), a
    position, or ``"table"``. Positions are within the rows given. A label that is a number (``task="regression"``) has no
    wrong class and no side: only near copies are asked. ``totals``, when given, receives the counts before any cap
    (every pair over the threshold, every suspect label), the figures a person reads."""
    regression = task == "regression"
    X = np.stack([np.asarray(i, dtype=np.float32) for i in images])
    src = np.asarray([str(s) for s in sources])
    lab = np.asarray([str(v) for v in labels])
    V = _pictures(X)
    S = V @ V.T
    n = len(V)
    out: list[dict[str, Any]] = []
    i, j = np.triu_indices(n, 1)
    other = src[i] != src[j]
    i, j = i[other], j[other]
    sim = S[i, j]
    over = np.flatnonzero(sim >= threshold)
    if totals is not None:
        totals["near_copies_across_sources"] = int(len(over))
        totals["pairs_across_sources"] = int(len(sim))
    rest = np.flatnonzero(sim < threshold)
    rest = rest[np.argsort(-sim[rest], kind="stable")]
    over = over[np.argsort(-sim[over], kind="stable")]
    if len(over) > MAX_PAIRS // 2:  # spread over the similarities, so the set also holds the closest pairs that are not copies
        over = over[np.linspace(0, len(over) - 1, MAX_PAIRS // 2).round().astype(int)]
    chosen = list(over) + list(rest[: MAX_PAIRS - len(over)])
    for k in sorted(chosen, key=lambda k: (int(i[k]), int(j[k]))):
        a, b, s = int(i[k]), int(j[k]), float(sim[k])
        out.append({"point": "image.near_copy", "key": f"{a}-{b}", "value": bool(s >= threshold),
                    "state": {"similarity": _round(s), "near_threshold": _round(threshold), "margin": _round(s - threshold),
                              "same_label": bool(lab[a] == lab[b]), "images": n}})
    if regression:
        return out
    np.fill_diagonal(S, -np.inf)
    suspects = 0
    for a in range(n):
        copies = np.flatnonzero(S[a] >= threshold)
        others = np.flatnonzero(src != src[a])
        near = others[np.argsort(-S[a, others], kind="stable")[:NEIGHBOURS]] if len(others) else np.array([], dtype=int)
        copies_agree = float((lab[copies] == lab[a]).mean()) if len(copies) else None
        neighbours_agree = float((lab[near] == lab[a]).mean()) if len(near) else None
        suspect = (copies_agree < 0.5) if copies_agree is not None else (neighbours_agree is not None and neighbours_agree <= 0.2)
        suspects += bool(suspect)
        state = {"copies": int(len(copies)), "copies_agree": None if copies_agree is None else _round(copies_agree),
                 "neighbours_agree": None if neighbours_agree is None else _round(neighbours_agree), "images": n}
        out.append({"point": "image.label_suspect", "key": str(a), "value": bool(suspect),
                    "state": {k: v for k, v in state.items() if v is not None}})
    if totals is not None:
        totals["labels_suspect"] = suspects
    flip = _flip_drop(X, src, lab, seed)
    if flip is not None:
        out.append({"point": "image.flip_safe", "key": "table", "value": bool(flip["drop"] <= FLIP_DROP),
                    "state": {"flip_drop": _round(flip["drop"]), "flip_accuracy": _round(flip["mirrored"]),
                              "plain_accuracy": _round(flip["plain"]), "image_side": int(X.shape[-1]), "images": n,
                              "classes": len(set(lab))}})
    return out


def truths(replica: Any) -> dict[str, Any]:
    """What is true on a replica, from its facts: the scenes filed again, the shots given the other label, the flip."""
    facts = replica.facts
    same = {frozenset(p) for p in facts.get("reshot_scenes") or []}
    return {"same_subject": same, "mislabelled": set(facts.get("mislabelled_items") or []),
            "flip_keeps_label": bool(facts.get("flip_keeps_label", True))}


def decide(replica: Any, *, table: str, at: str, group: Optional[str] = None, threshold: float = THRESHOLD) -> tuple[list[Any], list[Any]]:
    """The rules' decisions on a shapes replica's images, each with its benchmark label (true by construction)."""
    from .ledger import Answer, DecisionLabel, DecisionRecord

    t = replica.table
    sources, labels = t["scene"].to_numpy(), t["label"].to_numpy()
    truth = truths(replica)
    records, out = [], []
    for d in rule_decisions(replica.items, sources, labels, threshold=threshold):
        if d["point"] == "image.near_copy":
            a, b = (int(x) for x in d["key"].split("-"))
            value = frozenset((str(sources[a]), str(sources[b]))) in truth["same_subject"]
        elif d["point"] == "image.label_suspect":
            value = int(t["item"].iloc[int(d["key"])]) in truth["mislabelled"]
        else:
            value = truth["flip_keeps_label"]
        digest = hashlib.sha256(f"{table}|{d['key']}|{d['point']}".encode()).hexdigest()[:20]
        record = DecisionRecord(id=f"ig{digest}", point=d["point"], subject={"kind": "project", "key": f"{table}:{d['key']}"[:200]},
                                state=d["state"], options=["yes", "no"],
                                answers=[Answer(by="rule", answerer=f"rule:{d['point']}", value=d["value"])], value_used=d["value"],
                                used="rule", dataset=(group or table)[:80], at=at)
        records.append(record)
        out.append(DecisionLabel(id=f"il{digest}", record=record.id, point=d["point"], value=bool(value), source="benchmark",
                                 by="generated:image_decisions_v1", at=at))
    return records, out


__all__ = ["POINTS", "THRESHOLD", "FLIP_DROP", "rule_decisions", "truths", "decide"]
