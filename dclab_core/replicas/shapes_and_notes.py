"""Images and long texts, invented, with planted labels and near copies (replicas for package 16.12).

``shapes``: small grey images of a circle or a square (the label) at random places and sizes, photographed several
times per scene with a little noise: the shots of one scene are near copies, so a random split puts one shot in
training and its twin in the test.

``notes``: long support tickets whose label (urgent or not) depends on what the text says, written by authors with
their own phrasing; a few tickets were pasted again by another author with a new first line: near copies across
authors.

Each item's scene or author is its source: folds must keep a source on one side. The label rule and the copied pairs
are in ``facts`` (for judging only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    items: list
    table: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def shapes(seed: int = 0, scenes: int = 80, shots: int = 4, size: int = 32) -> Replica:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    images, rows = [], []
    for s in range(scenes):
        label = int(rng.integers(0, 2))
        cx, cy, r = rng.uniform(9, size - 9, 2).tolist() + [rng.uniform(4, 8)]
        base = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) if label == 0 else ((abs(xx - cx) <= r) & (abs(yy - cy) <= r))
        background = rng.uniform(0.1, 0.4)
        for k in range(shots):
            img = np.where(base, 0.9, background) + rng.normal(0, 0.03, (size, size))
            images.append(np.clip(img, 0, 1).astype(np.float32))
            rows.append({"item": len(rows), "scene": f"S{s:03d}", "shot": k, "label": "square" if label else "circle"})
    return Replica(images, pd.DataFrame(rows), {"source": "scene", "label": "label", "kind": "image", "synthetic": True})


WORDS = {"urgent": ["outage", "cannot log in", "payments failing", "data lost", "all users affected", "production is down"],
         "calm": ["question about", "would like to know", "small typo", "feature idea", "when convenient", "thank you"]}
FILL = ["we use the service every day", "our team has twelve people", "the dashboard shows the usual numbers", "we checked the guide",
        "the account was opened last year", "nothing else changed on our side", "the export runs at night", "we use the mobile app too"]


def notes(seed: int = 0, authors: int = 60, per_author: int = 5, copies: int = 12) -> Replica:
    rng = np.random.default_rng(seed)
    texts, rows = [], []
    for a in range(authors):
        style = rng.choice(["Hello team,", "Hi,", "Dear support,", "Good morning,"])
        sign = rng.choice(["Best", "Thanks", "Regards", "Cheers"]) + f", author {a}"
        for _ in range(per_author):
            urgent = int(rng.uniform() < 0.4)
            key = rng.choice(WORDS["urgent" if urgent else "calm"], 2, replace=False)
            body = list(rng.choice(FILL, 6)) + list(key)
            rng.shuffle(body)
            texts.append(f"{style} " + ". ".join(body) + f". {sign}")
            rows.append({"item": len(rows), "author": f"A{a:03d}", "urgent": urgent})
    pairs = []
    for _ in range(copies):  # pasted again by another author, with a new first line
        i = int(rng.integers(0, len(texts)))
        own = int(rows[i]["author"][1:])
        other = f"A{int(rng.choice([a for a in range(authors) if a != own])):03d}"  # another author, never the same one
        texts.append("Forwarding this again. " + texts[i])
        rows.append({"item": len(rows), "author": other, "urgent": rows[i]["urgent"]})
        pairs.append((i, len(rows) - 1))
    return Replica(texts, pd.DataFrame(rows), {"source": "author", "label": "urgent", "kind": "text", "copied_pairs": pairs, "synthetic": True})
