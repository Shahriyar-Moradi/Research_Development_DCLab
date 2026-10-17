"""Images and long texts, invented, with planted labels and near copies (replicas for package 16.12).

``shapes``: small grey images of a circle or a square (the label) at random places and sizes, photographed several
times per scene with a little noise: the shots of one scene are near copies, so a random split puts one shot in
training and its twin in the test.

``notes``: long support tickets whose label (urgent or not) depends on what the text says, written by authors with
their own phrasing; a few tickets were pasted again by another author with a new first line: near copies across
authors. With ``outcomes=True`` each ticket also has a ``topic`` (seven classes, from the phrase it was written around)
and ``minutes_to_resolve`` (a number: its two phrases' usual times, shorter when urgent, with noise), for the multiclass and
regression forms of the kind (16.15 C follow-up); the tickets, authors and copies are the same either way.

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


def shapes(seed: int = 0, scenes: int = 80, shots: int = 4, size: int = 32, *, reshot: int = 0, mislabel: float = 0.0,
           arrows: bool = False) -> Replica:
    """The shapes; 26.8's traps on request, each drawn from a stream of its own so the defaults draw the same images:
    ``reshot`` scenes photographed again under a new scene name (the same subject on two sources: a near copy the scene
    column cannot see), ``mislabel`` the share of shots given the other label, and ``arrows`` a triangle pointing left or
    right as the label (a mirror flip changes it; for a circle or a square it does not)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    names = ("left", "right") if arrows else ("circle", "square")
    images, rows, drawn = [], [], []
    for s in range(scenes):
        label = int(rng.integers(0, 2))
        cx, cy, r = rng.uniform(9, size - 9, 2).tolist() + [rng.uniform(4, 8)]
        if arrows:  # the wide end on one side, the point on the other
            inside = (xx >= cx - r) & (xx <= cx + r)
            base = inside & (abs(yy - cy) <= ((xx - (cx - r)) if label == 0 else (cx + r - xx)) * 0.5)
        else:
            base = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) if label == 0 else ((abs(xx - cx) <= r) & (abs(yy - cy) <= r))
        background = rng.uniform(0.1, 0.4)
        drawn.append((base, background, label))
        for k in range(shots):
            img = np.where(base, 0.9, background) + rng.normal(0, 0.03, (size, size))
            images.append(np.clip(img, 0, 1).astype(np.float32))
            rows.append({"item": len(rows), "scene": f"S{s:03d}", "shot": k, "label": names[label]})
    facts: dict[str, Any] = {"source": "scene", "label": "label", "kind": "image", "synthetic": True}
    if reshot or mislabel or arrows:
        traps = np.random.default_rng([seed, 26, 8])
        pairs = []
        for n, s in enumerate(sorted(traps.choice(scenes, size=min(reshot, scenes), replace=False).tolist())):
            base, background, label = drawn[s]
            new = f"S{scenes + n:03d}"
            for k in range(shots):  # the same subject again, a little more noise: another day's photograph
                img = np.where(base, 0.9, background) + traps.normal(0, 0.04, (size, size))
                images.append(np.clip(img, 0, 1).astype(np.float32))
                rows.append({"item": len(rows), "scene": new, "shot": k, "label": names[label]})
            pairs.append([f"S{s:03d}", new])
        wrong = sorted(traps.choice(len(rows), size=int(round(mislabel * len(rows))), replace=False).tolist()) if mislabel else []
        for i in wrong:
            rows[i]["label"] = names[1 - names.index(rows[i]["label"])]
        facts.update(reshot_scenes=pairs, mislabelled_items=[int(i) for i in wrong], flip_keeps_label=not arrows)
    return Replica(images, pd.DataFrame(rows), facts)


def shapes_table(seed: int = 0, scenes: int = 80, shots: int = 4, size: int = 32, prefix: str = "px_") -> Replica:
    """26.8: the shapes as a person's export holds images: one row per image, its scene and label, and its grey pixels as
    columns (``px_0000`` … row by row), as image tables come (a digits table, a scan export). The same scenes, shots and
    near copies as :func:`shapes`."""
    base = shapes(seed, scenes, shots, size)
    pixels = pd.DataFrame(np.stack([np.asarray(i, dtype=np.float32).reshape(-1) for i in base.items]).round(4),
                          columns=[f"{prefix}{k:04d}" for k in range(size * size)])
    table = pd.concat([base.table.reset_index(drop=True), pixels], axis=1)
    return Replica(base.items, table, {**base.facts, "pixel_prefix": prefix, "size": size, "shots_per_scene": shots})


WORDS = {"urgent": ["outage", "cannot log in", "payments failing", "data lost", "all users affected", "production is down"],
         "calm": ["question about", "would like to know", "small typo", "feature idea", "when convenient", "thank you"]}
FILL = ["we use the service every day", "our team has twelve people", "the dashboard shows the usual numbers", "we checked the guide",
        "the account was opened last year", "nothing else changed on our side", "the export runs at night", "we use the mobile app too"]


TOPIC = {"outage": "availability", "all users affected": "availability", "production is down": "availability",
         "cannot log in": "access", "payments failing": "billing", "data lost": "data",
         "question about": "question", "would like to know": "question", "when convenient": "question", "thank you": "question",
         "small typo": "content", "feature idea": "feature"}
MINUTES = {"availability": 90, "access": 45, "billing": 120, "data": 240, "question": 30, "content": 20, "feature": 60}


def notes(seed: int = 0, authors: int = 60, per_author: int = 5, copies: int = 12, outcomes: bool = False) -> Replica:
    rng = np.random.default_rng(seed)
    chosen: list[list[str]] = []  # each ticket's planted phrases: the first is its topic
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
            chosen.append([str(k) for k in key])
    pairs = []
    for _ in range(copies):  # pasted again by another author, with a new first line
        i = int(rng.integers(0, len(texts)))
        own = int(rows[i]["author"][1:])
        other = f"A{int(rng.choice([a for a in range(authors) if a != own])):03d}"  # another author, never the same one
        texts.append("Forwarding this again. " + texts[i])
        rows.append({"item": len(rows), "author": other, "urgent": rows[i]["urgent"]})
        chosen.append(chosen[i])
        pairs.append((i, len(rows) - 1))
    table = pd.DataFrame(rows)
    facts = {"source": "author", "label": "urgent", "kind": "text", "copied_pairs": pairs, "synthetic": True}
    if outcomes:  # drawn from their own generator: the tickets above are the same with or without them
        noise = np.random.default_rng(seed + 7919)
        topic = [TOPIC[k[0]] for k in chosen]
        table["topic"] = topic
        table["minutes_to_resolve"] = [round(float(np.mean([MINUTES[TOPIC[k]] for k in ks])) * (0.5 if u else 1.0) * float(noise.lognormal(0, 0.25)), 1)
                                       for ks, u in zip(chosen, table["urgent"])]
        facts.update({"topic_rule": "the topic of the first planted phrase (the text holds two, in any order)",
                      "minutes_rule": "the mean of the two planted phrases' topic minutes, halved when urgent, times log-normal noise (sd 0.25)"})
    return Replica(texts, table, facts)
