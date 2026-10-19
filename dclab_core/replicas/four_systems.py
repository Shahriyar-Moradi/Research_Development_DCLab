"""One customer in four systems, invented (the foundation replica, package 30.23).

A company keeps its customers in a CRM, a billing system, a support desk and a web shop, each with its own identifier and its own way of
writing a name, an e-mail and a phone: the same person as "Ana Lopez" with "ana.lopez@example.org" in one, "LOPEZ, ANA" with
"Ana.Lopez@Example.org" in another, a typo in a third, the phone with or without its country code. Most customers are in two or three of
the systems; some share a surname or a city with others. ``truth`` maps every record to its person; ``matches`` lists the pairs known to be
the same person. Invented data: a resolution on it says the method holds, never how a real company's records match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dclab_core.replicas import LABELS

SYSTEMS = ("crm", "billing", "support", "web")
FIRST = ("Ana", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea", "Mo", "Nia", "Oto", "Pia", "Raj", "Sia", "Tom", "Uma")
LAST = ("Lopez", "Kim", "Okafor", "Novak", "Silva", "Haddad", "Berg", "Moreau", "Tanaka", "Rossi", "Patel", "Adeyemi", "Kowalski", "Nguyen")
CITIES = ("Lisbon", "Lagos", "Leeds", "Lyon", "Lima", "Lahore")


@dataclass(frozen=True)
class Replica:
    systems: dict[str, pd.DataFrame]
    matches: pd.DataFrame
    truth: dict[str, int] = field(default_factory=dict)  # "system:id" -> the person
    info: dict[str, Any] = field(default_factory=dict)


def _typo(rng: np.random.Generator, word: str) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def generate(seed: int = 0, people: int = 1_200) -> Replica:
    rng = np.random.default_rng(seed)
    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in SYSTEMS}
    truth: dict[str, int] = {}
    for p in range(people):
        first, last = str(rng.choice(FIRST)), str(rng.choice(LAST))
        city = str(rng.choice(CITIES))
        phone = f"{int(rng.integers(200, 999))}{int(rng.integers(1_000_000, 9_999_999))}"
        email = f"{first.lower()}.{last.lower()}{int(rng.integers(1, 99))}@example.org"
        present = [s for s in SYSTEMS if rng.random() < 0.6] or [str(rng.choice(SYSTEMS))]
        for s in present:
            ident = f"{s[:2].upper()}{len(rows[s]) + 1:06d}"
            name = f"{last.upper()}, {first.upper()}" if s == "billing" else f"{first} {last}"
            if rng.random() < 0.15:
                name = _typo(rng, name)
            mail = email.title() if s == "support" and rng.random() < 0.5 else email
            tel = ("+351 " + phone) if s == "web" else phone
            rows[s].append({"id": ident, "name": name, "email": mail, "phone": tel, "city": city, "since": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 600))))[:10]})
            truth[f"{s}:{ident}"] = p
    systems = {s: pd.DataFrame(r) for s, r in rows.items()}
    by_person: dict[int, list[str]] = {}
    for record, p in truth.items():
        by_person.setdefault(p, []).append(record)
    pairs = [(a, b) for recs in by_person.values() for a, b in zip(recs, recs[1:])]  # a chain per person: every pair inside follows
    matches = pd.DataFrame(pairs, columns=["record", "same_as"])
    return Replica(systems=systems, matches=matches, truth=truth,
                   info={**LABELS, "seed": seed, "people": people, "records": len(truth), "in_two_or_more": sum(1 for r in by_person.values() if len(r) > 1)})


__all__ = ["Replica", "generate", "SYSTEMS"]
