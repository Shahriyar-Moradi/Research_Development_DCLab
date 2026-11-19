"""How a holdout should be split, read by code from the person's words and the rows (2026-11-19).

Two rules that a planner applies when nobody is there to ask:

- ``new_entities`` and ``entity_column``: when the brief says the model must work for entities it has not seen (new
  stores, new presses, next year's intake), rows of one entity on both sides of the split flatter the score, so the
  split keeps each entity on one side. Only then: a model used on the same customers again is judged honestly by a split
  in time, with the customer on both sides (the end-to-end future set, 2026-11-19).
- ``first_known_time``: a table with two dates (booked and appointment, invoiced and due) has one that is known first
  on every row; ordering the split by it is honest, a random split is not. None when no date is first on almost every
  row, or the dates lie in different periods (a birth date beside an order date).

Pure: names and the rows given, nothing written, no model.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import pandas as pd

from .bands import name_tokens

# the things a business keeps rows about; a brief's "new <thing>" is about new entities only when the thing is one of these
ENTITY_NOUNS = ("stores", "shops", "branches", "sites", "locations", "outlets", "customers", "clients", "accounts", "users",
                "members", "subscribers", "patients", "students", "learners", "pupils", "machines", "presses", "devices",
                "meters", "sensors", "assets", "buildings", "properties", "hotels", "restaurants", "merchants", "sellers",
                "suppliers", "vendors", "products", "items", "skus", "cards", "drivers", "vehicles", "employees", "agents",
                "schools", "clinics", "hospitals", "plants", "factories", "warehouses", "depots", "routes", "farms")
_NOUN = "(" + "|".join(ENTITY_NOUNS) + ")"
NEW_ENTITY = re.compile(
    r"\b(?:new|newly\s+\w+|unseen|incoming|upcoming|future|additional)\s+(?:[a-z-]+\s+)?" + _NOUN + r"\b"  # three new presses
    r"|\b" + _NOUN + r"\s+(?:that\s+)?(?:we|you|they)\s+(?:will\s+|are\s+|plan\s+to\s+)?"
    r"(?:open|opening|add|adding|acquire|acquiring|take\s+over|taking\s+over|onboard|onboarding|launch|launching|buy|buying)\b"  # the stores we open
    r"|\b(?:new|next\s+year'?s|next|incoming)\s+intake\b"  # next year's intake
    r"|\b(?:never|not\s+yet)\s+seen\s+before\b",
    re.I)
# words before a match, in its sentence, that say the new ones are not wanted ("new stores are out of scope", "we do not
# open new branches"), and words that make "new accounts" a kind of row rather than unseen entities ("fraud on new accounts")
NOT_WANTED = re.compile(r"\b(?:not|no|never|don'?t|doesn'?t|won'?t|without|excluding|exclude[sd]?|except|only\s+existing|existing\s+\w+\s+only)\b", re.I)
NOT_WANTED_AFTER = re.compile(r"^[^.;!?]*\b(?:out\s+of\s+scope|excluded|not\s+needed|not\s+in\s+scope|are\s+ignored)\b", re.I)
A_KIND_OF_ROW = re.compile(r"\b(?:from|on|of|by)\s+$", re.I)
KEY_TOKENS = frozenset({"id", "nr", "ref", "serial", "code", "key", "uuid", "acct", "sku"})  # and "no" as the last word: store_no, not no_show
FIRST_SHARE = 0.95  # a date is known first when it is at most the other on this share of rows


def new_entities(brief: str) -> Optional[str]:
    """The kind of entity the brief says the model must also serve when new ("presses", "stores"), "" when it says so
    without naming one (an intake, things not seen before), or None when it does not say so."""
    text = str(brief or "")
    for match in NEW_ENTITY.finditer(text):
        start = max(text.rfind(".", 0, match.start()), text.rfind(";", 0, match.start()), text.rfind("!", 0, match.start()),
                    text.rfind("?", 0, match.start())) + 1
        before, after = text[start:match.start()], text[match.end():]
        words = match.group(0).split()
        if words[0] == "New" and len(words) > 2 and words[1][:1].isupper():
            continue  # a place: our New York stores
        if NOT_WANTED.search(before) or NOT_WANTED_AFTER.search(after) or (match.group(1) and A_KIND_OF_ROW.search(before)):
            continue
        return next((g.lower() for g in match.groups() if g), "")
    return None


def _singular(word: str) -> str:
    word = word.lower()
    for ending, back in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("s", "")):
        if word.endswith(ending):
            return word[: -len(ending)] + back
    return word


def entity_candidates(frame: Any, columns: Sequence[str], exclude: Sequence[Any] = ()) -> list[str]:
    """Columns code reads as an entity's key: named as a key (``store_no``, ``meter_id``, ``machine_serial``), text or
    whole numbers, at least ten distinct values, and on average at least two rows each."""
    out = []
    rows = len(frame) if frame is not None else 0
    skip = {str(c) for c in exclude if c}
    for name in columns:
        if str(name) in skip or frame is None or name not in getattr(frame, "columns", ()):
            continue
        tokens = name_tokens(str(name))
        if not (KEY_TOKENS & set(tokens) or (len(tokens) > 1 and tokens[-1] == "no")):
            continue  # a count (num_items) or a flag (no_show) is no entity's key
        series = frame[name].dropna()
        if getattr(series.dtype, "kind", "O") not in ("O", "i", "u", "U", "S"):
            continue
        distinct = int(series.nunique())
        if 10 <= distinct <= 0.5 * max(rows, 1):
            out.append(str(name))
    return out


def entity_column(frame: Any, columns: Sequence[str], noun: Optional[str], exclude: Sequence[Any] = ()) -> Optional[str]:
    """The entity the split should keep on one side: the candidate the brief's noun names (``stores`` -> ``store_no``);
    with no noun (an intake), the finest one (most distinct values: the student, not the course). None when no candidate
    is named by the noun: a split by customer is never the answer to "the stores we open" (the review, 2026-11-19)."""
    found = entity_candidates(frame, columns, exclude)
    if not found:
        return None
    if noun:
        named = [c for c in found if _singular(noun) in name_tokens(c)]
        return named[0] if len(named) == 1 else None
    return max(found, key=lambda c: (int(frame[c].nunique()), c))


def first_known_time(frame: Any, names: Sequence[str]) -> Optional[str]:
    """Of several date columns, the one known first on (almost) every row, when they lie in the same period; else None."""
    if frame is None or len(names) < 2:
        return None
    try:  # one way for every column: a zone on one and none on the other once raised (the review, 2026-11-19)
        parsed = {}
        for name in names:
            if name not in getattr(frame, "columns", ()):
                return None
            parsed[name] = pd.to_datetime(frame[name], errors="coerce", utc=True, format="mixed")
        return _first(parsed, names)
    except Exception:  # noqa: BLE001 - unreadable dates: no choice, as before
        return None


def _first(parsed: dict[str, Any], names: Sequence[str]) -> Optional[str]:
    spans = {n: (v.min(), v.max()) for n, v in parsed.items()}
    if any(pd.isna(a) or pd.isna(b) for a, b in spans.values()):
        return None
    for name, values in parsed.items():
        others = [n for n in names if n != name]
        first = all(float((values <= parsed[o]).mean()) >= FIRST_SHARE for o in others)
        # the same period: they share at least half of the longer span (a birth date lies decades before; a sign-up date
        # over seven years beside six months of snapshots is the customer's age, not the row's time)
        same = all(_overlap(spans[name], spans[o]) >= 0.5 for o in others)
        if first and same:
            return str(name)
    return None


def _overlap(a: tuple[Any, Any], b: tuple[Any, Any]) -> float:
    shared = (min(a[1], b[1]) - max(a[0], b[0])).total_seconds()
    longer = max((a[1] - a[0]).total_seconds(), (b[1] - b[0]).total_seconds())
    return 1.0 if longer <= 0 and shared >= 0 else max(0.0, shared) / longer if longer > 0 else 0.0


__all__ = ["ENTITY_NOUNS", "FIRST_SHARE", "entity_candidates", "entity_column", "first_known_time", "new_entities"]
