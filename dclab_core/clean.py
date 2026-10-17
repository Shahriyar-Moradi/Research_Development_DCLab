"""Cleaning a training set (package 27.5): duplicates, personal data, values, evaluation cases, quality, its card.

One pass every gold set goes through before a model trains on it:

- **pure**: a decision sample's features are a state of bands, tokens, counts and names (``ledger.check_state``; a table-level
  sample's column list checked column by column), never a table's value;
- **personal**: no text holds what ``redact`` would replace (an e-mail, a phone number, a card, a key, an address);
- **duplicates**: one copy of each example per split, exact or near (the same text once case, digits, spacing and
  punctuation are set aside), and none in train or dev that is (nearly) an evaluation row;
- **evaluation cases**: an example built from a table an evaluation reads (a replica's seed a benchmark, a journey or a
  sealed set uses) is never in train;
- **quality**: a score from 0 to 1 per training example (verified, cited where the family answers from evidence, a target
  of a sensible length), kept on the example;
- **card**: what the set holds, where it comes from, what it was cleaned of, and what it can and cannot claim.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Optional

from .ledger import check_state
from .redact import redact_text

VERSION = "clean_v1"
CITATION = re.compile(r"\b(?:DCLAB|PIT|AEV|SLV|AML|DMG|DWM|DWF|JRN|HRN|TWN)-[A-Z]?\d{2,4}\b")
_PUNCT = re.compile(r"[^a-z0-9]+")


def normal(text: str) -> str:
    """Text as near-duplicates compare it: lower case, punctuation and spacing one space. The digits stay: in families
    whose content is numbers (a result read, a signal measured) two examples with other numbers teach other things."""
    return _PUNCT.sub(" ", str(text or "").lower()).strip()


def near_key(messages: Iterable[Mapping[str, Any]]) -> str:
    """A training example's near-duplicate key: its user and assistant turns, normalised (the system text is shared)."""
    body = "\n".join(normal(m.get("content", "")) for m in messages if m.get("role") != "system")
    return hashlib.sha256(body.encode()).hexdigest()


def impure(features: Mapping[str, Any]) -> list[str]:
    """Why a decision sample's features are not a pure state (empty when they are): a table-level sample's columns are
    checked one by one; an empty field is no field."""
    state = {k: v for k, v in features.items() if v is not None}
    columns = state.pop("columns", None)
    target_state = state.pop("target_state", None)  # a table-level sample's outcome column, as a column state
    if isinstance(columns, int) and not isinstance(columns, bool):
        state["columns"], columns = columns, None  # a count of columns
    problems = list(check_state(state))
    if target_state is not None:
        problems += [f"target_state.{p}" for p in (check_state({k: v for k, v in target_state.items() if v is not None})
                                                    if isinstance(target_state, dict) else ["a column state"])]
    if columns is not None:
        if not isinstance(columns, list):
            problems.append("columns: a list of column states")
        else:
            for i, c in enumerate(columns[:400]):
                if not isinstance(c, dict):
                    problems.append(f"columns[{i}]: a column state")
                    continue
                problems += [f"columns[{i}].{p}" for p in check_state({k: v for k, v in c.items() if v is not None})]
    return problems


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


# what identifies a person or reaches them (redact's other kinds, an amount, a long number or a URL, are the content of
# research write-ups and measurements: a row count, a feature range, a price in a study)
PERSONAL = ("EMAIL", "IBAN", "CARD", "IP", "PHONE", "ADDRESS")


def personal(texts: Iterable[Optional[str]]) -> dict[str, int]:
    """What ``redact`` would replace in these texts that identifies a person (``PERSONAL``), by kind (empty when nothing).
    A text that is JSON is read as JSON and only its strings are checked: a measured number's digits are not a card."""
    import json

    found: Counter[str] = Counter()
    for t in texts:
        if not t:
            continue
        try:
            parts = list(_strings(json.loads(t)))
        except (ValueError, TypeError):
            parts = [t]
        for part in parts:
            _, counts = redact_text(part)
            found.update({k: v for k, v in counts.items() if v and k in PERSONAL})
    return dict(found)


def quality(family: str, messages: list[Mapping[str, Any]], verified: bool, reasoned: bool = False, agreement: Optional[bool] = None) -> float:
    """0 to 1: verified (half the score); the rest the mean of a target of a sensible length, a verified reasoning trace
    (27.9), a citation where the family answers from evidence, and, where a second answerer exists, its agreement."""
    target = str((messages[-1] if messages else {}).get("content", ""))
    parts = [1.0 if 2 <= len(target) <= 4000 else 0.5, 1.0 if reasoned else 0.0]
    if family == "evidence_answer":
        parts.append(1.0 if CITATION.search(target) else 0.0)
    if agreement is not None:
        parts.append(1.0 if agreement else 0.0)
    return round(0.5 * (1.0 if verified else 0.0) + 0.5 * sum(parts) / len(parts), 3)


def dedupe(rows: list[Any], key: Callable[[Any], str], split: Callable[[Any], str]) -> tuple[list[Any], Counter]:
    """One copy per split; a train row (nearly) equal to a dev or sealed row, and a dev row equal to a sealed row, dropped."""
    by_split: dict[str, set[str]] = {}
    for r in rows:
        by_split.setdefault(split(r), set()).add(key(r))
    held = {"train": by_split.get("dev", set()) | by_split.get("sealed", set()), "dev": by_split.get("sealed", set())}
    seen: set[tuple[str, str]] = set()
    out, why = [], Counter()
    for r in rows:
        k, s = key(r), split(r)
        if k in held.get(s, ()):
            why[f"{s} row (nearly) identical to an evaluation row"] += 1
            continue
        if (k, s) in seen:
            why["duplicate"] += 1
            continue
        seen.add((k, s))
        out.append(r)
    return out, why


def card(name: str, kind: str, manifest: Mapping[str, Any], dropped: Mapping[str, int], licences: Mapping[str, int],
         claims: Iterable[str], limits: Iterable[str]) -> dict[str, Any]:
    """A dataset card: what the set is, where it comes from, what it was cleaned of, what it can and cannot claim."""
    return {"name": name, "kind": kind, "cleaning": VERSION, "rows": manifest.get("rows"), "by_split": manifest.get("by_split"),
            "by_origin": manifest.get("by_origin"), "groups": manifest.get("groups"), "licences": dict(licences),
            "dropped": dict(dropped), "can_claim": list(claims), "cannot_claim": list(limits)}


def card_markdown(c: Mapping[str, Any]) -> str:
    lines = [f"# {c['name']}", "", f"Kind: {c['kind']}. Cleaned by {c['cleaning']}. Rows: {c['rows']} in {c['groups']} groups.", "",
             "| Split | Rows |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted((c.get("by_split") or {}).items())], "",
             "| Origin | Rows |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted((c.get("by_origin") or {}).items())], "",
             "Licences: " + (", ".join(f"{k} ({v})" for k, v in sorted((c.get("licences") or {}).items())) or "none recorded"), "",
             "Dropped: " + (", ".join(f"{k}: {v}" for k, v in sorted((c.get("dropped") or {}).items())) or "nothing"), "",
             "## What it can claim", "", *[f"- {x}" for x in c.get("can_claim") or []], "",
             "## What it cannot claim", "", *[f"- {x}" for x in c.get("cannot_claim") or []], ""]
    return "\n".join(lines)


__all__ = ["VERSION", "PERSONAL", "normal", "near_key", "impure", "personal", "quality", "dedupe", "card", "card_markdown"]
