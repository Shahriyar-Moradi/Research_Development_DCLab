"""Memory upkeep (package 35.6): keep memory short and true without deleting anything.

- **Merge**: two active lines of the same kind and scope with the same words in the same order, filler words aside (a
  negation is never filler; numbers compared as numbers), are one fact said twice; the older is retired, named by the
  newer (``merged``). A person's line is never merged into an agent's.
- **Retire**: an agent's line shown in at least :data:`RETIRE_AFTER` briefs and never cited, or any line marked harmful
  at least :data:`HARMFUL` times, is retired (``never cited``, ``harmful``). A person's line is never retired for not
  being cited (a preference is followed without being named).
- **Conflicts**: any other two lines of one scope with much the same words (another number, a negation, another name,
  swapped words, a correction beside an agent's line) disagree; code never settles that. They are listed for a person,
  who corrects one (``POST /api/memory/{id}/correct``).

A person's correction is never merged or retired by upkeep. Every retirement is a new version with its event; the store
keeps every version (the owner's rule). ``python -m dclab_rnd.memory upkeep`` prints the plan; ``--apply`` writes it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from dclab_core.memory import MemoryItem, retire

from . import MemoryStore, event, latest

RETIRE_AFTER = 30   # briefs that showed a line without any reply citing it
HARMFUL = 2         # harmful marks that retire a line
CONFLICT = 0.6      # word overlap (Jaccard, numbers aside) at which two lines that are not the same say different things
FILLER = frozenset("a an the of for to in on at by is are was were be this that it its and or with as from".split())
NEGATION = frozenset("not no never none without نه نمی ن".split())  # a word that turns a line around is never filler
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _tokens(text: str) -> list[str]:
    """The line's words in order, lower-cased, numbers in a normal form (Persian digits read, thousands separators dropped)."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in folded)
    out = []
    for t in re.findall(r"\w+(?:[.,]\d+)*", folded):
        out.append(_number(t) if NUMBER.fullmatch(t) else t)
    return out


def _number(text: str) -> str:
    digits = re.sub(r"[.,](?=\d{3}\b)", "", text).replace(",", ".")
    try:
        value = float(digits)
        return str(int(value)) if value == int(value) else str(value)
    except ValueError:
        return text


def _same(a: list[str], b: list[str]) -> bool:
    """One fact said twice: the same words in the same order once filler words are set aside (a negation is never filler)."""
    return [t for t in a if t not in FILLER] == [t for t in b if t not in FILLER]


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def usage(log: Any) -> tuple[Counter, Counter]:
    """How many briefs showed each item, and how many replies cited it: every memory event, never only the newest few."""
    shown, cited = Counter(), Counter()
    events = log.read(kinds=("memory.shown", "memory.cited"), limit=10**9) if log is not None else []
    for e in events:
        if e.kind == "memory.shown":
            shown.update(e.payload.get("items") or [])
        elif e.kind == "memory.cited":
            cited.update(e.payload.get("items") or [])
    return shown, cited


def plan(store: MemoryStore, log: Any = None) -> dict[str, list[dict[str, Any]]]:
    """What upkeep would do now: merges, retirements and conflicts, by item id (nothing written). Only a line said again
    in the same words is merged; any other near pair (another number, a negation, another name, swapped words) is a
    conflict for a person. A person's line is never merged into an agent's, and never retired for not being cited."""
    active = [i for i in latest(store.log()) if i.status == "active"]
    shown, cited = usage(log)
    merge, conflicts, gone = [], [], set()
    by_scope: dict[tuple[str, str], list[MemoryItem]] = {}
    for i in active:
        by_scope.setdefault((i.scope, i.scope_id), []).append(i)
    for items in by_scope.values():
        items = sorted(items, key=lambda i: (i.created_at, i.id))
        for n, a in enumerate(items):
            for b in items[n + 1:]:
                if a.id in gone or b.id in gone:
                    continue
                ta, tb = _tokens(a.text), _tokens(b.text)
                mergeable = (a.kind == b.kind and not a.correction and not b.correction
                             and not (a.who.startswith("person:") and not b.who.startswith("person:")))
                if _same(ta, tb):
                    if mergeable:
                        merge.append({"item": a.id, "into": b.id})
                        gone.add(a.id)
                    continue
                words_a = {t for t in ta if not NUMBER.fullmatch(t)}
                words_b = {t for t in tb if not NUMBER.fullmatch(t)}
                if _overlap(words_a, words_b) >= CONFLICT or (words_a == words_b and ta != tb):
                    conflicts.append({"items": [a.id, b.id], "texts": [a.text, b.text], "why": "much the same words, saying something different"})
    retire_ = []
    for i in active:
        if i.id in gone or i.correction:
            continue
        if i.counters.harmful >= HARMFUL:
            retire_.append({"item": i.id, "reason": "harmful"})
        elif shown[i.id] >= RETIRE_AFTER and not cited[i.id] and not i.who.startswith("person:"):
            retire_.append({"item": i.id, "reason": "never cited", "shown": shown[i.id]})
    return {"merge": merge, "retire": retire_, "conflicts": conflicts}


def apply(store: MemoryStore, todo: dict[str, list[dict[str, Any]]], log: Any = None) -> int:
    """Write the plan's merges and retirements (conflicts stay for a person); how many items were retired."""
    state = {i.id: i for i in latest(store.log())}
    done = 0
    for m in todo.get("merge") or []:
        item = state.get(m["item"])
        if item is not None and item.status == "active":
            store.write(retire(item, by=m["into"]))
            event(log, "memory.retired", item, reason="merged", by="rule:upkeep", by_item=m["into"])
            done += 1
    for r in todo.get("retire") or []:
        item = state.get(r["item"])
        if item is not None and item.status == "active":
            store.write(retire(item))
            event(log, "memory.retired", item, reason=r["reason"], by="rule:upkeep")
            done += 1
    return done


__all__ = ["RETIRE_AFTER", "HARMFUL", "CONFLICT", "FILLER", "NEGATION", "usage", "plan", "apply"]
