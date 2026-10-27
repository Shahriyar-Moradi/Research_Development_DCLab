"""Writing to memory (package 35.3): every way an item enters, changes or leaves, each an event.

- :func:`record_fact`: a fact or an episode with its source is written at once (a saved solution, a person's statement).
- :func:`draft_know_how`: know-how is a draft until reviewed (:func:`review`: accept, edit or reject, by a reviewer's
  role, as lessons are reviewed); a draft is never shown to an agent.
- :func:`correct`: a person's correction, a new item superseding the old one (stored or read where it lives); a stored
  old item is retired, named by the correction.
- :func:`propose_preference`: a preference heard in a person's own words ("from now on, answer simply") is proposed as
  a draft fact of that person, never stored silently; only that person (or the owner) confirms it.

Text from a person or an agent becomes one line first: the first line only (the rest dropped, and said), cleaned of
invisible marks and cut to the contract's length, so a second line can never ride in as an instruction. Every change is
an event (``memory.written``, ``memory.proposed``, ``memory.retired``) with the item's id, scope and source.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from dclab_core.memory import MemoryItem, item_id, new, retire

from . import MemoryStore, Refused, event, latest, remember, secret
from .readers import line

REVIEWERS = ("reviewer", "owner")  # who reviews know-how (the lessons' reviewers)
BREAKS = re.compile("[\n\r\u2028\u2029\u0085\v\f\x1c\x1d\x1e]")  # every line end Python's splitlines knows
_HOW = r"(answer|reply|respond|write|explain|talk|speak|use|give|keep|show|ask|be|say)"
PREFERENCE = re.compile(  # a person telling the agents how to work with them, in English or Persian
    r"^\s*(please\s+)?(from now on\b|i prefer\b|i[\u2019']d prefer\b|i would prefer\b|"
    r"(always|never)\s+" + _HOW + r"\b|i want you to\s+" + _HOW + r"\b|i[\u2019']d like you to\s+" + _HOW + r"\b|"
    r"(answer|reply|respond|write)\s+(in|with)\b|keep (your )?(answers?|replies)\b|use simple\b|be (brief|short|critical|honest)\b)"
    r"|^\s*(\u0644\u0637\u0641\u0627|\u0644\u0637\u0641\u0627\u064b)?\s*(\u0627\u0632 \u0627\u06cc\u0646 \u0628\u0647 \u0628\u0639\u062f|\u0647\u0645\u06cc\u0634\u0647|\u0647\u0631\u06af\u0632)"
    r"|\u062a\u0631\u062c\u06cc\u062d \u0645\u06cc\u200c?\u062f\u0647\u0645", re.IGNORECASE)


class ReviewError(ValueError):
    """A review or a correction the memory does not take."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def first_line(text: Any) -> tuple[str, bool]:
    """The first non-empty line, cleaned (one visible line, at most the contract's length), and whether more was dropped."""
    parts = [p for p in BREAKS.split(str(text or "")) if p.strip()]
    if not parts:
        return "", False
    return line(parts[0]), len(parts) > 1


def _text(text: Any) -> str:
    kept, _ = first_line(text)
    if not kept:
        raise Refused("no words to remember")
    return kept


def record_fact(store: MemoryStore, *, scope: str, scope_id: str, text: Any, source: str, who: str, kind: str = "fact",
                at: str | None = None, protected: Iterable[str] = (), log: Any = None) -> MemoryItem:
    """A sourced fact (or episode) written at once."""
    if kind not in ("fact", "episode"):
        raise Refused("only facts and episodes are written at once: know-how is drafted for review")
    item = remember(store, new(kind=kind, scope=scope, scope_id=scope_id, text=_text(text), source=source, who=who, at=at or _now()), protected)
    event(log, "memory.written", item)
    return item


def draft_know_how(store: MemoryStore, *, scope: str, scope_id: str, text: Any, source: str, who: str, at: str | None = None,
                   protected: Iterable[str] = (), log: Any = None) -> MemoryItem:
    """Know-how as a draft: stored, never shown, until a reviewer accepts it."""
    item = remember(store, new(kind="know_how", scope=scope, scope_id=scope_id, text=_text(text), source=source, who=who,
                               at=at or _now(), status="draft"), protected)
    event(log, "memory.proposed", item, origin="agent")
    return item


def propose_preference(store: MemoryStore, *, person: str, message: Any, source: str, at: str | None = None, log: Any = None) -> MemoryItem | None:
    """A draft fact of the person when their words state how they want to be worked with; None otherwise."""
    lines = [line(p) for p in BREAKS.split(str(message or "")) if p.strip()]
    kept = next((p for p in lines if p and PREFERENCE.search(p) and not secret(p)), "")  # the first line that states one
    if not kept:
        return None
    item = new(kind="fact", scope="person", scope_id=person, text=kept, source=source, who=f"person:{person}", at=at or _now(), status="draft")
    held = next((v for v in latest(store.log()) if v.id == item.id), None)
    if held is not None:  # already proposed, confirmed or retired: never a second proposal of the same words
        return held if held.status == "draft" else None
    store.write(item)
    event(log, "memory.proposed", item, origin="chat")
    return item


def _draft(store: MemoryStore, item_id: str) -> MemoryItem:
    found = next((v for v in latest(store.log()) if v.id == item_id), None)
    if found is None:
        raise KeyError(item_id)
    if found.status != "draft":
        raise ReviewError(f"only a draft is reviewed; this item is {found.status}")
    return found


def review(store: MemoryStore, item_id: str, action: str, *, by: str, role: str | None, text: Any = None, at: str | None = None,
           log: Any = None) -> MemoryItem:
    """A draft accepted, edited or rejected. Know-how needs a reviewer's role; a person's preference, that person or the owner."""
    draft = _draft(store, item_id)
    if draft.scope == "person":  # a person's preference: that person confirms or edits it; the owner may only accept or reject it
        if not (draft.scope_id == by or (role == "owner" and action in ("accept", "reject"))):
            raise PermissionError("a person's preference is confirmed by that person (the owner may only accept or reject it)")
    elif role not in REVIEWERS:
        raise PermissionError("know-how is reviewed by a reviewer or the owner")
    if action == "accept":
        accepted = MemoryItem.model_validate({**draft.model_dump(), "status": "active"})
        store.write(accepted)
        event(log, "memory.written", accepted, by=f"person:{by}", reason="accepted")
        return accepted
    if action == "reject":
        gone = retire(draft)
        store.write(gone)
        event(log, "memory.retired", gone, reason="rejected", by=f"person:{by}")
        return gone
    if action == "edit":
        edited = new(kind=draft.kind, scope=draft.scope, scope_id=draft.scope_id, text=_text(text), source=draft.source, who=f"person:{by}",
                     at=at or _now())
        if edited.id == draft.id:
            raise ReviewError("an edit changes the words: accept it as it is instead")
        remember(store, edited)  # the door's checks: no secret, no retired words stored unseen
        store.write(retire(draft, by=edited.id))
        event(log, "memory.written", edited, reason="edited")
        event(log, "memory.retired", draft, reason="edited", by=f"person:{by}", by_item=edited.id)
        return edited
    raise ReviewError("a review accepts, edits or rejects")


def correct(store: MemoryStore, old: MemoryItem, text: Any, *, by: str, role: str | None = None, reason: str = "", at: str | None = None,
            protected: Iterable[str] = (), log: Any = None) -> MemoryItem:
    """A person's correction of an active item (stored, or read where it lives): a new item superseding it; a stored one
    retired. A draft is reviewed, never corrected; know-how is corrected by a reviewer only (it is reviewed knowledge)."""
    if old.status != "active":
        raise ReviewError("only an active item is corrected; a draft is reviewed")
    if old.kind == "know_how" and role not in REVIEWERS:
        raise PermissionError("know-how is corrected by a reviewer or the owner")
    words = _text(text)
    if item_id(old.kind, old.scope, old.scope_id, words) == old.id:
        raise ReviewError("a correction changes the words")
    fixed = new(kind=old.kind, scope=old.scope, scope_id=old.scope_id, text=words, source=f"correction:{old.id}", who=f"person:{by}",
                at=at or _now(), correction=True, supersedes=old.id)
    remember(store, fixed, protected)
    event(log, "memory.written", fixed, reason=reason)
    held = next((v for v in latest(store.log()) if v.id == old.id), None)
    if held is not None and held.status != "retired":
        store.write(retire(held, by=fixed.id))
        event(log, "memory.retired", held, reason="corrected", by=f"person:{by}", by_item=fixed.id)
    return fixed


__all__ = ["REVIEWERS", "ReviewError", "first_line", "record_fact", "draft_know_how", "propose_preference", "review", "correct"]
