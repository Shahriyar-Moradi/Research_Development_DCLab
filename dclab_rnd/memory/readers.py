"""Memory read where it lives (35.2): a project's notes (``studio/memory.py``) as ``fact`` items of the project, and the
workspace's accepted lessons (``lessons.py``) as ``know_how`` items of the workspace. Nothing is copied into the store:
an item read here has the same id every time it is read, so it is one item, and a deleted project's notes simply stop
being read.

A note's or a lesson's text is made one visible line first (line breaks and invisible marks become spaces, Persian's
joiners kept) and cut to the contract's length; a text left with no word is not read.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from dclab_core.memory import JOINERS, TEXT, MemoryItem, new

PEOPLE = ("human", "person", "owner", "reviewer")  # the authors studio notes and lesson reviews write for a person


def line(text: Any) -> str:
    """One visible line of at most ``TEXT`` characters ("" when no word is left)."""
    kept = "".join(" " if unicodedata.category(c) in ("Cc", "Zl", "Zp") or (unicodedata.category(c) == "Cf" and c not in JOINERS) else c
                   for c in str(text or ""))
    kept = " ".join(kept.split())
    if len(kept) > TEXT:
        kept = kept[:TEXT - 1].rstrip() + "…"
    return kept if re.search(r"\w", kept) else ""


def author(who: Any) -> str:
    who = str(who or "").strip() or "unknown"
    if who.startswith(("person:", "agent:", "rule:")):
        return who[:120]
    return (f"person:{who}" if who.split(":")[0] in PEOPLE else f"agent:{who}")[:120]


def from_project(project: dict[str, Any]) -> list[MemoryItem]:
    """A project's notes as facts of the project. A removed note is not read at all: the notes are the source of truth,
    and a retired copy would hide the same words written again later (an id is its words)."""
    pid = str(project.get("id") or "")
    if not pid:
        return []
    out = []
    for note in project.get("memory") or []:
        if note.get("removed"):
            continue
        text = line(". ".join(t for t in (note.get("decision"), note.get("reason")) if t))
        if not text or not note.get("at"):
            continue
        who = author(note.get("who"))
        try:
            out.append(new(kind="fact", scope="project", scope_id=pid, text=text, source=f"note:{pid}:{note.get('id')}", who=who,
                           at=note["at"], correction=note.get("kind") == "correction" and who.startswith("person:")))
        except ValueError:  # a note the contract cannot hold (a time without its zone): left out, never an error for the reader
            continue
    return out


def from_lessons(lessons: Iterable[dict[str, Any]], workspace_id: str) -> list[MemoryItem]:
    """The workspace's accepted lessons as its know-how."""
    out = []
    for lesson in lessons:
        if lesson.get("status") != "accepted":
            continue
        text = line(lesson.get("claim"))
        at = lesson.get("updated") or lesson.get("proposed_at") or lesson.get("created")
        if not text or not at:
            continue
        try:
            out.append(new(kind="know_how", scope="workspace", scope_id=workspace_id, text=text, source=f"lesson:{lesson.get('id')}",
                           who=author((lesson.get("review") or {}).get("by") or "reviewer"), at=at))
        except ValueError:
            continue
    return out


__all__ = ["PEOPLE", "line", "author", "from_project", "from_lessons"]
