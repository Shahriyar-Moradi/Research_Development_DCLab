"""A project's memory notes survive every write of the project document.

Jobs (a stage run, an improvement run) read a project, work for minutes, and save the whole document. A note a person
or the chat added meanwhile would be dropped, and a note removed meanwhile would come back. Notes are only ever added
or marked removed (``studio.memory``), never deleted, so every write merges them by id with what is stored: a note in
either is kept, and a removal on either side wins.
"""

from __future__ import annotations

from typing import Any, Optional


def merge(stored: Optional[list[dict[str, Any]]], incoming: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The notes to write: the stored ones in their order (each updated from the incoming copy), then the new ones."""
    stored, incoming = list(stored or []), list(incoming or [])
    if not stored:
        return incoming
    new = {n.get("id"): n for n in incoming if isinstance(n, dict) and n.get("id")}
    out, seen = [], set()
    for old in stored:
        if not isinstance(old, dict) or not old.get("id"):
            continue
        note = dict(new.get(old["id"], old))
        if old.get("removed") and not note.get("removed"):  # removed meanwhile: the removal wins
            note["removed"] = old["removed"]
        out.append(note)
        seen.add(old["id"])
    out += [n for n in incoming if isinstance(n, dict) and n.get("id") not in seen]
    return out


def into(doc: dict[str, Any], stored: Optional[dict[str, Any]]) -> dict[str, Any]:
    """``doc`` with its memory merged with the stored document's (unchanged when neither has notes)."""
    if stored and (stored.get("memory") or doc.get("memory")):
        doc["memory"] = merge(stored.get("memory"), doc.get("memory"))
    return doc


__all__ = ["merge", "into"]
