"""Consent as a level stamped on every record at write time (package 23.1).

A workspace (and a project inside it) allows two uses, each a switch: **improve** (what happened may be read and turned
into evaluation cases) and **train** (it may become training data). Read together they are one level on a ladder:

    operate  < improve < train

``operate`` is every workspace's default: the record is kept to run the product, for support and for audit, and never
leaves. Each event and each decision record carries the level it was written under (``consent``), so a later change of
the switches never re-labels old data: an event written at ``operate`` never enters a dataset, however the workspace
opts in afterwards. A host that stores its own rows passes the level it reads from its settings when it maps them
(``dclab_core.adapters.main_product_experience.events``); a row with no level is ``operate``.

A change of a switch is itself an event (``consent.changed``: the scope, the use, the old and new value, the level now,
and the version of the notice the person was shown), so the history of consent is in the record beside everything it
governs.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

Level = Literal["operate", "improve", "train"]
LEVELS: tuple[str, ...] = ("operate", "improve", "train")
USES: tuple[str, ...] = ("improve", "train")
LADDER = {"operate": 0, "improve": 1, "train": 2}
NOTICE_VERSION = "2026-10-16"  # the version of the notice that lists every stream kept (docs/guides/EXPERIENCE_ARCHIVE.md)


def level_of(workspace: Mapping[str, Any], project: Optional[Mapping[str, Any]] = None) -> str:
    """The highest use the switches allow now: the workspace's, and the project's when there is one (a use is on only
    when both allow it; a session with no project follows the workspace alone)."""

    def on(use: str) -> bool:
        return workspace.get(use) is True and (project is None or project.get(use) is True)

    if on("train"):
        return "train"
    if on("improve"):
        return "improve"
    return "operate"


def exportable(level: Any, use: str) -> bool:
    """Whether a record stamped ``level`` may leave for ``use``. ``train`` allows both uses, ``improve`` one, ``operate``
    none; an unknown stamp allows nothing."""
    if use not in USES:
        raise ValueError(f"use is one of {', '.join(USES)}")
    return LADDER.get(str(level), -1) >= LADDER[use]


def changes(before: Mapping[str, Any], after: Mapping[str, Any], scope: str = "workspace",
            notice_version: str = NOTICE_VERSION) -> list[dict[str, Any]]:
    """The ``consent.changed`` payloads for the switches that moved between ``before`` and ``after`` (meta only)."""
    out = []
    for use in USES:
        was, now = before.get(use) is True, after.get(use) is True
        if was != now:
            out.append({"scope": scope, "use": use, "on": now, "was": was, "level": level_of(after), "notice_version": notice_version})
    return out


__all__ = ["Level", "LEVELS", "USES", "LADDER", "NOTICE_VERSION", "level_of", "exportable", "changes"]
