"""One memory for DCLab's agents, stored (package 35.2, docs/guides/MEMORY_PLAN.md); the contract is ``dclab_core.memory``.

The store is an append-only log of item versions: a write, a retirement, a counter moved or a correction is a new
version, and the last version of an id is its state (a retired version is final). It is a JSONL file in the workspace
folder (``memory/items.jsonl``), or the ``memory_item_version`` table when ``DCLAB_DATABASE_URL`` is set; the two answer
the same calls. Nothing is ever deleted: deleting a project retires its items (:func:`retire_scope`), and the versions
before stay.

Project notes and accepted lessons are not copied in: :mod:`dclab_rnd.memory.readers` reads them as items where they
live, so each keeps one source of truth and nothing is written twice. :func:`view` is what a reader sees: the store's
items and the read ones, through ``dclab_core.memory.resolve``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Protocol

from dclab_core.memory import MemoryItem, check, resolve, retire

FILE = Path("memory") / "items.jsonl"


class Refused(ValueError):
    """An item the memory does not take."""


class MemoryStore(Protocol):
    def write(self, item: MemoryItem) -> MemoryItem: ...
    def write_many(self, items: list[MemoryItem]) -> list[MemoryItem]: ...  # all or none
    def log(self) -> list[MemoryItem]: ...  # every version, in the order written


def latest(versions: Iterable[MemoryItem]) -> list[MemoryItem]:
    """One item per id: its last version, or its retired one (retirement is final), in the order ids first appeared."""
    state: dict[str, MemoryItem] = {}
    for v in versions:
        held = state.get(v.id)
        if held is not None and held.status == "retired" and v.status != "retired":
            continue
        state[v.id] = v
    return list(state.values())


class FileMemory:
    """``memory/items.jsonl`` in the workspace folder: one version a line, appended and flushed to disk."""

    def __init__(self, home: Path):
        self.path = Path(home) / FILE  # the folder is made by the first write, never by reading
        self.unreadable = 0

    def _held(self):
        from ..filelock import held

        return held(self.path.parent)  # between processes too (the API and the worker write the same file)

    def write(self, item: MemoryItem) -> MemoryItem:
        return self.write_many([item])[0]

    def write_many(self, items: list[MemoryItem]) -> list[MemoryItem]:
        text = "".join(i.model_dump_json() + "\n" for i in items)
        with self._held(), self.path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():  # a line cut by a crash has no line break: end it, so this write is a line of its own
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        return items

    def log(self) -> list[MemoryItem]:
        out, bad = [], 0
        if not self.path.is_file():
            return []
        with self._held():
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                out.append(MemoryItem.read(line))
            except (ValueError, json.JSONDecodeError):  # a line cut by a crash, or one edited by hand: skipped and counted, never fatal
                bad += 1
        if bad and bad != self.unreadable:
            print(f"memory: {bad} unreadable line(s) in {self.path.name} skipped", file=sys.stderr, flush=True)
        self.unreadable = bad
        return out


class PgMemory:
    """The ``memory_item_version`` table, one row a version (append-only: the database refuses an update or a delete)."""

    def __init__(self, workspace_id: str, url: str | None = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def write(self, item: MemoryItem) -> MemoryItem:
        return self.write_many([item])[0]

    def write_many(self, items: list[MemoryItem]) -> list[MemoryItem]:
        """Every version in one transaction: all of them or none."""
        from ..storage.models import memory_item_version as t

        rows = [{"workspace_id": self.workspace_id, "item_id": i.id, "kind": i.kind, "scope": i.scope, "scope_id": i.scope_id,
                 "status": i.status, "doc": json.loads(i.model_dump_json())} for i in items]
        if rows:
            with self.engine.begin() as c:
                c.execute(t.insert(), rows)
        return items

    def log(self) -> list[MemoryItem]:
        import sqlalchemy as sa

        from ..storage.models import memory_item_version as t

        with self.engine.connect() as c:
            rows = c.execute(sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id).order_by(t.c.seq)).scalars().all()
        return [MemoryItem.read(r) for r in rows]


def open_memory(home: Path) -> MemoryStore:
    """The workspace's memory: the database when one is set, else the folder's file."""
    from ..storage import db

    if db.database_url(required=False):
        return PgMemory(db.workspace_for(Path(home)))
    return FileMemory(Path(home))


def remember(store: MemoryStore, item: MemoryItem, protected: Iterable[str] = ()) -> MemoryItem:
    """Write one item after the contract's screen (a protected column's name never enters memory)."""
    why = check(item, tuple(protected))
    if why:
        raise Refused(why)
    held = next((v for v in latest(store.log()) if v.id == item.id), None)
    if held is not None and held.status == "retired":  # the same words again would be stored and never seen: say so
        raise Refused("these words were retired: write a correction of what replaced them, in other words")
    return store.write(item)


def retire_scope(store: MemoryStore, scope: str, scope_id: str) -> int:
    """Every item of one scope retired, as new versions (a deleted project's items): how many. The versions before stay."""
    gone = [retire(i) for i in latest(store.log()) if i.scope == scope and i.scope_id == scope_id and i.status != "retired"]
    store.write_many(gone)
    return len(gone)


def view(store: MemoryStore | None, chain: list[tuple[str, str]], read: Iterable[MemoryItem] = ()) -> list[MemoryItem]:
    """What a reader with these (scope, id) pairs sees: the store's items and the ones read where they live."""
    read = list(read)
    mine = {i.id for i in read}  # a note or a lesson read where it lives is the truth: a retired stored copy of its words never hides it
    stored = [i for i in (latest(store.log()) if store is not None else []) if not (i.status == "retired" and i.id in mine)]
    return resolve([*stored, *read], chain)


__all__ = ["FILE", "Refused", "MemoryStore", "latest", "FileMemory", "PgMemory", "open_memory", "remember", "retire_scope", "view"]
