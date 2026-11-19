"""Problem memory's bases and summaries: JSON files in the workspace folder, or PostgreSQL when DCLAB_DATABASE_URL is set.

``dclab_rnd/memory/bases.py`` and ``summary.py`` keep every rule (revisions, stale marks, the cursor); this module only
keeps their documents, in one of two places that behave the same:

- Files (``FileMemoryBases``): ``memory/bases/<scope>_<hash>.json`` and ``memory/summaries/sum_<hash>.json``, each
  folder behind its lock (``filelock.held``: a thread lock and ``flock``, so one host only). Unchanged on disk.
- PostgreSQL (``PgMemoryBases``, migration 0024): one ``memory_base`` row a base and one ``memory_summary`` row a
  summary, so every API and worker host on the database reads and writes the same documents.

How a base change stays one change on PostgreSQL: ``change`` takes a transaction-scoped advisory lock on the scope (so
two hosts never run the caller's step on the same base at once, and the memory items that step writes are written once),
reads the live row, and updates it only where its ``version`` is still the one read; zero rows updated is
``BaseChanged``, never a lost write. A first write inserts the row, and the unique index on the live scope refuses a
second. A purge sets ``purged_at`` (the tombstone hides the row from every read); a trigger refuses DELETE.

How a summary stays monotonic: ``publish`` is one ``INSERT … ON CONFLICT DO UPDATE … WHERE stored.cursor <= new.cursor``,
so a late worker's older cursor never replaces a newer summary, whichever host it runs on. Summaries are derived, so
``invalidate`` deletes them (the next read rebuilds), as the files are deleted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

Document = dict[str, Any]
Change = Callable[[Document], "tuple[bool, Any]"]  # the caller's step: changes the document in place, says whether to write it

BASES_DIR = Path("memory") / "bases"
PURGED_DIR = BASES_DIR / "purged"
SUMMARY_DIR = Path("memory") / "summaries"
EMPTY_CURSOR = (0, 0, 0, 0, 0)


class BaseChanged(RuntimeError):
    """A base row changed between its read and its write (only a purge can do that while a change holds its lock)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summary_key(workspace_id: str, project_id: str, problem_id: str, visibility_class: str = "shared") -> str:
    return f"{workspace_id}:{project_id}:{problem_id}:{visibility_class}"


def _cursor(value: Iterable[Any] | None) -> tuple[int, ...]:
    return tuple(int(x) for x in (value or EMPTY_CURSOR))


# ---------------------------------------------------------------------------------------------------------- files
class FileMemoryBases:
    """Today's files: one JSON document a base or a summary, each folder behind its lock."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.bases_dir = self.home / BASES_DIR
        self.summary_dir = self.home / SUMMARY_DIR

    def location(self) -> str:
        return f"files in {self.home / 'memory'}"

    # ------------------------------------------------------------------ bases
    def _base_file(self, scope: str, scope_id: str) -> Path:
        digest = hashlib.sha256(f"{scope}:{scope_id}".encode("utf-8")).hexdigest()[:24]
        return self.bases_dir / f"{scope}_{digest}.json"

    @staticmethod
    def _load(path: Path, scope: str, scope_id: str) -> Optional[Document]:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) and doc.get("scope") == scope and doc.get("scope_id") == scope_id else None

    @staticmethod
    def _write(path: Path, doc: Document) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.part")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def read(self, scope: str, scope_id: str) -> Optional[Document]:
        from ..filelock import held

        path = self._base_file(scope, scope_id)
        if not path.is_file():
            return None
        with held(self.bases_dir):
            return self._load(path, scope, scope_id)

    def change(self, scope: str, scope_id: str, blank: Document, fn: Change) -> Any:
        from ..filelock import held

        path = self._base_file(scope, scope_id)
        with held(self.bases_dir):
            doc = (self._load(path, scope, scope_id) if path.is_file() else None) or copy.deepcopy(blank)
            write, value = fn(doc)
            if write:
                self._write(path, doc)
        return value

    def purge(self, belongs: Callable[[Document], bool], reason: str) -> list[Document]:
        """Every live base ``belongs`` picks moves to ``bases/purged/`` with a tombstone: never read again, never deleted."""
        from ..filelock import held

        out: list[Document] = []
        if not self.bases_dir.is_dir():
            return out
        with held(self.bases_dir):
            for path in sorted(self.bases_dir.glob("*.json")):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(doc, dict) or not belongs(doc):
                    continue
                folder = path.parent / "purged"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / path.name
                if target.exists():  # the same scope purged before: keep both
                    target = folder / f"{path.stem}.{secrets.token_hex(3)}{path.suffix}"
                self._write(target, {**doc, "purged": {"at": _now(), "reason": reason, "from": path.name}})
                path.unlink()
                out.append(doc)
        return out

    def all_bases(self) -> list[tuple[str, Document]]:
        """Every base, live and purged, with its path under ``memory/bases`` (what a move to the database copies)."""
        out = []
        for path in sorted(self.bases_dir.glob("*.json")) + sorted((self.bases_dir / "purged").glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("scope") and doc.get("scope_id"):
                out.append((path.relative_to(self.bases_dir).as_posix(), doc))
        return out

    # ------------------------------------------------------------------ summaries
    def _summary_file(self, key: str) -> Path:
        return self.summary_dir / f"sum_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}.json"

    def summary(self, key: str) -> Optional[Document]:
        path = self._summary_file(key)
        if not path.is_file():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def publish(self, key: str, doc: Document, before: Callable[[], Optional[Document]] | None = None) -> Document:
        """Write ``doc`` unless ``before`` refuses or the stored summary's cursor is newer; one step under the folder's lock."""
        from ..filelock import held

        path = self._summary_file(key)
        candidate = _cursor(doc.get("cursor"))
        with held(self.summary_dir):
            refused = before() if before is not None else None
            if refused is not None:
                return refused
            if path.is_file():
                try:
                    stored = _cursor(json.loads(path.read_text(encoding="utf-8")).get("cursor"))
                    if candidate < stored:
                        return {"published": False, "reason": "older_than_stored_summary", "candidate_cursor": list(candidate),
                                "stored_cursor": list(stored)}
                except (OSError, ValueError, AttributeError):
                    pass
            self._write(path, dict(doc))
        return {"published": True, "cursor": list(candidate)}

    def invalidate(self, project_id: str, problem_id: Optional[str] = None) -> int:
        """Delete the derived summaries of a project (and problem), matched on the raw ids whatever workspace id they carry."""
        from ..filelock import held

        if not self.summary_dir.is_dir() or not project_id:
            return 0
        removed = 0
        with held(self.summary_dir):
            for p in list(self.summary_dir.glob("sum_*.json")):
                try:
                    doc = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if str(doc.get("project_id") or "") != str(project_id):
                    continue
                if problem_id and str(doc.get("problem_id") or "") != str(problem_id):
                    continue
                p.unlink()
                removed += 1
        return removed

    def summaries(self) -> list[tuple[str, Document]]:
        out = []
        for path in sorted(self.summary_dir.glob("sum_*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("project_id") and doc.get("problem_id"):
                key = summary_key(str(doc.get("workspace_id")), str(doc["project_id"]), str(doc["problem_id"]), str(doc.get("visibility_class") or "shared"))
                out.append((key, doc))
        return out

    def counts(self) -> dict[str, int]:
        return {"bases": len(self.all_bases()), "summaries": len(self.summaries())}


# ---------------------------------------------------------------------------------------------------------- PostgreSQL
class PgMemoryBases:
    """The ``memory_base`` and ``memory_summary`` rows of one workspace."""

    def __init__(self, workspace_id: str, url: str | None = None):
        from . import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def location(self) -> str:
        return f"PostgreSQL workspace {self.workspace_id}"

    # ------------------------------------------------------------------ bases
    def _lock(self, c: Any, scope: str, scope_id: str) -> None:
        """One base at a time across hosts, until the transaction ends (commit or rollback releases it)."""
        import sqlalchemy as sa

        digest = hashlib.sha256(f"memory_base:{self.workspace_id}:{scope}:{scope_id}".encode("utf-8")).digest()
        c.execute(sa.text("select pg_advisory_xact_lock(:key)"), {"key": int.from_bytes(digest[:8], "big", signed=True)})

    def _live(self, scope: str, scope_id: str):
        import sqlalchemy as sa

        from .models import memory_base as t

        return (sa.select(t.c.id, t.c.version, t.c.doc)
                .where(t.c.workspace_id == self.workspace_id, t.c.scope == scope, t.c.scope_id == scope_id, t.c.purged_at.is_(None)))

    def read(self, scope: str, scope_id: str) -> Optional[Document]:
        with self.engine.connect() as c:
            row = c.execute(self._live(scope, scope_id)).first()
        return dict(row.doc) if row is not None else None

    @staticmethod
    def _columns(doc: Document) -> dict[str, Any]:
        return {"revision": int(doc.get("revision") or 0), "project_id": doc.get("project_id"), "problem_id": doc.get("problem_id")}

    def change(self, scope: str, scope_id: str, blank: Document, fn: Change) -> Any:
        import sqlalchemy as sa

        from .models import memory_base as t

        with self.engine.begin() as c:
            self._lock(c, scope, scope_id)
            row = c.execute(self._live(scope, scope_id)).first()
            doc = dict(row.doc) if row is not None else copy.deepcopy(blank)
            write, value = fn(doc)  # an exception here rolls back and releases the lock: nothing is written
            if write:
                if row is None:
                    c.execute(sa.insert(t).values(id=uuid.uuid4().hex, workspace_id=self.workspace_id, scope=scope, scope_id=scope_id,
                                                  doc=doc, **self._columns(doc)))
                else:
                    done = c.execute(sa.update(t).where(t.c.id == row.id, t.c.version == row.version, t.c.purged_at.is_(None))
                                     .values(doc=doc, version=t.c.version + 1, updated=sa.func.now(), **self._columns(doc)))
                    if done.rowcount != 1:
                        raise BaseChanged(f"the {scope} base changed while it was being written; nothing was written")
        return value

    def purge(self, belongs: Callable[[Document], bool], reason: str) -> list[Document]:
        """Tombstone every live base ``belongs`` picks (``purged_at``): hidden from every read, the row kept."""
        import sqlalchemy as sa

        from .models import memory_base as t

        out: list[Document] = []
        with self.engine.begin() as c:
            rows = c.execute(sa.select(t.c.id, t.c.scope, t.c.scope_id, t.c.doc)
                             .where(t.c.workspace_id == self.workspace_id, t.c.purged_at.is_(None))
                             .order_by(t.c.scope, t.c.scope_id)).all()  # one lock order, so two purges never wait on each other
            for row in rows:
                if not belongs(row.doc):
                    continue
                self._lock(c, row.scope, row.scope_id)
                held = c.execute(sa.select(t.c.doc).where(t.c.id == row.id, t.c.purged_at.is_(None))).scalar()  # as it is now, once locked
                if not isinstance(held, dict):
                    continue
                doc = {**held, "purged": {"at": _now(), "reason": reason, "from": row.id}}
                done = c.execute(sa.update(t).where(t.c.id == row.id, t.c.purged_at.is_(None))
                                 .values(doc=doc, purged_at=sa.func.now(), purge_reason=reason, version=t.c.version + 1, updated=sa.func.now()))
                if done.rowcount == 1:
                    out.append(dict(held))
        return out

    def import_base(self, row_id: str, doc: Document) -> bool:
        """One base from a file workspace (``storage migrate``); a row already there, or a live base of the same scope the
        database workspace already has, is left as it is. True when copied."""
        from sqlalchemy.dialects.postgresql import insert

        from .models import memory_base as t

        purged = doc.get("purged") if isinstance(doc.get("purged"), dict) else None
        when = None
        if purged:
            try:
                when = datetime.fromisoformat(str(purged.get("at")))
            except ValueError:
                when = datetime.now(timezone.utc)
        with self.engine.begin() as c:
            done = c.execute(insert(t).values(id=row_id, workspace_id=self.workspace_id, scope=str(doc["scope"]), scope_id=str(doc["scope_id"]),
                                              doc=doc, purged_at=when, purge_reason=(str(purged.get("reason") or "") if purged else None),
                                              **self._columns(doc)).on_conflict_do_nothing().returning(t.c.id))
            return done.first() is not None  # an INSERT's rowcount is not reliable here: the returned row is

    def all_bases(self) -> list[tuple[str, Document]]:
        import sqlalchemy as sa

        from .models import memory_base as t

        with self.engine.connect() as c:
            rows = c.execute(sa.select(t.c.id, t.c.doc).where(t.c.workspace_id == self.workspace_id).order_by(t.c.updated, t.c.id)).all()
        return [(r.id, dict(r.doc)) for r in rows]

    # ------------------------------------------------------------------ summaries
    def summary(self, key: str) -> Optional[Document]:
        import sqlalchemy as sa

        from .models import memory_summary as t

        with self.engine.connect() as c:
            doc = c.execute(sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id, t.c.key == key)).scalar()
        return dict(doc) if isinstance(doc, dict) else None

    def _upsert(self, key: str, doc: Document, guarded: bool):
        from sqlalchemy.dialects.postgresql import insert

        from .models import memory_summary as t

        stmt = insert(t).values(workspace_id=self.workspace_id, key=key, project_id=doc.get("project_id"), problem_id=doc.get("problem_id"),
                                visibility_class=str(doc.get("visibility_class") or "shared"), cursor=list(_cursor(doc.get("cursor"))), doc=dict(doc))
        if not guarded:
            return stmt.on_conflict_do_nothing().returning(t.c.key)
        new = stmt.excluded
        import sqlalchemy as sa

        return stmt.on_conflict_do_update(
            index_elements=[t.c.workspace_id, t.c.key],
            set_={"doc": new.doc, "cursor": new.cursor, "project_id": new.project_id, "problem_id": new.problem_id,
                  "visibility_class": new.visibility_class, "updated": sa.func.now()},
            where=t.c.cursor <= new.cursor).returning(t.c.key)  # the compare and the swap in one statement: arrays compare in order

    def publish(self, key: str, doc: Document, before: Callable[[], Optional[Document]] | None = None) -> Document:
        """Write ``doc`` only when its cursor is not older than the stored one: one conditional upsert, atomic across hosts."""
        import sqlalchemy as sa

        from .models import memory_summary as t

        candidate = _cursor(doc.get("cursor"))
        refused = before() if before is not None else None
        if refused is not None:
            return refused
        with self.engine.begin() as c:
            if c.execute(self._upsert(key, doc, guarded=True)).first() is not None:
                return {"published": True, "cursor": list(candidate)}
            stored = c.execute(sa.select(t.c.cursor).where(t.c.workspace_id == self.workspace_id, t.c.key == key)).scalar()
        return {"published": False, "reason": "older_than_stored_summary", "candidate_cursor": list(candidate),
                "stored_cursor": list(_cursor(stored))}

    def invalidate(self, project_id: str, problem_id: Optional[str] = None) -> int:
        """Delete the derived summaries of a project (and problem): the next read rebuilds them from the bases."""
        import sqlalchemy as sa

        from .models import memory_summary as t

        if not project_id:
            return 0
        query = sa.delete(t).where(t.c.workspace_id == self.workspace_id, t.c.project_id == str(project_id))
        if problem_id:
            query = query.where(t.c.problem_id == str(problem_id))
        with self.engine.begin() as c:
            return int(c.execute(query).rowcount or 0)

    def import_summary(self, key: str, doc: Document) -> bool:
        with self.engine.begin() as c:
            return c.execute(self._upsert(key, doc, guarded=False)).first() is not None

    def summaries(self) -> list[tuple[str, Document]]:
        import sqlalchemy as sa

        from .models import memory_summary as t

        with self.engine.connect() as c:
            rows = c.execute(sa.select(t.c.key, t.c.doc).where(t.c.workspace_id == self.workspace_id).order_by(t.c.key)).all()
        return [(r.key, dict(r.doc)) for r in rows]

    def counts(self) -> dict[str, int]:
        import sqlalchemy as sa

        from .models import memory_base, memory_summary

        with self.engine.connect() as c:
            return {name: int(c.execute(sa.select(sa.func.count()).select_from(t).where(t.c.workspace_id == self.workspace_id)).scalar())
                    for name, t in (("bases", memory_base), ("summaries", memory_summary))}


def import_id(workspace_id: str, relative: str) -> str:
    """A base row's id when it comes from a file workspace: the same file into the same workspace is the same row."""
    return "f" + hashlib.sha256(f"{workspace_id}:{relative}".encode("utf-8")).hexdigest()[:40]


def open_memory_bases(home: Path, url: str | None = None) -> "FileMemoryBases | PgMemoryBases":
    """The bases and summaries of the workspace in ``home``: PostgreSQL when DCLAB_DATABASE_URL is set, as the other
    stores choose (``storage.open_stores``), else the folder's files."""
    from . import db

    if url or db.database_url(required=False):
        return PgMemoryBases(db.workspace_for(Path(home), url), url)
    return FileMemoryBases(Path(home))


__all__ = ["BASES_DIR", "PURGED_DIR", "SUMMARY_DIR", "BaseChanged", "FileMemoryBases", "PgMemoryBases", "import_id",
           "open_memory_bases", "summary_key"]
