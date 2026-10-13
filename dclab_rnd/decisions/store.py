"""Where the decision samples and their labels live: two append-only files in the workspace folder, or the
``decision_sample`` and ``decision_label`` tables when ``DCLAB_DATABASE_URL`` is set (the same choice every other log makes).

A write that fails never stops the work that produced the decision: callers go through ``record`` and ``label``,
which report a failure on stderr and return. A project's rows are deleted with the project (``delete``); the export
(``dclab_rnd.decisions.export``) is how rows outlive it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Protocol

from .schema import LABEL_FIELDS, SAMPLE_FIELDS

SAMPLES_FILE = "decision_samples.jsonl"
LABELS_FILE = "decision_labels.jsonl"


class SampleStore(Protocol):
    def record(self, rows: Iterable[dict[str, Any]]) -> None: ...
    def label(self, rows: Iterable[dict[str, Any]]) -> None: ...
    def samples(self, project_id: str | None = None, point: str | None = None, draft_id: str | None = None) -> list[dict[str, Any]]: ...
    def labels(self, project_id: str | None = None) -> list[dict[str, Any]]: ...
    def latest(self, project_id: str | None, key: str) -> dict[str, Any] | None: ...
    def delete(self, project_id: str, draft_id: str | None = None) -> None: ...


_LOCKS: dict[Path, threading.Lock] = {}
_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.Lock())


class _Lines:
    """One append-only JSON-lines file, torn-line safe, replaced whole on a delete."""

    def __init__(self, path: Path, fields: tuple[str, ...]):
        self.path, self.fields = Path(path), fields
        self._lock = _lock_for(self.path)

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        lines = [json.dumps({k: r.get(k) for k in self.fields}, ensure_ascii=False, default=str) for r in rows]
        if not lines:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torn = self.path.is_file() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(("\n" if torn else "") + "\n".join(lines) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line
            if isinstance(row, dict):
                out.append(row)
        return out

    def keep(self, predicate) -> None:
        with self._lock:
            rows = [r for r in self.rows() if predicate(r)]
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
            os.replace(temporary, self.path)


class FileSamples:
    def __init__(self, home: Path):
        self.home = Path(home)
        self._samples = _Lines(self.home / SAMPLES_FILE, SAMPLE_FIELDS)
        self._labels = _Lines(self.home / LABELS_FILE, LABEL_FIELDS)

    def record(self, rows: Iterable[dict[str, Any]]) -> None:
        self._samples.append(rows)

    def label(self, rows: Iterable[dict[str, Any]]) -> None:
        self._labels.append(rows)

    def samples(self, project_id: str | None = None, point: str | None = None, draft_id: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self._samples.rows() if (project_id is None or r.get("project_id") == project_id)
                and (point is None or r.get("point") == point) and (draft_id is None or r.get("draft_id") == draft_id)]

    def labels(self, project_id: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self._labels.rows() if project_id is None or r.get("project_id") == project_id]

    def latest(self, project_id: str | None, key: str) -> dict[str, Any] | None:
        found = [r for r in self._samples.rows() if r.get("project_id") == project_id and r.get("key") == key]
        return found[-1] if found else None

    def delete(self, project_id: str, draft_id: str | None = None) -> None:
        gone = lambda r: r.get("project_id") == project_id or (draft_id is not None and r.get("draft_id") == draft_id)  # noqa: E731
        if self._samples.path.is_file():
            self._samples.keep(lambda r: not gone(r))
        if self._labels.path.is_file():
            self._labels.keep(lambda r: not gone(r))


class PgSamples:
    def __init__(self, workspace_id: str, url: str | None = None, engine: Any = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, engine if engine is not None else db.engine(url)

    def _insert(self, table, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
        import sqlalchemy as sa

        values = [{"workspace_id": self.workspace_id, **{k: r.get(k) for k in fields}} for r in rows]
        if not values:
            return
        with self.engine.begin() as c:
            c.execute(sa.insert(table), values)

    def record(self, rows: Iterable[dict[str, Any]]) -> None:
        from ..storage.models import decision_sample

        self._insert(decision_sample, SAMPLE_FIELDS, rows)

    def label(self, rows: Iterable[dict[str, Any]]) -> None:
        from ..storage.models import decision_label

        self._insert(decision_label, LABEL_FIELDS, rows)

    def _select(self, table, fields: tuple[str, ...], **where: Any) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        q = sa.select(*[table.c[k] for k in fields]).where(table.c.workspace_id == self.workspace_id)
        for column, value in where.items():
            if value is not None:
                q = q.where(table.c[column] == value)
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(q.order_by(table.c.id))]

    def samples(self, project_id: str | None = None, point: str | None = None, draft_id: str | None = None) -> list[dict[str, Any]]:
        from ..storage.models import decision_sample

        return self._select(decision_sample, SAMPLE_FIELDS, project_id=project_id, point=point, draft_id=draft_id)

    def labels(self, project_id: str | None = None) -> list[dict[str, Any]]:
        from ..storage.models import decision_label

        return self._select(decision_label, LABEL_FIELDS, project_id=project_id)

    def latest(self, project_id: str | None, key: str) -> dict[str, Any] | None:
        import sqlalchemy as sa

        from ..storage.models import decision_sample as t

        q = sa.select(*[t.c[k] for k in SAMPLE_FIELDS]).where(t.c.workspace_id == self.workspace_id, t.c.key == key)
        q = q.where(t.c.project_id == project_id) if project_id is not None else q.where(t.c.project_id.is_(None))
        with self.engine.connect() as c:
            row = c.execute(q.order_by(t.c.id.desc()).limit(1)).first()
        return dict(row._mapping) if row else None

    def delete(self, project_id: str, draft_id: str | None = None) -> None:
        import sqlalchemy as sa

        from ..storage.models import decision_label, decision_sample

        with self.engine.begin() as c:
            c.execute(sa.delete(decision_label).where(decision_label.c.workspace_id == self.workspace_id, decision_label.c.project_id == project_id))
            where = decision_sample.c.project_id == project_id
            if draft_id is not None:
                where = sa.or_(where, decision_sample.c.draft_id == draft_id)
            c.execute(sa.delete(decision_sample).where(decision_sample.c.workspace_id == self.workspace_id, where))


def open_samples(home: Path) -> SampleStore:
    from ..storage import db

    if db.database_url(required=False):
        return PgSamples(db.workspace_for(Path(home)))
    return FileSamples(Path(home))


def for_store(store: Any) -> SampleStore | None:
    """The sample store of the workspace a project or draft store belongs to (the same rule as the audit: a PostgreSQL
    store by its workspace id, a file store by the workspace folder its audit lives in; a bare store made for a test or
    a script writes nothing unless it is given a ``samples_home``)."""
    if getattr(store, "workspace_id", None):
        return PgSamples(store.workspace_id, engine=getattr(store, "engine", None))
    home = getattr(store, "samples_home", None) or getattr(store, "audit_path", None)
    if not home:
        return None
    home = Path(home)
    return FileSamples(home if home.is_dir() else home.parent)


def record(target: SampleStore | None, rows: list[dict[str, Any]]) -> int:
    """Append sample rows; a store that cannot be written is reported, never a reason to fail the decision."""
    if target is None or not rows:
        return 0
    try:
        target.record(rows)
        return len(rows)
    except Exception as exc:  # noqa: BLE001
        print(f"decisions: could not record {len(rows)} samples: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 0


def label(target: SampleStore | None, rows: list[dict[str, Any]]) -> int:
    if target is None or not rows:
        return 0
    try:
        target.label(rows)
        return len(rows)
    except Exception as exc:  # noqa: BLE001
        print(f"decisions: could not record {len(rows)} labels: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 0
