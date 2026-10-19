"""The append-only store (32.1): a source kept whole, compressed, with its hash; a changed source a new version beside the
old; nothing rewritten in place, nothing removed (the owner's ``store forget`` is the only removal).

``Store.keep(kind, workspace, source, data)`` copies the bytes into ``raw/<kind>/<workspace>/<yyyy>/<mm>/<dd>/`` and appends a
:class:`dclab_core.store.StoreEntry` to the kind's ``index.jsonl``; the same bytes kept again are not a new entry.
``Store.keep_log`` keeps an append-only log by its new tail (cut at its last line break, so a line half written is kept on
the next run). ``Store.read(entry)`` gives the bytes back and checks their hash; ``Store.whole`` rebuilds a log.

Every source is read under its kind's lock, so two runs at once (the worker's job and a removal's) never keep an older
reading after a newer one. Each kind's index is read once and then only its new lines (a cache keyed by the index's size);
a line that does not parse is set aside to ``index.torn.jsonl`` (a write cut short), never lost; a line written by a newer
version of the entry stays in the index for the code that reads it.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

from dclab_core.store import StoreEntry

from .layout import Layout, slug

Data = Union[bytes, Path, Callable[[], bytes]]
ENTRY_VERSION = "dclab_core.store_entry.v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bytes(data: Data) -> bytes:
    if callable(data):
        return data()
    return Path(data).read_bytes() if isinstance(data, Path) else bytes(data)


class _Index:
    """One kind's index as read so far: the latest entry per (workspace, source), how far the file was read, and whether a line
    did not parse."""

    def __init__(self) -> None:
        self.offset, self.latest, self.entries, self.torn = 0, {}, [], False


class Store:
    def __init__(self, layout: Optional[Layout] = None):
        self.layout = layout or Layout()
        self._indexes: dict[str, _Index] = {}

    @contextmanager
    def _locked(self, kind: str) -> Iterator[None]:
        lock = self.layout.lock(kind)
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # ------------------------------------------------------------------ the index
    @staticmethod
    def _parse(line: str) -> tuple[Optional[StoreEntry], bool]:
        """(the entry, whether the line is torn): a line of a newer entry version is neither (kept, not read here)."""
        try:
            doc = json.loads(line)
        except ValueError:
            return None, True
        if isinstance(doc, dict) and str(doc.get("schema_version", "")).startswith("dclab_core.store_entry.") and doc["schema_version"] != ENTRY_VERSION:
            return None, False
        try:
            return StoreEntry.read(doc), False
        except Exception:  # noqa: BLE001
            return None, True

    def _index(self, kind: str) -> _Index:
        """The kind's index, its new lines read (the whole file again only when it shrank: it was mended)."""
        path = self.layout.index(kind)
        idx = self._indexes.setdefault(kind, _Index())
        size = path.stat().st_size if path.exists() else 0
        if size < idx.offset:
            idx = self._indexes[kind] = _Index()
        if size > idx.offset:
            with open(path, "rb") as fh:
                fh.seek(idx.offset)
                chunk = fh.read()
            complete = chunk[: chunk.rfind(b"\n") + 1]  # a last line still being written is read next time
            for line in complete.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                entry, torn = self._parse(line)
                idx.torn = idx.torn or torn
                if entry is not None:
                    idx.entries.append(entry)
                    key = (entry.workspace, entry.source)
                    if key not in idx.latest or entry.version > idx.latest[key].version:
                        idx.latest[key] = entry
            idx.offset += len(complete)
            if len(complete) < len(chunk):
                idx.torn = idx.torn or not chunk.endswith(b"\n")
        return idx

    def entries(self, kind: str) -> list[StoreEntry]:
        """The kind's kept versions, in the order they were kept."""
        return list(self._index(kind).entries)

    def _mend(self, kind: str) -> None:
        """Under the kind's lock, only when a line did not parse or the file does not end a line: the torn lines set aside to
        ``index.torn.jsonl``, the index rewritten with the rest (newer versions' lines kept)."""
        path = self.layout.index(kind)
        if not path.exists():
            return
        idx = self._index(kind)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not idx.torn and (not text or text.endswith("\n")):
            return
        good, torn = [], []
        for line in text.splitlines():
            if line.strip():
                (torn if self._parse(line)[1] else good).append(line)
        if torn:
            with open(self.layout.torn(kind), "a", encoding="utf-8") as fh:
                fh.write("\n".join(torn) + "\n")
        tmp = path.with_suffix(".jsonl.part")
        tmp.write_text("".join(g + "\n" for g in good), encoding="utf-8")
        tmp.replace(path)
        self._indexes.pop(kind, None)

    def latest(self, kind: str, workspace: str, source: str) -> Optional[StoreEntry]:
        return self._index(kind).latest.get((self.layout.workspace(workspace), source))

    # ------------------------------------------------------------------ writing
    def _write(self, kind: str, entry: StoreEntry, target: Path, data: bytes) -> None:
        """The blob first, flushed to disk, then the index line, flushed: an index line never names a blob that is not whole."""
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        with open(tmp, "wb") as raw_fh:
            with gzip.GzipFile(fileobj=raw_fh, mode="wb") as fh:
                fh.write(data)
            raw_fh.flush()
            os.fsync(raw_fh.fileno())
        tmp.replace(target)
        with open(self.layout.index(kind), "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @staticmethod
    def _names(kind: str, workspace: str, source: str, version: int, digest: str) -> tuple[str, str]:
        """The entry's id and its file's name: the source's hash makes them unique (two sources never share one)."""
        key = hashlib.sha1(source.encode()).hexdigest()[:12]
        return f"{kind}:{workspace}:{key}:v{version}", f"{slug(source, 60)}-{key}.v{version}.{digest[:12]}.gz"

    def _entry(self, kind: str, workspace: str, source: str, version: int, digest: str, size: int, when: str, consent: str,
               run: Optional[str], note: Optional[str], offset: int, whole: str) -> tuple[StoreEntry, Path]:
        ident, name = self._names(kind, workspace, source, version, digest)
        target = self.layout.raw_file(kind, workspace, when[:10], name)
        entry = StoreEntry(id=ident, kind=kind, source=source, workspace=workspace, version=version, sha256=digest, bytes=size,
                           captured_at=when, consent=consent, run=run, path=self.layout.relative(target.parent) + "/" + name,
                           note=(note or None) and note[:600], offset=offset, whole_sha256=whole)  # validated before anything is written
        return entry, target

    def keep(self, kind: str, workspace: str, source: str, data: Data, *, consent: str = "operate", run: Optional[str] = None,
             note: Optional[str] = None, at: Optional[str] = None, transform: Optional[Callable[[bytes], bytes]] = None) -> Optional[StoreEntry]:
        """Keep ``data`` (bytes, a file read whole, or a function giving the bytes; read under the lock) as the newest version of
        ``source``; None when those bytes are already its newest version."""
        workspace = self.layout.workspace(workspace)
        with self._locked(kind):
            self._mend(kind)
            raw = _bytes(data)
            if transform is not None:
                raw = transform(raw)
            digest = hashlib.sha256(raw).hexdigest()
            last = self.latest(kind, workspace, source)
            if last is not None and last.whole_sha256 == digest:
                return None
            entry, target = self._entry(kind, workspace, source, (last.version + 1) if last else 1, digest, len(raw), at or now(), consent, run,
                                        note, 0, digest)
            self._write(kind, entry, target, raw)
        return entry

    def keep_log(self, kind: str, workspace: str, source: str, data: Data, *, consent: str = "operate", run: Optional[str] = None,
                 note: Optional[str] = None, at: Optional[str] = None, transform: Optional[Callable[[bytes], bytes]] = None) -> Optional[StoreEntry]:
        """Keep an append-only log: only the lines added since its last kept version (``offset``), when what was kept before is still
        the log's beginning (checked by ``whole_sha256``); the whole log again, as a new version, when it was rewritten. A last line
        not yet ended is left for the next run. None when nothing was added."""
        workspace = self.layout.workspace(workspace)
        with self._locked(kind):
            self._mend(kind)
            raw = _bytes(data)
            if transform is not None:
                raw = transform(raw)
            raw = raw[: raw.rfind(b"\n") + 1] if b"\n" in raw else b""  # whole lines only
            whole = hashlib.sha256(raw).hexdigest()
            last = self.latest(kind, workspace, source)
            if (last is not None and last.whole_sha256 == whole) or (last is None and not raw):
                return None
            end = (last.offset + last.bytes) if last else 0
            grows = last is not None and bool(last.whole_sha256) and len(raw) > end and hashlib.sha256(raw[:end]).hexdigest() == last.whole_sha256
            start = end if grows else 0
            part = raw[start:]
            entry, target = self._entry(kind, workspace, source, (last.version + 1) if last else 1, hashlib.sha256(part).hexdigest(), len(part),
                                        at or now(), consent, run,
                                        note or (None if grows or last is None else "rewritten: kept whole again"), start, whole)
            self._write(kind, entry, target, part)
        return entry

    # ------------------------------------------------------------------ reading
    def read(self, entry: StoreEntry) -> bytes:
        """The kept bytes, checked against the entry's hash (a mismatch is an error, never silently returned)."""
        with gzip.open(self.layout.entry(entry.path), "rb") as fh:
            raw = fh.read()
        if hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ValueError(f"{entry.id}: the kept bytes do not match their hash")
        return raw

    def whole(self, kind: str, workspace: str, source: str) -> bytes:
        """A source as it is now in the store: its last whole version and every tail after it, checked against the last hash."""
        workspace = self.layout.workspace(workspace)
        found = sorted((e for e in self.entries(kind) if e.workspace == workspace and e.source == source), key=lambda e: e.version)
        if not found:
            raise KeyError(f"{kind}/{workspace}/{source} was never kept")
        start = max(i for i, e in enumerate(found) if e.offset == 0)
        out = b"".join(self.read(e) for e in found[start:])
        last = found[-1]
        if last.whole_sha256 and hashlib.sha256(out).hexdigest() != last.whole_sha256:
            raise ValueError(f"{kind}/{workspace}/{source}: its kept parts do not make the source it was")
        return out

    def counts(self) -> dict[str, dict[str, Any]]:
        from dclab_core.store import KINDS

        out = {}
        for kind in KINDS:
            entries = self.entries(kind)
            out[kind] = {"entries": len(entries), "sources": len({(e.workspace, e.source) for e in entries}),
                         "bytes": sum(e.bytes for e in entries), "workspaces": sorted({e.workspace for e in entries}),
                         "last": max((e.captured_at for e in entries), default=None)}
        return out


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


__all__ = ["Store", "now", "write_json"]
