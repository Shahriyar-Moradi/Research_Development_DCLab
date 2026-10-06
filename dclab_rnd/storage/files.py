"""Tables and artifacts behind one interface (package 9.3): put, open, path, exists and delete.

Files do not belong in the database. Each one lives under the workspace folder at a *key*, its path relative to that
folder (``projects/<id>/data/telco.csv``, ``drafts/<id>/data/clean.parquet``), so a workspace folder that is moved,
or copied to another machine, still finds its files. The database (or ``.dclab_files.json`` beside the files, without
one) records each file's key, size, SHA-256 and content type, and every read checks the hash: a table that changed
on disk since it was recorded is refused, not modelled.

Two implementations:

- ``LocalFiles``: the workspace folder itself.
- ``S3Files``: an S3-compatible bucket (``DCLAB_FILES_URL=s3://bucket/prefix``; ``DCLAB_FILES_ENDPOINT`` for MinIO and
  the like). The workspace folder is its local copy: a write goes to both, a read that finds no local copy, or one
  whose hash differs, fetches the object and checks it. boto3 is imported only here, and credentials come from its
  default chain (environment, profile, instance role): they are never passed in, logged or recorded.

A key is made of the same safe names as an upload (letters, digits, dot, dash, underscore; no ``..``), and a file is at
most ``MAX_BYTES``, as every upload route already enforces.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol

MAX_BYTES = 200 * 1024 * 1024  # the upload limit of every route (server, drafts, connectors)
PART = re.compile(r"[^/\\\x00]{1,255}")  # any one name of a path: the safety is that no part is "", "." or ".." and none holds a separator
RECORDS = ".dclab_files.json"
TYPES = {".csv": "text/csv", ".tsv": "text/tab-separated-values", ".tab": "text/tab-separated-values", ".json": "application/json",
         ".jsonl": "application/x-ndjson", ".parquet": "application/vnd.apache.parquet", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         ".xls": "application/vnd.ms-excel", ".ipynb": "application/x-ipynb+json", ".md": "text/markdown", ".txt": "text/plain", ".log": "text/plain"}


class FileError(ValueError):
    """A key that is not a safe name, a file over the limit, or a file whose bytes are not the ones recorded."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_key(key: str) -> str:
    parts = str(key).split("/")
    if not key or str(key).startswith("/") or any(p in ("", ".", "..") or not PART.fullmatch(p) for p in parts):
        raise FileError(f"not a safe file key: {key!r}")
    return key


def content_type(key: str) -> str:
    return TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


_HASHES: dict[tuple[Any, ...], str] = {}
_HASH_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    """The file's SHA-256, remembered while its size and modification time stay the same."""
    stat = path.stat()
    # the inode and the change time too: a rewrite of the same size whose modification time was restored still counts
    cache = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)
    with _HASH_LOCK:
        if cache in _HASHES:
            return _HASHES[cache]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _HASH_LOCK:
        if len(_HASHES) > 512:
            _HASHES.clear()
        _HASHES[cache] = value
    return value


# ---------------------------------------------------------------------- the records


class Records(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def put(self, record: dict[str, Any]) -> None: ...
    def delete(self, key: str) -> None: ...
    def forget(self, prefix: str) -> list[str]: ...  # delete every record under a prefix; returns their keys


class JsonRecords:
    """``.dclab_files.json`` in the workspace folder (a workspace without a database), replaced whole on each write."""

    _locks: dict[str, threading.Lock] = {}

    def __init__(self, root: Path):
        self.path = Path(root) / RECORDS
        self._lock = JsonRecords._locks.setdefault(str(self.path.resolve()), threading.Lock())

    def _all(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, rows: dict[str, dict[str, Any]]) -> None:
        import tempfile

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".dclab_files.", suffix=".tmp", dir=self.path.parent)  # one per writer
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(rows, indent=1, sort_keys=True))
        os.replace(temporary, self.path)

    def _locked(self):
        """This process's lock and, where the system has one, a lock on a file beside the records: the server and the MCP
        server can write the same workspace's records at once."""
        import contextlib

        @contextlib.contextmanager
        def held():
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path.with_name(self.path.name + ".lock"), "a+") as lock:
                    try:
                        import fcntl

                        fcntl.flock(lock, fcntl.LOCK_EX)
                    except ImportError:  # Windows: the process lock only
                        pass
                    yield
        return held()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._all().get(key)  # a reader sees the whole file before or after a write (os.replace)

    def put(self, record: dict[str, Any]) -> None:
        with self._locked():
            rows = self._all()
            rows[record["key"]] = record
            self._write(rows)

    def delete(self, key: str) -> None:
        with self._locked():
            rows = self._all()
            if rows.pop(key, None) is not None:
                self._write(rows)

    def forget(self, prefix: str) -> list[str]:
        with self._locked():
            rows = self._all()
            gone = [k for k in rows if k.startswith(prefix)]
            for k in gone:
                rows.pop(k)
            if gone:
                self._write(rows)
        return gone


class PgRecords:
    def __init__(self, workspace_id: str, url: str | None = None):
        from . import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def get(self, key: str) -> dict[str, Any] | None:
        import sqlalchemy as sa

        from .models import stored_file as t

        with self.engine.connect() as c:
            row = c.execute(sa.select(t.c.key, t.c.size, t.c.sha256, t.c.content_type, t.c.backend, t.c.recorded)
                            .where(t.c.workspace_id == self.workspace_id, t.c.key == key)).first()
        return dict(row._mapping) if row else None

    def put(self, record: dict[str, Any]) -> None:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from .models import stored_file as t

        values = {k: record[k] for k in ("key", "size", "sha256", "content_type", "backend", "recorded")}
        with self.engine.begin() as c:
            c.execute(insert(t).values(workspace_id=self.workspace_id, **values).on_conflict_do_update(
                index_elements=[t.c.workspace_id, t.c.key], set_={k: values[k] for k in ("size", "sha256", "content_type", "backend", "recorded")}))

    def delete(self, key: str) -> None:
        import sqlalchemy as sa

        from .models import stored_file as t

        with self.engine.begin() as c:
            c.execute(sa.delete(t).where(t.c.workspace_id == self.workspace_id, t.c.key == key))

    def forget(self, prefix: str) -> list[str]:
        import sqlalchemy as sa

        from .models import stored_file as t

        like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self.engine.begin() as c:
            gone = [r[0] for r in c.execute(sa.delete(t).where(t.c.workspace_id == self.workspace_id, t.c.key.like(like, escape="\\")).returning(t.c.key))]
        return gone


def open_records(root: Path) -> Records:
    from . import db

    if db.database_url(required=False):
        return PgRecords(db.workspace_for(Path(root)))
    return JsonRecords(Path(root))


# ---------------------------------------------------------------------- the files


class LocalFiles:
    backend = "local"

    def __init__(self, root: Path, records: Records | None = None):
        self.root = Path(root)
        self.records = records if records is not None else open_records(self.root)

    def local_path(self, key: str) -> Path:
        return self.root / check_key(key)

    def key_of(self, path: Path) -> str:
        """The key of a file under the workspace folder. Paths are made absolute, not resolved: a folder inside the
        workspace that is a link to another disk keeps its key."""
        try:
            return check_key(Path(os.path.abspath(path)).relative_to(os.path.abspath(self.root)).as_posix())
        except ValueError as exc:
            if isinstance(exc, FileError):
                raise
            raise FileError(f"{Path(path).name} is not inside the workspace folder") from None

    def record(self, key: str) -> dict[str, Any] | None:
        return self.records.get(check_key(key))

    def put(self, key: str, source: Path | bytes, kind: str | None = None) -> dict[str, Any]:
        """Store bytes, or a file, at ``key`` and record it. A file already at its place is recorded as it is."""
        target = self.local_path(key)
        if isinstance(source, (bytes, bytearray)):
            if len(source) > MAX_BYTES:
                raise FileError(f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            temporary.write_bytes(bytes(source))
            os.replace(temporary, target)
        else:
            source = Path(source)
            if source.stat().st_size > MAX_BYTES:
                raise FileError(f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
            if source.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        record = {"key": key, "size": target.stat().st_size, "sha256": sha256(target), "content_type": kind or content_type(key),
                  "backend": self.backend, "recorded": now()}
        self._upload(key, target, record)
        self.records.put(record)
        return record

    def _upload(self, key: str, path: Path, record: dict[str, Any]) -> None:
        """A local workspace is its own storage."""

    def _fetch(self, key: str, path: Path) -> bool:
        return False

    def exists(self, key: str) -> bool:
        return self.local_path(key).is_file()

    def path(self, key: str, expected: str | None = None) -> Path:
        """A local file with the recorded bytes (or ``expected``, a SHA-256 the caller holds). A file that is missing
        or changed is fetched again where there is a copy elsewhere; otherwise this is an error, never other bytes."""
        local = self.local_path(key)
        record = self.records.get(key)
        want = expected or (record or {}).get("sha256")
        if local.is_file() and (want is None or sha256(local) == want):
            return local
        if self._fetch(key, local) and (want is None or sha256(local) == want):
            return local
        if not local.is_file():
            raise FileError(f"the file {key} is missing")
        raise FileError(f"the file {key} changed since it was recorded (its SHA-256 is not the recorded one); attach it again")

    def open(self, key: str, expected: str | None = None) -> BinaryIO:
        return self.path(key, expected).open("rb")

    def delete(self, key: str) -> None:
        self.local_path(key).unlink(missing_ok=True)
        self._remove(key)
        self.records.delete(check_key(key))

    def _remove(self, key: str) -> None:
        pass

    def forget(self, prefix: str) -> list[str]:
        """Forget every file under a prefix (a deleted project or draft; its folder is removed by its store)."""
        gone = self.records.forget(check_key(prefix.rstrip("/")) + "/")
        for key in gone:
            self._remove(key)
        return gone


class S3Files(LocalFiles):
    """An S3-compatible bucket, with the workspace folder as its local copy. boto3 is imported here only."""

    backend = "s3"

    def __init__(self, root: Path, url: str, records: Records | None = None, client: Any = None):
        super().__init__(root, records)
        match = re.fullmatch(r"s3://([^/]+)/?(.*)", url.strip())
        if not match:
            raise FileError("DCLAB_FILES_URL must look like s3://bucket/prefix")
        self.bucket, self.prefix = match.group(1), match.group(2).strip("/")
        self._client = client

    def _s3(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # noqa: F841
                raise FileError("S3 file storage needs boto3 installed (pip install boto3)") from None
            endpoint = os.environ.get("DCLAB_FILES_ENDPOINT") or None
            self._client = boto3.client("s3", endpoint_url=endpoint)  # credentials from boto3's default chain only
        return self._client

    def _object(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _upload(self, key: str, path: Path, record: dict[str, Any]) -> None:
        with path.open("rb") as body:
            self._s3().put_object(Bucket=self.bucket, Key=self._object(key), Body=body, ContentType=record["content_type"],
                                  Metadata={"sha256": record["sha256"]})

    def _fetch(self, key: str, path: Path) -> bool:
        try:
            got = self._s3().get_object(Bucket=self.bucket, Key=self._object(key))
        except Exception:  # noqa: BLE001 — no object, no access: the caller says the file is missing
            return False
        if int(got.get("ContentLength") or 0) > MAX_BYTES:
            return False  # refused before a byte is read
        stream = got["Body"]
        body = stream.read(MAX_BYTES + 1) if hasattr(stream, "read") else bytes(stream)[:MAX_BYTES + 1]
        if len(body) > MAX_BYTES:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(body)
        os.replace(temporary, path)
        return True

    def exists(self, key: str) -> bool:
        if super().exists(key):
            return True
        try:
            self._s3().head_object(Bucket=self.bucket, Key=self._object(key))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _remove(self, key: str) -> None:
        try:
            self._s3().delete_object(Bucket=self.bucket, Key=self._object(key))
        except Exception:  # noqa: BLE001 — a record without an object is harmless; an object without a record is never read
            pass


def open_files(root: Path) -> LocalFiles:
    """The file storage of the workspace in ``root``: S3 when DCLAB_FILES_URL names a bucket, the folder otherwise."""
    url = os.environ.get("DCLAB_FILES_URL", "").strip()
    if url.startswith("s3://"):
        return S3Files(Path(root), url)
    return LocalFiles(Path(root))


def files_for(store: Any) -> LocalFiles:
    """The file storage behind a project or draft store: the workspace folder it keeps its files in."""
    root = getattr(store, "files_home", None) or Path(store.home).parent
    return open_files(Path(root))


__all__ = ["FileError", "LocalFiles", "S3Files", "open_files", "files_for", "MAX_BYTES", "check_key", "sha256"]
