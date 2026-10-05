"""Data connectors: bring a table into a draft from Kaggle, Hugging Face, a database or cloud storage.

Every connector ends the same way: one file in a folder the caller chose (a draft's ``data/``), plus a
``source`` record saying where it came from (reference, revision, licence, sha256 of the bytes). The
draft routes register that file as an asset and run the same background pipeline as an upload.

    kaggle.search(query) / kaggle.files(ref) / kaggle.download(ref, directory, file=None)
    hf.fetch(dataset, directory, revision=None, split="train", config=None)
    db.connections() / db.fetch(connection, directory, table=None, query=None, limit=200_000)
    cloud.fetch("s3://bucket/key" | "gs://bucket/key", directory)

Secrets stay on the server: credentials come from environment variables or the provider's usual config
file, and never appear in a ``source`` record, in ``status()`` or in an error message. Optional client
libraries are imported lazily; when one is missing the error names the package to install.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import secrets
from pathlib import Path
from typing import Any

MAX_BYTES = 200 * 1024 * 1024
CHUNK = 1024 * 1024
TABLE_SUFFIXES = (".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx")
RESERVED = {"clean.parquet"}  # the pipeline writes the cleaned table under this name


class ConnectorError(ValueError):
    """A problem the user can act on; ``str(error)`` is a plain-English message without secrets."""


class BadInput(ConnectorError):
    """The request itself is malformed (a bad reference, URI or query): the routes answer 422."""


def status() -> dict[str, Any]:
    """What each connector can do on this server. Names and booleans only, never a secret or a URL."""
    from . import db, kaggle

    kaggle_from = kaggle.credentials_source()
    hf_token = bool(os.environ.get("HF_TOKEN"))
    names = db.connections()
    s3, gcs = _has("boto3"), _has("google.cloud.storage") or _has("gcsfs")
    return {
        "kaggle": {"configured": kaggle_from is not None,
                   "note": f"Credentials found ({kaggle_from})." if kaggle_from else
                   "Set KAGGLE_USERNAME and KAGGLE_KEY on the server (or put kaggle.json in ~/.kaggle) to search and import Kaggle datasets."},
        "hf": {"configured": True, "token": hf_token,
               "note": "HF_TOKEN is set: private and gated datasets you accepted can be read." if hf_token else
               "Public datasets work without a token; set HF_TOKEN on the server for private or gated ones."},
        "database": {"configured": bool(names), "connections": names,
                     "note": "Read-only queries through the connections the server defines." if names else
                     "No connection is configured: the server defines them as DCLAB_DB_<NAME>=<SQLAlchemy URL>."},
        "cloud": {"s3": s3, "gcs": gcs,
                  "note": "S3 uses the server's default AWS credentials (public buckets work without them); "
                          + ("gs:// uses the server's Google credentials." if gcs else "gs:// needs `google-cloud-storage` installed on the server.")},
        "max_bytes": MAX_BYTES,
    }


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a missing parent package raises instead of returning None
        return False


# ---------------------------------------------------------------------------- shared helpers
def safe_filename(name: str) -> str:
    """The last path part, safe characters only, no leading dots (same rule as uploads)."""
    base = re.split(r"[\\/]", str(name or ""))[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip("._")[:120]


def target_path(directory: Path, name: str) -> Path:
    """A free path for ``name`` in ``directory``: never overwrite an earlier file or the cleaned table."""
    directory = Path(directory)
    name = safe_filename(name) or "data"
    stem, suffix = (name.rsplit(".", 1)[0], "." + name.rsplit(".", 1)[1]) if "." in name else (name, "")
    path, n = directory / name, 1
    while path.exists() or path.name in RESERVED:
        path = directory / f"{stem}-{n}{suffix}"
        n += 1
    return path


def temp_path(directory: Path) -> Path:
    """A hidden scratch file in ``directory`` (same file system, so the final move is atomic)."""
    return Path(directory) / f".part-{secrets.token_hex(6)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_capped(reader, path: Path, limit: int | None = None, what: str = "The file") -> int:
    """Stream ``reader.read`` into ``path`` with a running byte count; refuse past ``limit`` (default MAX_BYTES)."""
    limit = MAX_BYTES if limit is None else limit
    total = 0
    try:
        with Path(path).open("wb") as out:
            while True:
                block = reader.read(CHUNK)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise ConnectorError(too_big(what, None, limit))
                out.write(block)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return total


def too_big(what: str, size: int | None, limit: int | None = None) -> str:
    limit = MAX_BYTES if limit is None else limit
    shown = f" is {size / 1e6:,.0f} MB and" if size else " is larger than the limit:"
    return (f"{what}{shown} the limit is {limit / 1e6:,.0f} MB. Import a smaller file or a sample "
            "(for a database, a query with fewer rows).")


def is_table(name: str, suffixes=TABLE_SUFFIXES) -> bool:
    return str(name).lower().endswith(tuple(suffixes))
