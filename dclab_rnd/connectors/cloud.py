"""One object from cloud storage: ``s3://bucket/key`` (boto3) or ``gs://bucket/key`` (google-cloud-storage or gcsfs).

Credentials are the server's: boto3's default chain (environment, ``AWS_PROFILE``, instance role…) and
Google's application default credentials. When S3 finds no credentials at all the object is tried once
anonymously, which is how public buckets are read. The object's size is checked before the download
and counted again while it streams; the file lands under a temporary name and is moved into place.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from .. import connectors as base
from . import BadInput, ConnectorError, copy_capped, is_table, sha256_file, target_path, temp_path, too_big

SUFFIXES = (".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx", ".log", ".txt")
URI = re.compile(r"^(s3|gs)://([^/]+)/(.+)$")
BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")


def parse(uri: str) -> tuple[str, str, str]:
    """(scheme, bucket, key) of a valid URI, else BadInput."""
    uri = str(uri or "").strip()
    match = URI.match(uri)
    if not match:
        raise BadInput("Write the object as s3://bucket/path/file.csv or gs://bucket/path/file.parquet.")
    scheme, bucket, key = match.groups()
    if not BUCKET.match(bucket) or ".." in bucket:
        raise BadInput(f"{bucket[:80]} is not a valid bucket name (lowercase letters, digits, dots and hyphens).")
    parts = key.split("/")
    if (len(key) > 1024 or ".." in parts or "." in parts or key.endswith("/") or "\\" in key
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in key)):
        raise BadInput("The object key must be a plain path to one file (no '..', no trailing '/').")
    if not is_table(key, SUFFIXES):
        raise BadInput(f"DCLab reads {', '.join(s.lstrip('.') for s in SUFFIXES)} files; "
                       f"{PurePosixPath(key).name[:80]} is not one of them.")
    return scheme, bucket, key


def fetch(uri: str, directory: Path) -> dict[str, Any]:
    """Download the object into ``directory``; returns {path, filename, source}."""
    scheme, bucket, key = parse(uri)
    uri = f"{scheme}://{bucket}/{key}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = target_path(directory, f"{bucket}_{PurePosixPath(key).name}")
    scratch = temp_path(directory)
    try:
        if scheme == "s3":
            meta = _s3(bucket, key, uri, scratch)
        else:
            meta = _gcs(bucket, key, uri, scratch)
        scratch.replace(final)
    finally:
        scratch.unlink(missing_ok=True)
    source = {"kind": "cloud", "uri": uri, "etag": meta.get("etag"), "size_bytes": final.stat().st_size,
              "sha256": sha256_file(final)}
    if meta.get("anonymous"):
        source["anonymous"] = True
    return {"path": str(final), "filename": final.name, "source": source}


# ---------------------------------------------------------------------------- S3
def _s3(bucket: str, key: str, uri: str, scratch: Path) -> dict[str, Any]:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError
    except ImportError:
        raise ConnectorError("Reading s3:// needs `boto3` installed in the server environment.") from None

    def attempt(client) -> dict[str, Any]:
        head = client.head_object(Bucket=bucket, Key=key)
        size = int(head.get("ContentLength") or 0)
        if size > base.MAX_BYTES:
            raise ConnectorError(too_big(uri, size))
        etag = str(head.get("ETag") or "").strip('"') or None
        body = client.get_object(Bucket=bucket, Key=key, **({"IfMatch": head["ETag"]} if head.get("ETag") else {}))["Body"]
        try:
            copy_capped(body, scratch, what=uri)
        finally:
            close = getattr(body, "close", None)
            if close:
                close()
        return {"etag": etag}

    try:
        try:
            return attempt(boto3.client("s3"))
        except NoCredentialsError:
            from botocore import UNSIGNED
            from botocore.config import Config

            try:  # no credentials on the server: a public bucket still answers anonymous reads
                return {**attempt(boto3.client("s3", config=Config(signature_version=UNSIGNED))), "anonymous": True}
            except ClientError as exc:
                if _status(exc) in (401, 403):
                    raise ConnectorError("The server has no AWS credentials and the object is not public. Configure the "
                                         "server's default AWS credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                                         "AWS_PROFILE or an instance role).") from None
                raise
    except ClientError as exc:
        status, code = _status(exc), str((getattr(exc, "response", {}) or {}).get("Error", {}).get("Code", ""))
        if status in (401, 403) or code in ("AccessDenied", "Forbidden", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
            raise ConnectorError(f"Access denied to {uri}: the server's AWS identity may not read it.") from None
        if status == 404 or code in ("NoSuchKey", "NoSuchBucket", "NotFound", "404"):
            raise ConnectorError(f"{uri} was not found (check the bucket and the key).") from None
        if status == 412 or code == "PreconditionFailed":
            raise ConnectorError(f"{uri} changed during the download. Try again.") from None
        raise ConnectorError(f"S3 refused the request for {uri} ({code or status or 'error'}).") from None
    except (NoCredentialsError, PartialCredentialsError):
        raise ConnectorError("The server's AWS credentials are missing or incomplete.") from None
    except BotoCoreError as exc:
        raise ConnectorError(f"Could not reach S3 ({type(exc).__name__}). Check the server's network and AWS region.") from None


def _status(exc) -> int | None:
    meta = (getattr(exc, "response", {}) or {}).get("ResponseMetadata", {})
    try:
        return int(meta.get("HTTPStatusCode"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------- Google Cloud Storage
def _gcs(bucket: str, key: str, uri: str, scratch: Path) -> dict[str, Any]:
    try:
        from google.cloud import storage  # type: ignore[import-not-found]
    except ImportError:
        storage = None
    if storage is not None:
        return _gcs_client(storage, bucket, key, uri, scratch)
    try:
        import gcsfs  # type: ignore[import-not-found]
    except ImportError:
        raise ConnectorError("Reading gs:// needs `google-cloud-storage` (or `gcsfs`) installed in the server "
                             "environment.") from None
    return _gcs_fs(gcsfs, bucket, key, uri, scratch)


def _gcs_client(storage, bucket: str, key: str, uri: str, scratch: Path) -> dict[str, Any]:
    from google.api_core import exceptions as gexc  # type: ignore[import-not-found]
    from google.auth.exceptions import DefaultCredentialsError  # type: ignore[import-not-found]

    anonymous = False
    try:
        client = storage.Client()
    except (DefaultCredentialsError, OSError):
        client, anonymous = storage.Client.create_anonymous_client(), True
    try:
        blob = client.bucket(bucket).blob(key)
        blob.reload()
        if blob.size is not None and int(blob.size) > base.MAX_BYTES:
            raise ConnectorError(too_big(uri, int(blob.size)))
        with blob.open("rb") as reader:
            copy_capped(reader, scratch, what=uri)
        return {"etag": blob.etag, "anonymous": anonymous}
    except gexc.NotFound:
        raise ConnectorError(f"{uri} was not found (check the bucket and the object name).") from None
    except (gexc.Forbidden, gexc.Unauthorized):
        raise ConnectorError(f"Access denied to {uri}: the server's Google credentials may not read it.") from None
    except gexc.GoogleAPIError as exc:
        raise ConnectorError(f"Google Cloud Storage refused the request for {uri} ({type(exc).__name__}).") from None


def _gcs_fs(gcsfs, bucket: str, key: str, uri: str, scratch: Path) -> dict[str, Any]:
    fs = gcsfs.GCSFileSystem()
    path = f"{bucket}/{key}"
    try:
        info = fs.info(path)
        size = int(info.get("size") or 0)
        if size > base.MAX_BYTES:
            raise ConnectorError(too_big(uri, size))
        with fs.open(path, "rb") as reader:
            copy_capped(reader, scratch, what=uri)
        return {"etag": info.get("etag") or info.get("md5Hash")}
    except FileNotFoundError:
        raise ConnectorError(f"{uri} was not found (check the bucket and the object name).") from None
    except PermissionError:
        raise ConnectorError(f"Access denied to {uri}: the server's Google credentials may not read it.") from None
    except OSError as exc:
        raise ConnectorError(f"Could not read {uri} ({type(exc).__name__}).") from None
