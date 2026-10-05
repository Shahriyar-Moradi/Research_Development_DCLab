"""Kaggle datasets through Kaggle's public REST API (``/api/v1``), with HTTP basic auth and ``urllib`` only.

Credentials: ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` in the environment, else ``kaggle.json`` in
``$KAGGLE_CONFIG_DIR`` or ``~/.kaggle``. The key is sent as an unredirected header (so it never follows
Kaggle's redirect to the storage host) and never appears in a message.

The response shapes are the ones the official ``kaggle`` client reads (camelCase keys such as
``totalBytes``, ``licenseName``, ``lastUpdated``, ``datasetFiles``); lookups ignore case and
underscores so snake_case answers work too.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .. import connectors as base
from . import (TABLE_SUFFIXES, BadInput, ConnectorError, copy_capped, is_table, safe_filename, sha256_file,
               target_path, temp_path, too_big)

API = "https://www.kaggle.com/api/v1"
SITE = "https://www.kaggle.com/datasets/"
REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TIMEOUT = 60


# ---------------------------------------------------------------------------- credentials
def _config_file() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle") / "kaggle.json"


def credentials_source() -> str | None:
    """Where the credentials come from ("environment" or "kaggle.json"), or None. Never the values."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "environment"
    return "kaggle.json" if _from_file() else None


def _from_file() -> tuple[str, str] | None:
    try:
        data = json.loads(_config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and data.get("username") and data.get("key"):
        return str(data["username"]), str(data["key"])
    return None


def _credentials() -> tuple[str, str]:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    found = _from_file()
    if found:
        return found
    raise ConnectorError("Kaggle is not configured on this server. Set KAGGLE_USERNAME and KAGGLE_KEY "
                         "(or put kaggle.json in ~/.kaggle) where the DCLab server runs, then try again.")


# ---------------------------------------------------------------------------- HTTP
def _open(path: str, what: str):
    """GET ``API + path`` with basic auth; returns the open response. Maps HTTP errors to plain messages."""
    user, key = _credentials()
    request = urllib.request.Request(API + path, headers={"User-Agent": "dclab-rnd", "Accept": "*/*"})
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    request.add_unredirected_header("Authorization", f"Basic {token}")
    try:
        return urllib.request.urlopen(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code in (401, 403):
            raise ConnectorError(f"Kaggle refused the credentials (HTTP {code}). Check KAGGLE_USERNAME and KAGGLE_KEY "
                                 "on the server; for some datasets you must first accept the terms on kaggle.com.") from None
        if code == 404:
            raise ConnectorError(f"Kaggle has no {what} (or it is private to another account).") from None
        if code == 429:
            raise ConnectorError("Kaggle is rate-limiting this server (HTTP 429). Wait a minute and try again.") from None
        raise ConnectorError(f"Kaggle answered HTTP {code} for {what}. Try again later.") from None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        raise ConnectorError(f"Could not reach Kaggle ({type(reason).__name__ if reason else 'network error'}). "
                             "Check the server's internet connection.") from None
    except (TimeoutError, OSError) as exc:
        raise ConnectorError(f"Could not reach Kaggle ({type(exc).__name__}). Check the server's internet connection.") from None


def _json(path: str, what: str) -> Any:
    with _open(path, what) as response:
        raw = response.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ConnectorError("Kaggle sent an answer that is not JSON. Try again later.") from None


def _pick(item: dict[str, Any], *names: str, default: Any = None) -> Any:
    """``item[name]`` ignoring case and underscores (``totalBytes`` == ``total_bytes``)."""
    folded = {str(k).replace("_", "").lower(): v for k, v in item.items()}
    for name in names:
        value = folded.get(name.replace("_", "").lower())
        if value not in (None, ""):
            return value
    return default


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def valid_ref(ref: str) -> bool:
    """``owner/slug`` with safe characters, and neither part made only of dots."""
    return isinstance(ref, str) and bool(REF.match(ref)) and ".." not in ref and not any(p.startswith(".") for p in ref.split("/"))


def check_ref(ref: str) -> tuple[str, str]:
    ref = str(ref or "").strip()
    if not valid_ref(ref):
        raise BadInput("A Kaggle dataset is written owner/dataset-name, for example blastchar/telco-customer-churn.")
    owner, slug = ref.split("/")
    return owner, slug


def _url(ref: str, url: Any = None) -> str:
    if isinstance(url, str) and url.startswith("https://"):
        return url
    if isinstance(url, str) and url.startswith("/"):
        return "https://www.kaggle.com" + url
    return SITE + ref


# ---------------------------------------------------------------------------- public API
def search(query: str, page: int = 1) -> list[dict[str, Any]]:
    """Datasets matching ``query``: [{ref, title, size_bytes, license, last_updated, url}]."""
    query = str(query or "").strip()
    if not query:
        raise BadInput("Type a few words to search Kaggle.")
    page = max(1, min(int(page or 1), 100))
    data = _json("/datasets/list?" + urllib.parse.urlencode({"search": query[:200], "page": page}), "search result")
    if isinstance(data, dict):  # tolerate a wrapped answer
        data = _pick(data, "datasets", default=[])
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        ref = _pick(item, "ref")
        if not ref and _pick(item, "ownerRef", "ownerSlug") and _pick(item, "datasetSlug", "slug"):
            ref = f"{_pick(item, 'ownerRef', 'ownerSlug')}/{_pick(item, 'datasetSlug', 'slug')}"
        if not valid_ref(ref):
            continue
        out.append({"ref": ref, "title": _pick(item, "title", default=ref), "size_bytes": _int(_pick(item, "totalBytes", "size")),
                    "license": _pick(item, "licenseName", "license"), "last_updated": _pick(item, "lastUpdated"),
                    "url": _url(ref, _pick(item, "url"))})
    return out


def files(ref: str) -> list[dict[str, Any]]:
    """The files of one dataset: [{name, size_bytes}]."""
    owner, slug = check_ref(ref)
    data = _json(f"/datasets/list/{owner}/{slug}", f"dataset {owner}/{slug}")
    if isinstance(data, dict):
        message = _pick(data, "errorMessage")
        entries = _pick(data, "datasetFiles", "files", default=[])
        if message and not entries:
            raise ConnectorError(f"Kaggle could not list the files of {owner}/{slug}: {str(message)[:200]}")
    else:
        entries = data
    out = []
    for item in entries if isinstance(entries, list) else []:
        if isinstance(item, dict):
            name = _pick(item, "name", "nameNullable", "ref")
            if isinstance(name, str) and name:
                out.append({"name": name, "size_bytes": _int(_pick(item, "totalBytes", "totalBytesNullable", "size"))})
    return out


def metadata(ref: str) -> dict[str, Any]:
    """Best effort: licence, last update and version number from ``/datasets/view``; {} when unavailable."""
    owner, slug = check_ref(ref)
    try:
        data = _json(f"/datasets/view/{owner}/{slug}", f"dataset {owner}/{slug}")
    except ConnectorError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {"license": _pick(data, "licenseName", "license"), "last_updated": _pick(data, "lastUpdated"),
            "version": _int(_pick(data, "currentVersionNumber", "versionNumber")), "url": _url(ref, _pick(data, "url"))}


def download(ref: str, directory: Path, file: str | None = None) -> dict[str, Any]:
    """Download the dataset (or one ``file`` of it) and keep one table in ``directory``.

    Kaggle answers with a zip (always for a whole dataset, often for one file). Only plain members
    are considered (no directories, no absolute or ``..`` paths); the largest table is kept unless
    ``file`` names one. Downloads and extracted members are both capped at ``MAX_BYTES``.
    """
    owner, slug = check_ref(ref)
    ref = f"{owner}/{slug}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wanted = None
    if file is not None and str(file).strip():
        wanted = str(file).strip()
        if len(wanted) > 255 or "\x00" in wanted or ".." in PurePosixPath(wanted).parts:
            raise BadInput("Give the file name as Kaggle lists it, for example WA_Fn-UseC_-Telco-Customer-Churn.csv.")
    info = metadata(ref)
    path = f"/datasets/download/{owner}/{slug}" + (f"/{urllib.parse.quote(wanted, safe='')}" if wanted else "")
    scratch = temp_path(directory)
    try:
        with _open(path, f"file {safe_filename(wanted)} in {ref}" if wanted else f"dataset {ref}") as response:
            length = _int(response.headers.get("Content-Length")) if getattr(response, "headers", None) else None
            if length and length > base.MAX_BYTES:
                raise ConnectorError(too_big(f"The Kaggle download {ref}", length))
            copy_capped(response, scratch, what=f"The Kaggle download {ref}")
            disposition = response.headers.get("Content-Disposition", "") if getattr(response, "headers", None) else ""
        if zipfile.is_zipfile(scratch):
            final, chosen = _extract(scratch, directory, wanted, ref)
        else:
            chosen = wanted or _disposition_name(disposition) or f"{slug}.csv"
            if not is_table(chosen):
                raise ConnectorError(f"{safe_filename(chosen) or 'The file'} is not a table DCLab reads "
                                     f"({', '.join(s.lstrip('.') for s in TABLE_SUFFIXES)}).")
            final = target_path(directory, f"{slug}_{PurePosixPath(chosen).name}")
            scratch.replace(final)
    finally:
        scratch.unlink(missing_ok=True)
    source = {"kind": "kaggle", "ref": ref, "file": chosen, "license": info.get("license"),
              "sha256": sha256_file(final), "url": info.get("url") or SITE + ref}
    if info.get("version") is not None:
        source["version"] = info["version"]
    if info.get("last_updated"):
        source["last_updated"] = info["last_updated"]
    return {"path": str(final), "filename": final.name, "source": source}


def _disposition_name(header: str) -> str | None:
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", header or "")
    return urllib.parse.unquote(match.group(1)) if match else None


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        parts = PurePosixPath(name).parts
        if member.is_dir() or not parts or name.startswith("/") or ".." in parts or ":" in parts[0]:
            continue  # directories, absolute paths, traversal and drive letters are never extracted
        if parts[0] == "__MACOSX" or parts[-1].startswith("."):
            continue
        members.append(member)
    return members


def _extract(archive_path: Path, directory: Path, wanted: str | None, ref: str) -> tuple[Path, str]:
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive)
        if wanted:
            basename = PurePosixPath(wanted).name
            matches = [m for m in members if m.filename.replace("\\", "/") in (wanted, wanted.lstrip("/"))] or \
                      [m for m in members if PurePosixPath(m.filename.replace("\\", "/")).name == basename]
            if not matches and len(members) == 1:
                matches = members
            if not matches:
                raise ConnectorError(f"The Kaggle download {ref} has no file named {safe_filename(basename)}.")
            member = matches[0]
            if not is_table(member.filename):
                raise ConnectorError(f"{safe_filename(PurePosixPath(member.filename).name)} is not a table DCLab reads "
                                     f"({', '.join(s.lstrip('.') for s in TABLE_SUFFIXES)}).")
        else:
            tables = [m for m in members if is_table(m.filename)]
            if not tables:
                raise ConnectorError(f"The Kaggle dataset {ref} has no table file DCLab reads "
                                     f"({', '.join(s.lstrip('.') for s in TABLE_SUFFIXES)}).")
            member = max(tables, key=lambda m: (m.file_size, m.filename))
        name = PurePosixPath(member.filename.replace("\\", "/")).name
        if member.file_size > base.MAX_BYTES:
            raise ConnectorError(too_big(name, member.file_size))
        final = target_path(directory, f"{ref.split('/')[1]}_{name}")
        with archive.open(member) as reader:
            copy_capped(reader, final, what=name)
    return final, member.filename.replace("\\", "/")
