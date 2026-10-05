"""Hugging Face datasets through ``huggingface_hub`` (no ``datasets`` package needed).

The connector looks for one data file of the requested split in the dataset repository (parquet first,
then csv, jsonl, json). When the repository has none (a loading script, an archive, an unusual layout)
it falls back to the parquet conversion Hugging Face publishes on the ``refs/convert/parquet`` branch
(``<config>/<split>/0000.parquet``). Only one file is taken: a split stored in several shards is
imported from its first shard, and ``source.shards`` says so.

``HF_TOKEN`` (server environment) is used when set: private datasets and gated datasets whose terms
the token's account accepted.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .. import connectors as base
from . import BadInput, ConnectorError, safe_filename, sha256_file, target_path, too_big

DATASET = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$")
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
REVISION = re.compile(r"^[A-Za-z0-9_./-]{1,200}$")
PREFERENCE = (".parquet", ".csv", ".tsv", ".jsonl", ".json")
CONVERTED = "refs/convert/parquet"


def check(dataset: str, revision: str | None, split: str, config: str | None) -> str:
    dataset = str(dataset or "").strip()
    if not DATASET.match(dataset) or ".." in dataset:
        raise BadInput("A Hugging Face dataset is written owner/name, for example scikit-learn/churn-prediction.")
    if revision is not None and (not REVISION.match(revision) or ".." in revision):
        raise BadInput("The revision must be a branch, tag or commit id.")
    if not NAME.match(split or "") or ".." in split:
        raise BadInput("The split must be a name such as train, validation or test.")
    if config is not None and (not NAME.match(config) or ".." in config):
        raise BadInput("The configuration (subset) must be a name such as default.")
    return dataset


def _token(path: str, name: str) -> bool:
    """``name`` appears in ``path`` as a word (``train`` matches ``data/train-0000.parquet``, not ``constrained``)."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", path, re.IGNORECASE) is not None


def pick_file(files: list[str], split: str, config: str | None) -> tuple[str | None, int]:
    """The best data file of ``split`` (and ``config``) on a normal branch, and how many shards share its format."""
    data = [f for f in files if f.lower().endswith(PREFERENCE) and not PurePosixPath(f).name.startswith(".")]
    if config and config != "default":  # "default" is the name of the unnamed configuration, not a folder
        data = [f for f in data if _token(f, config)]
    matched = [f for f in data if _token(f, split)]
    if not matched and len(data) == 1 and split == "train":
        matched = data  # a single data file with no split in its name is the train split by convention
    for suffix in PREFERENCE:
        group = sorted(f for f in matched if f.lower().endswith(suffix))
        if group:
            return group[0], len(group)
    return None, 0


def pick_converted(files: list[str], split: str, config: str | None) -> tuple[str | None, int, str | None]:
    """The first parquet shard of ``split`` on the conversion branch, its shard count and the config used."""
    shards: dict[str, list[str]] = {}
    for f in files:
        parts = PurePosixPath(f).parts
        if len(parts) >= 3 and f.endswith(".parquet") and parts[1] in (split, f"partial-{split}"):
            shards.setdefault(parts[0], []).append(f)
        elif len(parts) == 2 and f.endswith(f"-{split}.parquet"):  # the older layout: <config>/<name>-<split>.parquet
            shards.setdefault(parts[0], []).append(f)
    if not shards:
        return None, 0, None
    if config:
        chosen = config if config in shards else None
    else:
        chosen = "default" if "default" in shards else sorted(shards)[0]
    if chosen is None:
        return None, 0, None
    group = sorted(shards[chosen])
    return group[0], len(group), chosen


def fetch(dataset: str, directory: Path, revision: str | None = None, split: str = "train", config: str | None = None) -> dict[str, Any]:
    """Download one table of ``dataset`` into ``directory``; returns {path, filename, source}."""
    split = (split or "train").strip()
    config = (config or "").strip() or None
    revision = (revision or "").strip() or None
    dataset = check(dataset, revision, split, config)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ConnectorError("Hugging Face import needs `huggingface_hub` installed in the server environment.") from None
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi()
    rev = revision or "main"
    with _errors(dataset, rev):
        listed = api.list_repo_files(dataset, repo_type="dataset", revision=rev, token=token)
    path, shards = pick_file(listed, split, config)
    used_config, converted = config, False
    if path is None:
        try:
            with _errors(dataset, CONVERTED):
                converted_files = api.list_repo_files(dataset, repo_type="dataset", revision=CONVERTED, token=token)
        except ConnectorError:
            converted_files = []
        path, shards, used_config = pick_converted(converted_files, split, config)
        if path is None:
            found = sorted({p for f in listed for p in PurePosixPath(f).parts[:-1]})[:8]
            hint = f" Folders in the repository: {', '.join(found)}." if found else ""
            raise ConnectorError(f"No {split} split{f' for the configuration {config}' if config else ''} was found in {dataset} "
                                 f"as parquet, csv or json, and Hugging Face has no parquet conversion of it.{hint}")
        rev, converted = CONVERTED, True
    sha, license_ = _info(api, dataset, rev, token)
    pinned = sha or rev  # download the commit that is recorded, even if the branch moves meanwhile
    size = _size(api, dataset, path, pinned, token)
    if size is not None and size > base.MAX_BYTES:
        raise ConnectorError(too_big(f"{dataset} ({path})", size))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".hf-", dir=directory))
    try:
        with _errors(dataset, rev):
            local = Path(hf_hub_download(dataset, path, repo_type="dataset", revision=pinned, token=token, local_dir=scratch))
        actual = local.stat().st_size
        if actual > base.MAX_BYTES:
            raise ConnectorError(too_big(f"{dataset} ({path})", actual))
        repo = dataset.split("/")[-1]
        name = (f"{repo}_{used_config or 'default'}_{split}{PurePosixPath(path).suffix}" if converted
                else f"{repo}_{PurePosixPath(path).name}")
        final = target_path(directory, safe_filename(name))
        shutil.move(str(local), final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    source = {"kind": "hf", "dataset": dataset, "revision": rev, "sha": sha, "split": split, "config": used_config,
              "file": path, "license": license_, "sha256": sha256_file(final), "url": f"https://huggingface.co/datasets/{dataset}"}
    if converted:
        source["converted"] = True  # read from Hugging Face's parquet conversion, not the files the authors uploaded
    if shards > 1:
        source["shards"] = shards
        source["note"] = f"Only the first of {shards} files of the {split} split was imported."
    return {"path": str(final), "filename": final.name, "source": source}


def _size(api, dataset: str, path: str, revision: str, token: str | None) -> int | None:
    try:
        infos = api.get_paths_info(dataset, [path], repo_type="dataset", revision=revision, token=token)
    except Exception:  # noqa: BLE001 — the size is a pre-check; the download is measured again afterwards
        return None
    for item in infos or []:
        size = getattr(item, "size", None)
        if isinstance(size, int):
            return size
    return None


def _info(api, dataset: str, revision: str, token: str | None) -> tuple[str | None, str | None]:
    """The commit sha of ``revision`` and the dataset's licence (card first, then a ``license:`` tag). Best effort."""
    try:
        info = api.dataset_info(dataset, revision=revision, token=token)
    except Exception:  # noqa: BLE001
        return None, None
    license_ = None
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    if isinstance(card, dict):
        license_ = card.get("license")
    elif card is not None:
        license_ = getattr(card, "license", None)
    if not license_:
        license_ = next((t.split(":", 1)[1] for t in (getattr(info, "tags", None) or []) if isinstance(t, str) and t.startswith("license:")), None)
    if isinstance(license_, list):
        license_ = ", ".join(str(x) for x in license_)
    return getattr(info, "sha", None), license_


class _errors:
    """Turn ``huggingface_hub`` errors into ConnectorError with a plain message (gated first: it subclasses not-found)."""

    def __init__(self, dataset: str, revision: str):
        self.dataset, self.revision = dataset, revision

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, tb):
        if exc is None or isinstance(exc, ConnectorError) or not isinstance(exc, Exception):
            return False
        from huggingface_hub import errors

        d = self.dataset
        if isinstance(exc, errors.GatedRepoError):
            message = (f"{d} is a gated dataset: accept the dataset's terms on Hugging Face with your account "
                       "and set HF_TOKEN on the server, then try again.")
        elif isinstance(exc, errors.RepositoryNotFoundError):
            message = f"Hugging Face has no dataset {d} (if it is private, set HF_TOKEN on the server)."
        elif isinstance(exc, errors.RevisionNotFoundError):
            message = f"{d} has no revision {self.revision}."
        elif isinstance(exc, errors.EntryNotFoundError):
            message = f"The file was not found in {d} at revision {self.revision}."
        elif isinstance(exc, errors.DisabledRepoError):
            message = f"{d} has been disabled on Hugging Face."
        elif isinstance(exc, errors.HFValidationError):
            message = f"{d} is not a valid Hugging Face dataset id."
        elif isinstance(exc, errors.HfHubHTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                message = "Hugging Face refused the token (HTTP 401). Check HF_TOKEN on the server."
            elif status == 429:
                message = "Hugging Face is rate-limiting this server (HTTP 429). Wait a minute and try again."
            else:
                message = f"Hugging Face answered HTTP {status or 'error'} for {d}. Try again later."
        elif isinstance(exc, OSError) or type(exc).__module__.startswith(("httpx", "requests", "urllib3")):
            message = f"Could not reach Hugging Face ({type(exc).__name__}). Check the server's internet connection."
        else:
            return False
        raise ConnectorError(message) from None
