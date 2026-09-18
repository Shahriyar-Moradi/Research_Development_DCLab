"""Reproducibility metadata for newly executed experiments."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
    "torch",
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def capture_provenance(
    root: Path,
    *,
    data_paths: Iterable[Path] = (),
    random_state: int | None = None,
) -> dict:
    """Capture enough state to trace a result back to code, data, and runtime."""
    root = root.resolve()
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue

    data: dict[str, str] = {}
    for item in data_paths:
        path = Path(item)
        if not path.is_file():
            continue
        try:
            label = str(path.resolve().relative_to(root))
        except ValueError:
            label = path.name
        data[label] = file_sha256(path)

    status = _git(root, "status", "--porcelain")
    return {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "random_state": random_state,
        "packages": versions,
        "data_sha256": data,
    }

