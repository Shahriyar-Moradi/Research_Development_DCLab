"""Secrets from files (package 12.2): ``NAME_FILE=/run/secrets/name`` sets ``NAME`` from that file when ``NAME`` is not set.

A key or a connection string comes from a Docker secret or a platform's secret store, never from an image or a
committed file. Read when the package is imported, so the server, the worker and every command-line tool see the
same values (``python -m dclab_rnd.storage upgrade`` in the container's start, ``accounts add``, the MCP server).

Only the product's own secret settings are read this way: a name in ``NAMES``, a read-only data source
``DCLAB_DB_<NAME>`` or a model tier's key ``DCLAB_TIER_<TIER>_API_KEY``. Other ``*_FILE`` variables (AWS's own
``AWS_SHARED_CREDENTIALS_FILE``, ``DCLAB_PRICES_FILE``) keep their meaning and are left alone.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

NAMES = {"OPENAI_API_KEY", "DCLAB_DATABASE_URL", "DCLAB_TEST_DATABASE_URL", "DCLAB_OIDC_CLIENT_SECRET", "DCLAB_SECRET_KEY", "HF_TOKEN",
         "KAGGLE_USERNAME", "KAGGLE_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "DCLAB_NEW_PASSWORD",
         "DCLAB_FILES_URL"}
PATTERNS = (re.compile(r"DCLAB_DB_[A-Z0-9_]+"), re.compile(r"DCLAB_TIER_[A-Z0-9]+_API_KEY"))
NOT_SECRETS = {"DCLAB_DB_POOL_SIZE", "DCLAB_DB_MAX_OVERFLOW", "DCLAB_DB_PASSWORD"}
ERRORS: list[str] = []  # what could not be read at import: Settings.problems refuses to start with it


def secret_name(target: str) -> bool:
    return target in NAMES or (target not in NOT_SECRETS and any(p.fullmatch(target) for p in PATTERNS))


def load(strict: bool = True) -> list[str]:
    """Set each secret from its file; returns the names set (never their values). ``strict``: an unreadable file
    raises RuntimeError; otherwise it is remembered in ERRORS for the start-up check."""
    set_now = []
    for name, path in list(os.environ.items()):
        if not name.endswith("_FILE") or not secret_name(name[: -len("_FILE")]):
            continue
        target = name[: -len("_FILE")]
        if os.environ.get(target):  # a value set directly wins
            continue
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            message = f"{name} names a file that cannot be read ({type(exc).__name__})"
            if strict:
                raise RuntimeError(message) from None
            if message not in ERRORS:
                ERRORS.append(message)
            continue
        if value:
            os.environ[target] = value
            set_now.append(target)
    return set_now
