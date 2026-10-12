"""The prompts of dclab_core's roles, as versioned files (package 18.1).

Each ``<role>.md`` is the system prompt of one role. A host's Advisor builds its request from ``text(purpose)`` and the
role's evidence, so the R&D studio and the main product send the same words. Prompts are code: a person edits them,
``HASHES.json`` pins them, and a test fails when a prompt changes without its hash being recorded
(``python -m dclab_core.prompts --record``), which is the moment to measure the role again (Part 7's gates).
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HASHES = HERE / "HASHES.json"


@functools.cache
def text(name: str) -> str:
    raw = (HERE / f"{name}.md").read_text(encoding="utf-8")
    return raw[:-1] if raw.endswith("\n") else raw


def names() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.md"))


def fingerprints() -> dict[str, str]:
    return {name: hashlib.sha256((HERE / f"{name}.md").read_bytes()).hexdigest()[:16] for name in names()}


def recorded() -> dict[str, str]:
    return json.loads(HASHES.read_text(encoding="utf-8")) if HASHES.exists() else {}

