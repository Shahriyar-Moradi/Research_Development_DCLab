"""Whether a provider model's outputs (Claude's, GPT's) may enter training sets: the owner's recorded decision.

``owner_decisions.json`` beside this file holds it (decided 2026-10-25: yes, for evaluation and for training, kept silver
and apart by source). A set of such outputs writes its manifest's consent from here: ``train`` when the owner said yes,
``improve`` (evaluation only; ``export_chat`` refuses it for training) when not. One change of that file moves every such
set together, and each example's provenance still names where it came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DECISIONS = Path(__file__).with_name("owner_decisions.json")


def for_training(path: Optional[Path] = None) -> bool:
    try:
        return bool(json.loads((path or DECISIONS).read_text(encoding="utf-8"))["provider_outputs"]["for_training"])
    except (OSError, ValueError, KeyError, TypeError):
        return False  # no record of the owner's yes: evaluation only


def consent(path: Optional[Path] = None) -> str:
    return "train" if for_training(path) else "improve"


__all__ = ["for_training", "consent", "DECISIONS"]
