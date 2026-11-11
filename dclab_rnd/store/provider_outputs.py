"""Whether a provider model's outputs (Claude's, GPT's, Qwen's) may enter training sets: the owner's recorded decisions.

``owner_decisions.json`` beside this file holds them, one entry per kind of answerer:

- ``provider_outputs`` (decided 2026-10-25): a commercial provider's model (Claude, GPT): yes, for evaluation and for
  training, kept silver (weight 0.5) and apart by source;
- ``openrouter_qwen`` (decided 2026-11-11): a Qwen model served through OpenRouter (``source_class`` ``open_weights_model``):
  yes, for training at full weight (1.0), kept apart by answerer; its licence is checked before any fine-tune.

A set of such outputs writes its manifest's consent from here: ``train`` when the owner said yes, ``improve`` (evaluation
only; ``export_chat`` refuses it for training) when not. One change of that file moves every such set together, and each
example's provenance still names where it came from. :func:`openrouter_qwen` is the one rule, for the lake and the store,
of which model answers fall under the 2026-11-11 decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DECISIONS = Path(__file__).with_name("owner_decisions.json")
PROVIDER_OUTPUTS = "provider_outputs"
OPENROUTER_QWEN = "openrouter_qwen"
OPEN_WEIGHTS = "open_weights_model"  # the gateway's source class of an open-weights model served by a host (step 2 of the switch)


def _decision(path: Optional[Path], decision: str) -> dict[str, Any]:
    try:
        doc = json.loads((path or DECISIONS).read_text(encoding="utf-8"))[decision]
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def for_training(path: Optional[Path] = None, decision: str = PROVIDER_OUTPUTS) -> bool:
    return _decision(path, decision).get("for_training") is True  # no record of the owner's yes: evaluation only


def consent(path: Optional[Path] = None, decision: str = PROVIDER_OUTPUTS) -> str:
    return "train" if for_training(path, decision) else "improve"


def weight(path: Optional[Path] = None, decision: str = PROVIDER_OUTPUTS, default: float = 0.5) -> float:
    """The weight the owner gave a decision's examples (Claude's silver 0.5 when unsaid; Qwen's 1.0 as recorded)."""
    try:
        return float(_decision(path, decision).get("weight", default))
    except (TypeError, ValueError):
        return default


def openrouter_qwen(model: Any, provider: Any = None, source_class: Any = None) -> bool:
    """Whether a model's answer falls under the owner's 2026-11-11 decision: a Qwen model served through OpenRouter.

    The exchange's own ``provider``/``source_class`` (written by the gateway) win when present: ``openrouter`` and, when a
    class is said, ``open_weights_model``. Without them, only OpenRouter's own slug counts (``qwen/...`` in lower case, or
    ``openrouter/qwen/...``): a Hugging Face id (``Qwen/Qwen2.5-72B-Instruct:novita``) is another host's, and an Ollama tag
    (``qwen2.5-coder:1.5b``) runs on this machine; neither is this decision's."""
    name = str(model or "").strip()
    if "qwen" not in name.lower():
        return False
    if source_class not in (None, "") and str(source_class) != OPEN_WEIGHTS:
        return False
    if provider not in (None, ""):
        return str(provider).strip().lower() == "openrouter"
    return name.startswith(("qwen/", "openrouter/qwen/"))


def answerer_name(model: Any) -> str:
    """The lake's answerer for a served model, from its name: ``qwen/qwen3.8-flash`` → ``qwen3.8-flash`` (an OpenRouter
    variant such as ``:free`` is the same weights); any other name made safe as a folder name."""
    name = str(model or "").strip().lower()
    if name.startswith("openrouter/"):
        name = name.removeprefix("openrouter/")
    if name.startswith("qwen/"):
        return name.split("/", 1)[1].split(":", 1)[0] or "unknown-model"
    safe = "".join(c if c.isalnum() or c in ".-_" else "-" for c in name).strip("-")
    return safe or "unknown-model"


__all__ = ["for_training", "consent", "weight", "openrouter_qwen", "answerer_name", "DECISIONS", "PROVIDER_OUTPUTS", "OPENROUTER_QWEN",
           "OPEN_WEIGHTS"]
