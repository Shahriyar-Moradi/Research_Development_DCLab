"""The owner's rules for live test scripts: never a Claude model in a test, arm, person or judge, at any provider
(2026-11-15), and every answered call keeps its thinking (rule 9: ``thinking_record``).

    from no_claude import refuse_claude
    load_dotenv(...)      # first: the settings the run will use
    refuse_claude()       # exits when any of them would reach a Claude model

``claude_settings`` reads what the run would use, after ``.env`` is loaded: every model tier as the gateway resolves it
(its model, provider and endpoint), and the model and endpoint settings themselves. A name or an endpoint that says
"claude" or "anthropic" refuses the run, so a Claude model through OpenRouter ("anthropic/claude-…") is refused as
well as ``DCLAB_LLM_PROVIDER=anthropic``. ``ANTHROPIC_MODEL`` alone is not read: it names the model only for Anthropic's
own provider, which the tiers then show. The refusal names the settings, never their values (a key is never printed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORDS = ("claude", "anthropic")
_SETTING_PARTS = ("MODEL", "PROVIDER", "BASE_URL")
_SETTING_PREFIXES = ("DCLAB_", "OPENROUTER_", "OPENAI_")


def _says_claude(value: object) -> bool:
    text = str(value or "").lower()
    return any(w in text for w in WORDS)


def claude_settings() -> list[str]:
    """The settings (by name) that would send this run to a Claude model; empty when none would."""
    from dclab_rnd.models import settings

    # reviewer round 2: DCLAB_BLOCK_PROVIDERS=anthropic blocks Claude (it names no model), and an endpoint the owner
    # blocked serves nothing: neither refuses a run
    found = [name for name, value in sorted(os.environ.items())
             if name.startswith(_SETTING_PREFIXES) and any(p in name for p in _SETTING_PARTS) and "KEY" not in name
             and "BLOCK" not in name and _says_claude(value) and not ("BASE_URL" in name and settings.blocked(value))]
    for name in settings.ROUTABLE:
        try:
            tier = settings.tier(name)
        except Exception:  # noqa: BLE001 — a tier that cannot be resolved serves nothing
            continue
        if tier.get("blocked"):
            continue
        found += [f"tier {name}: {field}" for field in ("model", "provider", "base_url") if _says_claude(tier.get(field))]
    try:
        local = settings.embedding_tier()
    except Exception:  # noqa: BLE001
        local = None
    if local:
        found += [f"tier embedding_local: {field}" for field in ("model", "provider", "base_url") if _says_claude(local.get(field))]
    return found


def refuse_claude(model_used: bool = True) -> None:
    """Exit before any model call when a Claude model is configured (call it after ``.env`` is loaded). A run that calls no
    model (``model_used=False``: the rules, or a report of a finished run) is not refused: live models are turned off for
    it instead (DCLAB_NO_LIVE_MODELS=1), so nothing in it can reach the Claude model the settings name."""
    found = claude_settings()
    if found and not model_used:
        os.environ["DCLAB_NO_LIVE_MODELS"] = "1"
        print("note: a Claude model is configured (" + "; ".join(found) + "); this run calls no model, so live models are off")
        return
    if found:
        sys.exit("refused: a Claude model is configured (" + "; ".join(found) + "); the owner's rule: never a Claude model "
                 "in a test. Configure an open model (Qwen on OpenRouter) and run again.")


def thinking_record(home: Path) -> dict:
    """Rule 9's check on a live run's kept exchanges (``<home>/model_exchanges/**.jsonl``): every call that answered
    (``outcome`` "ok") kept its thinking; a call that failed (a timeout, a 429: ``outcome`` "error: …") has none to keep
    and is counted apart (reviewer round 2: one timeout failed the whole run)."""
    import json

    folder = Path(home) / "model_exchanges"
    kept = [json.loads(line) for f in (sorted(folder.rglob("*.jsonl")) if folder.exists() else ())
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    answered = [k for k in kept if k.get("outcome", "ok") == "ok"]
    thought = sum(1 for k in answered if k.get("thinking") or k.get("reasoning") or k.get("reasoning_details"))
    out = {"model_calls": len(kept), "answered_calls": len(answered), "errored_calls": len(kept) - len(answered),
           "errors": sorted({str(k.get("outcome")) for k in kept if k.get("outcome", "ok") != "ok"}),
           "calls_with_thinking": thought, "purposes": sorted({k.get("purpose", "") for k in kept})}
    out["thinking_kept"] = thought == len(answered)  # an answered call without its thinking fails the run
    if not out["thinking_kept"]:
        out["failed"] = f"{len(answered) - thought} answered model call(s) without their thinking"
    return out


__all__ = ["WORDS", "claude_settings", "refuse_claude", "thinking_record"]
