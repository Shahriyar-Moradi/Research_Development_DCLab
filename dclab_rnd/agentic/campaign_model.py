"""Which provider the research campaign's NOOA agents call through litellm.

The product runs on Claude (``DCLAB_LLM_PROVIDER=anthropic``, the owner's choice; OpenAI blocked on this machine since
2026-10-25): the campaign then calls the standard tier's Claude model with its key through litellm's own Anthropic path
(``anthropic/<model>``), never an OpenAI endpoint. Otherwise it keeps its OpenAI-compatible endpoint (``OPENAI_API_KEY``,
``OPENAI_BASE_URL``), refused when the owner blocked that provider. The key is handed to litellm only, never logged.
"""
from __future__ import annotations

import os
from typing import Any, Optional

OPENAI_DEFAULT = "gpt-5.6-terra"
# NOOA's typed answers reach Anthropic through litellm as a forced tool call; these models refuse forced tool use
# ("tool_choice: type tool and any are not supported"), so the campaign runs on CLAUDE_TYPED (same price as Sonnet 5.5)
# unless DCLAB_CAMPAIGN_MODEL names another.
NO_FORCED_TOOL = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1")
CLAUDE_TYPED = "claude-sonnet-5"


def choice() -> Optional[dict[str, Any]]:
    """The provider, its model, its endpoint and the key for litellm, or None when neither is configured."""
    from ..models import settings as model_settings
    from ..models.live_guard import blocked

    if model_settings.anthropic_default():
        t = model_settings.tier("standard")
        if t.get("provider") == "anthropic" and t.get("api_key") and t.get("model"):
            model = os.environ.get("DCLAB_CAMPAIGN_MODEL", "").strip() or typed(t["model"])
            return {"provider": "anthropic", "model": model, "base_url": t["base_url"], "api_key": t["api_key"]}
    if os.environ.get("OPENAI_API_KEY"):
        url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        if blocked(url):
            return None
        return {"provider": "openai", "model": os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT), "base_url": url, "api_key": None}
    return None


def why_not() -> str:
    """What to set when the campaign has no model to call."""
    return ("The research campaign needs a model: the product's Claude (DCLAB_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY "
            "in .env), or an OpenAI-compatible endpoint (OPENAI_API_KEY) that is not blocked; restart after setting it. "
            "Never paste a key in a research goal.")


def priced(model: str) -> bool:
    """Whether the gateway prices this model: an unpriced one is never counted by the euro caps, so it is not run."""
    from ..models.prices import price_of

    return price_of(bare(model)) is not None


def typed(model: str) -> str:
    """A Claude model that answers NOOA's typed questions: the model itself, or CLAUDE_TYPED for one that refuses forced tools."""
    return CLAUDE_TYPED if bare(model) in NO_FORCED_TOOL else model


def model_for(requested: str, picked: dict[str, Any]) -> str:
    """The model a run uses: the request's when it belongs to the chosen provider (a Claude that answers typed questions),
    else the provider's own."""
    claude = requested.startswith("claude-")
    if (picked["provider"] == "anthropic") == claude:
        if not claude:
            return requested
        chosen = typed(requested)
        return chosen if priced(chosen) else picked["model"]  # a Claude the caps cannot count is never sent
    return picked["model"]


def client_args(model: str, provider: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    """litellm's model name and extra settings for NOOA's client: ``anthropic/<model>`` with the key (and fields only
    OpenAI takes dropped), or ``openai/<model>``. ``provider``: the one the run began on; a run never moves to another
    (its model belongs to it). The live guard refuses a blocked provider, and a real client inside a test; litellm's own
    HTTP client never passes through the gateway's, so the guard is asked here."""
    from ..models import live_guard

    picked = choice()
    if picked is None:
        raise RuntimeError(why_not())
    if provider and provider != picked["provider"]:
        raise RuntimeError(f"This run began on {provider}; the campaign now runs on {picked['provider']}: start a new run")
    live_guard.check(picked["provider"], picked["base_url"])
    if picked["provider"] == "anthropic" and not priced(model):
        raise RuntimeError(f"{bare(model)} has no price, so the euro caps could not count it: name a priced Claude model")
    if picked["provider"] == "anthropic":
        return "anthropic/" + bare(model), {"api_key": picked["api_key"], "drop_params": True}
    return "openai/" + bare(model), {}


def endpoint() -> str:
    """Where the requests go, for the usage log and its prices; an OpenAI endpoint the owner blocked raises."""
    from ..models.live_guard import ProviderBlocked, blocked

    picked = choice()
    url = picked["base_url"] if picked else (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    if blocked(url):
        raise ProviderBlocked(f"This provider is blocked on this machine (DCLAB_BLOCK_PROVIDERS in .env): nothing is sent to {url}")
    return url


def bare(model: str) -> str:
    """The model's own name, without litellm's provider prefix."""
    return str(model).split("/", 1)[-1]


__all__ = ["CLAUDE_TYPED", "NO_FORCED_TOOL", "OPENAI_DEFAULT", "bare", "choice", "client_args", "endpoint", "model_for", "typed", "why_not"]
