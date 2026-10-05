"""The model gateway: every model request DCLab makes goes through here (package A1.1).

    gateway = Gateway(open_usage(home))
    client = gateway.client("home_agent", draft_id=...)      # None when no model serves that purpose
    client.complete(messages, tools) / client.stream(messages, tools, on_text=...)

A client has the same ``complete``/``stream`` shape as ``models.client.ChatClient``, so callers do not change. The
gateway adds what no caller should do by itself: the purpose decides the tier and the timeout; a rate limit, a
provider error or an unreachable endpoint is retried with backoff (a stream only before its first word reached the
user); and every request is written to the usage log with its tokens, time, attempts and outcome.

Not here yet: prices and budgets (A1.2).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from . import settings
from .client import ChatClient
from .usage import UsageLog, now

RETRIES = int(os.environ.get("DCLAB_MODEL_RETRIES", "2"))  # attempts after the first, for transient failures only
BACKOFF = float(os.environ.get("DCLAB_MODEL_BACKOFF", "1.5"))  # seconds, doubled after each attempt


def _transport(t: dict[str, Any], p: settings.Purpose) -> ChatClient:
    return ChatClient(model=t["model"], base_url=t["base_url"], api_key=t["api_key"], timeout=p.timeout, max_retries=0)


class Gateway:
    def __init__(self, usage: UsageLog | None = None, transport: Callable[[dict[str, Any], settings.Purpose], Any] = _transport,
                 sleep: Callable[[float], None] = time.sleep):
        self.usage, self.transport, self.sleep = usage, transport, sleep

    def available(self, purpose: str) -> bool:
        """A model serves this purpose: its tier has a key, and the SDK is installed (or a test supplies the transport).
        With DCLAB_NO_LIVE_MODELS=1 (set by make rd-check, make test-pg and make product-e2e) no live endpoint serves
        anything, whatever .env says: tests use scripted transports only."""
        p = settings.purpose(purpose)
        if not settings.tier(p.tier)["api_key"]:
            return False
        if self.transport is _transport and os.environ.get("DCLAB_NO_LIVE_MODELS") == "1":
            return False
        return self.transport is not _transport or settings.public(p.tier)["available"]

    def client(self, purpose: str, *, project_id: str | None = None, draft_id: str | None = None, model: str | None = None) -> "Bound | None":
        """A client for one purpose, or None when no model serves it (callers then take their deterministic path).
        ``model`` replaces the tier's model on the same endpoint (a command-line choice); the usage log records it."""
        p = settings.purpose(purpose)
        if not self.available(purpose):
            return None
        tier = settings.tier(p.tier)
        return Bound(self, purpose, p, {**tier, "model": model} if model else tier, project_id, draft_id)

    def record(self, purpose: str, model: str, usage: dict[str, Any] | None, seconds: float, attempts: int, outcome: str,
               project_id: str | None = None, draft_id: str | None = None, prompt: str | None = None) -> None:
        """Write one usage entry. Clients that cannot be routed through the gateway (the campaign agent's NOOA client)
        call this after each request, so every request is still counted."""
        if self.usage is None:
            return
        p = settings.purpose(purpose)
        try:
            self.usage.record({"at": now(), "purpose": purpose, "tier": p.tier, "model": model, "endpoint": settings.public(p.tier)["endpoint"],
                               "project_id": project_id, "draft_id": draft_id, "input_tokens": int((usage or {}).get("input_tokens") or 0),
                               "output_tokens": int((usage or {}).get("output_tokens") or 0), "seconds": round(seconds, 2),
                               "attempts": attempts, "outcome": outcome, "prompt": prompt if os.environ.get("DCLAB_LOG_PROMPTS") == "1" else None})
        except Exception:  # noqa: BLE001 — a usage row that cannot be written must not lose the answer
            pass

    def summary(self) -> dict[str, Any]:
        """What the pages show: tiers without keys, purposes with what they may see, and usage."""
        return {"tiers": [settings.public(t) for t in settings.TIERS],
                "purposes": [{"purpose": k, "tier": v.tier, "timeout": v.timeout, "may_see": v.may_see} for k, v in settings.PURPOSES.items()],
                "usage": self.usage.totals() if self.usage else None}


class Bound:
    def __init__(self, gateway: Gateway, purpose: str, p: settings.Purpose, tier: dict[str, Any], project_id: str | None, draft_id: str | None):
        self.gateway, self.purpose, self.p, self.tier, self.project_id, self.draft_id = gateway, purpose, p, tier, project_id, draft_id
        self.model = tier["model"]
        self._transport = gateway.transport(tier, p)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, **options: Any) -> dict[str, Any]:
        return self._call(lambda: self._transport.complete(messages, tools, max_tokens, **options), messages, can_retry=lambda: True)

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None) -> dict[str, Any]:
        heard = threading.Event()

        def relay(delta: str) -> None:
            heard.set()
            if on_text:
                on_text(delta)
        send = getattr(self._transport, "stream", None)
        if send is None:  # a transport without streaming: answer whole, then hand the text over in one piece
            out = self.complete(messages, tools, max_tokens)
            if on_text and out.get("content"):
                on_text(out["content"])
            return out
        # once a word has reached the user a retry would repeat it, so only a stream that sent nothing is retried
        return self._call(lambda: send(messages, tools, max_tokens, on_text=relay), messages, can_retry=lambda: not heard.is_set())

    def _call(self, send: Callable[[], dict[str, Any]], messages: list[dict[str, Any]], can_retry: Callable[[], bool]) -> dict[str, Any]:
        started, attempts, delay = time.monotonic(), 0, BACKOFF
        while True:
            attempts += 1
            try:
                out = send()
            except RuntimeError as error:
                transient = bool(getattr(error, "transient", False))
                if transient and attempts <= RETRIES and can_retry():
                    self.gateway.sleep(delay)
                    delay *= 2
                    continue
                self._record(messages, None, time.monotonic() - started, attempts, str(error).split(": ", 1)[-1][:200])
                raise
            self._record(messages, out, time.monotonic() - started, attempts, "ok")
            return out

    def _record(self, messages: list[dict[str, Any]], out: dict[str, Any] | None, seconds: float, attempts: int, outcome: str) -> None:
        prompt = "\n\n".join(f"[{m.get('role')}] {m.get('content') or ''}" for m in messages)[:20000] if os.environ.get("DCLAB_LOG_PROMPTS") == "1" else None
        self.gateway.record(self.purpose, self.model, (out or {}).get("usage"), seconds, attempts, outcome, self.project_id, self.draft_id, prompt)


# One gateway per process for code that runs outside a request (the engine's stage notes). The server installs its own.
_INSTALLED: Gateway | None = None


def install(gateway: Gateway) -> None:
    global _INSTALLED
    _INSTALLED = gateway


def installed() -> Gateway:
    return _INSTALLED or Gateway()


def for_workspace(home=None) -> Gateway:
    """A gateway that logs to the workspace's usage log, for command-line tools that run outside the server."""
    from pathlib import Path

    from .usage import open_usage

    root = Path(__file__).resolve().parents[2]
    return Gateway(open_usage(Path(home or os.environ.get("DCLAB_AGENT_HOME") or root / "agent_runs")))
