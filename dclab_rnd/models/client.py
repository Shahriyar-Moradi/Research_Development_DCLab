"""An OpenAI-compatible chat client with tool calling (the transport under ``dclab_rnd.models.gateway``).

Works against OpenAI, the Hugging Face router (``OPENAI_BASE_URL=https://router.huggingface.co/v1``
with an ``hf_…`` token, the same setup Hugging Face's ML Intern uses), a local Ollama
(``http://127.0.0.1:11434/v1``) or any other endpoint that speaks the chat-completions API.
The key is read from the environment or ``.env``; it is never typed into the UI.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "gpt-5.6-terra"
PAUSE_SECONDS = 600  # after a refusal that will not go away by itself (no credit, a refused key, an unknown model)
# the pause is shared by every app instance and worker (models/pause.py: a row with an expiry on PostgreSQL)


def failure(exc: BaseException) -> tuple[str, bool]:
    """Why a model request failed, in words safe to show, and whether trying again soon is pointless.

    Only the HTTP status and the provider's short error code are read. The provider's message is never passed on:
    it can repeat request headers or parts of a key.
    """
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    code = " ".join(str(error.get(k) or "") for k in ("code", "type")).lower() if isinstance(error, dict) else ""
    if "quota" in code or "credit" in code or "billing" in code:
        return "the model account has no credit left", True
    if status in (401, 403):
        return "the provider refused the key", True
    if status == 404 or "model_not_found" in code:
        return "the model name is not available to this key", True
    if status == 429:
        return "the provider is limiting requests for now", False
    if isinstance(status, int) and status >= 500:
        return "the provider had an error", False
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"):
        return "the model endpoint could not be reached", False
    return "the model request failed", False


def resume() -> None:
    """Forget every pause (a new key or endpoint was configured, or a test starts)."""
    from . import pause

    pause.clear()


def settings() -> dict[str, Any]:
    base_url = os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = os.environ.get("DCLAB_INTERN_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    key = os.environ.get("OPENAI_API_KEY") or ("ollama" if "11434" in base_url else "")
    try:
        import openai  # noqa: F401

        sdk = True
    except ImportError:
        sdk = False
    return {"available": bool(key) and sdk, "model": model, "endpoint": urlparse(base_url).netloc or base_url,
            "sdk_installed": sdk, "key_configured": bool(key)}


class ChatClient:
    """One ``complete`` call = one chat-completions request. A failure is raised as ``RuntimeError("<Type>: <reason>")``
    with a reason safe to show (see ``failure``); after a refusal that will not go away, requests pause for a while."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 180, max_retries: int = 1):
        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = base_url or os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("ollama" if "11434" in self.base_url else "")
        self._client = None
        self.timeout, self.max_retries = timeout, max_retries  # the gateway passes 0: it retries itself, and counts each attempt

    def _limit(self, max_tokens: int | None) -> dict[str, Any]:
        """OpenAI's newer models take ``max_completion_tokens`` (they refuse ``max_tokens``); other OpenAI-compatible
        servers (Ollama, the Hugging Face router, vLLM) take ``max_tokens``. None sends no limit."""
        if max_tokens is None:
            return {}
        return {"max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens": max_tokens}

    def _pause_key(self) -> str:
        from . import pause

        return pause.key(self.base_url, self.model, self.api_key)

    def _ready(self) -> None:
        from . import pause

        until, reason = pause.get(self._pause_key())
        if until > time.time():  # no request is sent: it would be refused again, and each attempt makes the user wait
            raise RuntimeError(f"ModelPaused: {reason}")

    def _failed(self, exc: BaseException) -> RuntimeError:
        reason, pointless = failure(exc)
        if pointless:
            from . import pause

            pause.put(self._pause_key(), time.time() + PAUSE_SECONDS, reason)
        error = RuntimeError(f"{type(exc).__name__}: {reason}")
        # worth trying again soon: a rate limit, a provider error or an unreachable endpoint; never a refusal that stays
        error.transient = not pointless and reason != "the model request failed"  # type: ignore[attr-defined]
        return error

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None) -> dict[str, Any]:
        """Like ``complete`` (same return value), but ``on_text(delta)`` hears the reply as it is written.

        Tool-call arguments arrive in pieces and are joined here; a call whose arguments are not valid JSON keeps the
        raw text under ``_raw``, as in ``complete``.
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, **self._limit(max_tokens), "stream": True,
                                  "stream_options": {"include_usage": True}}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        text: list[str] = []
        pieces: dict[int, dict[str, Any]] = {}
        finish, usage = None, None
        self._ready()
        try:
            for chunk in self.client.chat.completions.create(**kwargs):
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "content", None):
                    text.append(delta.content)
                    if on_text:
                        on_text(delta.content)
                for call in getattr(delta, "tool_calls", None) or []:
                    piece = pieces.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                    piece["id"] = call.id or piece["id"]
                    if call.function:
                        piece["name"] += call.function.name or ""
                        piece["arguments"] += call.function.arguments or ""
                finish = choice.finish_reason or finish
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            raise self._failed(exc) from None
        calls = []
        for index in sorted(pieces):
            piece = pieces[index]
            try:
                arguments = json.loads(piece["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": piece["arguments"]}
            calls.append({"id": piece["id"] or f"call_{index}", "name": piece["name"], "arguments": arguments})
        content = "".join(text)
        return {
            "content": content, "tool_calls": calls, "finish_reason": finish,
            "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0, "output_tokens": getattr(usage, "completion_tokens", 0) or 0},
            "assistant_message": {"role": "assistant", "content": content,
                                  **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})},
        }

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, **options: Any) -> dict[str, Any]:
        """``options`` are passed to the provider as they are (for example ``response_format`` or ``temperature``)."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, **self._limit(max_tokens), **options}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        self._ready()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            raise self._failed(exc) from None
        choice = response.choices[0]
        calls = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call.function.arguments}
            calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
        usage = getattr(response, "usage", None)
        return {
            "content": choice.message.content or "",
            "tool_calls": calls,
            "finish_reason": choice.finish_reason,
            "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0, "output_tokens": getattr(usage, "completion_tokens", 0) or 0},
            "assistant_message": {"role": "assistant", "content": choice.message.content or "",
                                  **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})},
        }
