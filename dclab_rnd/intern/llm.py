"""An OpenAI-compatible chat client with tool calling.

Works against OpenAI, the Hugging Face router (``OPENAI_BASE_URL=https://router.huggingface.co/v1``
with an ``hf_…`` token, the same setup Hugging Face's ML Intern uses), a local Ollama
(``http://127.0.0.1:11434/v1``) or any other endpoint that speaks the chat-completions API.
The key is read from the environment or ``.env``; it is never typed into the UI.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "gpt-5.6-terra"


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
    """One ``complete`` call = one chat-completions request. Errors surface by type only."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = base_url or os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("ollama" if "11434" in self.base_url else "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=180, max_retries=1)
        return self._client

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 1800, on_text=None) -> dict[str, Any]:
        """Like ``complete`` (same return value), but ``on_text(delta)`` hears the reply as it is written.

        Tool-call arguments arrive in pieces and are joined here; a call whose arguments are not valid JSON keeps the
        raw text under ``_raw``, as in ``complete``.
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                                  "stream_options": {"include_usage": True}}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        text: list[str] = []
        pieces: dict[int, dict[str, Any]] = {}
        finish, usage = None, None
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
            raise RuntimeError(f"{type(exc).__name__}: the model request failed (check the key, the endpoint and the model name)") from None
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

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 1800) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            raise RuntimeError(f"{type(exc).__name__}: the model request failed (check the key, the endpoint and the model name)") from None
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
