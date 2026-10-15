"""Anthropic's Messages API behind the same ``complete``/``stream`` shape as ``models.client.ChatClient``.

The gateway hands every caller OpenAI-style messages and tools; this transport translates them to the Messages API
and the reply back, so no caller changes: a ``system`` message becomes the system prompt, an assistant's
``tool_calls`` become ``tool_use`` blocks, ``tool`` messages become one user turn of ``tool_result`` blocks, and a
reply's ``tool_use`` blocks come back as ``tool_calls``. ``response_format`` asks for JSON: a schema is sent as
``output_config.format`` (structured outputs); a plain JSON object is asked for in the system prompt. Anthropic serves
no embeddings: ``embed`` is refused, and the gateway does not offer the ``embedding`` purpose on this provider.

Set it with ``DCLAB_LLM_PROVIDER=anthropic`` and ``ANTHROPIC_API_KEY`` in ``.env`` (models/settings.py).
"""

from __future__ import annotations

import json
from typing import Any

from .client import ChatClient

BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-sonnet-5-5"
from .prices import DEFAULT_OUTPUT_ESTIMATE as DEFAULT_MAX_TOKENS  # the Messages API needs a limit; the cost estimate assumes the same one
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length", "refusal": "content_filter"}
JSON_ONLY = "Reply with one JSON object and nothing else."


def _text(content: Any) -> str:
    """An OpenAI message content (a string or a list of parts) as plain text."""
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def to_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """(system prompt, Messages API turns). Consecutive turns of one role are joined: the API alternates roles."""
    system: list[str] = []
    turns: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            text = _text(m.get("content"))
            if text:
                system.append(text)
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _text(m.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                raw = fn.get("arguments")
                try:
                    arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    arguments = {"_raw": raw}
                blocks.append({"type": "tool_use", "id": call.get("id") or "", "name": fn.get("name") or "", "input": arguments})
            add("assistant", blocks)
        elif role == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "", "content": _text(m.get("content")) or "(empty)"}])
        else:
            text = _text(m.get("content"))
            add("user", [{"type": "text", "text": text or "(empty)"}])
    return "\n\n".join(system), turns


def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for t in tools or []:
        fn = t.get("function") or t
        out.append({"name": fn.get("name"), "description": fn.get("description") or "",
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _options(options: dict[str, Any], system: str) -> tuple[dict[str, Any], str]:
    """The caller's options as the Messages API takes them, and the system prompt (a JSON request may add a line)."""
    out: dict[str, Any] = {}
    for name in ("temperature", "top_p"):
        if options.get(name) is not None:
            out[name] = options[name]
    if options.get("stop"):
        stop = options["stop"]
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    fmt = options.get("response_format") or {}
    if fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema") or fmt.get("schema")
        if schema:
            out["extra_body"] = {"output_config": {"format": {"type": "json_schema", "schema": schema}}}
    elif fmt.get("type") == "json_object":
        system = f"{system}\n\n{JSON_ONLY}" if system else JSON_ONLY
    return out, system


def from_message(message: Any) -> dict[str, Any]:
    """A Messages API reply in the shape ``ChatClient.complete`` returns."""
    text, calls = [], []
    for block in getattr(message, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            text.append(getattr(block, "text", "") or "")
        elif kind == "tool_use":
            arguments = getattr(block, "input", None)
            calls.append({"id": getattr(block, "id", ""), "name": getattr(block, "name", ""),
                          "arguments": arguments if isinstance(arguments, dict) else {"_raw": arguments}})
    usage = getattr(message, "usage", None)
    content = "".join(text)
    stop = getattr(message, "stop_reason", None)
    return {
        "content": content, "tool_calls": calls, "finish_reason": FINISH.get(stop, stop),
        "usage": {"input_tokens": (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                                  + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
                  "output_tokens": getattr(usage, "output_tokens", 0) or 0},
        "assistant_message": {"role": "assistant", "content": content,
                              **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                                 for c in calls]} if calls else {})},
    }


class AnthropicClient(ChatClient):
    """``ChatClient`` over the Messages API: the same pause after a refusal that stays, the same safe failure reasons
    (Anthropic's errors carry a status and an error type, read by ``client.failure``; their message is never passed on)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 180, max_retries: int = 1):
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or BASE_URL
        self.api_key = api_key or ""
        self._client = None
        self.timeout, self.max_retries = timeout, max_retries

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int | None, options: dict[str, Any]) -> dict[str, Any]:
        system, turns = to_messages(messages)
        extra, system = _options(options, system)
        kwargs: dict[str, Any] = {"model": self.model, "messages": turns, "max_tokens": max_tokens or DEFAULT_MAX_TOKENS, **extra}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs.update(tools=to_tools(tools), tool_choice={"type": "auto"})
        return kwargs

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, **options: Any) -> dict[str, Any]:
        kwargs = self._request(messages, tools, max_tokens, options)
        self._ready()
        try:
            message = self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            raise self._failed(exc) from None
        return from_message(message)

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None) -> dict[str, Any]:
        kwargs = self._request(messages, tools, max_tokens, {})
        self._ready()
        try:
            with self.client.messages.stream(**kwargs) as events:
                for delta in events.text_stream:
                    if on_text and delta:
                        on_text(delta)
                message = events.get_final_message()
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            raise self._failed(exc) from None
        return from_message(message)

    def embed(self, texts: list[str]) -> dict[str, Any]:
        raise RuntimeError("NotSupported: Anthropic serves no embeddings; evidence search uses its lexical method")
