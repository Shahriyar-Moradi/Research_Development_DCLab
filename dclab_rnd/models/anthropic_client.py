"""Anthropic's Messages API behind the same ``complete``/``stream`` shape as ``models.client.ChatClient``.

The gateway hands every caller OpenAI-style messages and tools; this transport translates them to the Messages API
and the reply back, so no caller changes: a ``system`` message becomes the system prompt, an assistant's
``tool_calls`` become ``tool_use`` blocks, ``tool`` messages become one user turn of ``tool_result`` blocks, and a
reply's ``tool_use`` blocks come back as ``tool_calls``. ``response_format`` asks for JSON: a schema is sent as
``output_config.format`` (structured outputs); a plain JSON object is asked for in the system prompt. Anthropic serves
no embeddings: ``embed`` is refused, and the gateway does not offer the ``embedding`` purpose on this provider.

Thinking (25.3): ``thinking="adaptive"`` or ``thinking="enabled:<budget tokens>"`` (an option of ``complete``, or the
client's ``thinking`` attribute) asks for Claude's extended thinking; the reply's thinking blocks come back as
``thinking`` (``[{"text", "signature"}]``, and ``redacted_thinking``, a count) and ride on the assistant message as
``thinking_blocks``, which ``to_messages`` sends back first in that assistant turn, as the API requires during tool use.
A temperature is not sent with thinking (the API refuses the two together).

Prompt caching (platform.claude.com/docs/en/build-with-claude/prompt-caching): every request marks its stable prefix for
the cache, the last tool (so every tool definition) and the system prompt; a conversation already under way (an
assistant turn in it, so it will be sent again with more) also gets the API's automatic breakpoint, which moves to its
end. A single request with a new question gets none there, so it never pays the write premium on text no request will
read again. ``prices.caching()`` gives the time to live (DCLAB_PROMPT_CACHE_TTL, five minutes or an hour) or turns it
off (DCLAB_PROMPT_CACHE=off); a prompt below the model's minimum (512 tokens for Sonnet 5.5) is simply not cached. The
reply's usage keeps every input token in ``input_tokens`` and says how many were read from the cache and written to it.

Set it with ``DCLAB_LLM_PROVIDER=anthropic`` and ``ANTHROPIC_API_KEY`` in ``.env`` (models/settings.py).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from . import prices
from .client import ChatClient

BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-sonnet-5-5"
from .prices import DEFAULT_OUTPUT_ESTIMATE as DEFAULT_MAX_TOKENS  # the Messages API needs a limit; the cost estimate assumes the same one
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length", "refusal": "content_filter"}
JSON_ONLY = "Reply with one JSON object and nothing else."
ADAPTIVE_HEADROOM = 8000  # 25.3: tokens added to a request's limit for adaptive thinking, above the reply's own


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
            blocks: list[dict[str, Any]] = [dict(b) for b in m.get("thinking_blocks") or [] if isinstance(b, dict)
                                            and b.get("type") in ("thinking", "redacted_thinking")]
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


# Models that still take a thinking budget; every newer one refuses it with a 400 (2026-10-28, on claude-sonnet-5-5:
# '"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort"')
BUDGET_MODELS = re.compile(r"claude-(?:haiku-4-5|sonnet-4-5|opus-4-5|opus-4-1|opus-4-0|sonnet-4-0|sonnet-4-2|opus-4-2|sonnet-4@|opus-4@|3)")
NO_THINKING_MODELS: set[str] = set()  # models that refused a thinking request in this process: asked again without, never twice


def thinking_param(setting: Any, model: Any = None) -> dict[str, Any] | None:
    """``adaptive`` -> adaptive with a summary of the thinking; ``enabled:N`` -> a budget of N on a model that takes one,
    adaptive on any newer model (which refuses a budget); off -> None. The summary is asked for explicitly: newer models
    omit the thinking's text by default ("display": "omitted"), which is why earlier exchanges kept empty blocks."""
    if not setting or str(setting).strip().lower() in ("0", "off", "none", "false"):
        return None
    text = str(setting).strip().lower()
    if model and str(model) in NO_THINKING_MODELS:
        return None
    if text.startswith("enabled") and model and not BUDGET_MODELS.search(str(model)):
        return {"type": "adaptive", "display": "summarized"}
    if not text.startswith("enabled") and model and BUDGET_MODELS.search(str(model)):
        return {"type": "enabled", "budget_tokens": 4000, "display": "summarized"}  # an older model takes a budget, not adaptive
    if text.startswith("enabled"):
        _, _, n = text.partition(":")
        try:
            budget = int(n or 4000)
        except ValueError:
            budget = 4000  # a misspelt budget is the default one, never a failure of the request
        return {"type": "enabled", "budget_tokens": max(1024, budget), "display": "summarized"}
    return {"type": "adaptive", "display": "summarized"}  # newer models omit the thinking's text unless a summary is asked for


EFFORTS = ("low", "medium", "high", "xhigh", "max")  # output_config.effort; unset leaves the model's default
SAMPLING = ("temperature", "top_p")
NO_SAMPLING: set[str] = set()  # models that refused a sampling option in this process (the API says it is deprecated for them)


def _refuses_thinking(exc: BaseException) -> bool:
    """A 400 that refuses the thinking request for this model (a budget on a newer model, adaptive on an older one)."""
    text = str(getattr(exc, "message", "") or exc).lower()
    return getattr(exc, "status_code", 400) == 400 and "thinking" in text and any(
        w in text for w in ("not supported", "unsupported", "not allowed", "invalid", "use \"thinking"))


def _refuses_sampling(exc: BaseException) -> bool:
    """A 400 that refuses ``temperature`` or ``top_p`` for this model (not any other bad request)."""
    text = str(getattr(exc, "message", "") or exc).lower()
    return getattr(exc, "status_code", 400) == 400 and any(k in text for k in SAMPLING) and any(
        w in text for w in ("deprecated", "not supported", "unsupported", "not allowed"))


def _options(options: dict[str, Any], system: str, model: Any = None) -> tuple[dict[str, Any], str]:
    """The caller's options as the Messages API takes them, and the system prompt (a JSON request may add a line)."""
    out: dict[str, Any] = {}
    thinking = thinking_param(options.get("thinking"), model)
    if thinking:
        out["thinking"] = thinking
    for name in ("temperature", "top_p"):
        if options.get(name) is not None and not thinking:
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
    effort = str(options.get("effort") or os.environ.get("DCLAB_THINKING_EFFORT", "")).strip().lower()
    if thinking and thinking.get("type") == "adaptive" and effort in EFFORTS:  # how deeply the model thinks (newer models)
        body = out.setdefault("extra_body", {})
        body["output_config"] = {**(body.get("output_config") or {}), "effort": effort}
    return out, system


def from_message(message: Any) -> dict[str, Any]:
    """A Messages API reply in the shape ``ChatClient.complete`` returns."""
    text, calls, thought, raw_thinking, redacted = [], [], [], [], 0
    for block in getattr(message, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "thinking":
            thought.append({"text": getattr(block, "thinking", "") or "", "signature": getattr(block, "signature", "") or ""})
            raw_thinking.append({"type": "thinking", "thinking": thought[-1]["text"], "signature": thought[-1]["signature"]})
        elif kind == "redacted_thinking":
            redacted += 1
            raw_thinking.append({"type": "redacted_thinking", "data": getattr(block, "data", "") or ""})
        elif kind == "text":
            text.append(getattr(block, "text", "") or "")
        elif kind == "tool_use":
            arguments = getattr(block, "input", None)
            calls.append({"id": getattr(block, "id", ""), "name": getattr(block, "name", ""),
                          "arguments": arguments if isinstance(arguments, dict) else {"_raw": arguments}})
    usage = getattr(message, "usage", None)
    content = "".join(text)
    stop = getattr(message, "stop_reason", None)
    out = {
        "content": content, "tool_calls": calls, "finish_reason": FINISH.get(stop, stop),
        "usage": _usage(usage),
        "assistant_message": {"role": "assistant", "content": content,
                              **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                                 for c in calls]} if calls else {}),
                              **({"thinking_blocks": raw_thinking} if raw_thinking else {})},
    }
    if thought or redacted:
        out["thinking"] = thought
        out["redacted_thinking"] = redacted
    return out


def _usage(usage: Any) -> dict[str, int]:
    """The reply's tokens: ``input_tokens`` counts every input token (read from the cache, written to it, or neither), and
    the cache's share is said apart, the hour-long writes within the writes."""
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    write_1h = int(getattr(getattr(usage, "cache_creation", None), "ephemeral_1h_input_tokens", 0) or 0)
    return {"input_tokens": int(getattr(usage, "input_tokens", 0) or 0) + read + write, "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_read_tokens": read, "cache_write_tokens": write, "cache_write_1h_tokens": min(write_1h, write)}


def _cache_mark(ttl: str) -> dict[str, str]:
    return {"type": "ephemeral", **({"ttl": "1h"} if ttl == "1h" else {})}  # five minutes is the API's default


class AnthropicClient(ChatClient):
    """``ChatClient`` over the Messages API: the same pause after a refusal that stays, the same safe failure reasons
    (Anthropic's errors carry a status and an error type, read by ``client.failure``; their message is never passed on)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 180, max_retries: int = 1,
                 thinking: str | None = None):
        self.model = model or DEFAULT_MODEL
        self.thinking = thinking  # 25.3: "adaptive", "enabled:<budget>", or None (off)
        self.base_url = base_url or BASE_URL
        self.api_key = api_key or ""
        self._client = None
        self.timeout, self.max_retries = timeout, max_retries

    @property
    def client(self):
        if self._client is None:
            from .live_guard import check

            check("Anthropic", self.base_url)  # D3: never from a test, unless the test asked
            import anthropic

            self._client = anthropic.Anthropic(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def effective_max_tokens(self, max_tokens: int | None, options: dict[str, Any] | None = None) -> int:
        """The limit sent: the caller's for the reply, plus the thinking's room on top (a budget, or a fixed headroom for
        adaptive thinking), so thinking never eats the reply. The cost estimate reads the same number."""
        options = dict(options or {})
        if self.thinking and "thinking" not in options:
            options["thinking"] = self.thinking
        limit = max_tokens or DEFAULT_MAX_TOKENS
        thinking = thinking_param(options.get("thinking"), self.model)
        if thinking:
            limit += int(thinking.get("budget_tokens") or ADAPTIVE_HEADROOM)
        return limit

    def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int | None, options: dict[str, Any]) -> dict[str, Any]:
        system, turns = to_messages(messages)
        if self.thinking and "thinking" not in options:
            options = {**options, "thinking": self.thinking}
        extra, system = _options(options, system, self.model)
        limit = self.effective_max_tokens(max_tokens, options)
        kwargs: dict[str, Any] = {"model": self.model, "messages": turns, "max_tokens": limit, **extra}
        ttl = prices.caching()
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": _cache_mark(ttl)}] if ttl else system
        if tools:
            listed = to_tools(tools)
            if ttl:
                listed[-1] = {**listed[-1], "cache_control": _cache_mark(ttl)}  # the last tool's mark caches every tool before it
            kwargs.update(tools=listed, tool_choice={"type": "auto"})
        if ttl and any(t["role"] == "assistant" for t in turns):  # a conversation under way: the automatic breakpoint follows its end
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "cache_control": _cache_mark(ttl)}
        return kwargs

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, **options: Any) -> dict[str, Any]:
        if self.model in NO_SAMPLING:  # 29.18: a model that refuses a sampling option is never sent one again
            options = {k: v for k, v in options.items() if k not in SAMPLING}
        kwargs = self._request(messages, tools, max_tokens, options)
        self._ready()
        try:
            message = self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            if any(k in kwargs for k in SAMPLING) and _refuses_sampling(exc):  # live, 2026-10-18: Sonnet 5.5 refuses `temperature`
                NO_SAMPLING.add(self.model)
                return self.complete(messages, tools, max_tokens, **options)
            if "thinking" in kwargs and _refuses_thinking(exc):  # the call matters more than its thinking: asked again without, once
                NO_THINKING_MODELS.add(str(self.model))
                return self.complete(messages, tools, max_tokens, **options)
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
            if "thinking" in kwargs and _refuses_thinking(exc):  # as complete(): asked again without thinking, once
                NO_THINKING_MODELS.add(str(self.model))
                return self.stream(messages, tools, max_tokens, on_text)
            raise self._failed(exc) from None
        return from_message(message)

    def embed(self, texts: list[str]) -> dict[str, Any]:
        raise RuntimeError("NotSupported: Anthropic serves no embeddings; evidence search uses its lexical method")
