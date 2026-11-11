"""An OpenAI-compatible chat client with tool calling (the transport under ``dclab_rnd.models.gateway``).

Works against OpenAI, the Hugging Face router (``OPENAI_BASE_URL=https://router.huggingface.co/v1``
with an ``hf_…`` token, the same setup Hugging Face's ML Intern uses), a local Ollama
(``http://127.0.0.1:11434/v1``) or any other endpoint that speaks the chat-completions API.
The key is read from the environment or ``.env``; it is never typed into the UI.
"""

from __future__ import annotations

import json
import re
import os
import time
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "gpt-5.6-terra"
PAUSE_SECONDS = 600  # after a refusal that will not go away by itself (no credit, a refused key, an unknown model)
# the pause is shared by every app instance and worker (models/pause.py: a row with an expiry on PostgreSQL)


THINKING_REFUSED = "the model would not take extended thinking"
# the reasoning a local model writes before its answer: "<think>...</think>" at the start, or, when its chat template put the
# opening tag in the prompt, the text before the first "</think>"
_THINK = re.compile(r"\s*(?:<think>)?(.*?)</think>", re.S)
REASONING_FIELDS = ("reasoning_content", "reasoning")  # the names OpenAI-compatible servers give a reply's reasoning (D2)


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
    message = str(error.get("message") or "").lower() if isinstance(error, dict) else ""
    if status == 400 and "credit balance" in message:  # Anthropic says it in a 400; the words are tested, never passed on
        return "the model account has no credit left", True
    if status == 400 and "thinking" in message:  # 25.3: the model or the conversation would not take extended thinking (tested, never passed on)
        return THINKING_REFUSED, False
    if status in (401, 403):
        return "the provider refused the key", True
    if status == 404 or "model_not_found" in code:
        return "the model name is not available to this key", True
    if status == 429:
        return "the provider is limiting requests for now", False
    if (isinstance(status, int) and status >= 500) or "overloaded" in code or "api_error" in code:  # an overload inside a stream comes with 200
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
    model = os.environ.get("DCLAB_INTERN_MODEL") or os.environ.get("OPENAI_MODEL") or (DEFAULT_MODEL if urlparse(base_url).hostname == "api.openai.com" else "")
    key = os.environ.get("OPENAI_API_KEY") or ("ollama" if "11434" in base_url else "")
    try:
        import openai  # noqa: F401

        sdk = True
    except ImportError:
        sdk = False
    return {"available": bool(key) and sdk and bool(model), "model": model, "endpoint": urlparse(base_url).netloc or base_url,
            "sdk_installed": sdk, "key_configured": bool(key)}


# Models whose chat-completions endpoint takes function tools only with reasoning_effort "none" (OpenAI's gpt-6-luna
# refuses tools with its default reasoning: "use /v1/responses or set reasoning_effort to 'none'", found in 14.2).
# Learned from the refusal itself, once per process, so no model name is hard-coded and a model that changes is followed.
TOOLS_WITHOUT_REASONING: set[str] = set()


def _wants_no_reasoning(exc: BaseException) -> bool:
    text = str(exc)
    return "reasoning_effort" in text and "'none'" in text and "tool" in text.lower()


# 2026-11-11: OpenRouter's reasoning effort, from DCLAB_THINKING_EFFORT (the setting Claude's requests read). OpenRouter takes
# minimal to high; Claude's xhigh and max become high. Unset (or a word it does not know) sends {"enabled": true}, the model's default.
OPENROUTER_EFFORTS = ("minimal", "low", "medium", "high")
_EFFORT_MAP = {"xhigh": "high", "max": "high"}
# models that refused an effort in this process: sent {"enabled": true} from then on (learned from the refusal, no name in code)
EFFORT_REFUSED: set[str] = set()


def reasoning_allowance() -> int:
    """The tokens added to a reply's limit for an OpenRouter model's reasoning (``DCLAB_REASONING_TOKENS``, default 2048)."""
    raw = os.environ.get("DCLAB_REASONING_TOKENS", "").strip()
    try:
        return max(0, int(raw)) if raw else 2048
    except ValueError:
        return 2048


def openrouter_effort() -> str | None:
    raw = os.environ.get("DCLAB_THINKING_EFFORT", "").strip().lower()
    raw = _EFFORT_MAP.get(raw, raw)
    return raw if raw in OPENROUTER_EFFORTS else None


def _refuses_effort(exc: BaseException, kwargs: dict[str, Any]) -> bool:
    """A 400/422 about the reasoning effort, on a request that sent one. Only the status and the words are read: the words
    must name the effort ("reasoning" alone is not enough: a refused reasoning_details passback is another error)."""
    sent = (kwargs.get("extra_body") or {}).get("reasoning")
    if not (isinstance(sent, dict) and "effort" in sent):
        return False
    if getattr(exc, "status_code", None) not in (400, 422):
        return False
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    text = (str(exc) + " " + (str(error.get("message") or "") if isinstance(error, dict) else "")).lower()
    return "effort" in text


def _reasoning_sent(kwargs: dict[str, Any]) -> dict[str, Any]:
    sent = (kwargs.get("extra_body") or {}).get("reasoning")
    return {"reasoning_request": dict(sent)} if isinstance(sent, dict) else {}


class ChatClient:
    """One ``complete`` call = one chat-completions request. A failure is raised as ``RuntimeError("<Type>: <reason>")``
    with a reason safe to show (see ``failure``); after a refusal that will not go away, requests pause for a while."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 180, max_retries: int = 1,
                 reasoning_field: str | None = None, chat_template_kwargs: dict[str, Any] | None = None, tool_arguments: str = "json",
                 max_tokens_floor: int | None = None):
        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = base_url or os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("ollama" if "11434" in self.base_url else "")
        self._client = None
        self.timeout, self.max_retries = timeout, max_retries  # the gateway passes 0: it retries itself, and counts each attempt
        # D2 (2026-10-23), the endpoint's options from models.settings.options: the reply field that carries the model's
        # reasoning (read from every reply, sent back with every earlier assistant turn, so a local model keeps its plan
        # from turn to turn), the chat-template setting sent with every request, and how tool arguments go back
        self.reasoning_field = reasoning_field or None
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tool_arguments = tool_arguments
        # 2026-10-29: a local thinking model writes its whole reasoning before the answer and can spend a hosted model's
        # limit on thinking alone (Qwen3.5-4B: 1,800 tokens of reasoning, an empty reply); the tier's floor raises every limit
        self.max_tokens_floor = int(max_tokens_floor) if max_tokens_floor else None

    def _extra(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The endpoint's chat-template setting on every request, in ``extra_body`` beside whatever the caller put there.
        On OpenRouter (``openrouter.ai``), ``"reasoning": {"effort": …}`` (``openrouter_effort``) is sent, or ``{"enabled": True}``
        with no effort set or after the model refused one, unless turned off (OPENROUTER_REASONING, DCLAB_THINKING, the purpose)."""
        thinking = kwargs.pop("thinking", getattr(self, "thinking", None))
        extra = dict(kwargs.get("extra_body") or {})
        if self.chat_template_kwargs:
            extra["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if urlparse(str(getattr(self, "base_url", "") or "")).hostname == "openrouter.ai":
            env_off = (
                os.environ.get("OPENROUTER_REASONING", "1").strip().lower() in ("0", "off", "none", "false")
                or os.environ.get("DCLAB_THINKING", "").strip().lower() in ("0", "off", "none", "false")
            )
            think_off = str(thinking or "").strip().lower() in ("0", "off", "false")
            if not think_off and not env_off and "reasoning" not in extra:
                effort = openrouter_effort()
                extra["reasoning"] = {"effort": effort} if effort and self.model not in EFFORT_REFUSED else {"enabled": True}
            if extra.get("reasoning") and extra["reasoning"] != {"enabled": False}:
                # 2026-11-11: OpenRouter counts the reasoning inside the reply's limit, Claude does not: a purpose that
                # asked for 300 tokens got an empty answer after 300 tokens of reasoning (604 of 1,800 in SEX-008). The
                # reasoning gets its own allowance on top, so the answer keeps the room the purpose asked for.
                for key in ("max_tokens", "max_completion_tokens"):
                    if isinstance(kwargs.get(key), int):
                        kwargs[key] += reasoning_allowance()
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def _reasoning_details(self, message: Any) -> list[Any] | None:
        """OpenRouter's ``reasoning_details`` array on an assistant message or delta, normalized to plain dicts/items so it
        can be passed back unmodified in the next turn's ``assistant_message``."""
        raw = getattr(message, "reasoning_details", None)
        if raw is None and isinstance(message, dict):
            raw = message.get("reasoning_details")
        if not raw or not isinstance(raw, list):
            return None
        return [d.model_dump() if hasattr(d, "model_dump") else (dict(d) if isinstance(d, dict) else d) for d in raw]

    def _reasoning(self, message: Any) -> str:
        """The reasoning a reply (or a stream's delta) carries: under this endpoint's field, or, with none configured, under
        the first known field it has (the owner, 2026-10-28: a local model's reasoning is kept for training, not dropped
        for want of a setting; it is sent back to the model only under a configured field)."""
        for field in ((self.reasoning_field,) if self.reasoning_field else REASONING_FIELDS):
            value = getattr(message, field, None)
            if value:
                return str(value)
        details = self._reasoning_details(message)
        if details:
            text = "".join(str((d.get("text") if isinstance(d, dict) else getattr(d, "text", None)) or "") for d in details)
            if text:
                return text
        return ""

    def _reply(self, content: str, calls: list[dict[str, Any]], finish: Any, usage: Any, reasoning: str, note: dict[str, Any],
               reasoning_details: list[Any] | None = None) -> dict[str, Any]:
        """What ``complete`` and ``stream`` return: the reply parsed (``tool_calls`` with dict arguments; ``reasoning`` and
        ``thinking`` when the model reasoned, in the shape the exchange log keeps) and ``assistant_message`` as the next
        request sends it back: the reasoning under the endpoint's field (and ``reasoning_details`` unmodified when returned
        by OpenRouter), the tool arguments as its template wants them."""
        if not reasoning and "</think>" in (content or ""):  # a model that writes its reasoning inline: kept apart, out of the reply
            opened = content.lstrip().startswith("<think>")
            # only at the reply's start ("<think>...</think>"), or a lone closing tag with no opening one anywhere (the template
            # opened it in the prompt); an answer that mentions the tags later (code, an explanation) is kept whole
            match = _THINK.match(content) if opened or "<think>" not in content else None
            if match:
                reasoning, content = match.group(1).strip(), content[match.end():].strip()
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning and self.reasoning_field:
            message[self.reasoning_field] = reasoning
        if reasoning_details:
            message["reasoning_details"] = reasoning_details
        if calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"] if self.tool_arguments == "object" else json.dumps(c["arguments"])}}
                                     for c in calls]
        out = {"content": content, "tool_calls": calls, "finish_reason": finish,
               "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0, "output_tokens": getattr(usage, "completion_tokens", 0) or 0},
               "assistant_message": message, **note}
        if reasoning:
            out["reasoning"] = reasoning
            out["thinking"] = [{"text": reasoning}]
        if reasoning_details:
            out["reasoning_details"] = reasoning_details
        return out

    def _fields_seen(self, message: Any) -> dict[str, Any]:
        """With no field configured: which known reasoning fields the reply carried anyway, so the probe (and a person
        reading the usage) can name the right one for the tier. Read, never kept or sent back."""
        if self.reasoning_field:
            return {}
        seen = [f for f in REASONING_FIELDS if getattr(message, f, None)]
        return {"reasoning_fields_seen": seen} if seen else {}

    def effective_max_tokens(self, max_tokens: int | None, options: dict[str, Any] | None = None) -> int | None:
        """The actual reply/reasoning limit, including an endpoint's configured minimum."""
        return max(max_tokens, self.max_tokens_floor or 0) if max_tokens is not None else None

    def _limit(self, max_tokens: int | None) -> dict[str, Any]:
        """OpenAI's newer models take ``max_completion_tokens`` (they refuse ``max_tokens``); other OpenAI-compatible
        servers (Ollama, the Hugging Face router, vLLM) take ``max_tokens``. None sends no limit."""
        if max_tokens is None:
            return {}
        if self.max_tokens_floor:
            max_tokens = max(int(max_tokens), self.max_tokens_floor)
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

    def _create(self, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """One chat-completions request, and what the result should say about how it was sent. With tools, a model
        that takes them only without reasoning gets reasoning_effort "none" (learned from its refusal, then sent from
        the start); nothing else changes. The refused request is counted as an attempt, and the answer says it was
        written without reasoning, so a usage row or a live result never hides either."""
        if kwargs.get("tools") and self.model in TOOLS_WITHOUT_REASONING and "reasoning_effort" not in kwargs:
            kwargs = {**kwargs, "reasoning_effort": "none"}
        note = {"reasoning_effort": kwargs["reasoning_effort"]} if "reasoning_effort" in kwargs else {}
        try:
            return self.client.chat.completions.create(**kwargs), {**note, **_reasoning_sent(kwargs)}
        except Exception as exc:  # noqa: BLE001 — retried only for these two refusals; anything else goes on as it was
            if kwargs.get("tools") and "reasoning_effort" not in kwargs and _wants_no_reasoning(exc):
                TOOLS_WITHOUT_REASONING.add(self.model)
                return (self.client.chat.completions.create(**{**kwargs, "reasoning_effort": "none"}),
                        {"reasoning_effort": "none", "refused_attempts": 1, **_reasoning_sent(kwargs)})
            if _refuses_effort(exc, kwargs):  # 2026-11-11: once, with the model's default reasoning; said in the reply, never hidden
                refused = kwargs["extra_body"]["reasoning"]["effort"]
                EFFORT_REFUSED.add(self.model)
                retry = {**kwargs, "extra_body": {**kwargs["extra_body"], "reasoning": {"enabled": True}}}
                return self.client.chat.completions.create(**retry), {**note, "refused_attempts": 1, "reasoning_effort_refused": refused,
                                                                      **_reasoning_sent(retry)}
            raise

    @property
    def client(self):
        if self._client is None:
            from .live_guard import check

            check("OpenAI-compatible", self.base_url)  # D3: never from a test, unless the test asked
            import openai

            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None,
               **options: Any) -> dict[str, Any]:
        """Like ``complete`` (same return value), but ``on_text(delta)`` hears the reply as it is written.

        Tool-call arguments arrive in pieces and are joined here; a call whose arguments are not valid JSON keeps the
        raw text under ``_raw``, as in ``complete``. The reasoning arrives in pieces too and is joined.
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, **self._limit(max_tokens), **options, "stream": True,
                                  "stream_options": {"include_usage": True}}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        self._extra(kwargs)
        text: list[str] = []
        thought: list[str] = []
        details: list[Any] = []
        pieces: dict[int, dict[str, Any]] = {}
        finish, usage = None, None
        self._ready()
        try:
            chunks, note = self._create(kwargs)
            for chunk in chunks:
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
                piece = self._reasoning(delta)
                if piece:
                    thought.append(piece)
                delta_details = self._reasoning_details(delta)
                if delta_details:
                    details.extend(delta_details)
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
        return self._reply("".join(text), calls, finish, usage, "".join(thought), note, reasoning_details=details or None)

    def embed(self, texts: list[str]) -> dict[str, Any]:
        """Embeddings of the texts, in order: {"vectors": [[...]], "usage": {"input_tokens", "output_tokens": 0}}."""
        self._ready()
        try:
            response = self.client.embeddings.create(model=self.model, input=list(texts))
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            raise self._failed(exc) from None
        usage = getattr(response, "usage", None)
        rows = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        return {"vectors": [list(d.embedding) for d in rows],
                "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0, "output_tokens": 0}}

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, **options: Any) -> dict[str, Any]:
        """``options`` are passed to the provider as they are (for example ``response_format`` or ``temperature``)."""
        single_attempt = options.pop("_single_attempt", False)
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, **self._limit(max_tokens), **options}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        self._extra(kwargs)
        self._ready()
        try:
            if single_attempt:
                response, note = self.client.chat.completions.create(**kwargs), _reasoning_sent(kwargs)
            else:
                response, note = self._create(kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors can carry headers; keep only the type
            if single_attempt and _refuses_effort(exc, kwargs):  # one attempt only: not retried, but the next request sends enabled
                EFFORT_REFUSED.add(self.model)
            raise self._failed(exc) from None
        choice = response.choices[0]
        calls = []
        for call in choice.message.tool_calls or []:
            arguments = call.function.arguments
            if arguments is None:  # a call without arguments
                arguments = {}
            elif isinstance(arguments, str):  # the wire format; a server set to objects answers with a dict already
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": call.function.arguments}
            calls.append({"id": call.id, "name": call.function.name, "arguments": arguments if isinstance(arguments, dict) else {"_raw": arguments}})
        return {**self._reply(choice.message.content or "", calls, choice.finish_reason, getattr(response, "usage", None),
                              self._reasoning(choice.message), note, reasoning_details=self._reasoning_details(choice.message)),
                **self._fields_seen(choice.message)}
