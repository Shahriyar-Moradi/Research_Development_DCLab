"""Claude as the AI-first lane's model, and the capture of every request and answer (packages A6.6 and 25.2).

Claude is reached through the gateway package's native Messages transport (``models.anthropic_client``): the
OpenAI-style messages and tools every caller uses are translated both ways, and Claude's thinking (25.3) comes back
beside each reply. The key is ``ANTHROPIC_API_KEY``; the model is ``DCLAB_TWIN_MODEL`` (default ``claude-sonnet-5-5``).

``ClaudeClient`` is what the runtime, the intern, the Home agent and the role advisors see: ``complete(messages,
tools)`` in the gateway's shape. Unlike a gateway client it keeps the whole exchange: every request's messages and
tools and every answer are appended to the lane's ``transcripts/model_calls.jsonl``, because seeing exactly what the
AI-first lane was told and said is the point of a twin run. A tool result can carry cell values (``describe_data``
shows three examples per column), so tool messages are written through the trace's own scrub (``agents.traces.scrub``,
the keys that carry cells removed) unless the run asks for ``capture_cells``; the model's own words are kept whole.
It still writes the usage row every other client writes (``model_requests.jsonl``, through ``Gateway.record``) and
stops at the run's caps: a number of requests, and euros (the price table's units). A cap is checked before a
request, whose cost is known only after it, so the last request can pass the euro cap by its own cost.

``TwinGateway`` hands that client to every purpose, so the stage notes, the leakage reviewer and every role of
dclab_core run on Claude in the AI-first lane; ``NoModels`` hands none to anyone, so the workflow lane runs the
rules alone. On main (25.2) this replaces the A6.6 branch's OpenAI-compatible endpoint and ``models.endpoint_client``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"
TRANSCRIPT = "model_calls.jsonl"
THINKING = "thinking.jsonl"  # 25.3: Claude's thinking beside each reply, in the lane's transcripts/


def thinking_setting() -> str | None:
    """``DCLAB_TWIN_THINKING``: ``adaptive`` (the default: as much of Claude's reasoning as the API returns),
    ``enabled:<budget tokens>``, or ``off``."""
    value = os.environ.get("DCLAB_TWIN_THINKING", "adaptive").strip()
    return None if value.lower() in ("", "0", "off", "none", "false") else value


def model_name() -> str:
    return os.environ.get("DCLAB_TWIN_MODEL", "").strip() or DEFAULT_MODEL


def api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def transport(model: str | None = None, key: str | None = None, timeout: float = 180) -> Any:
    """The real transport: the gateway package's native Messages client."""
    from ..models.anthropic_client import AnthropicClient

    key = key or api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return AnthropicClient(model or model_name(), api_key=key, timeout=timeout, thinking=thinking_setting())


class Transcript:
    """Append-only: one line per request, the messages and tools as sent and the answer as received."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.n = 0

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.n += 1
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"n": self.n, **row}, ensure_ascii=False, default=str) + "\n")


class ClaudeClient:
    """A client in the gateway's shape whose every exchange is kept. ``inner`` is the transport (the real one, or a
    scripted one in tests); ``usage`` is the lane's gateway, which writes the usage rows; ``purpose`` names what the
    client is for (the intern, a role, the stage notes), so the transcript and the usage rows say who asked."""

    def __init__(self, inner: Any, transcript: Transcript, usage: Any = None, purpose: str = "intern", *, lane: str = "ai_first",
                 project_id: str | None = None, draft_id: str | None = None, max_requests: int = 200, cap_eur: float | None = None,
                 shared: dict[str, Any] | None = None, capture_cells: bool = False):
        self.inner, self.transcript, self.usage, self.purpose, self.lane = inner, transcript, usage, purpose, lane
        self.capture_cells = capture_cells
        self.project_id, self.draft_id = project_id, draft_id
        self.observer: Any = None  # A6.7: the design monitor on the agent's client, called before each request
        self.model = getattr(inner, "model", model_name())
        self.base_url = getattr(inner, "base_url", "https://api.anthropic.com")
        # what the intern's _Model and the role advisors read and set
        self.run_limit_eur: float | None = cap_eur
        self.spent_eur = 0.0
        self.reserved_eur = 0.0
        # one budget for every client of the lane: requests and euros are counted together
        self.shared = shared if shared is not None else {"requests": 0, "eur": 0.0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0,
                                                         "max_requests": max_requests, "cap_eur": cap_eur, "lock": threading.Lock()}

    def _refusal(self, estimate: float = 0.0) -> str | None:
        """Why the next request may not be sent: a cap reached, or one this request could pass by its upper-bound cost
        (``estimate``, as ``Gateway.reserve`` counts it), or the workspace's monthly cap with this run's spend on top."""
        s = self.shared
        if s["requests"] >= s["max_requests"]:
            return f"the twin run's request cap is reached ({s['max_requests']} requests)"
        cap = s.get("cap_eur")
        if cap is not None and s["eur"] + estimate > cap:
            return f"the twin run's euro cap would be passed (€{s['eur']:.2f} spent, up to €{estimate:.2f} more, cap €{cap:.2f})"
        if self.run_limit_eur is not None and self.spent_eur + estimate > self.run_limit_eur:
            return f"this client's euro limit would be passed (€{self.spent_eur:.2f} of €{self.run_limit_eur:.2f})"
        monthly = s.get("workspace_cap")
        if monthly is not None and float(s.get("month_spent") or 0.0) + s["eur"] + estimate > monthly:
            return f"the workspace's monthly euro cap would be passed (€{monthly:.2f}, DCLAB_WORKSPACE_MONTHLY_EUR)"
        return None

    def _estimate(self, messages: list[dict[str, Any]], max_tokens: int | None, options: dict[str, Any] | None = None) -> float:
        from ..models import prices

        chars = sum(len(json.dumps(m.get("content"), default=str)) for m in messages)
        limit = self.inner.effective_max_tokens(max_tokens, options) if hasattr(self.inner, "effective_max_tokens") else max_tokens
        value = prices.estimate(self.model, False, chars, limit)
        return float(value or 0.0)

    def _kept(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages as the transcript keeps them: a tool result without the keys that carry cells, unless asked."""
        if self.capture_cells:
            return messages
        from ..agents.traces import scrub

        out = []
        for m in messages:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                try:
                    out.append({**m, "content": json.dumps(scrub(json.loads(m["content"])), ensure_ascii=False)})
                    continue
                except (ValueError, TypeError):
                    pass
            out.append(m)
        return out

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800,
                 **options: Any) -> dict[str, Any]:
        if self.observer is not None:  # reads the steps that finished, may revise the design and hand it to the agent
            try:
                self.observer.before(messages)
            except Exception as exc:  # noqa: BLE001 — monitoring never stops the agent
                self.transcript.write({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lane": self.lane, "purpose": "monitor",
                                       "outcome": f"monitor failed: {type(exc).__name__}: {str(exc)[:300]}"})
        thinking_on = bool(getattr(self.inner, "thinking", None)) and not self.shared.get("thinking_refused")
        if getattr(self.inner, "thinking", None) and not thinking_on:
            options = {**options, "thinking": "off"}
        estimate = self._estimate(messages, max_tokens, options)
        with self.shared["lock"]:
            refusal = self._refusal(estimate)
            if refusal is None:
                self.shared["requests"] += 1
        row: dict[str, Any] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lane": self.lane, "purpose": self.purpose, "model": self.model,
                               "project_id": self.project_id, "draft_id": self.draft_id, "messages": self._kept(messages), "tools": tools,
                               "max_tokens": max_tokens, "options": {k: v for k, v in options.items() if k != "on_text"}}
        if refusal:
            row.update(outcome="refused: " + refusal, answer=None, seconds=0.0)
            self.transcript.write(row)
            self._usage_row(messages, None, 0.0, 0, "refused: " + refusal)
            error = RuntimeError("BudgetExceeded: " + refusal)
            error.transient = False  # type: ignore[attr-defined]
            raise error
        started = time.monotonic()
        try:
            out = self.inner.complete(messages, tools, max_tokens, **options)
        except Exception as exc:  # noqa: BLE001 — the failure is captured like an answer, then raised as the gateway would
            from ..models.client import THINKING_REFUSED

            if thinking_on and THINKING_REFUSED in str(exc):
                # the model or the conversation would not take thinking: once without it, and never again in this run
                self.shared["thinking_refused"] = THINKING_REFUSED
                self.transcript.write({**{k: row[k] for k in ("at", "lane", "purpose", "model")}, "outcome": "thinking refused, asked again without it"})
                self._usage_row(messages, None, time.monotonic() - started, 1, THINKING_REFUSED)
                return self.complete(messages, tools, max_tokens, **options)
            seconds = time.monotonic() - started
            row.update(outcome=f"{type(exc).__name__}: {str(exc)[:300]}", answer=None, seconds=round(seconds, 3))
            self.transcript.write(row)
            self._usage_row(messages, None, seconds, 1, str(exc).split(": ", 1)[-1][:200])
            raise
        seconds = time.monotonic() - started
        if out.get("thinking") or out.get("redacted_thinking"):
            self.thoughts().write({"at": row["at"], "lane": self.lane, "purpose": self.purpose, "model": self.model, "request": self.transcript.n + 1,
                                   "thinking": [t.get("text") for t in out.get("thinking") or []], "redacted": int(out.get("redacted_thinking") or 0),
                                   "said": out.get("content"), "tools": [c.get("name") for c in out.get("tool_calls") or []]})
        usage = out.get("usage") or {}
        eur = self._usage_row(messages, out, seconds, 1, "ok")
        with self.shared["lock"]:
            self.shared["input_tokens"] += int(usage.get("input_tokens") or 0)
            self.shared["output_tokens"] += int(usage.get("output_tokens") or 0)
            if eur is None:
                self.shared["unpriced"] += 1
            else:
                self.shared["eur"] += eur
                self.spent_eur += eur
        row.update(outcome="ok", answer={k: v for k, v in out.items() if k != "assistant_message"}, seconds=round(seconds, 3), cost_eur=eur)
        self.transcript.write(row)
        return out

    def thoughts(self) -> Transcript:
        """The lane's ``transcripts/thinking.jsonl``: one row per reply that carried Claude's thinking."""
        if "thoughts" not in self.shared:
            with self.shared["lock"]:
                self.shared.setdefault("thoughts", Transcript(self.transcript.path.parent / THINKING))
        return self.shared["thoughts"]

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None) -> dict[str, Any]:
        out = self.complete(messages, tools, max_tokens)
        if on_text and out.get("content"):
            on_text(out["content"])
        return out

    def _usage_row(self, messages, out, seconds, attempts, outcome) -> float | None:
        usage = (out or {}).get("usage") or {}
        if self.usage is None:
            from ..models import prices

            eur, _basis = prices.cost(self.model, False, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
        else:
            try:
                eur = self.usage.record(self.purpose, self.model, (out or {}).get("usage"), seconds, attempts, outcome, self.project_id,
                                        self.draft_id, None, base_url=self.base_url, tier="strong")
            except Exception:  # noqa: BLE001 — a usage row that cannot be written never loses the answer
                eur = None
        try:  # the lane's session record has every request, as the gateway's own clients write it (19.2)
            from .. import experience
            from ..experience import hooks

            hooks.from_model_request({"id": experience.new_model_request(), "purpose": self.purpose, "tier": "twin", "model": self.model,
                                      "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "cost_eur": eur,
                                      "seconds": round(seconds, 3), "outcome": outcome, "project_id": self.project_id, "draft_id": self.draft_id},
                                     messages, (out or {}).get("content"))
        except Exception:  # noqa: BLE001
            pass
        return eur

    def output(self, passed: bool, reason: str = "") -> None:
        """A caller's verdict on the last answer (the gateway's ``Bound.output``): kept in the transcript."""
        self.transcript.write({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lane": self.lane, "purpose": self.purpose, "model": self.model,
                               "check": {"passed": bool(passed), "reason": reason[:300]}})


class TwinGateway:
    """The AI-first lane's gateway: Claude for every purpose, one shared budget, every exchange in the transcript."""

    def __init__(self, inner: Any, transcript: Transcript, usage: Any = None, *, max_requests: int = 200, cap_eur: float | None = None,
                 capture_cells: bool = False, month_spent: float = 0.0):
        self.inner, self.transcript, self.usage, self.capture_cells = inner, transcript, usage, capture_cells
        from ..models.gateway import workspace_cap

        self.shared = {"requests": 0, "eur": 0.0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0,
                       "max_requests": max_requests, "cap_eur": cap_eur, "workspace_cap": workspace_cap(), "month_spent": month_spent,
                       "lock": threading.Lock()}
        self.shadows = None
        self.routing = None

    def client(self, purpose: str, *, project_id: str | None = None, draft_id: str | None = None, model: str | None = None) -> ClaudeClient | None:
        from ..models.gateway import _PRIVATE

        if _PRIVATE.get():  # a private scope (a replay, an acceptance) asks no model, whoever's gateway it is
            return None
        return ClaudeClient(self.inner, self.transcript, self.usage, purpose, project_id=project_id, draft_id=draft_id, shared=self.shared,
                            capture_cells=self.capture_cells)

    def available(self, purpose: str) -> bool:
        return True

    def refusal(self, project_id: str | None = None, run: Any = None, purpose: str | None = None) -> str | None:
        probe = ClaudeClient(self.inner, self.transcript, None, purpose or "check", shared=self.shared)
        if run is not None:
            probe.run_limit_eur, probe.spent_eur = getattr(run, "run_limit_eur", None), float(getattr(run, "spent_eur", 0.0) or 0.0)
        return probe._refusal()

    def totals(self) -> dict[str, Any]:
        out = {k: v for k, v in self.shared.items() if k not in ("lock", "thoughts")}
        out["thinking_replies"] = self.shared["thoughts"].n if "thoughts" in self.shared else 0
        return out


class NoModels:
    """The workflow lane's gateway: no purpose gets a model, so the current system runs its rules alone."""

    routing = None

    def client(self, *args: Any, **kwargs: Any) -> None:
        return None

    def available(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def refusal(self, *args: Any, **kwargs: Any) -> str | None:
        return None

    def totals(self) -> dict[str, Any]:
        return {"requests": 0, "eur": 0.0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0}
