"""Claude as the AI-first lane's model, and the capture of every request and answer.

Claude is reached through Anthropic's OpenAI-compatible endpoint (``https://api.anthropic.com/v1/``), so the
studio's own ``ChatClient`` carries the requests and nothing in the gateway changes. The key is ``ANTHROPIC_API_KEY``;
the model is ``DCLAB_TWIN_MODEL`` (default ``claude-sonnet-5-5``).

``ClaudeClient`` is what the runtime, the intern, the Home agent and the role advisors see: ``complete(messages,
tools)`` in the gateway's shape. Unlike a gateway client it keeps the whole exchange: every request's messages and
tools and every answer are appended to the lane's ``transcripts/model_calls.jsonl``, because seeing exactly what the
AI-first lane was told and said is the point of a twin run. A tool result can carry cell values (``describe_data``
shows three examples per column), so tool messages are written through the trace's own scrub (``agents.traces.scrub``,
the keys that carry cells removed) unless the run asks for ``capture_cells``; the model's own words are kept whole.
The holdout score is in the transcript when the agent asked ``get_results`` for the final stage, as it is in the
session: ``summary.json`` says how many times the holdout was scored. It still writes the usage row every other
client writes (``model_requests.jsonl``, through ``Gateway.record``) and stops at the run's caps: a number of
requests, and euros when the model has a price (``DCLAB_PRICES_FILE``; without one the euro cap cannot be applied
and the manifest says so). A cap is checked before a request, whose cost is known only after it, so the last request
can pass the euro cap by its own cost.

``TwinGateway`` hands that client to every purpose, so the stage notes, the leakage reviewer and the seven roles of
dclab_core all run on Claude in the AI-first lane; ``NoModels`` hands none to anyone, so the workflow lane runs the
rules alone.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

ANTHROPIC_OPENAI_URL = "https://api.anthropic.com/v1/"
DEFAULT_MODEL = "claude-sonnet-5-5"
TRANSCRIPT = "model_calls.jsonl"


def model_name() -> str:
    return os.environ.get("DCLAB_TWIN_MODEL", "").strip() or DEFAULT_MODEL


def api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def transport(model: str | None = None, key: str | None = None, timeout: float = 180) -> Any:
    """The real transport: the gateway's own client on Anthropic's OpenAI-compatible endpoint."""
    from ..models import endpoint_client

    key = key or api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return endpoint_client(ANTHROPIC_OPENAI_URL, model or model_name(), key, timeout=timeout)


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
        self.model = getattr(inner, "model", model_name())
        self.base_url = getattr(inner, "base_url", ANTHROPIC_OPENAI_URL)
        # what the intern's _Model and the role advisors read and set
        self.run_limit_eur: float | None = cap_eur
        self.spent_eur = 0.0
        self.reserved_eur = 0.0
        # one budget for every client of the lane: requests and euros are counted together
        self.shared = shared if shared is not None else {"requests": 0, "eur": 0.0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0,
                                                         "max_requests": max_requests, "cap_eur": cap_eur, "lock": threading.Lock()}

    def _refusal(self) -> str | None:
        s = self.shared
        if s["requests"] >= s["max_requests"]:
            return f"the twin run's request cap is reached ({s['max_requests']} requests)"
        cap = s.get("cap_eur")
        if cap is not None and s["eur"] >= cap:
            return f"the twin run's euro cap is reached (€{s['eur']:.2f} of €{cap:.2f})"
        if self.run_limit_eur is not None and self.spent_eur >= self.run_limit_eur:
            return f"this client's euro limit is reached (€{self.spent_eur:.2f} of €{self.run_limit_eur:.2f})"
        return None

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
        with self.shared["lock"]:
            refusal = self._refusal()
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
            seconds = time.monotonic() - started
            row.update(outcome=f"{type(exc).__name__}: {str(exc)[:300]}", answer=None, seconds=round(seconds, 3))
            self.transcript.write(row)
            self._usage_row(messages, None, seconds, 1, str(exc).split(": ", 1)[-1][:200])
            raise
        seconds = time.monotonic() - started
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

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None) -> dict[str, Any]:
        out = self.complete(messages, tools, max_tokens)
        if on_text and out.get("content"):
            on_text(out["content"])
        return out

    def _usage_row(self, messages, out, seconds, attempts, outcome) -> float | None:
        if self.usage is None:
            from ..models import prices

            usage = (out or {}).get("usage") or {}
            eur, _basis = prices.cost(self.model, False, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
            return eur
        try:
            return self.usage.record(self.purpose, self.model, (out or {}).get("usage"), seconds, attempts, outcome, self.project_id,
                                     self.draft_id, None, base_url=self.base_url, tier="strong")
        except Exception:  # noqa: BLE001 — a usage row that cannot be written never loses the answer
            return None

    def output(self, passed: bool, reason: str = "") -> None:
        """A caller's verdict on the last answer (the gateway's ``Bound.output``): kept in the transcript."""
        self.transcript.write({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lane": self.lane, "purpose": self.purpose, "model": self.model,
                               "check": {"passed": bool(passed), "reason": reason[:300]}})


class TwinGateway:
    """The AI-first lane's gateway: Claude for every purpose, one shared budget, every exchange in the transcript."""

    def __init__(self, inner: Any, transcript: Transcript, usage: Any = None, *, max_requests: int = 200, cap_eur: float | None = None,
                 capture_cells: bool = False):
        self.inner, self.transcript, self.usage, self.capture_cells = inner, transcript, usage, capture_cells
        self.shared = {"requests": 0, "eur": 0.0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0,
                       "max_requests": max_requests, "cap_eur": cap_eur, "lock": threading.Lock()}
        self.shadows = None

    def client(self, purpose: str, *, project_id: str | None = None, draft_id: str | None = None, model: str | None = None) -> ClaudeClient:
        return ClaudeClient(self.inner, self.transcript, self.usage, purpose, project_id=project_id, draft_id=draft_id, shared=self.shared,
                            capture_cells=self.capture_cells)

    def available(self, purpose: str) -> bool:
        return True

    def refusal(self, project_id: str | None = None, run: Any = None) -> str | None:
        probe = ClaudeClient(self.inner, self.transcript, None, "check", shared=self.shared)
        if run is not None:
            probe.run_limit_eur, probe.spent_eur = getattr(run, "run_limit_eur", None), float(getattr(run, "spent_eur", 0.0) or 0.0)
        return probe._refusal()

    def totals(self) -> dict[str, Any]:
        return {k: v for k, v in self.shared.items() if k != "lock"}


class NoModels:
    """The workflow lane's gateway: no purpose gets a model, so the current system runs its rules alone."""

    def client(self, *args: Any, **kwargs: Any) -> None:
        return None

    def available(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def refusal(self, *args: Any, **kwargs: Any) -> str | None:
        return None

    def totals(self) -> dict[str, Any]:
        return {"requests": 0, "eur": 0.0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0}
