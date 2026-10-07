"""The model gateway: every model request DCLab makes goes through here (package A1.1).

    gateway = Gateway(open_usage(home))
    client = gateway.client("home_agent", draft_id=...)      # None when no model serves that purpose
    client.complete(messages, tools) / client.stream(messages, tools, on_text=...)

A client has the same ``complete``/``stream`` shape as ``models.client.ChatClient``, so callers do not change. The
gateway adds what no caller should do by itself: the purpose decides the tier and the timeout; a rate limit, a
provider error or an unreachable endpoint is retried with backoff (a stream only before its first word reached the
user); and every request is written to the usage log with its tokens, time, attempts and outcome.

Money (A1.2): each request's cost comes from dclab_rnd/models/prices.py (never guessed; local endpoints are free), and a
request that would pass a run, project or workspace cap is refused before it is sent.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from typing import Any, Callable

from . import prices, settings
from .client import ChatClient
from .usage import UsageLog, month_start, now

def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:  # a bad value: the server refuses to start and says why (Settings.problems); a script keeps the default
        return default


RETRIES = int(_number("DCLAB_MODEL_RETRIES", 2))  # attempts after the first, for transient failures only
BACKOFF = _number("DCLAB_MODEL_BACKOFF", 1.5)  # seconds, doubled after each attempt


def _transport(t: dict[str, Any], p: settings.Purpose) -> ChatClient:
    return ChatClient(model=t["model"], base_url=t["base_url"], api_key=t["api_key"], timeout=p.timeout, max_retries=0)


def user_cap() -> tuple[float | None, str | None]:
    """The signed-in person's monthly euro limit and id (package 10.6); (None, None) without accounts or a person."""
    from ..accounts.principal import current

    who = current()
    if who is None or not who.user_id:
        return None, None
    from .. import limits

    if not limits.on():
        return None, None
    return limits.limits()["model_eur"]["user"], who.user_id


def workspace_cap() -> float | None:
    """The workspace's monthly euro cap, DCLAB_WORKSPACE_MONTHLY_EUR (none when unset)."""
    raw = os.environ.get("DCLAB_WORKSPACE_MONTHLY_EUR", "").strip()
    return float(raw) if raw else None


class Gateway:
    def __init__(self, usage: UsageLog | None = None, transport: Callable[[dict[str, Any], settings.Purpose], Any] = _transport,
                 sleep: Callable[[float], None] = time.sleep, project_cap: Callable[[str], float | None] | None = None,
                 routing: Any = None, shadows: Any = None, shadow_runner: Callable[[Callable[[], None]], None] | None = None,
                 clock: Callable[[], str] | None = None):
        """``project_cap(project_id)`` returns a project's monthly euro cap (the wizard's budget) or None. ``routing``
        (models/routing.py) moves a purpose to another tier or names a shadow; ``shadows`` (models/shadow.py) logs how
        the shadow's answers compare; ``shadow_runner`` runs a shadow request (a background thread unless a test runs it
        in line); ``clock`` stamps the shadow rows."""
        self.usage, self.transport, self.sleep, self.project_cap = usage, transport, sleep, project_cap
        self.routing, self.shadows, self.clock = routing, shadows, clock or now
        self.shadow_runner = shadow_runner or (lambda fn: threading.Thread(target=fn, daemon=True).start())
        self._lock, self._held = threading.Lock(), {}  # euros held by requests in flight, per project ("p:<id>") and workspace ("w")

    def reserve(self, project_id: str | None, run: "Bound | None", estimate: float | None) -> tuple[str | None, Callable[[], None]]:
        """Before a request: refuse it if it could pass a cap, otherwise hold its upper-bound cost until it ends.

        Returns (refusal or None, release). The check and the hold happen under one lock, so requests sent at the
        same time cannot together pass a cap. A model without a price (estimate None) cannot be stopped in euros.
        """
        nothing = lambda: None  # noqa: E731
        if estimate is None:
            return self.refusal(project_id, run), nothing  # only "already over" can be said without a price
        with self._lock:
            if run is not None and run.run_limit_eur is not None and run.spent_eur + run.reserved_eur + estimate > run.run_limit_eur:
                left = max(0.0, run.run_limit_eur - run.spent_eur - run.reserved_eur)
                return (f"this run has used its budget: €{run.spent_eur:.2f} of €{run.run_limit_eur:.2f}, €{left:.2f} left; "
                        f"the next request could cost up to €{estimate:.2f}"), nothing
            keys = []
            since = month_start() if self.usage is not None else None
            cap = self.project_cap(project_id) if (project_id and self.project_cap) else None
            if cap is not None and since:
                spent = self.usage.spent(since, project_id) + self._held.get("p:" + project_id, 0.0)
                if spent + estimate > cap:
                    return (f"this project has used its budget: €{spent:.2f} of €{cap:.2f} this month, €{max(0.0, cap - spent):.2f} left; "
                            f"the next request could cost up to €{estimate:.2f}. Raise the project's budget or wait for next month"), nothing
                keys.append("p:" + project_id)
            wcap = workspace_cap()
            if wcap is not None and since:
                spent = self.usage.spent(since) + self._held.get("w", 0.0)
                if spent + estimate > wcap:
                    return (f"the workspace has used its monthly budget: €{spent:.2f} of €{wcap:.2f}, €{max(0.0, wcap - spent):.2f} left"), nothing
                keys.append("w")
            ucap, uid = user_cap()
            if ucap is not None and since:
                spent = self.usage.spent(since, user_id=uid) + self._held.get("u:" + uid, 0.0)
                if spent + estimate > ucap:
                    return (f"you have used your monthly model budget: €{spent:.2f} of €{ucap:.2f}, €{max(0.0, ucap - spent):.2f} left; "
                            "it resets on the 1st, or the owner raises DCLAB_LIMIT_USER_MONTHLY_EUR"), nothing
                keys.append("u:" + uid)
            for key in keys:
                self._held[key] = self._held.get(key, 0.0) + estimate
            if run is not None:
                run.reserved_eur += estimate

        def release() -> None:
            with self._lock:
                for key in keys:
                    self._held[key] = max(0.0, self._held.get(key, 0.0) - estimate)
                if run is not None:
                    run.reserved_eur = max(0.0, run.reserved_eur - estimate)
        return None, release

    def refusal(self, project_id: str | None, run: "Bound | None" = None) -> str | None:
        """The cap is already reached (used where no estimate is possible: an unpriced model, a client the gateway does not send for)."""
        if run is not None and run.run_limit_eur is not None and run.run_limit_eur > 0 and run.spent_eur >= run.run_limit_eur:
            return f"this run has used its budget: €{run.spent_eur:.2f} of €{run.run_limit_eur:.2f}"
        if self.usage is None:
            return None
        since = month_start()
        cap = self.project_cap(project_id) if (project_id and self.project_cap) else None
        if cap is not None and self.usage.spent(since, project_id) >= cap > 0:
            return f"this project has used its budget of €{cap:.2f} this month. Raise the project's budget or wait for next month"
        wcap = workspace_cap()
        if wcap is not None and self.usage.spent(since) >= wcap > 0:
            return f"the workspace has used its monthly budget of €{wcap:.2f}"
        ucap, uid = user_cap()
        if ucap is not None and self.usage.spent(since, user_id=uid) >= ucap > 0:
            return f"you have used your monthly model budget of €{ucap:.2f}"
        return None

    def _tier_ready(self, name: str) -> bool:
        if not settings.tier(name)["api_key"]:
            return False
        if self.transport is _transport and os.environ.get("DCLAB_NO_LIVE_MODELS") == "1":
            return False
        return self.transport is not _transport or settings.public(name)["available"]

    def route(self, purpose: str) -> tuple[str, str | None]:
        """The tier that serves a purpose now and the tier that shadows it (A6.4). A routed tier that cannot answer
        falls back to the purpose's own: a switch never leaves a purpose without its model."""
        own = settings.purpose(purpose).tier
        entry = {"shadow": None, "serve": None}
        approved = False
        if self.routing is not None:
            try:
                from .routing import NOT_ROUTED, approved as is_approved, route

                if purpose not in NOT_ROUTED:
                    entry = route(self.routing.get(), purpose)
                    approved = is_approved(entry)
            except Exception:  # noqa: BLE001 — unreadable routing: the purpose keeps its own tier and no shadow
                pass
        # the routed tier serves only while it is the model the reviewer approved, and only while it can answer
        serve = entry["serve"] if entry["serve"] in settings.ROUTABLE and approved and self._tier_ready(entry["serve"]) else own
        shadow = entry["shadow"]
        if shadow == serve or shadow not in settings.ROUTABLE or not self._tier_ready(shadow) or not settings.public(shadow)["local"]:
            shadow = None
        return serve, shadow

    def available(self, purpose: str) -> bool:
        """A model serves this purpose: its tier has a key, and the SDK is installed (or a test supplies the transport).
        With DCLAB_NO_LIVE_MODELS=1 (set by make rd-check, make test-pg and make product-e2e) no live endpoint serves
        anything, whatever .env says: tests use scripted transports only."""
        return self._tier_ready(self.route(purpose)[0])

    def client(self, purpose: str, *, project_id: str | None = None, draft_id: str | None = None, model: str | None = None) -> "Bound | None":
        """A client for one purpose, or None when no model serves it (callers then take their deterministic path).
        ``model`` replaces the tier's model on the same endpoint (a command-line choice); the usage log records it."""
        p = settings.purpose(purpose)
        if not self.available(purpose):
            return None
        return Bound(self, purpose, p, None, project_id, draft_id, model)

    def record(self, purpose: str, model: str, usage: dict[str, Any] | None, seconds: float, attempts: int, outcome: str,
               project_id: str | None = None, draft_id: str | None = None, prompt: str | None = None, base_url: str | None = None,
               tier: str | None = None) -> float | None:
        """Write one usage entry. Clients that cannot be routed through the gateway (the campaign agent's NOOA client)
        call this after each request, so every request is still counted. ``tier`` is the tier that answered (a routed
        purpose's), the purpose's own by default."""
        p = settings.purpose(purpose)
        tier = tier or p.tier
        public = settings.public(tier)
        if base_url:  # a client the gateway does not send for (NOOA) names where its request went
            public = {**public, "endpoint": settings._host(base_url), "local": settings.is_local(base_url)}
        tokens_in, tokens_out = int((usage or {}).get("input_tokens") or 0), int((usage or {}).get("output_tokens") or 0)
        try:
            eur, basis = prices.cost(model, public["local"], tokens_in, tokens_out)
        except ValueError:  # a broken price file: the answer is kept and the request counted in tokens
            eur, basis = None, "no price"
        if self.usage is None:
            return eur
        try:
            self.usage.record({"at": now(), "purpose": purpose, "tier": tier, "model": model, "endpoint": public["endpoint"],
                               "project_id": project_id, "draft_id": draft_id, "input_tokens": tokens_in, "output_tokens": tokens_out,
                               "seconds": round(seconds, 2), "attempts": attempts, "outcome": outcome,
                               "prompt": prompt if os.environ.get("DCLAB_LOG_PROMPTS") == "1" else None, "cost_eur": eur, "cost_basis": basis})
            from .. import observe

            observe.observe_model(purpose, str(tier), outcome, tokens_in or 0, tokens_out or 0, eur)  # 12.4: model usage counters
        except Exception:  # noqa: BLE001 — a usage row that cannot be written must not lose the answer
            pass
        return eur

    def check(self, purpose: str, model: str, passed: bool, reason: str = "", tier: str | None = None) -> None:
        """Record whether a model's output passed the code that checks it (never the output itself)."""
        if self.usage is None:
            return
        try:
            self.usage.record_check({"at": now(), "purpose": purpose, "tier": tier or settings.purpose(purpose).tier, "model": model,
                                     "passed": bool(passed), "reason": (reason or "")[:300] or None})
        except Exception:  # noqa: BLE001 — a check that cannot be logged must not change the answer
            pass

    def summary(self) -> dict[str, Any]:
        """What the pages show: tiers without keys, purposes with what they may see, usage, and models that keep failing."""
        routes = {k: self.route(k) for k in settings.PURPOSES}
        doc = {}
        if self.routing is not None:
            try:
                doc = self.routing.get()
            except Exception:  # noqa: BLE001
                doc = {}
        from .routing import NOT_ROUTED, approved as is_approved, route as routed
        rows = []
        if self.shadows is not None:
            from .shadow import agreement
            try:
                rows = agreement(self.shadows.rows())
            except Exception:  # noqa: BLE001
                rows = []
        return {"tiers": [settings.public(t) for t in settings.TIERS], "tuned": settings.public(settings.TUNED),
                "purposes": [{"purpose": k, "tier": v.tier, "timeout": v.timeout, "may_see": v.may_see, "cell_values": v.cell_values,
                              "serving": routes[k][0], "shadow": routes[k][1], "routing": routed(doc, k), "routable": k not in NOT_ROUTED,
                              "approval_holds": is_approved(routed(doc, k)) if routed(doc, k)["serve"] else None}
                             for k, v in settings.PURPOSES.items()],
                "agreement": rows, "routing_history": (doc.get("history") or [])[-20:],
                "usage": self.usage.totals() if self.usage else None,
                "failing": self.usage.failing(month_start()) if self.usage else []}


class Bound:
    def __init__(self, gateway: Gateway, purpose: str, p: settings.Purpose, tier: dict[str, Any] | None, project_id: str | None,
                 draft_id: str | None, model: str | None = None):
        self.gateway, self.purpose, self.p, self.project_id, self.draft_id, self._model_override = gateway, purpose, p, project_id, draft_id, model
        self.tier, self._shadow_tier, self._serving = {}, None, None
        if tier is not None:  # a fixed tier (older callers); otherwise the routing decides before each request
            self.tier, self._serving = tier, tier["name"]
            self.model, self._transport = tier["model"], gateway.transport(tier, p)
        else:
            self._route()
        self.run_limit_eur: float | None = None  # a cap for one run (an intern session); set by the caller
        self.spent_eur = 0.0  # what this run has cost so far (the intern restores it from the session before each request)
        self.reserved_eur = 0.0

    def _route(self) -> None:
        """Follow the workspace's routing (A6.4): a long-lived client (the intern's) moves when a person moves the purpose."""
        serve, shadow = self.gateway.route(self.purpose)
        if serve != self._serving:
            tier = settings.tier(serve)
            self.tier = {**tier, "model": self._model_override} if self._model_override else tier
            self.model, self._transport, self._serving = self.tier["model"], self.gateway.transport(self.tier, self.p), serve
        self._shadow_tier = settings.tier(shadow) if shadow else None

    def output(self, passed: bool, reason: str = "") -> None:
        """The caller's verdict on this client's last answer: it passed the code that checks it, or why not."""
        self.gateway.check(self.purpose, self.model, passed, reason, self.tier.get("name"))

    def _shadow(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int | None, out: dict[str, Any]) -> None:
        """Send a copy of the request to the shadow, after the served model answered, and log how they compare. The
        shadow's reply goes nowhere else: not to the caller, not to the validator, not into any budget."""
        shadow, gateway = self._shadow_tier, self.gateway
        if shadow is None or gateway.shadows is None:
            return
        snapshot, served = copy.deepcopy(messages), {"tier": self.tier.get("name"), "model": self.model}
        offered = {str((t.get("function") or t).get("name")) for t in tools or [] if isinstance(t, dict)}
        project_id, draft_id = self.project_id, self.draft_id

        def run() -> None:
            from .shadow import compare

            started, reply, outcome = time.monotonic(), None, "ok"
            try:
                reply = gateway.transport(shadow, self.p).complete(snapshot, tools, max_tokens)
            except Exception as error:  # noqa: BLE001 — a shadow that fails is a row, never an error for the caller
                outcome = f"error: {type(error).__name__}"
            usage = (reply or {}).get("usage") or {}
            try:
                gateway.shadows.record({"at": gateway.clock(), "purpose": self.purpose, "primary_tier": served["tier"], "primary_model": served["model"],
                                        "shadow_tier": shadow["name"], "shadow_model": shadow["model"], "project_id": project_id, "draft_id": draft_id,
                                        **compare(out, reply, offered), "outcome": outcome, "seconds": round(time.monotonic() - started, 2),
                                        "input_tokens": int(usage.get("input_tokens") or 0), "output_tokens": int(usage.get("output_tokens") or 0)})
            except Exception:  # noqa: BLE001
                pass
        try:
            gateway.shadow_runner(run)
        except Exception:  # noqa: BLE001
            pass

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, **options: Any) -> dict[str, Any]:
        self._route()
        snapshot = copy.deepcopy(messages) if self._shadow_tier else None  # the caller appends to its list after the answer
        out = self._call(lambda: self._transport.complete(messages, tools, max_tokens, **options), messages, can_retry=lambda: True,
                         max_tokens=max_tokens, tools=tools)
        if snapshot is not None:
            self._shadow(snapshot, tools, max_tokens, out)
        return out

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = 1800, on_text=None) -> dict[str, Any]:
        self._route()
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
        snapshot = copy.deepcopy(messages) if self._shadow_tier else None
        # once a word has reached the user a retry would repeat it, so only a stream that sent nothing is retried
        out = self._call(lambda: send(messages, tools, max_tokens, on_text=relay), messages, can_retry=lambda: not heard.is_set(),
                         max_tokens=max_tokens, tools=tools)
        if snapshot is not None:
            self._shadow(snapshot, tools, max_tokens, out)  # the shadow answers whole, after the user has the served answer
        return out

    def _call(self, send: Callable[[], dict[str, Any]], messages: list[dict[str, Any]], can_retry: Callable[[], bool],
              max_tokens: int | None = None, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        started, attempts, delay = time.monotonic(), 0, BACKOFF
        try:
            chars = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or [])) for m in messages) + len(json.dumps(tools or []))
            estimate = prices.estimate(self.model, settings.public(self.tier.get("name") or self.p.tier)["local"], chars, max_tokens)
            refusal, release = self.gateway.reserve(self.project_id, self, estimate)
        except ValueError as error:  # a broken price file: nothing is sent until it is fixed
            refusal, release = str(error), (lambda: None)
        if refusal:  # checked before sending: nothing reaches the provider
            self._record(messages, None, 0.0, 0, f"refused: {refusal}")
            error = RuntimeError(f"BudgetExceeded: {refusal}")
            error.transient = False  # type: ignore[attr-defined]
            raise error
        try:
            return self._send(send, messages, can_retry, started)
        finally:
            release()

    def _send(self, send: Callable[[], dict[str, Any]], messages: list[dict[str, Any]], can_retry: Callable[[], bool], started: float) -> dict[str, Any]:
        attempts, delay = 0, BACKOFF
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
        eur = self.gateway.record(self.purpose, self.model, (out or {}).get("usage"), seconds, attempts, outcome, self.project_id, self.draft_id, prompt,
                                  tier=self.tier.get("name"))
        self.spent_eur += eur or 0.0


def check_external(purpose: str, project_id: str | None = None) -> None:
    """For clients the gateway does not send for (NOOA): refuse before the request when a cap is already reached."""
    refusal = installed().refusal(project_id)
    if refusal:
        error = RuntimeError(f"BudgetExceeded: {refusal}")
        error.transient = False  # type: ignore[attr-defined]
        raise error


# One gateway per process for code that runs outside a request (the engine's stage notes). The server installs its own.
_INSTALLED: Gateway | None = None


def install(gateway: Gateway) -> None:
    global _INSTALLED
    _INSTALLED = gateway


def installed() -> Gateway:
    """The gateway of the workspace this request or job acts in (``dclab_rnd.context``), else the process's."""
    from .. import context

    current = getattr(context.services(), "gateway", None)
    return current or _INSTALLED or Gateway()


def for_workspace(home=None) -> Gateway:
    """A gateway that logs to the workspace's usage log, for command-line tools that run outside the server."""
    from pathlib import Path

    from .usage import open_usage

    from .routing import open_routing
    from .shadow import open_shadows

    root = Path(__file__).resolve().parents[2]
    home = Path(home or os.environ.get("DCLAB_AGENT_HOME") or root / "agent_runs")
    return Gateway(open_usage(home), routing=open_routing(home), shadows=open_shadows(home))  # the same routing as the server
