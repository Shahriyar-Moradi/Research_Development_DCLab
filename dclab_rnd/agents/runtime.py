"""The agent loop every DCLab agent runs on (package A2.1).

    result = run(policy, client, registry, messages, context)

The loop asks the model (through the gateway client), checks each tool call against the tool's schema (a bad call
goes back to the model as an error it can read), runs the tool, and repeats until the model answers without a tool
call, calls one of the policy's terminal tools, or the step or time budget is used up. What the model sees and does
is returned step by step, so callers can show it, store it and replay it (package A2.3).

Deterministic code still owns every decision: a tool that changes a project goes through the workflow validator
(package A2.2), and the model never computes a split, a metric or a selection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .registry import Registry



@dataclass(frozen=True)
class Policy:
    """Who the agent is: its instructions, the tools it may call, its budget, and the tools that end a run."""
    name: str
    system: str
    tools: tuple[str, ...]
    max_steps: int = 24
    max_seconds: float = 20 * 60
    terminal: tuple[str, ...] = ()  # a successful call to one of these ends the run; never refused for the step budget
    stop_after: tuple[str, ...] = ()  # a successful call to one of these ends the turn (for example a question to the user)
    stop_when: Callable[["Step"], bool] | None = None  # or a successful step whose result says so (a simulation that started)
    max_replies: int | None = None  # model requests per run() call, besides the tool-call budget
    result_chars: int = 4000  # what one tool result may add to the conversation


@dataclass
class Step:
    n: int
    tool: str
    arguments: dict[str, Any]
    result: Any
    seconds: float
    ok: bool
    reply: int = 0  # which model reply (0, 1, ...) in this run() asked for it


@dataclass
class RunResult:
    messages: list[dict[str, Any]]
    steps: list[Step] = field(default_factory=list)
    final: str | None = None
    stopped: str = "answered"  # answered, terminal, stop_after, steps, replies, time (a failing model or tool raises)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


def _text(result: Any, chars: int = 4000) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= chars else text[:chars] + f"… [{len(text) - chars} more characters omitted]"


def run(policy: Policy, client: Any, registry: Registry, messages: list[dict[str, Any]], context: Any = None,
        on_step: Callable[[Step], None] | None = None, stream: Callable[[str], None] | None = None,
        steps_used: int = 0, started: float | None = None, trace: Any = None, contexts: dict[str, Any] | None = None) -> RunResult:
    """Run the policy until it answers, ends, or uses its budget. ``messages`` is extended in place and returned.

    ``trace`` (a ``traces.Tracer``) gets one row per step, with the reply's tokens on its first step (package A2.3).
    ``contexts`` gives a tool scope its own context (the intern: its toolbox for project tools, its session for
    write_plan and finish); a scope without one gets ``context``.
    """
    started = time.monotonic() if started is None else started
    unknown = [n for n in policy.tools if registry.get(n) is None]
    if unknown:
        raise ValueError(f"policy {policy.name} names tools that are not registered: {', '.join(unknown)}")
    allowed = [registry.get(n).name for n in policy.tools]
    terminal = {registry.get(n).name for n in policy.terminal if registry.get(n)}
    stop_after = {registry.get(n).name for n in policy.stop_after if registry.get(n)}
    schemas = registry.schemas(names=allowed)
    result = RunResult(messages=messages)
    count = steps_used
    reply = -1
    if trace is not None:
        trace.next_call()  # a tracer reused across run() calls continues its reply numbers

    def spent(ending: bool = False) -> str | None:
        if count >= policy.max_steps and not ending:  # a tool that ends the run may always end it
            return "steps"
        if time.monotonic() - started >= policy.max_seconds:
            return "time"
        return None

    while True:
        end = spent() or ("replies" if policy.max_replies is not None and reply + 1 >= policy.max_replies else None)
        if end:
            result.stopped = end
            return result
        if stream is not None and hasattr(client, "stream"):
            out = client.stream(messages, schemas, on_text=stream)
        else:
            out = client.complete(messages, schemas)
        reply += 1
        tokens = {key: int((out.get("usage") or {}).get(key) or 0) for key in ("input_tokens", "output_tokens")}
        for key in tokens:
            result.usage[key] += tokens[key]
        messages.append(out["assistant_message"])
        calls = out.get("tool_calls") or []
        if not calls:
            result.final, result.stopped = (out.get("content") or "").strip(), "answered"
            return result
        for call in calls:
            # Every call in the reply gets an answer (the API requires one per call), but none runs after the run ended
            # or its budget was used: the budget is checked before each call, not only before each model request.
            name = call["name"]
            tool = registry.get(name)
            ending = tool is not None and tool.name in terminal
            if end in (None, "stop_after") or (end == "steps" and ending):  # a used-up budget still lets the run end
                end = spent(ending=ending) or (None if end == "steps" else end)
            if end in ("terminal", "steps", "time"):
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": _text({"error": f"Not run: the run ended ({end})."})})
                continue
            if trace is not None:
                trace.before()  # the row records what the agent saw, not the state after its move
            clock = time.monotonic()
            if tool is None or tool.name not in allowed:
                value: Any = {"error": f"Tool {name!r} is not available here. Available: {', '.join(allowed)}"}
            else:
                value = registry.call(name, call.get("arguments") or {}, (contexts or {}).get(tool.scope, context))
            count += 1
            step = Step(count, name, call.get("arguments") or {}, value, round(time.monotonic() - clock, 3),
                        not (isinstance(value, dict) and "error" in value), reply)
            result.steps.append(step)
            if trace is not None:
                first = sum(1 for s in result.steps if s.reply == reply) == 1
                trace(step.tool, step.arguments, step.result, step.seconds, reply=reply,
                      tokens=tokens if first else None)
            if on_step:
                on_step(step)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _text(value, policy.result_chars)})
            # Only a call that ran ends the run or the turn: a refused finish or question goes back to the model to correct.
            if step.ok and tool is not None and tool.name in terminal:
                end = "terminal"
            elif step.ok and ((tool is not None and tool.name in stop_after) or (policy.stop_when is not None and policy.stop_when(step))):
                end = end or "stop_after"
        if end:
            result.stopped = end
            return result
