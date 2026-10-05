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

RESULT_CHARS = 4000  # what one tool result may add to the conversation


@dataclass(frozen=True)
class Policy:
    """Who the agent is: its instructions, the tools it may call, its budget, and the tools that end a run."""
    name: str
    system: str
    tools: tuple[str, ...]
    max_steps: int = 24
    max_seconds: float = 20 * 60
    terminal: tuple[str, ...] = ()  # calling one of these ends the run after it executes
    stop_after: tuple[str, ...] = ()  # calling one of these ends the turn (for example a question to the user)


@dataclass
class Step:
    n: int
    tool: str
    arguments: dict[str, Any]
    result: Any
    seconds: float
    ok: bool


@dataclass
class RunResult:
    messages: list[dict[str, Any]]
    steps: list[Step] = field(default_factory=list)
    final: str | None = None
    stopped: str = "answered"  # answered, terminal, stop_after, steps, time (a failing model or tool raises)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


def _text(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)[:RESULT_CHARS]


def run(policy: Policy, client: Any, registry: Registry, messages: list[dict[str, Any]], context: Any = None,
        on_step: Callable[[Step], None] | None = None, stream: Callable[[str], None] | None = None,
        steps_used: int = 0, started: float | None = None) -> RunResult:
    """Run the policy until it answers, ends, or uses its budget. ``messages`` is extended in place and returned."""
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

    def spent() -> str | None:
        if count >= policy.max_steps:
            return "steps"
        if time.monotonic() - started >= policy.max_seconds:
            return "time"
        return None

    while True:
        end = spent()
        if end:
            result.stopped = end
            return result
        if stream is not None and hasattr(client, "stream"):
            out = client.stream(messages, schemas, on_text=stream)
        else:
            out = client.complete(messages, schemas)
        for key in ("input_tokens", "output_tokens"):
            result.usage[key] += int((out.get("usage") or {}).get(key) or 0)
        messages.append(out["assistant_message"])
        calls = out.get("tool_calls") or []
        if not calls:
            result.final, result.stopped = (out.get("content") or "").strip(), "answered"
            return result
        for call in calls:
            # Every call in the reply gets an answer (the API requires one per call), but none runs after the run ended
            # or its budget was used: the budget is checked before each call, not only before each model request.
            if end in (None, "stop_after"):
                end = spent() or end
            if end in ("terminal", "steps", "time"):
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": _text({"error": f"Not run: the run ended ({end})."})})
                continue
            clock = time.monotonic()
            name = call["name"]
            tool = registry.get(name)
            if tool is None or tool.name not in allowed:
                value: Any = {"error": f"Tool {name!r} is not available here. Available: {', '.join(allowed)}"}
            else:
                value = registry.call(name, call.get("arguments") or {}, context)
            count += 1
            step = Step(count, name, call.get("arguments") or {}, value, round(time.monotonic() - clock, 3),
                        not (isinstance(value, dict) and "error" in value))
            result.steps.append(step)
            if on_step:
                on_step(step)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _text(value)})
            if tool is not None and tool.name in terminal:
                end = "terminal"
            elif tool is not None and tool.name in stop_after:
                end = "stop_after"
        if end:
            result.stopped = end
            return result
