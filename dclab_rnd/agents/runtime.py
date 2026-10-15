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

import functools
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import prompts
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
    must_act: bool = False  # an answer before any tool call is not one (the intern's first turn: it has not looked at anything)


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
    stopped: str = "answered"  # answered, terminal, stop_after, steps, replies, time, tools_as_text (a failing model or tool raises)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


# A reply that writes its tool calls as text instead of calling them (small models served without native tool calling
# do this; package 8.1 found it with qwen2.5-coder in Ollama, in several shapes: a ```json fence, a <tools> tag, a bare
# object): nothing ran. A known tool's name inside such a block, in a run where no tool has run, is that case.
_BLOCKS = re.compile(r"```.*?(?:```|$)|<tools?\b.*?(?:</tools?>|$)|\{(?:[^{}]|\{[^{}]*\})*\}", re.S)


def writes_tool_calls(text: str, allowed: list[str] | set[str]) -> bool:
    for block in _BLOCKS.findall(text or ""):
        if any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", block) for name in allowed):
            return True
    return False


def _text(result: Any, chars: int = 4000) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= chars else text[:chars] + f"… [{len(text) - chars} more characters omitted]"


def _in_span(fn: Callable[..., Any]) -> Callable[..., Any]:
    """23.11: a run is a span of the trace running (a new trace when none runs: a script, the campaign): its steps, tool calls and model
    requests are spans inside it, so they share one trace instead of one each."""
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from .. import observe

        with observe.span():
            return fn(*args, **kwargs)
    return wrapped


@_in_span
def run(policy: Policy, client: Any, registry: Registry, messages: list[dict[str, Any]], context: Any = None,
        on_step: Callable[[Step], None] | None = None, stream: Callable[[str], None] | None = None,
        steps_used: int = 0, started: float | None = None, trace: Any = None, contexts: dict[str, Any] | None = None) -> RunResult:
    """Run the policy until it answers, ends, or uses its budget. ``messages`` is extended in place and returned.

    ``trace`` (a ``traces.Tracer``) gets one row per step, with the reply's tokens on its first step (package A2.3).
    ``contexts`` gives a tool scope its own context (the intern: its toolbox for project tools, its session for
    write_plan and finish); a scope without one gets ``context``. Each call is a run in the session record (19.5).
    """
    from datetime import datetime, timezone

    from ..experience import hooks

    if trace is None:  # 21.1: every run is traced; without a store of the caller's, its steps reach the session record only
        from .harness import tracer

        trace = tracer(policy.name)
    began, clock, result, error, interrupted = datetime.now(timezone.utc).isoformat(timespec="microseconds"), time.monotonic(), None, None, False
    from .. import experience

    conversation = trace.run_id if str(getattr(trace, "agent", "")).split(".")[0] in ("home", "intern") else None
    try:
        with experience.bind(run=trace.run_id, conversation=conversation):  # 21.1: what a tool asks (a role, a model) joins this run
            result = _run(policy, client, registry, messages, context, on_step, stream, steps_used, started, trace, contexts)
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
        interrupted = not isinstance(exc, Exception)  # a person's Stop (jobs.worker.Stopped), not a failure
        raise
    finally:
        hooks.from_agent_run(policy, result, trace, began, time.monotonic() - clock, error, interrupted)
        from ..ledger import capture  # 19.7: why the run ended

        capture.stopped(policy, result, trace, interrupted)


def _run(policy: Policy, client: Any, registry: Registry, messages: list[dict[str, Any]], context: Any, on_step: Any, stream: Any,
         steps_used: int, started: float | None, trace: Any, contexts: dict[str, Any] | None) -> RunResult:
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
    nudged = False
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
            text = (out.get("content") or "").strip()
            if trace is not None:  # 21.1: a reply without a tool call is a step of the record too (an answer, or calls as text)
                trace.said(text, reply=reply, tokens=tokens, verdict="answered" if result.steps or not (nudged or policy.must_act
                                                                                                       or writes_tool_calls(text, allowed)) else "no_call")
            # Only in a run where no tool has run: a report that quotes a call after real work is an answer.
            if not result.steps and (nudged or policy.must_act or writes_tool_calls(text, allowed)):
                if not nudged:  # once: the model is told; any reply without a call after that ends the run
                    nudged = True
                    messages.append({"role": "user", "content": prompts.text("tools_as_text")})
                    continue
                result.final, result.stopped = text, "tools_as_text"
                return result
            result.final, result.stopped = text, "answered"
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
                      tokens=tokens if first else None, text=(out.get("content") or "").strip() or None if first else None)
            if on_step:
                on_step(step)
            from ..ledger import capture  # 19.7: the tool the model chose among those it was offered

            capture.tool_choice(policy, step, allowed, trace)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _text(value, policy.result_chars)})
            # Only a call that ran ends the run or the turn: a refused finish or question goes back to the model to correct.
            if step.ok and tool is not None and tool.name in terminal:
                end = "terminal"
            elif step.ok and ((tool is not None and tool.name in stop_after) or (policy.stop_when is not None and policy.stop_when(step))):
                end = end or "stop_after"
        if end:
            result.stopped = end
            return result
