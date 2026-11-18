"""The plain arm: the model alone with a locked Python box on the table, until it calls finish (or its calls run out)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dclab_rnd import prompts

from . import sandbox

PROMPT = "e2e_plain_agent"
PURPOSE = "e2e_plain_agent"
MAX_CALLS = 24
MAX_TOKENS = 16000
LAST = ("[the test's runner] That was your last tool call. Call finish now with your answer, from what you have seen.")
TOOLS = [
    {"type": "function", "function": {"name": "run_python", "description": "Run a Python script in your working folder (the table is there).",
                                      "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the work with your answer.", "parameters": {
        "type": "object", "required": ["target", "task", "metric", "value", "low", "high", "features", "report"],
        "properties": {"target": {"type": "string"}, "task": {"type": "string"}, "metric": {"type": "string"},
                       "value": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
                       "features": {"type": "array", "items": {"type": "string"},
                                    "description": "The table's own columns your final model reads, named as in its header."}, "report": {"type": "string"}}}}},
]


def run(case: Any, client: Any, folder: Path) -> dict[str, Any]:
    """The arm's outcome: what it finished with (or None), every tool output it saw, and its calls."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    case.table.to_csv(folder / case.filename, index=False)
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": case.request()}]
    seen: list[str] = []
    code: list[str] = []  # the scripts it ran: a number it set there (a fold's size, an iteration cap) is its choice, not made up
    calls: list[dict[str, Any]] = []
    finished: Optional[dict[str, Any]] = None
    for n in range(MAX_CALLS):
        last = n == MAX_CALLS - 1
        if last and messages[-1].get("role") == "tool":
            messages.append({"role": "user", "content": LAST})
        offered = [t for t in TOOLS if t["function"]["name"] == "finish"] if last else TOOLS
        out = client.complete(messages, offered, max_tokens=MAX_TOKENS)
        messages.append(out.get("assistant_message") or {"role": "assistant", "content": out.get("content") or ""})
        tool_calls = out.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user", "content": "[the test's runner] Use run_python to work, and finish to answer."})
            continue
        for c in tool_calls:
            args = c.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if c["name"] == "finish":
                finished = dict(args)
                calls.append({"tool": "finish", "ok": True})
                value: dict[str, Any] = {"ok": True}
            elif c["name"] == "run_python":
                code.append(str(args.get("code") or ""))
                value = sandbox.run_python(code[-1], folder, len(calls) + 1)
                seen.append(str(value.get("stdout") or "") + str(value.get("stderr") or "") + str(value.get("error") or ""))
                calls.append({"tool": "run_python", "ok": value.get("returncode") == 0})
            else:
                value = {"error": f"no tool {c['name']!r}"}
                calls.append({"tool": c["name"], "ok": False})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(value, default=str)})
        if finished is not None:
            break
    return {"arm": "plain", "finished": finished, "seen": seen, "code": code, "calls": calls}
