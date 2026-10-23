"""D2 (2026-10-23): what a local OpenAI-compatible server does with a model's reasoning, measured, never assumed.

    python -m dclab_rnd.models.probe_reasoning --tier cheap [--purpose intern]

One two-turn tool conversation through the gateway (so it is capped, logged and kept like any request; its project id
is ``probe``): the model is asked to think of a code word and keep it to itself, then to call a tool; the second turn
asks for the code word. The report says which reasoning field the reply carried (the tier's setting; with none set,
the known fields the reply carried anyway, ``reasoning_fields_seen``), whether the chat-template setting was accepted
(not refused; a server may still ignore it), and whether the second reply knew the earlier turn's reasoning. The owner
runs it against each server (Ollama's /v1, Ollama's own API behind a shim, mlx_lm.server, vLLM) before choosing one; a
test never does (D3). Record the result in research_notes/open_models_2026-10/servers.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

TOOL = [{"type": "function", "function": {"name": "note", "description": "Say that you are ready.",
                                           "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
FIRST = ("Think of a random code word of exactly six capital letters and keep it in your reasoning only: do not write it in "
         "your reply. Then call the tool note with the text 'ready'.")
SECOND = "What was the code word you thought of a moment ago? Answer with the word only."
CODE = re.compile(r"\b[A-Z]{6}\b")


def purpose_for(gateway: Any, tier: str, wanted: str | None) -> str:
    from . import settings

    if wanted:
        return wanted
    for name in settings.PURPOSES:
        if gateway.route(name)[0] == tier and name != "embedding":
            return name
    raise SystemExit(f"no purpose is served by the tier {tier!r}: name one with --purpose")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tier", default="cheap")
    parser.add_argument("--purpose", default=None, help="a purpose served by the tier (default: the first one routed to it)")
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args(argv)
    from . import settings
    from .gateway import Gateway

    t = settings.tier(args.tier)
    report: dict[str, Any] = {"tier": args.tier, "endpoint": settings._host(t["base_url"]) if t["base_url"] else None, "model": t["model"],
                              "reasoning_field": t["reasoning_field"], "chat_template_kwargs": t["chat_template_kwargs"],
                              "tool_arguments": t["tool_arguments"]}
    if not t["model"] or not t["base_url"]:
        report["error"] = "the tier has no endpoint or no model"
        print(json.dumps(report, indent=1))
        return 2
    gateway = Gateway()
    purpose = purpose_for(gateway, args.tier, args.purpose)
    report["purpose"] = purpose
    bound = gateway.client(purpose, project_id="probe")  # marked: never a project's, never a training example's source
    if bound is None:
        report["error"] = f"no model serves the purpose {purpose!r} (is the tier available?)"
        print(json.dumps(report, indent=1))
        return 2
    messages: list[dict[str, Any]] = [{"role": "system", "content": "You are careful and brief."}, {"role": "user", "content": FIRST}]
    try:
        first = bound.complete(messages, TOOL, max_tokens=args.max_tokens)
    except RuntimeError as error:
        report["first_turn"] = f"failed: {error}"
        report["chat_template_kwargs_accepted"] = False if t["chat_template_kwargs"] else None
        print(json.dumps(report, indent=1))
        return 1
    reasoning = str(first.get("reasoning") or "")
    report["reasoning_fields_seen"] = first.get("reasoning_fields_seen") or []  # with no field set: what the reply carried anyway
    report["chat_template_kwargs_accepted"] = (True if t["chat_template_kwargs"] else None)  # not refused; a server may still ignore it
    report["first_reply"] = {"reasoning_chars": len(reasoning), "content_chars": len(first.get("content") or ""),
                             "tool_calls": [c["name"] for c in first.get("tool_calls") or []],
                             "arguments_type": type((first.get("assistant_message") or {}).get("tool_calls", [{}])[0].get("function", {}).get("arguments")).__name__
                             if first.get("tool_calls") else None}
    codes = CODE.findall(reasoning)
    report["code_in_reasoning"] = codes[-1] if codes else None
    messages.append(first["assistant_message"])
    for call in first.get("tool_calls") or []:
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"ok": True})})
    messages.append({"role": "user", "content": SECOND})
    try:
        second = bound.complete(messages, TOOL, max_tokens=args.max_tokens)
    except RuntimeError as error:
        report["second_turn"] = f"failed: {error}"
        print(json.dumps(report, indent=1))
        return 1
    answer = str(second.get("content") or "")
    report["second_reply"] = {"content": answer[:200], "reasoning_chars": len(str(second.get("reasoning") or ""))}
    if codes:
        report["earlier_reasoning_seen"] = codes[-1] in answer.upper()
    else:
        report["earlier_reasoning_seen"] = "unknown: no code word was read from the first reply's reasoning"
    report["verdict"] = ("reasoning kept and sent back" if report["earlier_reasoning_seen"] is True
                         else "reasoning read but the second reply did not know it (the server dropped it, or the template setting is off)" if codes
                         else ("reasoning read; the model wrote no code word in it, so whether it is sent back is not shown (run again)"
                               if reasoning else
                               f"no reasoning read: set DCLAB_TIER_{args.tier.upper()}_REASONING_FIELD to one of reasoning_fields_seen, or the server returns none"))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
