"""D2 / P5 (2026-10-23): an export rendered with the model's own chat template, the same setting as at run time, and checked.

    python -m dclab_rnd.store.template_check EXPORT.jsonl --tokenizer PATH [--kwargs '{"preserve_thinking": true}']
    python -m dclab_rnd.store.template_check EXPORT.jsonl --template chat_template.jinja

If training and running use different templates, the model learns one habit and is used with another. So each line is
rendered with the template the tokenizer folder carries (``chat_template.jinja``, or ``chat_template`` in
tokenizer_config.json; no model library needed, Jinja alone) with the same keyword settings the gateway sends
(``DCLAB_TIER_<T>_CHAT_TEMPLATE_KWARGS``, ``preserve_thinking`` for Qwen), and the rendered text must hold every assistant
turn's ``reasoning_content`` (the earlier turns too: a template that drops them teaches the model to forget) and every tool
call's arguments as their JSON object (never ``[object`` or an escaped string). A line that fails is named with its turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def template_of(tokenizer: Path | None, template: Path | None) -> str:
    if template is not None:
        return Path(template).read_text(encoding="utf-8")
    if tokenizer is None:
        raise ValueError("a --tokenizer folder or a --template file is needed")
    folder = Path(tokenizer)
    if (folder / "chat_template.jinja").is_file():
        return (folder / "chat_template.jinja").read_text(encoding="utf-8")
    config = folder / "tokenizer_config.json"
    if config.is_file():
        text = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(text, str) and text:
            return text
    raise FileNotFoundError(f"no chat template in {folder} (chat_template.jinja or tokenizer_config.json's chat_template)")


def _tojson(value: Any, **kwargs: Any) -> str:
    """As the model libraries' template filter renders it: insertion order (no sorting), non-ASCII kept."""
    return json.dumps(value, ensure_ascii=False, **kwargs)


def _raise(message: str) -> None:
    raise ValueError(message)


def render(template: str, line: dict[str, Any], kwargs: dict[str, Any]) -> str:
    """The line rendered as the model libraries render it: a sandboxed environment (a downloaded template reaches no Python
    internals), ``tojson``, ``raise_exception`` and ``strftime_now`` (Llama-style templates date their prompt)."""
    from datetime import datetime

    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"], keep_trailing_newline=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(template).render(messages=line["messages"], tools=line.get("tools") or [], add_generation_prompt=False, **kwargs)


def check_line(rendered: str, line: dict[str, Any]) -> list[str]:
    """Every problem of one line: a reasoning not in the rendered text (blank lines around it allowed: templates strip them),
    a tool call whose arguments did not render as their JSON object (in the recorded or the sorted key order; a Python repr
    or an escaped string fails, whatever the schema in the system prompt says)."""
    problems = []
    for n, m in enumerate(line["messages"]):
        if m.get("role") != "assistant":
            continue
        reasoning = str(m.get("reasoning_content") or "").strip()
        if reasoning and reasoning not in rendered:
            problems.append(f"turn {n}: its reasoning is not in the rendered text (an earlier turn's thinking dropped?)")
        for call in m.get("tool_calls") or []:
            arguments = (call.get("function") or {}).get("arguments")
            if not isinstance(arguments, dict):
                problems.append(f"turn {n}: the tool call's arguments are not an object")
                continue
            if "_raw" in arguments:
                problems.append(f"turn {n}: the tool call's arguments were not valid JSON when recorded (_raw)")
            forms = {_tojson(arguments), _tojson(arguments, sort_keys=True), json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))}
            if not any(form in rendered for form in forms):
                problems.append(f"turn {n}: the tool call's arguments did not render as their JSON object")
    if "[object" in rendered or "&#34;" in rendered:
        problems.append("the template rendered an object as text or escaped a quote")
    return problems


def check(path: Path, template: str, kwargs: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    lines = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    failed = []
    for line in lines:
        try:
            problems = check_line(render(template, line, kwargs), line)
        except Exception as error:  # noqa: BLE001 — a template that cannot render the line is that line's problem
            problems = [f"the template failed on it: {type(error).__name__}: {str(error)[:200]}"]
        if problems:
            failed.append({"id": line.get("id"), "problems": problems})
    return {"lines": len(lines), "failed": len(failed), "kwargs": kwargs, "first": failed[:limit]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export", type=Path, help="an export's train.jsonl or dev.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=None, help="the model's tokenizer folder (chat_template.jinja or tokenizer_config.json)")
    parser.add_argument("--template", type=Path, default=None, help="a chat template file instead of a tokenizer folder")
    parser.add_argument("--kwargs", default='{"preserve_thinking": true}', help="the template's keyword settings, as sent at run time (JSON)")
    args = parser.parse_args(argv)
    try:
        template = template_of(args.tokenizer, args.template)
        kwargs = json.loads(args.kwargs)
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    report = check(args.export, template, kwargs)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
