"""A model writes a query plan; code decides whether it runs (package H6, used by Home's answers in H7).

The model (purpose ``home_query``) sees the person's question, the brief Home holds for it, and the table's column
summaries: names, kinds, missing share and number of distinct values. Never a row, a value, a category name or a
result. Its reply is a plan, which ``validate`` checks; a refused plan goes back to the model once, with the reasons.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from dclab_rnd import prompts

from .validate import PlanError, kinds, validate

PROMPT = "home_query"


def summaries(frame) -> list[dict[str, Any]]:
    """What the model may know about the table: per column its name, kind, missing share and distinct count."""
    k = kinds(frame)
    rows = max(len(frame), 1)
    return [{"name": str(c), "kind": k[str(c)], "missing": round(float(frame[c].isna().mean()), 3), "distinct": int(frame[c].nunique(dropna=True))}
            for c in frame.columns][:120] + ([{"rows": rows}] if rows else [])


def _first_json(text: str) -> Optional[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def propose(question: str, frame, client: Any, brief: Optional[dict[str, Any]] = None) -> tuple[Optional[dict[str, Any]], list[str]]:
    """A valid plan for the question, or None with the reasons. One retry with the validator's reasons."""
    state = {"question": str(question)[:600], "brief": {k: v for k, v in (brief or {}).items() if isinstance(v, str)}, "columns": summaries(frame)}
    messages = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]
    problems: list[str] = []
    for attempt in range(2):
        try:
            reply = client.complete(messages, None)
        except Exception as exc:  # noqa: BLE001 — no model: no plan, the caller answers without one
            return None, [f"the model did not answer ({type(exc).__name__})"]
        content = reply.get("content") if isinstance(reply, dict) else str(reply)
        plan = _first_json(content or "")
        if plan is None:
            problems = ["the reply was not a JSON plan"]
        else:
            try:
                checked = validate(plan, frame)
                if getattr(client, "output", None):
                    client.output(True, "")
                return checked, []
            except PlanError as exc:
                problems = exc.problems
        if getattr(client, "output", None):
            client.output(False, "; ".join(problems)[:200])
        if attempt == 0:
            messages += [{"role": "assistant", "content": (content or "")[:4000]},
                         {"role": "user", "content": "The plan was refused: " + "; ".join(problems)[:1500] + ". Write a corrected plan."}]
    return None, problems
