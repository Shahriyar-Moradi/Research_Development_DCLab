"""The Home agent's plan: what it knows about the problem, what it does not, and what to ask next (package A3.1).

For each field the plan keeps a status (stated, inferred or unknown), where it came from (the problem sentence, an
answer, the model, or the data) and the user's words that support it. The questions come from the plan: only an
unknown field is asked, in the order that matters most for leakage (the outcome, then the prediction moment, then
what is done and what an error costs, then the data).

Code decides what enters the plan. A model may propose a reading of the user's text, but only with a quote, and a
quote that is not in the user's own words (the problem sentence or their messages) is dropped. Without a model the
keyword rules below do the same job for the outcome and the moment, the two fields leakage depends on; they claim
an outcome only when it names who or what ("which", "whether"…) and has a window ("within 30 days"), because an
outcome without one is not yet a label, and a moment only when the sentence sets it apart from the outcome (after a
comma or a verb such as "scored"), because "cancel before renewal" is part of the outcome, not the moment.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dclab_rnd import prompts

FIELDS = ("target", "prediction_moment", "action", "data_plan")  # in the order they are asked
STATUSES = ("stated", "inferred", "unknown")
SOURCES = ("problem", "answer", "model", "data")
QUOTE_CHARS = 300
WHY = {
    "target": "Everything else follows from the outcome; it decides the task, the metric and which columns could leak it.",
    "prediction_moment": "Anything written after this moment must stay out of the model; it is the main source of leakage.",
    "action": "The cost of errors sets the metric and the operating point.",
    "data_plan": "A sample (even 1,000 rows) lets DCLab check the columns against the prediction moment.",
}

_UNIT = r"(?:hours?|days?|weeks?|months?|quarters?|years?)"
_WINDOW = rf"(?:within|in the next|over the next|during the next|in the coming|by the end of the|next)\s+(?:\d+\s+|one\s+|two\s+|three\s+|a\s+|the\s+)?{_UNIT}"
_TARGET = re.compile(rf"\b(?:predict|forecast|estimate|flag|detect|score|know)\s+((?:whether|which|if|who|how many|how much)\b[^,.;?]*?{_WINDOW})", re.I)
_QUESTION_TARGET = re.compile(rf"^\s*((?:which|who|whether|will)\b[^,.;?]*?{_WINDOW})", re.I)
# A moment is claimed only after a comma or a verb that sets it apart from the outcome ("…within 30 days, scored on
# the first day of each month"); the forms are tried in this order, so "2 days before dispatch" keeps its offset.
_SET_APART = r"(?:,\s*(?:and\s+)?|\b(?:scored|predicted|computed|checked|evaluated|refreshed)\s+)"  # not "made" or "run": "a purchase is made at checkout" is an outcome
_MOMENTS = [re.compile(_SET_APART + "(" + p + ")", re.I) for p in (
    r"(?:\d+|one|two|three|a)\s+(?:hours?|days?|weeks?|months?)\s+(?:before|ahead of|in advance of)\s+[^,.;?]+",
    r"(?:at|on)\s+the\s+(?:moment|time)\s+(?:of|when)\s+[^,.;?]+",
    r"(?:at|on)\s+the\s+(?:start|end|beginning|first day|last day)\s+of\s+(?:each|every|the)\s+[^,.;?]+",
    r"when\s+(?:an?|the)\s+[a-z ]+?\s+(?:is|are)\s+(?:placed|created|submitted|opened|received|booked|made|filed)\b[^,.;?]*",
    r"(?:at|before|on)\s+(?:sign-?up|checkout|booking|order time|dispatch|application|admission|login|renewal|onboarding|registration)\b[^,.;?]*",
    r"(?:every|each)\s+(?:morning|evening|night|day|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|month|quarter)\b[^,.;?]*",
)]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def user_text(draft: dict[str, Any]) -> list[str]:
    """What the user wrote: the problem sentence and their messages (never the agent's own words)."""
    return [draft.get("problem") or ""] + [m.get("text") or "" for m in draft.get("messages") or [] if m.get("role") == "user"]


def quoted(quote: Any, texts: list[str]) -> bool:
    """True when the quote is the user's own words: whole words (at least two, or eight characters), found in one of
    their texts. A fragment of a word ("cus" in "customers") is not a quote."""
    q = _norm(quote or "").strip(" \"'“”‘’.,;:")
    if not (len(q.split()) >= 2 or len(q) >= 8) or len(q) > QUOTE_CHARS:
        return False
    pattern = re.compile(r"(?<![\w])" + re.escape(q) + r"(?![\w])")
    return any(pattern.search(_norm(t)) for t in texts)


def supports(quote: Any, value: Any) -> bool:
    """True when the value says something the quote says: they share a content word (compared as 5-letter stems)."""
    stems = lambda text: {w[:5] for w in re.findall(r"[a-z0-9]{3,}", _norm(text))} - {"the", "and", "each", "every", "with", "that", "this", "which", "whether"}  # noqa: E731
    return bool(stems(quote) & stems(value))


def extract(text: str) -> dict[str, str]:
    """The keyword rules: the outcome (with its window) and the prediction moment, quoted from ``text``."""
    found: dict[str, str] = {}
    match = _TARGET.search(text) or _QUESTION_TARGET.search(text)
    if match:
        found["target"] = match.group(1).strip()
    taken = match.span(1) if match else (0, 0)
    for pattern in _MOMENTS:
        for moment in pattern.finditer(text):
            if moment.start(1) >= taken[1] or moment.end(1) <= taken[0]:  # never the outcome's own window again
                found["prediction_moment"] = re.split(r"\s+(?:and|so|to)\s+", moment.group(1).strip(), maxsplit=1)[0]  # "every Monday and the team calls…"
                return found
    return found


def empty() -> dict[str, Any]:
    return {f: {"status": "unknown", "source": None, "quote": None} for f in FIELDS} | {"next": None, "why": None}


def entry(status: str, source: str, quote: str | None) -> dict[str, Any]:
    return {"status": status, "source": source, "quote": (quote or "").strip()[:QUOTE_CHARS] or None}


def refresh(draft: dict[str, Any]) -> dict[str, Any]:
    """The plan for the draft as it is now: a known field keeps its entry, one the user answered is stated, and the
    next field to ask is the first unknown one that was not asked yet (the data only when none is on its way)."""
    plan = {**empty(), **{k: v for k, v in (draft.get("plan") or {}).items() if k in FIELDS}}
    u = draft.get("understanding") or {}
    answered = {q["field"]: q["answered"] for q in draft.get("questions") or [] if q.get("answered") and not q.get("superseded")}
    has_data = any(a.get("status") in ("ready", "queued", "structuring", "cleaning", "analysing") for a in draft.get("assets") or [])
    for field in FIELDS:
        if not u.get(field):
            if field == "data_plan" and has_data:
                plan[field] = entry("stated", "data", None)
            else:
                plan[field] = {"status": "unknown", "source": None, "quote": None}
        elif plan[field]["status"] == "unknown":
            plan[field] = entry("stated", "answer", answered.get(field) or u[field])
    asked = {q["field"] for q in draft.get("questions") or [] if not q.get("superseded")}
    plan["next"] = next((f for f in FIELDS if plan[f]["status"] == "unknown" and f not in asked), None)
    plan["why"] = WHY.get(plan["next"]) if plan["next"] else None
    return plan


PROPOSE = prompts.text("home_plan")  # dclab_rnd/prompts/home_plan.md (A4.3: prompts are versioned files)


def proposal(content: str, texts: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """The model's reading, checked: the entries whose quote is in the user's words, and what was dropped and why."""
    text = re.sub(r"^```(?:json)?|```$", "", str(content or "").strip(), flags=re.M).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}, ["the reply was not JSON"]
    if not isinstance(value, dict):
        return {}, ["the reply was not a JSON object"]
    kept, dropped = {}, []
    for field in ("target", "prediction_moment", "action"):
        item = value.get(field)
        if not isinstance(item, dict) or item.get("status") not in ("stated", "inferred"):
            continue
        reading = str(item.get("value") or item.get("quote") or "").strip()[:400]
        if not reading or not quoted(item.get("quote"), texts) or not supports(item.get("quote"), reading):
            dropped.append("a quote is not in the user's text")
            continue
        kept[field] = {"status": item["status"], "value": reading, "quote": str(item["quote"]).strip()}
    return kept, sorted(set(dropped))
