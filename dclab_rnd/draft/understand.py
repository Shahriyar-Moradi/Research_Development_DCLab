"""Home's understanding conversation (the owner's Telco test, 2026-10-29: "be more open, do not ask many questions
without a minimum understanding, talk naturally like Claude for about three turns, then give options based on context").

When Home's first reading of a request is unclear and a model serves ``home_understand``, the model leads a short
conversation instead of six buttons: each turn it says back what it understood and asks one open question; it keeps an
understanding card (goal, decision, subject, data said, the kinds of help still possible) and, once it understands
enough, proposes two to four options built from the context. The model writes; code decides:

- every card field is kept only when its quote is the person's words and it carries no number they did not write;
- every sentence said to the person keeps only the person's numbers (and the table's column names), as everywhere else;
- every option is checked: a known kind of help and tool, columns the table has, nothing the table cannot support
  (periods without a date column, a prediction with no outcome to learn from);
- "enough to offer" is the code's rule (:func:`enough`): a goal and the data known with a kind of help in view, the
  person asking for options, or the turn limit.

Without a model, Home keeps its open question with every kind of help as a button (the rules' fallback)."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .. import prompts

PROMPT = "home_understand"
TURNS = 3  # questions of the understanding conversation before Home offers options whatever it knows
MIN_TURNS = 2  # answered questions before any option (the owner: about three turns of talk; the review: never options first)
TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")
TOOLS = ("ml_project", "query", "segmentation_project", "recipe", "dashboard", "business_brief", "cited_answer")
TOOL_OF_TRACK = {"prediction": "ml_project", "analysis": "query", "preparation": "recipe", "reporting": "dashboard",
                 "business": "business_brief", "advice": "cited_answer"}
TRACK_OF_TOOL = {**{v: k for k, v in TOOL_OF_TRACK.items()}, "segmentation_project": "prediction"}  # a segmentation is a project
CARD_FIELDS = ("goal", "decision", "subject", "data_said")
SHOW_OPTIONS = "Show me what you can do"  # the one button of an understanding question: skip ahead to the options
SOMETHING_ELSE = "Something else"


def columns_view(draft: dict[str, Any]) -> list[dict[str, Any]]:
    """What the model may know of the table: each column's name and kind and what DCLab noticed, never a value."""
    analysis = draft.get("analysis") or {}
    flags = {str(c.get("name")): c for c in ((analysis.get("profile") or {}).get("columns") or [])}
    out = []
    for c in analysis.get("columns") or []:
        name = str(c.get("name") or "")
        if not name:
            continue
        p = flags.get(name, {})
        noticed = [label for key, label in (("id_like", "identifier"), ("name_id_like", "identifier"), ("target_name_like", "outcome-like name"),
                                            ("time_like", "date or time")) if p.get(key)]
        if c.get("kind") == "numeric" and 1 < int(c.get("unique") or 0) <= 10 and not p.get("target_name_like"):
            noticed.append("small code or flag")
        out.append({"name": name, "kind": c.get("kind"), "noticed": sorted(set(noticed))})
    return out


def conversation(draft: dict[str, Any]) -> list[dict[str, str]]:
    """The person's words and Home's own understanding questions, in order, ending on the person's words."""
    turns = [{"role": "user", "content": str(draft.get("problem") or "")}]
    for m in draft.get("messages") or []:
        if m.get("role") == "user" and str(m.get("text") or "").strip():
            turns.append({"role": "user", "content": str(m["text"])})
        elif m.get("role") == "agent" and m.get("kind") == "question" and (m.get("question") or {}).get("understand"):
            turns.append({"role": "assistant", "content": str(m.get("text") or "")})
        elif m.get("role") == "agent" and m.get("kind") == "question" and (m.get("question") or {}).get("offer"):
            titles = "; ".join(str(o.get("title") or "") for o in m["question"]["offer"])  # the review: free words after an offer are read
            turns.append({"role": "assistant", "content": f"{m.get('text') or ''} (Options offered: {titles}; or something else.)"})
    merged: list[dict[str, str]] = []
    for t in turns:  # two of the same role in a row are one turn (a button, then words)
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] += "\n" + t["content"]
        else:
            merged.append(dict(t))
    while merged and merged[-1]["role"] != "user":
        merged.pop()
    return merged


def _texts(draft: dict[str, Any]) -> list[str]:
    from .plan import user_text

    return [t for t in user_text(draft) if t]


def _numbers_ok(sentence: str, allowed_texts: list[str]) -> bool:
    from .. import cited

    allowed = set().union(*(cited.numbers(t, source=True) for t in allowed_texts)) if allowed_texts else set()
    return not (cited.numbers(sentence) - allowed)


def _sentences(text: str, allowed_texts: list[str]) -> str:
    kept = [s for s in re.split(r"(?<=[.?!])\s+", str(text or "").strip()) if s and _numbers_ok(s, allowed_texts)]
    return " ".join(kept)[:600]


def _card(raw: Any, texts: list[str]) -> dict[str, Any]:
    from .plan import quoted

    raw = raw if isinstance(raw, dict) else {}
    card: dict[str, Any] = {}
    for name in CARD_FIELDS:
        entry = raw.get(name)
        if isinstance(entry, dict) and entry.get("value") and quoted(entry.get("quote"), texts) and len(str(entry["quote"]).split()) >= 3:
            # the review: a two-word quote ("my customer") filled the goal; the field is the person's own words, the model's beside
            card[name] = {"value": str(entry["quote"])[:200], "quote": str(entry["quote"])[:200], "model_value": str(entry["value"])[:200]}
    card["tracks"] = [t for t in raw.get("tracks") or [] if t in TRACKS][:4]
    card["unknowns"] = [str(u)[:120] for u in raw.get("unknowns") or [] if isinstance(u, str)][:5]
    return card


def capabilities(draft: dict[str, Any]) -> dict[str, bool]:
    """What the draft's table can support (no table: everything is still possible, the data comes later)."""
    view = columns_view(draft)
    if not view:
        return {"table": False, "dates": True, "outcome": True}
    return {"table": True, "dates": any("date or time" in c["noticed"] or c["kind"] == "datetime" for c in view),
            "outcome": any("outcome-like name" in c["noticed"] for c in view)}


def validate(options: Any, draft: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The options code accepts, and why each refused one was refused (kept for the record)."""
    names = {c["name"] for c in columns_view(draft)}
    can = capabilities(draft)
    texts = [*_texts(draft), *names]
    kept, refused = [], []
    for o in options if isinstance(options, list) else []:
        if not isinstance(o, dict):
            continue
        title, what = str(o.get("title") or "").strip()[:80], str(o.get("what") or "").strip()[:200]
        tool = o.get("tool") if o.get("tool") in TOOLS else None
        track = TRACK_OF_TOOL[tool] if tool else (o.get("track") if o.get("track") in TRACKS else None)  # the review: the tool decides
        tool = tool or (TOOL_OF_TRACK[track] if track else None)
        columns = [str(c) for c in o.get("columns") or [] if isinstance(c, str)]
        why = None
        if not title or not track:
            why = "no title or no known kind of help"
        elif columns and (not names or not set(columns) <= names):
            why = "a column the table does not have"
        elif not (_numbers_ok(title, texts) and _numbers_ok(what, texts)):
            why = "a number nobody gave"
        elif track == "business" and can["table"] and not can["dates"]:
            why = "a comparison of periods on a table with no date column"
        elif tool == "ml_project" and can["table"] and not can["outcome"]:
            why = "a prediction with no outcome to learn from in the table"
        elif title.casefold() in {k["title"].casefold() for k in kept}:
            why = "the same option twice"
        if why:
            refused.append(f"{title or '?'}: {why}")
            continue
        kept.append({"title": title, "what": what, "track": track, "tool": tool, "columns": columns})
    return kept[:4], refused


WANT_OPTIONS = ("You now understand enough to help: answer with enough true and 2 to 4 concrete options built from what they said "
                "(and their table, if any), with a one-sentence lead-in as say.")
DECLINED = ("The person passed over the options you offered. Do not offer them again: say back what you understood and ask one "
            "open question about what they need instead (enough false).")


def ask(client: Any, draft: dict[str, Any], want_options: bool = False, declined: bool = False) -> dict[str, Any]:
    """One turn of the model: ``{"say", "card", "model_enough", "options", "refused", "request"}``, or ``{"failed": why}``.
    ``want_options``: code decided it is time for options and the model gave none (HOMECONV-002, p-06): asked once more for
    them; ``declined``: the person said "Something else": a new question, never the same options again."""
    view = columns_view(draft)
    state = ("The person's table, as DCLab described it (no values): " + json.dumps(view, ensure_ascii=False)) if view else \
        "No table has been shared yet."
    steer = [{"role": "system", "content": WANT_OPTIONS}] if want_options else [{"role": "system", "content": DECLINED}] if declined else []
    messages = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "system", "content": state}, *steer, *conversation(draft)]
    try:
        reply = client.complete(messages, None)
    except Exception as error:  # noqa: BLE001 - Home falls back to its open question and buttons
        return {"failed": (str(error) or type(error).__name__)[:200], "request": getattr(client, "last_request", None)}
    request = getattr(client, "last_request", None)
    from .intent import _first_json

    answer = _first_json(reply.get("content") if isinstance(reply, dict) else str(reply)) or {}
    if not answer:
        return {"failed": "the understanding reply was not JSON", "request": request}
    texts = _texts(draft)
    say = _sentences(str(answer.get("say") or ""), [*texts, *(c["name"] for c in view)])
    options, refused = validate(answer.get("options"), draft)
    return {"say": say, "card": _card(answer.get("card"), texts), "model_enough": bool(answer.get("enough")),
            "options": options, "refused": refused, "request": request}


def enough(card: dict[str, Any], draft: dict[str, Any], turns: int, asked_for_options: bool = False) -> bool:
    """The code's rule for offering options: the person asked for them, the turn limit is reached, or their goal and their
    data are known with at least one kind of help in view."""
    if asked_for_options or turns >= TURNS:
        return True
    if turns < MIN_TURNS:
        return False
    has_data = bool(card.get("data_said") or draft.get("assets"))
    return bool(card.get("goal")) and has_data and bool(card.get("tracks"))


GROUPS = re.compile(r"\b(?:segment\w*|cluster\w*|groups?|types? of|kinds? of|personas?)\b", re.I)


def fallback_options(card: dict[str, Any], reading: Optional[dict[str, Any]] = None, texts: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """The rules' options when the model gave fewer than two valid ones: new groups first when the person's words ask for
    groups or types (HOMECONV-002, p-06: "types of travellers" got no segmentation), then the likely kinds of help."""
    from . import intent as intents

    first = [t for t in [*(card.get("tracks") or []), *((reading or {}).get("tracks") or [])] if t in TRACKS]
    order = list(dict.fromkeys(first + list(TRACKS)))[:4]
    out = [{"title": intents.LABELS[t], "what": "", "track": t, "tool": TOOL_OF_TRACK[t], "columns": []} for t in order]
    if any(GROUPS.search(str(t or "")) for t in texts or []):
        out = [{"title": "Find groups in your data", "what": "", "track": "prediction", "tool": "segmentation_project", "columns": []}, *out][:4]
    return out
