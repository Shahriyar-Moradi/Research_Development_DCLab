"""A brief per track, and questions that matter (package H4).

A prediction has its plan (``plan.py``: the outcome, the moment, the action, the data). The other five tracks had
nothing to hold what the person said, so Home could not ask, record or show it. This module gives each track its own
brief, with the plan's machinery: every field has a status (stated, inferred, unknown), a source (the problem
sentence, an answer, the data, or a default) and the person's words that support it.

- ``TRACK_FIELDS``: each track's fields, why each matters, its default (when a safe one exists) and where its options
  come from (the table's numeric, categorical or time columns, or a fixed list).
- ``extract``: keyword rules that read a field from the person's words, in English and Persian (``lang``), claiming a
  field only when its words are there; every value is a quote of the person's text.
- ``refresh``: the brief as the draft stands now; a field that names a column the table does not have (a snake_case
  name, "lifetime_spend") is dropped and listed in ``missing``, never taken.
- ``next_field``: what to ask, if anything: only a field that is unknown and has no safe default, within the track's
  budget (``BUDGET``); a field with a default is recorded with ``source="default"`` and shown, not asked.

Code decides what enters the brief; a model never writes it directly (H6 onwards add the model's helpers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from . import lang
from . import plan as plans


@dataclass(frozen=True)
class Field:
    name: str
    why: str
    question: str
    default: Optional[str] = None  # None: no safe default, so Home asks when it is unknown
    options_from: Optional[str] = None  # "numeric", "categorical", "time", or None (free answer)


TRACK_FIELDS: dict[str, tuple[Field, ...]] = {
    "analysis": (
        Field("measure", "The number the answer is about decides the query.", "What should I measure, for example a total, an average or a count of what?", None, "numeric"),
        Field("breakdown", "A breakdown splits the answer into groups.", "Should I split it by anything?", "no breakdown", "categorical"),
        Field("period", "The period sets which rows count.", "Which period?", "all rows", "time"),
        Field("filters", "A filter keeps only some rows.", "Should I keep only some rows?", "no filter"),
    ),
    "preparation": (
        Field("inputs", "The tables the result is made from.", "Which tables should I use?", "the tables you upload"),
        Field("rules", "What to do to the data, step by step.", "What should I do to the data?", None),
        Field("output_grain", "What one row of the result stands for.", "What should one row of the result be?", "as in the table"),
        Field("destination", "Where the result goes.", "How do you want the result?", "a cleaned table to download"),
    ),
    "reporting": (
        Field("kpis", "The numbers the report shows.", "Which numbers should the report show?", None, "numeric"),
        Field("audience", "Who reads it sets its level of detail.", "Who reads it?", "you"),
        Field("refresh", "How often it is brought up to date.", "How often should it be refreshed?", "weekly"),
        Field("decisions_supported", "What the readers decide with it.", "What will people decide with it?", "not stated"),
    ),
    "business": (
        Field("decision", "The decision the answer must support.", "What decision is this for?", "understand what happened"),
        Field("success_measure", "The number that says whether it is going well.", "Which number tells you whether it is going well?", None, "numeric"),
        Field("options", "The choices on the table.", "Which options are you choosing between?", "not stated"),
        Field("period_compared", "The periods compared.", "Which periods should I compare?", "the latest period against the one before", "time"),
        Field("constraints", "What the decision must respect.", "Is there anything the decision must respect?", "none stated"),
    ),
    "advice": (
        Field("topic", "What the explanation is about.", "What would you like explained?", None),
    ),
}
BUDGET = {"prediction": 4, "analysis": 2, "preparation": 3, "reporting": 4, "business": 3, "advice": 1}  # questions per track

_STOP = r"(?:by|per|for|in|over|from|between|during|last|this|and|with|on|across|to|of|that|which|so|every|each|than|has|have|is|are|was|were|did|do|the|a|an|me)"
_WORD = rf"(?!{_STOP}\b)[\w-]+(?:'s)?"
_NOUNS = rf"{_WORD}(?:\s+{_WORD}){{0,2}}"  # one to three words, never a joining word, no trailing space
_MONTH = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
_GROUP = r"(?:type|category|channel|plan|segment|group|region|city|country)"
_END = r"(?=\s+(?:by|for|every|each|per|to|that|which|so|in|over|with)\b|[.,;?!:]|$)"
_LIST_END = r"(?=,?\s+(?:by|for|every|each|per|to|that|which|so|in|over|with|refreshed|updated|daily|weekly|monthly)\b|[.;?!:]|$)"  # a list of numbers keeps its commas
_FA_END = r"(?=\s+(?:برای|در|که|تا|و|چقدر|چند|چه|است|بود|بودند|هست|را)(?:\s|[.,;؟?!،:]|$)|[.,;؟?!،:]|$)"

# (track, field, pattern, group): the group is the value, quoted from the person's text. Written in the normalised form
# (lang.normalize); English first, then Persian. Several patterns per field: the first that matches wins.
RULES: list[tuple[str, str, str, int]] = [
    # analysis
    ("analysis", "measure", rf"\b((?:average|mean|median|total|sum|number|count|share|percentage|rate)\s+(?:of\s+)?{_NOUNS})", 1),
    ("analysis", "measure", r"\bhow\s+(?:much|many)\s+((?!do\b|did\b|does\b|is\b|are\b|was\b|were\b|will\b)[\w-]+)", 1),
    ("analysis", "measure", rf"\bshow\s+(?:me\s+)?(?:the\s+)?((?!me\b|the\b|breakdown\b|trend\b|split\b|list\b|report\b|dashboard\b|numbers\b|data\b|table\b|results?\b)[\w-]+){_END}", 1),
    ("analysis", "measure", r"(?:^|\s)((?:میانگین|مجموع|جمع|تعداد)\s+\S+(?:\s+(?!(?:بر|در|به|از|برای|که|و|هر|را|با)(?:\s|$))\S+)?)", 1),
    ("analysis", "breakdown", rf"\bbetween\s+(?:the\s+)?\w+\s+and\s+(?:the\s+)?\w+\s+({_GROUP})\b", 1),
    ("analysis", "breakdown", rf"\b(?:by|per|for each|in each|across)\s+(?!(?:the\s+)?(?:end|way|time|then|now|default|far|hand|next|last|this|day|week|month|year|quarter|monday)\b)(?:(?:the|our|their|my|your|all)\s+)?((?!the\b|our\b|their\b|my\b|your\b|all\b)[\w-]+(?:\s+{_GROUP})?)\b", 1),
    ("analysis", "breakdown", r"(?:به\s+تفکیک|بر\s+اساس)\s+(\S+(?:\s+\S+)?)" + _FA_END, 1),
    ("analysis", "breakdown", r"(?:در|برای)\s+هر\s+(\S+)", 1),
    ("analysis", "breakdown", r"(?:^|\s)هر\s+(\S+)", 1),
    ("analysis", "period", rf"\b((?:last|this|previous|past)\s+(?:week|month|quarter|year)|(?:the\s+)?last\s+\d+\s+days)\b", 1),
    ("analysis", "period", rf"\b(?:in|over|during|for)\s+((?:{_MONTH}\s+)?(?:19|20)\d\d|{_MONTH})\b", 1),
    ("analysis", "period", r"((?:سال\s+)?\d{4}|ماه\s+گذشته|پارسال|امسال|سه\s+ماه\s+(?:اول|دوم|سوم|آخر|گذشته))", 1),
    ("analysis", "filters", rf"\bonly\s+(?:the\s+)?({_NOUNS})", 1),
    ("analysis", "filters", r"\bbetween\s+(?:the\s+)?(\w+\s+and\s+(?:the\s+)?\w+)\s+\w+", 1),
    ("analysis", "filters", rf"\b(?:more|less|fewer)\s+than\s+\d+\s+({_NOUNS})", 1),
    # preparation
    ("preparation", "inputs", r"\b(?:join|merge|combine)\s+(.+?)\s+(?:on|so|into|by|then|and\s+(?:remove|drop|export|save))\b", 1),
    ("preparation", "rules", r"\b((?:remove|drop|delete|keep only|join|merge|combine|pivot|turn|convert|rename|fill|filter|deduplicate)\b(?:[^.;?!]|\.(?=\w))*?)(?=\s+and\s+(?:export|save|give|send)\b|\s+then\b|[.;?!](?!\w)|$)", 1),
    ("preparation", "rules", r"((?:\S+\s+){1,6}(?:حذف|تبدیل|ادغام|ترکیب)\s+(?:کن|کنید)|(?:یک\s+)?pivot\s+بساز[^.؟?]*)", 1),
    ("preparation", "output_grain", r"\b(?:one\s+row\s+per|a\s+row\s+for\s+each|table\s+per|so\s+every)\s+(\w+)", 1),
    ("preparation", "output_grain", r"هر\s+(\S+)\s+یک\s+ردیف", 1),
    ("preparation", "destination", r"\b((?:as|to|into)\s+(?:an?\s+)?(?:excel|csv|xlsx|parquet)(?:\s+file)?)\b", 1),
    ("preparation", "destination", r"(خروجی\s+(?:\S+\s+){0,2}(?:اکسل|csv)|فایل\s+اکسل)", 1),
    # reporting
    ("reporting", "kpis", rf"\b(?:dashboard|report|kpis?)\s+(?:of|on|listing|with)\s+(.+?){_LIST_END}", 1),
    ("reporting", "kpis", rf"\b(?:track|monitor)\s+((?!how\b|what\b|whether\b|if\b|why\b|it\b|this\b|that\b).+?){_LIST_END}", 1),
    ("reporting", "kpis", r"(?:داشبورد|گزارش)\s+(?:\S+\s+)?(?:از|برای)\s+(.+?)(?=\s+(?:برای\s+مدیر|برای\s+تیم|که|هر)\s|[.,;؟?!،:]|$)", 1),
    ("reporting", "audience", r"\bfor\s+(?:the\s+)?((?:[\w-]+\s+){0,2}(?:team|managers?|leadership|board|director|ceo|executives?)(?:\s+team)?)\b", 1),
    ("reporting", "audience", r"برای\s+((?:مدیر|مدیران|تیم)\s+\S+)", 1),
    ("reporting", "refresh", r"(?:\b[\w-]+\s+)?\b(daily|weekly|monthly|quarterly|every\s+(?:day|week|month|quarter|monday|morning))\b", 1),
    ("reporting", "refresh", r"(روزانه|هفتگی|ماهانه|هر\s+(?:روز|هفته|ماه))", 1),
    # business
    ("business", "decision", r"\b(which\s+\w+(?:\s+\w+)?\s+should\s+we\s+[^?.!:]+)", 1),
    ("business", "decision", r"\bwhether\s+we\s+should\s+([^?.!]+)", 1),
    ("business", "decision", r"\b(?:should|shall)\s+we\s+([^?.!]+)", 1),
    ("business", "decision", r"(چه\s+کار\s+(?:باید\s+)?(?:بکنیم|کنیم)|چه\s+باید\s+کرد)", 1),
    ("business", "success_measure", r"\bwhy\s+(?:did|do|does|is|are|was|were|has|have)\s+(?:our\s+|the\s+)?([\w-]+(?:\s+[\w-]+)?)\s+(?:drop|fall|rise|grow|increase|decrease|go|change|decline|fell|dropped)", 1),
    ("business", "success_measure", r"\bwhat\s+would\s+(?:it|that|this)\s+do\s+to\s+(?:the\s+|our\s+)?([\w-]+)", 1),
    ("business", "success_measure", r"چرا\s+(\S+)\s+[^.؟?]*?(?:کم|زیاد|کاهش|افزایش|افت)\s+(?:شد|یافت|پیدا\s+کرد)", 1),
    ("business", "options", r"\b(\w+(?:,\s*\w+)+,?\s+or\s+\w+)\b", 1),
    ("business", "period_compared", rf"\b((?:in\s+)?{_MONTH}(?:\s+(?:19|20)\d\d)?\s+(?:compared\s+(?:with|to)|versus|vs\.?|against)\s+{_MONTH}(?:\s+(?:19|20)\d\d)?)\b", 1),
    ("business", "period_compared", r"(سه\s+ماه\s+(?:اول|دوم|سوم|آخر)(?:\s+(?:سال\s+)?\d{4})?)", 1),
    # advice
    ("advice", "topic", r"\bdifference\s+between\s+([^?,.]+)", 1),
    ("advice", "topic", r"\b(?:is\s+it\s+(?:ok|okay|fine|right|safe|wrong)\s+to|should\s+i)\s+([^?.]+)", 1),
]
_COMPILED = [(t, f, re.compile(p, re.I), g) for t, f, p, g in RULES]
_QUESTION = re.compile(r"[^.?!؟]+[.?!؟]?")


def fields(track: str) -> tuple[Field, ...]:
    return TRACK_FIELDS.get(track, ())


def field(track: str, name: str) -> Optional[Field]:
    return next((f for f in fields(track) if f.name == name), None)


_TRIM = " ,.;:?!\"'“”،؟«»\u200c"


def extract(track: str, text: str) -> dict[str, tuple[str, str]]:
    """The fields of ``track`` the person's words say: ``{field: (value, quote)}``. The value is the words the rule
    took ("region"); the quote is the person's words around it that say so ("per region"), which pass the product's
    quote rule. Rules in both languages, on the normalised text, cut from the original."""
    found: dict[str, tuple[str, str]] = {}
    norm, index = lang.normalize(text)
    for t, name, pattern, group in _COMPILED:
        if t != track or name in found:
            continue
        match = pattern.search(norm)
        if not match or not (match.group(group) or "").strip():
            continue
        a, b = lang.span(text, index, match.start(group), match.end(group))
        qa, qb = lang.span(text, index, match.start(0), match.end(0))
        value, quote = text[a:b].strip(_TRIM), text[qa:qb].strip(_TRIM)
        if value and plans.quoted(quote, [text]):
            found[name] = (value[:300], quote[:300])
    if track == "advice" and "topic" not in found:  # an explanation is about the whole question
        first = _QUESTION.match(text.strip())
        value = (first.group(0) if first else text).strip(_TRIM)[:200]
        if plans.quoted(value, [text]):
            found["topic"] = (value, value)
    return found


def _columns(draft: dict[str, Any]) -> set[str]:
    return {str(c.get("name")) for c in ((draft.get("analysis") or {}).get("columns") or []) if c.get("name")}


def _missing_column(value: str, columns: set[str]) -> Optional[str]:
    """A snake_case name in a value that the table does not have ("lifetime_spend"), when a table is there."""
    if not columns:
        return None
    lowered = {c.lower() for c in columns}
    for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b", value):
        if token.lower() not in lowered:
            return token
    return None


def refresh(draft: dict[str, Any]) -> dict[str, Any]:
    """The briefs of the draft's tracks as they stand: what the problem sentence says, what the person answered, and
    the defaults of what is still unknown. Returns ``{"briefs": {track: {field: entry}}, "missing": [(field, column), ...]}``."""
    intent = draft.get("intent") or {}
    tracks = [t for t in [intent.get("track"), *(intent.get("tracks") or [])] if t in TRACK_FIELDS]
    tracks = list(dict.fromkeys(tracks))
    previous = draft.get("track_brief") or {}
    answered = {(q.get("track"), q["field"]): q["answered"] for q in draft.get("questions") or []
                if q.get("answered") and not q.get("superseded") and q.get("track")}
    columns = _columns(draft)
    problem = draft.get("problem") or ""
    briefs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for track in tracks:
        said = extract(track, problem)
        entries: dict[str, Any] = {}
        for f in fields(track):
            old = (previous.get(track) or {}).get(f.name) or {}
            answer = answered.get((track, f.name))
            unsure = bool(answer) and lang.not_sure(answer)
            default = {"value": f.default, "status": "inferred", "source": "default", "quote": None} if f.default is not None else None
            unknown = {"value": None, "status": "unknown", "source": None, "quote": None}
            checks = f.options_from is not None  # only a field whose values are the table's columns is checked against them
            if old.get("source") == "answer" and old.get("value") and not (checks and _missing_column(old["value"], columns)):
                entries[f.name] = old  # what the person answered stands
            elif answer and not unsure:  # their reply is the value; the reply itself is their words (no quote needed)
                absent = _missing_column(answer, columns) if checks else None
                if absent:
                    missing.append((f.name, absent))
                    entries[f.name] = default or unknown
                else:
                    entries[f.name] = {"value": answer.strip()[:300], "status": "stated", "source": "answer", "quote": None}
            elif f.name in said:  # what the sentence says; a "not sure" never erases it
                value, quote = said[f.name]
                absent = _missing_column(value, columns) if checks else None
                if absent:
                    missing.append((f.name, absent))
                    entries[f.name] = default or unknown
                else:
                    entries[f.name] = {"value": value, "status": "stated", "source": "problem", "quote": quote}
            else:
                entries[f.name] = default or unknown
        briefs[track] = entries
    return {"briefs": briefs, "missing": sorted(set(missing))}  # a field and the column it named that the table lacks


def options(draft: dict[str, Any], f: Field) -> list[str]:
    """A question's buttons, from the table when one is there: numeric columns for a measure, low-cardinality
    categories for a breakdown, the time column for a period. Names only, never values."""
    profile = ((draft.get("analysis") or {}).get("profile") or {})
    cols = (draft.get("analysis") or {}).get("columns") or []
    names: list[str] = []
    if f.options_from == "numeric":
        names = [c["name"] for c in cols if c.get("kind") == "numeric" and not c.get("id_like")]
    elif f.options_from == "categorical":
        names = [c["name"] for c in cols if c.get("kind") in ("categorical", "boolean") and 1 < (c.get("unique") or 0) <= 30]
    elif f.options_from == "time":
        names = list(profile.get("time_candidates") or [])[:1]
    return [str(n) for n in names][:3]


def next_field(draft: dict[str, Any], track: str, asked: int) -> Optional[tuple[Field, list[str]]]:
    """The one field worth asking now, or None: unknown, no safe default, not asked yet, and within the budget."""
    if asked >= BUDGET.get(track, 0):
        return None
    entries = (refresh(draft)["briefs"].get(track) or {})
    already = {q["field"] for q in draft.get("questions") or [] if q.get("track") == track and not q.get("superseded")}
    for f in fields(track):
        entry = entries.get(f.name) or {}
        if entry.get("status") == "unknown" and f.default is None and f.name not in already:
            return f, options(draft, f)
    return None
