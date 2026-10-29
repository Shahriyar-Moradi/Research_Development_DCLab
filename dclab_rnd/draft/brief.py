"""A brief per track (package H4): what Home needs to know for each kind of help, read from the person's words first.

A prediction keeps its plan (``plan.py``: outcome, moment, action, data). The other tracks get their own fields, each
with why it matters, a stated default (or none) and where its options come from (the table's columns or a fixed list):

- analysis: question, measure, breakdown, period, filters
- preparation: inputs, output_grain, rules, destination
- reporting: kpis, audience, refresh, decisions_supported
- business: decision, options, success_measure, period_compared, constraints
- advice: topic

Rules, in English and Persian (``lang.normalize``), read what the words say, quoted from the person's text; a rule that
is not sure leaves the field unknown (a field left unknown costs a question; a wrong one misleads). A value naming a
column the data does not have is refused and kept apart (``missing``), never taken: Home says the column is not there.

Home asks a field only when it is unknown and has no safe default (``asks``); every other field takes its default,
recorded with source "default" and shown, so the person can change it. A "Not sure" answer records the default too, and
never touches a field already known. Each track has its own question budget (``BUDGET``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from . import lang

TRACKS = ("analysis", "preparation", "reporting", "business", "advice")


@dataclass(frozen=True)
class Field:
    name: str
    why: str
    question: str = ""
    default: Optional[str] = None  # None: no safe default, so Home asks when it is unknown
    options_from: str = "fixed"  # numeric, categorical, time or fixed
    options: tuple[str, ...] = ()
    asked: bool = True  # False: never asked (read from the words or defaulted)


TRACK_FIELDS: dict[str, tuple[Field, ...]] = {
    "analysis": (
        Field("question", "The question the answer must settle.", default="the request as written", asked=False),
        Field("measure", "What is counted or summed decides the query and the chart.",
              "What should I measure?", options_from="numeric", options=("Number of rows",)),
        Field("breakdown", "A breakdown splits the answer into groups.", "Split it by what?",
              default="overall, no breakdown", options_from="categorical", asked=False),
        Field("period", "Which rows count: a period changes every number.", "Which period?",
              default="all rows", options_from="time", asked=False),
        Field("filters", "Rows left out change the answer.", default="none", asked=False),
    ),
    "preparation": (
        Field("inputs", "The tables the result is made from.", default="the uploaded tables", asked=False),
        Field("output_grain", "What one row of the result is decides joins and totals.",
              default="one row per input row", asked=False),
        Field("rules", "The steps to apply, in order.", "What should I do to the data?",
              options=("Remove duplicate rows", "Fix types and dates", "Drop empty rows")),
        Field("destination", "Where the prepared table goes.", default="a prepared table to download (CSV)", asked=False),
    ),
    "reporting": (
        Field("kpis", "The numbers on the report or dashboard.", "Which numbers should it show?", options_from="numeric"),
        Field("audience", "Who reads it sets its level of detail; it never shows individual people's rows.",
              default="you", asked=False),
        Field("refresh", "How often it updates sets how it is built.", "How often should it update?",
              options=("Daily", "Weekly", "Monthly", "Only when I ask")),
        Field("decisions_supported", "What it helps decide.", default="not said", asked=False),
    ),
    "business": (
        Field("decision", "The decision the brief supports.", default="explain what changed and why", asked=False),
        Field("options", "The choices being compared.", default="not said", asked=False),
        Field("success_measure", "The number that says whether things got better.", "Which number tells you it worked?",
              options_from="numeric"),
        Field("period_compared", "The periods compared.", default="the latest period against the one before", asked=False),
        Field("constraints", "Limits on the options (budget, time, policy).", default="none said", asked=False),
    ),
    "advice": (
        Field("topic", "What the explanation is about.", default="the question as written", asked=False),
    ),
}
BUDGET = {"analysis": 2, "preparation": 3, "reporting": 4, "business": 3, "advice": 1}  # a prediction keeps chat.MAX_QUESTIONS
NOT_SURE = "Not sure"


def field(track: str, name: str) -> Optional[Field]:
    return next((f for f in TRACK_FIELDS.get(track, ()) if f.name == name), None)


# ------------------------------------------------------------------------------------------------ reading the words
_STOP = r"(?:last|this|next|previous|in|over|during|since|between|from|to|with|and|or|the|for|of|on|at|by|per|across|within|among|when|where|compared|vs|did|do|does|was|were|is|are|has|have|each|every|only|so|then|too)"
_WORDS = r"[A-Za-z_][\w\-]*(?:\s+(?!" + _STOP + r"\b)[A-Za-z_][\w\-]*)?"  # one name, or two when the second is not a joining word
_MONTH = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
# a month name only after a time word or before a year ("I may want" and "march on" are not periods: the review of H4)
_EN_PERIOD = (rf"\b(?:(?:last|this|previous|past)\s+(?:\d+\s+)?(?:quarters?|months?|years?|weeks?|days?)|(?:in|over|during|for)\s+(?:the\s+)?(?:year\s+)?20\d\d|of\s+(?:the\s+year\s+)?20\d\d(?=\s*(?:$|[.?!,;]|by\b|per\b))"
              rf"|(?:in|during|for|since|from|until|between)\s+{_MONTH}(?:\s+20\d\d)?|{_MONTH}\s+20\d\d)\b")
_NOT_A_GROUP = r"(?!(?:end|way|time|bridge|now|then|far|default|hand|chance|mistake|today|tomorrow|myself|itself|far)\b)"
_ARTICLE = r"(?:(?:the|a|an|our|my)\s+)?"
_FA_PERIOD = r"(?:(?:سه\s+ماه|ماه|فصل|سال|هفته)\s+(?:اول|دوم|سوم|چهارم|گذشته|قبل|پیش)(?:\s+(?:سال\s+)?\d{4})?|سال\s+\d{4})"
_REFRESH_WORDS = r"(?:daily|weekly|monthly|quarterly|hourly)"

# (track, field, pattern, group) on the normalized text; the first that matches wins. Precision first: no pattern guesses.
PATTERNS: list[tuple[str, str, str, int]] = [
    # analysis
    ("analysis", "measure", rf"\b((?:average|avg|mean|median|total|sum\s+of|number\s+of|count\s+of|share\s+of|percentage\s+of|rate\s+of)\s+{_ARTICLE}{_WORDS})", 1),
    ("analysis", "measure", rf"\bhow\s+much\s+({_WORDS})\s+(?:did|do|does|have|has|was|were|is|are)\b", 1),
    ("analysis", "measure", r"\bhow\s+many\s+([A-Za-z_][\w\-]*)", 1),
    ("analysis", "measure", r"\bwhat\s+((?:share|percentage|proportion|fraction)\s+of\s+[A-Za-z_][\w\-]*)", 1),
    ("analysis", "measure", rf"\bshow\s+(?:me\s+)?(?:the\s+)?({_WORDS})\s+(?:per|by|for\s+each)\b", 1),
    ("analysis", "measure", r"((?:میانگین|مجموع|جمع|تعداد)\s+[^\s،,.;؛?؟]+(?:\s+(?!در\b|به\b|هر\b|از\b|را\b|چقدر\b)[^\s،,.;؛?؟]+)?)", 1),
    ("analysis", "breakdown", rf"\b(?:by|per|for\s+each|in\s+each|across)\s+(?!(?:the\s+|each\s+)?{_NOT_A_GROUP[3:-1]})(?:the\s+|each\s+)?(?!(?:the|each)\b)({_WORDS})", 1),
    ("analysis", "breakdown", r"(?:به\s*تفکیک|بر\s*حسب)\s+([^\s،,.;؛?؟]+(?:\s+(?!چقدر\b|در\b)[^\s،,.;؛?؟]+)?)", 1),
    ("analysis", "breakdown", r"(?:^|\s)(?:در\s+)?هر\s+([^\s،,.;؛?؟]+)", 1),
    ("analysis", "period", rf"({_EN_PERIOD})", 1),
    ("analysis", "period", rf"({_FA_PERIOD})", 1),
    ("analysis", "filters", r"\bonly\s+(?!(?:interested|want|wants|need|needs|if|when|because|to|a|one|in|for|about)\b)((?:the\s+)?[A-Za-z_][\w\-]*(?:\s+[A-Za-z_][\w\-]*)?)", 1),
    ("analysis", "filters", r"\b(?:with|has|have)\s+((?:more|less|fewer)\s+than\s+\d+\s+[A-Za-z_][\w\-]*(?:\s+[A-Za-z_][\w\-]*)?)", 1),
    # preparation
    ("preparation", "rules", r"\b((?:remove|drop|delete|dedupe|deduplicate|keep\s+only|filter|join|merge|combine|convert|turn|rename|fill|split|pivot|aggregate|group|fix|clean)\b[^.;:?]*?)(?=\s+and\s+(?:then\s+)?(?:export|save|give|send|show|download)\b|[.;:?]|$)", 1),
    ("preparation", "rules", r"([^\s،,.;؛:?؟]+(?:\s+[^\s،,.;؛:?؟]+){0,5}?\s+(?:تبدیل|حذف|ادغام|ترکیب|پاک|تمیز)\s*(?:کن|کنید))", 1),
    ("preparation", "rules", r"((?:یک\s+)?pivot[^.;?؟]*)", 1),
    ("preparation", "output_grain", r"\bone\s+row\s+(?:per|for\s+each)\s+([A-Za-z_][\w\-]*)", 1),
    ("preparation", "output_grain", r"هر\s+([^\s،,.;؛:?؟]+)\s+یک\s+ردیف", 1),
    ("preparation", "destination", r"\b(?:as|to|into)\s+(?:an?\s+)?((?:excel|csv|parquet|json|xlsx|google\s+sheets?)(?:\s+file)?)\b", 1),
    ("preparation", "destination", r"((?:فایل\s+)?(?:اکسل|csv))", 1),
    ("preparation", "inputs", r"\b(?:join|merge|combine)\s+(the\s+[A-Za-z_][\w\-]*\s+(?:table\s+)?(?:with|and)\s+the\s+[A-Za-z_][\w\-]*(?:\s+table)?)", 1),
    # reporting
    ("reporting", "kpis", r"\b(?:dashboard|report|model|chart)\s+(?:of|on|with|listing)\s+(?:(?:last|this)\s+\w+'s\s+)?(.+?)(?=\s+(?:by|for|per|every|each|to|so)\b|[.?,;]|$)", 1),
    ("reporting", "kpis", r"\btrack\s+(.+?)(?=\s+(?:by|for|per|every|each|to)\b|[.?,;]|$)", 1),
    ("reporting", "kpis", r"داشبورد\s+(?:روزانه\s+|هفتگی\s+|ماهانه\s+)?از\s+([^،,.;؛?؟]+?)(?=\s+برای\s|\s+می\s*خواهم|[،,.;؛?؟]|$)", 1),
    ("reporting", "kpis", r"dashboard\s+برای\s+([^،,.;؛?؟]+?)(?=\s+هر\s|\s+می\s*خواهم|[،,.;؛?؟]|$)", 1),
    ("reporting", "audience", r"\bfor\s+(?:the\s+|our\s+|my\s+)?((?:[a-z]+\s+){0,2}(?:team|managers?|leadership|board|executives?|ceo|cfo|department|staff|analysts?))\b", 1),
    ("reporting", "audience", r"برای\s+((?:مدیر|مدیران|تیم|هیئت)(?:\s+[^\s،,.;؛?؟]+)?)", 1),
    ("reporting", "refresh", r"\b(every\s+(?:day|week|month|quarter|morning|monday|tuesday|wednesday|thursday|friday))\b", 1),
    ("reporting", "refresh", rf"\b({_REFRESH_WORDS})\s+(?:dashboard|report|update|refresh|email|summary)\b", 1),
    ("reporting", "refresh", rf"\b(?:updated|refreshed|sent)\s+({_REFRESH_WORDS})\b", 1),
    ("reporting", "refresh", r"(?:داشبورد|گزارش)\s+(روزانه|هفتگی|ماهانه)", 1),
    ("reporting", "refresh", r"(هر\s+(?:روز|هفته|ماه))\s+(?:refresh|به\s*روز)", 1),
    # business
    ("business", "success_measure", r"\bwhy\s+(?:did|has|have|is|are|do|does)\s+(?:our\s+|the\s+)?([a-z_]+(?:\s+[a-z_]+)?)\s+(?:drop|fall|fell|rise|grow|go\s+down|go\s+up|decline|increase|decrease|change|dropped|fallen|risen|grown)", 1),
    ("business", "success_measure", r"\bwhat\s+would\s+it\s+do\s+to\s+(?:our\s+|the\s+)?([a-z_]+(?:\s+[a-z_]+)?)", 1),
    ("business", "success_measure", r"چرا\s+([^\s،,.;؛?؟]+)(?:\s+[^\s،,.;؛?؟]+){0,8}?\s+(?:کم|زیاد|کمتر|بیشتر)\s+(?:شد|شده|می\s*شود)", 1),
    ("business", "decision", r"\b(?:whether\s+)?(?:we|i)\s+should\s+(.+?)(?=[?:.]|$)", 1),
    ("business", "decision", r"\bshould\s+(?:we|i)\s+(.+?)(?=[?:.]|$)", 1),
    ("business", "decision", r"(چه\s*کار\s+(?:باید\s+)?(?:بکنیم|کنیم))", 1),
    ("business", "options", r":\s*([a-z][a-z ]*(?:,\s*[a-z][a-z ]*)*\s+or\s+[a-z][a-z ]*)\s*\?", 1),
    ("business", "period_compared", rf"\b(?:in\s+)?({_MONTH}(?:\s+20\d\d)?\s+compared\s+(?:with|to)\s+{_MONTH}(?:\s+20\d\d)?)", 1),
    ("business", "period_compared", rf"({_FA_PERIOD})", 1),
    ("business", "period_compared", rf"({_EN_PERIOD})", 1),
]
_COMPILED = [(t, f, re.compile(p, re.I), g) for t, f, p, g in PATTERNS]
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
# sealed v6 (s6-30, 2026-10-29): "average discount per order" on a table with no discount column was kept as the measure; a
# measure's plain head word (after average, total, number of ...) that is no part of any column's name is a missing column too
_AGG = re.compile(r"\b(?:average|avg|mean|median|total|sum\s+of|sum|count\s+of|number\s+of|share\s+of|rate\s+of|maximum|minimum|max|min)"
                  r"\s+(?:the\s+)?([a-z][a-z]+)\b")
_GENERIC = {"order", "orders", "value", "values", "revenue", "sales", "sale", "spend", "spending", "count", "number", "rows", "row",
            "customers", "customer", "transactions", "transaction", "size", "amount", "time", "rate", "basket", "ticket", "money",
            "income", "growth", "change", "people", "users", "user", "items", "item", "records", "record", "daily", "weekly",
            "monthly", "yearly", "annual", "per", "each", "of", "all",
            # the review: function words and modifiers after "total", "average", "count of" are never a column
            "by", "for", "in", "on", "across", "and", "or", "this", "that", "last", "next", "how", "distinct", "unique", "different",
            "over", "from", "to", "at", "with", "without", "their", "our", "my", "your", "its", "the", "a", "an", "between", "among",
            "number", "amount", "quantity", "level", "score", "share", "percentage", "percent"}
MEASURE_FIELDS = ("measure", "kpis", "success_measure")


def absent_columns(value: Any, names: Optional[list[str]], field: str = "") -> list[str]:
    """The column names a field's words rely on that the table does not have: snake_case names, and, in a measure, the plain
    word after average, total, number of ... when it is no part of any column's name. Empty when there is no table."""
    if not names:
        return []
    text = str(value or "").lower()
    known = {str(n).lower() for n in names}
    out = [w for w in _SNAKE.findall(text) if w not in known]
    if field in MEASURE_FIELDS:
        parts = {p for n in names for p in re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(n)).lower()) if p}
        for m in _AGG.finditer(text):
            w = m.group(1)
            one = w[:-1] if w.endswith("s") else w
            if len(w) < 4 or w in _GENERIC or one in _GENERIC or w in parts or one in parts:
                continue
            if len(one) >= 4 and any(p.startswith(one) or one.startswith(p) and len(p) >= 4 for p in parts):
                continue
            out.append(w)
    return list(dict.fromkeys(out))
_FIRST = re.compile(r"[^.?!؟]+[.?!؟]?")


def _quotable(text: str, value: str, start: int, end: int) -> str:
    """The person's words that say the value, as a quote the product accepts (whole words, two of them or eight characters):
    the value itself when it is long enough, else grown over its neighbours ("region" is quoted as "per region")."""
    from . import intent, plan

    if plan.quoted(value, [text]):
        return value[:200]
    grown = intent._quote(text, start, end)
    return (grown if plan.quoted(grown, [text]) else value)[:200]


def columns_of(draft: dict[str, Any]) -> list[dict[str, Any]]:
    """The described columns of the draft's prepared table (names and kinds only, never a value)."""
    return [c for c in ((draft.get("analysis") or {}).get("columns") or []) if c.get("name")]


def read(track: str, text: str, columns: Optional[list[str]] = None) -> dict[str, Any]:
    """The fields the person's words give for ``track``: {field: {value, status, source, quote}} quoted from ``text``,
    and the values refused because they name a column the data does not have (``missing``)."""
    text = str(text or "")
    norm, at = lang.normalize_map(text)
    found: dict[str, dict[str, Any]] = {}
    missing: dict[str, list[str]] = {}
    known = {c.lower() for c in columns} if columns else None
    for t, name, pattern, group in _COMPILED:
        if t != track or name in found:
            continue
        match = pattern.search(norm)
        if not match:
            continue
        start, end = lang.span(at, *match.span(group))
        value = text[start:end].strip(" ,.;:!?\"'،؛؟")
        if not value:
            continue
        absent = absent_columns(value, columns, name) if columns else []
        if absent:  # never a field built on a column that is not there (the H1 trap): Home says so instead
            missing[name] = absent
            continue
        found[name] = {"value": value[:200], "status": "stated", "source": "rules", "quote": _quotable(text, value, start, end)}
    if track == "advice" and "topic" not in found:  # an explanation's topic is the question itself, in the person's words
        first = (_FIRST.match(text.strip()) or [""])[0].strip()
        if first and not (known is not None and [w for w in _SNAKE.findall(first.lower()) if w not in known]):
            found["topic"] = {"value": first[:200], "status": "stated", "source": "rules", "quote": first[:200]}
    return {"fields": found, "missing": missing}


def read_each(track: str, texts: list[str], columns: Optional[list[str]] = None) -> dict[str, Any]:
    """``read`` on each of the person's messages apart, a later message's field over an earlier one's (the conversation's
    answers refine the first sentence): a quote never spans two messages (HOMECONV-002: six quotes joined across them)."""
    fields: dict[str, dict[str, Any]] = {}
    missing: dict[str, list[str]] = {}
    for text in texts:
        if not str(text or "").strip():
            continue
        one = read(track, text, columns)
        fields.update(one["fields"])
        for name, absent in one["missing"].items():
            missing[name] = sorted(set(missing.get(name, [])) | set(absent))
            fields.pop(name, None) if name not in one["fields"] else None
    return {"fields": fields, "missing": {k: v for k, v in missing.items() if k not in fields}}


def fill(track: str, read_fields: dict[str, dict[str, Any]], before: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, dict[str, Any]]:
    """The track's brief: what was known before (never replaced by a default or a rule), then what the words gave, then
    each remaining field's default (source "default"), or unknown when it has none."""
    before = dict(before or {})
    out: dict[str, dict[str, Any]] = {}
    for f in TRACK_FIELDS.get(track, ()):
        old = before.get(f.name)
        if old and old.get("source") != "default" and old.get("value") not in (None, ""):
            out[f.name] = old
        elif f.name in read_fields:
            out[f.name] = read_fields[f.name]
        elif old:
            out[f.name] = old
        elif f.default is not None:
            out[f.name] = {"value": f.default, "status": "inferred", "source": "default", "quote": None}
        else:
            out[f.name] = {"value": None, "status": "unknown", "source": None, "quote": None}
    return out


def asks(track: str, brief: dict[str, dict[str, Any]], asked: int, has_columns: bool = True) -> Optional[Field]:
    """The next field worth a question: unknown, with no safe default, never asked before, within the track's budget.
    A field whose buttons come from the table's columns waits for the table (2026-10-29: "What should I measure?" was
    asked before any upload, with "Number of rows" as its only choice)."""
    if asked >= BUDGET.get(track, 0):
        return None
    for f in TRACK_FIELDS.get(track, ()):
        entry = brief.get(f.name) or {}
        if f.options_from and not has_columns and len(f.options) <= 1:  # its buttons would be the table's columns and one fixed choice
            continue
        if f.asked and entry.get("status") == "unknown" and not entry.get("asked"):
            return f
    return None


def options(f: Field, columns: list[dict[str, Any]], profile: Optional[dict[str, Any]] = None, words: str = "") -> list[str]:
    """The question's buttons: the table's columns of the right kind (names only), the field's fixed ones, then Not sure.

    The owner's Telco test (2026-10-29): "Which number tells you it worked?" offered SeniorCitizen (a 0/1 flag) and not Churn.
    With the profile's flags a number is a measure only when it is no key and no small code (a 0/1 flag, a 1-5 rating), and an
    outcome-like yes/no column (its rate) is a measure too, offered first; the columns the person's words name come first of all."""
    kinds = {"numeric": ("numeric",), "categorical": ("categorical", "boolean"), "time": ("datetime",)}.get(f.options_from)
    flags = {str(c.get("name")): c for c in ((profile or {}).get("columns") or [])}
    if f.options_from == "numeric":
        def measure(c: dict[str, Any]) -> bool:
            p = flags.get(str(c["name"]), {})
            if p.get("id_like") or p.get("name_id_like"):
                return False
            if c.get("kind") == "numeric":
                return int(c.get("unique") or 0) > 10 or bool(p.get("target_name_like"))
            return c.get("kind") == "boolean" and bool(p.get("target_name_like"))
        chosen = [c for c in columns if measure(c)]
    elif f.options_from == "categorical":
        chosen = [c for c in columns if c.get("kind") in kinds and 1 < int(c.get("unique") or 0) <= 30
                  and not flags.get(str(c["name"]), {}).get("id_like")]
    else:
        chosen = [c for c in columns if kinds and c.get("kind") in kinds]
    said = re.sub(r"[^a-z0-9]+", " ", str(words or "").lower())
    spoken = lambda c: re.sub(r"[^a-z0-9]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(c["name"])).lower()).strip()  # noqa: E731
    chosen.sort(key=lambda c: (not (spoken(c) and f" {spoken(c)} " in f" {said} "), not flags.get(str(c["name"]), {}).get("target_name_like")))
    named = [c["name"] for c in chosen]
    return list(dict.fromkeys([*named[:5], *f.options, NOT_SURE]))


def answered(track: str, brief: dict[str, dict[str, Any]], name: str, said: str, not_known: bool) -> dict[str, dict[str, Any]]:
    """The person's answer to a brief question: their words, or, when they do not know, the field's default (or still
    unknown, marked asked so it is not asked again). Nothing else in the brief changes."""
    f = field(track, name)
    brief = {k: dict(v) for k, v in brief.items()}
    if f is None:
        return brief
    if not_known or said.strip().lower() == NOT_SURE.lower():
        brief[name] = ({"value": f.default, "status": "inferred", "source": "default", "quote": None, "asked": True} if f.default is not None
                       else {"value": None, "status": "unknown", "source": None, "quote": None, "asked": True})
    else:
        brief[name] = {"value": said.strip()[:200], "status": "stated", "source": "user", "quote": said.strip()[:200], "asked": True}
    return brief


__all__ = ["TRACKS", "Field", "TRACK_FIELDS", "BUDGET", "NOT_SURE", "field", "read", "fill", "asks", "options", "answered", "columns_of"]
