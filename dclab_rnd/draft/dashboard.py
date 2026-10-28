"""A dashboard from a reporting request (package H10 of the Home plan), the studio's side.

``read`` turns the person's words into ``dclab_core.dashboard`` KPIs by rules (English and Persian): the list of numbers
they named ("revenue, orders and refunds"), each as a query plan: a total of a number column ("revenue", "sales": the
amount), an average ("average monthly spend"), a count of rows ("orders", "bookings"), a rate of a yes/no column ("churn
rate": the share of rows where the churn column is 1; "cancellation rate"), or a count of rows holding a value the table has
("refunds": status = refunded, found by code); split by a column they named ("by region", "per channel", "هر destination"),
over time at the grain of the refresh they asked for (weekly: per week) when the table has one date column. A KPI the rules
cannot read is said, never guessed; a list of rows ("every churned customer with their ID") is never a tile.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from dclab_core import dashboard as D
from dclab_core import query as Q

from . import lang

REFRESH = [(r"\b(?:daily|every\s+day|every\s+morning)\b|روزانه|هر\s+روز", "daily", "day"),
           (r"\b(?:weekly|every\s+week|every\s+monday|every\s+friday)\b|هفتگی|هر\s+هفته", "weekly", "week"),
           (r"\b(?:monthly|every\s+month)\b|ماهانه|هر\s+ماه", "monthly", "month")]
LIST = re.compile(r"\b(?:list(?:ing)?|every\s+\w+\s+with\s+(?:their|its)|each\s+\w+\s+with\s+(?:their|its))\b|فهرست|لیست", re.I)
_KPIS = [r"\b(?:dashboard|report|reporting|kpis?|model)\s+(?:of|on|with|showing|tracking)\s+(?:(?:last|this|previous|past)\s+\w+'s\s+)?(.+?)(?=\s+(?:by|per|for\s+the|for\s+our|every|each|to\s+the|so)\b|[.?;]|$)",
         r"\btrack(?:ing)?\s+(.+?)(?=\s+(?:by|per|for\s+the|for\s+our|every|each|to\s+the)\b|[.?;]|$)",
         r"(?:داشبورد|dashboard)\s+(?:روزانه\s+|هفتگی\s+|ماهانه\s+)?(?:از|برای)\s+(.+?)(?=\s+برای\s|\s+هر\s|\s+می\s*خواهم|\s+که\s|[.;؛?؟]|$)"]
# a split over time only when the words ask for one ("per week", "by month", "trend"): "a weekly dashboard" is how often it
# is refreshed, never a split by the month of a column such as signup_date (the rules cannot tell an event date from another)
OVER_TIME = [(r"\b(?:per|by|each)\s+day\b|\bdaily\s+(?:trend|totals?)\b", "day"),  # "هر روز" is a cadence (the review of H10a)
             (r"\b(?:per|by|each)\s+week\b|\bweek\s+by\s+week\b", "week"), (r"\b(?:per|by|each)\s+month\b|\bmonth\s+by\s+month\b", "month"),
             (r"\b(?:trend|over\s+time)\b|روند", "month")]
RELATIVE = re.compile(r"\b(?:last|this|previous|past)\s+(?:day|week|month|quarter|year)(?:'s)?\b", re.I)
_SPLIT = r"(?:\bby|\bper|\beach|\bfor\s+each|هر)\s+([A-Za-z_][\w]*|[\u0600-\u06FF]+)(?:\s*(?:,|\band\b|\s+و)\s*([A-Za-z_][\w]*|[\u0600-\u06FF]+))?"
RATE = re.compile(r"\b(?:rate|share|percentage|percent|ratio)\b|نرخ|درصد|سهم", re.I)
MEAN = re.compile(r"\b(?:average|avg|mean)\b|میانگین|متوسط", re.I)
COUNT_NOUNS = re.compile(r"^(?:orders?|bookings?|transactions?|sales\s+count|visits?|tickets?|customers?|سفارش|رزرو|booking)", re.I)
TOTAL_NOUNS = {"revenue": ("amount", "revenue", "sales"), "sales": ("amount", "sales", "revenue"), "فروش": ("amount", "sales", "revenue"),
               "income": ("amount", "revenue"), "spend": ("spend", "amount"), "gmv": ("amount",)}


def _columns(frame: pd.DataFrame, personal: set[str]) -> list[dict[str, Any]]:
    out = []
    for c in frame.columns:
        s = frame[c]
        kind = "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        if str(c) not in personal and not Q.ID_NAME.search(str(c)):
            out.append({"name": str(c), "kind": kind})
    return out


def _date_column(frame: pd.DataFrame) -> Optional[str]:
    named = [str(c) for c in frame.columns if re.search(r"(?:^|_)date(?:$|_)", str(c).lower()) or pd.api.types.is_datetime64_any_dtype(frame[c])]
    return named[0] if len(named) == 1 else None


def _pieces(phrase: str) -> list[str]:
    parts = re.split(r",|\band\b|\s+و\s+|،", phrase)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _flag(word: str, frame: pd.DataFrame, cols: list[dict[str, Any]]) -> Optional[str]:
    """A yes/no column whose name covers the word ("churn": churned_next_month; "cancellation": cancelled)."""
    from .. import home_query

    stem = home_query._STEM(word)
    hits = [c["name"] for c in cols if c["kind"] == "numeric" and set(pd.unique(frame[c["name"]].dropna())) <= {0, 1, True, False}
            and any(home_query._STEM(t) == stem for t in re.split(r"[_\W]+", c["name"].lower()) if len(t) > 2)]
    whole = [h for h in hits if len([t for t in re.split(r"[_\W]+", h.lower()) if len(t) > 2]) == 1]  # "cancelled", not "free_cancellation"
    return hits[0] if len(hits) == 1 else whole[0] if len(whole) == 1 else None


def _value(word: str, frame: pd.DataFrame, cols: list[dict[str, Any]]) -> Optional[tuple[str, str]]:
    """A text column holding the value the word names: the word itself, its singular, or the singular with "ed"
    ("refunds": refunded), never a shared five-letter stem ("complaints" is not "completed"; the review of H10a)."""
    w = word.lower()
    forms = {w, w.rstrip("s"), w.rstrip("s") + "ed", w.rstrip("s") + "d"}
    hits = []
    for c in cols:
        if c["kind"] != "categorical" or frame[c["name"]].nunique() > 50:
            continue
        for v in pd.unique(frame[c["name"]].dropna().astype(str)):
            if v.lower() in forms:
                hits.append((c["name"], v))
    return hits[0] if len(hits) == 1 else None


_PLAIN = re.compile(r"^(?:the|of|a|an|our|my|total|number|count|all|کل|ها|های|its|their|and|in|for|per|by|trend|trends|over|time|روند)$")  # "trend" is the time split, read apart
NEGATION = re.compile(r"\b(?:not|no|excluding|exclude|except|without|net|other\s+than)\b|n't|بجز|به\s*جز|غیر\s+از|بدون", re.I)


def kpi_of(piece: str, frame: pd.DataFrame, cols: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
    """One KPI phrase as a measure (name, op, column, where), or None with the phrase (said as not read). A phrase with a
    condition the rules do not apply ("excluding refunds", "net revenue", "in region A") is never built without it."""
    from .. import home_query

    norm = lang.normalize(piece).lower()
    if NEGATION.search(norm):
        return None, piece
    words = [w for w in re.findall(r"[A-Za-z_]+|[\u0600-\u06FF]+", norm) if not _PLAIN.match(w)]
    if not words:
        return None, ""
    numeric = [c for c in cols if c["kind"] == "numeric"]
    used: set[str] = set()

    def done(k: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
        rest = [w for w in words if w not in used and not RATE.fullmatch(w) and not MEAN.fullmatch(w)]
        return (k, "") if not rest else (None, piece)  # a word no part of the tile used: never dropped in silence

    if RATE.search(norm):
        for w in words:
            flag = _flag(w, frame, cols)
            hit = None if flag else _value(w, frame, cols)
            if flag or hit:
                used.add(w)
                col, value = (flag, 1) if flag else hit
                return done({"name": f"share of rows where {col} = {value}", "op": "rate", "where": {"column": col, "op": "eq", "value": value}})
        return None, piece
    if MEAN.search(norm):
        rest = " ".join(w for w in words if not MEAN.fullmatch(w))
        col = home_query._column(rest, numeric, ("numeric",))
        used.update(w for w in words if not MEAN.fullmatch(w))
        return done({"name": f"mean of {col}", "op": "mean", "column": col}) if col else (None, piece)
    col = home_query._column(" ".join(words), numeric, ("numeric",))  # a number column first ("refund amount": refund_amount)
    if col:  # "revenue": amount, never discount_amount (one column, covering every word)
        used.update(words)
        return done({"name": f"total of {col}", "op": "sum", "column": col})
    if COUNT_NOUNS.match(" ".join(words)):
        noun = words[0].rstrip("s")
        key = [c for c in frame.columns if re.fullmatch(rf"{re.escape(noun)}_(?:id|key|no|number)", str(c).lower())]
        used.add(words[0])
        if len(key) == 1:  # "customers": the distinct customers, never rows (an order table has many per customer)
            return done({"name": f"distinct {key[0]}", "op": "count_distinct", "column": str(key[0])})
        return done({"name": "rows", "op": "count", "assumed": f"each row counted as one {noun}"})
    for w in words:  # a value the table holds: its rows counted ("refunds")
        hit = _value(w, frame, cols)
        if hit:
            used.add(w)
            return done({"name": f"rows where {hit[0]} = {hit[1]}", "op": "count", "where_filter": {"column": hit[0], "op": "eq", "value": hit[1]}})
    return None, piece


TIME_WORDS = re.compile(r"^(?:day|days|week|weeks|month|months|quarter|year|روز|هفته|ماه|سال)$", re.I)


def read(text: str, frame: pd.DataFrame, columns: Optional[list[dict[str, Any]]] = None,
         kpis_said: Optional[str] = None) -> tuple[Optional[dict[str, Any]], list[str]]:
    """The dashboard the words ask for ({title, audience, refresh, kpis}), or None, and the phrases not read. ``kpis_said``:
    the person's answer to "Which numbers should it show?", read before their first sentence (the review of H10a)."""
    from .. import home_query

    norm = lang.normalize(text)
    if LIST.search(norm):
        return None, ["a list of rows: a dashboard shows numbers per group, never single rows"]
    personal = {str(c.get("name")) for c in columns or [] if str(c.get("sensitivity") or "none") in home_query.SENSITIVE}
    cols = _columns(frame, personal)
    source = lang.normalize(kpis_said) if kpis_said else norm
    phrase = lang.normalize(kpis_said) if kpis_said else next((m.group(1) for p in _KPIS if (m := re.search(p, norm, re.I))), None)
    if kpis_said:  # the answer may carry its own split: "revenue and orders by region"
        phrase = re.split(r"\s+(?:by|per|for\s+each|each)\s+|\s+هر\s+", phrase, maxsplit=1)[0]
    if not phrase:
        return None, ["which numbers to show"]
    unread: list[str] = []
    measures: list[dict[str, Any]] = []
    for piece in _pieces(phrase):
        kpi, why = kpi_of(piece, frame, cols)
        if kpi:
            measures.append(kpi)
            if kpi.get("assumed"):
                unread.append(kpi.pop("assumed"))
        elif why:
            unread.append(why)
    refresh = next((r for p, r, _ in REFRESH if re.search(p, norm + " " + source, re.I)), "on_new_data")
    grain = next((g for p, g in OVER_TIME if re.search(p, source + " " + norm, re.I)), None)
    if RELATIVE.search(norm):  # "last month's sales": a period the rules do not apply yet, said
        unread.append(f"the period “{RELATIVE.search(norm).group(0)}” (the tiles cover every row)")
    groups: list[str] = []
    for m in re.finditer(_SPLIT, source + (" " + norm if kpis_said else ""), re.I):
        for word in [w for w in m.groups() if w]:  # "by region and channel": both
            if TIME_WORDS.match(word):
                continue  # "per week" is the grain, read above
            g = home_query._column(word, [c for c in cols if c["kind"] == "categorical"], ("categorical",))
            if g and g not in groups:
                groups.append(g)
            elif g is None and f"by {word}" not in unread and word.lower() not in ("the", "our"):
                unread.append(f"by {word}")
    groups = groups[:2]  # at most two splits besides time (the engine's three groups)
    date = _date_column(frame)
    if grain and date is None:
        unread.append(f"per {grain} (the table has no single date column)")
    kpis = []
    for k in measures:
        where = k.pop("where_filter", None)
        plan = {"measures": [{**{x: y for x, y in k.items() if x != "name"}, "name": k["name"][:80]}],
                "group_by": groups, "filters": [where] if where else [], "limit": Q.MAX_LIMIT,
                "period": {"column": date, "grain": grain} if grain and date else None}
        try:
            kpis.append(D.Kpi(name=k["name"][:120], plan=Q.QueryPlan.model_validate(plan), said=(kpis_said or text)[:200]))
        except ValueError:
            unread.append(k["name"])
    if not kpis:
        return None, unread or ["which numbers to show"]
    audience = re.search(r"\bfor\s+(?:the\s+|our\s+|my\s+)?((?:[a-z]+\s+){0,2}(?:team|managers?|leadership|board|executives?|ceo|cfo))\b|برای\s+((?:مدیر|مدیران|تیم)(?:\s+[^\s،,.;؛?؟]+)?)", norm, re.I)
    title = ", ".join(k.name for k in kpis) + (f" by {', '.join(groups)}" if groups else "") + (f", per {grain}" if grain and date else "")
    return {"title": title[:200], "audience": (audience.group(1) or audience.group(2)).strip() if audience else None,
            "refresh": refresh, "kpis": kpis}, unread


__all__ = ["read", "kpi_of", "REFRESH"]
