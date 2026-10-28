"""A query plan from what Home understood, without a model (package H7).

The analysis brief (``draft/brief.py``) holds the person's words for the measure, the breakdown, the period and the
filters. Code reads the table to turn them into a plan: words matched to column names, an aggregate word to an
operation, a period phrase to dates, a filter word to a category value that is in the table. Everything it had to
assume is listed in ``notes`` so the answer can say it ("I took amount as revenue"; "last quarter in the data").

A model is not needed for this; when the words do not lead to a plan, the caller may ask one (``propose``).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..draft import lang
from .validate import kinds

_STEM = 5
# words for an amount of money, compared as the first letters of a word, as the column names are
MONEY = {w[:_STEM] for w in ("revenue", "sales", "spend", "money", "income", "value", "amount", "price", "fare", "cost", "paid", "total",
                             "turnover", "درآمد", "فروش", "مبلغ", "هزینه", "قیمت")}
OPS = [  # (operation, words): the first that the measure's words hold
    ("mean", ("average", "mean", "avg", "میانگین", "متوسط")),
    ("median", ("median", "میانه")),
    ("sum", ("total", "sum", "how much", "مجموع", "جمع", "چقدر")),
    ("count", ("number", "count", "how many", "تعداد", "چند تا", "چند")),
    ("rate", ("share", "percentage", "rate", "proportion", "درصد", "نرخ")),
]
GRAIN_WORDS = {"day": ("day", "daily", "روز"), "week": ("week", "weekly", "هفته"), "month": ("month", "monthly", "ماه"),
               "quarter": ("quarter", "quarterly", "فصل"), "year": ("year", "yearly", "annual", "سال")}
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W_]+", lang.fold(text or "")) if len(w) > 1]


def _stems(text: str) -> set[str]:
    return {w[:_STEM] for w in _words(text)}


def _column(text: str, columns: list[str]) -> Optional[str]:
    """The column whose name shares the most words with the text (whole names first); Persian words count by their
    English column words (``lang.english_words``)."""
    text = " ".join([text or "", *lang.english_words(text or "")])
    folded = lang.fold(text)
    whole = [c for c in columns if c.lower() in folded or c.lower().replace("_", " ") in folded]
    if whole:
        return max(whole, key=len)
    want = _stems(text)
    scored = sorted(((len(want & _stems(c.replace("_", " "))), c) for c in columns), reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _op(text: str) -> str:
    folded = lang.fold(text or "")
    for op, words in OPS:
        if any(re.search(rf"(?<![^\W_]){re.escape(w)}(?![^\W_])", folded) for w in words):
            return op
    return "sum"


def _containing(day, unit: str):
    """The first and last day of the quarter, month or year a day falls in."""
    import pandas as pd

    if unit == "quarter":
        s = pd.Timestamp(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
        return s, s + pd.offsets.QuarterEnd(0)
    if unit == "month":
        s = pd.Timestamp(day.year, day.month, 1)
        return s, s + pd.offsets.MonthEnd(0)
    return pd.Timestamp(day.year, 1, 1), pd.Timestamp(day.year, 12, 31)


def _period(text: str, latest) -> Optional[tuple[str, str, str]]:
    """(from, to, how it was read) for a period phrase, relative to the table's latest date, or None."""
    import pandas as pd

    t = lang.fold(text or "")
    year = re.search(r"\b(19\d\d|20\d\d)\b", t)
    month = next((i + 1 for i, m in enumerate(MONTHS) if re.search(rf"\b{m}\b", t)), None)
    if month and year:
        start = pd.Timestamp(int(year.group(1)), month, 1)
        return start.date().isoformat(), (start + pd.offsets.MonthEnd(0)).date().isoformat(), f"{MONTHS[month - 1].capitalize()} {year.group(1)}"
    if year:
        return f"{year.group(1)}-01-01", f"{year.group(1)}-12-31", year.group(1)
    if latest is None:
        return None
    end = pd.Timestamp(latest).normalize()
    if month:
        y = end.year if month <= end.month else end.year - 1
        start = pd.Timestamp(y, month, 1)
        return start.date().isoformat(), (start + pd.offsets.MonthEnd(0)).date().isoformat(), f"{MONTHS[month - 1].capitalize()} {y} (the latest in the data)"
    days = re.search(r"\blast\s+(\d+)\s+days\b", t)
    if days:
        return (end - pd.Timedelta(days=int(days.group(1)) - 1)).date().isoformat(), end.date().isoformat(), f"the last {days.group(1)} days of the data"
    for unit in ("quarter", "month", "year"):
        if re.search(rf"\b(?:last|previous|past)\s+{unit}\b", t):
            s, e = _containing(end, unit)
            if e != end:  # the data's latest period is not complete: the last full one is the one before
                s, e = _containing(s - pd.Timedelta(days=1), unit)
            return s.date().isoformat(), e.date().isoformat(), f"the last full {unit} in the data ({s.date()} to {e.date()})"
        if re.search(rf"\bthis\s+{unit}\b", t):
            s, _ = _containing(end, unit)
            return s.date().isoformat(), end.date().isoformat(), f"this {unit} so far in the data ({s.date()} to {end.date()})"
    if re.search(r"امسال", t):
        return f"{end.year}-01-01", end.date().isoformat(), f"{end.year} so far in the data"
    return None


def plan_from_brief(entries: dict[str, Any], frame) -> tuple[Optional[dict[str, Any]], list[str]]:
    """A plan for the analysis brief's fields on this table, and what was assumed; (None, reasons) when the words do
    not lead to one."""
    import pandas as pd

    k = kinds(frame)
    value = {f: (e or {}).get("value") for f, e in (entries or {}).items() if (e or {}).get("source") in ("problem", "answer")}
    numeric = [c for c, kind in k.items() if kind == "numeric"]
    groups = [c for c, kind in k.items() if kind in ("categorical", "boolean")]
    dates = [c for c, kind in k.items() if kind == "datetime"]
    notes: list[str] = []
    measure_text = value.get("measure") or ""
    if not measure_text:
        return None, ["the measure is not known yet"]
    op = _op(measure_text)
    column = _column(measure_text, numeric)
    if op in ("sum", "mean", "median") and column is None:
        money = [c for c in numeric if _stems(c.replace("_", " ")) & MONEY]
        if len(money) == 1 and _stems(measure_text) & MONEY:
            column = money[0]
            notes.append(f"I took {column} as the {measure_text}.")
        else:
            return None, [f"no number column matches “{measure_text}”"]
    if op == "count" and column is not None and _stems(measure_text) & _stems(column.replace("_", " ")) and k.get(column) == "numeric":
        op = "sum" if not re.search(r"how many|number of|تعداد", lang.fold(measure_text)) else "count"
    if op == "rate":
        return None, ["a share needs the condition it counts, which I ask for"]
    name = re.sub(r"[^A-Za-z0-9_]+", "_", lang.fold(measure_text))[:40].strip("_") or op
    if not re.match(r"^[A-Za-z0-9_]", name) or not name.isascii():
        name = f"{op}_{column}" if column else op
    measure = {"name": name, "op": op} | ({"column": column} if op != "count" else {})
    plan: dict[str, Any] = {"measures": [measure], "filters": [], "group_by": []}
    breakdown = value.get("breakdown")
    if breakdown:
        grain = next((g for g, words in GRAIN_WORDS.items() if any(w == x for x in _words(breakdown) for w in words)), None)
        group = _column(breakdown, groups)
        if group:
            plan["group_by"] = [group]
        elif grain and dates:
            plan["period"] = {"column": dates[0], "grain": grain}
        else:
            notes.append(f"I found no column for “{breakdown}”, so the answer is not split.")
    period_text = value.get("period")
    if period_text and dates:
        latest = pd.to_datetime(frame[dates[0]], errors="coerce").max()
        read = _period(period_text, None if pd.isna(latest) else latest)
        if read:
            plan.setdefault("period", {"column": dates[0]})
            plan["period"].update({"from": read[0], "to": read[1]})
            notes.append(f"Period: {read[2]}.")
        else:
            notes.append(f"I could not read the period “{period_text}”, so every date counts.")
    elif period_text:
        notes.append("The table has no date column, so the period could not be applied.")
    filters_text = value.get("filters")
    if filters_text:
        wanted = set(_words(filters_text))
        for col in groups:
            values = [str(v) for v in frame[col].dropna().unique()[:200]]
            hit = [v for v in values if lang.fold(v) in wanted or lang.fold(v).rstrip("s") in {w.rstrip("s") for w in wanted}]
            if hit:
                plan["filters"].append({"column": col, "op": "in" if len(hit) > 1 else "eq", "value": hit if len(hit) > 1 else hit[0]})
        if not plan["filters"]:
            notes.append(f"I could not match “{filters_text}” to the table, so no rows are left out.")
    if plan["group_by"]:
        plan["sort"] = [{"by": measure["name"], "desc": True}]
    elif (plan.get("period") or {}).get("grain"):
        plan["sort"] = [{"by": "period", "desc": False}]
    return plan, notes
