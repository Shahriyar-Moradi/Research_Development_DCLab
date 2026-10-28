"""Preparing a table on request, the studio's side (package H9 of the Home plan).

``read_steps`` turns the person's sentence into ``dclab_core.recipe`` steps by rules (English and Persian), each step with
the words it came from; what the rules cannot read is returned as words, never guessed. A value the person names ("only
the paid orders") is looked up in the table's own columns by code; it never goes to a model. ``destination`` reads the
format asked for (CSV by default; Excel; Parquet).

The table a recipe makes is a new file beside the original (``prepared.parquet``); the original is never written.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from dclab_core import recipe as R

from . import lang

FORMATS = {"csv": "csv", "excel": "xlsx", "xlsx": "xlsx", "اکسل": "xlsx", "parquet": "parquet"}
PREPARED = "prepared.parquet"  # the old single name, read for drafts prepared before the per-table name


def prepared_name(asset_id: str) -> str:
    """The prepared table's file: one per source table, under a name an upload never takes."""
    return f"prepared__{re.sub(r'[^A-Za-z0-9]', '', str(asset_id))[:40]}.parquet"


YES = re.compile(r"\s*(?:yes|yeah|yep|y|ok|okay|sure|apply|apply\s+it|yes,?\s+apply\s+it|do\s+it|go\s+ahead|بله|آره|باشه|اعمال\s+کن|اعمالش\s+کن)\s*[.!]*\s*", re.I)
NO = re.compile(r"\s*(?:no|nope|n|skip|skip\s+it|no,?\s+skip\s+it|don'?t|do\s+not|نه|نکن|اعمال\s+نکن)\s*[.!]*\s*", re.I)
_W = r"[A-Za-z_][\w\-]*"


def destination(text: str) -> str:
    """The file format the person asked for (CSV when they named none)."""
    low = lang.normalize(text).lower()
    return next((fmt for word, fmt in FORMATS.items() if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", low)), "csv")


def _column(words: str, frame: pd.DataFrame, *, dates: bool = False) -> Optional[str]:
    """The one column the words name (the answer's matching rules: every content word covers a part of the name)."""
    from .. import home_query

    kinds = []
    for c in frame.columns:
        s = frame[c]
        kind = "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        kinds.append({"name": str(c), "kind": kind})
    said = home_query._content(words)
    hits = [c["name"] for c in kinds if said and home_query._covers(said, c["name"])]
    if dates:
        hits = [h for h in hits if "date" in h.lower() or kinds[[k["name"] for k in kinds].index(h)]["kind"] == "datetime"] or \
            [c["name"] for c in kinds if "date" in c["name"].lower() and not said]
    return hits[0] if len(hits) == 1 else None


def _value_column(value: str, frame: pd.DataFrame) -> Optional[str]:
    """The one text column that holds the value the person named (read by code from the table, never sent anywhere)."""
    hits = [str(c) for c in frame.columns if frame[c].dtype == object and frame[c].astype(str).str.lower().eq(value.lower()).any()]
    return hits[0] if len(hits) == 1 else None


def _date_column(frame: pd.DataFrame) -> Optional[str]:
    """The one date column: named with the word date (never "updated_at"), or held as dates; None when there are several."""
    named = [str(c) for c in frame.columns if re.search(r"(?:^|_)date(?:$|_)", str(c).lower()) or pd.api.types.is_datetime64_any_dtype(frame[c])]
    return named[0] if len(named) == 1 else None


NEGATION = re.compile(r"\b(?:not|no|never|except|excluding|exclude|without|non|other\s+than|but\s+not)\b|n't|(?<![\u0600-\u06FF])(?:بجز|به\s*جز|غیر\s+از|نه|نباشد|نیست)(?![\u0600-\u06FF])", re.I)
_YEAR = re.compile(r"\b(?:from|in|of|during|for)\s+(?:the\s+year\s+)?(20\d\d)\b|سال\s+(20\d\d)")


def describe(step: R.Step) -> str:
    """What a step does, in words built from the step itself (the guard's question says this, not the person's words)."""
    k = step.kind
    if k == "filter":
        ops = {"eq": "=", "ne": "≠", "gt": ">", "ge": "≥", "lt": "<", "le": "≤", "in": "in", "not_in": "not in"}
        parts, fs = [], list(step.filters)
        for f in fs:
            pair = next((g for g in fs if g is not f and g.column == f.column and {f.op, g.op} == {"ge", "le"}), None)
            if pair is not None and f.op == "ge":  # a range on one column, said as one: "order_date from 2026-01-01 to 2026-12-31"
                parts.append(f"{f.column} from {str(f.value)[:10]} to {str(pair.value)[:10]}")
            elif pair is None:
                parts.append(f"{f.column} {ops.get(f.op, f.op)} {str(f.value)[:40]}")
        return "keep only the rows where " + " and ".join(parts)
    if k == "dedupe":
        return f"remove rows with the same {', '.join(step.keys)}" if step.keys else "remove rows that are exact copies"
    if k == "drop_missing":
        return f"remove rows with no {', '.join(step.columns)}"
    if k == "to_date":
        return f"read {', '.join(step.columns)} as dates"
    if k == "months_since":
        return f"add {step.to}: whole months from {step.columns[0]} to {step.at or 'its latest date'}"
    if k == "join":
        return f"add {', '.join(step.columns) or 'the columns'} from {step.table}, matched on {', '.join(step.keys)}"
    if k == "pivot":
        return f"one row per {step.index}, one column per {step.spread}, the {step.agg} of {step.values}"
    if k == "rename":
        return f"rename {step.columns[0]} to {step.to}"
    if k == "sort":
        return f"sort by {', '.join(step.columns)}" + (" (largest first)" if step.desc else "")
    return f"drop {', '.join(step.columns)}"


def read_steps(text: str, frame: pd.DataFrame, tables: Optional[dict[str, pd.DataFrame]] = None) -> tuple[list[R.Step], list[str]]:
    """The steps the person's sentence asks for, in the order said, and the parts the rules could not read."""
    norm = lang.normalize(text)
    low = norm.lower()
    steps: list[tuple[int, R.Step]] = []
    unread: list[str] = []

    def add(pos: int, why: str, **kw: Any) -> None:
        try:
            steps.append((pos, R.Step(why=why[:400], **kw)))
        except ValueError:
            unread.append(why)

    # duplicates: on the id column of the thing named ("duplicate orders": order_id), else every column
    for m in re.finditer(rf"\b(?:remove|drop|delete)\s+(?:the\s+)?(?:duplicate|duplicated)\s+({_W})|\bdedupe?\w*\s+(?:the\s+)?({_W})?|(?:ردیف|سطر)\S*\s+تکراری\S*", low):
        thing = m.group(1) or m.group(2) or ""
        key = [str(c) for c in frame.columns if thing and re.fullmatch(rf"{re.escape(thing.rstrip('s'))}s?_(?:id|key|no|number)", str(c).lower())]
        add(m.start(), m.group(0), kind="dedupe", keys=key[:1])
    # dates: "convert <column> to a date", "the date column ... right format", Persian "ستون تاریخ ... تبدیل"
    for m in re.finditer(rf"\b(?:convert|parse|turn|fix)\s+(?:the\s+)?({_W}(?:\s+column)?)\s+(?:to|into|as)\s+(?:a\s+|the\s+)?(?:proper\s+|right\s+|correct\s+)?(?:date|datetime)s?\b|ستون\s+تاریخ\S*[^.؟!،]*?(?:تبدیل|درست)", low):
        col = _column(m.group(1) or "", frame, dates=True) if m.group(1) else _date_column(frame)
        if col:
            add(m.start(), m.group(0), kind="to_date", columns=[col])
        else:
            unread.append(m.group(0))
    # rows missing a value: "rows without an amount", "rows that have no amount", Persian "ردیف‌هایی که مبلغ ندارند حذف"
    for m in re.finditer(rf"\b(?:remove|drop|delete)\s+(?:the\s+)?rows\s+(?:without|with\s+no|that\s+have\s+no|missing)\s+(?:an?\s+|the\s+)?({_W})|ردیف\S*(?:\s+ها\S*)?\s+(?:که|بدون)\s+(\S+)\s+(?:ندارند|ندارد|خالی)", low):  # "ردیف‌هایی": the joiner reads as a space
        col = _column(m.group(1) or m.group(2), frame)
        if col:
            add(m.start(), m.group(0), kind="drop_missing", columns=[col])
        else:
            unread.append(m.group(0))
    # pivot: each column by its role words, never by the order they are said (the review of H9a): "per/each A" or "هر A یک ردیف"
    # the rows, "by B" or "هر B یک ستون" the columns, "sum/average/count of C" or "جمع C" the values; a sum or mean of a number only
    m = re.search(r"pivot", low)
    if m:
        names = {str(c).lower(): str(c) for c in frame.columns}
        name = rf"({'|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))})" if names else r"(?!x)x"
        rows = re.search(rf"(?:\b(?:per|each|for\s+each|one\s+row\s+(?:per|for\s+each))\s+|هر\s+){name}(?:\s+یک\s+ردیف|\b)", low)
        cols = re.search(rf"(?:\bby\s+|\bone\s+column\s+(?:per|for\s+each)\s+|هر\s+){name}(?:\s+یک\s+ستون|\b)", low[rows.end():] if rows else low)
        vals = re.search(rf"(?:\b(?:sum|total|average|mean|count)\s+of\s+(?:the\s+)?|(?:جمع|مجموع|میانگین|تعداد)\s+){name}", low)
        agg = "mean" if re.search(r"\b(?:average|mean)\b|میانگین", low) else "count" if re.search(r"\bcount\b|تعداد", low) else "sum"
        index, spread, values = (names[x.group(1)] if x else None for x in (rows, cols, vals))
        numeric = values is not None and pd.api.types.is_numeric_dtype(frame[values])
        if index and spread and values and len({index, spread, values}) == 3 and (numeric or agg == "count"):
            add(m.start(), norm[m.start():m.start() + 200], kind="pivot", index=index, spread=spread, values=values, agg=agg)
        else:
            unread.append(norm[m.start():m.start() + 120])
    # keep only: a value found in one column and a year on the one date column; a negation or a year with no single date
    # column leaves the whole phrase unread, never a partial or opposite filter (the review of H9a)
    for m in re.finditer(rf"\bkeep\s+only\s+(?:the\s+)?([^.;?]+?)(?=\s+and\s+(?:give|export|save|send|show|download)\b|[.;?]|$)", low):
        words = m.group(1)
        if NEGATION.search(words):
            unread.append(m.group(0))
            continue
        filters, whole = [], True
        years = [a or b for a, b in _YEAR.findall(words)]
        if years:
            col = _date_column(frame)
            if col is None or len(set(years)) > 1:
                unread.append(m.group(0))
                continue
            filters += [{"column": col, "op": "ge", "value": f"{years[0]}-01-01"}, {"column": col, "op": "le", "value": f"{years[0]}-12-31 23:59:59"}]
        rest = [w for w in re.findall(_W, _YEAR.sub(" ", words)) if w not in ("from", "in", "of", "the", "only", "rows", "orders", "records", "that", "are", "is", "with")]
        for w in rest:
            col = _value_column(w, frame)
            if col:
                filters.append({"column": col, "op": "eq", "value": str(frame[col][frame[col].astype(str).str.lower() == w].iloc[0])})
            else:
                whole = False
        if filters and whole:
            if any(not pd.api.types.is_datetime64_any_dtype(frame[f["column"]]) and f["op"] in ("ge", "le") for f in filters):
                add(m.start() - 1, "the dates read as dates", kind="to_date", columns=sorted({f["column"] for f in filters if f["op"] in ("ge", "le")}))
            add(m.start(), m.group(0), kind="filter", filters=filters)
        else:
            unread.append(m.group(0))
    # months since: "turn <date column> into months since <...>"
    m = re.search(rf"\b(?:turn|convert)\s+({_W})\s+into\s+(?:the\s+)?months\s+since\s+({_W})?", low)
    if m:
        col = _column(m.group(1), frame, dates=True) or (m.group(1) if m.group(1) in map(str, frame.columns) else None)
        if col:  # counted to the column's latest date, stated in the step (reproducible, and said)
            latest = pd.to_datetime(frame[col], errors="coerce").max()
            add(m.start(), m.group(0), kind="months_since", columns=[col], to=f"months_since_{m.group(2) or col}",
                at=None if pd.isna(latest) else latest.date().isoformat())
        else:
            unread.append(m.group(0))
    # join: "join ... on <key>" / "combine ... on <key>" with another table that has the key
    m = re.search(rf"\b(?:join|merge|combine)\b[^.;?]*?\bon\s+({_W})", low)
    if m and tables:
        key = m.group(1)
        other = next((name for name, t in tables.items() if key in t.columns), None)
        if key in frame.columns and other:
            cols = next((found for b in re.finditer(rf"\b(?:has|gets|with)\s+the\s+\w+'?s?\s+({_W})(?:\s+and\s+({_W}))?", low)
                         if (found := [c for c in b.groups() if c and c in tables[other].columns])), [])
            add(m.start(), m.group(0), kind="join", table=other, keys=[key], columns=cols)
        else:
            unread.append(m.group(0))
    return [s for _, s in sorted(steps, key=lambda x: x[0])], unread


def safe_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """For a spreadsheet: a text cell starting = + - @ would open as a live formula, so it is kept as text (a leading ')."""
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return out


JOIN = re.compile(rf"\b(?:join|merge|combine)\b[^.;?]*?\bon\s+({_W})|ادغام|ترکیب", re.I)  # a join with its key, not any "combine"


def main_table(text: str, frames: dict[str, pd.DataFrame]) -> Optional[str]:
    """The table a join adds to: the one where the key repeats (each order has one customer), never the upload order."""
    m = JOIN.search(lang.normalize(text).lower())
    key = m.group(1) if m and m.group(1) else None
    if key is None:
        return None
    many = [name for name, t in frames.items() if key in t.columns and t[key].duplicated().any()]
    return many[0] if len(many) == 1 else None


__all__ = ["FORMATS", "PREPARED", "JOIN", "NEGATION", "describe", "destination", "main_table", "prepared_name", "read_steps", "safe_cells"]
