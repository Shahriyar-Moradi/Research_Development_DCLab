"""Answering questions from a draft's own table (package H6 of the Home plan): the studio's side of dclab_core.query
(dclab_rnd/analysis.py is another thing: the research registry's cross-dataset evidence).

``table`` reads the draft's cleaned table only when its SHA-256 matches the one the pipeline recorded; ``write_plan`` asks
the ``home_query`` purpose for a plan from column summaries alone (never a row or a cell value) and keeps it only when
it reads as a plan; ``run`` executes a plan with the row floor (DCLAB_ROW_FLOOR, default 5) and appends what it related
to the draft's exposure log. The answer's words (H7) come later; nothing here shows a row.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from dclab_core import query as Q
from dclab_rnd import prompts

PROMPT = "home_query"
SUMMARY = ("name", "kind", "missing_rate", "unique")  # what the model sees of each column: never a value


def floor() -> int:
    try:
        return max(1, int(os.environ.get("DCLAB_ROW_FLOOR", Q.FLOOR)))
    except ValueError:
        return Q.FLOOR


def table(store: Any, draft: dict[str, Any]) -> pd.DataFrame:
    """The draft's cleaned table, checked against the hash the pipeline recorded (a changed file is refused)."""
    from .audit import sha256_of

    asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is None:
        raise ValueError("there is no prepared table yet: upload or connect one")
    path = Path(store.data_dir(draft["id"])) / str(asset.get("clean_file") or "clean.parquet")
    if not path.is_file():
        raise ValueError("the prepared table is not there; prepare it again")
    found = sha256_of(path)
    if asset.get("clean_sha256") and found != asset["clean_sha256"]:
        raise ValueError("the prepared table changed since it was checked; prepare it again")
    if not asset.get("clean_sha256") and found:  # a table prepared before H6 (or a version's rewrite): its hash from now on
        store.update(draft["id"], lambda d: [a.update(clean_sha256=found) for a in d.get("assets") or [] if a.get("id") == asset["id"]])
    return pd.read_parquet(path)


def summaries(draft: dict[str, Any]) -> list[dict[str, Any]]:
    return [{k: c.get(k) for k in SUMMARY} for c in ((draft.get("analysis") or {}).get("columns") or []) if c.get("name")]


def write_plan(client: Any, question: str, draft: dict[str, Any], brief: Optional[dict[str, Any]] = None) -> tuple[Optional[Q.QueryPlan], str]:
    """The model's plan for the question, or None with the reason (no model, a refusal, an unreadable answer)."""
    if client is None:
        return None, "no model serves home_query here"
    user = json.dumps({"question": str(question)[:2000], "brief": {k: (v or {}).get("value") for k, v in (brief or {}).items()},
                       "columns": summaries(draft)}, ensure_ascii=False)
    try:
        reply = client.complete([{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": user}], None)
    except Exception as error:  # noqa: BLE001 - no plan, said why
        return None, f"the model did not answer ({type(error).__name__})"
    text = reply.get("content") if isinstance(reply, dict) else str(reply)
    match = re.search(r"\{.*\}", str(text or ""), re.S)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except ValueError:
        answer = {}
    if answer.get("refuse"):
        return None, str(answer["refuse"])[:300]
    if not answer.get("measures"):
        return None, "the answer was not a plan (no measure)"
    try:
        return Q.QueryPlan.model_validate({k: v for k, v in answer.items() if k in Q.QueryPlan.model_fields and k != "schema_version"}), ""
    except Exception as error:  # noqa: BLE001 - an answer that is not a plan
        return None, f"the answer was not a plan ({str(error).splitlines()[0][:160]})"


OPS = [(re.compile(r"\b(?:average|avg|mean)\b|میانگین|متوسط", re.I), "mean"), (re.compile(r"\bmedian\b|میانه", re.I), "median"),
       (re.compile(r"\b(?:total|sum)\b|مجموع|جمع", re.I), "sum"), (re.compile(r"\b(?:max|maximum|highest|largest)\b|بیشترین", re.I), "max"),
       (re.compile(r"\b(?:min|minimum|lowest|smallest)\b|کمترین", re.I), "min"),
       (re.compile(r"\b(?:number\s+of|count\s+of|count|how\s+many)\b|تعداد|چند\s*تا", re.I), "count")]
SHARE = re.compile(r"\b(?:share|percentage|percent|proportion|rate|ratio|fraction)\b|درصد|نسبت|سهم", re.I)  # needs its condition: not by rules
_ROWS = re.compile(r"\b(?:rows?|records?|entries|lines)\b|ردیف|رکورد", re.I)
_WORD = re.compile(r"[a-z0-9]+|[؀-ۿ]+", re.I)
_STEM = lambda w: w.lower()[:5]  # noqa: E731  "orders" and "order", "spending" and "spend" meet at five letters
_FILLER = re.compile(r"^(?:the|a|an|of|per|by|in|for|our|my|each|all|every|and|was|were|is|are|did|do|does|what|how|much|many|"
                     r"average|avg|mean|median|total|sum|max|maximum|min|minimum|highest|lowest|largest|smallest|number|count|"
                     r"میانگین|متوسط|میانه|مجموع|جمع|بیشترین|کمترین|تعداد|ها|های|را|از|به|در|و|هر)$", re.I)
_LOOSE = re.compile(r"^(?:type|types|kind|kinds|نوع|انواع)$", re.I)  # dropped only on a second try
# a breakdown that is a time grain, not a column: the period's grain on the one date column
GRAINS = {"day": "day", "daily": "day", "روز": "day", "روزانه": "day", "week": "week", "weekly": "week", "هفته": "week",
          "هفتگی": "week", "month": "month", "monthly": "month", "ماه": "month", "ماهانه": "month", "quarter": "quarter",
          "quarterly": "quarter", "فصل": "quarter", "year": "year", "yearly": "year", "annual": "year", "سال": "year", "سالانه": "year"}
SENSITIVE = ("pii", "restricted", "personal", "sensitive")  # analyze's sensitivity classes never grouped, filtered or measured

# the person's word -> the words a column for it is usually named with (English and Persian); every content word must match
SYNONYMS = {"revenue": ("revenue", "sales", "amount"), "sales": ("sales", "revenue", "amount"), "spend": ("spend",), "price": ("price",),
            "cost": ("cost", "spend"), "هزینه": ("spend", "cost"), "فروش": ("sales", "revenue", "amount"), "درآمد": ("revenue",),
            "قیمت": ("price",), "مبلغ": ("amount",), "پلن": ("plan",), "شهر": ("city",), "منطقه": ("region",), "کانال": ("channel",),
            "ماهانه": ("monthly",), "ماه": ("month",), "مشتری": ("customer",), "سفارش": ("order",), "تیکت": ("ticket",)}
OP_WORDS = {"mean": "mean", "median": "median", "sum": "total", "max": "maximum", "min": "minimum", "count": "rows",
            "count_distinct": "distinct", "rate": "share", "percentile": "percentile"}


def _content(words: str) -> list[str]:
    return [w for w in _WORD.findall(lang_digits(str(words or "")).lower()) if len(w) > 2 and not _FILLER.match(w)]


def _matches(word: str, name_tokens: set[str]) -> bool:
    alts = next((a for key, a in SYNONYMS.items() if word == key or _STEM(word) == _STEM(key)), ())
    return _STEM(word) in name_tokens or any(_STEM(a) in name_tokens for a in alts)


def _column(words: str, columns: list[dict[str, Any]], kinds: tuple[str, ...]) -> Optional[str]:
    """The one column, among the given kinds and never a sensitive one, whose name covers every content word the person
    used (a synonym counting as the word); none when no column does, or when two do (the review of H7a: one shared stem
    picked a different quantity, "revenue" summing a discount)."""
    said = _content(words)
    for words_tried in (said, [w for w in said if not _LOOSE.match(w)]):  # "plan type" is "plan" only when no column says "type"
        if not words_tried:
            return None
        hits = [c["name"] for c in columns if c.get("kind") in kinds and str(c.get("sensitivity") or "none") not in SENSITIVE
                and _covers(words_tried, str(c["name"]))]
        if hits:
            return hits[0] if len(hits) == 1 else None
    return None


def _covers(said: list[str], name: str) -> bool:
    """Every word said matches a part of the name; and when one matched only through a synonym, the name holds no part the
    person did not say ("revenue" is "amount", never "discount_amount")."""
    tokens = {_STEM(t) for t in re.split(r"[_\W]+", name.lower()) if len(t) > 2}
    if not all(_matches(w, tokens) for w in said):
        return False
    if all(_STEM(w) in tokens for w in said):
        return True
    named = {_STEM(w) for w in said} | {_STEM(a) for w in said for key, alts in SYNONYMS.items()
                                        if w == key or _STEM(w) == _STEM(key) for a in alts}
    return tokens <= named


def plan_from_brief(brief: dict[str, Any], columns: list[dict[str, Any]]) -> tuple[Optional[Q.QueryPlan], str]:
    """H7: the rules' plan from what the person said (the brief's measure, breakdown and period) and the column names, or
    None with what is missing (then the model, or Home asks). With a plan, the second value lists what of the question
    the rules could not apply, which the answer says. No model, no value: the plan names columns only."""
    value = lambda f: str(((brief or {}).get(f) or {}).get("value") or "") if ((brief or {}).get(f) or {}).get("source") != "default" else ""  # noqa: E731
    measure = value("measure")
    if not measure:
        return None, "what to measure is not known yet"
    if SHARE.search(measure):
        return None, f"a share (“{measure}”) needs the condition it counts, which I can't read from words yet"
    op = next((o for rx, o in OPS if rx.search(measure)), None)
    words = _content(measure)
    if op == "count":
        if not words or _ROWS.search(measure):
            measures = [{"name": "rows", "op": "count"}]
        else:  # "how many customers": the distinct values of the column that names them, never rows called customers
            column = _column(measure, columns, ("categorical", "numeric", "boolean", "text"))
            if column is None:
                return None, f"no column names the “{' '.join(words)}” to count; say which column holds them"
            measures = [{"name": f"distinct {column}", "op": "count_distinct", "column": column}]
    else:
        column = _column(measure, columns, ("numeric",))
        if column is None:
            return None, f"no one numeric column matches “{measure}”; say which column to use"
        op = op or "sum"
        measures = [{"name": f"{OP_WORDS[op]} {column}", "op": op, "column": column}]
    group_by = []
    when = [c["name"] for c in columns if c.get("kind") == "datetime"]
    split = _content(value("breakdown"))
    grain = GRAINS.get(split[0]) if len(split) == 1 and len(when) == 1 else None
    if value("breakdown") and grain is None:
        group = _column(value("breakdown"), columns, ("categorical", "boolean"))
        if group is None:
            return None, f"no one column matches the breakdown “{value('breakdown')}”; say which column to split by"
        group_by.append(group)
    period = {"column": when[0], "grain": grain} if grain else None
    unused: list[str] = []
    said = lang_digits(value("period")).strip().lower()
    year = re.fullmatch(r"(?:(?:in|for|during|over)\s+)?(?:the\s+year\s+)?(?:سال\s+)?(20\d\d)", said)  # a bare year only, never widened
    if said and year and len(when) == 1:
        period = {"column": when[0], "grain": grain or "year", "start": f"{year.group(1)}-01-01", "end": f"{year.group(1)}-12-31"}
    elif said:
        unused.append(f"the period “{value('period')}”")
    if value("filters"):
        unused.append(f"the filter “{value('filters')}”")
    try:
        order = [{"by": period["column"], "desc": False}] if period and grain else [{"by": measures[0]["name"], "desc": True}]  # time reads in order
        return Q.QueryPlan(measures=measures, group_by=group_by, period=period, sort=order), \
            " and ".join(unused)
    except Exception as error:  # noqa: BLE001
        return None, f"the plan could not be made ({type(error).__name__})"


def sensitive_columns(plan: Q.QueryPlan, columns: list[dict[str, Any]]) -> list[str]:
    """The plan's columns the studio marks as personal (a model's plan is checked too)."""
    marked = {c["name"] for c in columns if str(c.get("sensitivity") or "none") in SENSITIVE}
    used = {*plan.group_by, *(f.column for f in plan.filters), *(m.column for m in plan.measures if m.column),
            *(m.where.column for m in plan.measures if m.where), *([plan.period.column] if plan.period else [])}
    return sorted(used & marked)


def lang_digits(text: str) -> str:
    from .draft import lang

    return lang.normalize(text)


def shown(v: Any) -> str:
    """How code writes a result's number in an answer: two decimals at most, thousands separated; a withheld value says so."""
    if v is None:
        return "withheld (too few rows)"
    return f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)


def shown_numbers(result: Q.QueryResult) -> list[str]:
    """Every number as the answer shows it, computed by code from the result (kept beside the answer, so each shown number
    is traced to the result it was rounded from)."""
    return [shown(v) for row in result.rows for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)]


def describe(plan: Q.QueryPlan) -> str:
    """What the plan computed, in words built from the plan itself (never the person's or a model's label): the
    operation and column, the filters, the period and its date column."""
    m = plan.measures[0]
    what = {"count": "number of rows", "count_distinct": f"number of distinct {m.column}", "rate": f"share of rows where {m.where.column if m.where else ''}"
            }.get(m.op, f"{OP_WORDS.get(m.op, m.op)} of {m.column}")
    if plan.group_by:
        what += " by " + ", ".join(plan.group_by)
    if plan.filters:
        what += ", where " + " and ".join(f"{f.column} {f.op.replace('_', ' ')} {f.value}" for f in plan.filters)
    if plan.period:
        span = f" from {plan.period.start}" if plan.period.start else ""
        span += f" to {plan.period.end}" if plan.period.end else ""
        what += f", per {plan.period.grain} of {plan.period.column}{span}"
    return what[0].upper() + what[1:]


def answer_text(plan: Q.QueryPlan, result: Q.QueryResult, note: str = "") -> str:
    """The answer in words, written by code from the plan and the result alone: what was computed (operation, column,
    filters, period), then the numbers; ``note`` says what of the question was not applied."""
    keys = len(result.columns) - len(plan.measures)
    head = describe(plan)
    if not result.rows:
        text = f"{head}: no group had enough rows to show."
    elif keys == 0:
        text = f"{head}: {shown(result.rows[0][-len(plan.measures)])}."
    else:
        parts = [f"{' '.join(str(k) for k in row[:keys])} {shown(row[keys])}" for row in result.rows[:12]]
        more = f", and {len(result.rows) - 12} more" if len(result.rows) > 12 else ""
        text = f"{head}: " + "; ".join(parts) + more + "."
    if note:
        text += f" I could not apply {note} yet, so it is not in this answer; tell me how to apply it."
    if result.suppressed_groups:
        text += f" {result.suppressed_groups} small group{'s' if result.suppressed_groups != 1 else ''} hidden (fewer than {floor()} rows)."
    return text


def run(store: Any, draft_id: str, plan: Q.QueryPlan) -> Q.QueryResult:
    """Execute a plan on the draft's checked table (refused plans raise ValueError with the reasons) and log what it related."""
    from .draft.store import now

    draft = store.get(draft_id)
    result = Q.execute(plan, table(store, draft), floor=floor())
    entry = {**Q.exposure(plan), "at": now()}
    store.update(draft_id, lambda d: d.setdefault("exposure", []).append(entry))
    return result


__all__ = ["floor", "table", "summaries", "write_plan", "run"]
