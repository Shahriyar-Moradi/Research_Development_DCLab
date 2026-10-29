"""Business questions on Home: where a number changed between two periods, said as a decision brief (package H11).

``read`` turns the person's question into what ``dclab_core.change`` needs, by rules (English and Persian): the two periods
("March 2026 compared with February", "Q1 2026", "سه ماه اول ۲۰۲۶": a quarter against the one before), the measure (a
column, or a count of rows), the one date column, values the person named that narrow the rows ("Sharjah", found in the
table by code, never sent anywhere), and the grouping columns that may explain the change. ``brief`` writes the decision
brief by a template from the result alone: the change, the segments that carry most of it with their volume, mix and rate,
and always that the data shows association, not cause. What it cannot read it returns as a reason, never a guess.
"""

from __future__ import annotations

import calendar
import re
from typing import Any, Optional

import pandas as pd

from dclab_core import change as C
from dclab_core import query as Q
from dclab_core.times import is_text  # object on pandas 2, the string type on pandas 3

from . import lang

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
FA_QUARTERS = {"اول": 1, "دوم": 2, "سوم": 3, "چهارم": 4}
MAX_SPLIT = 30  # a grouping column with more values than this explains nothing a person can read
_MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"


def _month(name: str, year: int) -> C.Window:
    m = MONTHS[name[:3] if name[:3] in MONTHS and name not in MONTHS else name]
    last = calendar.monthrange(year, m)[1]
    return C.Window(start=f"{year}-{m:02d}-01", end=f"{year}-{m:02d}-{last}", label=f"{calendar.month_name[m]} {year}")


def _quarter(q: int, year: int) -> C.Window:
    first = 3 * (q - 1) + 1
    last = calendar.monthrange(year, first + 2)[1]
    return C.Window(start=f"{year}-{first:02d}-01", end=f"{year}-{first + 2:02d}-{last}", label=f"Q{q} {year}")


def windows(text: str) -> Optional[tuple[C.Window, C.Window]]:
    """(baseline A, the period asked about B), or None when the words name no two periods the rules can read."""
    low = lang.normalize(text).lower()
    vs = r"(?:compared\s+(?:with|to)|vs\.?|versus|against|than)\s+(?:in\s+|with\s+)?"
    m = re.search(rf"\b{_MONTH}\s*(20\d\d)?\b[^.?!]{{0,40}}?\b{vs}{_MONTH}\s*(20\d\d)?\b", low)
    # 36.3: "compare March 2026 with March 2025" (the second is the baseline, as with "compared with") and "from January to
    # February 2026" (the first is the baseline): the written people's own ways of naming two periods
    m = m or re.search(rf"\bcompare\s+(?:the\s+)?{_MONTH}\s*(20\d\d)?\s+(?:with|to|against|and)\s+(?:the\s+)?{_MONTH}\s*(20\d\d)?\b", low)
    span = None if m else re.search(rf"\bfrom\s+{_MONTH}\s*(20\d\d)?\s+(?:to|until|till|through)\s+{_MONTH}\s*(20\d\d)?\b", low)
    if m or span:
        if m:
            b_name, b_year, a_name, a_year = m.groups()
        else:
            a_name, a_year, b_name, b_year = span.groups()
        year_b = int(b_year or a_year or 0)
        year_a = int(a_year or b_year or 0)
        if not year_b:
            return None  # no year is ever assumed
        if not a_year and MONTHS[a_name[:3]] > MONTHS[b_name[:3]]:  # "January compared with December": the year before
            year_a -= 1
        elif a_year and not b_year and MONTHS[b_name[:3]] < MONTHS[a_name[:3]]:  # "January compared with December 2025": the year after
            year_b += 1
        a, b = _month(a_name, year_a), _month(b_name, year_b)
        return None if a.start == b.start else (a, b)
    quarter = r"\bq([1-4])\s*(20\d\d)\b|\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(20\d\d)\b|سه\s+ماه\s+(اول|دوم|سوم|چهارم)\s+(?:سال\s+)?(20\d\d)"

    def q_of(g: tuple) -> tuple[int, int]:
        n = int(g[0]) if g[0] else {"first": 1, "second": 2, "third": 3, "fourth": 4}.get(g[2] or "") or FA_QUARTERS[g[4]]
        return n, int(g[1] or g[3] or g[5])
    two = re.search(rf"(?:{quarter})[^.?!]{{0,40}}?\b{vs}(?:{quarter})", low)
    if two:  # "Q2 2026 compared with Q2 2025": the second one is the baseline (the review of H11a)
        g = two.groups()
        b, a = q_of(g[:6]), q_of(g[6:])
        return (None if a == b else (_quarter(*a), _quarter(*b)))
    q = re.search(quarter, low)
    if q:
        n, year = q_of(q.groups())
        prev = (n - 1, year) if n > 1 else (4, year - 1)
        return _quarter(*prev), _quarter(n, year)  # a quarter alone: against the one before it
    return None


def _date_columns(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if re.search(r"(?:^|_)date(?:$|_)", str(c).lower()) or pd.api.types.is_datetime64_any_dtype(frame[c])]


RATIO = re.compile(r"\b(?:average|avg|mean|median|per|value|aov|rate|ratio|share|percent(?:age)?|margin)\b|میانگین|نرخ|درصد|سهم", re.I)
COUNTS = re.compile(r"\b(?:orders?|bookings?|customers?|count|number|transactions?|sales\s+count)\b|تعداد|سفارش", re.I)
# words never read as a value that narrows the rows: a sentence's own words, however a cell happens to be spelled
_COMMON = re.compile(r"^(?:the|and|for|our|why|did|does|was|were|what|how|this|that|with|from|into|than|then|drop|fell|fall|rise|rose|"
                     r"grow|grew|revenue|sales|orders?|month|year|quarter|compared|versus|against|home|store|online|paid|total|all|any|"
                     r"new|old|other|more|less|last|next|data|table|lower|higher|down|up)$", re.I)


NO_DATES = "a date column to place each row in a period: this table has none"  # the owner's Telco test (2026-10-29)


def _example(frame: pd.DataFrame, date: str) -> str:
    """Two periods the table really holds (its latest month and the one before), as an example; none when they cannot be read."""
    try:
        days = pd.to_datetime(frame[date], errors="coerce").dropna()
    except (TypeError, ValueError):
        return ""
    if days.empty:
        return ""
    last = days.max().to_period("M")
    before = last - 1
    if not (days.dt.to_period("M") == before).any():
        return ""
    return f" (for example: {last.strftime('%B %Y')} compared with {before.strftime('%B %Y')})"


def read(text: str, frame: pd.DataFrame, measure_words: str = "", columns: Optional[list[dict[str, Any]]] = None
         ) -> tuple[Optional[dict[str, Any]], str]:
    """What the comparison needs, or None with what is missing (said to the person). ``columns``: the studio's column
    summaries; a column it marks personal is never summed, split or used to narrow (the review of H11a)."""
    from .. import home_query

    personal = {str(c.get("name")) for c in columns or [] if str(c.get("sensitivity") or "none") in home_query.SENSITIVE}
    dates = [d for d in _date_columns(frame) if d not in personal]
    if not dates:  # the owner's Telco test: periods were asked of a table that has no dates, with an invented example
        return None, NO_DATES
    pair = windows(text)
    if pair is None:
        return None, "which two periods to compare" + _example(frame, dates[0])
    said_dates = [d for d in dates if re.search(rf"(?<![\w]){re.escape(d.lower())}(?![\w])", lang.normalize(text).lower())]
    date = dates[0] if len(dates) == 1 else (said_dates[0] if len(said_dates) == 1 else None)
    if date is None:
        return None, "which date column places each row in a period" + (f" ({', '.join(dates)})" if dates else "")
    words = lang.normalize(measure_words or text)
    if RATIO.search(words):
        return None, "a total or a count to explain: an average, a rate or a value per row is not split this way here"
    usable = lambda c: str(c) not in personal and not Q.ID_NAME.search(str(c))  # noqa: E731
    numeric = [{"name": str(c), "kind": "numeric"} for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and usable(c)]
    measure = None
    for chunk in re.findall(r"[A-Za-z_\u0600-\u06FF]+", words):
        hit = home_query._column(chunk, numeric, ("numeric",))
        if hit:
            measure = hit
            break
    if measure is None and not COUNTS.search(words):
        return None, "which number to explain (a column such as amount, or the number of rows)"
    named: dict[str, str] = {}
    clash: set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z\-]*", text):  # the person's own spelling: a name, not a common word
        if len(w) < 3 or _COMMON.match(w) or not (w[0].isupper() or w.isupper()):
            continue
        hits = [str(c) for c in frame.columns if is_text(frame[c]) and usable(c)
                and frame[c].astype(str).str.lower().eq(w.lower()).any()]
        if len(hits) == 1:
            value = str(frame[hits[0]][frame[hits[0]].astype(str).str.lower() == w.lower()].iloc[0])
            if hits[0] in named and named[hits[0]] != value:
                clash.add(hits[0])  # two values of one column: neither is guessed
            named[hits[0]] = value
    for c in clash:
        named.pop(c, None)
    splits = [str(c) for c in frame.columns if is_text(frame[c]) and str(c) not in named and str(c) != date
              and usable(c) and 1 < frame[c].nunique() <= MAX_SPLIT]
    if not splits:
        return None, "a column to split the change by (a region, a channel, a category)"
    return {"a": pair[0], "b": pair[1], "date_column": date, "measure": measure, "narrow": named, "splits": splits}, ""


def compare(spec: dict[str, Any], frame: pd.DataFrame) -> Optional[C.ChangeResult]:
    rows = frame
    for column, value in spec["narrow"].items():
        rows = rows[rows[column].astype(str) == value]
    kw = {"date_column": spec["date_column"], "a": spec["a"], "b": spec["b"], "measure": spec["measure"]}
    spec["scores"] = C.split_scores(rows, spec["splits"], **kw)  # every column offered and its score, kept with the answer
    return C.best_split(rows, spec["splits"], **kw)


def _n(v: float) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def shown(result: C.ChangeResult) -> list[str]:
    """Every number the brief can show, written by code from the result (kept beside it, so each is traced to the result)."""
    values = [result.total_a, result.total_b, result.change, float(result.missing_measure)]
    for x in result.segments:
        values += [x.change, x.volume, x.mix, x.rate]
    return sorted({t for v in values for t in (_n(abs(v)), f"{abs(v):,.0f}" if float(v).is_integer() else _n(abs(v)),
                                               ("−" if v < 0 else "+") + _n(abs(v)), ("-" if v < 0 else "+") + _n(abs(v)))})


def brief(result: C.ChangeResult, narrow: dict[str, str] | None = None, top: int = 3) -> str:
    """The decision brief, by template from the result alone: every number in it is the result's (see ``shown``)."""
    what = f"the total of {result.measure}" if result.measure else "the number of rows"
    where = "".join(f" for {k} = {v}" for k, v in (narrow or {}).items())
    direction = "rose" if result.change > 0 else "fell" if result.change < 0 else "did not change"
    lines = [f"{what[0].upper() + what[1:]}{where} {direction} by {_n(abs(result.change))}, from {_n(result.total_a)} in {result.a.label or result.a.start} "
             f"to {_n(result.total_b)} in {result.b.label or result.b.start}. By {result.by}, the change came most from:"]
    if result.change == 0:
        return (f"{what[0].upper() + what[1:]}{where} did not change: {_n(result.total_a)} in {result.a.label or result.a.start} and in "
                f"{result.b.label or result.b.start}. {result.caveat}")
    sign = 1.0 if result.change >= 0 else -1.0
    named = [x for x in result.segments if x.segment != C.OTHER]
    drivers = sorted([x for x in named if x.change * sign > 0], key=lambda x: -abs(x.change))[:top]  # moving with the total first
    against = sorted([x for x in named if x.change * sign < 0], key=lambda x: -abs(x.change))[:1]

    def line(s: C.Segment) -> str:
        parts = [f"{name} {'+' if v >= 0 else '−'}{_n(abs(v))}" for name, v in (("volume", s.volume), ("mix", s.mix), ("rate", s.rate)) if abs(v) >= 0.005]
        return f"- {s.segment}: {'+' if s.change >= 0 else '−'}{_n(abs(s.change))} ({'; '.join(parts) or 'no change'})"
    other = next((x for x in result.segments if x.segment == C.OTHER), None)
    lines += [line(s) for s in drivers]
    if other is not None and other.change * sign > 0:  # the pool under the heading its sign belongs to (the review of H11a)
        lines.append(f"- {C.OTHER}: {'+' if other.change >= 0 else '−'}{_n(abs(other.change))}")
    if against or (other is not None and other.change * sign < 0):
        lines.append("Against the change:")
        lines += [line(s) for s in against]
        if other is not None and other.change * sign < 0:
            lines.append(f"- {C.OTHER}: {'+' if other.change >= 0 else '−'}{_n(abs(other.change))}")
    lines.append("Volume is more or fewer rows, mix a shift between groups, rate the value per row "
                 f"(split in the order {result.order}).")
    return "\n".join(lines + notes(result, narrow))


def notes(result: C.ChangeResult, narrow: dict[str, str] | None = None) -> list[str]:
    """The lines code always says itself, whoever words the rest: the caveat, the rows left out, what the rows cover."""
    out = [result.caveat]
    if result.missing_measure:
        out.append(f"{result.missing_measure:,} rows with no {result.measure} are left out of the sums.")
    if not narrow:  # a place or a product named in other words than the table's is never matched by guess: said instead
        out.append("This covers every row of the two periods; if you meant a part of them (a region, a channel), name it as "
                   "it is written in the table.")
    return out


PROMPT = "home_business"
MAX_GROUPS = 8  # the group names the model may see (cell values of the person's table: the purpose's budget)


# a cause, a reason or a recommendation in a reworded sentence: the template stands (the review of H11b: cited.py's own
# overclaim words caught "causes" only, not "because", "drove", "should" or Persian "به دلیل", "باید")
CAUSAL = re.compile(r"\b(?:because|due\s+to|owing\s+to|driv(?:e|en|es|ing)|drove|led\s+to|leads?\s+to|result(?:s|ed)?\s+in|reason|"
                    r"thanks\s+to|explain(?:s|ed)?\s+by|should|must|recommend\w*|suggest\w*|advis\w*|need\s+to|ought)\b|"
                    r"به\s*دلیل|باعث|چون|زیرا|به\s*خاطر|باید|پیشنهاد|توصیه", re.I)
UP = re.compile(r"\b(?:rose|rise|risen|rising|grew|grow|grown|increas\w*|up|higher|gain\w*|more)\b|افزایش|بیشتر|رشد", re.I)
DOWN = re.compile(r"\b(?:fell|fall|fallen|falling|dropp?\w*|decreas\w*|declin\w*|down|lower|los[st]\w*|less|fewer|shrank)\b|کاهش|کمتر|افت", re.I)


def _faithful(sentence: str, result: C.ChangeResult, lines: dict[str, str], head: str) -> bool:
    """A reworded sentence says each number of the group it names, in the direction that group moved (the total's when it
    names none): never a number moved to another group, never a fall called a rise."""
    from .. import cited

    low = sentence.lower()
    groups = [g for g in lines if g.lower() in low]
    if len(groups) > 1:
        return False  # one group a sentence: its numbers are then that group's own
    line = lines[groups[0]] if groups else head
    if any(n not in cited.numbers(line, source=True) for n in cited.numbers(sentence)):
        return False
    change = next((x.change for x in result.segments if x.segment == groups[0]), 0.0) if groups else result.change
    if (UP.search(sentence) and change < 0) or (DOWN.search(sentence) and change > 0):
        return False
    return True


def reword(client: Any, result: C.ChangeResult, template: str, narrow: dict[str, str] | None = None) -> tuple[str, dict[str, Any]]:
    """H11b: the brief in plainer words by the home_business model, kept only when every sentence passes cited.py against
    the template, says each number of its own group in the direction it moved, claims no cause and gives no advice, and
    the groups the template names as drivers are all there. The scope and the notes (the caveat, the rows left out, what
    the rows cover) are always the code's. The template stands otherwise. Returns the text and what happened (never the
    model's own words)."""
    from .. import cited, prompts

    if client is None:
        return template, {"by": "template", "why": "no model serves home_business here"}
    named = [s.segment for s in result.segments if s.segment != C.OTHER]
    if len(named) + len(narrow or {}) > MAX_GROUPS:  # more names of the table's values than the purpose may see
        return template, {"by": "template", "why": "more groups than the model may see"}
    try:
        reply = client.complete([{"role": "system", "content": prompts.text(PROMPT)},
                                 {"role": "user", "content": template}], None)
    except Exception as error:  # noqa: BLE001 - the template stands, said why
        return template, {"by": "template", "why": f"the model did not answer ({type(error).__name__})"}
    text = str((reply.get("content") if isinstance(reply, dict) else reply) or "")
    checked = cited.check(text, {"CHANGE": template})
    kept = [cited.CITE.sub("", k["text"]).replace(" .", ".").strip() for k in checked.get("kept") or []]
    if not kept or checked.get("dropped") or len(kept) > 6:
        return template, {"by": "template", "why": "the model's sentences did not pass the citation check", "dropped": len(checked.get("dropped") or [])}
    if any(CAUSAL.search(k) for k in kept):
        return template, {"by": "template", "why": "a reworded sentence claimed a cause or gave advice"}
    rows = template.split("\n")
    lines = {s.segment: next((r for r in rows if r.startswith(f"- {s.segment}:")), "") for s in result.segments}
    lines = {g: line for g, line in lines.items() if line}
    if not all(_faithful(k, result, lines, rows[0]) for k in kept):
        return template, {"by": "template", "why": "a reworded sentence moved a number or a direction"}
    drivers = [r[2:].split(":")[0] for r in rows if r.startswith("- ")]
    if not all(d.lower() in " ".join(kept).lower() for d in drivers):
        return template, {"by": "template", "why": "the rewording left out a group the brief names"}
    scope = "".join(f"For {k} = {v}: " for k, v in (narrow or {}).items())
    return "\n".join([scope + " ".join(kept), *notes(result, narrow)]), {"by": "model", "kept": len(kept)}


__all__ = ["windows", "read", "compare", "brief", "notes", "shown", "reword", "PROMPT"]
