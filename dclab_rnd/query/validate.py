"""The plan validator (package H6): a plan runs only when every column is known and every use of it is allowed.

The kinds of a column are read from the table itself (``kinds``), not from what a model believed: numeric, boolean,
datetime, categorical, text (long or mostly unique strings) and identifier (an id-like name or almost one value per
row). Text and identifier columns may be counted (``count_distinct``) but never grouped on or filtered by: a group per
customer id, or a filter on a free-text note, is a way to read single rows.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from . import plan as plans

ID_NAME = re.compile(r"(?:^|[_\s.-])(?:id|uuid|guid|key|code|number|no)$|^id[_\s.-]|[_\s.-]id[_\s.-]", re.I)
CAMEL_ID = re.compile(r"[a-z0-9](?:ID|Id)$")  # customerId, CustomerID; case matters ("paid" is not an id)
# a column about a person is an identifier whatever its uniqueness: a group per e-mail is one person's rows
PERSONAL = re.compile(r"e[_\s.-]?mail(?:[_\s.-]?address)?$|(?:^|[_\s.-])(?:phone|mobile|msisdn|whatsapp)(?:[_\s.-]?(?:number|no))?$"
                      r"|(?:^|[_\s.-])(?:first|last|full|user|customer|client|contact|owner|person|employee|patient|student|guest|passenger)[_\s.-]?names?$"
                      r"|^names?$|^username$|(?:^|[_\s.-])(?:account|iban|passport|national[_\s.-]?id|ssn|imei)(?:[_\s.-]?(?:id|number|no))?$"
                      r"|(?:^|[_\s.-])(?:street[_\s.-]?)?address$|(?:^|[_\s.-])ip[_\s.-]?(?:addr|address)?$|(?:^|[_\s.-])(?:license|number)[_\s.-]?plate$", re.I)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE = re.compile(r"^\+?[\d\s().-]{7,20}$")
MAX_IN = 20  # values in an "in" filter
MAX_GROUP_UNIQUE = 60  # a numeric column with more distinct values than this is not a group


class PlanError(ValueError):
    """A plan the validator refused; ``problems`` lists every reason."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def kinds(frame) -> dict[str, str]:
    """Each column's kind, from the table."""
    import pandas as pd

    out: dict[str, str] = {}
    rows = max(len(frame), 1)
    for name in frame.columns:
        s = frame[name]
        unique = s.nunique(dropna=True)
        if pd.api.types.is_bool_dtype(s):
            out[str(name)] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[str(name)] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            # a number is an identifier only by its name ("customer_id", "phone"): amounts are often unique per row too
            named = _id_name(str(name)) and unique > max(20, 0.05 * rows)
            out[str(name)] = "identifier" if named or PERSONAL.search(str(name)) else "numeric"
        else:
            text = s.dropna().astype(str)
            long = bool(len(text)) and text.str.len().mean() > 40
            sample = text.head(500)
            shaped = bool(len(sample)) and (sample.str.match(EMAIL).mean() > 0.5 or (sample.str.match(PHONE).mean() > 0.5
                                                                                      and sample.str.count(r"\d").mean() >= 7))
            if PERSONAL.search(str(name)) or shaped:  # e-mails, phone numbers, names: a person, never a group
                out[str(name)] = "identifier"
                continue
            # an id-like name with many values is an identifier even when it repeats (a customer_id in an orders table)
            if _id_name(str(name)) and unique > max(20, 0.05 * rows) or (unique >= 0.9 * rows and rows >= 20 and not long):
                out[str(name)] = "identifier"
            elif long or (unique > 0.5 * rows and rows >= 20):
                out[str(name)] = "text"
            else:
                out[str(name)] = "categorical"
    return out


def _id_name(name: str) -> bool:
    return bool(ID_NAME.search(name) or CAMEL_ID.search(name))


def _a(kind: str) -> str:
    return ("an " if kind[:1] in "aeiou" else "a ") + kind


def _value_ok(kind: str, op: str, value: Any) -> bool:
    if op in ("is_null", "not_null"):
        return True
    values = value if op in ("in", "not_in", "between") else [value]
    if not isinstance(values, list) or not values:
        return False
    if op == "between" and len(values) != 2:
        return False
    if op in ("in", "not_in") and len(values) > MAX_IN:
        return False
    return all(isinstance(v, (str, int, float, bool)) and not (isinstance(v, str) and len(v) > 120) for v in values)


def validate(plan: dict[str, Any], frame) -> dict[str, Any]:
    """The plan, checked and completed (version, source, limit); raises PlanError with every reason it is refused.
    Any other error while checking is a refusal too: a malformed plan never reaches the executor."""
    try:
        return _validate(plan, frame)
    except PlanError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PlanError([f"the plan is malformed ({type(exc).__name__})"]) from None


def _typed(kind: str, op: str, value: Any) -> bool:
    """A filter value fits its column: numbers for a numeric column, dates for a date column."""
    import pandas as pd

    if op in ("is_null", "not_null"):
        return True
    values = value if isinstance(value, list) else [value]
    if kind == "numeric":
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if kind == "datetime":
        return all(isinstance(v, str) and not pd.isna(pd.to_datetime(v, errors="coerce")) for v in values)
    if kind == "boolean":
        return all(isinstance(v, (bool, int)) for v in values)
    return all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in values)


def _validate(plan: dict[str, Any], frame) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanError(["a plan is a JSON object"])
    p = copy.deepcopy(plan)
    problems: list[str] = []
    k = kinds(frame)
    if p.get("version", plans.SCHEMA_VERSION) != plans.SCHEMA_VERSION:
        problems.append(f"plan version {p.get('version')} is not {plans.SCHEMA_VERSION}")
    p["version"], p["source"] = plans.SCHEMA_VERSION, "active_asset"
    unknown = sorted(set(p) - {"version", "source", "filters", "group_by", "period", "measures", "sort", "limit"})
    if unknown:
        problems.append(f"unknown keys: {', '.join(unknown)}")

    def known(column: Any, where: str) -> bool:
        if not isinstance(column, str) or column not in k:
            problems.append(f"{where}: no column named {str(column)[:60]!r}")
            return False
        return True

    filters = p.get("filters") or []
    if not isinstance(filters, list):
        problems.append("filters must be a list")
        filters = []
    for f in filters:
        if not isinstance(f, dict):
            problems.append("a filter is an object with column, op and value")
            continue
        if not known(f.get("column"), "filter"):
            continue
        col, op, kind = f["column"], f.get("op"), k[f["column"]]
        if op not in plans.FILTER_OPS:
            problems.append(f"filter on {col}: unknown op {op!r}")
        elif kind in ("text", "identifier") and op not in ("is_null", "not_null"):
            problems.append(f"filter on {col}: {_a(kind)} column cannot be filtered by value (that reads single rows)")
        elif op in ("gt", "ge", "lt", "le", "between") and kind not in ("numeric", "datetime"):
            problems.append(f"filter on {col}: {op} needs a numeric or date column")
        elif not _value_ok(kind, op, f.get("value")) or not _typed(kind, op, f.get("value")):
            problems.append(f"filter on {col}: the value does not fit {op} on {_a(kind)} column")
    groups = p.get("group_by") or []
    if not isinstance(groups, list) or len(groups) > plans.MAX_GROUPS:
        problems.append(f"group_by is a list of at most {plans.MAX_GROUPS} columns")
        groups = []
    period_grain = isinstance(p.get("period"), dict) and bool((p.get("period") or {}).get("grain"))
    for g in groups:
        if not known(g, "group_by"):
            continue
        if g == "period" and period_grain:
            problems.append("group_by period: a column named period cannot be grouped beside a period grain")
            continue
        if k[g] in ("text", "identifier"):
            problems.append(f"group_by {g}: {_a(k[g])} column cannot be a group (a group per row is a row listing)")
        elif k[g] == "datetime":
            problems.append(f"group_by {g}: group a date with period and a grain")
        elif k[g] == "numeric" and frame[g].nunique(dropna=True) > MAX_GROUP_UNIQUE:
            problems.append(f"group_by {g}: a numeric column with many values is not a group")
    period = p.get("period")
    if period:
        if not isinstance(period, dict):
            problems.append("period is an object with column, grain, from and to")
            period = None
        elif not known(period.get("column"), "period"):
            pass
        elif k[period["column"]] != "datetime":
            problems.append(f"period {period['column']}: not a date column")
        elif period.get("grain") not in (None, *plans.GRAINS):
            problems.append(f"period: grain {period.get('grain')!r} is not one of {', '.join(plans.GRAINS)}")
        else:
            import pandas as pd

            for edge in ("from", "to"):
                if period.get(edge) is not None and pd.isna(pd.to_datetime(period[edge], errors="coerce")):
                    problems.append(f"period: {edge} {str(period[edge])[:20]!r} is not a date")
    measures = p.get("measures") or []
    if not isinstance(measures, list) or not measures:
        problems.append("a plan needs at least one measure: a question is answered with numbers, never by listing rows")
        measures = []
    names = set()
    for m in measures:
        if not isinstance(m, dict):
            problems.append("a measure is an object")
            continue
        name, op, col = m.get("name"), m.get("op"), m.get("column")
        if not isinstance(name, str) or not plans.NAME.match(name) or name in names or name in groups or (name == "period" and period_grain):
            problems.append(f"measure name {str(name)[:40]!r} is missing, repeated or not a plain name")
        else:
            names.add(name)
        if op not in plans.MEASURE_OPS and not (isinstance(op, str) and plans.PERCENTILE.match(op)):
            problems.append(f"measure {name}: unknown op {op!r}")
            continue
        if op == "rate":
            where = m.get("where")
            if not isinstance(where, dict) or not known(where.get("column"), f"measure {name}") or where.get("op") not in plans.FILTER_OPS \
                    or not _value_ok(k.get(where.get("column"), ""), where.get("op"), where.get("value")) \
                    or not _typed(k.get(where.get("column"), ""), where.get("op"), where.get("value")):
                problems.append(f"measure {name}: a rate needs one condition in where")
            elif k[where["column"]] in ("text", "identifier") and where.get("op") not in ("is_null", "not_null"):
                problems.append(f"measure {name}: a rate cannot test {_a(k[where['column']])} column by value")
            continue
        if op == "count":
            continue
        if not known(col, f"measure {name}"):
            continue
        numeric = k[col] in ("numeric",) or (k[col] == "boolean" and op == "mean")
        if (op in plans.NUMERIC_OPS or plans.PERCENTILE.match(op)) and not numeric:
            problems.append(f"measure {name}: {op} needs a numeric column, {col} is {k[col]}")
        if op in ("min", "max") and k[col] not in ("numeric", "datetime"):
            problems.append(f"measure {name}: {op} of {_a(k[col])} column would show one row's value")
    sorts = p.get("sort") or []
    if not isinstance(sorts, list):
        problems.append("sort must be a list")
        sorts = []
    for s in sorts:
        if not isinstance(s, dict) or not isinstance(s.get("by"), str) or s.get("by") not in names | set(groups) | ({"period"} if period else set()):
            problems.append(f"sort by {str((s or {}).get('by'))[:40]!r}: sort by a measure or a group")
    limit = p.get("limit", 100)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= plans.MAX_LIMIT:
        problems.append(f"limit must be 1 to {plans.MAX_LIMIT}")
    p.update(filters=filters, group_by=groups, measures=measures, sort=sorts, limit=limit if isinstance(limit, int) else 100)
    if not period:
        p.pop("period", None)
    if problems:
        raise PlanError(problems)
    return p
