"""Does a written person's story fit the table they share? (2026-11-01, after the route test)

The exam gives every person who shares data one of three fixed invented tables (``home.tables``: customers, sales, bookings).
The practice set ``people_dev_route`` was written by a writer who never saw them, and at least three of its eight played
people with a table told a story the table could not support (r-17 wanted Indian city names cleaned in a table of UAE cities
with no repeated customer, r-20 ran trips to Machu Picchu in a table of Paris and Dubai, r-06 was in Turkey). The simulated
person then doubted the file, the conversation drifted, and neither student's behaviour could be read from it.

A person can now say what they believe about their table, in ``table_claims``, and code checks it against the real table:

    "table_claims": {"columns": ["customer_id", "city"],            # columns the story names; each must exist
                     "values": {"city": ["Dubai", "Sharjah"]},       # values the story names; each must be in that column
                     "duplicate_rows": false,                         # the story says the table has exact duplicate rows (or none)
                     "repeated": ["customer_id"],                     # id columns the story says repeat (one person on several rows)
                     "unique": ["order_id"],                          # id columns the story says never repeat (one row each)
                     "missing": ["monthly_spend"]}                    # columns the story says have gaps; each must have some

The claims are opt-in for the sets that exist; a new set that must be written to fit (``REQUIRE_CLAIMS`` in the exam) refuses a
person with a table and no claims, or with claims that say nothing. Only what the story relies on needs to be claimed; a claim code
cannot check (a column with too many distinct values to list) is refused rather than trusted. A badly shaped claim is refused with
the shape it should have, never a traceback.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

CLAIM_KEYS = ("columns", "values", "duplicate_rows", "repeated", "unique", "missing")
TABLES = ("customers", "sales", "bookings")
MAX_VALUES = 30  # a column with more distinct values than this is listed by no one: it cannot be claimed


@lru_cache(maxsize=None)
def facts(name: str) -> dict[str, Any]:
    """What the invented table holds, as code sees it: the columns, the values of each column with few distinct values (text,
    numbers and yes/no alike), the ranges of the other numbers, the span of the dates, an example of each other text column,
    exact duplicate rows, the id columns that repeat and the columns with gaps."""
    import pandas as pd

    from ..home import tables

    df = tables.build(name)
    ids = [str(c) for c in df.columns if str(c).endswith("_id")]
    values, ranges, dates, examples = {}, {}, {}, {}
    for column in df.columns:
        c, series = str(column), df[column].dropna()
        if series.empty or c in ids:
            continue
        flag = pd.api.types.is_bool_dtype(series)
        numeric = pd.api.types.is_numeric_dtype(series) and not flag
        if series.nunique() <= MAX_VALUES:
            found = series.unique()
            values[c] = [str(v) for v in sorted(found)] if numeric or flag else sorted({str(v) for v in found})
        elif c.endswith("_date"):
            dates[c] = [str(series.min())[:10], str(series.max())[:10]]
        elif numeric:
            ranges[c] = [series.min().item(), series.max().item()]
        else:
            examples[c] = str(series.iloc[0])
    return {"rows": int(len(df)), "columns": [str(c) for c in df.columns], "values": values, "ranges": ranges, "dates": dates, "examples": examples,
            "duplicate_rows": int(df.duplicated().sum()), "missing": sorted(str(c) for c, n in df.isna().sum().items() if n),
            "repeated": [c for c in ids if bool(df[c].dropna().duplicated().any())]}


def _shape(claims: Mapping[str, Any]) -> list[str]:
    """What is wrong with the shape of the claims (the writer's typos), before anything is compared."""
    bad = []
    for key in ("columns", "repeated", "unique", "missing"):
        value = claims.get(key)
        if value is not None and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            bad.append(f"{key} must be a list of column names")
    values = claims.get("values")
    if values is not None and not (isinstance(values, Mapping) and all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v) for k, v in values.items())):
        bad.append("values must map a column name to a list of values")
    if "duplicate_rows" in claims and not isinstance(claims["duplicate_rows"], bool):
        bad.append("duplicate_rows must be true or false")
    return bad


def fit(person: Mapping[str, Any], require: bool = False) -> list[str]:
    """What is wrong between a person's claims and their table (empty when they fit)."""
    name, claims = person.get("upload"), person.get("table_claims")
    if not name:
        return ["table_claims without a table to share"] if claims is not None else []
    if name not in TABLES:
        return []  # check_person already refuses an unknown table
    if claims is None:
        return [f"a person who shares the {name} table must say what the story relies on in it (table_claims)"] if require else []
    if not isinstance(claims, Mapping) or set(claims) - set(CLAIM_KEYS):
        return [f"table_claims is an object that takes only {', '.join(CLAIM_KEYS)}"]
    shape = _shape(claims)
    if shape:
        return ["malformed table_claims: " + "; ".join(shape)]
    if require and not (any(claims.get(k) for k in CLAIM_KEYS if k != "duplicate_rows") or "duplicate_rows" in claims):
        return [f"table_claims says nothing about the {name} table: name the columns, values or properties the story relies on"]
    have, problems = facts(name), []
    for column in claims.get("columns") or []:
        if column not in have["columns"]:
            problems.append(f"the story names the column {column!r}, which the {name} table does not have ({', '.join(have['columns'])})")
    for column, wanted in (claims.get("values") or {}).items():
        if column not in have["columns"]:
            problems.append(f"values claimed for {column!r}, which the {name} table does not have")
            continue
        if column not in have["values"]:
            problems.append(f"{column!r} has too many distinct values to claim any of them")
            continue
        held = {v.casefold() for v in have["values"][column]}
        for value in wanted:
            if str(value).casefold() not in held:
                problems.append(f"the story names {value!r} in {column!r}, which holds only {', '.join(have['values'][column])}")
    if "duplicate_rows" in claims:
        says, has = claims["duplicate_rows"], have["duplicate_rows"] > 0
        if says != has:
            problems.append("the story says the table " + ("has" if says else "has no") + f" exact duplicate rows; the {name} table "
                            + (f"has {have['duplicate_rows']}" if has else "has none"))
    for key, wants_repeat in (("repeated", True), ("unique", False)):
        for column in claims.get(key) or []:
            word = "repeats" if wants_repeat else "is unique"
            if not column.endswith("_id"):
                problems.append(f"'{key}' is about id columns (ending in _id); {column!r} is not one")
            elif column not in have["columns"]:
                problems.append(f"the story says {column!r} {word}; the {name} table has no such column")
            elif (column in have["repeated"]) != wants_repeat:
                problems.append(f"the story says {column!r} {word}; in the {name} table it " + ("never does" if wants_repeat else "repeats")
                                + (f" (repeated: {', '.join(have['repeated'])})" if have["repeated"] else ""))
    for column in claims.get("missing") or []:
        if column not in have["missing"]:
            problems.append(f"the story says {column!r} has gaps; in the {name} table " + ("it has none" if column in have["columns"] else "there is no such column")
                            + (f" (gaps are in: {', '.join(have['missing'])})" if have["missing"] else ""))
    return problems
