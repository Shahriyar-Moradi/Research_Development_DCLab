"""Data cards and synthetic tables (package 19.11): a table described without its rows, and invented rows from that.

A :class:`DataCard` describes the tables of one problem in the ledger's vocabulary (``dclab_core.ledger``): each
column's role, dtype and bands, the aggregates needed to draw it again (rounded quantiles, the shares of its
categories, the shape of its identifiers, a date range), the relations between columns and tables, and the problems
those relations show (found in the data, or declared planted by a replica). :func:`synthesize` draws invented tables
from a card: the same columns, roles, bands and relations, and no row of the source, so a training example can carry
a table without carrying a person's rows.

What a card never holds: a row, or a value a person could be found by. Category names are replaced by codes
(``status_a``, by share) unless the caller keeps them (``keep_categories``: a public or synthetic table); identifiers
keep only their shape (``ph_`` and sixteen hexadecimal characters); numbers only rounded quantiles over many rows (a
column with fewer than ``MIN_ROWS`` values keeps none). A rounded quantile still says roughly where values lie (the
coordinates of a city, the scale of a fare): that is the aggregate the card is for.

Relations it finds and :func:`synthesize` reproduces:

* ``key``: a labels table matched to the main table on a column (how many rows match, how many do not, how many main
  rows each matched row finds), and the entity the matched rows concentrate on (a few bikers);
* ``join``: what joining on other shared columns does (the pair of an account and a biker: each row finds many);
* ``disjoint``: a shared column name whose values never meet (one name, two meanings);
* ``agrees``: how often a shared number agrees across the key (the fares of a sample against the table's);
* ``equals``: two columns of a row holding the same value for a share of rows (the receiver's phone is the biker's);
* ``null_when``: a column empty exactly for one category of another (a cancelled ride has no biker);
* ``one_to_one``: two identifiers naming the same entity (a biker and their phone).
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from dclab_core import sensitivity
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.bands import CARDINALITY_BANDS, DTYPE_KINDS, RATIO_BANDS, VALUE_PATTERNS, cardinality_band, name_tokens, ratio_band

if TYPE_CHECKING:  # the fit verdict is defined beside the check that makes it; the two modules resolve it between them (30.13)
    from dclab_core.meet import FitVerdict

MIN_ROWS = 20                 # a column with fewer values keeps no quantiles and no shares
MAX_CATEGORIES = 50           # the shares kept; the rest are one share for "others"
SAMPLE_ROWS = 100_000         # relations are measured on at most this many rows of a table
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)  # never the least or the most: those are one row's value
MAX_ID_COLUMNS = 12   # identifier columns compared pairwise within a table (66 pairs at most)
MIN_EQUALS = 0.01     # two columns hold the same value on at least this share of rows
MIN_LEVEL = 5  # a number kept as a category only when every kept one has at least this many rows
KEY_TOKENS = frozenset({"id", "ids", "uuid", "guid", "key", "ref", "phone", "email", "hash", "token", "nr", "number"})
HEX = frozenset("0123456789abcdef")
Role = Literal["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"]
# 20.3: what a column is for in the problem, each with its reason
Purpose = Literal["identifier", "time", "entity", "target_candidate", "feature", "text", "place", "constant", "leak_suspect"]
NOT_A_TARGET = frozenset({"identifier", "time", "entity", "text", "place", "constant", "leak_suspect"})
TableRole = Literal["main", "labels", "related"]
Source = Literal["found", "planted"]


def _sig(value: float, digits: int = 3) -> float:
    """``value`` rounded to ``digits`` significant digits: an aggregate, not a person's number."""
    if value is None or not math.isfinite(value) or value == 0:
        return 0.0 if value == 0 else float("nan")
    return float(round(value, -int(math.floor(math.log10(abs(value)))) + digits - 1))


# ------------------------------------------------------------------------------------------------ the contract

class ColumnCard(Contract):
    """One column: what it is, in bands, and what it takes to draw it again."""

    name: str = Field(min_length=1, max_length=200)
    dtype: Literal[DTYPE_KINDS]  # type: ignore[valid-type]
    role: Role
    nulls: Literal[RATIO_BANDS]  # type: ignore[valid-type]
    uniqueness: Literal[RATIO_BANDS]  # type: ignore[valid-type]
    cardinality: Literal[CARDINALITY_BANDS]  # type: ignore[valid-type]
    value_pattern: Literal[VALUE_PATTERNS]  # type: ignore[valid-type]
    null_rate: float = Field(ge=0, le=1, description="Rounded to three decimals.")
    distinct: int = Field(ge=0)
    monotonic: bool = False
    integer: bool = False
    quantiles: Optional[list[float]] = Field(None, max_length=len(QUANTILES), description="Numbers: rounded, at QUANTILES.")
    shares: Optional[list[float]] = Field(None, max_length=MAX_CATEGORIES + 1, description="Categories by share, largest first.")
    categories: Optional[list[str]] = Field(None, max_length=MAX_CATEGORIES, description="Their names, only when kept.")
    numeric_levels: bool = Field(False, description="The categories are numbers (kept only when not named like a key).")
    repeats: Optional[list[float]] = Field(None, max_length=3, description="Identifiers: rows per value at p50, p90 and the most.")
    prefix: Optional[str] = Field(None, max_length=6, description="Identifiers: a prefix every value shares (letters only).")
    shape: Optional[str] = Field(None, max_length=64, description="Identifiers: x hex, 9 digit, a letter, A capital, others kept.")
    digits: Optional[int] = Field(None, ge=1, le=20, description="Numeric identifiers: how many digits.")
    start: Optional[str] = Field(None, max_length=10, description="Dates: the first day.")
    end: Optional[str] = Field(None, max_length=10, description="Dates: the last day.")
    date_format: Optional[str] = Field(None, max_length=20, description="Dates stored as text: their format.")
    length: Optional[float] = Field(None, ge=0, description="Text: mean characters, rounded.")
    step: Optional[float] = Field(None, gt=0, description="Numbers: nearly every value a multiple of it (1, 2, 2.5 or 5 times a power of ten).")
    purpose: Optional[Purpose] = Field(None, description="v2 (20.3): what the column is for in the problem.")
    why: Optional[str] = Field(None, max_length=300, description="v2: the rule that gave the purpose, in words.")
    unit: Optional[str] = Field(None, max_length=20, description="v2: the unit its name says (km, minutes, EUR…).")
    sensitivity: Optional[Literal["restricted", "pii", "sensitive", "none"]] = Field(None, description="v3 (23.7): what the name and the shape of "
                                                                                         "the values say the column holds; not none: it reaches a model by name only.")
    sensitivity_why: Optional[str] = Field(None, max_length=80, description="v3: the rule in words, never a value.")
    known_at_prediction: Optional[Literal["yes", "no", "unknown"]] = Field(None, description="v4 (30.13): whether it is known when a "
                                                                          "prediction is made: no for a leak suspect, yes for a time or a key, else unknown until a person says")


class TableCard(Contract):
    name: str = Field(min_length=1, max_length=200)
    role: TableRole = "main"
    rows: int = Field(ge=0)
    columns: list[ColumnCard] = Field(default_factory=list, max_length=2000)
    grain: Optional[str] = Field(None, max_length=300, description="v2: what one row is, in words.")
    key: list[str] = Field(default_factory=list, max_length=4, description="v2: the fewest columns that tell rows apart.")
    duplicate_rows: Optional[float] = Field(None, ge=0, le=1, description="v2: the share of rows repeated whole.")
    target_candidates: list[str] = Field(default_factory=list, max_length=20, description="v2: allowed outcomes, likeliest first.")
    joins_on: list[str] = Field(default_factory=list, max_length=4, description="v2: a labels table's key into the main table.")


class Relation(Contract):
    """How columns or tables relate: measured on the source, drawn again by :func:`synthesize`."""

    kind: Literal["key", "join", "disjoint", "agrees", "equals", "null_when", "one_to_one"]
    table: str = Field(max_length=200)
    columns: list[str] = Field(min_length=1, max_length=4)
    other_table: Optional[str] = Field(None, max_length=200)
    other_columns: Optional[list[str]] = Field(None, max_length=4)
    matched: Optional[int] = Field(None, ge=0, description="key: rows of the table that find a row in the other.")
    unmatched: Optional[int] = Field(None, ge=0, description="key: rows that find none.")
    fanout: Optional[float] = Field(None, ge=0, description="key, join: rows of the other table each matched row finds.")
    share: Optional[float] = Field(None, ge=0, le=1, description="agrees, equals, disjoint: the share of rows it holds for.")
    category: Optional[int] = Field(None, ge=0, description="null_when: the category's rank by share (0 the largest).")
    concentrated_on: Optional[str] = Field(None, max_length=200, description="key: the entity column the matched rows gather on.")
    entities: Optional[int] = Field(None, ge=0, description="key: how many of those entities the matched rows hold.")


class Problem(Contract):
    """Something a careful data scientist must see in these tables, with the numbers that show it."""

    code: str = Field(pattern=r"^[a-z_]{3,60}$")
    source: Source = "found"
    tables: list[str] = Field(default_factory=list, max_length=4)
    columns: list[str] = Field(default_factory=list, max_length=8)
    statement: str = Field(max_length=400)
    numbers: dict[str, float] = Field(default_factory=dict)


class DetectedModifier(Contract):
    """v4 (30.13): a recipe modifier the rows show, as ``dclab_core.modifiers.detect`` says it (kept here so the card never
    imports the modifiers, which read cards' facts)."""

    name: str = Field(max_length=60)
    applies: bool
    reason: str = Field(max_length=600)
    strength: float = Field(ge=0, le=1)
    value: Optional[str] = Field(default=None, max_length=200)
    asks_user: bool = False


class DataCard(Versioned):
    """The tables of one problem without their rows: columns in bands, relations, problems."""

    schema_version: Literal["dclab_core.data_card.v4"] = _version("data_card", 4)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.data_card.v1", "dclab_core.data_card.v2", "dclab_core.data_card.v3"})  # every stored card stays readable
    name: str = Field(min_length=1, max_length=200)
    synthetic: bool = Field(False, description="Built from a synthetic table (a replica, or drawn from another card).")
    tables: list[TableCard] = Field(min_length=1, max_length=20)
    relations: list[Relation] = Field(default_factory=list, max_length=500)
    problems: list[Problem] = Field(default_factory=list, max_length=100)
    limits: list[str] = Field(default_factory=list, max_length=20)

    target: Optional[str] = Field(None, max_length=200, description="v2: the outcome the problem names, when it names one.")
    detected_modifiers: list[DetectedModifier] = Field(default_factory=list, max_length=12, description="v4 (30.13): the recipe modifiers the "
                                               "rows show, each with its reason (dclab_core.modifiers), read on the earlier rows when a column orders them.")
    fit: Optional[FitVerdict] = Field(None, description="v4: G2's verdict against a problem card, filled by a host that has one (meet.fit).")

    def table(self, name: str) -> TableCard:
        return next(t for t in self.tables if t.name == name)

    @classmethod
    def read(cls, document):
        """A card as stored; a v1, v2 or v3 card is read as v4: each only added fields, all optional."""
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        if data.get("schema_version") in ("dclab_core.data_card.v1", "dclab_core.data_card.v2"):
            data = sensitivity.upgrade_card(data)  # a card from before 23.7 may hold category names and ranges of a personal column
            data["schema_version"] = cls.model_fields["schema_version"].default
        elif data.get("schema_version") == "dclab_core.data_card.v3":
            data = {**data, "schema_version": cls.model_fields["schema_version"].default}
        return super().read(data)


CONTRACTS["data_card"] = DataCard


def _resolve_fit() -> None:
    """``DataCard.fit`` names ``meet.FitVerdict``, and ``meet`` reads data cards: whichever module loads last completes
    the card's type (both are loaded by ``dclab_core.contracts``), so neither imports the other at load."""
    meet = sys.modules.get("dclab_core.meet")
    if meet is not None and hasattr(meet, "FitVerdict"):
        DataCard.model_rebuild(_types_namespace={"FitVerdict": meet.FitVerdict})


_resolve_fit()


# ------------------------------------------------------------------------------------------------ reading a column

def _dates(series: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None
    sample = series.dropna().astype(str).head(200)
    if sample.empty or not sample.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return None
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed if parsed.notna().sum() >= 0.99 * series.notna().sum() else None


def _shape(values: pd.Series) -> tuple[Optional[str], Optional[str]]:
    """The prefix every value shares (letters and punctuation, at most six) and the commonest shape of the rest."""
    text = values.dropna().astype(str).head(500)
    if text.empty:
        return None, None
    prefix = text.iloc[0]
    for v in text:
        while prefix and not v.startswith(prefix):
            prefix = prefix[:-1]
    prefix = re.sub(r"[0-9].*$", "", prefix)[:6] if text.nunique() >= MIN_ROWS else ""
    rests = text.str[len(prefix):]
    hexlike = all(set(c for c in r if c.isalnum()) <= HEX for r in rests.head(200))

    def one(r: str) -> str:
        return "".join("x" if hexlike and c in HEX else "9" if c.isdigit() else "A" if c.isupper() else "a" if c.isalpha() else c
                       for c in r[:64])

    shapes = rests.map(one)
    return (prefix or None), (str(shapes.mode().iloc[0]) if not shapes.empty else None)


def dtype_band(series: pd.Series) -> str:
    """A column's storage type as a data card and a decision's state say it (integer, float, boolean, datetime,
    categorical, string, other)."""
    return _dtype(series)


def value_pattern(series: pd.Series, is_date: bool = False) -> str:
    """The shape of a column's written values (digits, mixed, date_like, long_text, short_text, or not_text for a number
    or a boolean), the band a data card and a decision's state use: never a value."""
    return _pattern(series, series if is_date else None)


def _pattern(series: pd.Series, dates: Optional[pd.Series]) -> str:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return "not_text"
    if dates is not None:
        return "date_like"
    text = series.dropna().astype(str).head(500)
    if text.empty:
        return "short_text"
    if text.str.fullmatch(r"\d+").all():
        return "digits"
    if text.str.len().mean() >= 40:
        return "long_text"
    if text.str.contains(r"\d").mean() > 0.5:
        return "mixed"
    return "short_text"


def _dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "string"
    return "other"


def _integer(values: pd.Series) -> bool:
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    return bool(len(numbers)) and bool((numbers % 1 == 0).all())


def _step(numbers: pd.Series) -> Optional[float]:
    """The largest step (1, 2, 2.5 or 5 times a power of ten) that 98% of the values are multiples of: fares in 250s."""
    sample = numbers.head(2000).to_numpy(dtype=float)
    sample = sample[np.isfinite(sample)]
    if not len(sample) or np.mean(sample % 1 == 0) < 0.98:
        return None
    best = None
    for power in range(0, 10):
        for unit in (1, 2, 2.5, 5):
            step = unit * 10 ** power
            if step % 1 == 0 and np.mean(sample % step == 0) >= 0.98:
                best = float(step)
    return best


def _digits(values: pd.Series) -> int:
    """How many digits the largest value has: invented numbers are drawn above it, never among the source's."""
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    numbers = numbers[np.isfinite(numbers)].abs()
    return int(min(18, max(1, len(str(int(numbers.max()))) if len(numbers) else 1)))


def key_named(name: str) -> bool:
    return bool(set(name_tokens(name)) & KEY_TOKENS)


def role_of(name: str, series: pd.Series) -> str:
    """The column's role in the ledger's vocabulary (``ledger.ROLES``), by rule: its dtype, its bands, its name."""
    values = series.dropna()
    distinct = int(values.nunique())
    if _dates(series) is not None:
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical_code"
    textual = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    unique_ratio = distinct / max(len(values), 1)
    if textual and len(values) and values.astype(str).str.len().mean() >= 40 and unique_ratio > 0.5:
        return "free_text"
    low = cardinality_band(distinct) in ("constant", "binary", "low")
    integral = textual or _integer(values)
    if not low and integral and (key_named(name) or (unique_ratio >= 0.98 and len(values) > MIN_ROWS)):
        return "identifier"
    if pd.api.types.is_numeric_dtype(series) and not low:
        return "numeric"
    return "categorical_code" if distinct else "other"


def column_card(name: str, series: pd.Series, keep_categories: bool = False) -> ColumnCard:
    values = series.dropna()
    n, distinct = len(values), int(values.nunique())
    dates = _dates(series)
    role = role_of(name, series)
    out: dict[str, Any] = dict(name=str(name)[:200], dtype=_dtype(series), role=role, nulls=ratio_band(float(series.isna().mean())) if len(series) else "none",
                               uniqueness=ratio_band(distinct / n) if n else "none", cardinality=cardinality_band(distinct),
                               value_pattern=_pattern(series, dates), null_rate=round(float(series.isna().mean()), 3) if len(series) else 0.0,
                               distinct=distinct, integer=_integer(values) if pd.api.types.is_numeric_dtype(series) else False)
    ordered = dates.dropna() if dates is not None else values
    out["monotonic"] = bool(n > 1 and distinct > 1 and (pd.api.types.is_numeric_dtype(ordered) or dates is not None)
                            and ordered.is_monotonic_increasing)
    if role == "datetime" and n >= MIN_ROWS:
        days = dates.dropna()
        if getattr(days.dt, "tz", None) is not None:
            days = days.dt.tz_convert(None)
        out.update(start=str(days.quantile(0.01).date()), end=str(days.quantile(0.99).date()))
    if role == "datetime" and n and not pd.api.types.is_datetime64_any_dtype(series):
        out["date_format"] = "%Y-%m-%d" if values.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all() else "%Y-%m-%dT%H:%M:%S"
    elif role == "identifier" and n:
        counts = values.value_counts()
        out["repeats"] = [_sig(float(counts.quantile(0.5)), 2), _sig(float(counts.quantile(0.9)), 2), float(counts.max())] \
            if distinct >= MIN_ROWS else None
        if pd.api.types.is_numeric_dtype(series):
            out["digits"] = _digits(values)
        else:
            out["prefix"], out["shape"] = _shape(values)
    elif role == "free_text" and n:
        out["length"] = _sig(float(values.astype(str).str.len().mean()), 2)
    elif role == "categorical_code" and n:
        counts = values.value_counts(normalize=True)
        top = counts.head(MAX_CATEGORIES)
        out["shares"] = [round(float(x), 4) for x in top] + ([round(float(counts.iloc[MAX_CATEGORIES:].sum()), 4)] if len(counts) > MAX_CATEGORIES else [])
        numeric = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        common = bool((values.value_counts().head(MAX_CATEGORIES) >= MIN_LEVEL).all())  # an age held by one person is that person
        if n >= MIN_ROWS and (keep_categories or (numeric and not key_named(name) and common)):
            out["categories"] = [str(v)[:80] for v in top.index]
            out["numeric_levels"] = bool(numeric)
        elif numeric and not pd.api.types.is_bool_dtype(series):
            out["digits"] = _digits(values)
    elif role == "numeric":
        numbers = pd.to_numeric(values, errors="coerce").astype(float)
        numbers = numbers[np.isfinite(numbers)]  # an inf or two never cost the whole card
        if len(numbers) >= MIN_ROWS:
            q = numbers.quantile(list(QUANTILES)).to_numpy()
            if len(numbers) < 100:  # with few rows the 1st and 99th percentiles are nearly one row's value
                q[0], q[-1] = q[1], q[-2]
            spread = float(q[-2] - q[1]) or float(q[-1] - q[0]) or abs(float(q[3])) or 1.0
            decimals = -int(math.floor(math.log10(spread))) + 2  # a hundredth of the spread: where values lie, not one of them
            out["quantiles"] = [float(round(float(x), decimals)) for x in q]
            out["step"] = _step(numbers)
    cls, why = sensitivity.classify(name, values)
    out.update(sensitivity=cls, sensitivity_why=why)
    return ColumnCard(**sensitivity.strip(out, cls))  # a column that holds a person's data keeps no value, however it was drawn


# ------------------------------------------------------------------------------------------------ relations

def _norm(series: pd.Series) -> pd.Series:
    """Values as text in one form: a number read as 622233 in one file and 622233.0 in another is the same value."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce").map(lambda x: None if pd.isna(x) else f"{float(x):.12g}")
    return series.map(lambda x: None if pd.isna(x) else str(x))


def _repeating(card: ColumnCard) -> bool:
    """An identifier whose values recur (an entity: a biker, a customer), not one value per row."""
    return card.role == "identifier" and card.uniqueness in ("under_1pct", "from_1_to_10pct", "from_10_to_50pct", "from_50_to_90pct")


def _sample(frame: pd.DataFrame) -> pd.DataFrame:
    return frame if len(frame) <= SAMPLE_ROWS else frame.sample(SAMPLE_ROWS, random_state=0)


def _within(name: str, frame: pd.DataFrame, cards: dict[str, ColumnCard]) -> list[Relation]:
    out: list[Relation] = []
    frame = _sample(frame)
    cats = [c for c, k in cards.items() if k.role == "categorical_code" and k.cardinality in ("binary", "low") and k.null_rate == 0]
    for col, card in cards.items():
        if not 0 < card.null_rate < 1:
            continue
        empty = frame[col].isna()
        for cat in cats:
            if cat == col:
                continue
            ranked = list(frame[cat].value_counts().index)
            values = frame.loc[empty, cat].unique()
            if len(values) == 1 and set(frame.loc[frame[cat] == values[0], col].isna()) == {True}:
                out.append(Relation(kind="null_when", table=name, columns=[col], other_columns=[cat], category=ranked.index(values[0])))
                break
    ids = [c for c, k in cards.items() if k.role == "identifier" and k.distinct >= 2][:MAX_ID_COLUMNS]
    norm = {c: _norm(frame[c]) for c in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            both = norm[a].notna() & norm[b].notna()
            if not both.any():
                continue
            if _repeating(cards[a]) and cards[a].distinct == cards[b].distinct and frame.loc[both].groupby(a)[b].nunique().max() == 1:
                out.append(Relation(kind="one_to_one", table=name, columns=[a], other_columns=[b]))
                continue
            same = float((norm[a][both] == norm[b][both]).sum() / len(frame))
            if same >= MIN_EQUALS:  # a chance coincidence of two numbers is not a relation
                out.append(Relation(kind="equals", table=name, columns=[a], other_columns=[b], share=round(same, 3)))
    return out


def _between(left: str, lf: pd.DataFrame, right: str, rf: pd.DataFrame, lc: dict[str, ColumnCard], rc: dict[str, ColumnCard]) -> list[Relation]:
    shared = [c for c in lc if c in rc]
    lf, rf = _sample(lf), _sample(rf)
    out: list[Relation] = []
    norm = {(side, c): _norm(f[c]) for side, f in (("l", lf), ("r", rf)) for c in shared}
    overlap = {c: float(norm["l", c].dropna().isin(set(norm["r", c].dropna())).mean()) if lf[c].notna().any() else 0.0 for c in shared}
    for c in shared:
        if overlap[c] == 0 and lc[c].role == "identifier" and rc[c].role == "identifier" and lf[c].notna().any():
            out.append(Relation(kind="disjoint", table=left, columns=[c], other_table=right, other_columns=[c], share=0.0))
    keys = [c for c in shared if lc[c].role == "identifier" and lc[c].uniqueness in ("from_90_to_99pct", "over_99pct", "all") and overlap[c] >= 0.5]
    if not keys:
        return out

    def fanout(cols: list[str]) -> tuple[int, float]:
        lk = pd.DataFrame({c: norm["l", c] for c in cols}).dropna()
        sizes = pd.DataFrame({c: norm["r", c] for c in cols}).dropna().groupby(cols).size().rename("n").reset_index()
        merged = lk.merge(sizes, on=cols, how="left")
        hit = merged["n"].notna()
        return int(hit.sum()), float(merged.loc[hit, "n"].mean()) if hit.any() else 0.0

    key = max(keys, key=lambda c: (overlap[c], -fanout([c])[1]))
    matched, fan = fanout([key])
    hit = norm["l", key].isin(set(norm["r", key].dropna()))
    entities = [c for c in shared if c != key and lc[c].role == "identifier" and _repeating(rc[c]) and overlap[c] > 0 and rc[c].distinct >= MIN_ROWS]
    on = min(entities, key=lambda c: lf.loc[hit, c].nunique() / max(rc[c].distinct, 1)) if entities else None
    out.append(Relation(kind="key", table=left, columns=[key], other_table=right, other_columns=[key], matched=int(hit.sum()),
                        unmatched=int((~hit).sum()), fanout=round(fan, 2), concentrated_on=on,
                        entities=int(lf.loc[hit, on].nunique()) if on else None))
    coarse = [c for c in shared if c != key and overlap[c] > 0 and (lc[c].role == "categorical_code" or _repeating(rc[c]))
              and c not in {r.columns[0] for r in out if r.kind == "disjoint"}]
    combos = [[c] for c in coarse if lc[c].role == "identifier"] + [[a, b] for i, a in enumerate(coarse) for b in coarse[i + 1:]
                                                                    if "identifier" in (lc[a].role, lc[b].role)]
    for cols in combos[:30]:
        n, f = fanout(cols)
        if n:
            out.append(Relation(kind="join", table=left, columns=cols, other_table=right, other_columns=cols, matched=n, fanout=round(f, 1)))
    left_rows = lf.loc[hit].assign(__key=norm["l", key][hit])
    right_rows = rf.assign(__key=norm["r", key]).drop_duplicates("__key")
    both = left_rows.merge(right_rows, on="__key", suffixes=("", "__other"))
    for c in shared:
        if c != key and rc[c].role == "numeric" and pd.api.types.is_numeric_dtype(lf[c]) and len(both):
            a, b = pd.to_numeric(both[c], errors="coerce"), pd.to_numeric(both[c + "__other"], errors="coerce")
            out.append(Relation(kind="agrees", table=left, columns=[c], other_table=right, other_columns=[c],
                                share=round(float(np.isclose(a, b, rtol=1e-6).mean()), 3)))
    return out


def _short(name: Any) -> str:
    text = str(name)
    return text if len(text) <= 60 else text[:57] + "..."


def _problem(**fields: Any) -> Problem:
    return Problem(**{**fields, "statement": fields["statement"][:400]})


def _problems(card_tables: list[TableCard], relations: list[Relation]) -> list[Problem]:
    out: list[Problem] = []
    keys = {(r.table, r.other_table): r for r in relations if r.kind == "key"}
    for (table, other), key in keys.items():  # one problem per pair of tables: the join that inflates most, and how many do
        coarse = [r for r in relations if r.kind == "join" and (r.table, r.other_table) == (table, other)
                  and r.fanout and r.fanout >= 10 * max(key.fanout or 1, 1)]
        if coarse:
            worst = max(coarse, key=lambda r: (r.fanout or 0, -len(r.columns)))
            out.append(_problem(code="coarse_join_inflates", tables=[table, other or ""], columns=worst.columns,
                               statement=f"Joining {_short(table)} to {_short(other)} on {' and '.join(_short(c) for c in worst.columns)} finds about {worst.fanout:g} rows for each "
                                         f"({len(coarse)} such joins); the key is {_short(key.columns[0])}.",
                               numbers={"fanout": worst.fanout or 0, "key_fanout": key.fanout or 0, "joins": len(coarse)}))
    for r in relations:
        if r.kind == "key":
            if r.unmatched:
                out.append(_problem(code="rows_not_in_table", tables=[r.table, r.other_table], columns=r.columns,
                                   statement=f"{r.unmatched} rows of {_short(r.table)} find no row in {_short(r.other_table)}.",
                                   numbers={"unmatched": r.unmatched, "matched": r.matched or 0}))
            if (r.fanout or 0) > 1:
                out.append(_problem(code="several_rows_per_key", tables=[r.other_table], columns=r.columns,
                                   statement=f"{_short(r.columns[0])} repeats in {_short(r.other_table)}: a matched row finds {r.fanout:g} rows on average.",
                                   numbers={"fanout": r.fanout or 0}))
            other = next((t for t in card_tables if t.name == r.other_table), None)
            col = next((c for c in other.columns if c.name == r.concentrated_on), None) if other and r.concentrated_on else None
            if col is not None and r.entities is not None and r.entities <= 0.25 * col.distinct:
                out.append(_problem(code="labels_concentrate_on_entities", tables=[r.table, r.other_table], columns=[col.name],
                                   statement=f"The matched rows hold {r.entities} of {col.distinct} {_short(col.name)} values: a random split "
                                             "lets a model recognise them.", numbers={"entities": r.entities, "distinct": col.distinct}))
        elif r.kind == "disjoint":
            out.append(_problem(code="same_name_other_meaning", tables=[r.table, r.other_table], columns=r.columns,
                               statement=f"{_short(r.columns[0])} never holds a value of {_short(r.other_table)}'s {_short(r.columns[0])}: one name, two meanings.",
                               numbers={"overlap": 0.0}))
        elif r.kind == "agrees" and r.share is not None and r.share < 0.5:
            out.append(_problem(code="values_disagree", tables=[r.table, r.other_table], columns=r.columns,
                               statement=f"{_short(r.columns[0])} agrees with {_short(r.other_table)}'s on {r.share:.0%} of matched rows.", numbers={"share": r.share}))
        elif r.kind == "null_when":
            out.append(_problem(code="empty_by_category", tables=[r.table], columns=r.columns + (r.other_columns or []),
                               statement=f"{_short(r.columns[0])} is empty exactly when {_short((r.other_columns or ['?'])[0])} is one category.",
                               numbers={"category_rank": r.category or 0}))
    return out


# ------------------------------------------------------------------------------------------------ purposes (20.3)

UNITS = {"km": "km", "m2": "m²", "mm": "mm", "cm": "cm", "kg": "kg", "minutes": "minutes", "mins": "minutes",
         "hours": "hours", "hrs": "hours", "days": "days", "weeks": "weeks", "months": "months", "years": "years", "seconds": "seconds",
         "pct": "percent", "percent": "percent", "eur": "EUR", "usd": "USD", "gbp": "GBP", "aed": "AED"}
COORDINATES = frozenset({"lat", "lon", "lng", "latitude", "longitude"})
PLACE_WORDS = COORDINATES | frozenset({"city", "country", "postcode", "zip", "zipcode", "district", "address", "street"})
# words a column written after the outcome often carries: a suspicion to check, never a verdict
AFTER_WORDS = frozenset({"final", "resolved", "refund", "refunded", "rating", "rated", "reply", "replied", "response", "delivered",
                         "closed", "completed", "settled", "lifetime"})
AFTER_PHRASES = ("so_far",)
# an outcome by its whole name (churned, fraud, label) or by its form (is_late, fraud_flag); a token inside a longer
# name (default_rate, class_size, positive_reviews) says nothing
OUTCOME_NAMES = frozenset({"target", "label", "outcome", "result", "class", "churn", "churned", "fraud", "default", "defaulted",
                           "converted", "bought", "negative", "positive", "urgent", "accepted", "cancelled", "canceled", "renewed",
                           "failed", "y"})
FLAG_HEADS = frozenset({"is", "has", "was", "did", "will"})
FLAG_TAILS = frozenset({"flag", "label", "target", "outcome"})
ENTITY_LEVELS = 20  # a text column with this many recurring names is a thing rows belong to, not a category


def _unit(name: str) -> Optional[str]:
    return next((UNITS[t] for t in reversed(name_tokens(name)) if t in UNITS), None)


def outcome_named(name: str) -> bool:
    tokens = name_tokens(name)
    return bool(tokens) and ("_".join(tokens) in OUTCOME_NAMES or tokens[0] in FLAG_HEADS and len(tokens) > 1
                             or tokens[-1] in FLAG_TAILS and len(tokens) > 1)


def _after_named(name: str) -> Optional[str]:
    hit = sorted(set(name_tokens(name)) & AFTER_WORDS) or [p for p in AFTER_PHRASES if p in name.lower()]
    return ", ".join(hit) if hit else None


def _row_number(series: pd.Series) -> bool:
    numbers = pd.to_numeric(series, errors="coerce")
    if numbers.isna().any() or len(numbers) < MIN_ROWS or not (numbers % 1 == 0).all():
        return False
    return bool(numbers.nunique() == len(numbers) and numbers.max() - numbers.min() + 1 == len(numbers))


def _auc(x: pd.Series, y: pd.Series) -> float:
    """How well one number separates a yes from a no (the area under its ROC curve, 0.5 when it says nothing)."""
    ranks = x.rank()
    pos = y.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return 0.5
    auc = (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def _purity(x: pd.Series, y: pd.Series) -> Optional[float]:
    """The share of rows whose value names their outcome: each value's commonest outcome, counted over all rows."""
    pair = pd.DataFrame({"x": _norm(x), "y": _norm(y)}).dropna()
    if len(pair) < MIN_ROWS or pair["y"].value_counts(normalize=True).iloc[0] >= 0.9:
        return None  # an outcome nine rows in ten share: any column looks pure
    return float(pair.groupby("x")["y"].agg(lambda v: v.value_counts().iloc[0]).sum() / len(pair))


def _leak(name: str, frame: pd.DataFrame, card: ColumnCard, target: Optional[str], cards: dict[str, ColumnCard],
          dates: dict[str, Optional[pd.Series]]) -> Optional[str]:
    """Why a column looks written after the outcome, or None: by its name, and, when the outcome is known, by what it
    holds: empty for exactly one outcome, naming it almost perfectly, separating it alone, tracking a number outcome,
    or a time that gives it with another time. A suspicion to check, never proof."""
    words = _after_named(name)
    if words:
        return f"its name ({words}) may say it is written after the outcome: check it is known at the prediction moment"
    if not target or target not in frame or target == name:
        return None
    tokens = name_tokens(name)
    if tokens and ("_".join(tokens) in OUTCOME_NAMES or (tokens[-1] in FLAG_TAILS and len(tokens) > 1)):  # is_/has_ forms are often inputs
        return f"its name says an outcome too, beside {target}: a second outcome is not an input"
    known = frame[target].notna()
    if known.sum() < MIN_ROWS:
        return None
    y, x = frame[target][known], frame[name][known]
    outcome = cards.get(target)
    categorical = outcome is not None and outcome.cardinality in ("binary", "low", "medium")
    if categorical and 0 < x.isna().mean() < 1:
        empty = x.isna()
        for value in y.unique()[:20]:
            hit = y == value
            both = (empty & hit).sum()  # both ways: that outcome's rows are the empty ones, and the empty ones are that outcome
            if both >= 0.95 * hit.sum() and both >= 0.95 * empty.sum():
                return f"it is empty exactly when {target} is one value"
    if categorical and card.role == "categorical_code" and card.distinct >= 2 and len(x) / max(card.distinct, 1) >= 5:
        pure = _purity(x, y)
        if pure is not None and pure >= 0.98:
            return f"each of its values names one {target} ({pure:.0%} of rows)"
    if card.role == "numeric" and outcome is not None:
        numbers = pd.to_numeric(x, errors="coerce")
        both = numbers.notna()
        if outcome.cardinality == "binary" and both.sum() >= MIN_ROWS:
            auc = _auc(numbers[both], y[both] == y[both].unique()[0])
            if auc >= 0.97:
                return f"it alone separates the values of {target} (AUC {auc:.2f}): check it is known at the prediction moment"
        elif outcome.role == "numeric" and both.sum() >= MIN_ROWS:
            corr = abs(float(numbers[both].corr(pd.to_numeric(y[both], errors="coerce"), method="spearman") or 0))
            if corr >= 0.97:
                return f"it follows {target} almost exactly (rank correlation {corr:.2f})"
    if card.role == "datetime" and outcome is not None and outcome.role == "numeric" and dates.get(name) is not None:
        later = dates[name][known]
        for other, earlier in dates.items():
            if other in (name, target) or earlier is None:
                continue
            gap = (later - earlier[known]).dt.total_seconds()
            both = gap.notna() & pd.to_numeric(y, errors="coerce").notna()
            if both.sum() >= MIN_ROWS and (gap[both] >= 0).mean() > 0.95:
                corr = abs(float(gap[both].corr(pd.to_numeric(y[both]), method="spearman") or 0))
                if corr >= 0.95:
                    return f"its time minus {other} gives {target} (rank correlation {corr:.2f})"
    return None


def purposes(frame: pd.DataFrame, cards: dict[str, ColumnCard], target: Optional[str] = None,
             full: Optional[pd.DataFrame] = None) -> dict[str, tuple[str, str]]:
    """Each column's purpose and the reason, by rule (``frame`` may be a sample; ``full`` the whole table, for a row
    number). The outcome the problem names is judged as an outcome: a number with a different value in every row is a
    regression target, not an identifier; it is refused only when its name says it is a key, it counts the rows, it is
    one value, free text, a time, a coordinate or written after the outcome."""
    full = full if full is not None else frame
    dates = {c: _dates(frame[c]) for c, k in cards.items() if k.role == "datetime"}  # parsed once
    out: dict[str, tuple[str, str]] = {}
    for name, card in cards.items():
        tokens = set(name_tokens(name))
        is_target = name == target
        unique = card.uniqueness in ("from_90_to_99pct", "over_99pct", "all")
        numeric_values = card.dtype in ("integer", "float") or pd.to_numeric(frame[name].dropna().head(200), errors="coerce").notna().all()
        if card.distinct <= 1 and not 0 < card.null_rate < 1:  # one value sometimes empty is a flag, not a constant
            out[name] = ("constant", "one value in every row" if card.distinct else "empty in every row")
        elif key_named(name):  # customer_id, order_ref, biker_phone: never an outcome, whatever its values
            out[name] = ("identifier", "its name says it is a key, with a different value in nearly every row") if unique else \
                ("entity", "its name says it is a key, and its values recur: rows of one customer, biker or store belong together")
        elif card.dtype in ("integer", "float") and _row_number(full[name]):
            out[name] = ("identifier", "the numbers 1 to n (or a step apart), one per row: a row number")
        elif card.role == "identifier" and unique and not (is_target and numeric_values):
            out[name] = ("identifier", "a different value in nearly every row")
        elif not is_target and (why := _leak(name, frame, card, target, cards, dates)):
            out[name] = ("leak_suspect", why)
        elif card.role == "free_text":
            out[name] = ("text", "long, varied text: words to read, not a category")
        elif card.role == "datetime":
            out[name] = ("time", "dates or times")
        elif tokens & COORDINATES or (not is_target and tokens & PLACE_WORDS):
            out[name] = ("place", f"its name says where ({', '.join(sorted(tokens & PLACE_WORDS))})")
        elif card.role == "identifier" and not is_target:
            out[name] = ("entity", "a key whose values recur: rows of one customer, biker or store belong together")
        elif not is_target and card.role == "categorical_code" and card.dtype in ("string", "categorical") \
                and card.distinct >= ENTITY_LEVELS and card.uniqueness in ("under_1pct", "from_1_to_10pct", "from_10_to_50pct"):
            out[name] = ("entity", f"{card.distinct} names that recur: rows of one machine, author or series belong together")
        elif is_target:
            out[name] = ("target_candidate", "the outcome the problem names")
        elif not target and (card.cardinality == "binary" or outcome_named(name)):  # with no outcome named: what could be one
            out[name] = ("target_candidate", "a yes or no" if card.cardinality == "binary" else "its name says an outcome")
        else:
            out[name] = ("feature", "known about the row: a candidate input")
    if not target:  # a yes or no whose name says written-after is a likely outcome when none is named yet
        for name, card in cards.items():
            if out[name][0] == "leak_suspect" and (card.cardinality == "binary" or outcome_named(name)):
                out[name] = ("target_candidate", "a yes or no, or named as an outcome; its name may also mean written after one")
    for name, card in cards.items():  # an entity whose rows all share one outcome: the group a split must keep together
        if target and name != target and out[name][0] == "entity" and target in frame:
            pure = _purity(frame[name], frame[target])
            if pure is not None and pure >= 0.98:
                out[name] = ("entity", out[name][1] + f"; its rows share one {target}: split by it")
    return out


def _grain(frame: pd.DataFrame, cards: dict[str, ColumnCard], purpose: dict[str, tuple[str, str]]) -> tuple[str, list[str], float]:
    """What one row is: the fewest columns (up to three) that tell rows apart, in words."""
    sample = _sample(frame)
    duplicates = float(sample.duplicated().mean()) if len(sample) else 0.0
    sample = sample.drop_duplicates()  # a row repeated whole is a duplicate to remove, not a second grain
    repeated = f", with {duplicates:.1%} of rows repeated whole" if duplicates else ""
    keys = [c for c, (p, _) in purpose.items() if p == "identifier" and cards[c].null_rate == 0 and not sample[c].duplicated().any()]
    if keys:
        return f"one row per {keys[0]}{repeated}", [keys[0]], duplicates
    pool = [c for c, (p, _) in purpose.items() if p in ("identifier", "entity", "time") or (cards[c].role == "categorical_code" and cards[c].distinct > 1)]
    pool = sorted(pool, key=lambda c: -cards[c].distinct)[:8]
    from itertools import combinations

    for size in (1, 2, 3):
        for combo in combinations(pool, size):
            part = sample[list(combo)]
            if part.notna().all(axis=None) and not part.duplicated().any():
                return f"one row per {' and '.join(combo)}{repeated}", list(combo), duplicates
    return f"no column or set of three tells the rows apart{repeated}", [], duplicates


def _candidates(cards: dict[str, ColumnCard], purpose: dict[str, tuple[str, str]], target: Optional[str]) -> list[str]:
    allowed = [c for c, (p, _) in purpose.items() if p == "target_candidate"]
    return sorted(allowed, key=lambda c: (c != target, not outcome_named(c), cards[c].cardinality != "binary", c))[:20]


KNOWN = {"leak_suspect": "no", "identifier": "yes", "entity": "yes", "constant": "yes"}  # v4: known at prediction time; a time may be
# written after the outcome (a chargeback date): unknown until the person names the one that orders the rows


def _modifiers(frame: pd.DataFrame, table: "TableCard", target: Optional[str]) -> list[DetectedModifier]:
    """v4: the recipe modifiers the main table's rows show for the named outcome, read on the rows the proposal may read
    (``modifiers.provisional_reading``: the earliest 75% by distinct times, or a seeded 75%), never the rows a later
    holdout takes."""
    from dclab_core import modifiers as M

    if not target or target not in frame.columns:
        return []
    rows = frame[frame[target].notna()]
    if len(rows) < MIN_ROWS:
        return []
    times = [c.name for c in table.columns if c.purpose == "time"]
    time_column = times[0] if times else None
    read, _ = M.provisional_reading(rows.index, rows[time_column] if time_column else None)
    rows = rows.loc[read]
    if len(rows) < MIN_ROWS:
        return []
    y = rows[target]
    values = y.dropna().unique()
    numeric = pd.api.types.is_numeric_dtype(y)
    task = "binary" if len(values) == 2 else "regression" if numeric and len(values) > 20 else "multiclass"
    positive = rate = None
    if task == "binary":
        counts = y.astype(str).value_counts()
        positive = counts.index[-1]  # the rarer class: the event a screen is asked to catch
        rate = float(counts.iloc[-1] / counts.sum())
    X = rows.drop(columns=[target])
    keys = [c.name for c in table.columns if c.purpose == "entity" and c.name in X]
    sources = M.label_source_columns(X)
    try:
        facts = M.facts_of(X, y, task, time_column, keys, rate, positive, label_source=sources[0] if sources else None)
        return [DetectedModifier(**d.model_dump()) for d in M.detect(facts)]
    except Exception:  # noqa: BLE001 - a card is never refused for its modifiers
        return []


def card_of(tables: Mapping[str, pd.DataFrame], name: str = "tables", roles: Optional[Mapping[str, str]] = None,
            keep_categories: bool = False, synthetic: bool = False, planted: Optional[list[Problem]] = None,
            target: Optional[str] = None) -> DataCard:
    """The card of these tables. ``roles``: each table's role (the first is ``main``, the rest ``labels`` by default).
    ``target``: the outcome the problem names, when it names one (in the main table): it sharpens the leak suspicions
    and is checked against what may be an outcome."""
    tables = {str(t): f.rename(columns=str) for t, f in tables.items()}  # a column named 0 is the column "0"
    names = list(tables)
    roles = dict(roles or {})
    columns = {t: {str(c): column_card(str(c), tables[t][c], keep_categories) for c in tables[t].columns} for t in names}
    main_name = next((t for i, t in enumerate(names) if roles.get(t, "main" if i == 0 else "labels") == "main"), names[0])
    cards, refused = [], []
    for i, t in enumerate(names):
        aim = target if t == main_name else None
        purpose = purposes(_sample(tables[t]), columns[t], aim, full=tables[t])
        from dclab_core.modifiers import label_source_columns

        sources = set(label_source_columns(tables[t]))  # where each label came from: never known when a prediction is made
        cols = [c.model_copy(update={"purpose": purpose[c.name][0], "why": purpose[c.name][1][:300], "unit": _unit(c.name),
                                     "known_at_prediction": None if c.name == aim else "no" if c.name in sources
                                     else KNOWN.get(purpose[c.name][0], "unknown")})
                for c in columns[t].values()]
        columns[t] = {c.name: c for c in cols}
        grain, key, duplicates = _grain(tables[t], columns[t], purpose)
        if aim and aim in purpose and purpose[aim][0] in NOT_A_TARGET:
            refused.append(_problem(code="target_refused", tables=[str(t)[:200]], columns=[str(aim)[:200]],
                                    statement=f"{_short(aim)} cannot be the outcome: it is {purpose[aim][0].replace('_', ' ')} ({purpose[aim][1]}).",
                                    numbers={}))
        cards.append(TableCard(name=str(t)[:200], role=roles.get(t, "main" if i == 0 else "labels"), rows=int(len(tables[t])), columns=cols,
                               grain=grain[:300], key=key[:4], duplicate_rows=round(duplicates, 4),
                               target_candidates=_candidates(columns[t], purpose, aim)))
    relations: list[Relation] = []
    main = next((c.name for c in cards if c.role == "main"), names[0])
    for t in names:  # between the tables first: a key, a coarse join and a name with two meanings are the traps
        if t != main:
            relations += _between(t, tables[t], main, tables[main], columns[t], columns[main])
    for t in names:
        relations += _within(t, tables[t], columns[t])
    keys = {r.table: list(r.columns) for r in relations if r.kind == "key"}
    cards = [c.model_copy(update={"joins_on": keys.get(c.name, [])[:4]}) for c in cards]
    found = refused + _problems(cards, relations)
    detected = _modifiers(tables[main_name], next(c for c in cards if c.name == str(main_name)[:200]), target)
    return DataCard(name=name[:200], synthetic=synthetic, tables=cards, relations=relations[:500], target=(target or None) and str(target)[:200],
                    detected_modifiers=detected,
                    problems=(found + [p.model_copy(update={"source": "planted"}) for p in planted or []])[:100],
                    limits=["Relations are measured on at most 100,000 rows of each table.",
                            "A relation not listed here (a behaviour across rows, a formula between numbers) is not drawn again."])


# ------------------------------------------------------------------------------------------------ drawing it again

def _ids(rng: np.random.Generator, card: ColumnCard, n: int, taken: set[str] = frozenset()) -> np.ndarray:
    """``n`` invented, distinct identifiers of the card's shape."""
    out: list[Any] = []
    seen = set(taken)
    if card.digits is not None or (card.dtype in ("integer", "float") and card.shape is None):
        digits = card.digits or 6  # the source's values are all below 10**digits: invented ones are drawn above
        width = max(digits, int(math.ceil(math.log10(max(2 * (n + len(seen)) / 9, 1)))))
        low, high = 10 ** width, 10 ** (width + 1)
        chosen: set[int] = set()
        for attempt in range(1000):  # bounded: a crowded range widens rather than loops
            if len(chosen) >= n:
                break
            if attempt and attempt % 20 == 0:
                low, high = high, high * 10
            for x in rng.integers(low, high, size=max(16, 2 * (n - len(chosen)))).tolist():
                if x not in chosen and f"{float(x):.12g}" not in seen:
                    chosen.add(x)
                    if len(chosen) == n:
                        break
        values = np.array(sorted(chosen) if card.monotonic else rng.permutation(sorted(chosen)), dtype=np.int64)
        return values
    shape, prefix = card.shape or "x" * 12, card.prefix or ""
    alphabet = {"x": "0123456789abcdef", "9": "0123456789", "a": "abcdefghijklmnopqrstuvwxyz", "A": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    while len(out) < n:
        value = prefix + "".join(alphabet[c][rng.integers(len(alphabet[c]))] if c in alphabet else c for c in shape)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return np.array(out, dtype=object)


def _repeat(rng: np.random.Generator, values: np.ndarray, n: int, card: ColumnCard) -> np.ndarray:
    """``n`` rows holding every one of ``values`` at least once, the rest repeated as the card's rows per value say."""
    if not len(values):
        return np.array([None] * n, dtype=object)
    p50, p90, most = card.repeats or (1, 1, 1)
    sigma = max(0.05, math.log(max(p90, p50 + 1e-9) / max(p50, 1e-9)) / 1.2816)
    weights = np.clip(rng.lognormal(math.log(max(p50, 1e-9)), sigma, size=len(values)), 1e-9, max(most, 1))
    extra = rng.choice(len(values), size=max(0, n - len(values)), p=weights / weights.sum())
    draws = np.concatenate([np.arange(len(values)), extra])
    draws = np.sort(draws) if card.monotonic else rng.permutation(draws)
    return values[draws]


def _letters(i: int) -> str:
    """0 → a, 25 → z, 26 → ba: a category's invented code, with no digit in it."""
    out = ""
    while True:
        out = "abcdefghijklmnopqrstuvwxyz"[i % 26] + out
        i //= 26
        if not i:
            return out


def _column(rng: np.random.Generator, card: ColumnCard, n: int, source_rows: int) -> pd.Series:
    nonnull = n - int(round(card.null_rate * n))
    if card.role == "numeric" and card.quantiles:
        q = np.array(card.quantiles, dtype=float)
        values = np.interp(rng.uniform(0, 1, size=nonnull), QUANTILES, q)
        if card.step:
            values = np.round(values / card.step) * card.step
        elif card.integer:
            values = np.round(values)
    elif card.role == "categorical_code" and card.shares:
        shares = np.array(card.shares[:MAX_CATEGORIES], dtype=float)
        k = len(shares)
        if card.categories:
            levels = np.array([(c == "True") if card.dtype == "boolean" else float(c) if card.numeric_levels else c
                               for c in card.categories[:k]], dtype=object)
        elif card.digits is not None:
            levels = _ids(rng, card.model_copy(update={"monotonic": False}), k)
        else:
            levels = np.array([f"{card.name[:40]}_{_letters(i)}" for i in range(k)], dtype=object)
        values = levels[rng.choice(k, size=nonnull, p=shares / shares.sum())]
    elif card.role == "identifier":
        distinct = max(1, min(nonnull, int(round(card.distinct * n / max(source_rows, 1)))))
        if card.uniqueness == "all":
            distinct = nonnull
        pool = _ids(rng, card, distinct)
        values = pool if distinct == nonnull else _repeat(rng, pool, nonnull, card)
    elif card.role == "datetime" and card.start:
        days = pd.date_range(card.start, card.end or card.start, freq="D")
        picks = rng.choice(len(days), size=nonnull)
        values = days[np.sort(picks) if card.monotonic else picks]
        values = values.strftime(card.date_format).to_numpy(dtype=object) if card.date_format else values.to_numpy()
    elif card.role == "free_text":
        words = np.array("the order rider late box city door call wait note item price route stop".split())
        size = max(1, int((card.length or 40) / 5))
        values = np.array([" ".join(rng.choice(words, size=size)) for _ in range(nonnull)], dtype=object)
    else:
        values = np.array([None] * nonnull, dtype=object)
    out = np.empty(n, dtype=object)
    where = np.sort(rng.choice(n, size=nonnull, replace=False)) if card.monotonic else rng.permutation(n)[:nonnull]
    out[:] = None
    out[where] = values
    series = pd.Series(out)
    if card.dtype == "boolean":
        return series.astype(bool) if series.notna().all() else series
    if card.dtype == "datetime":
        return pd.to_datetime(series)
    if card.dtype in ("integer", "float") or (card.role == "categorical_code" and card.numeric_levels):
        series = pd.to_numeric(series, errors="coerce")
        if card.dtype == "integer" and series.notna().all():
            series = series.astype("int64")
    return series


def _rank(series: pd.Series) -> list[Any]:
    return list(series.value_counts().index)


def synthesize(card: DataCard, seed: int = 0, rows: Optional[Mapping[str, int]] = None) -> dict[str, pd.DataFrame]:
    """Invented tables drawn from ``card``: the same columns, roles, bands and relations, no row of the source. Each
    frame is marked ``attrs["dclab.synthetic"] = True``."""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    order = sorted(card.tables, key=lambda t: t.role != "main")
    for table in order:
        n = int((rows or {}).get(table.name, table.rows))
        key = next((r for r in card.relations if r.kind == "key" and r.table == table.name and r.other_table in out), None)
        frame = pd.DataFrame({c.name: _column(rng, c, n, table.rows) for c in table.columns})
        if key is not None:
            frame = _matched(rng, card, table, frame, key, out[key.other_table], n)
        for r in [r for r in card.relations if r.table == table.name]:
            frame = _apply(rng, card, r, frame, out)
        frame = frame[[c.name for c in table.columns]]
        frame.attrs["dclab.synthetic"] = True
        out[table.name] = frame
    return out


def _matched(rng, card, table, frame, key, other, n):
    """A labels table: its matched rows copied from invented rows of the other table (gathered on the same number of
    entities), the rest invented with values the other table does not hold."""
    matched = int(round(n * (key.matched or 0) / max((key.matched or 0) + (key.unmatched or 0), 1)))
    pool = other
    if key.concentrated_on and key.entities and key.concentrated_on in other:
        counts = other[key.concentrated_on].value_counts()
        chosen = rng.choice(counts.index.to_numpy(), size=min(key.entities, len(counts)), replace=False)
        pool = other[other[key.concentrated_on].isin(chosen)]
    pool = pool.drop_duplicates(key.other_columns[0])
    picks = pool.iloc[rng.choice(len(pool), size=min(matched, len(pool)), replace=False)] if len(pool) else pool
    shared = [c.name for c in table.columns if c.name in other.columns]
    frame = frame.astype(object)
    if len(picks):
        frame.loc[: len(picks) - 1, shared] = picks[shared].to_numpy()
    rest = frame.index[len(picks):]
    keycard = next(c for c in table.columns if c.name == key.columns[0])
    taken = set(_norm(other[key.other_columns[0]]).dropna())
    if len(rest):
        fresh = _ids(rng, keycard.model_copy(update={"monotonic": False}), len(rest), taken)
        frame.loc[rest, key.columns[0]] = fresh
    for c in table.columns:
        if c.dtype == "integer" and frame[c.name].notna().all():
            frame[c.name] = pd.to_numeric(frame[c.name]).astype("int64")
        elif c.dtype == "float":
            frame[c.name] = pd.to_numeric(frame[c.name], errors="coerce")
    return frame.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)


def _apply(rng, card, r: Relation, frame: pd.DataFrame, done: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = frame.copy()
    col = r.columns[0]
    if r.kind == "null_when" and r.other_columns and r.other_columns[0] in frame:
        ranked = _rank(frame[r.other_columns[0]])
        if r.category is not None and r.category < len(ranked):
            empty = frame[r.other_columns[0]] == ranked[r.category]
            if empty.any():
                filler = frame.loc[~empty & frame[col].notna(), col]
                frame.loc[~empty & frame[col].isna(), col] = rng.choice(filler.to_numpy(), size=int((~empty & frame[col].isna()).sum())) if len(filler) else None
                frame.loc[empty, col] = None
    elif r.kind == "one_to_one" and r.other_columns:
        other = r.other_columns[0]
        keys = frame[col].dropna().unique()
        targets = frame[other].dropna().unique()
        if len(targets) < len(keys):
            ocard = next(c for t in card.tables if t.name == r.table for c in t.columns if c.name == other)
            targets = _ids(rng, ocard, len(keys))
        mapping = dict(zip(keys, targets[: len(keys)]))
        frame[other] = frame[col].map(mapping)
    elif r.kind == "equals" and r.other_columns and r.share:
        k = int(round(r.share * len(frame)))
        rows = rng.choice(len(frame), size=k, replace=False)
        frame.loc[frame.index[rows], col] = frame.loc[frame.index[rows], r.other_columns[0]].to_numpy()
    elif r.kind == "disjoint" and r.other_table in done:
        theirs = set(_norm(done[r.other_table][col]).dropna())
        ccard = next(c for t in card.tables if t.name == r.table for c in t.columns if c.name == col)
        clash = _norm(frame[col]).isin(theirs)
        if clash.any():
            frame[col] = frame[col].astype(object)
            frame.loc[clash, col] = _ids(rng, ccard, int(clash.sum()), theirs | set(_norm(frame[col]).dropna()))
    elif r.kind == "agrees" and r.other_table in done and r.share is not None and pd.api.types.is_numeric_dtype(frame[col]):
        key = next((k for k in card.relations if k.kind == "key" and k.table == r.table and k.other_table == r.other_table), None)
        if key is not None:
            matched = _norm(frame[key.columns[0]]).isin(set(_norm(done[r.other_table][key.other_columns[0]]).dropna()))
            index = frame.index[matched]
            disagree = rng.random(len(index)) >= r.share
            frame.loc[index[disagree], col] = (frame.loc[index[disagree], col] * rng.uniform(0.6, 1.4, size=int(disagree.sum()))).round()
    return frame


__all__ = ["DataCard", "TableCard", "ColumnCard", "Relation", "Problem", "card_of", "column_card", "role_of", "synthesize", "key_named"]
