"""Synthetic tables from a description, for users who cannot share their data.

A user with confidential data can still design a DCLab solution. The Home agent (a model when one is
configured, otherwise a keyword-matched template) writes a small schema, :class:`SyntheticSpec`; the code
here validates it and draws seeded rows. The model only proposes the schema; the validators and the
generator decide what is produced (CLAUDE.md rule 5). No real data row is ever sent to the model.

Every output is labelled synthetic (``LABELS``): in :func:`describe`, in ``synthetic_spec.json`` and in the
parquet file's own schema metadata. Scores on synthetic rows show whether a pipeline runs and behaves
sensibly; they say nothing about real-world performance and never make a model production-ready.

How the target is made
----------------------
``effects`` map columns to weights in a linear predictor ``eta``:

- numeric and integer columns (key ``"col"``) are z-scored on the generated values, so a weight is
  "per standard deviation";
- boolean columns (key ``"col"``, or ``"col=true"`` / ``"col=false"``) enter as 0/1;
- categorical columns take one key per value, ``"col=value"``, entering as a 0/1 indicator;
- id, text and datetime columns cannot carry effects.

*Binary*: a latent logistic model, positive when ``eta / noise + e > t`` with ``e ~ Logistic(0, 1)``.
With ``positive_rate`` set, the threshold ``t`` (minus the intercept) is solved on the sample itself: the
``round(rows * positive_rate)`` rows with the highest latent score are positive. This is the intercept
calibration done exactly instead of in expectation, so even a 100-row table hits the requested rate.
Without ``positive_rate`` the intercept is ``base`` (``t = -base / noise``). Class 1 of ``classes`` is
the positive class (default ``["no", "yes"]``).

*Multiclass*: class 0 is the reference. Classes ``k = 1..K-1`` get a seeded offset drawn from N(0, 0.5)
and the centred predictor scaled by ``k / (K-1)``; each row's class is drawn from the softmax of
``(offset_k + k / (K-1) * eta) / noise`` (Gumbel-max). A positive weight therefore moves rows toward the
classes later in the list.

*Regression*: ``y = base + eta + N(0, noise)``, in target units.

Missing values are applied *after* the target is computed: the target reflects the true underlying values
and the gaps are missing completely at random, as when a sensor or a form field drops a value that still
existed. With ``time_order`` the rows are sorted by the first datetime column before ids and gaps are
assigned, so ids increase with time.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MIN_ROWS, MAX_ROWS = 100, 200_000
MODEL_MAX_COLUMNS = 30
LABELS = {
    "synthetic": True,
    "warning": "Synthetic data generated from a description. Use it to design and test the solution, not to report real performance.",
}

DIST_PARAMS: dict[str, tuple[str, ...]] = {
    "normal": ("mean", "sd"),
    "uniform": ("low", "high"),
    "lognormal": ("mean", "sigma"),
    "poisson": ("lam",),
    "exponential": ("scale",),
}
CLIP_KEYS = ("clip_min", "clip_max")
NUMERIC_TYPES = ("numeric", "integer")
EFFECT_TYPES = ("numeric", "integer", "boolean", "categorical")
TRUE_WORDS, FALSE_WORDS = {"true", "yes", "1"}, {"false", "no", "0"}
DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
DEFAULT_START, DEFAULT_END = "2024-01-01", "2024-12-31"
DEFAULT_VOCAB = ("billing", "delivery", "pricing", "support", "the app", "the contract", "quality",
                 "an account change", "a refund", "the website")
TEXT_TEMPLATES = ("Asked about {a}.", "Reported a problem with {a}.", "Mentioned {a} and {b}.",
                  "Follow-up on {a}.", "Positive comment on {a}.", "Requested a change to {a} after {b}.")


def _timestamp(value: str, where: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(str(value).strip())
    except (ValueError, TypeError, OverflowError):
        stamp = pd.NaT
    if pd.isna(stamp):
        raise ValueError(f"{where} {value!r} is not an ISO date such as 2024-01-31")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    if not 1900 <= stamp.year <= 2100:
        raise ValueError(f"{where} {value!r} is outside the years 1900 to 2100")
    return stamp.as_unit("ns")


def _as_labels(value: Any) -> Any:
    """Category and class labels are text; a model that writes ``[0, 1]`` means ``["0", "1"]``."""
    if isinstance(value, (list, tuple)):
        return [v.strip() if isinstance(v, str) else str(v) for v in value]
    return value


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _infer_dist(params: dict[str, float]) -> str | None:
    keys = set(params) - set(CLIP_KEYS)
    for dist, required in DIST_PARAMS.items():
        if keys == set(required):
            return dist
    return None


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    type: Literal["numeric", "integer", "categorical", "boolean", "datetime", "text", "id"]
    description: str = Field(default="", max_length=600)
    dist: Literal["normal", "uniform", "lognormal", "poisson", "exponential"] | None = None
    params: dict[str, float] = Field(default_factory=dict)
    categories: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    null_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    p_true: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("categories", mode="before")
    @classmethod
    def categories_as_text(cls, value: Any) -> Any:
        return _as_labels(value)

    @model_validator(mode="after")
    def check_column(self):
        label = f"Column {self.name!r}"
        if self.type in NUMERIC_TYPES:
            self._check_distribution(label)
        elif self.dist is not None or self.params:
            raise ValueError(f"{label} is {self.type}: dist and params apply only to numeric and integer columns")

        if self.type == "categorical" and not 2 <= len(self.categories) <= 50:
            raise ValueError(f"{label} is categorical: give 2 to 50 categories")
        if self.type == "text" and len(self.categories) > 50:
            raise ValueError(f"{label} is text: give at most 50 words or topics as categories")
        if self.type not in ("categorical", "text") and (self.categories or self.weights):
            raise ValueError(f"{label} is {self.type}: categories and weights apply only to categorical and text columns")
        if any(not c for c in self.categories):
            raise ValueError(f"{label} has an empty category")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"{label} lists the same category twice")
        if self.weights:
            if len(self.weights) != len(self.categories):
                raise ValueError(f"{label} has {len(self.weights)} weights for {len(self.categories)} categories; give one weight per category, or none")
            if any(not math.isfinite(w) or w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError(f"{label}: weights must be zero or positive numbers with a positive total")
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        if self.type == "datetime":
            if (self.start is None) != (self.end is None):
                raise ValueError(f"{label} is datetime: give both start and end, or neither")
            if self.start is None:
                self.start, self.end = DEFAULT_START, DEFAULT_END
            if _timestamp(self.start, f"{label} start") >= _timestamp(self.end, f"{label} end"):
                raise ValueError(f"{label} starts at {self.start} and ends at {self.end}: start must be before end")
        elif self.start is not None or self.end is not None:
            raise ValueError(f"{label} is {self.type}: start and end apply only to datetime columns")

        if self.type == "id" and self.null_rate > 0:
            raise ValueError(f"{label} is an id: ids are never missing, so null_rate must be 0")
        return self

    def _check_distribution(self, label: str) -> None:
        if self.dist is None:
            self.dist = _infer_dist(self.params)
            if self.dist is None:
                raise ValueError(f"{label} is {self.type}: set dist to normal (mean, sd), uniform (low, high), lognormal (mean, sigma), "
                                 "poisson (lam) or exponential (scale), and give those params")
        required = DIST_PARAMS[self.dist]
        missing = [k for k in required if k not in self.params]
        if missing:
            raise ValueError(f"{label} uses a {self.dist} distribution: params must include {', '.join(required)} (missing {', '.join(missing)})")
        unknown = sorted(set(self.params) - set(required) - set(CLIP_KEYS))
        if unknown:
            raise ValueError(f"{label}: unknown params {', '.join(unknown)}; a {self.dist} distribution takes {', '.join(required)} "
                             "and optionally clip_min and clip_max")
        p = self.params
        if any(not math.isfinite(v) or abs(v) > 1e12 for v in p.values()):
            raise ValueError(f"{label}: params must be finite numbers no larger than 1e12")
        if self.dist == "normal" and p["sd"] <= 0:
            raise ValueError(f"{label}: sd must be greater than 0")
        if self.dist == "uniform" and p["low"] >= p["high"]:
            raise ValueError(f"{label}: low must be below high")
        if self.dist == "lognormal" and not 0 < p["sigma"] <= 4:
            raise ValueError(f"{label}: sigma (the spread of the logarithm) must be above 0 and at most 4")
        if self.dist == "lognormal" and abs(p["mean"]) > 30:
            raise ValueError(f"{label}: mean is the mean of the logarithm (exp(11) is about 60,000); keep it between -30 and 30")
        if self.dist == "poisson" and not 0 < p["lam"] <= 1e6:
            raise ValueError(f"{label}: lam must be above 0 and at most 1,000,000")
        if self.dist == "exponential" and p["scale"] <= 0:
            raise ValueError(f"{label}: scale must be greater than 0")
        low, high = p.get("clip_min"), p.get("clip_max")
        if low is not None and high is not None and low >= high:
            raise ValueError(f"{label}: clip_min must be below clip_max")
        if self.type == "integer":
            if self.dist == "uniform" and math.ceil(p["low"]) > math.floor(p["high"]):
                raise ValueError(f"{label}: no whole number lies between low and high")
            if low is not None and high is not None and math.ceil(low) > math.floor(high):
                raise ValueError(f"{label}: no whole number lies between clip_min and clip_max")


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    type: Literal["binary", "multiclass", "regression"]
    description: str = Field(default="", max_length=600)
    classes: list[str] = Field(default_factory=list)
    positive_rate: float | None = None
    effects: dict[str, float] = Field(default_factory=dict)
    noise: float = Field(default=1.0, ge=0.0)
    base: float = 0.0

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("classes", mode="before")
    @classmethod
    def classes_as_text(cls, value: Any) -> Any:
        return _as_labels(value)

    @model_validator(mode="after")
    def check_target(self):
        label = f"Target {self.name!r}"
        if self.type == "binary":
            if not self.classes:
                self.classes = ["no", "yes"]
            if len(self.classes) != 2:
                raise ValueError(f"{label} is binary: give exactly two classes, the negative one first (for example [\"no\", \"yes\"])")
            if self.positive_rate is not None and not 0.005 <= self.positive_rate <= 0.95:
                raise ValueError(f"{label}: positive_rate must be between 0.005 and 0.95")
        else:
            if self.positive_rate is not None:
                raise ValueError(f"{label} is {self.type}: positive_rate applies only to a binary target")
            if self.type == "multiclass" and not 3 <= len(self.classes) <= 20:
                raise ValueError(f"{label} is multiclass: give 3 to 20 classes (two classes make it binary)")
            if self.type == "regression" and self.classes:
                raise ValueError(f"{label} is a regression target: it has no classes")
        if any(not c for c in self.classes) or len(set(self.classes)) != len(self.classes):
            raise ValueError(f"{label}: classes must be distinct, non-empty labels")
        if not math.isfinite(self.noise) or not math.isfinite(self.base):
            raise ValueError(f"{label}: noise and base must be finite numbers")
        if self.type != "regression" and self.noise <= 0:
            raise ValueError(f"{label}: noise must be greater than 0 for a {self.type} target (it sets how strongly the effects decide the class)")
        bad = [key for key, weight in self.effects.items() if not math.isfinite(weight)]
        if bad:
            raise ValueError(f"{label}: effect weights must be finite numbers ({', '.join(bad)})")
        return self


class SyntheticSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)
    rows: int = Field(ge=MIN_ROWS, le=MAX_ROWS)
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    time_order: bool = False
    columns: list[ColumnSpec] = Field(min_length=2, max_length=60)
    target: TargetSpec | None = None

    @model_validator(mode="after")
    def check_spec(self):
        seen: set[str] = set()
        for column in self.columns:
            if column.name.casefold() in seen:
                raise ValueError(f"Two columns are named {column.name!r}; column names must be unique")
            seen.add(column.name.casefold())
        if self.target is not None:
            if self.target.name.casefold() in seen:
                raise ValueError(f"The target {self.target.name!r} has the same name as a column; rename one of them")
            by_name = {c.name: c for c in self.columns}
            for key in self.target.effects:
                _resolve_effect(key, by_name)
        time_column = self.time_column
        if self.time_order and time_column is not None and time_column.null_rate > 0:
            raise ValueError(f"Column {time_column.name!r} orders the rows (time_order is true), so it cannot have missing values")
        return self

    @property
    def time_column(self) -> ColumnSpec | None:
        """The first datetime column: the one ``time_order`` sorts by."""
        return next((c for c in self.columns if c.type == "datetime"), None)


def _resolve_effect(key: str, by_name: dict[str, ColumnSpec]) -> tuple[ColumnSpec, str | None]:
    """``"col"`` or ``"col=value"`` → the column and the value (None for a whole-column effect)."""
    if key in by_name:
        column, value = by_name[key], None
    elif "=" in key and key.split("=", 1)[0].strip() in by_name:
        name, value = (part.strip() for part in key.split("=", 1))
        column = by_name[name]
    else:
        raise ValueError(f"Effect {key!r} names no column in the spec (use \"column\" or \"column=value\")")
    if column.type not in EFFECT_TYPES:
        raise ValueError(f"Effect {key!r} is on the {column.type} column {column.name!r}: effects can use numeric, integer, boolean "
                         "or categorical columns only")
    if column.type == "categorical":
        if value is None:
            raise ValueError(f"Effect {key!r} is on the categorical column {column.name!r}: use one key per value, "
                             f"like \"{column.name}={column.categories[0]}\"")
        if value not in column.categories:
            raise ValueError(f"Effect {key!r}: {value!r} is not one of the categories of {column.name!r}")
    elif column.type == "boolean":
        if value is not None and value.lower() not in TRUE_WORDS | FALSE_WORDS:
            raise ValueError(f"Effect {key!r}: a boolean column takes \"{column.name}\", \"{column.name}=true\" or \"{column.name}=false\"")
    elif value is not None:
        raise ValueError(f"Effect {key!r}: a {column.type} column takes its name alone as the key (\"{column.name}\")")
    return column, value


def problems(error: ValidationError) -> list[str]:
    """Plain-English lines from a pydantic error, for the UI and for the model's retry."""
    lines = []
    for item in error.errors():
        message = str(item.get("msg", "")).removeprefix("Value error, ")
        where = ".".join(str(part) for part in item.get("loc", ()))
        lines.append(f"{where}: {message}" if where else message)
    return lines


# --------------------------------------------------------------------------------------- generation

def generate(spec: SyntheticSpec) -> pd.DataFrame:
    """Seeded rows for ``spec``: the same spec always gives the same frame.

    Columns come in spec order with the target last. Integers are ``Int64`` and booleans ``boolean``
    (both nullable), datetimes ``datetime64[ns]``, everything else text (``object``).
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.rows
    clean = {c.name: _draw(c, n, rng) for c in spec.columns if c.type != "id"}
    target = _draw_target(spec, clean, n, rng) if spec.target is not None else None

    time_column = spec.time_column
    if spec.time_order and time_column is not None:
        order = np.argsort(clean[time_column.name], kind="stable")
        clean = {name: values[order] for name, values in clean.items()}
        target = target[order] if target is not None else None

    data: dict[str, Any] = {}
    for column in spec.columns:
        values = _ids(column.name, n) if column.type == "id" else clean[column.name]
        data[column.name] = _with_gaps(column, values, _missing(column.null_rate, n, rng))
    if target is not None:
        data[spec.target.name] = target
    return pd.DataFrame(data)


def _draw(column: ColumnSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    if column.type in NUMERIC_TYPES:
        return _draw_number(column, n, rng)
    if column.type == "categorical":
        index = rng.choice(len(column.categories), size=n, p=column.weights or None)
        return np.asarray(column.categories, dtype=object)[index]
    if column.type == "boolean":
        return rng.random(n) < column.p_true
    if column.type == "datetime":
        start, end = _timestamp(column.start, "start"), _timestamp(column.end, "end")
        if DATE_ONLY.fullmatch(column.start.strip()) and DATE_ONLY.fullmatch(column.end.strip()):
            offsets = rng.integers(0, (end - start).days, endpoint=True, size=n) * 86_400
        else:
            offsets = rng.integers(0, int((end - start).total_seconds()), endpoint=True, size=n)
        return start.value + offsets.astype(np.int64) * 10**9
    return _draw_text(column, n, rng)


def _draw_number(column: ColumnSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    p, integer = column.params, column.type == "integer"
    if column.dist == "normal":
        values = rng.normal(p["mean"], p["sd"], n)
    elif column.dist == "uniform":
        values = (rng.integers(math.ceil(p["low"]), math.floor(p["high"]), endpoint=True, size=n).astype(float)
                  if integer else rng.uniform(p["low"], p["high"], n))
    elif column.dist == "lognormal":
        values = rng.lognormal(p["mean"], p["sigma"], n)
    elif column.dist == "poisson":
        values = rng.poisson(p["lam"], n).astype(float)
    else:
        values = rng.exponential(p["scale"], n)
    low, high = p.get("clip_min", -np.inf), p.get("clip_max", np.inf)
    if integer:
        values = np.round(values)
        low, high = (math.ceil(low) if math.isfinite(low) else low), (math.floor(high) if math.isfinite(high) else high)
    values = np.clip(values, low, high)
    return np.clip(values, -1e15, 1e15).astype(np.int64) if integer else np.round(values, 4)


def _draw_text(column: ColumnSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    vocab = np.asarray(column.categories or DEFAULT_VOCAB, dtype=object)
    weights = column.weights or None
    first, second = rng.choice(len(vocab), size=n, p=weights), rng.choice(len(vocab), size=n, p=weights)
    if len(vocab) > 1:
        second = np.where(second == first, (second + 1) % len(vocab), second)
    template = rng.integers(0, len(TEXT_TEMPLATES), size=n)
    out = np.empty(n, dtype=object)
    for index, pattern in enumerate(TEXT_TEMPLATES):
        rows = template == index
        out[rows] = [pattern.format(a=a, b=b) for a, b in zip(vocab[first[rows]], vocab[second[rows]])]
    return out


def _ids(name: str, n: int) -> np.ndarray:
    prefix = next((ch.upper() for ch in name if ch.isascii() and ch.isalpha()), "R")
    width = max(6, len(str(n)))
    return np.asarray([f"{prefix}{i:0{width}d}" for i in range(1, n + 1)], dtype=object)


def _missing(rate: float, n: int, rng: np.random.Generator) -> np.ndarray | None:
    count = int(round(n * rate))
    if count == 0:
        return None
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=count, replace=False)] = True
    return mask


def _with_gaps(column: ColumnSpec, values: np.ndarray, missing: np.ndarray | None):
    mask = missing if missing is not None else np.zeros(len(values), dtype=bool)
    if column.type == "integer":
        return pd.arrays.IntegerArray(values.astype(np.int64), mask)
    if column.type == "boolean":
        return pd.arrays.BooleanArray(values.astype(bool), mask)
    if column.type == "numeric":
        out = values.astype(float).copy()
        out[mask] = np.nan
        return out
    if column.type == "datetime":
        out = values.astype("datetime64[ns]").copy()
        out[mask] = np.datetime64("NaT")
        return out
    out = values.copy()
    out[mask] = None
    return out


def _effect_feature(column: ColumnSpec, value: str | None, values: np.ndarray) -> np.ndarray:
    if column.type in NUMERIC_TYPES:
        x = values.astype(float)
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else np.zeros(len(x))
    if column.type == "boolean":
        flags = values.astype(bool)
        if value is not None and value.lower() in FALSE_WORDS:
            flags = ~flags
        return flags.astype(float)
    return (values == value).astype(float)


def _draw_target(spec: SyntheticSpec, clean: dict[str, np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    target = spec.target
    by_name = {c.name: c for c in spec.columns}
    eta = np.zeros(n)
    for key, weight in target.effects.items():
        column, value = _resolve_effect(key, by_name)
        eta += weight * _effect_feature(column, value, clean[column.name])

    if target.type == "regression":
        return np.round(target.base + eta + rng.normal(0.0, target.noise, n), 4)

    if target.type == "binary":
        latent = eta / target.noise + rng.logistic(0.0, 1.0, n)
        if target.positive_rate is None:
            positive = latent + target.base / target.noise > 0
        else:
            count = min(n - 1, max(1, int(round(n * target.positive_rate))))
            positive = np.zeros(n, dtype=bool)
            positive[np.argsort(-latent, kind="stable")[:count]] = True
        return np.where(positive, target.classes[1], target.classes[0]).astype(object)

    k = len(target.classes)
    offsets = np.concatenate([[0.0], rng.normal(0.0, 0.5, k - 1)])
    slopes = np.arange(k) / (k - 1)
    logits = (offsets[None, :] + (eta - eta.mean())[:, None] * slopes[None, :]) / target.noise
    choice = np.argmax(logits + rng.gumbel(size=(n, k)), axis=1)
    return np.asarray(target.classes, dtype=object)[choice]


# --------------------------------------------------------------------------------------- description

def describe(spec: SyntheticSpec, frame: pd.DataFrame) -> dict[str, Any]:
    """A JSON-safe summary of a generated table, always carrying the synthetic label."""
    rows = int(len(frame))
    columns = []
    for column in spec.columns:
        nulls = int(frame[column.name].isna().sum())
        columns.append({"name": column.name, "type": column.type, "dtype": str(frame[column.name].dtype),
                        "description": column.description, "nulls": nulls, "null_rate": round(nulls / max(rows, 1), 4)})
    time_column = spec.time_column
    return {
        "name": spec.name,
        "description": spec.description,
        "rows": rows,
        "n_columns": int(frame.shape[1]),
        "columns": columns,
        "seed": spec.seed,
        "time_order": spec.time_order,
        "time_column": time_column.name if time_column is not None and spec.time_order else None,
        "target": _describe_target(spec, frame),
        "labels": dict(LABELS),
        "spec": spec.model_dump(mode="json"),
    }


def _describe_target(spec: SyntheticSpec, frame: pd.DataFrame) -> dict[str, Any] | None:
    target = spec.target
    if target is None or target.name not in frame:
        return None
    values = frame[target.name]
    out: dict[str, Any] = {"name": target.name, "type": target.type}
    if target.type == "regression":
        numbers = values.astype(float)
        out.update(mean=round(float(numbers.mean()), 4), sd=round(float(numbers.std()), 4),
                   min=round(float(numbers.min()), 4), max=round(float(numbers.max()), 4))
        return out
    shares = values.value_counts(normalize=True)
    out["classes"] = list(target.classes)
    out["distribution"] = {label: round(float(shares.get(label, 0.0)), 4) for label in target.classes}
    if target.type == "binary":
        out.update(positive_label=target.classes[1], positive_rate=out["distribution"][target.classes[1]],
                   requested_positive_rate=target.positive_rate)
    return out


def save(frame: pd.DataFrame, spec: SyntheticSpec, directory: Path) -> dict[str, Any]:
    """Write ``synthetic.parquet`` (label in its schema metadata) and ``synthetic_spec.json`` (spec + labels)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, spec_path = directory / "synthetic.parquet", directory / "synthetic_spec.json"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"dclab.synthetic"] = json.dumps(LABELS).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)
    spec_path.write_text(json.dumps({"labels": LABELS, "rows": int(len(frame)), "columns": [str(c) for c in frame.columns],
                                     "spec": spec.model_dump(mode="json")}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "spec_path": str(spec_path), "rows": int(len(frame)), "columns": int(frame.shape[1])}


# --------------------------------------------------------------------------------------- templates

def _spec(data: dict[str, Any]) -> SyntheticSpec:
    return SyntheticSpec.model_validate(data)


TEMPLATES: dict[str, SyntheticSpec] = {
    "churn": _spec({
        "name": "Telco customer churn (synthetic)",
        "description": "Monthly subscribers of a telecom operator; predict who cancels in the next month.",
        "rows": 5000,
        "columns": [
            {"name": "customer_id", "type": "id"},
            {"name": "tenure_months", "type": "integer", "dist": "uniform", "params": {"low": 0, "high": 72}},
            {"name": "monthly_charges", "type": "numeric", "dist": "normal", "params": {"mean": 70, "sd": 25, "clip_min": 18, "clip_max": 120}},
            {"name": "contract", "type": "categorical", "categories": ["month-to-month", "one-year", "two-year"], "weights": [0.55, 0.24, 0.21]},
            {"name": "internet_service", "type": "categorical", "categories": ["dsl", "fiber", "none"], "weights": [0.34, 0.44, 0.22]},
            {"name": "payment_method", "type": "categorical", "null_rate": 0.01,
             "categories": ["electronic check", "mailed check", "bank transfer", "credit card"], "weights": [0.34, 0.23, 0.22, 0.21]},
            {"name": "senior_citizen", "type": "boolean", "p_true": 0.16},
            {"name": "paperless_billing", "type": "boolean", "p_true": 0.59},
            {"name": "support_calls_90d", "type": "integer", "dist": "poisson", "params": {"lam": 1.2}},
            {"name": "signup_date", "type": "datetime", "start": "2019-01-01", "end": "2024-12-31"},
        ],
        "target": {"name": "churned", "type": "binary", "classes": ["no", "yes"], "positive_rate": 0.265,
                   "effects": {"tenure_months": -1.0, "monthly_charges": 0.5, "contract=one-year": -0.8, "contract=two-year": -1.6,
                               "internet_service=fiber": 0.6, "payment_method=electronic check": 0.5, "support_calls_90d": 0.4,
                               "senior_citizen": 0.3}},
    }),
    "fraud": _spec({
        "name": "Card transaction fraud (synthetic)",
        "description": "Card transactions with a rare fraud label (about 0.5%); flag fraudulent ones at authorisation time.",
        "rows": 5000,
        "columns": [
            {"name": "transaction_id", "type": "id"},
            {"name": "transaction_time", "type": "datetime", "start": "2025-01-01T00:00:00", "end": "2025-06-30T23:59:59"},
            {"name": "amount", "type": "numeric", "dist": "lognormal", "params": {"mean": 3.6, "sigma": 1.0, "clip_max": 10000}},
            {"name": "merchant_category", "type": "categorical",
             "categories": ["grocery", "fuel", "restaurants", "online retail", "travel", "electronics", "gaming"],
             "weights": [0.28, 0.14, 0.18, 0.2, 0.06, 0.09, 0.05]},
            {"name": "channel", "type": "categorical", "categories": ["chip", "contactless", "online", "manual entry"], "weights": [0.35, 0.33, 0.28, 0.04]},
            {"name": "card_present", "type": "boolean", "p_true": 0.68},
            {"name": "foreign_transaction", "type": "boolean", "p_true": 0.05},
            {"name": "distance_from_home_km", "type": "numeric", "dist": "exponential", "params": {"scale": 15, "clip_max": 5000}},
            {"name": "transactions_last_24h", "type": "integer", "dist": "poisson", "params": {"lam": 3}},
            {"name": "account_age_days", "type": "integer", "dist": "uniform", "params": {"low": 1, "high": 3650}},
        ],
        "target": {"name": "is_fraud", "type": "binary", "classes": ["no", "yes"], "positive_rate": 0.005,
                   "effects": {"amount": 0.8, "channel=online": 0.9, "channel=manual entry": 1.2, "merchant_category=electronics": 0.6,
                               "merchant_category=gaming": 0.8, "card_present": -0.7, "foreign_transaction": 1.0,
                               "distance_from_home_km": 0.6, "transactions_last_24h": 0.5, "account_age_days": -0.6}},
    }),
    "demand": _spec({
        "name": "Daily store demand (synthetic)",
        "description": "Daily unit sales per store and product category over two years; forecast units sold.",
        "rows": 5000,
        "time_order": True,
        "columns": [
            {"name": "date", "type": "datetime", "start": "2023-01-01", "end": "2024-12-31"},
            {"name": "store", "type": "categorical", "categories": ["north", "south", "east", "west", "central"]},
            {"name": "product_category", "type": "categorical", "categories": ["grocery", "household", "beverages", "snacks"],
             "weights": [0.4, 0.2, 0.2, 0.2]},
            {"name": "promotion", "type": "boolean", "p_true": 0.2},
            {"name": "public_holiday", "type": "boolean", "p_true": 0.03},
            {"name": "unit_price", "type": "numeric", "dist": "normal", "params": {"mean": 4.5, "sd": 1.2, "clip_min": 0.5, "clip_max": 15}},
            {"name": "temperature_c", "type": "numeric", "dist": "normal", "params": {"mean": 15, "sd": 8}, "null_rate": 0.02},
        ],
        "target": {"name": "units_sold", "type": "regression", "base": 140, "noise": 15,
                   "effects": {"promotion": 35, "public_holiday": 25, "unit_price": -18, "temperature_c": 6, "store=central": 20,
                               "store=north": -10, "product_category=beverages": 15, "product_category=snacks": 8}},
    }),
    "credit": _spec({
        "name": "Loan default (synthetic)",
        "description": "Personal loan applications; predict which loans default within the first year.",
        "rows": 5000,
        "columns": [
            {"name": "applicant_id", "type": "id"},
            {"name": "application_date", "type": "datetime", "start": "2021-01-01", "end": "2024-06-30"},
            {"name": "age", "type": "integer", "dist": "normal", "params": {"mean": 41, "sd": 12, "clip_min": 18, "clip_max": 80}},
            {"name": "annual_income", "type": "numeric", "dist": "lognormal", "params": {"mean": 10.8, "sigma": 0.5}, "null_rate": 0.03},
            {"name": "loan_amount", "type": "numeric", "dist": "lognormal", "params": {"mean": 9.6, "sigma": 0.7, "clip_min": 500, "clip_max": 100000}},
            {"name": "loan_term_months", "type": "categorical", "categories": ["36", "60"], "weights": [0.7, 0.3]},
            {"name": "credit_score", "type": "integer", "dist": "normal", "params": {"mean": 680, "sd": 70, "clip_min": 300, "clip_max": 850}},
            {"name": "debt_to_income", "type": "numeric", "dist": "normal", "params": {"mean": 0.32, "sd": 0.12, "clip_min": 0, "clip_max": 1}},
            {"name": "employment_years", "type": "integer", "dist": "exponential", "params": {"scale": 6, "clip_max": 45}, "null_rate": 0.02},
            {"name": "home_ownership", "type": "categorical", "categories": ["rent", "mortgage", "own"], "weights": [0.45, 0.4, 0.15]},
            {"name": "purpose", "type": "categorical",
             "categories": ["debt consolidation", "home improvement", "car", "education", "small business", "other"],
             "weights": [0.45, 0.15, 0.12, 0.08, 0.06, 0.14]},
            {"name": "previous_defaults", "type": "integer", "dist": "poisson", "params": {"lam": 0.2}},
        ],
        "target": {"name": "defaulted", "type": "binary", "classes": ["no", "yes"], "positive_rate": 0.12,
                   "effects": {"credit_score": -1.2, "debt_to_income": 0.8, "loan_amount": 0.4, "annual_income": -0.5,
                               "previous_defaults": 0.7, "employment_years": -0.3, "purpose=small business": 0.6,
                               "home_ownership=own": -0.3, "loan_term_months=60": 0.4}},
    }),
    "maintenance": _spec({
        "name": "Machine failure from sensor readings (synthetic)",
        "description": "Hourly sensor readings from factory machines; predict a failure in the next 7 days.",
        "rows": 5000,
        "time_order": True,
        "columns": [
            {"name": "reading_id", "type": "id"},
            {"name": "reading_time", "type": "datetime", "start": "2024-01-01T00:00:00", "end": "2024-12-31T23:00:00"},
            {"name": "machine_type", "type": "categorical", "categories": ["pump", "compressor", "conveyor", "press"]},
            {"name": "temperature_c", "type": "numeric", "dist": "normal", "params": {"mean": 70, "sd": 8}, "null_rate": 0.02},
            {"name": "vibration_mm_s", "type": "numeric", "dist": "lognormal", "params": {"mean": 1.0, "sigma": 0.4}, "null_rate": 0.02},
            {"name": "pressure_bar", "type": "numeric", "dist": "normal", "params": {"mean": 6, "sd": 0.8, "clip_min": 0}, "null_rate": 0.02},
            {"name": "rpm", "type": "integer", "dist": "normal", "params": {"mean": 1500, "sd": 200, "clip_min": 0}},
            {"name": "operating_hours", "type": "integer", "dist": "uniform", "params": {"low": 0, "high": 40000}},
            {"name": "days_since_service", "type": "integer", "dist": "exponential", "params": {"scale": 60, "clip_max": 730}},
        ],
        "target": {"name": "failure_next_7d", "type": "binary", "classes": ["no", "yes"], "positive_rate": 0.04,
                   "effects": {"temperature_c": 0.7, "vibration_mm_s": 1.1, "pressure_bar": -0.3, "operating_hours": 0.6,
                               "days_since_service": 0.8, "machine_type=press": 0.4}},
    }),
}

TEMPLATE_KEYWORDS: dict[str, tuple[str, ...]] = {  # order breaks ties; churn is also the default
    "churn": ("churn", "cancel", "retention", "attrition", "subscriber", "subscription", "unsubscrib", "leave"),
    "fraud": ("fraud", "chargeback", "scam", "card transaction", "transaction", "suspicious", "anomal"),
    "demand": ("demand", "forecast", "sales", "units sold", "inventory", "time series", "time-series", "timeseries",
               "daily", "weekly", "footfall"),
    "credit": ("credit", "loan", "default", "lending", "borrower", "mortgage", "underwriting", "repay"),
    "maintenance": ("maintenance", "failure", "machine", "sensor", "equipment", "breakdown", "downtime", "vibration",
                    "iot", "fault"),
}


def template_for(prompt: str) -> tuple[str, SyntheticSpec]:
    """The template whose keywords the prompt mentions most (churn when none match). Returns a copy."""
    text = (prompt or "").lower()
    scores = {key: sum(1 for word in words if re.search(r"\b" + re.escape(word), text)) for key, words in TEMPLATE_KEYWORDS.items()}
    best = max(scores, key=lambda key: scores[key])  # max keeps the first key on ties
    key = best if scores[best] > 0 else "churn"
    return key, TEMPLATES[key].model_copy(deep=True)


# --------------------------------------------------------------------------------------- the model

SYSTEM_PROMPT = """You design a synthetic table for DCLab. The user cannot share their real data, so DCLab \
generates seeded rows from your schema to design and test a solution; the table is always labelled synthetic.

Reply with ONE JSON object and nothing else:
{"name": str, "description": str, "seed": int, "time_order": bool, "columns": [column, ...], "target": target or null}

column = {"name": str (1-60 chars, unique), "type": "numeric"|"integer"|"categorical"|"boolean"|"datetime"|"text"|"id",
  "description": str, "null_rate": 0-0.5 (share of missing values; 0 for ids), plus by type:
  numeric/integer: "dist" and "params": normal {"mean","sd"} | uniform {"low","high"} | lognormal {"mean","sigma"} \
(of the logarithm) | poisson {"lam"} | exponential {"scale"}; optional "clip_min", "clip_max" in params
  categorical: "categories": [2-50 strings], optional "weights": [same length]
  boolean: "p_true": 0-1
  datetime: "start", "end": ISO dates, start before end
  text: optional "categories" as topic words; id: nothing else}

target = {"name": str (not a column name), "type": "binary"|"multiclass"|"regression", "description": str,
  "classes": binary: two labels, negative first; multiclass: 3-20 labels (positive weights push toward later classes),
  "positive_rate": binary only, 0.005-0.95,
  "effects": {"column": weight} for numeric, integer (per standard deviation) and boolean columns, \
{"column=value": weight} for a categorical value; never on id, text or datetime columns,
  "noise": > 0 (regression: noise sd in target units), "base": regression intercept}

Rules:
- Realistic column names, units and ranges for the domain; at most 30 columns.
- No names, emails, phone numbers, addresses or other personal data of real people; use an id column instead.
- Only columns known at the moment the prediction is made: nothing computed from the target or recorded after it.
- Include a target only if the problem has something to predict; otherwise "target": null.
- Set time_order to true for forecasting and time-series problems (rows ordered by the first datetime column).
- DCLab chooses the number of rows; do not set it."""

_ROW_KEYS = {"rows", "records", "sample", "samples", "preview", "head", "data", "frame", "dataframe", "values", "examples", "table"}
_DROP = object()


def _plain(value: Any, depth: int = 0) -> Any:
    """Scalars and small nested lists/dicts of scalars; anything table-like is dropped."""
    if value is None or isinstance(value, (bool, int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if isinstance(value, str):
        return value[:1000]
    if depth >= 2:
        return _DROP
    if isinstance(value, (list, tuple)):
        items = [_plain(v, depth + 1) for v in list(value)[:20]]
        return [v for v in items if v is not _DROP]
    if isinstance(value, dict):
        items = {str(k)[:80]: _plain(v, depth + 1) for k, v in list(value.items())[:30] if str(k).lower() not in _ROW_KEYS}
        return {k: v for k, v in items.items() if v is not _DROP}
    return _DROP  # DataFrames, arrays, series and other objects never reach the model


def _safe_context(context: dict | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    out = _plain(context, depth=0)
    while out and len(json.dumps(out, ensure_ascii=False)) > 4000:
        out.pop(next(reversed(out)))
    return out


def _first_json_object(text: str) -> dict[str, Any]:
    text = (text or "")[:60_000]
    candidates = [m.group(1) for m in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("The reply contained no JSON object")


def _clamp_rows(rows: Any) -> int:
    try:
        rows = int(rows)
    except (TypeError, ValueError):
        rows = 5000
    return min(MAX_ROWS, max(MIN_ROWS, rows))


def _parse_reply(content: str, rows: int) -> SyntheticSpec:
    data = _first_json_object(content)
    if "columns" not in data and isinstance(data.get("spec"), dict):
        data = data["spec"]
    data = {**data, "rows": rows}
    if isinstance(data.get("columns"), list) and len(data["columns"]) > MODEL_MAX_COLUMNS:
        raise ValueError(f"The spec has {len(data['columns'])} columns; use at most {MODEL_MAX_COLUMNS}")
    try:
        return SyntheticSpec.model_validate(data)
    except ValidationError as error:
        raise ValueError("; ".join(problems(error))) from None


def spec_from_model(client: Any, prompt: str, context: dict | None = None, rows: int = 5000) -> tuple[SyntheticSpec, dict[str, Any]]:
    """Ask the model for a spec (one request, one retry); fall back to a template.

    The model sees the user's prompt and a scalar-only ``context`` (problem sentence, answers so far);
    table-like values are dropped, so no data row is ever sent. ``rows`` is clamped to 100–200,000 and
    overrides whatever the model wrote.
    """
    rows = _clamp_rows(rows)
    if client is None:
        key, spec = template_for(prompt)
        return spec.model_copy(update={"rows": rows}), {"source": "template", "template": key}

    request = f"Problem: {(prompt or '').strip()[:4000]}"
    context = _safe_context(context)
    if context:
        request += f"\nContext from the conversation: {json.dumps(context, ensure_ascii=False)}"
    request += "\nDesign the synthetic table as one JSON object."
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]

    reason = ""
    for attempt in (1, 2):
        content = None
        try:
            reply = client.complete(messages, max_tokens=4000)
            content = str((reply or {}).get("content") or "")
            spec = _parse_reply(content, rows)
            return spec, {"source": "model", "attempts": attempt}
        except Exception as error:  # noqa: BLE001 — any failure means one retry, then the template
            reason = (str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}")[:600]
        if attempt == 1 and content is not None:
            messages = [*messages, {"role": "assistant", "content": content[:8000]},
                        {"role": "user", "content": f"That spec was not valid: {reason}. Reply with the corrected JSON object only."}]

    key, spec = template_for(prompt)
    return spec.model_copy(update={"rows": rows}), {"source": "template", "template": key, "attempts": 2,
                                                     "fallback_reason": f"The model did not return a valid spec after one retry: {reason}"}


__all__ = ["ColumnSpec", "TargetSpec", "SyntheticSpec", "LABELS", "TEMPLATES", "generate", "describe", "save", "template_for",
           "spec_from_model", "problems"]
