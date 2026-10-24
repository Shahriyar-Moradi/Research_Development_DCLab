"""The prediction and outcome log of a model in use (34.9): what was scored, what was decided, and what really happened.

A :class:`PredictionRecord` is one row scored: the model and its version, the row's key, the score, the decision taken,
the batch it came in and the time. An :class:`OutcomeRecord` is the truth about a key once it is known: the outcome,
coded as the model predicts it, when it became known and when it was recorded. The two join by key (:func:`join`), which
is what monitoring with outcomes (34.10) and a retrain (34.11) read.

Neither record ever holds a value of the scored table. The key is a salted hash (:func:`key_hash`, HMAC-SHA-256 with a
secret the service keeps), never the caller's order number or customer id; a caller with no key of its own gets one made
for it (:func:`new_key`), a random number, and keeps it to post the outcome. A decision is a short code (``alert``), never
a sentence; the score is the model's output. Readers refuse any field beyond the contract, so a row cannot slip into a log
line unnoticed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from .contracts import CONTRACTS, Versioned, _version

KEY = r"^[0-9a-f]{32}$"
CODE = r"^[a-z][a-z0-9_]{0,39}$"
MIN_SECRET = 16  # characters: a shorter secret would let a guessable key (an order number) be found by trying them all
Task = Literal["binary", "multiclass", "regression"]


def _iso(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError(f"not an ISO 8601 time: {value!r}") from None
    if moment.tzinfo is None:
        raise ValueError(f"a time needs its zone (UTC): {value!r}")
    return value


class PredictionRecord(Versioned):
    """One row scored by a model in use. Never a value of the row: its key is a salted hash."""

    schema_version: Literal["dclab_core.prediction_record.v1"] = _version("prediction_record")
    model_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.:-]+$", description="The model (a project id).")
    model_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$",
                               description="Which fit of it: a fingerprint of the model file.")
    batch: str = Field(pattern=KEY, description="The request the row came in (rows of a batch were scored together).")
    position: int = Field(ge=0, description="The row's place in its batch.")
    key: str = Field(pattern=KEY, description="The row's key as a salted hash (key_hash), never the caller's value.")
    task: Task
    score: Optional[float] = Field(default=None, description="Binary: the score; multiclass: the chosen class's probability; "
                                                               "regression: the prediction.")
    predicted_class: Optional[int] = Field(default=None, ge=0, description="Multiclass: the chosen class, by its index in the "
                                                                            "model's class labels.")
    decision: Optional[str] = Field(default=None, pattern=CODE, description="The action taken, as a code (alert, no_alert); "
                                                                             "None when the service decides nothing.")
    threshold: Optional[float] = Field(default=None, description="The threshold the decision used, when it used one.")
    at: str = Field(description="When it was scored: ISO 8601 with its zone (UTC).")

    @field_validator("at")
    @classmethod
    def _time(cls, value: str) -> str:
        return _iso(value)

    @model_validator(mode="after")
    def _shape(self) -> "PredictionRecord":
        if (self.task == "multiclass") != (self.predicted_class is not None):
            raise ValueError("a multiclass prediction names its class by index, and only a multiclass one does")
        return self


class OutcomeRecord(Versioned):
    """What really happened to a key: the outcome coded as the model predicts it (:func:`code_outcome`)."""

    schema_version: Literal["dclab_core.outcome_record.v1"] = _version("outcome_record")
    key: str = Field(pattern=KEY, description="The row's key as a salted hash, as in its prediction.")
    outcome: float = Field(description="Binary: 1 positive, 0 not; multiclass: the class's index; regression: the value.")
    known_at: str = Field(description="When the outcome became known (ISO 8601 with its zone).")
    recorded_at: str = Field(description="When it was recorded (ISO 8601 with its zone).")

    @field_validator("known_at", "recorded_at")
    @classmethod
    def _times(cls, value: str) -> str:
        return _iso(value)


CONTRACTS["prediction_record"] = PredictionRecord
CONTRACTS["outcome_record"] = OutcomeRecord


_WHOLE = re.compile(r"^(-?\d+)\.0+$")


def _text(value: Any) -> str:
    """A key or a label as text, the same whichever way it arrived: a CSV's 123.0 (a number, or the text a float column
    was written as) is a JSON's 123; "007" stays "007"."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # a numpy scalar
        return _text(value.item())
    text = str(value)
    whole = _WHOLE.match(text.strip())
    return whole.group(1) if whole else text


def key_hash(key: Any, secret: str) -> str:
    """The key as stored: HMAC-SHA-256 of its text under the service's secret, 32 hex characters. The same key and secret
    always give the same hash (so outcomes join); without the secret, the key cannot be found from it."""
    if hasattr(key, "item") and not isinstance(key, (str, bytes)):  # a numpy scalar
        key = key.item()
    if key is None or (isinstance(key, float) and math.isnan(key)) or (isinstance(key, str) and not key.strip()):
        raise ValueError("a key is missing")
    if isinstance(key, bool) or not isinstance(key, (str, int, float)):
        raise ValueError("a key is a text or a number")
    if not secret or len(secret) < MIN_SECRET:
        raise ValueError(f"the secret must have at least {MIN_SECRET} characters")
    return hmac.new(secret.encode(), _text(key).encode(), hashlib.sha256).hexdigest()[:32]


def new_key() -> str:
    """A key made for a row its caller gave none: random, 32 hex characters, returned to the caller to post the outcome."""
    return secrets.token_hex(16)


def code_outcome(value: Any, task: str, *, positive_label: Any = None, negative_label: Any = None,
                 class_labels: Optional[Sequence[Any]] = None) -> float:
    """A true outcome as the model predicts it, or ValueError. Binary: 1 for the positive label, 0 for the other one; 1/0
    and true/false only when the labels are not known or are themselves those; a value that is neither is refused, never
    read as a negative. Multiclass: the class's index in the
    model's labels. Regression: a finite number."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("the outcome is missing")
    text = _text(value)
    if task == "binary":
        if positive_label is not None and text == _text(positive_label):
            return 1.0
        if negative_label is not None and text == _text(negative_label):
            return 0.0
        plain_words = {"1": 1.0, "true": 1.0, "0": 0.0, "false": 0.0}
        plain = plain_words.get(text.strip().lower())
        labels = {_text(label).strip().lower() for label in (positive_label, negative_label) if label is not None}
        if plain is not None and (negative_label is None or labels <= set(plain_words)):
            return plain
        raise ValueError(f"a binary outcome is the positive label ({positive_label}), the other one ({negative_label}), 1 or 0")
    if task == "multiclass":
        labels = [_text(label) for label in class_labels or []]
        if text in labels:
            return float(labels.index(text))
        raise ValueError("a multiclass outcome is one of the model's class labels")
    if task == "regression":
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("a regression outcome is a number") from None
        if not math.isfinite(number):
            raise ValueError("a regression outcome is a finite number")
        return number
    raise ValueError(f"unknown task {task!r}")


def read_log(source: Union[str, Path, Iterable[str]], contract: type[Versioned]) -> list[Any]:
    """A JSONL log read back, each line strictly its contract (its version, no field beyond it); an absent file is empty.
    A line that is not JSON at all is a torn write (a batch still being written, or a writer killed mid-line) and is
    skipped: it is no record. A line that is JSON but not the contract raises ValueError with its line number."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return []
        lines: Iterable[str] = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = source
    out = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except ValueError:
            continue
        try:
            out.append(contract.read(document))
        except (ValueError, TypeError) as error:
            raise ValueError(f"line {number} is not a {contract.__name__}: {str(error)[:300]}") from None
    return out


def join(predictions: Iterable[PredictionRecord], outcomes: Iterable[OutcomeRecord]) -> Any:
    """The predictions that have an outcome, as a table: one row per prediction, with the key's outcome (the one recorded
    last, so a correction wins), when it became known and the delay in seconds from scoring to knowing. A prediction with
    no outcome yet, and an outcome with no prediction, are left out."""
    import pandas as pd

    latest: dict[str, OutcomeRecord] = {}
    for outcome in outcomes:
        held = latest.get(outcome.key)
        if held is None or _moment(outcome.recorded_at) >= _moment(held.recorded_at):
            latest[outcome.key] = outcome
    rows = []
    for record in predictions:
        outcome = latest.get(record.key)
        if outcome is None:
            continue
        rows.append({"key": record.key, "model_id": record.model_id, "model_version": record.model_version, "batch": record.batch,
                     "position": record.position, "at": record.at, "task": record.task, "score": record.score,
                     "predicted_class": record.predicted_class, "decision": record.decision, "threshold": record.threshold,
                     "outcome": outcome.outcome, "known_at": outcome.known_at,
                     "delay_seconds": (_moment(outcome.known_at) - _moment(record.at)).total_seconds()})
    columns = ["key", "model_id", "model_version", "batch", "position", "at", "task", "score", "predicted_class", "decision",
               "threshold", "outcome", "known_at", "delay_seconds"]
    return pd.DataFrame(rows, columns=columns)


def _moment(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
