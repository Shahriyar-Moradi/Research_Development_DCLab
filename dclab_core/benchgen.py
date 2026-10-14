"""Generated decision benchmarks (package 22.1): tables with planted columns whose answers are known, and the decisions
DCLab's rules make on them, each recorded with its benchmark label.

:func:`plant` takes a base table (a replica, or any table with an outcome) and adds columns of known kinds: a unique
identifier, a key whose values recur (an entity), a numeric code for categories, a date in time order, free text, a
column written after the outcome, a copy of the outcome, and two parts that add up to it (one a leak, the other alone
not); and negatives with the same signatures (independent, gappy and genuinely predictive numbers, a category of
many levels, counts, a code with many levels, a date that is not when the row happened), so no single band gives a
label away. :func:`obfuscate` renames every
column ``f000``… so a model must learn from bands, not names. :func:`decide` runs the rules (``dclab_core.datacard``)
on a planted table and returns, for every planted column and every point with a planted truth, a
:class:`~dclab_core.ledger.DecisionRecord` (its state in bands, never a value; the rule's answer; ``dataset`` the
table's id) and its benchmark :class:`~dclab_core.ledger.DecisionLabel`. Base columns carry no planted truth and are
not labelled.

Points and their truths:

=========================  =============================================================================
``column.is_identifier``   a unique identifier and a recurring key are identifiers; every other kind is not
``column.semantic_role``   identifier, identifier, categorical_code, datetime, free_text, numeric…
``feature.leakage_suspect``  written after the outcome, a copy, a sum that gives it: yes; the others: no
``column.category_code``   the numeric code: yes; the other numeric kinds: no
``column.time``            the date in time order: yes; the others: no
``column.group``           the recurring key: yes; the others: no
=========================  =============================================================================
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pandas as pd

KINDS = ("identifier", "entity", "code", "date", "text", "after_outcome", "copy", "sum",
         # negatives with the positives' signatures: no single band gives the label away
         "part", "feature", "gappy", "signal", "category", "count", "small_count", "wide_code", "other_date")
NEG = {"column.is_identifier": False, "feature.leakage_suspect": False, "column.time": False, "column.group": False}
TRUTH: dict[str, dict[str, Any]] = {  # kind -> point -> the right answer
    "identifier": {"column.is_identifier": True, "column.semantic_role": "identifier", "feature.leakage_suspect": False,
                   "column.time": False, "column.group": False, "column.category_code": False},
    "entity": {"column.is_identifier": True, "column.semantic_role": "identifier", "feature.leakage_suspect": False,
               "column.time": False, "column.group": True},
    "code": {"column.is_identifier": False, "column.semantic_role": "categorical_code", "feature.leakage_suspect": False,
             "column.time": False, "column.group": False, "column.category_code": True},
    "date": {"column.is_identifier": False, "column.semantic_role": "datetime", "feature.leakage_suspect": False,
             "column.time": True, "column.group": False},
    "text": {"column.is_identifier": False, "column.semantic_role": "free_text", "feature.leakage_suspect": False,
             "column.time": False, "column.group": False},
    "after_outcome": {"column.is_identifier": False, "feature.leakage_suspect": True, "column.time": False, "column.group": False},
    "copy": {"column.is_identifier": False, "feature.leakage_suspect": True, "column.time": False, "column.group": False},
    "sum": {"column.is_identifier": False, "column.semantic_role": "numeric", "feature.leakage_suspect": True,
            "column.time": False, "column.group": False, "column.category_code": False},
    "part": {**NEG, "column.semantic_role": "numeric", "column.category_code": False},  # alone it says nothing of the outcome
    "feature": {**NEG, "column.semantic_role": "numeric", "column.category_code": False},
    "gappy": {**NEG, "column.semantic_role": "numeric", "column.category_code": False},  # empty at random: not a leak
    "signal": {**NEG, "column.semantic_role": "numeric", "column.category_code": False},  # a real input, related to the outcome
    "category": {**NEG, "column.semantic_role": "categorical_code"},
    "count": {**NEG, "column.semantic_role": "numeric", "column.category_code": False},
    "small_count": {**NEG, "column.semantic_role": "numeric", "column.category_code": False},  # few values, still a count
    "wide_code": {**NEG, "column.semantic_role": "categorical_code", "column.category_code": True},  # a code with many levels
    "other_date": {**NEG, "column.semantic_role": "datetime"},  # a date, but not when the row happened
}
NAMES = {  # names as people write them, so the named variant has realistic tokens
    "identifier": ("record_id", "ref_no", "uuid", "row_key"), "entity": ("customer_id", "account_ref", "store_code_id", "user_key"),
    "code": ("region_code", "segment_no", "status_cd", "class_code"), "date": ("created_on", "event_date", "logged_at", "opened"),
    "text": ("note", "comment_text", "description", "remarks"), "after_outcome": ("resolved_by", "final_amount", "closed_reason", "refund_total"),
    "copy": ("outcome_flag", "result_copy", "label_shadow", "status_final"), "sum": ("total_plus", "combined_value", "grand_total", "sum_value"),
    "part": ("adjustment", "offset_amount", "correction", "delta"), "feature": ("score_a", "measure_b", "metric_x", "reading"),
    "gappy": ("optional_value", "survey_score", "rating_given", "extra_metric"), "signal": ("risk_score", "usage_level", "engagement", "activity"),
    "category": ("channel", "colour", "device_type", "product_line"), "count": ("visits", "items_bought", "logins", "calls_made"),
    "small_count": ("children", "returns", "complaints", "late_payments"),
    "wide_code": ("branch_no", "zone_code", "tariff_code", "dept_code"), "other_date": ("birth_date", "first_contact", "license_date", "warranty_end"),
}
WORDS = np.array("the order arrived late again customer asked about refund driver waited near the door item missing box "
                 "broken please call back tomorrow thanks great service slow".split())


def _numeric_outcome(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return pd.to_numeric(y, errors="coerce").astype(float)
    codes, _ = pd.factorize(y.astype(str))
    return pd.Series(codes, index=y.index).astype(float)


def plant(base: pd.DataFrame, target: str, seed: int, rows: int = 1500, kinds: tuple[str, ...] = KINDS) -> tuple[pd.DataFrame, dict[str, str]]:
    """The base table (a sample of ``rows``) with one column of each kind planted; returns the table and, for each
    planted column, its kind."""
    rng = np.random.default_rng(seed)
    frame = base.sample(min(rows, len(base)), random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    n = len(frame)
    y = _numeric_outcome(frame[target])
    planted: dict[str, str] = {}

    def name(kind: str) -> str:
        pick = NAMES[kind][int(rng.integers(len(NAMES[kind])))]
        while pick in frame.columns:
            pick += "_x"
        planted[pick] = kind
        return pick

    part = rng.normal(0, max(float(y.std()), 1.0), n).round(6)  # drawn first: the sum needs it, wherever "part" is planted
    for kind in kinds:
        if kind == "identifier":
            frame[name(kind)] = rng.permutation(np.arange(10**6, 10**6 + n)) if rng.random() < 0.5 else \
                ["".join(rng.choice(list("0123456789abcdef"), 12)) for _ in range(n)]
        elif kind == "entity":
            entities = int(rng.integers(max(25, n // 40), max(26, n // 6)))
            frame[name(kind)] = [f"E{v:05d}" for v in rng.integers(0, entities, n)]
        elif kind == "code":
            frame[name(kind)] = rng.choice(rng.choice(np.arange(10, 99), int(rng.integers(3, 9)), replace=False), n)
        elif kind == "date":
            start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
            frame[name(kind)] = np.sort(start + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h"))
        elif kind == "text":
            frame[name(kind)] = [" ".join(rng.choice(WORDS, int(rng.integers(8, 16)))) for _ in range(n)]
        elif kind == "after_outcome":  # filled only once the outcome is known to be the rarer value
            rare = y == y.value_counts().idxmin() if y.nunique() <= 20 else y > y.median()
            frame[name(kind)] = np.where(rare, rng.normal(100, 30, n).round(2), np.nan)
        elif kind == "copy":
            frame[name(kind)] = y.to_numpy()
        elif kind == "sum":  # with the "part" column planted beside it, the two add up to the outcome exactly
            frame[name(kind)] = (y - part).round(6)
        elif kind == "part":
            frame[name(kind)] = part
        elif kind == "feature":
            frame[name(kind)] = rng.normal(rng.uniform(-50, 50), rng.uniform(1, 20), n).round(4)
        elif kind == "gappy":
            frame[name(kind)] = np.where(rng.random(n) < rng.uniform(0.05, 0.4), np.nan, rng.normal(10, 3, n).round(3))
        elif kind == "signal":
            z = (y - y.mean()) / (y.std() or 1.0)
            frame[name(kind)] = (0.6 * z + rng.normal(0, 1, n)).round(4)
        elif kind == "category":
            frame[name(kind)] = rng.choice([f"{w}_{i}" for i, w in enumerate(rng.choice(WORDS, 18))][: int(rng.integers(12, 19))], n)
        elif kind == "count":
            frame[name(kind)] = rng.poisson(rng.uniform(5, 200), n)
        elif kind == "small_count":
            frame[name(kind)] = rng.poisson(rng.uniform(0.5, 2.5), n)
        elif kind == "wide_code":
            frame[name(kind)] = rng.choice(rng.choice(np.arange(100, 999), int(rng.integers(12, 19)), replace=False), n)
        elif kind == "other_date":
            frame[name(kind)] = pd.Timestamp("1960-01-01") + pd.to_timedelta(rng.integers(0, 20000, n), unit="D")
    return frame, planted


def obfuscate(frame: pd.DataFrame, planted: Mapping[str, str], target: str, seed: int = 0) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Every column renamed ``f000``… in an order shuffled by ``seed``: a hidden name never says the kind (a column's
    position would), and a model must read the bands."""
    order = [str(c) for c in np.random.default_rng(seed).permutation(np.array(frame.columns, dtype=object))]
    names = {c: f"f{i:03d}" for i, c in enumerate(order)}
    return frame.rename(columns=names), {names[c]: k for c, k in planted.items()}, names[target]


def table_id(base: str, seed: int, obfuscated: bool) -> str:
    return f"{base}-s{seed}{'-ob' if obfuscated else ''}"


def decide(frame: pd.DataFrame, planted: Mapping[str, str], target: str, table: str, at: str,
           group: Optional[str] = None) -> tuple[list[Any], list[Any]]:
    """The rules' decisions on the planted columns, with their benchmark labels. ``group``: the dataset the record
    names (by default the table): a table and its hidden-name twin hold the same data and must share one, so the twin
    never sits in another split."""
    from dclab_core.bands import cardinality_band, name_tokens, ratio_band
    from dclab_core.datacard import card_of
    from dclab_core.ledger import POINTS, Answer, DecisionLabel, DecisionRecord

    card = card_of({table: frame}, name=table, target=target)
    columns = {c.name: c for c in card.tables[0].columns}
    records, labels = [], []
    for column, kind in planted.items():
        c = columns.get(column)
        if c is None:
            continue
        rule = {"column.is_identifier": c.purpose in ("identifier", "entity"), "column.semantic_role": c.role,
                "feature.leakage_suspect": c.purpose == "leak_suspect", "column.time": c.purpose == "time",
                "column.group": c.purpose == "entity", "column.category_code": c.role == "categorical_code" and c.dtype in ("integer", "float")}
        state = {"column": column[:200], "name_tokens": name_tokens(column) or None, "dtype": c.dtype, "uniqueness": c.uniqueness,
                 "nulls": c.nulls, "cardinality": c.cardinality, "value_pattern": c.value_pattern, "target": target[:200]}
        state = {k: v for k, v in state.items() if v is not None}
        for point, truth in TRUTH[kind].items():
            if point not in POINTS:
                continue
            value = rule[point]
            digest = hashlib.sha256(f"{table}|{column}|{point}".encode()).hexdigest()[:20]
            options = list(POINTS[point].choices) if POINTS[point].answer == "choice" else ["yes", "no"]
            record = DecisionRecord(id=f"dg{digest}", point=point, subject={"kind": "column", "key": column[:200]}, state=state,
                                    options=options, answers=[Answer(by="rule", answerer="datacard:purposes", value=value)],
                                    value_used=value, used="rule", dataset=(group or table)[:80], at=at)
            records.append(record)
            labels.append(DecisionLabel(id=f"bl{digest}", record=record.id, point=point, value=truth, source="benchmark",
                                        by=f"generated:benchgen_v1:{kind}", at=at))
    return records, labels


__all__ = ["KINDS", "TRUTH", "plant", "obfuscate", "decide", "table_id"]
