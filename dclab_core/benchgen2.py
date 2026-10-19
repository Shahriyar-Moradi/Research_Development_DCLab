"""Generated decision benchmarks, version 2 (package 29.2): table-level and column-level choices whose answers are
known because they were planted, for the students of choices that 22.1's planted columns do not reach.

Each generator draws one small invented table from a seed and returns it with its truth; the rules' answers are the
host's (the studio calls its own rule on the same table) and a record's state holds bands and counts, never a value.

- :func:`task_table`: the outcome's kind. Binary as booleans, yes/no or 1/2; multiclass as names, small codes, or
  **codes with many levels** (a rule that reads "many whole numbers" as a number is wrong); regression as floats,
  large integers, or **small counts** (a rule that reads "few whole numbers" as classes is wrong).
- :func:`split_table`: how rows must be split. The row's own time (an order placed, a visit) means the future is the
  holdout; an entity that repeats (a customer with several rows) means each stays on one side; both, both; neither,
  a stratified random split. Traps: a **date that is not the row's time** (a birth date, a contract start), an **id
  that never repeats**, an entity under a plain name.
- :func:`cases_table`: whether a table is a list of known cases (the labels of another table) rather than rows to
  score. Traps: a cases list **without** a constant status, a transaction table **with** a reason column, a labels file
  with negatives.
- :func:`dirty_table`: which cleaning step each column needs. Missing markers, numbers written as text, dates as text,
  yes/no, padded text; traps: a **country code "NA"** (Namibia, not missing), **codes with leading zeros**, **versions
  like 1.10** (not numbers), a **date-like id**.

Every table is small (``rows``), seeded, and its facts are what is true by construction; the generators read no data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VERSION = "benchgen_v2"

TASK_KINDS = ("binary_bool", "binary_yesno", "binary_12", "binary_rare", "multiclass_names", "multiclass_codes", "multiclass_many_codes",
              "regression_float", "regression_large_int", "regression_small_counts")
TASK_TRUTH = {"binary_bool": "binary", "binary_yesno": "binary", "binary_12": "binary", "binary_rare": "binary", "multiclass_names": "multiclass",
              "multiclass_codes": "multiclass", "multiclass_many_codes": "multiclass", "regression_float": "regression",
              "regression_large_int": "regression", "regression_small_counts": "regression"}
# one pool of outcome names for every kind (the review of 29.2: names drawn per truth let a student read the truth from the
# name); the kind is in the values, and half the tables hide every name besides (the host's hidden-name twins)
TARGET_NAMES = ("outcome", "target", "label", "result", "response", "value", "measure", "answer", "signal", "score_y", "status_y", "y")
SPLITS = ("stratified", "time", "group", "time_group")
CLEAN_STEPS = ("missing_markers", "numbers", "dates", "booleans", "trim", "none")


def _features(rng: np.random.Generator, n: int, k: int = 3) -> dict[str, Any]:
    names = ("age", "tenure", "score", "spend", "visits_last_month", "region")
    out: dict[str, Any] = {}
    for name in rng.choice(names, size=k, replace=False):
        out[str(name)] = (rng.choice(["north", "south", "east", "west"], n) if name == "region" else rng.normal(50, 15, n).round(2))
    return out


# ------------------------------------------------------------------------------------------------ the outcome's kind

def task_table(seed: int, rows: int = 300, kind: str | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """``kind`` (29.12): a curriculum's choice of the outcome's kind; by default the seed's, as the sets were built."""
    rng = np.random.default_rng([seed, 29, 2, 1])
    if kind is not None and kind not in TASK_KINDS:
        raise ValueError(f"kind is one of {', '.join(TASK_KINDS)}")
    kind = kind or TASK_KINDS[seed % len(TASK_KINDS)]
    truth = TASK_TRUTH[kind]
    target = str(rng.choice(TARGET_NAMES))
    n = rows
    if kind == "binary_bool":
        y: Any = rng.random(n) < rng.uniform(0.2, 0.5)
    elif kind == "binary_yesno":
        y = np.where(rng.random(n) < rng.uniform(0.2, 0.5), "yes", "no")
    elif kind == "binary_12":
        y = np.where(rng.random(n) < rng.uniform(0.2, 0.5), 2, 1)
    elif kind == "binary_rare":  # at least 3 positives: a table with none is constant, not binary
        y = np.where(rng.random(n) < rng.uniform(0.01, 0.05), "yes", "no")
        y[rng.choice(n, 3, replace=False)] = "yes"
    elif kind == "multiclass_names":
        y = rng.choice([f"{w}" for w in ("bronze", "silver", "gold", "platinum", "basic", "pro", "team", "enterprise")][: int(rng.integers(3, 8))], n)
    elif kind == "multiclass_codes":
        y = rng.choice(np.arange(1, int(rng.integers(4, 15))), n)
    elif kind == "multiclass_many_codes":  # category ids: many levels, still classes
        levels = rng.choice(np.arange(100, 999), int(rng.integers(25, 40)), replace=False)
        y = rng.choice(levels, n)
    elif kind == "regression_float":
        y = rng.gamma(2.0, 30.0, n).round(2)
    elif kind == "regression_large_int":
        y = rng.integers(0, 5000, n)
    else:  # regression_small_counts: a count of a few things, still a number to predict
        y = rng.poisson(rng.uniform(1.0, 4.0), n)
    frame = pd.DataFrame({**_features(rng, n), target: y})
    return frame, target, {"kind": kind, "task": truth, "target": target, "version": VERSION}


# ------------------------------------------------------------------------------------------------ the split

def split_table(seed: int, rows: int = 300) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A table whose right split is known: whether it holds the row's own time, a repeating entity, both or neither,
    with decoys (a date that is not the row's time, an id that never repeats, an entity under a plain name)."""
    rng = np.random.default_rng([seed, 29, 2, 2])
    n = rows
    has_time, has_entity = bool(seed % 4 in (1, 3)), bool(seed % 4 in (2, 3))
    cols: dict[str, Any] = _features(rng, n)
    decoys = []
    if has_time:
        name = str(rng.choice(["order_time", "created_at", "visit_date", "event_time", "placed_at"]))
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
        cols[name] = np.sort(start + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h"))
    if has_entity:
        plain = rng.random() < 0.35  # an entity under a plain name
        name = str(rng.choice(["owner", "shop", "driver", "member"])) if plain else str(rng.choice(["customer_id", "account_id", "store_id", "user_id"]))
        entities = int(rng.integers(max(10, n // 12), max(11, n // 3)))
        drawn = rng.integers(0, entities, n)
        cols[name] = [f"E{v:04d}" for v in drawn]
        if plain:
            decoys.append("entity_plain_name")
    else:
        drawn = None
    if rng.random() < 0.5:  # a date that is never when the row happened (a birth date, a licence's expiry): one per entity
        name = str(rng.choice(["birth_date", "license_expiry"]))
        days = rng.integers(0, 20000, (int(drawn.max()) + 1) if drawn is not None else n)
        cols[name] = pd.Timestamp("1960-01-01") + pd.to_timedelta(days[drawn] if drawn is not None else days, unit="D")
        decoys.append("date_not_row_time")
    if rng.random() < 0.6:  # an id that never repeats: not a group
        cols[str(rng.choice(["row_id", "ticket_id", "ref_no"]))] = rng.permutation(np.arange(10_000, 10_000 + n))
        decoys.append("unique_id")
    cols["outcome"] = rng.random(n) < 0.3
    frame = pd.DataFrame(cols)
    truth = "time_group" if has_time and has_entity else "time" if has_time else "group" if has_entity else "stratified"
    return frame, {"split": truth, "has_row_time": has_time, "has_entity": has_entity, "decoys": decoys, "version": VERSION}


# ------------------------------------------------------------------------------------------------ a list of known cases

def cases_table(seed: int, rows: int = 200) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """A table that is, or is not, a list of known cases."""
    rng = np.random.default_rng([seed, 29, 2, 3])
    n = rows
    variant = ("cases", "cases_no_status", "cases_plain_file", "transactions_with_reason", "labels_with_negatives", "plain_table")[seed % 6]
    reasons = rng.choice(["fake_address", "chargeback", "stolen_card", "duplicate_account", "refund_abuse"], int(rng.integers(3, 6)), replace=False)
    key = str(rng.choice(["order_id", "case_id", "transaction_id", "claim_id"]))
    ids = rng.permutation(np.arange(100_000, 100_000 + n))
    if variant in ("cases", "cases_no_status", "cases_plain_file"):
        cols: dict[str, Any] = {key: ids}
        if rng.random() < 0.85:  # the kind of case, under one of its names
            cols[str(rng.choice(["reason", "fraud_type", "case_kind", "cause"]))] = rng.choice(reasons, n)
        if variant != "cases_no_status" and rng.random() < 0.8:
            cols["status"] = str(rng.choice(["CONFIRMED", "DELIVERED", "CLOSED"]))
        if rng.random() < 0.5:
            cols["confirmed_at"] = pd.Timestamp("2024-06-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
        frame = pd.DataFrame(cols)
        named = variant != "cases_plain_file" and rng.random() < 0.8
        name = str(rng.choice(["known_frauds.csv", "confirmed_cases.csv", "fraud_cases.csv"])) if named else str(rng.choice(["export_3.csv", "list.csv", "sheet1.csv"]))
        return frame, name, {"cases": True, "variant": variant, "version": VERSION}
    if variant == "transactions_with_reason":  # a payment's own note, mostly empty, never a fraud's kind: rows to score
        notes = np.where(rng.random(n) < 0.8, None, rng.choice(["card", "bank_transfer", "voucher", "split_payment"], n))
        frame = pd.DataFrame({key: ids, "payment_note": notes, "status": (rng.choice(["OPEN", "CLOSED", "PENDING"], n) if rng.random() < 0.7
                                                                          else str(rng.choice(["SETTLED", "PAID"]))),
                              "amount": rng.gamma(2, 40, n).round(2)})
    elif variant == "labels_with_negatives":  # labelled rows, negatives included: the outcome, not a list of cases
        frame = pd.DataFrame({key: ids, "is_fraud": rng.choice(["yes", "no"], n, p=[0.1, 0.9]),
                              **({"status": "REVIEWED"} if rng.random() < 0.6 else {})})
    else:
        frame = pd.DataFrame({key: ids, **_features(rng, n), **({"type": rng.choice(["a", "b", "c"], n)} if rng.random() < 0.3 else {})})
    name = str(rng.choice(["orders.csv", "export_7.csv", "transactions.csv", "fraud_review.csv"]))
    return frame, name, {"cases": False, "variant": variant, "version": VERSION}


# ------------------------------------------------------------------------------------------------ cleaning

def dirty_table(seed: int, rows: int = 200) -> tuple[pd.DataFrame, dict[str, str]]:
    """A table whose every column needs one known cleaning step, or none, with traps that look dirty and are not."""
    rng = np.random.default_rng([seed, 29, 2, 4])
    n = rows
    cols: dict[str, Any] = {}
    truth: dict[str, str] = {}

    pool = list(rng.permutation(["info", "detail", "entry", "attribute", "field_a", "field_b", "item", "note", "record_value",
                                 "data_1", "data_2", "col_x", "property", "measure_1", "tag"]))

    def put(_hint: str, values: Any, step: str) -> None:
        name = str(pool.pop())  # one shared pool (the review of 29.2): a column's name never says its step
        cols[name] = values
        truth[name] = step

    numbers = rng.normal(100, 20, n).round(2)
    kinds = rng.choice(["markers", "text_numbers", "text_dates", "yesno", "padded", "clean_number", "clean_text",
                        "country_codes", "leading_zeros", "versions", "date_like_id"], size=6, replace=False)
    for kind in kinds:
        if kind == "markers":
            vals = numbers.astype(object)
            vals[rng.random(n) < 0.15] = str(rng.choice(["N/A", "-", "?", "null", "n/a"]))
            put(str(rng.choice(["income", "weight", "balance"])), vals, "missing_markers")
        elif kind == "text_numbers":
            put(str(rng.choice(["revenue", "price", "fee"])), [f"{v:,.2f}" if rng.random() < 0.5 else f"${v:,.2f}" for v in rng.gamma(2, 900, n)], "numbers")
        elif kind == "text_dates":
            base = pd.Timestamp("2023-01-01")
            put(str(rng.choice(["signup", "joined", "opened"])), [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 700, n)], "dates")
        elif kind == "yesno":
            put(str(rng.choice(["subscribed", "is_member", "opted_in"])), rng.choice(["yes", "no"], n), "booleans")
        elif kind == "padded":
            put(str(rng.choice(["city", "plan", "channel"])), [f"  {v} " if rng.random() < 0.6 else str(v) for v in rng.choice(["paris", "lyon", "nice"], n)], "trim")
        elif kind == "clean_number":
            put(str(rng.choice(["age", "tenure", "score"])), rng.normal(40, 10, n).round(1), "none")
        elif kind == "clean_text":
            put(str(rng.choice(["colour", "segment", "device"])), rng.choice(["red", "green", "blue"], n), "none")
        elif kind == "country_codes":  # "NA" is Namibia here, among other country codes: not a missing value
            put("country", rng.choice(["NA", "ZA", "BW", "ZM", "KE", "NG"], n), "none")
        elif kind == "leading_zeros":  # product codes: the zeros are part of the code
            put(str(rng.choice(["product_code", "sku", "zip"])), [f"{v:05d}" for v in rng.integers(0, 9999, n)], "none")
        elif kind == "versions":  # 1.10 is not 1.1
            put("app_version", rng.choice(["1.9", "1.10", "1.11", "2.0", "2.1"], n), "none")
        else:  # date_like_id: digits that read like a date and are an identifier
            put("batch_ref", [f"2024{int(m):02d}{int(d):02d}{int(k):03d}" for m, d, k in zip(rng.integers(1, 13, n), rng.integers(1, 28, n), rng.integers(0, 999, n))], "none")
    return pd.DataFrame(cols), {**truth}


__all__ = ["VERSION", "TASK_KINDS", "TASK_TRUTH", "SPLITS", "CLEAN_STEPS", "task_table", "split_table", "cases_table", "dirty_table"]
