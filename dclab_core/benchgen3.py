"""Generated decision benchmarks, version 3 (S7 of the 100-students plan, 2026-10-25): homework whose layout never names
its truth, and homework for an expert's craft, not only for traps.

Two reviews of benchgen v2 found students that read the generator instead of the table: its split tables always held
three features, so any further column was the planted entity, and its dirty tables made every unique digits-only text
column an identifier. Version 3 draws every layout apart from every truth and plants each trap beside its
counter-example:

- :func:`craft_table`: a table whose outcome follows one planted **structure** (a linear signal, an interaction, a
  threshold, a high-cardinality category, a seasonal date, a skewed outcome, informative missing values, or none), with
  the task, the size, the plain columns and the missing values drawn apart from it. Which model family wins on it is not
  planted: the engine's screen measures it. The structure is a fact for analysis, never a label.
- :func:`split_table`: whether rows repeat an entity and whether they carry their own time, with 0 to 6 plain columns of
  every kind; decoys beside the truths: an id-like name that never repeats, an entity under a plain name, a repeating
  category that is no entity, a date that is not the row's time.
- :func:`dirty_table`: each column's cleaning step, with counter-examples: numbers written as plain digits (they *are*
  numbers), digits that are an identifier, codes with leading zeros, versions with many or few values, the country code
  "NA" among two-letter codes and "NA" as a missing value.

Every table is small, seeded, and invented; the generators read no data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VERSION = "benchgen_v3"

STRUCTURES = ("linear", "interaction", "threshold", "high_card", "seasonal", "skewed", "missing_informative", "noise")
TASKS = ("binary", "multiclass", "regression")
PLAIN_NAMES = ("amount", "visits", "tenure", "score_a", "score_b", "weight", "distance", "items", "rating", "age", "spend", "load",
               "level", "share", "count_a", "count_b", "ratio", "index_a", "speed", "size")
CATEGORY_NAMES = ("region", "channel", "device", "segment", "plan", "colour", "source", "branch")
# one pool each for a big category and a date, whether it carries the structure or is a decoy: a name never says which
HIGH_CARD_NAMES = ("merchant", "store_code", "product", "zone", "agent", "seller", "account_code", "item_ref", "postcode_area", "batch")
DATE_NAMES = ("order_date", "visit_date", "booked_at", "created", "joined", "updated_at", "first_seen", "record_date")


def _names(rng: np.random.Generator, k: int, pool: tuple[str, ...], used: set[str]) -> list[str]:
    free = [n for n in pool if n not in used]
    picked = [str(n) for n in rng.choice(np.array(free, dtype=object), min(k, len(free)), replace=False)] if free and k else []
    used.update(picked)
    while len(picked) < k:  # the pool ran out: numbered names, still saying nothing of the truth
        name = f"x_{len(used):02d}"
        used.add(name)
        picked.append(name)
    return picked


def _plain(rng: np.random.Generator, n: int, used: set[str], numbers: int, categories: int) -> dict[str, Any]:
    cols: dict[str, Any] = {}
    for name in _names(rng, numbers, PLAIN_NAMES, used):
        cols[name] = (rng.gamma(2.0, 10.0, n) if rng.random() < 0.5 else rng.normal(50, 15, n)).round(2)
    for name in _names(rng, categories, CATEGORY_NAMES, used):
        levels = [f"{name[:2]}{i}" for i in range(int(rng.integers(2, 9)))]
        cols[name] = rng.choice(levels, n)
    return cols


def _outcome(rng: np.random.Generator, signal: np.ndarray, task: str, rare: bool = False) -> np.ndarray:
    """The outcome from a signal: a class by its quantile (``rare``: a rare positive class, 2 to 8%), or the signal itself
    with noise."""
    z = (signal - signal.mean()) / (signal.std() + 1e-9) + rng.normal(0, 0.6, len(signal))
    if task == "regression":
        return z.round(4)
    if task == "binary":
        rate = float(rng.uniform(0.02, 0.08)) if rare else float(rng.uniform(0.15, 0.5))
        return (z > np.quantile(z, 1 - rate)).astype(int)
    k = int(rng.integers(3, 6))
    return np.digitize(z, np.quantile(z, np.linspace(0, 1, k + 1)[1:-1]))


def craft_table(seed: int, rows: int | None = None, structure: str | None = None, task: str | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """A table whose outcome follows one planted structure, everything else drawn apart from it. Returns the table, the
    outcome's name and the facts (the structure, the task, the layout's counts)."""
    rng = np.random.default_rng([seed, 25, 10, 25])
    structure = structure or STRUCTURES[int(rng.integers(len(STRUCTURES)))]
    task = task or TASKS[int(rng.integers(len(TASKS)))]
    if structure not in STRUCTURES or task not in TASKS:
        raise ValueError(f"structure in {STRUCTURES}, task in {TASKS}")
    n = int(rows or rng.integers(600, 4000))
    used: set[str] = set()
    signal_cols = _names(rng, 3, PLAIN_NAMES, used)
    x = {c: rng.normal(0, 1, n) for c in signal_cols}
    a, b, c = (x[k] for k in signal_cols)
    cols: dict[str, Any] = {k: (v * float(rng.uniform(5, 20)) + float(rng.uniform(0, 100))).round(3) for k, v in x.items()}
    if structure == "linear":
        signal = 1.2 * a - 0.8 * b + 0.5 * c
    elif structure == "interaction":
        signal = np.sign(a) * np.sign(b) * 1.5 + 0.2 * c  # the classic XOR: no line separates it
    elif structure == "threshold":
        signal = (a > 0.6).astype(float) * 2 - (b < -0.4).astype(float) * 1.5 + ((c > 0) & (a < 0)).astype(float)
    elif structure == "high_card":
        levels = int(rng.integers(150, 400))
        name = _names(rng, 1, HIGH_CARD_NAMES, used)[0]
        codes = rng.integers(0, levels, n)
        effect = rng.normal(0, 1.5, levels)
        cols[name] = np.array([f"L{v:04d}" for v in codes])
        signal = effect[codes] + 0.3 * a
    elif structure == "seasonal":
        name = _names(rng, 1, DATE_NAMES, used)[0]
        days = rng.integers(0, 730, n)
        when = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")
        cols[name] = when.strftime("%Y-%m-%d")
        signal = np.sin(2 * np.pi * when.dayofyear.to_numpy() / 365.25) * 1.5 + (when.dayofweek.to_numpy() >= 5) * 1.0 + 0.3 * a
    elif structure == "skewed":
        signal = np.exp(0.9 * a + 0.4 * b) + 0.2 * c  # a few rows dominate the scale
        if task == "regression":
            cols_target = np.round(signal * float(rng.uniform(50, 500)) + rng.gamma(1.0, 5.0, n), 2)
    elif structure == "missing_informative":
        name = signal_cols[0]
        gone = rng.random(n) < 1 / (1 + np.exp(-(1.5 * b)))  # who leaves the field empty says something about the outcome
        cols[name] = np.where(gone, np.nan, cols[name])
        signal = 1.4 * gone.astype(float) + 0.5 * c
    else:  # noise: nothing in the table predicts the outcome
        signal = rng.normal(0, 1, n)
    structure_cols = set(cols) - set(signal_cols)  # the structure's own column (a category, a date), if it planted one
    plain_numbers, plain_categories = int(rng.integers(0, 7)), int(rng.integers(0, 4))
    cols.update(_plain(rng, n, used, plain_numbers, plain_categories))
    decoys = []  # the review of S7: a big category and a date in any structure, with no effect, so neither says the structure
    if rng.random() < 0.35:
        name = _names(rng, 1, HIGH_CARD_NAMES, used)[0]
        cols[name] = np.array([f"L{v:04d}" for v in rng.integers(0, int(rng.integers(150, 400)), n)])
        decoys.append("high_card_no_effect")
    if rng.random() < 0.35:
        name = _names(rng, 1, DATE_NAMES, used)[0]
        cols[name] = (pd.Timestamp("2022-06-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")).strftime("%Y-%m-%d")
        decoys.append("date_no_effect")
    free = [k for k in cols if k not in signal_cols and k not in structure_cols]
    gaps = [str(k) for k in rng.choice(np.array(free, dtype=object), min(len(free), int(rng.integers(0, 3))), replace=False)] if free else []
    for name in gaps:  # missing values apart from any truth, on columns drawn at random
        values = pd.Series(cols[name], dtype=object if isinstance(cols[name][0], str) else float)
        values[rng.random(n) < float(rng.uniform(0.02, 0.2))] = np.nan
        cols[name] = values.to_numpy()
    target = _names(rng, 1, ("outcome", "target", "label", "result", "y", "response"), used)[0]
    y = cols_target if structure == "skewed" and task == "regression" else _outcome(rng, signal, task, rare=structure == "skewed")
    frame = pd.DataFrame(cols)
    frame = frame[[str(k) for k in rng.permutation(np.array(list(frame.columns), dtype=object))]]  # an order that says nothing
    frame[target] = y
    return frame, target, {"structure": structure, "task": task, "rows": n, "plain_numbers": plain_numbers,
                           "plain_categories": plain_categories, "decoys": decoys, "missing_in": gaps, "version": VERSION}


# ------------------------------------------------------------------------------------------------ the split's truths

def split_table(seed: int, rows: int = 300) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Whether rows repeat an entity and carry their own time; plain columns of every kind (0 to 6) and decoys drawn apart
    from both truths: an id-like name that never repeats, an entity under a plain name, a repeating category that is no
    entity, a date that is not the row's time."""
    rng = np.random.default_rng([seed, 25, 10, 2])
    n = rows
    has_time, has_entity = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    used: set[str] = set()
    cols = _plain(rng, n, used, int(rng.integers(1, 6)), int(rng.integers(0, 3)))
    decoys = []
    if has_time:
        name = _names(rng, 1, ("order_time", "created_at", "visit_date", "event_time", "placed_at"), used)[0]
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
        cols[name] = np.sort(start + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h"))
    drawn = None
    if has_entity:
        plain = rng.random() < 0.4
        name = _names(rng, 1, ("owner", "shop", "driver", "member") if plain else ("customer_id", "account_id", "store_id", "user_id"), used)[0]
        entities = int(rng.integers(max(10, n // 12), max(11, n // 3)))
        drawn = rng.integers(0, entities, n)
        cols[name] = [f"E{v:04d}" for v in drawn]
        decoys += ["entity_plain_name"] if plain else []
    if rng.random() < 0.5:  # an id-like name that never repeats: no entity
        cols[_names(rng, 1, ("ticket_id", "ref_id", "order_no", "line_id"), used)[0]] = [f"T{v:06d}" for v in rng.permutation(n)]
        decoys.append("unique_id_named_id")
    if rng.random() < 0.5:  # a repeating category that is no entity: few values, many rows each
        cols[_names(rng, 1, ("tier", "status", "grade", "zone_code"), used)[0]] = rng.choice(["A", "B", "C", "D"], n)
        decoys.append("repeating_category")
    if rng.random() < 0.5:  # a date that is never when the row happened
        name = _names(rng, 1, ("birth_date", "license_expiry", "signup_date"), used)[0]
        days = rng.integers(0, 20000, (int(drawn.max()) + 1) if drawn is not None else n)
        cols[name] = pd.Timestamp("1960-01-01") + pd.to_timedelta(days[drawn] if drawn is not None else days, unit="D")
        decoys.append("date_not_row_time")
    cols["outcome"] = rng.random(n) < 0.3
    frame = pd.DataFrame(cols)
    frame = frame[[str(k) for k in rng.permutation(np.array(list(frame.columns), dtype=object))]]
    truth = "time_group" if has_time and has_entity else "time" if has_time else "group" if has_entity else "stratified"
    return frame, {"split": truth, "has_row_time": has_time, "has_entity": has_entity, "decoys": decoys, "version": VERSION}


# ------------------------------------------------------------------------------------------------ cleaning

CLEAN_STEPS = ("missing_markers", "numbers", "dates", "booleans", "trim", "none")
KINDS = ("markers", "text_numbers", "plain_digit_numbers", "text_dates", "yesno", "padded", "clean_number", "clean_text",
         "country_codes", "leading_zeros", "versions_few", "versions_many", "digit_ids", "date_like_id")


def dirty_table(seed: int, rows: int = 200) -> tuple[pd.DataFrame, dict[str, str]]:
    """Each column's needed cleaning step, or none; every trap beside its counter-example (plain digits that are numbers
    beside digits that are an identifier, few and many versions)."""
    rng = np.random.default_rng([seed, 25, 10, 4])
    n = rows
    pool = list(rng.permutation(["info", "detail", "entry", "attribute", "field_a", "field_b", "item", "note", "record_value",
                                 "data_1", "data_2", "col_x", "property", "measure_1", "tag", "value_b", "reading", "entry_2"]))
    cols: dict[str, Any] = {}
    truth: dict[str, str] = {}

    def put(values: Any, step: str) -> None:
        name = str(pool.pop())  # one shared pool: a column's name never says its step
        cols[name], truth[name] = values, step

    for kind in rng.choice(np.array(KINDS, dtype=object), size=int(rng.integers(4, 8)), replace=False):
        if kind == "markers":
            vals = rng.normal(100, 20, n).round(2).astype(object)
            vals[rng.random(n) < 0.15] = str(rng.choice(["N/A", "-", "?", "null", "NA"]))
            put(vals, "missing_markers")
        elif kind == "text_numbers":
            put([f"{v:,.2f}" if rng.random() < 0.5 else f"${v:,.2f}" for v in rng.gamma(2, 900, n)], "numbers")
        elif kind == "plain_digit_numbers":  # numbers written as plain digits, unique or nearly: they *are* numbers
            put([str(int(v)) for v in rng.gamma(2, 400, n)] if rng.random() < 0.5 else [f"{v:.2f}" for v in rng.gamma(2, 40, n)], "numbers")
        elif kind == "text_dates":
            base = pd.Timestamp("2023-01-01")
            put([(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 700, n)], "dates")
        elif kind == "yesno":
            put(rng.choice(["yes", "no"], n), "booleans")
        elif kind == "padded":
            put([f"  {v} " if rng.random() < 0.6 else str(v) for v in rng.choice(["paris", "lyon", "nice", "lille"], n)], "trim")
        elif kind == "clean_number":
            put(rng.normal(40, 10, n).round(1), "none")
        elif kind == "clean_text":
            put(rng.choice(["red", "green", "blue", "black"], n), "none")
        elif kind == "country_codes":  # "NA" is Namibia here, among other two-letter codes
            put(rng.choice(["NA", "ZA", "BW", "ZM", "KE", "NG"], n), "none")
        elif kind == "leading_zeros":
            put([f"{v:05d}" for v in rng.integers(0, 9999, n)], "none")
        elif kind == "versions_few":
            put(rng.choice(["1.9", "1.10", "1.11", "2.0", "2.1"], n), "none")
        elif kind == "versions_many":
            put([f"{int(a)}.{int(b)}.{int(c)}" for a, b, c in zip(rng.integers(1, 4, n), rng.integers(0, 20, n), rng.integers(0, 30, n))], "none")
        elif kind == "digit_ids":  # digits that are an identifier: unique, of one length
            put([str(v) for v in rng.permutation(np.arange(10_000_000, 10_000_000 + n))], "none")
        else:  # date_like_id
            put([f"2024{int(m):02d}{int(d):02d}{int(k):03d}" for m, d, k in zip(rng.integers(1, 13, n), rng.integers(1, 28, n), rng.integers(0, 999, n))], "none")
    return pd.DataFrame(cols), truth


__all__ = ["VERSION", "STRUCTURES", "TASKS", "CLEAN_STEPS", "craft_table", "split_table", "dirty_table"]
