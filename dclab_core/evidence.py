"""What may be called evidence: a compact summary, never rows.

One check, used where evidence is stored (``Finding.evidence``) and where it would leave the library (an ``Advisor``).
Only JSON-like values pass: strings, numbers, booleans, None and numpy scalars, inside dicts and lists/tuples, within
limits on size, depth, key length and the total number of values. Everything else (DataFrames, arrays of any shape,
sets, bytes, ranges, generators, dict views) is refused, as is a list of records that look like rows, and any number
that is not finite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Limits:
    items: int = 20        # entries in one list
    keys: int = 30         # entries in one mapping
    key_length: int = 100
    depth: int = 3
    leaves: int = 120      # values in the whole summary
    string: int = 500


FINDING = Limits()
ADVISOR = Limits(items=50, keys=60, depth=4, leaves=600, string=2000)
RECORD_KEYS = 4  # a list of more than this many mappings sharing the same keys is a table


class NotASummary(ValueError):
    """The value would carry rows or raw values."""


def check(value: Any, limits: Limits = FINDING) -> None:
    """Raise :class:`NotASummary` unless ``value`` is a compact JSON-like summary within ``limits``."""
    leaves = [0]

    def walk(v: Any, depth: int) -> None:
        if depth > limits.depth:
            raise NotASummary("nested too deep: a summary, not a table")
        if v is None or isinstance(v, bool):
            leaves[0] += 1
        elif isinstance(v, (int, float, np.integer, np.floating, np.bool_)):
            if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
                raise NotASummary("a number that is not finite (NaN or infinity)")
            leaves[0] += 1
        elif isinstance(v, str):
            if len(v) > limits.string:
                raise NotASummary(f"a string of {len(v)} characters (at most {limits.string})")
            leaves[0] += 1
        elif isinstance(v, Mapping) and not isinstance(v, (np.ndarray,)):
            if len(v) > limits.keys:
                raise NotASummary(f"a mapping of {len(v)} keys (at most {limits.keys}): summarise, never copy values")
            for key, item in v.items():
                if not isinstance(key, str) or len(key) > limits.key_length:
                    raise NotASummary("mapping keys are short strings")
                walk(item, depth + 1)
        elif type(v) in (list, tuple):
            if len(v) > limits.items:
                raise NotASummary(f"a list of {len(v)} items (at most {limits.items}): summarise, never copy rows")
            records = [tuple(sorted(x)) for x in v if isinstance(x, Mapping)]
            if len(records) > RECORD_KEYS and len(set(records)) == 1 and len(records[0]) > 1:
                raise NotASummary("a list of records with the same keys is a table, not a summary")
            for item in v:
                walk(item, depth + 1)
        else:
            raise NotASummary(f"{type(v).__name__} is not evidence: send aggregates, never rows or arrays")
        if leaves[0] > limits.leaves:
            raise NotASummary(f"more than {limits.leaves} values: a summary is smaller")

    walk(value, 0)


def plain(value: Any) -> Any:
    """The same summary with numpy scalars turned into Python numbers (what pandas aggregates return)."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def finite(value: Any) -> Any:
    """Raise ValueError if a float anywhere inside ``value`` is NaN or infinite (JSON cannot hold them)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("a number that is not finite (NaN or infinity) cannot be stored as JSON")
    if isinstance(value, Mapping):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)
    return value
