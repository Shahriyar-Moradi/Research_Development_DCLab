"""The query plan: what may be asked of a table (package H6).

```json
{"version": 1, "source": "active_asset",
 "filters": [{"column": "status", "op": "in", "value": ["paid"]}],
 "group_by": ["region"],
 "period": {"column": "order_date", "grain": "month", "from": "2026-01-01", "to": "2026-03-31"},
 "measures": [{"name": "revenue", "op": "sum", "column": "amount"}, {"name": "orders", "op": "count"}],
 "sort": [{"by": "revenue", "desc": true}], "limit": 50}
```

A ``rate`` measure is the share of rows where its ``where`` condition holds (one filter); ``pXX`` is a percentile
(``p90``). A period with a grain groups by that grain of its column; ``from`` and ``to`` bound it (inclusive dates).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = 1
FILTER_OPS = ("eq", "ne", "in", "not_in", "gt", "ge", "lt", "le", "between", "is_null", "not_null")
MEASURE_OPS = ("count", "count_distinct", "sum", "mean", "median", "min", "max", "rate")
PERCENTILE = re.compile(r"^p(\d{1,2})$")
GRAINS = ("day", "week", "month", "quarter", "year")
MAX_GROUPS = 3
MAX_LIMIT = 500
NUMERIC_OPS = ("sum", "mean", "median")  # and every percentile
NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ .-]{0,62}$")


def plan_hash(plan: dict[str, Any]) -> str:
    """A short fingerprint of a plan as validated (the same question gives the same hash)."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()[:16]


def measure_columns(plan: dict[str, Any]) -> list[str]:
    return [m["column"] for m in plan.get("measures") or [] if m.get("column")]


def related(plan: dict[str, Any]) -> list[list[str]]:
    """The pairs of columns a plan relates: a column grouped or filtered on, against a column measured (a rate's
    condition column counts as measured). This is what the exposure log keeps."""
    sides = list(dict.fromkeys([*(plan.get("group_by") or []), *[f["column"] for f in plan.get("filters") or []],
                                *([plan["period"]["column"]] if plan.get("period") else [])]))
    measured = list(dict.fromkeys([*measure_columns(plan), *[m["where"]["column"] for m in plan.get("measures") or [] if m.get("where")]]))
    return sorted({tuple(sorted((a, b))) for a in sides for b in measured if a != b})  # type: ignore[return-value]
