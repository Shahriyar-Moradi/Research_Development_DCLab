"""The memory item as a host's row (35.1): the main product has no memory table at the commit its fields were read from
(``main_product_fields.json``), so this is the row it would add, column by column, and the way back.

The vector never travels in the row: a host keeps it in its own index (pgvector, a file), named by ``vector_model``.
Counters are one JSON column. Reading a row goes through :meth:`MemoryItem.read`, so a row that breaks a rule (two
lines, no source, a forged id) is refused, never stored as it is.
"""

from __future__ import annotations

import json
from typing import Any

from dclab_core.memory import MemoryItem

COLUMNS = ("id", "schema_version", "kind", "scope", "scope_id", "text", "source", "who", "status", "correction", "supersedes",
           "retired_by", "counters", "created_at", "last_used_at", "vector_model")


def as_row(item: MemoryItem) -> dict[str, Any]:
    data = item.model_dump()
    row = {c: data[c] for c in COLUMNS if c != "counters"}
    row["counters"] = json.dumps(data["counters"], sort_keys=True)
    return row


def from_row(row: dict[str, Any]) -> MemoryItem:
    data = {c: row.get(c) for c in COLUMNS}
    for c in ("created_at", "last_used_at"):  # a host's timestamp column comes back as a datetime: read as ISO with its zone
        if hasattr(data[c], "isoformat"):
            data[c] = data[c].isoformat()
    data["counters"] = json.loads(row["counters"]) if isinstance(row.get("counters"), str) else (row.get("counters") or {})
    return MemoryItem.read(data)


__all__ = ["COLUMNS", "as_row", "from_row"]
