"""The curator served by a model (32.9): Claude (the product's model, through the gateway's ``store_curator`` purpose) places a
batch of records beside the rule curators, in shadow. Its answers are kept and compared, never used, until it measures
better on a person's review of a sample and a person switches it on.

It is shown each record's shape only (:func:`shape_of`): kind, type, flags, point, and its body's field names with their
short codes; never a value's words, a row or a cell. Its reply is validated by ``write.check`` like a rule's; a reply that
does not parse is kept as a failed answer. ``ask(records)`` is what ``curate(model=...)`` calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from dclab_core.store import StoreRecord

PURPOSE = "store_curator"
PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "store_curator.md"
CODE = re.compile(r"^[a-z0-9_.:\-]{1,40}$")
MAX_TOKENS = 6000  # about 150 tokens a placement, two placements a record, a batch of 10 to 20


def shape_of(record: StoreRecord) -> dict[str, Any]:
    """What the model may see of a record: no value's words, only names and short codes."""
    fields = {}
    for key, value in sorted(record.body.items()):
        if isinstance(value, bool) or value is None:
            fields[key] = value
        elif isinstance(value, str) and CODE.match(value):
            fields[key] = value
        else:
            fields[key] = type(value).__name__
    return {"record": record.id, "kind": record.kind, "type": record.type, "flags": record.flags,
            "point": record.body.get("point") if isinstance(record.body.get("point"), str) else None, "fields": fields}


def ask(records: list[StoreRecord], gateway: Any = None, client: Any = None) -> list[dict[str, Any]]:
    """The model's placements of the batch, or [] when no model serves the purpose here. ``client``: one client for a whole run,
    so its ``run_limit_eur`` caps every batch together."""
    from ...models import for_tools

    if client is None:
        gateway = gateway or for_tools()
        client = gateway.client(PURPOSE) if gateway is not None else None
    if client is None:
        return []
    messages = [{"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
                {"role": "user", "content": json.dumps([shape_of(r) for r in records], ensure_ascii=False)}]
    reply = client.complete(messages, max_tokens=MAX_TOKENS)  # a placement or more per record: room for the whole batch
    text = reply.get("content") if isinstance(reply, dict) else str(reply)
    try:
        found = json.loads(re.search(r"\[.*\]", text or "", re.S).group(0))
    except (AttributeError, ValueError):
        return [{"error": "the reply was not a JSON list", "raw": text}]
    return [a for a in found if isinstance(a, dict)]


__all__ = ["ask", "shape_of", "PURPOSE"]
