"""Thinking and reasoning for the past (the owner, 2026-10-28 night: "make sure all previous data has thinking and
reasoning parts, it's crucial"; plan ~/.claude/plans/deep-meandering-hippo.md, Part C).

Thinking that was never saved cannot be recovered; this module never pretends it can. ``plan`` counts the kept calls that
could be asked again with thinking on (whole, with a reply, no thinking text, consent to train, never exam material) and
what it would cost. The paid replay itself waits for its own package (the review of 2026-10-28): it must re-send the
original inputs (the shaped records have their tables removed), keep the original's split and consent, skip sealed
originals, and label its reasoning "replayed" in the contract, never "recorded".

``python -m dclab_rnd.store backfill``
"""

from __future__ import annotations

from typing import Any, Iterable

from dclab_core.store import StoreRecord

from . import shapers
from .layout import Layout

PRICE_IN, PRICE_OUT = 2.0 / 1_000_000, 10.0 / 1_000_000  # USD per token, claude-sonnet-5-5 (the product's model)
# never replayed: the purposes that write or grade the evaluation people (their outputs are exam material, some sealed; a
# replay would put exam-like text into training data) and the teacher's own calls
NOT_REPLAYED = {"journey_writer": "writes the evaluation people (dev and sealed)", "journey_checker": "grades the evaluation people",
                "reasoning_teacher": "a teacher's call is not an agent's work"}
THINKING_OUT = 800  # the output tokens a replay is planned at: its answer and its thinking
REMOVED = "[a table of"  # the shaper's mark where it removed a printed table: those inputs are not the originals


def _thinking_text(body: dict[str, Any]) -> str:
    return "\n\n".join(str(t.get("text") or t.get("thinking") or "") for t in body.get("thinking") or [] if isinstance(t, dict)).strip()


def replayable(records: Iterable[StoreRecord]) -> list[StoreRecord]:
    """Kept calls that could be asked again: whole, with a reply, no thinking text, consent to train, not sealed, not exam
    material, and with their inputs as sent (no table removed by the shaper)."""
    out = []
    for r in records:
        b = r.body
        if not (r.provenance.source.startswith("model_exchanges/") and b.get("whole") and isinstance(b.get("messages"), list)):
            continue
        if (not b.get("reply") and not b.get("tool_calls")) or _thinking_text(b) or r.provenance.consent != "train":
            continue
        if r.split == "sealed" or str(b.get("purpose")) in NOT_REPLAYED or REMOVED in str(b.get("messages")):
            continue
        out.append(r)
    return out


def estimate(records: list[StoreRecord]) -> float:
    tokens_in = sum(int(((r.body.get("usage") or {}).get("input_tokens")) or 2000) for r in records)
    return round(tokens_in * PRICE_IN + len(records) * THINKING_OUT * PRICE_OUT, 2)


def plan(layout: Layout) -> dict[str, Any]:
    todo = replayable(shapers.records(layout, "model_requests"))
    by_purpose: dict[str, int] = {}
    for r in todo:
        by_purpose[str(r.body.get("purpose"))] = by_purpose.get(str(r.body.get("purpose")), 0) + 1
    return {"replayable": len(todo), "replay_usd": estimate(todo), "by_purpose": dict(sorted(by_purpose.items(), key=lambda kv: -kv[1])),
            "note": "the replay itself waits for its own package (inputs as sent, the original's split and consent, a replayed source)"}


__all__ = ["replayable", "estimate", "plan", "NOT_REPLAYED"]
