"""Synthetic data, the second way: the person describes the table in the chat, sees its design, and approves it before it
is created (the owner, 2026-10-20: "1 build by AI, 2 describe the data and continue the chatting, get approval, then
create the data").

``propose`` designs the table from everything the person has described so far (the model when one serves the purpose,
else a built-in template, said so), keeps the design on the draft and puts it in the conversation as a message the
person approves, changes by describing more, or drops. Nothing is created until ``approve``; the table is then made
from the stored design exactly, never designed again. No row of any table is involved at any step: the design is
column names, kinds and rates.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from . import synthetic

APPROVE = ("approve", "approved", "yes", "yep", "ok", "okay", "create it", "create the data", "looks good", "go ahead", "do it", "generate it",
           "build it", "confirm")
MAX_DESCRIPTION = 4000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def approves(text: str) -> bool:
    """A reply that approves the design: a short yes, not a change (a change names something to do differently)."""
    t = " ".join((text or "").lower().replace("!", " ").replace(".", " ").split())
    return bool(t) and len(t) <= 40 and any(t == w or t.startswith(w + " ") or t.endswith(" " + w) for w in APPROVE)


def summary(spec: synthetic.SyntheticSpec, info: dict[str, Any], description: str) -> dict[str, Any]:
    """What the person is shown: the table's name, size, columns with their kinds, the outcome and its rate. No data."""
    target = spec.target
    return {"name": spec.name, "rows": spec.rows, "source": info.get("source"), "description": description[:600],
            "columns": [{"name": c.name, "type": c.type, **({"categories": c.categories[:8]} if c.type == "categorical" else {}),
                         **({"missing": c.null_rate} if c.null_rate else {})} for c in spec.columns],
            "target": ({"name": target.name, "type": target.type, **({"positive_rate": target.positive_rate} if target.positive_rate is not None else {}),
                        **({"classes": target.classes[:8]} if target.classes else {})} if target else None),
            "time_order": spec.time_order, "note": synthetic.template_note(spec, info)}


def propose(drafts: Any, agent: Any, client: Any, draft_id: str, description: str, rows: int) -> dict[str, Any]:
    """The person's description joins the conversation; the design it gives, with what was said before, is kept and said."""
    description = (description or "").strip()[:MAX_DESCRIPTION]
    if not description:
        raise ValueError("Describe the table first: its rows, columns, the outcome and how often it happens.")
    draft = drafts.get(draft_id)
    kept = draft.get("synthetic_proposal") or {}
    said = [*((kept.get("said") or [])), description][-6:]  # a change adds to the description, it does not replace it
    full = " ".join(said)[:MAX_DESCRIPTION]
    message = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "text", "text": description, "at": now()}
    drafts.update(draft_id, lambda d: d["messages"].append(message))
    drafts.emit(draft_id, "chat", message)
    context = {"problem": draft["problem"], "understanding": draft.get("understanding") or {}, "pack": (draft.get("pack") or {}).get("key")}
    spec, info = synthetic.spec_from_model(client, full, context, rows)
    shown = summary(spec, info, full)
    proposal = {"id": "sp" + secrets.token_hex(5), "spec": spec.model_dump(mode="json"), "info": info, "said": said, "rows": spec.rows, "at": now(), "summary": shown}
    drafts.update(draft_id, lambda d: d.update(synthetic_proposal=proposal))
    agent.say(draft_id, f"Here is the table I would create: {spec.rows:,} rows of “{spec.name}”. Approve it to create the data, or tell me what to change.",
              kind="synthetic_spec", proposal={"id": proposal["id"], **shown}, actions=["approve_synthetic", "change_synthetic"])
    return proposal


def stored(drafts: Any, draft_id: str, proposal_id: str) -> tuple[synthetic.SyntheticSpec, dict[str, Any]]:
    """The design the person approved, from the draft: only the one whose id was approved, validated again."""
    kept = drafts.get(draft_id).get("synthetic_proposal") or {}
    if not kept or kept.get("id") != proposal_id:
        raise KeyError("That design is not the one on this draft any more: describe the table again.")
    return synthetic.SyntheticSpec.model_validate(kept["spec"]), {**(kept.get("info") or {}), "approved": True, "chosen_by": "user"}


def close(drafts: Any, draft_id: str) -> None:
    """The design is used or dropped: it leaves the draft (the conversation keeps what was said)."""
    drafts.update(draft_id, lambda d: d.pop("synthetic_proposal", None))
