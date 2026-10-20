"""Each role's model on one page (package 21.4): who serves it, with which prompt, how it measured and what it cost.

For every role (``agents.harness.ROLES``) and agent purpose: its own tier and the tier serving it now (routed by a
reviewer, with a reason, in ``models/routing.py``), the model and endpoint behind that tier, the version of the prompt
it is sent (a content hash), its monthly cap and this month's spend by model, whether the product asks it (its gate,
``roles.status``), and its measured score by model on the stored conversation runs. No keys, no prompts' text.
"""

from __future__ import annotations

from typing import Any


def roles(gateway: Any) -> list[dict[str, Any]]:
    from ..agents import harness
    from ..extensions import optional
    from . import routing, settings
    from .usage import month_start

    doc = gateway.routing.get() if gateway.routing is not None else routing.empty()
    spent = (gateway.usage.totals(month_start()) if gateway.usage is not None else {}).get("by_purpose_model") or {}
    try:
        from .. import roles as judged_roles

        gate = {r["role"]: r for r in judged_roles.status()}
    except Exception:  # noqa: BLE001 - the judged table is optional here
        gate = {}
    conversation = optional("agent_eval.conversation")  # the open edition has no measurement: no score shown
    measured = conversation.by_model() if conversation is not None else {}
    purposes = [*harness.ROLES, *(p for a in harness.AGENTS.values() for p in a.purposes)]
    out = []
    for purpose in purposes:
        own = settings.purpose(purpose).tier
        current = routing.route(doc, purpose)
        serving = current["serve"] if current["serve"] and routing.approved(current) else own
        public = settings.public(serving)
        prompt = routing.prompt_of(purpose)
        scores = [{k: x[k] for k in ("run", "split", "model", "right", "of", "cost_eur", "prompt")} for x in measured if x["role"] == purpose]
        out.append({"role": purpose, "kind": "role" if purpose in harness.ROLES else "agent", "own_tier": own, "serving_tier": serving,
                    "model": public.get("model"), "endpoint": public.get("endpoint"), "local": public.get("local"),
                    "shadow": current["shadow"], "prompt": prompt, "cap_eur_month": routing.cap(doc, purpose),
                    "spent_this_month": [v for v in spent.values() if v.get("purpose") == purpose],
                    "on": (gate.get(purpose) or {}).get("on"), "why": (gate.get(purpose) or {}).get("why"),
                    "understanding": purpose in routing.UNDERSTANDING,
                    "measured": [s for s in scores if s["prompt"] == prompt][-5:],  # at the prompt it is sent now
                    "history": [h for h in (doc.get("history") or []) if h.get("purpose") == purpose][-5:]})
    return out


__all__ = ["roles"]
