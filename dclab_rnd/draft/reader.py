"""The conversation reader on Home (20.10 v4): ``dclab_core.conversation.READER`` asked through the gateway's
``conversation_reader`` purpose, reading the problem sentence and the answers that settle the problem card.

Every value it returns quotes the person's own words (``READER``'s parse refuses anything else) and the rules' reading
stands wherever the model is silent, refused or failing. It is on only when ``DCLAB_CONVERSATION_READER=1``, a model
serves the purpose, and the reader passed its sealed measurement (``journeys.reader_passed``: the fourth sealed set, both
arms on the same people, judged by ``GATE_V4``'s rule fixed before the run): a role serves the product only after it
measured better than the rules.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Optional


def switched_on() -> bool:
    return os.environ.get("DCLAB_CONVERSATION_READER", "").strip().lower() in ("1", "on", "yes", "true")


def reader(gateway: Any, *, draft_id: Optional[str] = None, cap_eur: Optional[float] = None,
           force: bool = False, prebuild: bool = False) -> Optional[Callable[[str, str, list[str]], dict[str, str]]]:
    """``(field, text, columns) -> fields the model read`` (only the model's part, validated), or None when the reader is
    off or no model serves it. ``force``: a measurement turns it on for its own run. ``prebuild``: the measurement's
    journeys run their engines in a private scope; its reader's client is made before, so the reader alone is asked.
    The product never prebuilds: a private scope then keeps the reader silent too."""
    if gateway is None or not gateway.available("conversation_reader"):
        return None
    if not force:
        from ..agent_eval.journeys import reader_passed

        if not (switched_on() and reader_passed()):
            return None
    from dclab_core.agents import run_role
    from dclab_core.conversation import READER

    from ..core_advisor import GatewayAdvisor

    advisor = GatewayAdvisor(gateway, draft_id=draft_id, cap_eur=cap_eur, max_tokens=1200)  # the model reasons before its short JSON: 300 cut a quarter of replies
    if prebuild:
        advisor._client("conversation_reader")

    alone = dataclasses.replace(READER, merge=lambda rules, model: model)  # the model's own reading; Home weighs it against the rules

    def read(field: str, text: str, columns: list[str]) -> dict[str, str]:
        result = run_role(alone, advisor, {"field": field, "answer": text, "columns": columns})
        read.asked += 1  # type: ignore[attr-defined]
        if result.source != "advisor" and result.reason == "failed":  # the provider failed or the cap was reached: the rules answered
            read.failures += 1  # type: ignore[attr-defined]
        return dict(result.output or {}) if result.source == "advisor" else {}

    read.asked, read.failures = 0, 0  # type: ignore[attr-defined]  # a measurement stores no reader arm the rules partly played

    read.advisor = advisor  # type: ignore[attr-defined]  # its spend and report, for a measurement
    return read
