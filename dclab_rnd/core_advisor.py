"""The R&D host's Advisor for dclab_core roles: each role's pinned prompt, sent through the model gateway (package 18.2).

The gateway does what it does for every purpose: routing to a tier, caps on euros, retries, the usage log and the
data-limit rules of ``models/settings.py`` (``problem_analyst`` may see the description and column names, never values).
The answer must be one JSON object; anything else is no answer, and the role falls back to its rules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Optional


class GatewayAdvisor:
    """``cap_eur`` caps everything this advisor sends (one client per purpose, each with what is left of the cap)."""

    def __init__(self, gateway: Any = None, *, project_id: Optional[str] = None, draft_id: Optional[str] = None,
                 max_tokens: int = 700, cap_eur: Optional[float] = None):
        if gateway is None:
            from dclab_rnd.models import installed

            gateway = installed()
        self.gateway, self.project_id, self.draft_id, self.max_tokens, self.cap_eur = gateway, project_id, draft_id, max_tokens, cap_eur
        self._clients: dict[str, Any] = {}

    def _client(self, purpose: str) -> Any:
        if purpose not in self._clients:
            client = self.gateway.client(purpose, project_id=self.project_id, draft_id=self.draft_id)
            if client is not None and self.cap_eur is not None:
                client.run_limit_eur = max(0.0, self.cap_eur - self.spent_eur())
            self._clients[purpose] = client
        return self._clients[purpose]

    def spent_eur(self) -> float:
        return sum(float(getattr(c, "spent_eur", 0.0) or 0.0) for c in self._clients.values() if c is not None)

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        from dclab_core import prompts

        client = self._client(purpose)
        if client is None:
            return None  # no model serves this purpose here: the role keeps to its rules
        messages = [{"role": "system", "content": prompts.text(purpose)},
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]
        reply = client.complete(messages, None, max_tokens=self.max_tokens)
        if reply.get("finish_reason") == "length":
            raise RuntimeError("the reply was cut at the token limit")  # a failure, not an abstention
        content = (reply.get("content") or "").strip()
        answer = _first_object(content)
        if answer is None and content:
            raise ValueError("the reply holds no JSON object")
        return answer


def _first_object(text: str) -> Optional[dict[str, Any]]:
    """The first JSON object in a reply (a model may wrap it in a code fence)."""
    start = text.find("{")
    while start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        return value if isinstance(value, dict) else None
    return None
