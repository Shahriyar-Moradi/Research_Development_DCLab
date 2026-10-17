"""The thin adapter: a graph node asks a model through DCLab's gateway and nothing else (28.1).

No LangChain model class, no key in the framework: ``complete`` takes the node's purpose (a name in
``models/settings.py``), goes through ``installed().client(purpose, ...)`` so the purpose's routing, caps, shadows and
the ``model.request`` record apply as for every other caller, and returns the reply, or None when no model serves the
purpose (the node then takes its deterministic path). Tests install a scripted gateway; the framework never sees a key.
"""

from __future__ import annotations

from typing import Any, Optional


def complete(purpose: str, messages: list[dict[str, Any]], *, draft_id: Optional[str] = None, project_id: Optional[str] = None,
             max_tokens: int = 1800, **options: Any) -> Optional[dict[str, Any]]:
    """One completion for ``purpose`` through the gateway: the reply dict (``content``, ``tool_calls``, ``usage``), or None
    when no model serves the purpose here."""
    from ..models import installed

    bound = installed().client(purpose, draft_id=draft_id, project_id=project_id)
    if bound is None:
        return None
    return bound.complete(messages, max_tokens=max_tokens, **options)


def text(purpose: str, messages: list[dict[str, Any]], **kw: Any) -> Optional[str]:
    """The reply's text, or None (no model, or an empty reply)."""
    reply = complete(purpose, messages, **kw)
    content = (reply or {}).get("content")
    return str(content).strip() or None if content else None


__all__ = ["complete", "text"]
