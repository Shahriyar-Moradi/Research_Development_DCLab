"""Home, conversation first (package 37.9, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): with ``DCLAB_CASE_AGENT`` on, Home's
draft conversation is the lead agent's (``agents.case_agent``) instead of the rules-first flow.

:class:`CaseHome` is Home with its conversation replaced: :meth:`start` reads the person's first sentence, :meth:`reply`
their next message (a pick of an option is their choice), :meth:`data_ready` records a table that became ready. Each turn
writes the agent's words as Home's messages, its options as one ``case_options`` message (cards on the page), the case file
on the draft (``draft["case"]``, the side panel the person can read) and the whole state (``draft["case_state"]``) so the
next turn, a restart or another server instance resumes it. A route the agent or the person starts becomes the draft's
kind of help, so the rest of DCLab (briefs, queries, the build) follows. Everything else of Home is unchanged, and with the
switch off nothing here runs: the old flow stays the fallback and the open edition's path.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core import case as C

from ..agents import case_agent as A
from ..agents import case_session
from .chat import HomeAgent


def table_of(store: Any, draft: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Any]:
    """The draft's ready table as the agent sees it (column view and row count, never a value) and the table itself."""
    import pandas as pd

    from . import understand as U

    asset = next((a for a in draft.get("assets") or [] if a["id"] == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is None:
        return None, None
    try:
        frame = pd.read_parquet(store.data_dir(draft["id"]) / asset.get("clean_file", "clean.parquet"))
    except Exception:  # noqa: BLE001 - a table that cannot be read is no table for the agent
        return None, None
    return A.table_of(str(asset.get("name") or "table"), U.columns_view(draft), len(frame)), frame


class CaseHome(HomeAgent):
    """Home whose conversation is the lead agent's (``DCLAB_CASE_AGENT``)."""

    def __init__(self, store: Any, models: Any, **kwargs: Any):
        super().__init__(store, None, on_request=kwargs.pop("on_request", lambda *a: None), **kwargs)
        self.models = models

    def _state(self, draft: dict[str, Any]) -> A.CaseState:
        saved = draft.get("case_state")
        if saved:
            state = A.CaseState(case=C.CaseFile.read(saved["case"]), table=saved.get("table"), computed=list(saved.get("computed") or []),
                                turns=list(saved.get("turns") or []), asking=bool(saved.get("asking")), build=saved.get("build"))
            state.spent_eur, state.cap_eur = float(saved.get("spent_eur") or 0.0), float(saved.get("cap_eur") or case_session.cap_eur())
            state.waiting_on_budget, state.consulted = bool(saved.get("waiting_on_budget")), list(saved.get("consulted") or [])
            case_session.restore(state, saved)  # names seen, exposure and the tracer's place (the review of 37.8)
        else:
            state = A.CaseState(case=C.new(f"draft-{draft['id']}"))
        table, frame = table_of(self.store, draft)
        if table is not None:
            state.table, state.frame = table, frame
        state.specialist_client = self.models.client("case_specialist", draft_id=draft["id"]) if self.models is not None else None
        return state

    def _client(self, draft_id: str) -> Any:
        return self.models.client(A.PURPOSE, draft_id=draft_id) if self.models is not None else None

    def _save(self, draft_id: str, state: A.CaseState, out: dict[str, Any]) -> None:
        snapshot = case_session.snapshot(state)
        view = {**C.views(state.case), "findings": [f.model_dump(mode="json") for f in state.case.findings],
                "route": state.case.route.model_dump(mode="json") if state.case.route else None}

        def put(d: dict[str, Any]) -> None:
            d["case_state"], d["case"] = snapshot, view
            route = state.case.route
            if route is not None and (d.get("intent") or {}).get("track") != route.track:  # the route is the draft's kind of help
                d["intent"] = {**(d.get("intent") or {}), "track": route.track, "status": "chosen" if route.by == "person" else "read",
                               "by": "case_agent", "tool": route.tool}
        self.store.update(draft_id, put)
        if out.get("text"):
            self.say(draft_id, out["text"], kind="text", by="case_agent")
        if out.get("options"):
            self.say(draft_id, "", kind="case_options", options=[o.model_dump(mode="json") for o in out["options"]])

    def _run(self, draft_id: str, words: str, pick: Optional[str] = None) -> None:
        client = self._client(draft_id)
        draft = self.store.get(draft_id)
        state = self._state(draft)
        out = A.pick(state, client, pick) if pick else A.turn(state, client, words)
        self._save(draft_id, state, out)

    def start(self, draft_id: str) -> None:
        if self._client(draft_id) is None:
            return super().start(draft_id)
        self._run(draft_id, str(self.store.get(draft_id).get("problem") or ""))

    def reply(self, draft_id: str, text: str) -> None:
        if self._client(draft_id) is None:  # no model: the rules-first Home answers, as with the switch off
            return super().reply(draft_id, text)
        draft = self.store.get(draft_id)
        offered = [o.get("title") for o in ((draft.get("case_state") or {}).get("case") or {}).get("options") or []]
        if not (draft.get("case_state") or {}).get("case", {}).get("route") and text in offered:
            self.store.update(draft_id, lambda d: d["messages"].append({"role": "user", "kind": "text", "text": text}))
            return self._run(draft_id, text, pick=text)
        self.store.update(draft_id, lambda d: d["messages"].append({"role": "user", "kind": "text", "text": text}))
        self._run(draft_id, text)

    def data_ready(self, draft_id: str, asset: dict[str, Any]) -> None:
        """A table is ready: the case reads it on the next turn; Home says so in one line (never the rules' questions)."""
        draft = self.store.get(draft_id)
        s = (draft.get("analysis") or {}).get("summary") or {}
        self.say(draft_id, f"Your table {asset.get('name', '')} is ready: {s.get('rows', 0):,} rows and {s.get('columns', 0)} columns. "
                           "Tell me what you would like to know from it, or carry on where we were.", kind="text", by="case_agent")


__all__ = ["CaseHome", "table_of"]
