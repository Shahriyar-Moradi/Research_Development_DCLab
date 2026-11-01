"""Home, conversation first (packages 37.9 and 37.11, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): Home's draft conversation is
the lead agent's (``agents.case_agent``) by default wherever a model serves it; ``DCLAB_CASE_AGENT=0`` or no model keeps the
rules-first flow.

:class:`CaseHome` is Home with its conversation replaced: :meth:`start` reads the person's first sentence, :meth:`reply`
their next message (a pick of an option is their choice), :meth:`data_ready` records a table that became ready. Each turn
writes the agent's words as Home's messages, its options as one ``case_options`` message (cards on the page), the case file
on the draft (``draft["case"]``, the side panel the person can read) and the whole state (``draft["case_state"]``) so the
next turn, a restart or another server instance resumes it. A route the agent or the person starts becomes the draft's
kind of help, so the rest of DCLab (briefs, queries, the build) follows. Everything else of Home is unchanged, and with the
switch off nothing here runs: the old flow stays the fallback and the open edition's path.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from dclab_core import case as C

from ..agents import case_agent as A
from ..agents import case_session
from .chat import HomeAgent, turn


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
            if route is None or route.tool not in A.BUILDS:  # the review of 37.11: a build question left open after the route moved on
                d.pop("build_waiting", None)
                d.pop("moment_said", None)
            u = d.setdefault("understanding", {})
            if route is not None and route.tool == "segmentation_project":  # the build reads no outcome for groups (as take_answer records)
                u.update(family="segmentation", family_by="answer")
            elif route is not None and u.get("family") == "segmentation" and u.get("family_by") == "answer":
                u.pop("family", None)  # the review: groups chosen, then a prediction: the build asks for the outcome again
                u.pop("family_by", None)
        self.store.update(draft_id, put)
        if out.get("text"):
            self.say(draft_id, out["text"], kind="text", by="case_agent")
        if out.get("options"):
            self.say(draft_id, "", kind="case_options", options=[o.model_dump(mode="json") for o in out["options"]])

    def _run(self, draft_id: str, words: str, pick: Optional[str] = None) -> None:
        client = self._client(draft_id)
        draft = self.store.get(draft_id)
        state = self._state(draft)
        state.build = None  # the review of 37.11: a yes counts in the turn it is said (a second yes after a refusal builds again)
        out = A.pick(state, client, pick) if pick else A.turn(state, client, words)
        self._save(draft_id, state, out)
        if state.build and state.build.get("tool") in A.BUILDS:
            self._build(draft_id)  # the person's yes starts the build, as "build it" does on the rules-first Home

    BUILD_NEEDS = ("outcome column", "prediction moment")  # "confirm moment" follows the second  # the build's own refusals whose answer code reads (draft/api.py quick_build)

    def _build(self, draft_id: str) -> None:
        """The same build step as the button and the rules-first Home's "build it". What it still needs is asked in the chat:
        a column check as the rules-first Home asks it (``built``), the outcome column or the prediction moment as the build's
        own question, whose answer code reads on the next message (``_build_answer``)."""
        done = self.on_request(draft_id, "build", {})
        if done is None:
            return
        rule = done.get("rule")
        self.store.update(draft_id, lambda d: d.update(build_waiting=rule) if rule in self.BUILD_NEEDS else d.pop("build_waiting", None))
        self.built(draft_id, done)

    # A new moment in the answer: "but", "instead", "rather", "actually", or a leading "no". A bare "not" or "no" later in a yes
    # ("Yes, no other changes") is no correction, and a yes of any length builds (the cloud review of 37.11: "Yes, that is
    # correct, go ahead" was said back as the moment, so a person answering in full sentences looped).
    CORRECTION = re.compile(r"\b(?:but|instead|rather|actually)\b|^\W*(?:no|nope|not)\b", re.IGNORECASE)
    HOLD_BUILD = re.compile(r"\bhold off\b|\b(?:don'?t|do not|not yet|wait|later|never|cancel)\b[^.!?]{0,40}\b(?:build|start|go ahead)",
                            re.IGNORECASE)  # "yes, but hold off on the build": nothing is built and the agent reads it
    LEADING_ANSWER = re.compile(r"^\W*(?:(?:yes|yeah|yep|ok(?:ay)?|sure|no|nope|not)\b\W*)?(?:but\b\W*)?", re.IGNORECASE)
    OUTCOME_SAID = re.compile(r"^\s*(?:(?:the\s+)?(?:outcome|target|label|answer)(?:\s+column)?\s*(?:is|=|:)\s*)?[`'\"]?([\w.\- ]+?)[`'\"]?\s*[.!]?\s*$",
                              re.IGNORECASE)

    def _build_answer(self, draft_id: str, text: str) -> bool:
        """The person's answer to the build's question, read by code only when it plainly is one (the review of 37.11), else
        False and the agent reads the words:

        - the outcome: the whole message is a column's name, or "the outcome is <column>", with exactly one column and no
          question in it;
        - the prediction moment: a sentence with no question in it is said back, and the build runs only on the person's
          yes to it (a moment is the prediction contract: never taken from words that only look like one)."""
        draft = self.store.get(draft_id)
        need, said = draft.get("build_waiting"), text.strip()
        if "?" in said:
            return False
        if need == "outcome column":
            columns = {str(c.get("name")).casefold(): str(c.get("name")) for c in (draft.get("analysis") or {}).get("columns") or [] if c.get("name")}
            m = self.OUTCOME_SAID.match(said)
            column = columns.get(m.group(1).strip().casefold()) if m else None
            if column is None:
                return False
            self.store.update(draft_id, lambda d: (d.setdefault("understanding", {}).update(target=column), d.pop("build_waiting", None)))
            self._build(draft_id)
            return True
        if need == "prediction moment":
            if len(said.split()) < 3:
                return False
            self.store.update(draft_id, lambda d: d.update(build_waiting="confirm moment", moment_said=said[:400]))
            self.say(draft_id, f"So the prediction is made {said[:400].rstrip('.')}, using only what is known then. Shall I build with that? "
                               "Say yes, or tell me the moment again.", kind="text", by="case_agent")
            return True
        if need == "confirm moment":
            if self.HOLD_BUILD.search(said):
                self.store.update(draft_id, lambda d: (d.pop("build_waiting", None), d.pop("moment_said", None)))
                return False
            if self.CORRECTION.search(said):
                # "Yes, but it is at checkout": a correction is said back again, never a yes to the old moment (the review)
                self.store.update(draft_id, lambda d: d.update(build_waiting="prediction moment"))
                return self._build_answer(draft_id, self.LEADING_ANSWER.sub("", said, count=1))
            if not A.says_yes(said):
                self.store.update(draft_id, lambda d: (d.pop("build_waiting", None), d.pop("moment_said", None)))
                return False
            moment = str(draft.get("moment_said") or "")
            self.store.update(draft_id, lambda d: (d.setdefault("understanding", {}).update(prediction_moment=moment),
                                                   d.pop("build_waiting", None), d.pop("moment_said", None)))
            self._build(draft_id)
            return True
        return False

    @turn  # the review of 37.11: one turn at a time per draft, seen by every instance (the rules-first Home's own lock)
    def start(self, draft_id: str) -> None:
        if self._client(draft_id) is None:
            return super().start(draft_id)
        self._run(draft_id, str(self.store.get(draft_id).get("problem") or ""))

    @turn
    def reply(self, draft_id: str, text: str) -> None:
        if self._client(draft_id) is None:  # no model: the rules-first Home answers, as with the switch off
            return super().reply(draft_id, text)
        from . import column_check as checks

        draft = self.store.get(draft_id)
        open_question = self.pending(draft)
        if open_question is not None and open_question.get("field") == checks.FIELD:
            # the review of 37.11: the build's column question is answered by code (known, written later, or DCLab's call),
            # then the build goes on, as on the rules-first Home; the agent never reads it
            self.store.update(draft_id, lambda d: d["messages"].append({"role": "user", "kind": "text", "text": text}))
            return self.take_answer(draft_id, open_question, text)
        if draft.get("build_waiting"):
            self.store.update(draft_id, lambda d: d["messages"].append({"role": "user", "kind": "text", "text": text}))
            if self._build_answer(draft_id, text):
                return
            return self._run(draft_id, text)
        offered = [o.get("title") for o in ((draft.get("case_state") or {}).get("case") or {}).get("options") or []]
        if not (draft.get("case_state") or {}).get("case", {}).get("route") and text in offered:
            self.store.update(draft_id, lambda d: d["messages"].append({"role": "user", "kind": "text", "text": text}))
            return self._run(draft_id, text, pick=text)
        self.store.update(draft_id, lambda d: d["messages"].append({"role": "user", "kind": "text", "text": text}))
        self._run(draft_id, text)

    @turn
    def data_ready(self, draft_id: str, asset: dict[str, Any]) -> None:
        """A table is ready: the case reads it on the next turn; Home says so in one line (never the rules' questions). A build
        question about the earlier table is dropped (the review of 37.11)."""
        self.store.update(draft_id, lambda d: (d.pop("build_waiting", None), d.pop("moment_said", None)))
        draft = self.store.get(draft_id)
        s = (draft.get("analysis") or {}).get("summary") or {}
        self.say(draft_id, f"Your table {asset.get('name', '')} is ready: {s.get('rows', 0):,} rows and {s.get('columns', 0)} columns. "
                           "Tell me what you would like to know from it, or carry on where we were.", kind="text", by="case_agent")


__all__ = ["CaseHome", "table_of"]
