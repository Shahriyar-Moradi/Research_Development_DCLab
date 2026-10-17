"""The spike (28.1): three nodes on a draft's first sentence, resumable, every step in the harness.

- ``interviewer``: the next question by what it unblocks (``dclab_core.conversation.next_question``), asked as an
  ``interrupt`` the person answers; the answer read by the rules (``read_answer``) into what was said.
- ``analyst``: the problem card from the words and what was said (``card_for``; a model advisor when one serves the
  problem analyst, else the rules), summarised for the person.
- ``confirm``: the summary as a ``confirmation`` interrupt; the person approves or corrects.

The graph pauses at each interrupt and the walk returns ``paused`` with what it asks; ``resume`` continues the same
thread with the answer, in this process or another (the checkpoint holds the state). Every node is a step of the
``orchestra`` session; a pause is a step too, and the run ends ``stopped`` until the person answers (as the
coordinator's does). Nothing here replaces the coordinator: this is the layer's first graph, on the same functions.
"""

from __future__ import annotations

import contextvars
import re
from typing import Any, Optional, TypedDict

from ..agents import harness
from . import thread_id

LIMIT = 8  # questions a walk may ask before it goes to the analyst with what it has
YES = re.compile(r"^\s*(yes|ok(ay)?|approved?|confirm(ed)?|correct|go ahead)\b", re.I)  # a person's approval in words
_SESSION: contextvars.ContextVar[Any] = contextvars.ContextVar("orchestra_session")  # the walk's harness session: one per walk, not per graph


class SpikeState(TypedDict, total=False):
    problem: str
    columns: Optional[list[str]]
    said: dict[str, str]  # what the person stated, as card fields
    asked: list[str]  # the fields asked, in order
    summary: dict[str, Any]
    confirmed: Optional[bool]
    limit: int
    done: bool  # the interviewer has nothing more to ask (or reached the limit): the edge reads this, never a card


class Spike:
    """The three-node graph of 28.1, built over an open checkpointer."""

    def __init__(self, checkpointer: Any, advisor: Any = None):
        self.checkpointer, self.advisor = checkpointer, advisor
        self.graph = None

    @property
    def _session(self) -> Any:
        return _SESSION.get()

    # ---------------------------------------------------------------------------------------------------- the nodes
    def _card(self, state: SpikeState) -> Any:
        from dclab_core.conversation import card_for

        return card_for(state["problem"], state.get("said") or {}, state.get("columns"), advisor=self.advisor)

    def interviewer(self, state: SpikeState) -> dict[str, Any]:
        from langgraph.types import interrupt

        from dclab_core.conversation import next_question, read_answer

        asked = list(state.get("asked") or [])
        question = next_question(self._card(state), asked=tuple(asked))
        if question is None or len(asked) >= state.get("limit", LIMIT):
            return {"done": True}
        field = getattr(question, "field", None) or getattr(question, "code", "")
        answer = interrupt({"kind": "question", "field": field, "text": getattr(question, "text", ""), "why": getattr(question, "why", "")})
        with self._session.timed("answer", {"field": field}) as out:  # on resume: the node runs again and the interrupt returns the answer
            read = read_answer(field, str(answer or ""), state.get("columns"))
            out["result"] = {"read": sorted(read)}
        return {"said": {**(state.get("said") or {}), **read}, "asked": asked + [field]}

    def analyst(self, state: SpikeState) -> dict[str, Any]:
        from dclab_core.conversation import summary

        with self._session.timed("analyst", {"asked": len(state.get("asked") or [])}) as out:
            card = self._card(state)
            shown = summary(card)
            out["result"] = {"ready": shown["ready"], "family": shown["family"], "blocking": len(shown["blocking"])}
        return {"summary": shown}

    def confirm(self, state: SpikeState) -> dict[str, Any]:
        from langgraph.types import interrupt

        answer = interrupt({"kind": "confirmation", "summary": state.get("summary") or {}})
        approved = answer is True or (isinstance(answer, str) and bool(YES.match(answer)))  # a correction is not an approval
        with self._session.timed("confirm", {"approved": approved}) as out:
            out["result"] = {"confirmed": approved}
        return {"confirmed": approved}

    # ---------------------------------------------------------------------------------------------------- the graph
    @staticmethod
    def _after_interviewer(state: SpikeState) -> str:
        """A pure function of the state: the interviewer said whether it is done (never a second card, never a model)."""
        return "analyst" if state.get("done") else "interviewer"

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(SpikeState)
        g.add_node("interviewer", self.interviewer)
        g.add_node("analyst", self.analyst)
        g.add_node("confirm", self.confirm)
        g.add_edge(START, "interviewer")
        g.add_conditional_edges("interviewer", self._after_interviewer, {"interviewer": "interviewer", "analyst": "analyst"})
        g.add_edge("analyst", "confirm")
        g.add_edge("confirm", END)
        self.graph = g.compile(checkpointer=self.checkpointer)
        return self.graph

    # ---------------------------------------------------------------------------------------------------- the walk
    def run(self, draft_id: str, problem: Optional[str] = None, columns: Optional[list[str]] = None, resume: Any = None,
            limit: int = LIMIT, store: Any = None) -> dict[str, Any]:
        """Start a walk (``problem`` given) or resume one (``resume``: the person's answer or approval) on the draft's
        thread. Returns ``{"status": "paused", "asks": {...}}`` at an interrupt, else ``{"status": "done", ...}``."""
        from langgraph.types import Command

        if self.graph is None:
            self.build()
        config = {"configurable": {"thread_id": thread_id("draft", draft_id)}}
        if resume is None and problem is None:
            raise ValueError("a new walk needs the problem's words")
        if resume is not None and not any(getattr(t, "interrupts", None) for t in self.graph.get_state(config).tasks):
            raise ValueError(f"nothing to resume on {config['configurable']['thread_id']}: no pause is open")
        with harness.session("orchestra", store=store, params={"draft": draft_id, "resumed": resume is not None}) as current:
            token = _SESSION.set(current)
            try:
                if resume is None:
                    self.graph.invoke({"problem": problem, "columns": columns, "said": {}, "asked": [], "limit": limit, "done": False}, config)
                else:
                    self.graph.invoke(Command(resume=resume), config)
            finally:
                _SESSION.reset(token)
            snapshot = self.graph.get_state(config)
            pending = [i for task in snapshot.tasks for i in (getattr(task, "interrupts", None) or ())]
            if pending:
                asks = dict(pending[0].value)
                current.step("interrupt", {"kind": asks.get("kind"), "field": asks.get("field")}, "paused")
                current.status = "stopped"  # the walk waits for the person, as the coordinator's does
                return {"status": "paused", "asks": asks, "asked": list(snapshot.values.get("asked") or [])}
            values = snapshot.values
            return {"status": "done", "confirmed": values.get("confirmed"), "summary": values.get("summary"), "asked": list(values.get("asked") or [])}


__all__ = ["LIMIT", "Spike", "SpikeState"]
