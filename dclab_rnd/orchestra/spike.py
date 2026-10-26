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
    board: dict[str, Any]  # 28.2: the team's board (dclab_core.orchestra.Board), advanced by every node


class Spike:
    """The three-node graph of 28.1, built over an open checkpointer."""

    TEAM = "understanding"

    def __init__(self, checkpointer: Any, advisor: Any = None, home: Any = None):
        self.checkpointer, self.advisor, self.home = checkpointer, advisor, home
        self.graph = None

    def _board(self, state: SpikeState) -> Any:
        from dclab_core.orchestra import Board

        return Board.model_validate(state["board"])

    def _advance(self, state: SpikeState, node: str, what: str, **changes: Any) -> dict[str, Any]:
        from . import board as B

        return B.advance(self._board(state), node, what, self.TEAM, home=self.home, **changes).model_dump(mode="json")

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
            return {"done": True, "board": self._advance(state, "interviewer", "questions_done", to="analyst")}  # what is still open stays open
        field = getattr(question, "field", None) or getattr(question, "code", "")
        answer = interrupt({"kind": "question", "field": field, "text": getattr(question, "text", ""), "why": getattr(question, "why", "")})
        with self._session.timed("answer", {"field": field}) as out:  # on resume: the node runs again and the interrupt returns the answer
            read = read_answer(field, str(answer or ""), state.get("columns"))
            out["result"] = {"read": sorted(read)}
        from . import board as B

        board = B.interrupt(self._board(state), "question", "interviewer", self.TEAM, field=field, text=getattr(question, "text", ""),
                            status="answered", home=self.home)
        still = [q for q in board.open_questions if q != field] + ([field] if not read else [])  # an unread answer leaves its field open
        board = B.advance(board, "interviewer", "answer", self.TEAM, home=self.home, open_questions=still)
        return {"said": {**(state.get("said") or {}), **read}, "asked": asked + [field], "board": board.model_dump(mode="json")}

    def analyst(self, state: SpikeState) -> dict[str, Any]:
        from dclab_core.conversation import summary

        with self._session.timed("analyst", {"asked": len(state.get("asked") or [])}) as out:
            card = self._card(state)
            shown = summary(card)
            out["result"] = {"ready": shown["ready"], "family": shown["family"], "blocking": len(shown["blocking"])}
        problem = {k: (None if shown.get(k) is None else str(shown.get(k))[:200]) for k in ("family", "labels", "outcome", "moment", "unit", "action", "ready")}
        return {"summary": shown, "board": self._advance(state, "analyst", "brief", to="confirm", problem=problem,
                                                         open_questions=[str(b)[:80] for b in shown.get("blocking") or []])}

    def confirm(self, state: SpikeState) -> dict[str, Any]:
        from langgraph.types import interrupt

        answer = interrupt({"kind": "confirmation", "summary": state.get("summary") or {}})
        approved = answer is True or (isinstance(answer, str) and bool(YES.match(answer)))  # a correction is not an approval
        with self._session.timed("confirm", {"approved": approved}) as out:
            out["result"] = {"confirmed": approved}
        from . import board as B

        board = B.interrupt(self._board(state), "confirmation", "confirm", self.TEAM, status="answered" if approved else "declined", home=self.home)
        board = B.advance(board, "confirm", "confirmation", self.TEAM, home=self.home, state="map" if approved else "problem")
        return {"confirmed": approved, "board": board.model_dump(mode="json")}

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
        from .. import experience

        with experience.bind(conversation=draft_id), \
                harness.session("orchestra", store=store, params={"draft": draft_id, "resumed": resume is not None}) as current:
            token = _SESSION.set(current)
            try:
                if resume is None:
                    from . import board as B

                    self.graph.invoke({"problem": problem, "columns": columns, "said": {}, "asked": [], "limit": limit, "done": False,
                                       "board": B.new(thread_id("draft", draft_id)).model_dump(mode="json")}, config)
                else:
                    self.graph.invoke(Command(resume=resume), config)
            finally:
                _SESSION.reset(token)
            snapshot = self.graph.get_state(config)
            pending = [i for task in snapshot.tasks for i in (getattr(task, "interrupts", None) or ())]
            if pending:
                asks = dict(pending[0].value)
                current.step("interrupt", {"kind": asks.get("kind"), "field": asks.get("field")}, "paused")
                from dclab_core.orchestra import Board

                from . import board as B

                B.interrupt(Board.model_validate(snapshot.values["board"]), asks.get("kind") or "question", "interviewer" if asks.get("kind") == "question" else "confirm",
                            self.TEAM, field=asks.get("field"), text=str(asks.get("text") or ""), status="open", home=self.home)  # opened: an event
                current.status = "stopped"  # the walk waits for the person, as the coordinator's does
                return {"status": "paused", "asks": asks, "asked": list(snapshot.values.get("asked") or [])}
            values = snapshot.values
            return {"status": "done", "confirmed": values.get("confirmed"), "summary": values.get("summary"), "asked": list(values.get("asked") or []),
                    "board": values.get("board")}


def main(argv: Optional[list[str]] = None) -> int:
    """The spike from a command line (2026-10-26, so a chat can step it with the person: DCLab Code runs this command, the
    person answers each question in the chat): start a walk with the problem's words, resume it with each answer, then the
    confirmation. The rules only (no model), its checkpoints and its events in ``--home`` (a folder of the person's
    choosing), one JSON line out per call."""
    import argparse
    import json
    from pathlib import Path

    from .. import experience
    from . import checkpoints

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.orchestra.spike", description=main.__doc__)
    parser.add_argument("--home", required=True, help="the folder for its checkpoints and events")
    parser.add_argument("--draft", required=True, help="the walk's name: the same name resumes it")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--words", help="start: the problem in the person's words")
    group.add_argument("--answer", help="resume: the person's answer, or their approval at the confirmation")
    parser.add_argument("--columns", default="", help="start: the table's column names, comma-separated")
    args = parser.parse_args(argv)
    home = Path(args.home).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    experience.install(experience.FileLog(home / "experience", "spike"))
    try:
        with checkpoints.open_checkpointer(home, database_url="") as saver:
            spike = Spike(saver, home=home)
            columns = [c.strip() for c in args.columns.split(",") if c.strip()] or None
            out = spike.run(args.draft, args.words, columns=columns) if args.words is not None else spike.run(args.draft, resume=args.answer)
    except ValueError as error:
        print(json.dumps({"status": "refused", "why": str(error)}))
        return 2
    finally:
        experience.install(None)
    out.pop("board", None)  # the board is in the checkpoint; the line says what the person needs
    print(json.dumps(out, default=str))
    return 0


__all__ = ["LIMIT", "Spike", "SpikeState", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
