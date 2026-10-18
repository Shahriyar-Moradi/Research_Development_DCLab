"""The contracts of a team of agents (package 28.2, ``docs/guides/PHASE6_MULTI_AGENT_PLAN.md`` Part D4): framework-free, so
the orchestration layer (LangGraph in the studio) can be swapped without touching what is kept.

- :class:`Board`: what the team knows at one moment, read and written by its nodes: the problem as understood (the card's
  fields, each a short name or code), the tables as counted (never a row), the map's parts, each part's route and the
  workflow's step names, the questions open and the blockers, the spend, the hand-offs so far, the ledger ids of the
  decisions made, the interrupts. A board holds names, codes and counts: never a table's value (:func:`values_in`).
- :class:`Handoff`: one node passing the board to the next, with the board's version before and after.
- :class:`Interrupt`: one of the three pauses a person answers (a question, a confirmation, an approval).
- :class:`TeamRecord`: a walk of a team, node by node (steps, seconds, cost, how it ended): the trajectory a solver
  model learns from.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import Field, field_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

NAME = 200  # a field holds a name, a code or a short phrase the system wrote: never a row
TEXT = 600  # a question or a summary DCLab wrote for the person


class Handoff(Contract):
    from_node: str = Field(min_length=1, max_length=80)
    to_node: str = Field(min_length=1, max_length=80)
    what: str = Field(min_length=1, max_length=NAME, description="What was handed over, as a code (brief, data_card, map, routes, workflow).")
    board_before: int = Field(ge=0, description="The board's version when the node read it.")
    board_after: int = Field(ge=0, description="The board's version the node left.")
    at: str = Field(max_length=40)


class Interrupt(Contract):
    kind: Literal["question", "confirmation", "approval"]
    node: str = Field(min_length=1, max_length=80)
    field: Optional[str] = Field(default=None, max_length=80, description="For a question: the card field it settles.")
    gate: Optional[str] = Field(default=None, max_length=80, description="For an approval: what is approved (build, holdout, promote).")
    text: str = Field(default="", max_length=TEXT, description="What DCLab asks, in its own words: never a row.")
    status: Literal["open", "answered", "declined"] = "open"
    at: str = Field(max_length=40)


class Part(Contract):
    part: str = Field(min_length=1, max_length=80)
    title: str = Field(default="", max_length=NAME)
    family: Optional[str] = Field(default=None, max_length=80)
    route: Optional[str] = Field(default=None, max_length=80)
    workflow: list[Annotated[str, Field(max_length=120)]] = Field(default_factory=list, max_length=40, description="The step names of its solution workflow.")
    status: Literal["open", "routed", "designed", "built", "blocked"] = "open"


class Board(Versioned):
    """What a team knows at one moment (28.2)."""

    schema_version: Literal["dclab_core.orchestra_board.v1"] = _version("orchestra_board")
    thread: str = Field(min_length=1, max_length=120, description="draft:<id> or project:<id>: the walk this board belongs to.")
    version: int = Field(default=0, ge=0, description="Raised by one at every node that changes the board.")
    state: str = Field(default="problem", max_length=40, description="The coordinator's state (dclab_core.coordinator.STATES).")
    problem: dict[str, Optional[str]] = Field(default_factory=dict, description=(
        "The card's fields as understood: family, labels, outcome (a column's name), moment, unit, action, ready: each a name, "
        "a code or a short phrase."))
    data: dict[str, int] = Field(default_factory=dict, description="The tables as counted: tables, rows, columns, labelled rows.")
    parts: list[Part] = Field(default_factory=list, max_length=40)
    open_questions: list[Annotated[str, Field(max_length=80)]] = Field(default_factory=list, max_length=40, description="The card fields still asked.")
    blockers: list[Annotated[str, Field(max_length=80)]] = Field(default_factory=list, max_length=40, description="Codes of what stops a build.")
    spent_eur: float = Field(default=0.0, ge=0)
    handoffs: list[Handoff] = Field(default_factory=list, max_length=400)
    interrupts: list[Interrupt] = Field(default_factory=list, max_length=200)
    decisions: list[str] = Field(default_factory=list, max_length=2000, description="Ledger ids of the decisions made in this walk.")

    @field_validator("problem", mode="before")
    @classmethod
    def _short(cls, value: Any) -> Any:
        """Whitespace made one line (a person's two-line answer never crashes a walk; the draft keeps their words whole);
        still at most NAME characters."""
        out = {}
        for k, v in dict(value or {}).items():
            v = None if v is None else " ".join(str(v).split())
            if len(str(k)) > 40 or (v is not None and len(v) > NAME):
                raise ValueError(f"problem.{str(k)[:40]}: a name, a code or a short phrase (at most {NAME} characters)")
            out[str(k)] = v
        return out


class NodeRun(Contract):
    node: str = Field(min_length=1, max_length=80)
    steps: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0)
    cost_eur: float = Field(default=0.0, ge=0)
    model: Optional[str] = Field(default=None, max_length=120)
    status: Literal["completed", "failed", "paused", "skipped"] = "completed"


class TeamRecord(Versioned):
    """A walk of a team, node by node: the trajectory kept for the solver model (28.2)."""

    schema_version: Literal["dclab_core.team_record.v1"] = _version("team_record")
    thread: str = Field(min_length=1, max_length=120)
    team: Literal["understanding", "solution", "orchestra"]
    nodes: list[NodeRun] = Field(default_factory=list, max_length=400)
    handoffs: int = Field(default=0, ge=0)
    interrupts: int = Field(default=0, ge=0)
    spent_eur: float = Field(default=0.0, ge=0)
    board_version: int = Field(default=0, ge=0)
    status: Literal["completed", "paused", "failed", "stopped"] = "completed"
    started: str = Field(max_length=40)
    finished: Optional[str] = Field(default=None, max_length=40)


CONTRACTS["orchestra_board"] = Board
CONTRACTS["team_record"] = TeamRecord


VOCABULARY = ("family", "labels", "ready")  # fields holding DCLab's own codes: never compared with a table's values
MIN_VALUE = 4  # values shorter than this are too common to tell (a code like "no", "1", "yes")


def values_in(board: Board, values: Any) -> list[str]:
    """Where a board holds one of ``values`` (a table's distinct cell values, as text): the fields to refuse. Every
    text-bearing field is read (the problem's, the open questions, the blockers, the parts', the interrupts'), a value
    found inside a longer text too; DCLab's own vocabulary fields (family, labels, ready) and short values are not."""
    seen = sorted({str(v) for v in values if v is not None and len(str(v)) >= MIN_VALUE}, key=len, reverse=True)

    def holds(text: Any) -> bool:
        t = str(text or "")
        return bool(t) and any(v in t for v in seen)

    found = []
    for k, v in board.problem.items():
        if k not in VOCABULARY and v is not None and holds(v):
            found.append(f"problem.{k}")
    for name in ("open_questions", "blockers"):
        for i, item in enumerate(getattr(board, name)):
            if holds(item):
                found.append(f"{name}[{i}]")
    for i, p in enumerate(board.parts):
        for name in ("part", "title", "route"):
            if holds(getattr(p, name)):
                found.append(f"parts[{i}].{name}")
        if any(holds(w) for w in p.workflow):
            found.append(f"parts[{i}].workflow")
    for i, q in enumerate(board.interrupts):
        if holds(q.text) or holds(q.field) or holds(q.gate):
            found.append(f"interrupts[{i}]")
    return found


def advance(board: Board, node: str, what: str, at: str, **changes: Any) -> Board:
    """The board a node leaves: its changes applied, the version raised, the hand-off recorded."""
    after = board.version + 1
    handoff = Handoff(from_node=node, to_node=changes.pop("to", node), what=what, board_before=board.version, board_after=after, at=at)
    data = {**board.model_dump(), **{k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in changes.items()},
            "version": after, "handoffs": [*(h.model_dump() for h in board.handoffs), handoff.model_dump()]}
    return Board.model_validate(data)  # validated again: a node never leaves a board its contract refuses


__all__ = ["Board", "Handoff", "Interrupt", "Part", "NodeRun", "TeamRecord", "values_in", "advance"]
