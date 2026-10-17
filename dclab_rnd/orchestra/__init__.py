"""The orchestration layer (phase 6, ``docs/guides/PHASE6_MULTI_AGENT_PLAN.md``): DCLab's agents as nodes of a graph, on
LangGraph used thinly.

What is taken from the framework: the graph and its typed state, checkpoints (``checkpoints``: PostgreSQL in the
workspace's database, SQLite in a file workspace), ``interrupt`` for the three pauses a person answers (a question, a
confirmation, an approval), and ``Send`` for the parts of a complex problem. What is not: its agents, tools and model
classes. A node is one of DCLab's functions; a model is asked through the gateway (``model``); every node is a step of a
``harness.session``; every pause is recorded; the rules' path serves whenever the layer is off, over budget or failing.

28.1 is the spike: three nodes (the interviewer, the analyst, a confirmation) on a draft, resumable from a checkpoint,
every step in the harness (``spike``). The contracts of the board and the hand-offs are 28.2's.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = "orchestra"  # under the workspace home: checkpoints and, later, the boards


def thread_id(kind: str, ident: str) -> str:
    """The checkpoint thread of one walk: the draft's or the project's id, prefixed by what it is."""
    return f"{kind}:{ident}"


__all__ = ["ROOT", "FOLDER", "thread_id"]
