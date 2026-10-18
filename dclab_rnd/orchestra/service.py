"""The teams in the product (package 28.8): what the Home page, the project page, the Admin page and a tester's CLI call.

``walk_draft`` starts the understanding walk on a draft, or resumes it with the person's answer (a question's words, or
an approval's yes or no), once: the walk runs until its next pause or its end and says which. ``walk_project`` does the
same for the solution team on a project. ``view`` reads a walk as it stands without running anything: its last board,
the pause waiting for the person (from the checkpoint), and its last record. ``walks`` lists the workspace's walks for the
Admin trace view: each walk's nodes, hand-offs, pauses, cost and the models it asked.

The understanding critic's model serves only once the team passed its sealed gate (``team_passed``); until then the walk
asks the critic's rules, as the measurement decided. Nothing here bypasses the walk: every move is the coordinator's, every
project write the validator's.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from . import checkpoints, thread_id

_PROJECT_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


class Busy(RuntimeError):
    """The project's team is already walking (another request, another person)."""


class WrongAnswer(ValueError):
    """An answer that does not fit the pause: words for a question, a yes or no for a confirmation or an approval."""


def pending(home: Path, thread: str) -> Optional[dict[str, Any]]:
    """The pause the walk waits on now, read from its checkpoint (its kind, field or gate, text, and the summary of what an
    approval approves: names and counts), or None. Nothing runs, and no store is made to read nothing."""
    place = checkpoints.where(home)
    if place["kind"] != "postgres" and not Path(place.get("path") or "").is_file():
        return None
    from .solution import SolutionTeam
    from .walk import Walk

    with checkpoints.open_checkpointer(home) as saver:
        graph = (SolutionTeam(None, saver) if thread.startswith("project:") else Walk(None, saver)).build()
        snap = graph.get_state({"configurable": {"thread_id": thread}})
        open_ = [i for t in snap.tasks for i in (getattr(t, "interrupts", None) or ())]
    return dict(open_[0].value) if open_ else None


def resume_value(home: Path, thread: str, body: dict[str, Any]) -> Any:
    """The person's answer from a request's body, checked against the pause it answers (the review of 28.8: a yes or no
    sent to a question was recorded as the prediction moment). None: start (or show the pause already waiting)."""
    if "answer" not in body and "approve" not in body:
        return None
    waiting = pending(home, thread)
    if waiting is None:
        raise ValueError("nothing to resume: no pause is open")
    if waiting.get("kind") == "question":
        if "answer" not in body or not isinstance(body["answer"], str):
            raise WrongAnswer("this pause is a question: answer it in words ({\"answer\": \"...\"}), or with an empty answer to skip it")
        return body["answer"]
    if "approve" not in body or not isinstance(body["approve"], bool):
        raise WrongAnswer(f"this pause is a {waiting.get('kind')}: answer yes or no ({{\"approve\": true}} or false)")
    return body["approve"]


def _lines(home: Path, name: str) -> list[dict[str, Any]]:
    path = Path(home) / "orchestra" / name
    out = []
    for line in (path.read_text(encoding="utf-8").splitlines() if path.is_file() else []):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def team_passed() -> bool:
    """Whether the team passed its sealed measurement (the fifth sealed set's team arm against ``TEAM_GATE``)."""
    from ..agent_eval.journeys import RESULTS

    for path in sorted(RESULTS.glob("JRN-*_sealed_v5.json")) if RESULTS.exists() else []:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("arm") == "team_live":
            return bool(((doc.get("team_against") or {}).get("passed")))
    return False


def _critic(gateway: Any, draft_id: str) -> Any:
    if gateway is None or not team_passed():
        return None  # the critic's rules: its model serves after the sealed gate
    from ..core_advisor import GatewayAdvisor

    return GatewayAdvisor(gateway, draft_id=draft_id, max_tokens=1200)


def view(home: Path, thread: str) -> dict[str, Any]:
    """A walk as it stands: the last board, the pause waiting for the person, the last record. Nothing runs."""
    boards = [d for d in _lines(home, "boards.jsonl") if (d.get("board") or {}).get("thread") == thread]
    records = [r for r in _lines(home, "walks.jsonl") if r.get("thread") == thread]
    found = pending(home, thread)
    waiting = ({k: found.get(k) for k in ("kind", "field", "gate", "text", "summary")} if found else None)  # what the person is asked, and what an approval approves
    last = boards[-1] if boards else None
    return {"thread": thread, "board": (last or {}).get("board"), "node": (last or {}).get("node"), "what": (last or {}).get("what"),
            "waiting": waiting, "walk": records[-1] if records else None, "handoffs": len(boards)}


def walk_draft(home: Path, drafts: Any, projects: Any, agent: Any, draft_id: str, resume: Any = None, gateway: Any = None) -> dict[str, Any]:
    """Start the understanding walk on a draft, or resume it with the person's answer; until its next pause or its end."""
    from ..draft.coordinator import Coordinator
    from .walk import Walk

    with checkpoints.open_checkpointer(home) as saver:
        walk = Walk(Coordinator(drafts, projects, agent), saver, home=Path(home), critic_advisor=_critic(gateway, draft_id))
        out = walk.run(draft_id, resume=resume)
    return {**{k: v for k, v in out.items() if k != "board"}, "view": view(home, thread_id("draft", draft_id))}


def walk_project(home: Path, projects: Any, project_id: str, resume: Any = None) -> dict[str, Any]:
    """Start the solution team on a project, or resume it with the person's answer; until its next pause or its end. One walk
    of a project at a time in this process (``Busy`` otherwise: two runs would approve and read the holdout twice)."""
    from .solution import SolutionTeam

    with _GUARD:
        lock = _PROJECT_LOCKS.setdefault(project_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise Busy(f"the team is already walking project {project_id}")
    try:
        with checkpoints.open_checkpointer(home) as saver:
            out = SolutionTeam(projects, saver, home=Path(home)).run(project_id, resume=resume)
    finally:
        lock.release()
    return {**{k: v for k, v in out.items() if k != "board"}, "view": view(home, thread_id("project", project_id))}


def walks(home: Path, limit: int = 200) -> dict[str, Any]:
    """Every finished walk of the workspace, newest first, for the Admin trace view: its team, thread, status, nodes (each
    with its steps and seconds), hand-offs, pauses, cost and the models it asked. Names, counts and codes only."""
    teams = {(r.get("thread"), r.get("started")): r for r in _lines(home, "teams.jsonl")}
    out = []
    for w in reversed(_lines(home, "walks.jsonl")):
        rec = teams.get((w.get("thread"), w.get("started"))) or {}
        out.append({"thread": w.get("thread"), "team": w.get("team"), "status": w.get("status"), "started": w.get("started"), "finished": w.get("finished"),
                    "spent_eur": w.get("spent_eur"), "models": w.get("models"), "handoffs": rec.get("handoffs"), "interrupts": rec.get("interrupts"),
                    "nodes": rec.get("nodes") or []})
        if len(out) >= limit:
            break
    return {"walks": out, "count": len(out), "team_serves_a_model": team_passed()}


__all__ = ["Busy", "WrongAnswer", "pending", "resume_value", "view", "walk_draft", "walk_project", "walks", "team_passed"]
