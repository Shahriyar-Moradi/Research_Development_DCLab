"""Lessons from use (34.12): what a model in use taught, proposed as workspace lessons and reviewed as any other.

A champion/challenger comparison (34.11), promoted or refused, and a live report's alerts (34.10) each become one proposed
lesson in the workspace's lessons table (``dclab_rnd.lessons``): its claim from the record's own numbers, what argues
against it (one window, one comparison), the next test, and the record it cites. Code writes them, never a model, so every
number in a lesson is the record's. A reviewer accepts, edits or rejects them as lessons from finished projects; until
then they are offered nowhere. Each record is proposed once.
"""

from __future__ import annotations

import secrets
from typing import Any, Optional

from .. import lessons


def _scope(store: Any, project: dict[str, Any]) -> dict[str, Any]:
    final = store.read_stage(project["id"], "final") or {}
    return {**lessons.scope_of(project, final), "from": "use"}


def _save(lesson_store: Any, project: dict[str, Any], basis: str, claim: str, against: str, next_test: str, cite: str, source: str,
          scope: dict[str, Any]) -> Optional[dict[str, Any]]:
    if any(l.get("basis") == basis for l in lesson_store.list(project_id=project["id"])):
        return None  # proposed once
    lesson = {"id": secrets.token_hex(5), "project_id": project["id"], "project_name": project.get("name"), "basis": basis,
              "experiment_id": None, "status": "proposed", "claim": lessons._line(claim), "against": lessons._line(against),
              "next_test": lessons._line(next_test), "cites": [cite], "sources": {cite: lessons._line(source, 1200)}, "scope": scope,
              "synthetic": bool((project.get("data") or {}).get("synthetic") or (project.get("draft") or {}).get("synthetic")),
              "written_by": "code", "proposed_at": lessons.now(), "review": None, "history": []}
    return lesson_store.save(lesson)


def propose(store: Any, project_id: str, entry: dict[str, Any], lesson_store: Any = None) -> Optional[dict[str, Any]]:
    """One lesson from a champion/challenger comparison (an entry of the project's ``retrains``)."""
    if lesson_store is None:
        return None
    project = store.get(project_id)
    cite = f"RETRAIN-{entry['id']}"
    lo, hi = entry["interval"]
    shown = f"90% interval {lo} to {hi}" if lo is not None else "no interval could be read"
    against = entry.get("against") or "the approved model"
    if entry["promoted"]:
        claim = (f"A challenger refitted with {entry['trained_on']['challenger'].split(' and ', 1)[-1]} was better than {against} "
                 f"on {entry['window']}: {entry['metric']} {entry['challenger']} against {entry['champion']}, better by {entry['difference']} "
                 f"({shown}).")
        against = "One comparison on one window: the newer rows may not stand for the rows that follow them."
        next_test = "Compare the promoted model with the next challenger on the next window of newer labelled rows."
    else:
        claim = (f"Refitting with {entry['trained_on']['challenger'].split(' and ', 1)[-1]} did not beat {against} on "
                 f"{entry['window']}: {entry['metric']} {entry['challenger']} against {entry['champion']}, a difference of "
                 f"{entry['difference']} ({shown}).")
        against = "One window of newer rows: a change smaller than the interval stays unseen."
        next_test = "Compare again when more newer labelled rows exist; a larger window narrows the interval."
    if entry.get("because"):
        claim += " It was compared because " + "; ".join(entry["because"]) + "."
    return _save(lesson_store, project, f"retrain:{entry['id']}", claim, against, next_test, cite, entry["reason"], _scope(store, project))


def propose_from_live(store: Any, project_id: str, report: dict[str, Any], lesson_store: Any = None) -> Optional[dict[str, Any]]:
    """One lesson from a live report's first alert (``dclab_core.live.LiveReport`` as a dict), when it has one."""
    if lesson_store is None or not report.get("first_alert"):
        return None
    project = store.get(project_id)
    window = next((w for w in report.get("windows") or [] if w.get("window") == report["first_alert"]), None)
    if window is None:  # a report whose first alert names no window of its own: nothing to cite
        return None
    said = "; ".join(window.get("alerts") or [])
    cite = f"LIVE-{report['first_alert']}"
    claim = f"The model in use raised its first alert in the window {report['first_alert']} ({window.get('rows')} rows): {said}."
    against = "Read from one model's logs: a shift in the inputs is not by itself a fall in performance until outcomes say so."
    next_test = "Read the next windows once their outcomes are known, and compare a challenger on the latest of them."
    return _save(lesson_store, project, f"live:{report.get('model_id')}:{report['first_alert']}", claim, against, next_test, cite,
                 report.get("note") or said, _scope(store, project))
