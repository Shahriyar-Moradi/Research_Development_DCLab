"""Tasks, actions, reports and the delivery outbox of the work passed between team agents (W4, 2026-11-20).

Two places that behave the same, as problem memory's bases (``storage/memory_bases.py``):

- Files (``FileTeamWork``): one document per project, ``teamspace/<project>/work.json`` in the workspace folder, read
  and replaced whole under that project's folder lock (``filelock.held``: a thread lock and ``flock``). A change is one
  atomic write, so a task and its pending delivery are saved together or not at all. One host only.
- PostgreSQL (``PgTeamWork``, migration 0026): ``team_task``, ``team_action``, ``team_report`` and ``team_delivery``.
  A change is one transaction holding an advisory lock on the project; a task's or a delivery's update also checks its
  ``version``. Ids are unique within a project on both.

What every change keeps (the rules are dclab_core.team_work's; this module only feeds them what is stored):

- A new task is saved with its pending delivery in one step, on a problem the project has. A child task is made only
  through ``delegate``, given its stored siblings; a status change only through ``validate_transition``, given its
  stored children and their joined reports. A clarification (``waiting_for_person``) and its answer each travel as a
  delivery of their own.
- An action is recorded, then finished once. A second, different terminal record for the same action is refused; so is
  a second succeeded or simulated record under one retry key (the same side effect twice). A replayed identical record
  is the same record: nothing is added.
- A report is built only by ``build_report`` from the stored actions and the joined child reports, and a child's report
  joins its parent only through ``accept_reply``, against the problem's current brief revision the caller names. A
  report is one row per (task, revision): storing it again returns the stored one. A correction is the next revision,
  linked both ways by ``supersede``.
- Words are kept whole (the owner's rule: never cut before saving); only what is shown is cut. A delivery's reason, which
  the requester sees, is fixed words and an error's class; the error's own text is kept apart, redacted.
- Reads are scoped: a task is read within its project, the store is one workspace's (its folder, or its rows).

Each delivery is a row: what it carries (a task, a clarification, its answer, a reply), from which team to which, the
words, the thread a reply goes back to, the member who started it, and its state (pending, delivered, blocked with the
reason the requester sees, or refused). The rows of a task are delivered in order; the rows of a problem in order are its
handoff history.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from dclab_core import team_work as tw

Row = dict[str, Any]
FOLDER = Path("teamspace")
FILE = "work.json"
DELIVERY_KINDS = ("task", "clarification", "answer", "reply")
DELIVERY_STATES = ("pending", "delivered", "blocked", "refused")
INITIATOR_FIELDS = ("user_id", "role", "workspace_id")
SHOWN = 500  # how much of a message the history shows; the row keeps it whole
_SAFE_DIR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_CREDENTIAL = re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|authorization)\b(\s*[=:]\s*|\s+)\S+")


class Conflict(RuntimeError):
    """A row changed between its read and its write (another host wrote it): nothing was written."""


@dataclass
class Written:
    """What a change saved: the record, the delivery saved with it (if any), the parent's verdict on a child's reply, and
    whether it was already there (a retried call adds nothing)."""

    item: Any
    delivery: Optional[Row] = None
    verdict: Optional[tw.Verdict] = None
    duplicate: bool = False
    replaced: Any = None  # the superseded report, on a correction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dump(model: Any) -> Row:
    return json.loads(model.model_dump_json())


def scrub(text: Any) -> str:
    """An error's text as the server keeps it: credentials removed (dclab_core.experience's shapes and ``password=…``),
    then dclab_core.redact (e-mails, phones, ids, names…). Never shown to the requester."""
    from dclab_core import redact
    from dclab_core.experience import SECRET

    cleaned = _CREDENTIAL.sub(lambda m: f"{m.group(1)}=[SECRET]", SECRET.sub("[SECRET]", str(text)))
    return redact.redact_text(cleaned)[0] or ""


def initiator_of(value: Any = None) -> Optional[Row]:
    """The member a change acts for, as the outbox keeps it (their id, role and workspace; never more): the one given, else
    the signed-in person of this request or job, else None (an agent's own step in a local workspace)."""
    if value is None:
        from ..accounts.principal import current

        value = current()
        if value is None or not getattr(value, "user_id", None):
            return None
    if isinstance(value, Mapping):
        out = {k: (str(value[k]) if value.get(k) is not None else None) for k in INITIATOR_FIELDS}
    else:
        out = {k: (str(getattr(value, k)) if getattr(value, k, None) is not None else None) for k in INITIATOR_FIELDS}
    if not out.get("user_id"):
        raise ValueError("an initiating member names their user id")
    return out


def team_label(name: str) -> str:
    """How a team is named to people: the directory's team (``data_science`` → "Data Science"), else its words."""
    config = tw.DIRECTORY.get(name)
    if config is not None and config.kind == "team_agent" and config.identity.endswith(" team"):
        return config.identity[: -len(" team")]
    base, _, qualifier = name.partition(":")
    words = " ".join(w.capitalize() for w in base.split("_"))
    return f"{words} ({qualifier})" if qualifier else words


def history_lines(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """The handoff history as the owner writes it (``1. Business → Data Science · Can we rank accounts …?``); a row not
    delivered (yet) says so (``· not delivered: blocked``)."""
    return [f"{e['n']}. {e['from_team']} → {e['to_team']} · {e['text']}" + ("" if e.get("state", "delivered") == "delivered"
                                                                            else f" · not delivered: {e['state']}") for e in entries]


def is_due(row: Mapping[str, Any], now: Optional[datetime] = None) -> bool:
    """Whether a pending row may be tried now (no next attempt set, or its time has come)."""
    at = row.get("next_attempt_at")
    return not at or datetime.fromisoformat(at) <= (now or datetime.now(timezone.utc))


# ---------------------------------------------------------------------------------------------------------- the rules
class _TeamWork:
    """What both places share: every change and read, written once against a project's transaction (``_tx``)."""

    workspace: str = "local"
    home: Optional[Path] = None
    _projects: Any = None

    # implemented by each place
    def _tx(self, project_id: str) -> contextlib.AbstractContextManager: ...                   # a change: locked, all or nothing
    def _read(self, project_id: Optional[str] = None) -> contextlib.AbstractContextManager: ...  # a read; None reads every project
    def location(self) -> str: ...

    @property
    def projects(self) -> Any:
        """The workspace's projects, to check a task's problem belongs to its project."""
        if self._projects is None:
            if self.home is None:
                raise LookupError("this team work store has no projects store to check problems against")
            from ..storage import open_stores

            self._projects = open_stores(self.home)[0]
        return self._projects

    # ------------------------------------------------------------------ helpers
    def _own(self, task: tw.TeamTask) -> None:
        if str(task.workspace_id) != str(self.workspace):
            raise ValueError(f"task {task.id} belongs to workspace {task.workspace_id}, not to this one ({self.workspace})")

    @staticmethod
    def _in_project(row: Optional[Row], project_id: str, task_id: str) -> tw.TeamTask:
        if row is None or row["task"].project_id != project_id:
            raise KeyError(f"no task {task_id} in project {project_id}")
        return row["task"]

    @staticmethod
    def _joined(tx: Any, parent_id: str) -> list[tw.WorkReport]:
        """The child reports the parent accepted (every revision; join_status reads the current one per child)."""
        children = [r["task"].id for r in tx.tasks(parent=parent_id)]
        return [r["report"] for r in tx.reports(children) if r["joined"]] if children else []

    def _delivery(self, kind: str, key: str, task: tw.TeamTask, *, item_id: str, from_team: str, to_team: str, text: str,
                  thread_id: Optional[str], initiator: Optional[Row]) -> Row:
        assert kind in DELIVERY_KINDS, kind
        return {"id": "d" + secrets.token_hex(8), "key": key, "kind": kind, "item_id": item_id, "task_id": task.id,
                "workspace_id": self.workspace, "project_id": task.project_id, "problem_id": task.problem_id,
                "from_team": from_team, "to_team": to_team, "text": str(text), "thread_id": thread_id,
                "initiator": initiator, "status": "pending", "attempts": 0, "next_attempt_at": None, "last_error": None,
                "reason": None, "created_at": _now(), "delivered_at": None, "posted": None, "receipt": None}

    @staticmethod
    def _send(tx: Any, row: Row) -> Row:
        """Save a delivery once per what it carries: the one already saved under its key is returned instead."""
        held = tx.delivery_by_key(row["key"])
        return held if held is not None else tx.insert_delivery(row)

    @staticmethod
    def _event(what: str, *, thread: Optional[str], team: str, from_node: str, to_node: str, state: str, project: str) -> dict:
        return {"what": what, "thread": thread, "team": team, "from_node": from_node, "to_node": to_node, "state": state,
                "project": project}

    @staticmethod
    def _emit(events: list[dict]) -> None:
        """After the change is saved: one ``team.handoff`` event each, names and states only (never the words)."""
        from .. import experience

        for e in events:
            project = e.pop("project", None)
            experience.record("team.handoff", "agent", e, project=project)

    def _check_problem(self, task: tw.TeamTask) -> None:
        """The task's problem is one its project has, in this workspace (AccessDenied otherwise)."""
        from ..memory.scope import resolve_problem

        resolve_problem(self.projects, task.project_id, task.problem_id, workspace_id=self.workspace)

    # ------------------------------------------------------------------ the facts work rests on (W6)
    def _facts_scope(self, project_id: str, problem_id: str) -> Any:
        """The problem's base as this store reaches it (the workspace's own write, as the team agents' steps are): the
        problem's recorded workspace keeps its key the one people's corrections use."""
        from ..memory.scope import AuthorizedScope, resolve_problem

        prob = resolve_problem(self.projects, project_id, problem_id, workspace_id=self.workspace)
        return AuthorizedScope(workspace_id=self.workspace, principal_id="teamspace", role="owner", project_id=project_id,
                               problem_id=problem_id, problem_workspace=str(prob.get("workspace_id") or "") or None)

    def _check_cites(self, project_id: str, problem_id: str, cites: Sequence[str]) -> None:
        """Refuse a citation of a fact this problem does not hold, or of one at a revision it was never written at, before
        anything is saved."""
        if cites and self.home is not None:
            from ..memory.bases import check_citations

            check_citations(self.home, self._facts_scope(project_id, problem_id), cites)

    def _link(self, project_id: str, task: Optional[tw.TeamTask] = None, report: Optional[tw.WorkReport] = None) -> None:
        """Link what was saved to the facts it cites, in the problem base's ledger, so a later correction marks it stale in
        its own write (whatever happens to the reconciliation after it). An error here is raised: the record is saved, and
        the same call again links it (a retried call is a duplicate that links once)."""
        if self.home is None:
            return
        if report is not None:
            task = task or self.task(project_id, report.task_id)
            cites = report_citations(report)
            link = _report_link(report, task, project_id, cites) if cites else None
        else:
            cites = task_citations(task) if task is not None else []
            link = _task_link(task, cites) if cites else None
        if link is None or task is None:
            return
        from ..memory.bases import link_dependents

        link_dependents(self.home, self._facts_scope(project_id, task.problem_id), [link])

    # ------------------------------------------------------------------ tasks
    def create_task(self, task: tw.TeamTask, *, initiator: Any = None) -> Written:
        """A new top-level task and its pending delivery to the owner, saved together. The same task asked again (same
        id, problem, requester, owner and question; its time may differ) returns the saved one; another task under a taken
        id is refused. A child task is made with :meth:`delegate`."""
        self._own(task)
        if task.parent_task_id is not None:
            raise ValueError("a child task is made with delegate(), which checks its siblings and its budget")
        if task.status != "open":
            raise ValueError("a new task starts open")
        self._check_problem(task)
        self._check_cites(task.project_id, task.problem_id, task_citations(task))
        who = initiator_of(initiator)
        with self._tx(task.project_id) as tx:
            held = tx.task(task.id)
            if held is not None:
                if _identity(held["task"]) != _identity(task):
                    raise ValueError(f"a task {task.id} already exists")
                again = Written(held["task"], tx.delivery_by_key(f"task:{task.id}"), duplicate=True)
            else:
                again = None
                tx.insert_task(task, who)
                row = self._send(tx, self._delivery("task", f"task:{task.id}", task, item_id=task.id, from_team=task.requester.team,
                                                    to_team=task.owner_team, text=task.question, thread_id=None, initiator=who))
        if again is not None:
            self._link(task.project_id, task=again.item)
            return again
        self._emit([self._event("task.created", thread=task.requester.thread_id, team=task.owner_team, from_node=task.requester.team,
                                to_node=task.owner_team, state=task.status, project=task.project_id)])
        self._link(task.project_id, task=task)  # after the save: an error is raised, and the same call again links it
        return Written(task, row)

    def delegate(self, project_id: str, parent_id: str, *, id: str, owner_team: str, question: str, expected_output: str,
                 done_criteria: Sequence[str], by: tw.Actor, thread_id: str, created_at: str, allowed_inputs: Sequence[str] = (),
                 allowed_tools: Sequence[str] = (), budget_eur: float = 0.0, initiator: Any = None, parent_waits: bool = False) -> Written:
        """A child task asked by the parent's owner (``team_work.delegate`` given the stored siblings), saved with its
        delivery. The same child asked again returns the saved one. A child acts for the parent's member when none is given.
        ``parent_waits``: an accepted parent moves to ``waiting_for_team`` in the same transaction (W5), so a crash never
        leaves a child out with its parent still saying it works alone."""
        who = initiator_of(initiator)
        cites = _citations(allowed_inputs)
        if cites:
            held_parent = self.task(project_id, parent_id)
            if held_parent is not None:
                self._check_cites(project_id, held_parent.problem_id, cites)
        child = None
        with self._tx(project_id) as tx:
            parent_row = tx.task(parent_id)
            parent = self._in_project(parent_row, project_id, parent_id)
            who = who or parent_row["initiator"]
            siblings = [r["task"] for r in tx.tasks(parent=parent_id)]
            held = tx.task(id)
            if held is not None:
                same = held["task"]
                if (same.parent_task_id, same.owner_team, same.question) != (parent_id, owner_team, question):
                    raise ValueError(f"a task {id} already exists")
                self._parent_waits(tx, parent_row, by, parent_waits)
                result = Written(same, tx.delivery_by_key(f"task:{id}"), duplicate=True)
            else:
                child = tw.delegate(parent, id=id, owner_team=owner_team, question=question, expected_output=expected_output,
                                    done_criteria=done_criteria, by=by, thread_id=thread_id, created_at=created_at,
                                    allowed_inputs=allowed_inputs, allowed_tools=allowed_tools, budget_eur=budget_eur, siblings=siblings)
                tx.insert_task(child, who)
                row = self._send(tx, self._delivery("task", f"task:{child.id}", child, item_id=child.id, from_team=parent.owner_team,
                                                    to_team=owner_team, text=question, thread_id=None, initiator=who))
                self._parent_waits(tx, parent_row, by, parent_waits)
                result = Written(child, row)
        if child is not None:
            self._emit([self._event("task.created", thread=thread_id, team=owner_team, from_node=parent.owner_team, to_node=owner_team,
                                    state=child.status, project=project_id)])
        self._link(project_id, task=result.item)  # after the save: an error is raised, and the same call again links it
        return result

    @staticmethod
    def _parent_waits(tx: Any, parent_row: Row, by: tw.Actor, wanted: bool) -> None:
        parent = parent_row["task"]
        if wanted and parent.status == "accepted":
            tx.update_task(tw.validate_transition(parent, "waiting_for_team", by), parent_row["version"])

    def move(self, project_id: str, task_id: str, new_status: str, by: tw.Actor, *, superseded_by: Optional[str] = None,
             note: Optional[str] = None, initiator: Any = None) -> Written:
        """The task in its new status (``team_work.validate_transition`` given its stored children and their joined reports).

        ``note`` travels with the move: the clarifying question when the owner waits for the person (required), and the
        person's answer when the owner takes the task up again (``accepted``); each is a delivery of its own."""
        who = initiator_of(initiator)
        if new_status == "waiting_for_person" and not (note or "").strip():
            raise ValueError("a clarification says what it asks: pass note")
        with self._tx(project_id) as tx:
            row = tx.task(task_id)
            task = self._in_project(row, project_id, task_id)
            children = [r["task"] for r in tx.tasks(parent=task_id)]
            moved = tw.validate_transition(task, new_status, by, superseded_by=superseded_by, children=children,
                                           child_reports=self._joined(tx, task_id))
            tx.update_task(moved, row["version"])
            sent = None
            if new_status == "waiting_for_person":
                sent = self._send(tx, self._delivery("clarification", f"clarification:{task_id}:{row['version']}", moved, item_id=task_id,
                                                     from_team=moved.owner_team, to_team=moved.requester.team, text=str(note),
                                                     thread_id=moved.requester.thread_id, initiator=who or row["initiator"]))
            elif task.status == "waiting_for_person" and new_status == "accepted" and (note or "").strip():
                sent = self._send(tx, self._delivery("answer", f"answer:{task_id}:{row['version']}", moved, item_id=task_id,
                                                     from_team=moved.requester.team, to_team=moved.owner_team, text=str(note),
                                                     thread_id=None, initiator=who or row["initiator"]))
        self._emit([self._event("task.moved", thread=moved.requester.thread_id, team=moved.owner_team, from_node=task.status,
                                to_node=new_status, state=new_status, project=project_id)])
        return Written(moved, sent)

    def send_answer(self, project_id: str, task_id: str, *, response_id: str, text: str, initiator: Any = None) -> Written:
        """A person's answer to a task waiting for them, carried by the requesting team to the owner as its own ``answer``
        delivery (W5): saved once per response (``answer:<task>:r-<response>``; the same response again returns the saved
        row and adds nothing). The task does not move here: its owner takes it up again when the row reaches it. Refused
        when the task does not wait for a person."""
        who = initiator_of(initiator)
        key = f"answer:{task_id}:r-{response_id}"
        with self._tx(project_id) as tx:
            row = tx.task(task_id)
            task = self._in_project(row, project_id, task_id)
            held = tx.delivery_by_key(key)
            if held is not None:
                return Written(task, held, duplicate=True)
            if task.status != "waiting_for_person":
                raise ValueError(f"{task_id} is {task.status}: it does not wait for an answer")
            sent = self._send(tx, self._delivery("answer", key, task, item_id=task.id, from_team=task.requester.team,
                                                 to_team=task.owner_team, text=str(text), thread_id=None,
                                                 initiator=who or row["initiator"]))
        self._emit([self._event("answer.sent", thread=task.requester.thread_id, team=task.owner_team, from_node=task.requester.team,
                                to_node=task.owner_team, state="pending", project=project_id)])
        return Written(task, sent)

    # ------------------------------------------------------------------ actions
    def record_action(self, project_id: str, action: tw.TeamAction) -> Written:
        """Record an attempt (usually planned or running; a denial may be recorded ended). The same record again is the same
        record; another record under a taken action id is refused (finish it with :meth:`finish_action`)."""
        events: list[dict] = []
        with self._tx(project_id) as tx:
            task = self._in_project(tx.task(action.task_id), project_id, action.task_id)
            held = tx.action(action.task_id, action.action_id)
            if held is not None:
                if held == action:
                    return Written(held, duplicate=True)
                raise ValueError(f"action {action.action_id} is already recorded: finish it with finish_action")
            self._one_done(tx, action)
            tx.insert_action(action)
            if tw.can_claim(action):
                events.append(self._finished(task, action))
        self._emit(events)
        return Written(action)

    def finish_action(self, project_id: str, task_id: str, action_id: str, status: str, *, ended_at: str,
                      started_at: Optional[str] = None, output_ref: Optional[str] = None, failure_ref: Optional[str] = None,
                      reason: Optional[str] = None, responded_by: Optional[tw.Actor] = None) -> Written:
        """The action's one terminal record. Finishing it again the same way changes nothing; finishing it another way, or
        a second success under its retry key, is refused."""
        if status not in tw.TERMINAL_ACTIONS:
            raise ValueError(f"{status!r} is not an ending: one of {sorted(tw.TERMINAL_ACTIONS)}")
        with self._tx(project_id) as tx:
            task = self._in_project(tx.task(task_id), project_id, task_id)
            held = tx.action(task_id, action_id)
            if held is None:
                raise KeyError(f"no action {action_id} on task {task_id}")
            ended = tw.TeamAction.model_validate({**held.model_dump(), "status": status, "ended_at": ended_at,
                                                  "started_at": held.started_at or started_at or ended_at, "output_ref": output_ref,
                                                  "failure_ref": failure_ref, "reason": reason,
                                                  "responded_by": (responded_by.model_dump() if responded_by is not None
                                                                   else held.model_dump()["responded_by"])})
            if tw.can_claim(held):
                if held == ended:
                    return Written(held, duplicate=True)
                raise ValueError(f"action {action_id} already ended as {held.status}: a second terminal record is refused")
            self._one_done(tx, ended)
            tx.update_action(ended)
        self._emit([self._finished(task, ended)])
        return Written(ended)

    @staticmethod
    def _one_done(tx: Any, action: tw.TeamAction) -> None:
        if action.status in tw.DONE_ACTIONS:
            done = tx.done_by_key(action.retry_key)
            if done is not None and (done.task_id, done.action_id) != (action.task_id, action.action_id):
                raise ValueError(f"retry key {action.retry_key} already has a completed attempt ({done.action_id}): "
                                 "the same side effect is not done twice")

    def _finished(self, task: tw.TeamTask, action: tw.TeamAction) -> dict:
        return self._event("action.finished", thread=task.requester.thread_id, team=action.actor.team or task.owner_team,
                           from_node=action.actor.id, to_node=action.action, state=action.status, project=task.project_id)

    # ------------------------------------------------------------------ reports
    def _store_report(self, project_id: str, task_id: str, *, current_brief_revision: int, answer: str, next_task: tw.NextStep,
                     evidence: Sequence[tw.EvidenceRef] = (), limits: Sequence[str] = (), private_notes: Sequence[str] = (),
                     revision: int = 1, status: Optional[str] = None, created_at: Optional[str] = None, initiator: Any = None) -> Written:
        """The task's report at ``revision``, built by ``team_work.build_report`` from the stored actions and joined child
        reports, and its reply to the requester's thread. A child's report joins its parent only when ``accept_reply``
        accepts it against ``current_brief_revision`` (the problem brief's revision now; else it is kept with the verdict,
        and nothing is sent). The same (task, revision) again returns the stored report and sends nothing. A later revision
        is made with :meth:`supersede_report`."""
        who = initiator_of(initiator)
        with self._tx(project_id) as tx:
            row = tx.task(task_id)
            task = self._in_project(row, project_id, task_id)
            held = tx.report(task_id, revision)
            if held is not None:
                return Written(held["report"], tx.delivery_by_key(f"reply:{held['report'].id}"), self._verdict(held), duplicate=True)
            if revision != 1 and any(r["report"].status != "superseded" for r in tx.reports([task_id])):
                raise ValueError("a later revision replaces the current report: use supersede_report")
            report = self._build(tx, task, answer=answer, next_task=next_task, evidence=evidence, limits=limits,
                                 private_notes=private_notes, revision=revision, status=status, created_at=created_at)
            written = self._keep_report(tx, task, report, current_brief_revision, who or row["initiator"])
        self._emit([self._stored(task, written.item)])
        return written

    def _supersede_report(self, project_id: str, task_id: str, *, current_brief_revision: int, answer: str, next_task: tw.NextStep,
                         evidence: Sequence[tw.EvidenceRef] = (), limits: Sequence[str] = (), private_notes: Sequence[str] = (),
                         status: Optional[str] = None, created_at: Optional[str] = None, initiator: Any = None) -> Written:
        """A correction: the next revision of the task's current report, built from the stored actions, linked both ways
        (``team_work.supersede``), the old one kept as superseded."""
        who = initiator_of(initiator)
        with self._tx(project_id) as tx:
            row = tx.task(task_id)
            task = self._in_project(row, project_id, task_id)
            current = [r for r in tx.reports([task_id]) if r["report"].status != "superseded"]
            if not current:
                raise ValueError(f"task {task_id} has no report to correct: store_report first")
            old = max(current, key=lambda r: r["report"].revision)
            new = self._build(tx, task, answer=answer, next_task=next_task, evidence=evidence, limits=limits,
                              private_notes=private_notes, revision=old["report"].revision + 1, status=status, created_at=created_at)
            old_report, new_report = tw.supersede(old["report"], new)
            tx.update_report({**old, "report": old_report})
            written = self._keep_report(tx, task, new_report, current_brief_revision, who or row["initiator"])
            written.replaced = old_report
        self._emit([self._stored(task, written.item)])
        return written

    def _report_task(self, project_id: str, task_id: str, *, revision: int, current_brief_revision: int, answer: str,
                    next_task: tw.NextStep, move_to: Optional[str] = None, by: Optional[tw.Actor] = None,
                    evidence: Sequence[tw.EvidenceRef] = (), limits: Sequence[str] = (), private_notes: Sequence[str] = (),
                    status: Optional[str] = None, created_at: Optional[str] = None, initiator: Any = None) -> Written:
        """The owner's move (``move_to``: complete or partial, through ``validate_transition``) and its report at
        ``revision`` with the reply, in ONE transaction (W5): a crash leaves both or neither, never a complete task without
        its report. Revision 1 is the first report; a later one supersedes the current report (a correction). The same
        revision again is a duplicate, not a failure, whatever the task's status now (two turns of one agent raced)."""
        who = initiator_of(initiator)
        events: list[dict] = []
        with self._tx(project_id) as tx:
            row = tx.task(task_id)
            task = self._in_project(row, project_id, task_id)
            held = tx.report(task_id, revision)
            if held is not None:
                return Written(held["report"], tx.delivery_by_key(f"reply:{held['report'].id}"), self._verdict(held), duplicate=True)
            current = [r for r in tx.reports([task_id]) if r["report"].status != "superseded"]
            top = max((r["report"].revision for r in current), default=0)
            if revision != top + 1:
                raise ValueError(f"revision {revision} does not follow the current report's revision ({top})")
            if move_to and task.status != move_to:
                if by is None:
                    raise ValueError("a move names who makes it")
                children = [r["task"] for r in tx.tasks(parent=task_id)]
                moved = tw.validate_transition(task, move_to, by, children=children, child_reports=self._joined(tx, task_id))
                tx.update_task(moved, row["version"])
                events.append(self._event("task.moved", thread=moved.requester.thread_id, team=moved.owner_team, from_node=task.status,
                                          to_node=move_to, state=move_to, project=project_id))
                task = moved
            report = self._build(tx, task, answer=answer, next_task=next_task, evidence=evidence, limits=limits,
                                 private_notes=private_notes, revision=revision, status=status, created_at=created_at)
            replaced = None
            if current:
                old = max(current, key=lambda r: r["report"].revision)
                replaced, report = tw.supersede(old["report"], report)
                tx.update_report({**old, "report": replaced})
            written = self._keep_report(tx, task, report, current_brief_revision, who or row["initiator"])
            written.replaced = replaced
        self._emit(events + [self._stored(task, written.item)])
        return written

    def _reported(self, how: Any, project_id: str, task_id: str, kw: dict) -> Written:
        """A report saved through ``how``, its fact citations checked before and linked after (W6)."""
        cites = _citations([v for e in kw.get("evidence") or () if not e.private for v in (e.source, e.ref)])
        if cites:
            task = self.task(project_id, task_id)
            if task is not None:
                self._check_cites(project_id, task.problem_id, cites)
        written = how(project_id, task_id, **kw)
        self._link(project_id, report=written.item)  # after the save: an error is raised, and the same call again links it
        return written

    def store_report(self, project_id: str, task_id: str, **kw: Any) -> Written:
        """The task's report at a revision (see :meth:`_store_report`); a fact it cites is checked first and linked after."""
        return self._reported(self._store_report, project_id, task_id, kw)

    def supersede_report(self, project_id: str, task_id: str, **kw: Any) -> Written:
        """A correction of the task's report (see :meth:`_supersede_report`); its fact citations checked and linked."""
        return self._reported(self._supersede_report, project_id, task_id, kw)

    def report_task(self, project_id: str, task_id: str, **kw: Any) -> Written:
        """The owner's move and its report in one transaction (see :meth:`_report_task`); its fact citations checked and
        linked."""
        return self._reported(self._report_task, project_id, task_id, kw)

    def _build(self, tx: Any, task: tw.TeamTask, **kw: Any) -> tw.WorkReport:
        children = [r["task"] for r in tx.tasks(parent=task.id)]
        return tw.build_report(task, tx.actions(task.id), self._joined(tx, task.id), children=children, **kw)

    def _keep_report(self, tx: Any, task: tw.TeamTask, report: tw.WorkReport, current_brief_revision: int,
                     initiator: Optional[Row]) -> Written:
        verdict = None
        if task.parent_task_id is not None:
            parent = tx.task(task.parent_task_id)["task"]
            accepted = [r.id for r in self._joined(tx, parent.id)]
            verdict = tw.accept_reply(parent, report, int(current_brief_revision), accepted=accepted)
        joined = verdict is not None and verdict.ok
        tx.insert_report({"report": report, "joined": joined, "verdict": _dump(verdict) if verdict else None})
        sent = None
        if verdict is None or verdict.ok:
            sent = self._send(tx, self._delivery("reply", f"reply:{report.id}", task, item_id=report.id, from_team=task.owner_team,
                                                 to_team=task.requester.team, text=reply_text(report), thread_id=task.requester.thread_id,
                                                 initiator=initiator))
        return Written(report, sent, verdict)

    def _stored(self, task: tw.TeamTask, report: tw.WorkReport) -> dict:
        return self._event("report.stored", thread=task.requester.thread_id, team=task.owner_team, from_node=task.owner_team,
                           to_node=task.requester.team, state=report.status, project=task.project_id)

    @staticmethod
    def _verdict(held: Row) -> Optional[tw.Verdict]:
        return tw.Verdict.model_validate(held["verdict"]) if held.get("verdict") else None

    # ------------------------------------------------------------------ reads
    def task(self, project_id: str, task_id: str) -> Optional[tw.TeamTask]:
        with self._read(project_id) as tx:
            row = tx.task(task_id)
        return row["task"] if row is not None and row["task"].project_id == project_id else None

    def task_view(self, project_id: str, task_id: str) -> Optional[Row]:
        """A task with its children, its actions, its reports (every revision), how its children stand, who acts next, and
        its deliveries (a blocked one says why). None outside the project."""
        with self._read(project_id) as tx:
            row = tx.task(task_id)
            if row is None or row["task"].project_id != project_id:
                return None
            task = row["task"]
            children = [r["task"] for r in tx.tasks(parent=task_id)]
            joined = self._joined(tx, task_id)
            reports = [r["report"] for r in tx.reports([task_id])]
            actions = tx.actions(task_id)
            deliveries = tx.deliveries(task_id=task_id)
        return {"task": task, "children": children, "actions": actions, "reports": reports, "child_reports": joined,
                "join": tw.join_status(task, joined, children=children) if children else None,
                "next_owner": tw.next_owner(task, reports + joined, children=children), "deliveries": deliveries}

    def open_tasks(self, team: str, project_id: Optional[str] = None) -> list[tw.TeamTask]:
        """The tasks a team owns that still wait on someone (a clarification keeps a task open), oldest first."""
        with self._read(project_id) as tx:
            rows = tx.tasks(owner=team)
        return [r["task"] for r in rows if tw.is_open(r["task"]) and (project_id is None or r["task"].project_id == project_id)]

    def handoff_history(self, project_id: str, problem_id: str) -> list[Row]:
        """Who asked whom and what came back, in order: ``{n, from_team, to_team, text, state}`` per thing that travelled
        or is on its way (a refused row never left, and is not shown). No thread id, no private note: other teams may read
        it. The text is cut for showing; the row keeps it whole."""
        with self._read(project_id) as tx:
            rows = [r for r in tx.deliveries(problem_id=problem_id) if r["status"] != "refused"]
        return [{"n": n, "from_team": team_label(r["from_team"]), "to_team": team_label(r["to_team"]), "text": _shown(_first_line(r["text"])),
                 "state": r["status"]} for n, r in enumerate(rows, 1)]

    def problem_records(self, project_id: str, problem_id: str) -> Row:
        """Every task of one problem in the project and every report of those tasks (each revision), in the order made:
        what a correction of the problem's facts is checked against (W6)."""
        with self._read(project_id) as tx:
            tasks = [r["task"] for r in tx.tasks(problem_id=problem_id) if r["task"].project_id == project_id]
            reports = [r["report"] for r in tx.reports([t.id for t in tasks])] if tasks else []
        return {"tasks": tasks, "reports": reports}

    # ------------------------------------------------------------------ deliveries (dclab_rnd/teamspace/delivery.py)
    def delivery(self, delivery_id: str, project_id: Optional[str] = None) -> Optional[Row]:
        with self._read(project_id) as tx:
            return tx.delivery(delivery_id)

    def deliveries(self, status: Optional[str] = None, project_id: Optional[str] = None, task_id: Optional[str] = None) -> list[Row]:
        """Rows in delivery order (a task's oldest first); ``project_id`` None reads every project of the workspace."""
        with self._read(project_id) as tx:
            return tx.deliveries(status=status, task_id=task_id)

    def pending_heads(self) -> list[Row]:
        """The oldest pending row of each task that has one: what each task delivers next."""
        heads: dict[tuple[str, str], Row] = {}
        for row in self.deliveries(status="pending"):
            heads.setdefault((row["project_id"], row["task_id"]), row)
        return list(heads.values())

    def _change_delivery(self, delivery_id: str, fn, project_id: Optional[str] = None) -> tuple[Row, bool]:
        first = self.delivery(delivery_id, project_id)
        if first is None:
            raise KeyError(delivery_id)
        with self._tx(first["project_id"]) as tx:
            row = tx.delivery(delivery_id)
            changed = copy.deepcopy(row)
            if row["status"] != "pending" or not fn(changed):
                return row, False
            return tx.update_delivery(changed, row["version"]), True

    def _ended(self, row: Row, status: str) -> None:
        self._emit([self._event(f"{row['kind']}.{status}", thread=row.get("thread_id"), team=row["to_team"], from_node=row["from_team"],
                                to_node=row["to_team"], state=status, project=row["project_id"])])

    def delivery_done(self, delivery_id: str, receipt: Any = None, project_id: Optional[str] = None) -> tuple[Row, bool]:
        """Mark it delivered, once: a reply is posted to its thread here and nowhere else. False when it was not pending."""
        at = _now()

        def put(row: Row) -> bool:
            row.update(status="delivered", delivered_at=at, next_attempt_at=None, receipt=_json(receipt))
            if row.get("thread_id"):
                row["posted"] = {"thread_id": row["thread_id"], "at": at}
            return True
        row, changed = self._change_delivery(delivery_id, put, project_id)
        if changed:
            self._ended(row, "delivered")
        return row, changed

    def delivery_failed(self, delivery_id: str, error: BaseException | str, *, max_attempts: int, backoff_seconds: float,
                        cap_seconds: float = 60.0, project_id: Optional[str] = None) -> Row:
        """One failed attempt: counted, and the next one waits longer (doubling, capped), its time on the row; at
        ``max_attempts`` the row is blocked. The requester's reason is fixed words and the error's class; the error's text is
        kept in ``last_error``, scrubbed."""
        kind = type(error).__name__ if isinstance(error, BaseException) else "DeliveryError"

        def put(row: Row) -> bool:
            row["attempts"] = int(row.get("attempts") or 0) + 1
            row["last_error"] = scrub(f"{kind}: {error}")
            if row["attempts"] >= max_attempts:
                row.update(status="blocked", next_attempt_at=None,
                           reason=f"Not delivered to {team_label(row['to_team'])} after {row['attempts']} attempts ({kind}).")
            else:
                wait = min(cap_seconds, backoff_seconds * (2 ** (row["attempts"] - 1)))
                row["next_attempt_at"] = (datetime.now(timezone.utc) + timedelta(seconds=wait)).isoformat(timespec="milliseconds")
            return True
        row, changed = self._change_delivery(delivery_id, put, project_id)
        if changed and row["status"] == "blocked":
            self._ended(row, "blocked")
        return row

    def delivery_stop(self, delivery_id: str, status: str, reason: str, *, error: Any = None, project_id: Optional[str] = None) -> Row:
        """End a pending delivery without delivering it: ``refused`` (the member no longer has the authority) or ``blocked``.
        ``reason`` is the caller's fixed words (the requester sees them); ``error`` is kept apart, scrubbed."""
        assert status in ("blocked", "refused"), status

        def put(row: Row) -> bool:
            row.update(status=status, reason=reason, next_attempt_at=None)
            if error is not None:
                row["last_error"] = scrub(f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error)
            return True
        row, changed = self._change_delivery(delivery_id, put, project_id)
        if changed:
            self._ended(row, status)
        return row

    def counts(self) -> dict[str, int]:
        with self._read(None) as tx:
            return tx.counts()


# ---------------------------------------------------------------------------------------------------------- facts (W6)
# Team work rests on the problem's facts by citing them, ``fact:<entry id>@r<revision>`` (``memory.bases.fact_citation``):
# a task in its allowed inputs, a report in an evidence source (or ref), a decision in its constraint revisions. A
# correction of the fact (``memory.bases.correct_base_fact``) calls :func:`mark_fact_dependents_stale`: every dependent
# citing the old revision is marked stale in the problem base's ledger, then what rested on it (a report joining a stale
# child, a task whose current report is stale, a decision on a stale report), and each requester's thread gets one alert
# through W3's post. Work that cites nothing corrected is never touched (the kitchen's correction is a superseded report,
# not a fact: its acceptance report keeps its snapshot).


def task_ref(project_id: str, task_id: str) -> str:
    return f"team_task:{project_id}/{task_id}"


def report_ref(project_id: str, report_id: str) -> str:
    return f"work_report:{project_id}/{report_id}"


def decision_ref(decision_id: str) -> str:
    return f"team_decision:{decision_id}"


def fact_citation(entry: Mapping[str, Any]) -> str:
    """A base fact at its revision, as a task, a report or a decision cites it (``fact:<entry id>@r<revision>``)."""
    from ..memory.bases import fact_citation as cite

    return cite(entry)


def fact_evidence(entry: Mapping[str, Any], *, uncertainty: Optional[str] = None) -> tw.EvidenceRef:
    """A report's evidence that it rests on a problem-base fact: its source is the citation (so the task's allowed inputs
    can name it), its ref the memory item, its input revision the fact's revision."""
    citation = fact_citation(entry)
    return tw.EvidenceRef(source=citation, ref=f"memory:{entry['item_id']}", input_revision=citation.rsplit("@", 1)[1],
                          uncertainty=uncertainty)


def _citations(values: Sequence[Any]) -> list[str]:
    from ..memory.bases import parse_citation

    return list(dict.fromkeys(str(v).strip() for v in values if parse_citation(v)))


def task_citations(task: tw.TeamTask) -> list[str]:
    return _citations(task.allowed_inputs)


def report_citations(report: tw.WorkReport) -> list[str]:
    return _citations([v for e in report.evidence for v in (e.source, e.ref)])


def _task_link(task: tw.TeamTask, cites: Sequence[str]) -> Row:
    return {"ref": task_ref(task.project_id, task.id), "kind": "team_task", "cites": list(cites), "project_id": task.project_id,
            "task_id": task.id, "owner_team": task.owner_team, "requester_team": task.requester.team,
            "thread_id": task.requester.thread_id, "label": f"task {task.id} ({team_label(task.owner_team)})"}


def _report_link(report: tw.WorkReport, task: Optional[tw.TeamTask], project_id: str, cites: Sequence[str]) -> Row:
    out = {"ref": report_ref(project_id, report.id), "kind": "work_report", "cites": list(cites), "project_id": project_id,
           "task_id": report.task_id, "report_id": report.id, "owner_team": report.owner, "label": f"report {report.id}"}
    if task is not None:
        out.update(requester_team=task.requester.team, thread_id=task.requester.thread_id)
    return out


def _problem_scope(scope: Any) -> None:
    from ..memory.scope import AccessDenied

    if not (getattr(scope, "project_id", None) and getattr(scope, "problem_id", None)):
        raise AccessDenied("a problem's team work needs an authorized project_id and problem_id")


def record_decision(home: Path, scope: Any, decision: tw.TeamDecision, *, thread_id: Optional[str] = None, log: Any = None) -> Row:
    """Keep a decision the owner made (or is asked to make) in the problem base's ledger, with the fact revisions it rests
    on (its constraint revisions that cite facts) and the reports it rests on (its evidence revisions). A decision citing
    a fact already corrected is stale at once. Recorded by a person who may write; the approval itself is the decision
    owner's alone (``team_work`` refuses any other approver). Recording the same decision again (its approval, its
    outcome) replaces the stored record; a record naming another decision owner is refused. Returns its ledger entry."""
    from ..memory import bases
    from ..memory.scope import AccessDenied

    _problem_scope(scope)
    if not scope.can_write:
        raise AccessDenied(f"role {scope.role!r} cannot record a decision")
    if decision.problem_id != scope.problem_id:
        raise AccessDenied(f"decision {decision.id} belongs to problem {decision.problem_id}, not {scope.problem_id}")
    bases.check_citations(home, scope, _citations(decision.constraint_revisions))  # a forged or foreign citation: refused
    link = {"ref": decision_ref(decision.id), "kind": "team_decision", "cites": _citations(decision.constraint_revisions),
            "project_id": scope.project_id, "decision_id": decision.id, "thread_id": thread_id,
            "decision_owner": decision.decision_owner.id, "label": f"decision {decision.id}", "doc": _dump(decision),
            "stale_via": [report_ref(scope.project_id, r) for r in decision.evidence_revisions]}
    deps = bases.link_dependents(home, scope, [link], log=log)  # kept even when it cites nothing here, to be read back
    return next(d for d in deps if d["ref"] == link["ref"])


def stale_marks(home: Path, scope: Any) -> list[Row]:
    """The problem's team work marked stale, each with what changed under it and whether its requester was alerted."""
    from ..memory import bases

    _problem_scope(scope)
    return [d for d in bases.read_base(home, scope, "problem").get("dependents", []) if d.get("stale")]


def decisions(home: Path, scope: Any) -> list[Row]:
    """The problem's recorded decisions with their ledger state (stale, needs_review)."""
    from ..memory import bases

    _problem_scope(scope)
    out = []
    for d in bases.read_base(home, scope, "problem").get("dependents", []):
        if d.get("kind") == "team_decision" and d.get("doc"):
            out.append({**d, "decision": tw.TeamDecision.model_validate(d["doc"])})
    return out


def _alert_text(group: Sequence[Row], old: Optional[Row], new: Optional[Row], revision: int) -> str:
    what = str((old or {}).get("field") or "a fact")
    if old is not None and new is not None:
        change = f"\"{old.get('text')}\" is now \"{new.get('text')}\""
    elif old is not None:
        change = f"\"{old.get('text')}\" was withdrawn"
    else:
        change = "a fact it rested on changed"
    parts = []
    for d in group:
        label = str(d.get("label") or d["ref"])
        if d.get("kind") == "team_decision":
            label += f" (needs {d.get('decision_owner') or 'its owner'}'s review)"
        parts.append(label)
    return (f"Correction on this problem ({what}): {change} (problem base revision {revision}). "
            f"Stale until rechecked: {', '.join(parts)}. Work that did not rest on it is unchanged.")


def mark_fact_dependents_stale(home: Path, scope: Any, *, work: Any = None, threads: Any = None, log: Any = None) -> Row:
    """Reconcile the problem's team work with its facts (called by every correction or retirement of a problem fact, and
    safe to call again at any time, from any session: every step is idempotent).

    1. Find the dependents: each task, report and recorded decision citing a fact of this problem's base, linked in the
       base's ledger; whatever cites a revision no longer current is marked stale there (a correction already marked the
       ones linked before it, in its own write).
    2. Mark what rested on a stale item: a report joining a stale child report, a task whose current report is stale, a
       decision resting on a stale report (a decision also waits for its owner's review).
    3. Alert each requester once per corrected fact: one message in its thread (W3's ``post``, keyed so a retry adds
       nothing), from the agent of the team whose work rested on it; then record that it was told.
    Tasks and reports citing nothing corrected are not touched. Returns the stale refs and the alerts posted."""
    from ..memory import bases

    from ..memory.scope import AccessDenied

    _problem_scope(scope)
    if not scope.can_write:
        raise AccessDenied(f"role {scope.role!r} cannot mark the problem's team work stale")
    base = bases.read_base(home, scope, "problem")
    facts = {str(e["entry_id"]): e for e in base["active"] + base["retired"] if e.get("kind") == "fact"}
    work = work if work is not None else open_team_work(home)
    records = work.problem_records(scope.project_id, scope.problem_id)
    tasks = {t.id: t for t in records["tasks"]}
    pid = scope.project_id

    def here(cites: Sequence[str]) -> list[str]:
        """The citations naming a fact of this base at the revision it was written at (one stored before citations were
        checked, naming another revision, is never a reason to mark anything)."""
        return [c for c in cites if (bases.parse_citation(c) or ("", 0))[0] in facts and not bases.citation_problems(facts.values(), [c])]

    links = [_task_link(t, here(task_citations(t))) for t in tasks.values() if here(task_citations(t))]
    links += [_report_link(r, tasks.get(r.task_id), pid, here(report_citations(r))) for r in records["reports"] if here(report_citations(r))]
    deps = {d["ref"]: d for d in (bases.link_dependents(home, scope, links, log=log) if links else base.get("dependents") or [])}
    def roots(ref: str) -> set[str]:
        return {str(c["old_entry_id"]) for c in (deps.get(ref) or {}).get("changed_sources") or [] if c.get("old_entry_id")}

    def behind(ref: str, via: Sequence[str]) -> bool:
        """Whether ``ref`` misses a stale mark, or a corrected fact, that what it rests on carries (a second correction
        under a dependent already stale from the first still reaches it)."""
        return bool(via) and (not (deps.get(ref) or {}).get("stale") or bool(set().union(*(roots(v) for v in via)) - roots(ref)))

    for _ in range(2 * (len(tasks) + len(records["reports"]) + len(deps)) + 1):  # a fixed point: each round adds a mark or a fact
        stale = {r for r, d in deps.items() if d.get("stale")}
        more: list[Row] = []
        for r in records["reports"]:
            via = [report_ref(pid, c) for c in r.child_reports if report_ref(pid, c) in stale]
            if behind(report_ref(pid, r.id), via):
                more.append({**_report_link(r, tasks.get(r.task_id), pid, []), "stale_via": via})
        for t in tasks.values():
            via = [report_ref(pid, r.id) for r in records["reports"] if r.task_id == t.id and r.status != "superseded"
                   and report_ref(pid, r.id) in stale]
            if behind(task_ref(pid, t.id), via):
                more.append({**_task_link(t, []), "stale_via": via})
        for ref, d in list(deps.items()):
            if d.get("kind") == "team_decision" and d.get("doc"):
                via = [report_ref(pid, r) for r in d["doc"].get("evidence_revisions") or [] if report_ref(pid, r) in stale]
                if behind(ref, via):
                    more.append({"ref": ref, "kind": "team_decision", "stale_via": via})
        if not more:
            break
        deps = {d["ref"]: d for d in bases.link_dependents(home, scope, more, log=log)}

    # 3. one alert per (thread, corrected fact), naming every item of that thread resting on it
    groups: dict[tuple[str, str], list[Row]] = {}
    for ref, d in deps.items():
        if not d.get("stale"):
            continue
        where = str(d.get("thread_id") or (f"team:{d['requester_team']}" if d.get("requester_team") else ""))
        if not where:
            continue  # nobody to tell (a decision recorded with no thread): it stays stale and visible in the ledger
        roots = {str(c["old_entry_id"]) for c in d.get("changed_sources") or [] if c.get("old_entry_id")}
        for root in sorted(roots - set(d.get("alerted") or [])):
            groups.setdefault((where, root), []).append(d)
    posted: list[Row] = []
    if groups:
        from .threads import open_threads, team_key

        threads = threads if threads is not None else open_threads(home)
        teams = None
        for (where, root), group in sorted(groups.items()):
            old = facts.get(root)
            new = facts.get(str((old or {}).get("retired_by") or "")) if old else None
            target = threads.thread(where) if not where.startswith("team:") else None
            if target is None or (target["project_id"], target["problem_id"]) != (scope.project_id, scope.problem_id):
                team = str(group[0].get("requester_team") or "")
                if not team:
                    continue
                if teams is None:
                    from ..storage import open_teams

                    teams = open_teams(Path(home))
                key, label = team_key(teams, team)
                target = threads.open_thread(scope.project_id, scope.problem_id, key, label=label)[0]
            refs = sorted(d["ref"] for d in group)
            author_team = next((str(d["owner_team"]) for d in group if d.get("kind") == "team_task" and d.get("owner_team")),
                               next((str(d["owner_team"]) for d in group if d.get("owner_team")), "problem"))
            digest = hashlib.sha256("|".join(refs).encode("utf-8")).hexdigest()[:12]
            message, made = threads.post(target["id"], {"kind": "agent", "id": author_team, "team": author_team},
                                         _alert_text(group, old, new, int(base["revision"])),
                                         key=f"stale:{scope.problem_id}:{root}:{target['id']}:{digest}")
            bases.mark_alerted(home, scope, {d["ref"]: [root] for d in group})
            posted.append({"thread_id": target["id"], "message_id": message["id"], "posted": made, "fact": root, "refs": refs})
    return {"stale": sorted(r for r, d in deps.items() if d.get("stale")), "alerts": posted}


def _identity(task: tw.TeamTask) -> tuple:
    """What makes two asks the same task (its time and status may differ on a retry)."""
    return (task.id, task.project_id, task.problem_id, task.parent_task_id, task.requester.team, task.requester.thread_id,
            task.owner_team, task.question)


def _first_line(text: str) -> str:
    """A reply's first line: its answer (labelled when simulated); the lines after it are the thread's details."""
    return str(text).split("\n", 1)[0]


def simulated_sources(report: tw.WorkReport) -> list[str]:
    """The fixtures a simulated report rests on (``E1``, ``A1``, ``sample:renewal_v1``), in order, once each."""
    out: list[str] = []
    for e in report.evidence:
        name = e.ref[len("fixture:"):] if e.ref.startswith("fixture:") else e.source if e.source.startswith("sample:") else None
        if name and name not in out:
            out.append(name)
    return out


def reply_text(report: tw.WorkReport) -> str:
    """What a reply says in its thread (W5), so a reader answers the four questions from the thread alone:

    1. what was decided or found: the answer, labelled ``Simulated (fixture E1/A1):`` when anything in it was simulated;
    2. ``Why:`` the evidence and the first limit (or the failed attempt), on one line;
    3. ``Next:`` who acts next and what unblocks them, or that nothing is owed;
    4. ``Changed:`` the report it replaces and the current one, or that it is the first.
    The row keeps it whole; the handoff history shows its first line."""
    head = ""
    if report.simulated:
        names = simulated_sources(report)
        head = f"Simulated (fixture {'/'.join(names)}): " if names else "Simulated: "
    why = [e.ref for e in report.evidence[:3]] + [f"{l.action} {l.status}: {l.ref}" for l in report.actions.failed[:2] if l.ref]
    why += report.limits[:1]
    nxt = report.next_task
    lines = [head + report.answer,
             "Why: " + ("; ".join(why) if why else "the report's actions"),
             "Next: closed, nothing is owed." if nxt.closed else f"Next: {team_label(str(nxt.owner))} — {nxt.unblocks}.",
             f"Changed: replaces {report.supersedes}; the current report is {report.id}." if report.supersedes
             else f"Changed: first report ({report.id})."]
    return "\n".join(lines)


def _shown(text: str) -> str:
    return text if len(text) <= SHOWN else text[: SHOWN - 1].rstrip() + "…"


def _json(value: Any) -> Any:
    """A target's receipt as kept on the row, whole: JSON as it is, anything else as its words."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


# ---------------------------------------------------------------------------------------------------------- files
def _task_row(entry: Row) -> Row:
    return {"task": tw.TeamTask.model_validate(entry["doc"]), "version": entry["version"], "initiator": entry.get("initiator"),
            "seq": entry["seq"]}


def _report_row(entry: Row) -> Row:
    return {"report": tw.WorkReport.model_validate(entry["doc"]), "joined": bool(entry.get("joined")), "verdict": entry.get("verdict"),
            "seq": entry["seq"]}


class _FileTx:
    """One project's ``work.json`` (or, for a read of every project, all of them side by side), changed in memory; the
    caller writes a project's document back whole, or drops it on an error."""

    def __init__(self, data: Row):
        self.data = data
        for k in ("tasks", "actions", "reports", "deliveries"):
            data.setdefault(k, {})
        data.setdefault("seq", 0)
        self.dirty = False

    def _next(self) -> int:
        self.data["seq"] += 1
        self.dirty = True
        return self.data["seq"]

    # tasks
    def task(self, task_id: str) -> Optional[Row]:
        entry = self.data["tasks"].get(task_id)
        return _task_row(entry) if entry else None

    def tasks(self, *, problem_id=None, parent=None, owner=None) -> list[Row]:
        out = []
        for entry in sorted(self.data["tasks"].values(), key=lambda e: e["seq"]):
            d = entry["doc"]
            if ((problem_id is None or d["problem_id"] == problem_id) and (parent is None or d.get("parent_task_id") == parent)
                    and (owner is None or d["owner_team"] == owner)):
                out.append(_task_row(entry))
        return out

    def insert_task(self, task: tw.TeamTask, initiator: Optional[Row]) -> None:
        self.data["tasks"][task.id] = {"doc": _dump(task), "version": 1, "initiator": initiator, "seq": self._next()}

    def update_task(self, task: tw.TeamTask, version: int) -> None:
        entry = self.data["tasks"][task.id]
        if entry["version"] != version:
            raise Conflict(f"task {task.id} changed while it was being written")
        entry.update(doc=_dump(task), version=version + 1)
        self.dirty = True

    # actions
    def action(self, task_id: str, action_id: str) -> Optional[tw.TeamAction]:
        entry = self.data["actions"].get(f"{task_id}/{action_id}")
        return tw.TeamAction.model_validate(entry["doc"]) if entry else None

    def actions(self, task_id: str) -> list[tw.TeamAction]:
        return [tw.TeamAction.model_validate(e["doc"]) for e in sorted(self.data["actions"].values(), key=lambda e: e["seq"])
                if e["doc"]["task_id"] == task_id]

    def done_by_key(self, retry_key: str) -> Optional[tw.TeamAction]:
        found = [e for e in self.data["actions"].values() if e["doc"]["retry_key"] == retry_key and e["doc"]["status"] in tw.DONE_ACTIONS]
        return tw.TeamAction.model_validate(found[0]["doc"]) if found else None

    def insert_action(self, action: tw.TeamAction) -> None:
        self.data["actions"][f"{action.task_id}/{action.action_id}"] = {"doc": _dump(action), "seq": self._next()}

    def update_action(self, action: tw.TeamAction) -> None:
        self.data["actions"][f"{action.task_id}/{action.action_id}"]["doc"] = _dump(action)
        self.dirty = True

    # reports
    def report(self, task_id: str, revision: int) -> Optional[Row]:
        entry = self.data["reports"].get(tw.report_id(task_id, revision))
        return _report_row(entry) if entry else None

    def reports(self, task_ids: Sequence[str]) -> list[Row]:
        wanted = set(task_ids)
        return [_report_row(e) for e in sorted(self.data["reports"].values(), key=lambda e: e["seq"]) if e["doc"]["task_id"] in wanted]

    def insert_report(self, row: Row) -> None:
        report = row["report"]
        self.data["reports"][report.id] = {"doc": _dump(report), "joined": row["joined"], "verdict": row["verdict"], "seq": self._next()}

    def update_report(self, row: Row) -> None:
        entry = self.data["reports"][row["report"].id]
        entry.update(doc=_dump(row["report"]), joined=row["joined"], verdict=row.get("verdict"))
        self.dirty = True

    # deliveries
    def delivery(self, delivery_id: str) -> Optional[Row]:
        row = self.data["deliveries"].get(delivery_id)
        return copy.deepcopy(row) if row else None

    def delivery_by_key(self, key: str) -> Optional[Row]:
        found = [r for r in self.data["deliveries"].values() if r["key"] == key]
        return copy.deepcopy(found[0]) if found else None

    def insert_delivery(self, row: Row) -> Row:
        stored = {**row, "seq": self._next(), "version": 1}
        self.data["deliveries"][row["id"]] = stored
        return copy.deepcopy(stored)

    def update_delivery(self, row: Row, version: int) -> Row:
        held = self.data["deliveries"][row["id"]]
        if held["version"] != version:
            raise Conflict(f"delivery {row['id']} changed while it was being written")
        self.data["deliveries"][row["id"]] = {**row, "version": version + 1}
        self.dirty = True
        return copy.deepcopy(self.data["deliveries"][row["id"]])

    def deliveries(self, *, status=None, problem_id=None, task_id=None) -> list[Row]:
        rows = sorted(self.data["deliveries"].values(), key=lambda r: (r["project_id"], r["seq"]))
        return [copy.deepcopy(r) for r in rows if (status is None or r["status"] == status)
                and (problem_id is None or r["problem_id"] == problem_id) and (task_id is None or r["task_id"] == task_id)]

    def counts(self) -> dict[str, int]:
        return {k: len(self.data[k]) for k in ("tasks", "actions", "reports", "deliveries")}


class FileTeamWork(_TeamWork):
    """``teamspace/<project>/work.json`` in the workspace folder: each read and replaced whole under its project's lock."""

    def __init__(self, home: Path, workspace: str = "local", projects: Any = None):
        self.home, self.workspace, self._projects = Path(home), workspace, projects
        self.folder = self.home / FOLDER

    def location(self) -> str:
        return f"files in {self.folder}"

    def project_dir(self, project_id: str) -> Path:
        name = project_id if _SAFE_DIR.match(project_id) and ".." not in project_id else \
            "p_" + hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:24]
        return self.folder / name

    @staticmethod
    def _load(path: Path) -> Row:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a team work document")
        return data

    @staticmethod
    def _write(folder: Path, data: Row) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".work.", suffix=".part", dir=folder)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        os.replace(temporary, folder / FILE)

    @contextlib.contextmanager
    def _tx(self, project_id: str) -> Iterator[_FileTx]:
        from ..filelock import held

        folder = self.project_dir(project_id)
        with held(folder):
            tx = _FileTx(self._load(folder / FILE))
            tx.data["project_id"] = project_id
            yield tx  # an exception here leaves the file as it was: the change is all or nothing
            if tx.dirty:
                self._write(folder, tx.data)

    def _project_folders(self) -> list[Path]:
        return sorted(p.parent for p in self.folder.glob(f"*/{FILE}")) if self.folder.is_dir() else []

    @contextlib.contextmanager
    def _read(self, project_id: Optional[str] = None) -> Iterator[_FileTx]:
        from ..filelock import held

        folders = [self.project_dir(project_id)] if project_id is not None else self._project_folders()
        merged: Row = {"tasks": {}, "actions": {}, "reports": {}, "deliveries": {}}
        for folder in folders:
            if not (folder / FILE).is_file():
                continue
            with held(folder):
                data = self._load(folder / FILE)
            if project_id is not None:
                merged = data
                break
            for k in ("tasks", "actions", "reports"):  # side by side: ids are unique only within a project
                merged[k].update({f"{data.get('project_id')}/{key}": v for key, v in (data.get(k) or {}).items()})
            merged["deliveries"].update(data.get("deliveries") or {})  # a delivery id is unique in the workspace
        yield _FileTx(merged)


# ---------------------------------------------------------------------------------------------------------- PostgreSQL
class _PgTx:
    """The rows of one workspace's project (or, for a read with no project, of every project) on one connection."""

    def __init__(self, c: Any, workspace_id: str, project_id: Optional[str]):
        import sqlalchemy as sa

        from ..storage import models

        self.c, self.ws, self.project, self.sa, self.m = c, workspace_id, project_id, sa, models

    def _scope(self, t) -> list:
        return [t.c.workspace_id == self.ws] + ([t.c.project_id == self.project] if self.project is not None else [])

    def _need(self) -> str:
        if self.project is None:
            raise ValueError("this read or change names its project")
        return self.project

    # tasks
    def _task_rows(self, *where) -> list[Row]:
        t = self.m.team_task
        q = self.sa.select(t.c.doc, t.c.version, t.c.initiator, t.c.seq).where(*self._scope(t), *where).order_by(t.c.seq)
        return [{"task": tw.TeamTask.model_validate(r.doc), "version": r.version, "initiator": r.initiator, "seq": r.seq}
                for r in self.c.execute(q)]

    def task(self, task_id: str) -> Optional[Row]:
        self._need()
        found = self._task_rows(self.m.team_task.c.id == task_id)
        return found[0] if found else None

    def tasks(self, *, problem_id=None, parent=None, owner=None) -> list[Row]:
        t, where = self.m.team_task, []
        for column, value in ((t.c.problem_id, problem_id), (t.c.parent_task_id, parent), (t.c.owner_team, owner)):
            if value is not None:
                where.append(column == value)
        return self._task_rows(*where)

    def insert_task(self, task: tw.TeamTask, initiator: Optional[Row]) -> None:
        t = self.m.team_task
        self.c.execute(self.sa.insert(t).values(workspace_id=self.ws, project_id=self._need(), id=task.id, problem_id=task.problem_id,
                                                parent_task_id=task.parent_task_id, owner_team=task.owner_team,
                                                requester_team=task.requester.team, status=task.status, initiator=initiator, doc=_dump(task)))

    def update_task(self, task: tw.TeamTask, version: int) -> None:
        t = self.m.team_task
        done = self.c.execute(self.sa.update(t).where(*self._scope(t), t.c.id == task.id, t.c.version == version)
                              .values(doc=_dump(task), status=task.status, version=t.c.version + 1, updated=self.sa.func.now()))
        if done.rowcount != 1:
            raise Conflict(f"task {task.id} changed while it was being written; nothing was written")

    # actions
    def _actions(self, *where) -> list[tw.TeamAction]:
        a = self.m.team_action
        q = self.sa.select(a.c.doc).where(*self._scope(a), *where).order_by(a.c.seq)
        return [tw.TeamAction.model_validate(r.doc) for r in self.c.execute(q)]

    def action(self, task_id: str, action_id: str) -> Optional[tw.TeamAction]:
        a = self.m.team_action
        found = self._actions(a.c.task_id == task_id, a.c.action_id == action_id)
        return found[0] if found else None

    def actions(self, task_id: str) -> list[tw.TeamAction]:
        return self._actions(self.m.team_action.c.task_id == task_id)

    def done_by_key(self, retry_key: str) -> Optional[tw.TeamAction]:
        a = self.m.team_action
        found = self._actions(a.c.retry_key == retry_key, a.c.status.in_(sorted(tw.DONE_ACTIONS)))
        return found[0] if found else None

    def insert_action(self, action: tw.TeamAction) -> None:
        a = self.m.team_action
        self.c.execute(self.sa.insert(a).values(workspace_id=self.ws, project_id=self._need(), task_id=action.task_id, action_id=action.action_id,
                                                retry_key=action.retry_key, status=action.status, doc=_dump(action)))

    def update_action(self, action: tw.TeamAction) -> None:
        a = self.m.team_action
        self.c.execute(self.sa.update(a).where(*self._scope(a), a.c.task_id == action.task_id, a.c.action_id == action.action_id)
                       .values(doc=_dump(action), status=action.status, updated=self.sa.func.now()))

    # reports
    def _reports(self, *where) -> list[Row]:
        r = self.m.team_report
        q = self.sa.select(r.c.doc, r.c.joined, r.c.verdict, r.c.seq).where(*self._scope(r), *where).order_by(r.c.seq)
        return [{"report": tw.WorkReport.model_validate(x.doc), "joined": bool(x.joined), "verdict": x.verdict, "seq": x.seq}
                for x in self.c.execute(q)]

    def report(self, task_id: str, revision: int) -> Optional[Row]:
        r = self.m.team_report
        found = self._reports(r.c.task_id == task_id, r.c.revision == revision)
        return found[0] if found else None

    def reports(self, task_ids: Sequence[str]) -> list[Row]:
        return self._reports(self.m.team_report.c.task_id.in_(list(task_ids))) if task_ids else []

    def insert_report(self, row: Row) -> None:
        r, report = self.m.team_report, row["report"]
        self.c.execute(self.sa.insert(r).values(workspace_id=self.ws, project_id=self._need(), task_id=report.task_id, revision=report.revision,
                                                parent_task_id=report.parent_task_id, status=report.status, joined=bool(row["joined"]),
                                                verdict=row.get("verdict"), doc=_dump(report)))

    def update_report(self, row: Row) -> None:
        r, report = self.m.team_report, row["report"]
        self.c.execute(self.sa.update(r).where(*self._scope(r), r.c.task_id == report.task_id, r.c.revision == report.revision)
                       .values(doc=_dump(report), status=report.status, joined=bool(row["joined"]), verdict=row.get("verdict"),
                               updated=self.sa.func.now()))

    # deliveries
    def _deliveries(self, *where) -> list[Row]:
        d = self.m.team_delivery
        q = self.sa.select(d.c.doc, d.c.seq, d.c.version).where(*self._scope(d), *where).order_by(d.c.project_id, d.c.seq)
        return [{**x.doc, "seq": x.seq, "version": x.version} for x in self.c.execute(q)]

    def delivery(self, delivery_id: str) -> Optional[Row]:
        found = self._deliveries(self.m.team_delivery.c.id == delivery_id)
        return found[0] if found else None

    def delivery_by_key(self, key: str) -> Optional[Row]:
        self._need()
        found = self._deliveries(self.m.team_delivery.c.key == key)
        return found[0] if found else None

    @staticmethod
    def _columns(row: Row) -> Row:
        at = row.get("next_attempt_at")
        doc = {k: v for k, v in row.items() if k not in ("seq", "version")}
        return {"status": row["status"], "attempts": int(row.get("attempts") or 0),
                "next_attempt_at": datetime.fromisoformat(at) if at else None, "doc": doc}

    def insert_delivery(self, row: Row) -> Row:
        d = self.m.team_delivery
        self.c.execute(self.sa.insert(d).values(id=row["id"], workspace_id=self.ws, key=row["key"], kind=row["kind"], project_id=self._need(),
                                                problem_id=row["problem_id"], task_id=row["task_id"], **self._columns(row)))
        return self.delivery(row["id"])

    def update_delivery(self, row: Row, version: int) -> Row:
        d = self.m.team_delivery
        done = self.c.execute(self.sa.update(d).where(*self._scope(d), d.c.id == row["id"], d.c.version == version)
                              .values(version=d.c.version + 1, updated=self.sa.func.now(), **self._columns(row)))
        if done.rowcount != 1:
            raise Conflict(f"delivery {row['id']} changed while it was being written; nothing was written")
        return self.delivery(row["id"])

    def deliveries(self, *, status=None, problem_id=None, task_id=None) -> list[Row]:
        d, where = self.m.team_delivery, []
        for column, value in ((d.c.status, status), (d.c.problem_id, problem_id), (d.c.task_id, task_id)):
            if value is not None:
                where.append(column == value)
        return self._deliveries(*where)

    def counts(self) -> dict[str, int]:
        m = self.m
        return {name: int(self.c.execute(self.sa.select(self.sa.func.count()).select_from(t).where(*self._scope(t))).scalar())
                for name, t in (("tasks", m.team_task), ("actions", m.team_action), ("reports", m.team_report), ("deliveries", m.team_delivery))}


class PgTeamWork(_TeamWork):
    """The ``team_*`` rows of one workspace: a change is one transaction under the project's advisory lock."""

    def __init__(self, workspace_id: str, url: Optional[str] = None, home: Optional[Path] = None, projects: Any = None):
        from ..storage import db

        self.workspace, self.engine = workspace_id, db.engine(url)
        self.home, self._projects = (Path(home) if home is not None else None), projects

    @property
    def workspace_id(self) -> str:  # memory.scope.workspace_of reads it, as for the other PostgreSQL stores
        return self.workspace

    def location(self) -> str:
        return f"PostgreSQL workspace {self.workspace}"

    @contextlib.contextmanager
    def _tx(self, project_id: str) -> Iterator[_PgTx]:
        import sqlalchemy as sa

        digest = hashlib.sha256(f"team_work:{self.workspace}:{project_id}".encode("utf-8")).digest()
        with self.engine.begin() as c:  # an exception rolls back and releases the lock: nothing is written
            c.execute(sa.text("select pg_advisory_xact_lock(:key)"), {"key": int.from_bytes(digest[:8], "big", signed=True)})
            yield _PgTx(c, self.workspace, project_id)

    @contextlib.contextmanager
    def _read(self, project_id: Optional[str] = None) -> Iterator[_PgTx]:
        with self.engine.connect() as c:
            yield _PgTx(c, self.workspace, project_id)


def _rehomed(value: Any, workspace_id: str) -> Any:
    """A file workspace's document as the database workspace keeps it: its workspace id ``local`` is the database's."""
    if isinstance(value, dict):
        return {k: (workspace_id if k == "workspace_id" and v in (None, "local") else _rehomed(v, workspace_id)) for k, v in value.items()}
    if isinstance(value, list):
        return [_rehomed(v, workspace_id) for v in value]
    return value


def import_project(work: "PgTeamWork", project_id: str, data: Row) -> dict[str, int]:
    """One project's ``work.json`` copied into the database workspace (``storage migrate``), every row in the order it was
    made (its ``seq``), so a task's deliveries and the handoff history keep their order. A row the database already holds
    (the same task, action, report, or delivery key) is left as it is, so a second run copies nothing. Returns how many of
    each kind were copied."""
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import insert

    from ..storage import models as m

    ws = work.workspace
    entries = [("tasks", e) for e in (data.get("tasks") or {}).values()] + [("actions", e) for e in (data.get("actions") or {}).values()]
    entries += [("reports", e) for e in (data.get("reports") or {}).values()]
    entries += [("deliveries", e) for e in (data.get("deliveries") or {}).values()]
    entries.sort(key=lambda kv: int(kv[1].get("seq") or 0))
    copied = {"tasks": 0, "actions": 0, "reports": 0, "deliveries": 0}
    with work.engine.begin() as c:
        for kind, entry in entries:
            if kind == "deliveries":
                row = _rehomed({k: v for k, v in entry.items() if k not in ("seq", "version")}, ws)
                values = dict(id=row["id"], workspace_id=ws, key=row["key"], kind=row["kind"], project_id=project_id,
                              problem_id=row["problem_id"], task_id=row["task_id"], **_PgTx._columns(row))
                q = insert(m.team_delivery).values(**values)
            else:
                doc = _rehomed(entry["doc"], ws)
                if kind == "tasks":
                    task = tw.TeamTask.model_validate(doc)
                    q = insert(m.team_task).values(workspace_id=ws, project_id=project_id, id=task.id, problem_id=task.problem_id,
                                                   parent_task_id=task.parent_task_id, owner_team=task.owner_team,
                                                   requester_team=task.requester.team, status=task.status, version=int(entry.get("version") or 1),
                                                   initiator=_rehomed(entry.get("initiator"), ws), doc=doc)
                elif kind == "actions":
                    action = tw.TeamAction.model_validate(doc)
                    q = insert(m.team_action).values(workspace_id=ws, project_id=project_id, task_id=action.task_id, action_id=action.action_id,
                                                     retry_key=action.retry_key, status=action.status, doc=doc)
                else:
                    report = tw.WorkReport.model_validate(doc)
                    q = insert(m.team_report).values(workspace_id=ws, project_id=project_id, task_id=report.task_id, revision=report.revision,
                                                     parent_task_id=report.parent_task_id, status=report.status, joined=bool(entry.get("joined")),
                                                     verdict=entry.get("verdict"), doc=doc)
            if c.execute(q.on_conflict_do_nothing().returning(sa.literal(1))).first() is not None:
                copied[kind] += 1
    return copied


def open_team_work(home: Path, url: Optional[str] = None, projects: Any = None) -> "FileTeamWork | PgTeamWork":
    """The team work of the workspace in ``home``: PostgreSQL when DCLAB_DATABASE_URL is set (as the other stores choose),
    else the folder's files. On files the workspace is ``local``, as ``memory.scope.workspace_of`` names it. ``projects``:
    the workspace's projects store (opened from ``home`` when it is first needed)."""
    from ..storage import db

    if url or db.database_url(required=False):
        return PgTeamWork(db.workspace_for(Path(home), url), url, home=Path(home), projects=projects)
    return FileTeamWork(Path(home), projects=projects)


__all__ = ["decision_ref", "decisions", "fact_citation", "fact_evidence", "mark_fact_dependents_stale", "record_decision",
           "report_citations", "report_ref", "stale_marks", "task_citations", "task_ref",
           "Conflict", "import_project", "reply_text", "simulated_sources", "DELIVERY_KINDS", "DELIVERY_STATES", "FileTeamWork", "PgTeamWork", "Written", "history_lines", "initiator_of",
           "is_due", "open_team_work", "scrub", "team_label"]
