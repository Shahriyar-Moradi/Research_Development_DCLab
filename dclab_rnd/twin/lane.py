"""One lane of a twin run: a complete workspace folder of its own, seeded with the same problem as the other lane.

A lane is ``<run>/<name>/`` with everything a workspace writes: ``projects/<id>/`` (project.json, data/, stages/,
activity.jsonl, transitions.jsonl), ``intern/<session>.json`` (the whole conversation: system prompt, task, every
model reply, every tool result, the final report), ``audit.jsonl``, ``agent_steps.jsonl``, ``model_requests.jsonl``,
``ledger/`` (every decision the lane made, with the rule's answer and the agent's, 19.7; on main it replaces the A6.5
decision samples of the branch), ``experience/`` (the lane's session record, 19.2), and ``transcripts/model_calls.jsonl``
(every request and answer, verbatim, when a model ran). Nothing of a lane is shared with the main workspace or the other
lane: the two can run at the same time in two threads, each inside ``context.using(lane.services)`` so that every
``installed()`` call in the engine, the roles and the tools finds its own lane's gateway, ledger, log and audit.

Seeds: a studied sample (``sample:<key>``), a project of another workspace (``project:<id>``, its table and known
solution copied, its stages left behind), or a draft (``draft:<id>``, its problem and the person's turns copied).
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK = ("Build a leakage-safe model for the project's table: understand the problem, describe the data, propose and set the "
        "solution (the target, the prediction moment, the forbidden columns, the identifiers, the split), run the five stages, "
        "approve each one with the option you can defend from its evidence, and report the honest score with the decisions and "
        "their proof.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Services:
    """What ``dclab_rnd.context`` hands to code that asks for the current workspace: the lane's gateway, ledger,
    experience log and audit (and nothing else, so a lane never reaches the main workspace's lessons or limits)."""

    def __init__(self, gateway: Any, workspace_id: str, home: Path | None = None):
        self.gateway, self.workspace_id = gateway, None  # a file workspace: no PostgreSQL id (a twin run keeps files only)
        self.label = workspace_id
        self.lesson_store = self.ledger = self.experience = self.audit = None
        self.projects = self.drafts = None  # set by the lane once its stores exist
        if home is not None:
            from .. import audit, experience, ledger, lessons

            # files, whatever the server's storage: a lane is a folder of its own even when the workspace is on PostgreSQL
            self.ledger = ledger.open_ledger(home)  # no workspace id: the file ledger
            self.experience = experience.open_log(home)  # no workspace id: the file log
            self.audit = audit.FileAudit(Path(home) / "audit.jsonl")
            self.lesson_store = lessons.open_lessons(home)  # the lane's own (empty): never the main workspace's lessons


@dataclass
class Lane:
    name: str  # workflow or ai_first
    home: Path
    gateway: Any
    label: str = ""
    project_id: str | None = None
    draft_id: str | None = None
    session_id: str | None = None
    started: str | None = None
    finished: str | None = None
    status: str = "pending"
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        from ..agents.traces import FileTraces
        from ..draft.store import DraftStore
        from ..intern.sessions import SessionStore
        from ..studio.store import ProjectStore

        self.home = Path(self.home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.projects = ProjectStore(self.home / "projects")
        self.drafts = DraftStore(self.home / "drafts")
        self.projects.audit_path = self.drafts.audit_path = self.home / "audit.jsonl"  # the lane's own audit
        self.sessions = SessionStore(self.home / "intern")
        self.traces = FileTraces(self.home / "agent_steps.jsonl")
        if getattr(self.gateway, "routing", None) is None:  # the lane's own routing document: never the process's workspace's
            from ..models.routing import FileRouting

            try:
                self.gateway.routing = FileRouting(self.home / "model_routing.json")
            except AttributeError:  # a gateway that keeps its own
                pass
        self.services = Services(self.gateway, f"twin:{self.name}", self.home)
        self.services.projects, self.services.drafts = self.projects, self.drafts

    # ------------------------------------------------------------------ seeding

    def seed_sample(self, key: str, goal: str, blind: bool = False) -> str:
        from ..studio import data as sd

        catalog = {s["key"]: s for s in sd.sample_catalog()}
        if key not in catalog:
            raise KeyError(f"unknown sample {key!r}; one of: {', '.join(sorted(catalog))}")
        project = self.projects.create(catalog[key]["name"][:60], catalog[key].get("industry") or "general", goal)
        project = sd.use_sample(self.projects, project["id"], key, actor="human")
        if blind:
            project["suggestion"] = None  # the R&D's known solution stays out: each lane must find the leak itself
            self.projects.save(project)
        self.project_id = project["id"]
        return self.project_id

    def seed_project(self, source_store: Any, source_id: str, blind: bool = False) -> str:
        """A copy of another workspace's project: its table (registered again, so the digest and the record are the
        lane's own), its known solution and labels; never its stages, decisions or approvals."""
        from ..studio import data as sd

        src = source_store.get(source_id)
        if not src.get("data"):
            raise ValueError("the source project has no data")
        project = self.projects.create(src.get("name") or "Twin", src.get("industry") or "general", src.get("goal") or "")
        pid = project["id"]
        filename = src["data"]["filename"]
        shutil.copyfile(sd.data_path(source_store, src), self.projects.data_dir(pid) / filename)
        sd.attach_data(self.projects, pid, filename, source={"kind": "twin", "project_id": source_id}, actor="human")
        project = self.projects.get(pid)
        for key in ("categorical",):
            if src["data"].get(key) is not None:
                project["data"][key] = src["data"][key]
        project["suggestion"] = None if blind else src.get("suggestion")
        project["pack"] = src.get("pack")
        if src.get("labels") and (Path(source_store.data_dir(source_id)) / "labels.parquet").is_file():
            shutil.copyfile(Path(source_store.data_dir(source_id)) / "labels.parquet", self.projects.data_dir(pid) / "labels.parquet")
            project["labels"] = src["labels"]
        project["settings"] = {**project["settings"], **{k: v for k, v in (src.get("settings") or {}).items() if k in ("max_rows", "quick", "critic")}}
        self.projects.save(project)
        self.project_id = pid
        return pid

    def seed_draft(self, source_store: Any, source_id: str) -> tuple[str, list[str]]:
        """A fresh draft with the source's problem and pack; the person's turns are returned, to be replayed."""
        src = source_store.get(source_id)
        draft = self.drafts.create(src.get("problem") or "", src.get("pack"))
        turns = [m.get("text") or "" for m in src.get("messages") or [] if m.get("role") == "user" and (m.get("text") or "").strip()]
        self.draft_id = draft["id"]
        return draft["id"], turns

    # ------------------------------------------------------------------ running

    def run_project(self, task: str, client: Any, max_steps: int, max_minutes: int, designer: Any = None) -> dict[str, Any]:
        """The intern on the lane's project: with ``client`` the model decides every move (the AI-first lane); without,
        the standard plan runs the current system's fixed order (the workflow lane). Every move goes through the graph
        validator either way, and every tool call through the trace. The engine's profile (quick or full) is the
        process's, set once by the run around both lanes: it is one global, never one per lane.

        ``designer`` (A6.7, the AI-first lane): Claude designs the solution and its workflow first, the design goes
        with the task, a monitor on the client writes the timeline and asks for revisions at the checkpoints, and the
        last revision, the adherence and ``design/DESIGN.md`` are written when the run ends, however it ends."""
        from .. import context
        from ..intern import Intern
        from ..intern.tools import Toolbox

        self.started, self.status = now(), "running"
        monitor, session = None, None
        try:
            with context.using(self.services):
                if designer is not None and client is not None:
                    from .design import Monitor

                    designer.initial()
                    follow = designer.follow_text()
                    task = task + ("\n\n" + follow if follow else "")
                    monitor = Monitor(designer)
                    client.observer = monitor
                intern = Intern(self.sessions, Toolbox(self.projects), client, traces=self.traces)
                session = intern.start(task, {"max_steps": max_steps, "max_minutes": max_minutes}, project_id=self.project_id)
                self.session_id = session["id"]
                session = intern.run(session["id"])
            status = session.get("status")
            self.status = "completed" if status == "completed" else "stopped" if status == "budget_exhausted" else str(status)
            self.error = session.get("error") or (str(session.get("final") or "").split(". ", 1)[0][:400] if self.status == "stopped" else None)
        except Exception as exc:  # noqa: BLE001 — the lane records its failure; the other lane goes on
            self.status, self.error = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
        finally:
            if monitor is not None:
                client.observer = None
                try:
                    with context.using(self.services):
                        stored = self.sessions.get(self.session_id) if self.session_id else session
                        monitor.finish(stored, self._outcome(stored))
                except Exception as exc:  # noqa: BLE001 — the design's files are research notes; the lane's result stands
                    self.notes.append(f"design finish failed: {type(exc).__name__}: {str(exc)[:300]}")
        self.finished = now()
        return self.summary()

    def _outcome(self, session: dict[str, Any] | None) -> dict[str, Any]:
        """What the run came to, for the design's retrospective record: the status, the holdout, the solution's fields."""
        out: dict[str, Any] = {"status": (session or {}).get("status") or self.status}
        if self.project_id:
            final = self.projects.records(self.project_id).get("final") or {}
            ev, metric = final.get("evidence") or {}, final.get("primary_metric")
            out.update(metric=metric, holdout=(ev.get("holdout_metrics") or {}).get(metric) if metric else None, holdout_ci=ev.get("holdout_primary_metric_ci"))
            sol = self.projects.get(self.project_id).get("solution") or {}
            out["solution"] = {"target": sol.get("target"), "task": sol.get("task"), "forbidden": sorted(f.get("column") for f in sol.get("forbidden") or []),
                               "identifiers": sorted(sol.get("identifiers") or [])}
        return out

    def run_draft(self, turns: list[str], client: Any) -> dict[str, Any]:
        """The Home agent on the lane's draft, the person's recorded turns replayed one by one: with ``client`` the
        model answers (the AI-first lane); without, the standard questions do (the workflow lane)."""
        from .. import context
        from ..draft.chat import HomeAgent

        self.started, self.status = now(), "running"
        try:
            with context.using(self.services):
                agent = HomeAgent(self.drafts, client, traces=self.traces)
                for text in turns:
                    agent.reply(self.draft_id, text)
            self.status = "completed"
        except Exception as exc:  # noqa: BLE001
            self.status, self.error = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
        self.finished = now()
        return self.summary()

    # ------------------------------------------------------------------ what the lane did

    def _rows(self, name: str) -> list[dict[str, Any]]:
        path = self.home / name
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _glob(self, folder: str) -> list[dict[str, Any]]:
        """Every row of every ``.jsonl`` under a folder of the lane, in the files' order (days, then lines)."""
        out: list[dict[str, Any]] = []
        for path in sorted((self.home / folder).rglob("*.jsonl")) if (self.home / folder).is_dir() else []:
            out.extend(self._rows(str(path.relative_to(self.home))))
        return out

    def decisions(self) -> list[dict[str, Any]]:
        return self._glob("ledger/decisions")

    def summary(self) -> dict[str, Any]:
        """The lane's decisions and numbers, read back from what it wrote (never from memory of the run)."""
        out: dict[str, Any] = {"lane": self.name, "label": self.label, "status": self.status, "error": self.error, "started": self.started,
                               "finished": self.finished, "project_id": self.project_id, "draft_id": self.draft_id, "session_id": self.session_id}
        if self.project_id:
            project = self.projects.get(self.project_id)
            out["data_sha256"] = (project.get("data") or {}).get("sha256")
            out["goal"] = (project.get("goal") or "")[:2000]
            out["given_solution"] = bool(project.get("suggestion"))
            sol = project.get("solution") or {}
            out["solution"] = {"target": sol.get("target"), "task": sol.get("task"), "metric": sol.get("metric"),
                               "prediction_moment": sol.get("prediction_moment"), "forbidden": sorted(f.get("column") for f in sol.get("forbidden") or []),
                               "identifiers": sorted(sol.get("identifiers") or []), "time_column": sol.get("time_column"),
                               "group_column": sol.get("group_column"), "labels": sol.get("labels")} if sol else None
            records = self.projects.records(self.project_id)
            stages = {}
            for stage, rec in records.items():
                ev, dec = rec.get("evidence") or {}, rec.get("decision") or {}
                entry: dict[str, Any] = {"status": project["stages"].get(stage, {}).get("status"), "selected": dec.get("selected"), "chosen": dec.get("chosen"),
                                         "overridden": dec.get("overridden", False), "elapsed_seconds": rec.get("elapsed_seconds")}
                if stage == "data":
                    entry["column_roles"] = ev.get("column_roles")
                    entry["risk_features"] = sorted((ev.get("risk_features") or {}).keys())
                if stage == "leakage":
                    entry["declared"] = sorted(ev.get("declared_leakage_features") or [])
                    entry["flagged"] = sorted(f.get("feature") for f in ev.get("heuristic_review_candidates") or [] if isinstance(f, dict))
                    entry["apparent_lift"] = ev.get("apparent_lift")
                if stage == "final":
                    metric = rec.get("primary_metric")
                    entry["primary_metric"] = metric
                    entry["holdout"] = (ev.get("holdout_metrics") or {}).get(metric)
                    entry["holdout_ci"] = ev.get("holdout_primary_metric_ci")
                    entry["cv_mean"] = ev.get("cv_selected_metric_mean")
                    entry["cv_to_holdout_gap"] = ev.get("cv_to_holdout_gap")
                    entry["holdout_uses"] = ev.get("holdout_uses_in_this_project")
                stages[stage] = entry
            out["stages"] = stages
            out["decisions"] = project.get("decisions") or {}
            out["approvals"] = [{k: a.get(k) for k in ("gate", "by", "at")} for a in project.get("approvals") or []]
            out["memory"] = [{k: n.get(k) for k in ("kind", "decision", "move", "stage")} for n in project.get("memory") or []]
            moves = self.projects.transitions(self.project_id, limit=5000)
            out["moves"] = {"total": len(moves), "allowed": sum(1 for m in moves if m.get("status") == "allowed"), "refused": sum(1 for m in moves if m.get("status") not in (None, "allowed")),
                            "sequence": [f"{m.get('move')}{':' + str(m['args'].get('stage')) if isinstance(m.get('args'), dict) and m['args'].get('stage') else ''}"
                                         f":{'ok' if m.get('status') == 'allowed' else 'refused'}" for m in moves]}
        if self.session_id:
            session = self.sessions.get(self.session_id)
            out["session"] = {"mode": session.get("mode"), "model": session.get("model"), "status": session.get("status"), "steps": len(session.get("steps") or []),
                              "tools": [s.get("tool") for s in session.get("steps") or []], "used": session.get("used"),
                              "final": (session.get("final") or "")[:4000], "error": session.get("error"), "plan": (session.get("plan") or "")[:2000] or None}
        if self.draft_id:
            draft = self.drafts.get(self.draft_id)
            out["draft"] = {"messages": len(draft.get("messages") or []), "agent_turns": sum(1 for m in draft.get("messages") or [] if m.get("role") == "agent"),
                            "questions_asked": [q.get("field") for q in draft.get("questions") or []], "understanding": draft.get("understanding"),
                            "plan": draft.get("plan"), "workflow": draft.get("workflow"), "status": draft.get("status"),
                            "model_note": (draft.get("agent") or {}).get("model_note")}  # set when a turn fell back to the standard questions
        rules: dict[str, Any] = {}
        split_used = None
        for row in self.decisions():  # the rule's latest answer per decision point and subject (19.7): the current system's design
            rule = next((a for a in row.get("answers") or [] if a.get("by") == "rule"), None)
            subject = row.get("subject") or {}
            key = f"{row.get('point')}:{subject.get('key') if subject.get('kind') == 'column' else '-'}"  # a project's own points: '-'
            if rule is not None:
                rules[key] = rule.get("value")
            if row.get("point") == "split.strategy":
                split_used = row.get("value_used")  # the data stage's decision: the split the lane's engine made
        out["rule_answers"] = rules
        out["split_used"] = split_used
        design_summary = self.home / "design" / "summary.json"
        if design_summary.is_file():
            try:
                out["design"] = json.loads(design_summary.read_text(encoding="utf-8"))
            except ValueError:
                out["design"] = {"error": "design/summary.json could not be read"}
        if self.notes:
            out["notes"] = list(self.notes)
        out["counts"] = {"audit": len(self._rows("audit.jsonl")), "agent_steps": len(self._rows("agent_steps.jsonl")),
                         "model_requests": len(self._rows("model_requests.jsonl")), "model_calls_captured": len(self._rows("transcripts/model_calls.jsonl")),
                         "ledger_decisions": len(self.decisions()), "ledger_labels": len(self._glob("ledger/labels")),
                         "experience_events": len(self._glob("experience")),
                         "design_versions": len(self._rows("design/designs.jsonl")), "design_timeline": len(self._rows("design/timeline.jsonl"))}
        if hasattr(self.gateway, "totals"):
            out["model"] = self.gateway.totals()
        else:  # the configured gateway (--workflow-models configured): counted from the rows it wrote
            rows = self._rows("model_requests.jsonl")
            out["model"] = {"requests": len(rows), "eur": round(sum(float(r.get("cost_eur") or 0.0) for r in rows), 6),
                            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows), "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
                            "unpriced": sum(1 for r in rows if r.get("cost_eur") is None), "refused": sum(1 for r in rows if str(r.get("outcome", "")).startswith("refused"))}
        out["model"]["refused_in_transcript"] = sum(1 for r in self._rows("transcripts/model_calls.jsonl") if str(r.get("outcome", "")).startswith("refused"))
        return out


def run_both(lanes: list[tuple[Lane, Any]]) -> list[dict[str, Any]]:
    """Each ``(lane, fn)`` in its own thread, at the same time; the summaries in the lanes' order."""
    results: list[dict[str, Any] | None] = [None] * len(lanes)

    def go(i: int, lane: Lane, fn: Any) -> None:
        try:
            results[i] = fn()
        except Exception as exc:  # noqa: BLE001
            lane.status, lane.error, lane.finished = "failed", f"{type(exc).__name__}: {str(exc)[:400]}", now()
            try:
                results[i] = lane.summary()
            except Exception as inner:  # noqa: BLE001 — the summary itself failed: the status and the error still reach the manifest
                results[i] = {"lane": lane.name, "status": "failed", "error": f"{lane.error}; summary: {type(inner).__name__}"}

    threads = [threading.Thread(target=go, args=(i, lane, fn), name=f"twin-{lane.name}", daemon=True) for i, (lane, fn) in enumerate(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r or {} for r in results]
