"""The solution team (package 28.5): a project's programme solved by a team whose nodes are the studio's own modules.

``situate`` reads the project and alone chooses the next node. The router routes every sub-problem **in parallel** (one
``Send`` a part, each the same ``solutions.route_with`` the programme calls); the designer draws the solution map from those
routes (``programme.refresh_plan``, unchanged but for taking them); the auditor screens the columns the leaves would read on
the fit rows before anything is read (``studio.leak_audit``) and sends a leak suspect back to the person, who decides
whether it is kept out (the move ``forbid_columns``, a person's), at most ``auditor.ROUNDS`` rounds before it stops with
the suspects as a named blocker; the composer runs the programme through the graph validator as an agent
(``programme.run``: the leaves, the composition chosen on validation, the holdout read once) after the person approves
reading the holdout; a part that cannot run is a question to the person (drop it with a reason, or stop); the reporter
closes the walk with the programme's report. The person is reached only through the three pauses (a question, a
confirmation, an approval: ``walk.Asker``), every node advances the board, and the walk's record is kept, as the
understanding walk's (28.3).
"""

from __future__ import annotations

import contextvars
import functools
from contextlib import nullcontext
import json
import operator
import threading
import time
from typing import Annotated, Any, Optional, TypedDict

from ..agents import harness
from . import NothingToResume, thread_id
from .walk import _PAUSES, Asker, _nodes, _requests, _walk_id

_SESSION: contextvars.ContextVar[Any] = contextvars.ContextVar("orchestra_solution_session")
_STEP = threading.Lock()  # parallel routes write their steps one at a time


def _solution_turn(fn):
    @functools.wraps(fn)
    def serialized(self, project_id, *args, **kwargs):
        # A separate lock: graph nodes may run in other threads and take the project's effect lock themselves.
        with self.projects.turn(project_id, purpose="solution", require_process=self.task_execution):
            self.projects.get(project_id)  # authenticate before reading a shared saver or returning a paused question
            return fn(self, project_id, *args, **kwargs)
    return serialized


def _union(a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {**(a or {}), **(b or {})}


class SolveState(TypedDict, total=False):
    project_id: str
    board: dict[str, Any]
    steps: Annotated[list[dict[str, Any]], operator.add]
    routes: Annotated[dict[str, Any], _union]
    card: dict[str, Any]
    sub: str
    routes_for: str      # what each step read, as a fingerprint (the review of 28.5): a change to the tables, the labels or the
    designed_for: str    # columns kept out makes the step old, and the team does it again; a checkpoint never answers for a
    audited_for: str     # project that changed
    approved_for: str
    send_backs: int
    reported: bool
    next: str
    says: str
    back_to: str         # 30.15: the phase a failed G4 sends the work back to (kept in the state: LangGraph drops a key it does not know)
    started: str
    runs: Annotated[list[str], operator.add]  # every session of the walk, appended
    task_checkpoint: str


class SolutionTeam:
    TEAM = "solution"

    def __init__(self, projects: Any, checkpointer: Any, home: Any = None, advisor: Any = None, max_steps: int = 40,
                 shadow: Any = None, shadow_name: str = "", max_eur: Optional[float] = None, task_execution: bool = False,
                 admit_proposals: bool = False, gateway: Any = None, planning_budget: Optional[dict[str, Any]] = None):
        self.projects, self.checkpointer, self.home, self.advisor = projects, checkpointer, home, advisor
        self.shadow, self.shadow_name = shadow, shadow_name  # 28.10: a model that answers the router's question beside it, never used
        self.max_steps = max_steps
        self.max_eur = max_eur  # 28.8: the team's euro budget across its resumes
        self.asker = Asker()
        self.graph = None
        self.task_execution = task_execution  # TP-04 developer opt-in; no product switch or default execution change
        self.admit_proposals = admit_proposals
        self.gateway = gateway
        self.planning_budget = dict(planning_budget) if planning_budget is not None else None

    def _tasks(self):
        from ..tasks.dispatch import TaskDispatcher
        return TaskDispatcher(self.projects, self.checkpointer, enabled=self.task_execution, admit_proposals=self.admit_proposals,
                              gateway=self.gateway, planning_budget=self.planning_budget,
                              budget={"max_tool_calls": min(self.max_steps, 100), "max_seconds": 3600, "max_cost_eur": 0.0})

    def _task_call(self, project_id, task):
        dispatcher = self._tasks()
        result = dispatcher.dispatch(project_id, task)
        if result.get("status") == "blocked" and result.get("reason") == "registry_validation_failed":
            recovery = dispatcher.replan(project_id, task)
            if recovery.get("status") == "ready":
                result = dispatcher.dispatch(project_id, task)
        return result if result["status"] in ("completed", "reconciled", "already_completed") else {
            **result, "blocked": True, "error": result.get("reason", "task_execution_refused")}

    # ------------------------------------------------------------------------------------------------ the board
    def _advance(self, state: SolveState, node: str, what: str, board: Any = None, **changes: Any) -> dict[str, Any]:
        from dclab_core.orchestra import Board

        from . import board as B

        b = board if board is not None else Board.model_validate(state["board"])
        return B.advance(b, node, what, self.TEAM, home=self.home, walk=_walk_id(state), **changes).model_dump(mode="json")

    def _close_pauses(self, state: SolveState, node: str) -> Any:
        from dclab_core.orchestra import Board

        from . import board as B

        board = Board.model_validate(state["board"])
        for p in _PAUSES.get(None) or []:
            board = B.interrupt(board, p["kind"], node, self.TEAM, field=p.get("field"), gate=p.get("gate"), text=str(p.get("text") or ""),
                                status=p["status"], home=self.home)
        return board

    def _parts(self, project: dict[str, Any]) -> list[dict[str, Any]]:
        from ..studio import programme

        plan = programme.plan_of(project) or {}
        routes = plan.get("routes") or {}
        stored = programme.stored(project)
        status = {l.block: l.status for l in stored.leaves} if stored is not None else {}
        blocks = {b.get("sub_problem"): b.get("part") for b in (plan.get("solution_map") or {}).get("blocks") or []}
        shown = {"built": "built", "blocked": "blocked", "dropped": "blocked", "planned": "designed"}
        out = []
        for s in (plan.get("problem_map") or {}).get("sub_problems") or []:
            route = (routes.get(s["id"]) or {}).get("chosen")
            leaf = status.get(blocks.get(s["id"])) if blocks.get(s["id"]) else None
            out.append({"part": str(s["id"])[:80], "title": str(s.get("title") or "")[:200], "family": s.get("family"), "route": str(route)[:80] if route else None,
                        "status": shown.get(str(leaf), "routed" if route else "open")})
        return out

    def _step(self, node: str, arguments: dict[str, Any], result: Any, seconds: float) -> dict[str, Any]:
        with _STEP:
            _SESSION.get().step(node, arguments, result, seconds)
        return {"action": node, "seconds": round(seconds, 3), "result": result}

    # ------------------------------------------------------------------------------------------------ the nodes
    def situate(self, state: SolveState) -> dict[str, Any]:
        """The next node, from the project as it is now and what each step last read; every choice a decision (29.6)."""
        out = self._situate(state)
        from ..ledger import capture

        project = self.projects.get(state["project_id"])
        capture.team_node(("project", state["project_id"]), "solution", "solution", out.get("next", "stop"), out.get("next", "stop"),
                          blockers=len(((project.get("programme_plan") or {}).get("solution_map") or {}).get("blockers") or []),
                          step=len(state.get("steps") or []))
        return out

    def _situate(self, state: SolveState) -> dict[str, Any]:
        from ..studio import programme

        project = self.projects.get(state["project_id"])
        if self.task_execution and programme.plan_of(project) is not None:
            return {"next": "stop", "says": "Task execution currently supports one binary workflow; the programme stays with its atomic owner."}
        if programme.plan_of(project) is None:
            return self._situate_single(state, project)  # 30.1: a project of one problem, through the engine's stages
        if len(state.get("steps") or []) >= self.max_steps:
            return {"next": "stop", "says": "The team's step budget was used."}
        if self.max_eur is not None:  # 28.8
            spent = _requests(state.get("runs") or [], project=state["project_id"])[0]
            if spent >= self.max_eur:
                return {"next": "stop", "says": f"The team's euro budget was used: €{spent:.2f} of €{self.max_eur:.2f}."}
        p = programme.stored(project)
        if p is not None and p.holdout is not None and not programme.stale_reasons(project):
            return {"next": "done" if state.get("reported") else "report"}
        routing, audited = programme.routing_inputs(project), programme.audit_inputs(project)
        if state.get("routes_for") != routing:
            return {"next": "route", "routes_for": routing, "card": json.loads(programme.plan_card(self.projects, project).model_dump_json())}
        if state.get("designed_for") != audited:
            return {"next": "design"}
        if state.get("audited_for") != audited:
            return {"next": "audit"}
        current = p is not None and not programme.stale_reasons(project)
        if current and state.get("approved_for") == audited and any(l.status == "blocked" for l in p.leaves):
            return {"next": "drop_gate"}
        return {"next": "compose"}

    # ------------------------------------------------------------------------------------------------ 30.1: one problem
    STAGES = ("data", "leakage", "features", "models", "final")

    def _situate_single(self, state: SolveState, project: dict[str, Any]) -> dict[str, Any]:
        """The engine's stages in order, the auditor after the leakage stage and before anything is chosen, the holdout read
        by the final stage only after the person approves it."""
        from ..studio import graph as studio_graph, leak_audit

        if not project.get("solution"):
            return {"next": "stop", "says": "Save the solution first: the team runs a project of one problem from its saved solution (the person's)."}
        if len(state.get("steps") or []) >= self.max_steps:
            return {"next": "stop", "says": "The team's step budget was used."}
        if self.max_eur is not None:  # 28.8
            spent = _requests(state.get("runs") or [], project=state["project_id"])[0]
            if spent >= self.max_eur:
                return {"next": "stop", "says": f"The team's euro budget was used: €{spent:.2f} of €{self.max_eur:.2f}."}
        status = {k: ((project.get("stages") or {}).get(k) or {}).get("status") for k in self.STAGES}
        done = ("completed", "approved")
        nxt = next((k for k in self.STAGES if status[k] not in done), None)

        if self.task_execution:
            tasks = self._tasks()
            problem = tasks.preflight(project["id"])
            if problem:
                return {"next": "stop", "says": "Task execution stopped: " + problem}
            if nxt is not None and tasks.state(project["id"]) is None:
                verdict = studio_graph.check(project, "run_stage", "agent", stage=nxt)
                if not verdict.allowed:
                    return {"next": "stop", "says": f"The {nxt} stage did not run: {verdict.message}"}
            if self.admit_proposals:
                tasks.prepare_model_graph(project["id"])
                problem = tasks.preflight(project["id"])
                if problem:
                    return {"next": "stop", "says": "Task execution stopped: " + problem}

        if nxt is None:
            if not state.get("reported"):
                return {"next": "report"}
            gate = (((project.get("stages") or {}).get("final") or {}).get("evidence") or {}).get("gate_g4") \
                or ((self.projects.read_stage(state["project_id"], "final") or {}).get("evidence") or {}).get("gate_g4")
            if gate and not gate.get("passed"):  # 30.15: a failed G4 names the phase the work goes back to; the team follows it, a person decides there
                return {"next": "stop", "says": (f"G4 sends the work back to the {gate.get('back_to')}: " + "; ".join(gate.get("reasons") or [])[:3 * 200])[:600],
                        "back_to": gate.get("back_to")}
            return {"next": "done"}
        if status["leakage"] in done and state.get("audited_for") != leak_audit.single_inputs(project):
            return {"next": "audit"}
        return {"next": "compose" if nxt == "final" else "stage"}

    def stage(self, state: SolveState) -> dict[str, Any]:
        """30.1: the next stage of the engine, run through the registry's tool and the validator, as an agent."""
        from ..intern.tools import Toolbox

        project = self.projects.get(state["project_id"])
        status = {k: ((project.get("stages") or {}).get(k) or {}).get("status") for k in self.STAGES}
        key = next(k for k in self.STAGES if status[k] not in ("completed", "approved"))
        clock = time.monotonic()
        out = self._task_call(state["project_id"], key) if self.task_execution else Toolbox(self.projects).call("run_stage", {"project_id": state["project_id"], "stage": key})  # its guard the validator
        failed = isinstance(out, dict) and (out.get("error") or out.get("blocked"))
        step = self._step("stage", {"stage": key}, {"refused": str(failed)[:300]} if failed else {"stage": key, "status": (out or {}).get("status") if isinstance(out, dict) else None},
                          time.monotonic() - clock)
        board = self._advance(state, "stage", key if not failed else "refused")
        if failed:
            return {"steps": [step], "board": board, "next": "stop", "says": f"The {key} stage did not run: {failed}"[:600]}
        return {"steps": [step], "board": board, "next": "situate", **({"task_checkpoint": out["checkpoint"]} if out.get("checkpoint") else {})}

    def _fan(self, state: SolveState) -> Any:
        """The router's fan-out: one ``Send`` a sub-problem, each routed on its own, all in the same step."""
        from langgraph.graph import END
        from langgraph.types import Send

        from ..studio import programme

        nxt = state.get("next")
        if nxt != "route":
            return nxt if nxt in ("design", "audit", "audit_gate", "compose", "drop_gate", "report", "stage") else END
        m, _ = programme.maps(self.projects.get(state["project_id"]))
        return [Send("route_part", {"project_id": state["project_id"], "sub": s.id, "card": state["card"]}) for s in m.sub_problems]

    def route_part(self, state: SolveState) -> dict[str, Any]:
        """One sub-problem routed by the router (the rules, or its model role when it serves)."""
        from dclab_core import solutions
        from dclab_core.datacard import DataCard

        from ..studio import programme

        from ..ledger import capture

        clock = time.monotonic()
        project = self.projects.get(state["project_id"])
        m, _ = programme.maps(project)
        sub = next(s for s in m.sub_problems if s.id == state["sub"])
        card = DataCard.read(state["card"])
        route, source = solutions.route_with(sub, card, m, None, self.advisor)
        shadow = None
        if self.shadow is not None:  # 28.10: the shadow's own route, from its model alone (no rules fallback is its answer)
            try:
                guess, how = solutions.route_with(sub, card, m, None, self.shadow)
                shadow = guess if how == "advisor" or getattr(guess, "source", None) == "advisor" else None
            except Exception:  # noqa: BLE001 - a shadow that fails is no answer, never the walk's failure
                shadow = None
        owner = (project.get("draft") or {}).get("id") or state["project_id"]
        rec = capture.route_chosen(owner, sub, route, card, len(m.parts), shadow=shadow, shadow_by=self.shadow_name)  # every route the team makes, a decision
        step = self._step("route", {"part": sub.id}, {"chosen": route.chosen, "source": route.source, "candidates": [c.approach for c in route.candidates][:16],
                                                     "shadow": getattr(shadow, "chosen", None) if self.shadow is not None else None,
                                                     "record": getattr(rec, "id", None)}, time.monotonic() - clock)
        return {"routes": {sub.id: json.loads(route.model_dump_json())}, "steps": [step]}

    def design(self, state: SolveState) -> dict[str, Any]:
        """The solution map drawn from the team's routes (the designer's checks are the solution map's own)."""
        from dclab_core import solutions

        from ..studio import programme

        clock = time.monotonic()
        routes = {k: solutions.Route.model_validate(v) for k, v in (state.get("routes") or {}).items()}
        changed = programme.refresh_plan(self.projects, state["project_id"], self.advisor, routes=routes, record=False)  # the routes were recorded as they were made
        project = self.projects.get(state["project_id"])
        sm = (programme.plan_of(project) or {}).get("solution_map") or {}
        blockers = [str(b)[:80] for b in sm.get("blockers") or []]
        step = self._step("design", {"routes": len(routes)}, {"changed": changed, "blockers": blockers, "fingerprint": sm.get("fingerprint"),
                                                              "routes": {k: r.chosen for k, r in routes.items()}}, time.monotonic() - clock)
        board = self._advance(state, "design", "solution_map", parts=self._parts(project), blockers=blockers[:40])
        return {"designed_for": programme.audit_inputs(project), "steps": [step], "board": board, "next": "situate"}

    def audit(self, state: SolveState) -> dict[str, Any]:
        """The auditor's screen before the holdout is read: pass, send a suspect back to the person, or stop."""
        from dclab_core.orchestra import Board

        from ..studio import leak_audit, programme

        clock = time.monotonic()
        rnd = int(state.get("send_backs") or 0) + 1  # the rounds are the send-backs: a restart is not one
        single = programme.plan_of(self.projects.get(state["project_id"])) is None
        try:
            v = (leak_audit.audit_single if single else leak_audit.audit)(self.projects, state["project_id"], rnd)
        except (leak_audit.AuditError, programme.ProgrammeError) as exc:  # a table too short for the split, no data: said, never a crash
            step = self._step("audit", {"round": rnd}, {"refused": str(exc)[:300]}, time.monotonic() - clock)
            return {"steps": [step], "next": "stop", "says": f"The auditor could not screen the columns: {exc}"[:600],
                    "board": self._advance(state, "audit", "refused")}
        step = self._step("audit", {"round": rnd}, json.loads(v.model_dump_json()), time.monotonic() - clock)  # the whole verdict, kept
        plan_blockers = [b for b in Board.model_validate(state["board"]).blockers if not b.startswith("leak suspect: ")]
        board = self._advance(state, "audit", v.verdict, blockers=(plan_blockers + [f"leak suspect: {c}"[:80] for c in leak_audit.suspects_of(v)])[:40])
        out: dict[str, Any] = {"steps": [step], "board": board}
        if v.verdict in ("pass", "not_screened"):
            return {**out, "audited_for": v.inputs, "next": "situate"}
        if v.verdict == "stop":
            from dclab_core.auditor import as_lines

            return {**out, "next": "stop", "says": ("The auditor stops the run before the holdout: " + "; ".join(as_lines(v)[:3]))[:600]}
        return {**out, "next": "audit_gate"}

    def audit_gate(self, state: SolveState) -> dict[str, Any]:
        """The auditor's suspects put to the person: keep them out (the plan is drawn again without them and screened again: a
        send-back), or keep them (their decision, recorded; the walk goes on)."""
        from dclab_core.auditor import AuditVerdict, as_lines

        from ..studio import leak_audit

        project = self.projects.get(state["project_id"])
        v = AuditVerdict.model_validate(((project.get("auditor") or {}).get("last") or {})["verdict"])
        token = _PAUSES.set([])
        try:
            keep_out = self.asker.approve("audit", {"suspects": as_lines(v)[:20], "round": v.round})
            board = self._close_pauses(state, "audit")
        finally:
            _PAUSES.reset(token)
        clock = time.monotonic()
        columns = leak_audit.suspects_of(v)
        if keep_out:
            from ..studio import programme as _programme

            single = _programme.plan_of(project) is None
            try:
                if single:  # 30.1: the solution's own forbidden list, through set_solution (the stages run again)
                    leak_audit.forbid_in_solution(self.projects, state["project_id"], columns, f"the auditor's screen, round {v.round}: kept out by the person")
                else:
                    leak_audit.forbid(self.projects, state["project_id"], columns, f"the auditor's screen, round {v.round}: kept out by the person", actor="human", verdict=v)
            except leak_audit.AuditError as exc:
                return {"next": "stop", "says": f"The columns could not be kept out: {exc}"[:600], "board": self._advance(state, "audit", "refused", board)}
        leak_audit.decided(self.projects, state["project_id"], v, keep_out)
        step = self._step("audit_gate", {"round": v.round}, {"kept_out": columns if keep_out else [], "kept": [] if keep_out else columns}, time.monotonic() - clock)
        board = self._advance(state, "audit", "send_back" if keep_out else "kept_by_person", board)
        if keep_out:  # sent back: the columns kept out change what the plan reads, so the designer and the auditor go again
            return {"send_backs": int(state.get("send_backs") or 0) + 1, "steps": [step], "board": board, "next": "situate"}
        return {"audited_for": v.inputs, "steps": [step], "board": board, "next": "situate"}

    def compose(self, state: SolveState) -> dict[str, Any]:
        """The programme run through the registry's tool and the graph validator, as an agent, once the person approves reading
        the holdout on these inputs (recorded as the project's approval of the gate)."""
        from ..intern.tools import Toolbox
        from ..studio import graph as studio_graph, programme

        project = self.projects.get(state["project_id"])
        if programme.plan_of(project) is None:
            return self._compose_single(state, project)
        audited = programme.audit_inputs(project)
        p = programme.stored(project)
        if p is not None and p.holdout is not None and not programme.stale_reasons(project):
            return {"next": "situate"}  # the run saved its result before a crash: it is not run again
        token = _PAUSES.set([])
        try:
            approved = state.get("approved_for") == audited or self.asker.approve("holdout", {"parts": [x["title"] or x["part"] for x in self._parts(project)][:12]})
            board = self._close_pauses(state, "compose")
        finally:
            _PAUSES.reset(token)
        if not approved:
            return {"next": "stop", "says": "Waiting for the person's approval to read the holdout.", "board": self._advance(state, "compose", "declined", board)}
        clock = time.monotonic()
        if state.get("approved_for") != audited:
            studio_graph.approve_gate(self.projects, state["project_id"], "holdout", by="person", reason="approved in the solution team, after its audit")
        out = Toolbox(self.projects).call("run_programme", {"project_id": state["project_id"], "keep_plan": True})  # the registry's tool, its guard the validator
        if isinstance(out, dict) and out.get("error"):
            step = self._step("compose", {}, {"refused": str(out.get("error"))[:300]}, time.monotonic() - clock)
            return {"next": "stop", "says": f"The programme did not run: {out.get('error')}"[:600], "steps": [step], "approved_for": audited,
                    "board": self._advance(state, "compose", "refused", board)}
        step = self._step("compose", {}, {"state": out.get("state"), "holdout_reads": out.get("holdout_reads"), "blockers": out.get("blockers")}, time.monotonic() - clock)
        project = self.projects.get(state["project_id"])
        board = self._advance(state, "compose", str(out.get("state"))[:80], board, parts=self._parts(project))
        p = programme.stored(project)
        if p is not None and p.holdout is None and not any(l.status == "blocked" for l in p.leaves):
            return {"approved_for": audited, "steps": [step], "board": board, "next": "stop",
                    "says": ("The programme did not read the holdout: " + "; ".join(out.get("blockers") or out.get("issues") or [str(out.get("state"))]))[:600]}
        return {"approved_for": audited, "steps": [step], "board": board, "next": "situate"}

    def _compose_single(self, state: SolveState, project: dict[str, Any]) -> dict[str, Any]:
        """30.1: the final stage (tuning on the training folds, then the holdout once) after the person approves reading it,
        the approval recorded as the project's gate; run through the registry's tool as an agent."""
        from ..intern.tools import Toolbox
        from ..studio import graph as studio_graph, leak_audit

        if self.task_execution:
            progress = self._tasks().state(state["project_id"])
            if progress and any(a["task"] == "final" for a in progress["attempts"]):
                out = self._task_call(state["project_id"], "final")
                if out.get("blocked"):
                    return {"next": "stop", "says": "Final task stopped: " + out["error"]}
                return {"next": "situate", "board": self._advance(state, "compose", "reconciled"),
                        **({"task_checkpoint": out["checkpoint"]} if out.get("checkpoint") else {})}

        fp = leak_audit.single_inputs(project)
        token = _PAUSES.set([])
        try:
            approved = state.get("approved_for") == fp or self.asker.approve("holdout", {"stages": [k for k in self.STAGES if k != "final"],
                                                                                         "target": (project.get("solution") or {}).get("target")})
            board = self._close_pauses(state, "compose")
        finally:
            _PAUSES.reset(token)
        if not approved:
            return {"next": "stop", "says": "Waiting for the person's approval to read the holdout.", "board": self._advance(state, "compose", "declined", board)}
        clock = time.monotonic()
        if state.get("approved_for") != fp:
            studio_graph.approve_gate(self.projects, state["project_id"], "holdout", by="person", reason="approved in the solution team, after its audit")
        out = self._task_call(state["project_id"], "final") if self.task_execution else Toolbox(self.projects).call("run_stage", {"project_id": state["project_id"], "stage": "final"})
        failed = isinstance(out, dict) and (out.get("error") or out.get("blocked"))
        step = self._step("compose", {"stage": "final"}, {"refused": str(failed)[:300]} if failed else {"final": "done"}, time.monotonic() - clock)
        if failed:
            return {"next": "stop", "says": f"The final stage did not run: {failed}"[:600], "steps": [step], "approved_for": fp,
                    "board": self._advance(state, "compose", "refused", board)}
        return {"approved_for": fp, "steps": [step], "board": self._advance(state, "compose", "final", board), "next": "situate"}

    def drop_gate(self, state: SolveState) -> dict[str, Any]:
        """A part that cannot run, put to the person: drop it with a reason (their move ``drop_leaf``), or stop."""
        from ..studio import programme

        project = self.projects.get(state["project_id"])
        p = programme.stored(project)
        blocked = [l for l in p.leaves if l.status == "blocked"] if p is not None else []
        token = _PAUSES.set([])
        try:
            reasons = {}
            for leaf in blocked:
                said = self.asker.ask(f"drop:{leaf.block}", f"The part {leaf.block} ({leaf.title[:120]}) cannot run: {leaf.why[:300]}. "
                                                            "To go on without it, say why; to stop here, say nothing.")
                if not said:
                    break
                reasons[leaf.block] = said
            board = self._close_pauses(state, "drop")
        finally:
            _PAUSES.reset(token)
        if len(reasons) < len(blocked):
            return {"next": "stop", "says": "A part cannot run and the person stops here: the holdout stays closed.", "board": self._advance(state, "drop", "stopped", board)}
        clock = time.monotonic()
        for block, reason in reasons.items():
            programme.drop(self.projects, state["project_id"], block, reason, actor="human")
        step = self._step("drop_gate", {"parts": list(reasons)}, {"dropped": list(reasons)}, time.monotonic() - clock)
        return {"steps": [step], "board": self._advance(state, "drop", "dropped", board), "next": "compose"}

    REPORT_SAID = 20000  # characters of the report the walk's answer carries; the whole one is GET /api/projects/{id}/export/solution-report

    def report(self, state: SolveState) -> dict[str, Any]:
        """The report from the records, every number traced to them: a programme's (``dclab_core.programme.report``), or for a
        project of one problem the solution report (``studio.solution_report.latest``, built once per set of records). The
        walk's answer carries it, so a team walked on a finished project reports what it found."""
        from dclab_core import programme as core_programme

        from ..studio import programme, solution_report

        clock = time.monotonic()
        project = self.projects.get(state["project_id"])
        if self.task_execution:
            out = self._task_call(state["project_id"], "report")
            if out.get("blocked"):
                return {"next": "stop", "says": "Report task stopped: " + out["error"]}
        from ..models.gateway import private_scope

        p = programme.stored(project)
        with private_scope(True) if self.task_execution else nullcontext():
            text = core_programme.markdown(p) if p is not None else solution_report.latest(self.projects, state["project_id"])["markdown"]
        step = self._step("report", {}, {"characters": len(text), "sections": len(core_programme.sections(p)) if p is not None else text.count("\n## ")},
                          time.monotonic() - clock)
        said = text if len(text) <= self.REPORT_SAID else text[:self.REPORT_SAID] + f"\n\n… (the whole report: GET /api/projects/{state['project_id']}/export/solution-report)"
        return {"reported": True, "steps": [step], "says": said, "board": self._advance(state, "report", "reported", parts=self._parts(project)), "next": "situate"}

    @staticmethod
    def _after(state: SolveState) -> str:
        nxt = state.get("next")
        return nxt if nxt in ("situate", "audit_gate", "compose") else "end"

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(SolveState)
        g.add_node("situate", self.situate)
        for name in ("route_part", "design", "audit", "audit_gate", "compose", "drop_gate", "report", "stage"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "situate")
        g.add_conditional_edges("situate", self._fan, ["route_part", "design", "audit", "audit_gate", "compose", "drop_gate", "report", "stage", END])
        g.add_edge("route_part", "design")  # every part routed, then the designer: the join of the fan-out
        for name in ("design", "audit", "audit_gate", "compose", "drop_gate", "report", "stage"):
            g.add_conditional_edges(name, self._after, {"situate": "situate", "audit_gate": "audit_gate", "compose": "compose", "end": END})
        self.graph = g.compile(checkpointer=self.checkpointer)
        return self.graph

    # ------------------------------------------------------------------------------------------------ the walk
    @_solution_turn
    def run(self, project_id: str, resume: Any = None, store: Any = None) -> dict[str, Any]:
        """Start the team on a project, or resume it with the person's answer: ``{"status": "paused", "asks": …}`` at a pause,
        else ``{"status": "done", "next": "done" | "stop", "says", "steps", "board"}``."""
        from langgraph.types import Command

        from .. import experience
        from . import board as B

        if self.graph is None:
            self.build()
        thread = thread_id("project", project_id)
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 6 * self.max_steps + 20}
        snap = self.graph.get_state(config)
        from ..tasks.dispatch import TaskDispatcher
        if not self.task_execution and self.checkpointer.get(TaskDispatcher.config(project_id)) is not None:
            return {"status": "done", "next": "stop", "says": "Existing task attempts were retained. Resume with task_execution=True."}
        open_ = [i for t in snap.tasks for i in (getattr(t, "interrupts", None) or ())]
        crash_resume = bool(self.task_execution and snap.values and snap.next and not open_ and resume is None)
        if resume is not None and not open_:
            raise NothingToResume(f"nothing to resume on {thread}: no pause is open")
        if resume is None and open_:  # started again while a pause waits for the person: the pause stands, nothing is redone
            return {"status": "paused", "asks": dict(open_[0].value)}
        if resume is None and snap.values and not self.task_execution:  # never delete task attempt namespaces on resume
            self.checkpointer.delete_thread(thread)
        if resume is not None and not (isinstance(resume, dict) and "key" in resume):
            resume = {"key": dict(open_[0].value).get("key"), "answer": resume}
        with experience.bind(project=project_id), harness.session("orchestra.solution", store=store, params={"project": project_id, "resumed": resume is not None}) as current:
            token = _SESSION.set(current)
            try:
                from ..models.gateway import private_scope
                with private_scope(True) if (self.task_execution and not self.admit_proposals) else nullcontext():
                    if crash_resume:
                        self.graph.invoke(None, config)
                    elif resume is None:
                        self.graph.invoke({"project_id": project_id, "board": B.new(thread, state="solution").model_dump(mode="json"), "steps": [],
                                           "started": B.now(), "runs": [current.run_id]}, config)
                    else:
                        self.graph.invoke(Command(resume=resume, update={"runs": [current.run_id]}), config)
            finally:
                _SESSION.reset(token)
            snap = self.graph.get_state(config)
            pending = [i for task in snap.tasks for i in (getattr(task, "interrupts", None) or ())]
            values = snap.values
            if pending:
                from dclab_core.orchestra import Board

                asks = dict(pending[0].value)
                current.step("interrupt", {"kind": asks.get("kind"), "field": asks.get("field"), "gate": asks.get("gate")}, "paused")
                current.status = "stopped"
                B.interrupt(Board.model_validate(values["board"]), asks.get("kind") or "question", "team", self.TEAM, field=asks.get("field"),
                            gate=asks.get("gate"), text=str(asks.get("text") or ""), status="open", home=self.home)
                return {"status": "paused", "asks": asks}
            nxt = values.get("next")
            if nxt != "done":
                current.status = "stopped"
            from dclab_core.orchestra import Board

            spent, models = _requests(values.get("runs") or [], project=project_id)
            B.record(thread, self.TEAM, _nodes(values.get("steps") or []), Board.model_validate(values["board"]), "completed" if nxt == "done" else "stopped",
                     values.get("started") or B.now(), home=self.home, spent_eur=spent, walk=_walk_id(values), models=models)
        return {"status": "done", "next": nxt, "says": values.get("says") or "", "steps": values.get("steps") or [], "board": values.get("board"),
                **({"back_to": values["back_to"]} if values.get("back_to") else {})}


def drive(team: SolutionTeam, project_id: str, person: Any, max_pauses: int = 40) -> dict[str, Any]:
    """Walk the team with a person who answers its pauses (``ask``/``answer`` and ``approve``), as ``walk.drive`` does."""
    out = team.run(project_id)
    for _ in range(max_pauses):
        if out.get("status") != "paused":
            return out
        asks = out["asks"]
        if asks.get("kind") == "question":
            ask = getattr(person, "ask", None) or getattr(person, "answer")
            said = ask(asks.get("field"), asks.get("text") or "")
            out = team.run(project_id, resume=said if said is not None else "")
        else:
            approve = getattr(person, "approve", None)
            out = team.run(project_id, resume=bool(approve(asks.get("gate") or "", asks.get("summary") or {})) if approve else False)
    return out


__all__ = ["SolutionTeam", "SolveState", "drive"]
