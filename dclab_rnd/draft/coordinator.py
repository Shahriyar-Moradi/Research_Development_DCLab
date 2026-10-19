"""The coordinator (package 21.2): one lead agent walks a draft from the problem to a built project.

It reads the draft's facts (the problem card, its files, the label join, the meet check, what the roles handed over,
the workflow, the build's own checks), asks ``dclab_core.coordinator.situation`` where it stands, and does the one
thing that state allows: the person answers the open question, the table these cases label is brought from another
draft, the data investigator hands over its reading (checked by ``coordinator.gate`` against the data card: a column
the tables do not hold refuses the hand-off and the rules stand), the workflow is drawn, the project is built. It never
builds while a blocker is open, and says which one stopped it. It runs in the agent harness (21.1): each action a
step of its run.

    Coordinator(drafts, projects, agent, advisor).run(draft_id, person)
"""

from __future__ import annotations

import json
from typing import Any

from dclab_core import coordinator as machine

from ..agents import harness
from . import plan as plans


class Recording:
    """An advisor that keeps every raw answer it passes on (the hand-off before a role's parse drops what it cannot use)."""

    def __init__(self, advisor: Any):
        self.advisor, self.answers = advisor, []

    def ask(self, purpose: str, evidence: Any) -> Any:
        answer = self.advisor.ask(purpose, evidence)
        self.answers.append((purpose, answer))
        return answer


class Coordinator:
    def __init__(self, drafts: Any, projects: Any, agent: Any, advisor: Any = None, max_steps: int = 40, map_advisor: Any = None):
        self.drafts, self.projects, self.agent, self.advisor, self.max_steps = drafts, projects, agent, advisor, max_steps
        self.map_advisor = map_advisor  # 24.2: the decomposer, the map reviewer and the data-need agent, when they serve (None: the rules)

    # ------------------------------------------------------------------ facts and where it stands
    def facts(self, draft_id: str) -> machine.Facts:
        from . import map as problem_map
        from .api import build_blockers
        from .card import card_of, meeting

        draft = self.drafts.get(draft_id)
        card = card_of(draft)
        pending = self.agent.pending(draft)
        assets = draft.get("assets") or []
        ready = [a for a in assets if a.get("status") == "ready"]
        waiting = [a for a in assets if a.get("status") in ("queued", "structuring", "cleaning", "analysing")]
        cases = (draft.get("analysis") or {}).get("cases_table") or {}
        label = plans.label_asset(draft)
        joined = label is not None and bool((label.get("label_review") or {}).get("key"))
        met = meeting(draft, self.drafts) if ready else None
        own = draft.get("coordinator") or {}
        workflow = draft.get("workflow") or {}
        built = bool(draft.get("project_id")) or draft.get("status") == "built"
        problems: list[str] = []
        if workflow.get("nodes"):  # the drawn workflow, checked as a model's proposal is (draft/workflow.py)
            from . import workflow as wflow

            _, problems = wflow.validate({"nodes": [{k: n.get(k) for k in ("wf", "label", "detail")} for n in workflow["nodes"]],
                                          "title": workflow.get("title", "")}, (draft.get("pack") or {}).get("key"))
        elsewhere = [] if cases else self._cases_elsewhere(draft_id)
        return machine.Facts(
            blocking_questions=[{"field": q.field, "text": q.text} for q in card.questions if q.blocks],
            pending_question=(pending or {}).get("field") if pending else None, open_field=None if pending or built else self._open_field(draft),
            files_ready=len(ready), files_pending=len(waiting), files_failed=sum(a.get("status") == "failed" for a in assets),
            cases_only=bool(cases) and label is None, offers=list(cases.get("tables") or []), labels_joined=joined,
            cases_elsewhere=elsewhere,
            meet=[{"code": m.code, "question": m.question, "evidence": m.evidence} for m in (met.mismatches if met else []) if m.blocking],
            meet_checked=met is not None or not ready, analysed=bool(own.get("analysed")),
            workflow_problems=problems, workflow_ready=bool(workflow.get("nodes")), **problem_map.facts(draft),
            build_refusals=[] if built else [{"rule": r.rule, "message": str(r)} for r in build_blockers(self.drafts, draft_id)],
            built=built)

    def _open_field(self, draft: dict[str, Any]) -> str | None:
        """The plan's next field never asked, while questions are left (the rules ask every one before a build; a model
        may end its turns without it)."""
        from .chat import MAX_QUESTIONS

        if len([q for q in draft.get("questions") or [] if not q.get("superseded")]) >= MAX_QUESTIONS:
            return None
        question = self.agent.next_question(draft)
        return question[0] if question else None

    def _cases_elsewhere(self, draft_id: str) -> list[dict[str, Any]]:
        """Other drafts whose file of known cases labels this draft's table (21.3's offers, seen from the table's side)."""
        out = []
        try:
            others = self.drafts.list(limit=200) if hasattr(self.drafts, "list") else []
        except Exception:  # noqa: BLE001 - no listing: nothing offered from this side
            return []
        for other in others:
            oid = other.get("id") if isinstance(other, dict) else None
            if not oid or oid == draft_id:
                continue
            try:
                tables = ((self.drafts.get(oid).get("analysis") or {}).get("cases_table") or {}).get("tables") or []
            except KeyError:
                continue
            out += [{"draft_id": oid, "key": t.get("key"), "coverage": t.get("coverage")} for t in tables if t.get("draft_id") == draft_id]
        return out

    def where(self, draft_id: str) -> machine.Situation:
        return machine.situation(self.facts(draft_id))

    # ------------------------------------------------------------------ the walk
    def run(self, draft_id: str, person: Any) -> dict[str, Any]:
        """Walk the draft until it is built, or a blocker only the person (or more data) can clear stops it."""
        from .. import experience

        steps: list[dict[str, Any]] = []
        project_id = None
        with self.drafts.turn(draft_id), experience.bind(conversation=draft_id), harness.session("coordinator") as current:
            last, seen = None, {}
            now = self.where(draft_id)
            for _ in range(self.max_steps):
                now = self.where(draft_id)
                if now.next in ("done", "stop", "wait_for_data"):
                    break
                with current.timed(now.next, {"state": now.state}) as out:
                    out["result"] = result = self.act(draft_id, now, person)
                steps.append({"state": now.state, "action": now.next, "says": now.says, "result": result})
                if result.get("stopped"):
                    now = now.model_copy(update={"next": "stop", "says": result["stopped"]})
                    break
                done = (now.state, now.next, json.dumps(result, sort_keys=True, default=str))
                seen[done] = seen.get(done, 0) + 1
                if done == last or seen[done] >= 3:  # the same outcome again, or a loop of questions: no progress
                    now = now.model_copy(update={"next": "stop", "says": f"No progress at {now.state}: {now.says}"})
                    break
                last = done
                project_id = result.get("project_id") or project_id
            else:
                now = now.model_copy(update={"next": "stop", "says": "The coordinator's step budget was used."})
            if now.next != "done":
                current.status = "stopped"
            self._remember(draft_id, now)
        return {"state": now.state, "next": now.next, "says": now.says, "blockers": [b.model_dump() for b in now.blockers],
                "passed": now.passed, "steps": steps, "project_id": project_id or self.drafts.get(draft_id).get("project_id")}

    def act(self, draft_id: str, now: machine.Situation, person: Any) -> dict[str, Any]:
        if now.next == "answer":
            return self._answer(draft_id, person)
        if now.next == "draw_map":
            from . import map as problem_map

            shown = problem_map.draw(self.drafts, draft_id, advisor=self.map_advisor)
            return {"map": shown["map"]["size"], "parts": len(shown["map"]["parts"]), "source": shown["source"]}
        if now.next == "confirm_map":
            return self._confirm_map(draft_id, person)
        if now.next == "bring_table":
            return self._bring(draft_id, now)
        if now.next == "analyse":
            return self._analyse(draft_id)
        if now.next == "ask":
            question = self.agent.next_question(self.drafts.get(draft_id))
            if question is None or not self.agent.ask(draft_id, *question):
                return {"stopped": f"The question about {now.blockers[0].field if now.blockers else 'the plan'} could not be asked."}
            from ..ledger import capture

            capture.next_question(self.drafts.get(draft_id), question)
            self.agent.moved(draft_id, "ask_user", {"field": question[0]})
            return {"asked": question[0]}
        if now.next == "plan_workflow":
            self.agent.refresh_workflow(draft_id)
            return {"workflow": bool((self.drafts.get(draft_id).get("workflow") or {}).get("nodes"))}
        if now.next == "build":
            from .api import BuildRefused, build_project

            try:
                project = build_project(self.drafts, self.projects, draft_id)
            except BuildRefused as refused:  # a check the facts did not see yet: said, never built around
                return {"stopped": f"Not built: {refused}", "rule": refused.rule}
            return {"project_id": project["id"]}
        return {"stopped": now.says}

    def _confirm_map(self, draft_id: str, person: Any) -> dict[str, Any]:
        """The person confirms the map of a problem of several parts (or it stays as it is and the walk waits)."""
        from . import map as problem_map

        shown = problem_map.view(self.drafts.get(draft_id), self.map_advisor)
        titles = ", ".join(p["title"] for p in shown["map"]["parts"][:6])
        approve = getattr(person, "approve", None)
        if approve is None or not approve("map", {"parts": [p["title"] for p in shown["map"]["parts"]], "families": [s["family"] for s in shown["map"]["sub_problems"]],
                                              "composition": shown["map"]["composition"]["method"], "questions": len(shown["review"]["questions"])}):
            return {"stopped": f"Waiting for the person to confirm the map of the problem ({titles})."}
        problem_map.edit(self.drafts, draft_id, "confirm", who=getattr(person, "who", "person"), advisor=self.map_advisor)
        return {"confirmed": True, "parts": len(shown["map"]["parts"])}

    def _answer(self, draft_id: str, person: Any) -> dict[str, Any]:
        q = self.agent.pending(self.drafts.get(draft_id))
        if q is None:
            return {"stopped": "No question is open to answer."}
        text = q.get("text") or q.get("question") or ""
        said = person.ask(q.get("field"), text) if hasattr(person, "ask") else person.answer(q.get("field"), text)
        if (said is None or not str(said).strip()) and q.get("source") == "second_opinion":
            self.agent.decline(draft_id, q)  # 28.6: an advisory question: no answer keeps the card, never a stop
            return {"declined": q.get("field"), "source": "second_opinion"}
        if said is None or not str(said).strip():  # the question that stopped the walk is in the person's record too
            return {"stopped": f"Waiting for the person: {text}"}
        self.agent.reply(draft_id, str(said).strip())
        return {"answered": q.get("field")}

    def _bring(self, draft_id: str, now: machine.Situation) -> dict[str, Any]:
        from . import pipeline
        from .api import BringRefused, bring_from_draft

        offer = now.blockers[0].evidence if now.blockers else {}
        try:
            asset = bring_from_draft(self.drafts, draft_id, str(offer.get("draft_id") or ""))
        except BringRefused as refused:
            return {"stopped": f"The table could not be brought: {refused}"}
        final = pipeline.run(self.drafts, draft_id, asset["id"], agent=self.agent)
        if (final or {}).get("status") != "ready":  # never built around a table that could not be read
            return {"stopped": f"The brought table could not be read: {(final or {}).get('error') or 'its pipeline did not finish'}"}
        return {"brought": offer.get("draft_id"), "asset": asset["id"]}

    def _analyse(self, draft_id: str) -> dict[str, Any]:
        """The data investigator's reading of the joined table, handed over and checked against the data card."""
        import pandas as pd
        from dclab_core.agents import run_role
        from dclab_core.analyse import DATA_INVESTIGATOR

        draft = self.drafts.get(draft_id)
        asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset") and a.get("status") == "ready"), None)
        u = draft.get("understanding") or {}
        record: dict[str, Any] = {"role": "data_investigator", "source": "rules"}
        if asset is not None and self.advisor is not None:
            frame = pd.read_parquet(self.drafts.data_dir(draft_id) / asset.get("clean_file", "clean.parquet"))
            target = u.get("target") if u.get("target") in frame.columns else "is_case"
            recording = Recording(self.advisor)
            out = run_role(DATA_INVESTIGATOR, recording, {"features": frame.drop(columns=[target], errors="ignore"), "target": target,
                                                          "task": u.get("task") or "binary", "moment": u.get("prediction_moment")})
            raw = next((a for p, a in recording.answers if p == "data_investigator" and isinstance(a, dict)), None)
            if raw is not None:
                verdict = machine.gate(machine.investigator_handoff(raw), draft.get("data_card") or {}, extra_columns=("is_case",))
                used = out.output if verdict.allowed else DATA_INVESTIGATOR.fallback({})
                record.update(allowed=verdict.allowed, refused_columns=verdict.refused_columns, reasons=verdict.reasons,
                              source="advisor" if verdict.allowed and out.source == "advisor" else "rules",
                              # the columns it proposed, never the model's words (kept only in its model request)
                              output={k: sorted(v) for k, v in used.items()})
                self._decided(draft_id, verdict)
            else:
                record.update(allowed=None, reason=out.reason)

        def keep(d: dict[str, Any]) -> None:
            own = d.setdefault("coordinator", {})
            own["analysed"] = True
            own.setdefault("handoffs", []).append(json.loads(json.dumps(record, default=str)))
        self.drafts.update(draft_id, keep)
        from . import map as problem_map

        problem_map.store_routes(self.drafts, draft_id, self.map_advisor)  # the data card is there now: each part is routed against it
        return {k: record.get(k) for k in ("role", "source", "allowed", "refused_columns")}

    def _decided(self, draft_id: str, verdict: machine.HandoffVerdict) -> None:
        """The gate's verdict on a hand-off, in the session record: a refusal names the columns and the rule."""
        from .. import experience

        if verdict.allowed:
            experience.record("decision.accepted", "system", {"point": "handoff", "choice": verdict.role, "by": "coordinator"}, conversation=draft_id)
        else:
            experience.record("decision.refused", "system", {"point": "handoff", "choice": verdict.role, "rule": "columns in the data card",
                                                            "reason": "; ".join(verdict.reasons)[:600]}, conversation=draft_id)

    def _remember(self, draft_id: str, now: machine.Situation) -> None:
        def keep(d: dict[str, Any]) -> None:
            own = d.setdefault("coordinator", {})
            own["situation"] = json.loads(now.model_dump_json())
        self.drafts.update(draft_id, keep)


__all__ = ["Coordinator", "Recording"]
