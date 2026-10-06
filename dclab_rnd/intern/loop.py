"""The budgeted tool loop, and the standard plan it follows when no model is configured.

A session has a budget of tool calls and minutes, enforced here on every step. In LLM
mode the model plans, calls tools and writes the report; in standard mode the same
tools run in the fixed DCLab order (sample → solution → five stages → report), so the
transcript looks the same and the product works offline.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from dclab_rnd.agents.registry import Registry, Tool
from dclab_rnd.agents.runtime import Policy, Step, run as run_policy
from dclab_rnd.agents.traces import Tracer
from dclab_rnd.studio import graph as studio_graph

from .sessions import SessionStore, now
from .tools import LEGACY_TOOLS, RESULT_CHARS, Toolbox, summarize, truncate

SESSION_TOOLS = ("write_plan", "finish")  # the intern's own notes on its session (scope "session")
SESSION_MOVES = {"write_plan": "plan", "finish": "finish"}

POLICY = """You are the DCLab intern: a careful ML engineer who builds models with evidence, not opinions.
You work only through the tools. Deterministic code owns splits, metrics and selection rules; you plan, choose, explain and cite.
Rules you never break:
1. Write the solution before any score: what is predicted, at which moment, which columns are unknown then. Use propose_solution and forbid every column that would be known only after the outcome; review_leakage checks the columns against the moment (forbid its forbid list; weigh its consider list and say why you forbid or keep each). Identifiers are never features.
2. Run the stages in order and read each result before the next. The final stage consumes the holdout once; do not rerun it to chase a score.
3. Report numbers with their uncertainty (fold std, the 95% interval) and name the rule or precedent record IDs behind each claim (search_evidence / get_record).
4. Never claim production readiness, causality or fairness from benchmark evidence.
5. Stay inside the budget: prefer quick mode first; use run_all when the solution is settled.
6. Every project move passes the workflow graph's validator. If a tool returns a blocked or needs_approval verdict, do not work around it: explain it, and ask the owner when a person must decide. get_graph shows where the project is and which moves are allowed.
Start by writing a short plan with write_plan. When the work is done, call finish with a report for the person: what was built, the honest score with its interval, the leakage findings, the decisions and their proof, what to do next. Keep every message concise."""

SAMPLE_HINTS = [
    (r"churn|telco|subscri", "telco_churn"), (r"fraud|credit.?card|transaction", "credit_card_fraud"), (r"bike|rental|demand|forecast|time.?series", "bike_sharing_daily"),
    (r"letter|26 class|multi.?class|handwrit", "letter_recognition"), (r"review|text|clothing|sentiment", "ecommerce_clothing_reviews"),
    (r"hyperack|delivery|dispatch|courier", "hyperack"), (r"bank|deposit|marketing", "bank_marketing"), (r"spam|email", "spambase"),
    (r"heart|cardio", "heart_disease"), (r"cancer|tumou?r", "breast_cancer"), (r"wine", "wine_quality"), (r"mushroom", "mushroom"),
    (r"shopper|purchase|session", "online_shoppers"), (r"german|loan|credit risk", "german_credit"), (r"default|billing", "credit_default"), (r"income|census|adult", "adult"),
]


class Intern:
    def __init__(self, sessions: SessionStore, toolbox: Toolbox, client: Any | None = None, traces: Any = None):
        self.sessions, self.toolbox, self.client, self.traces = sessions, toolbox, client, traces
        self.tracer: Tracer | None = None

    # ------------------------------------------------------------------ public
    def start(self, task: str, budget: dict[str, Any] | None = None, project_id: str | None = None, model: str | None = None) -> dict[str, Any]:
        mode = "llm" if self.client is not None else "standard"
        session = self.sessions.create(task, mode=mode, model=(model or getattr(self.client, "model", None)) if self.client else None, budget=budget, project_id=project_id)
        session["messages"] = [{"role": "system", "content": POLICY}, {"role": "user", "content": self._task_message(session)}]
        return self.sessions.save(session)

    def _trace(self, session: dict[str, Any]) -> Tracer:
        """The session's trace (package A2.3), continued across follow-ups; its state is the project's graph state."""
        def state() -> str:
            pid = session.get("project_id")
            return studio_graph.state_string(self.toolbox.projects.get(pid)) if pid else "no project"
        return Tracer.resume(self.traces, session["id"], "intern", state, floor=len(session.get("steps") or []))

    def run(self, session_id: str) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        self.tracer = self._trace(session)
        session["status"] = "running"
        session["error"] = None
        session["used"]["minutes_before"] = session["used"]["minutes"]
        self.sessions.save(session)
        started = time.monotonic()
        try:
            if session["mode"] == "llm":
                self._llm_loop(session, started)
            else:
                self._standard_plan(session, started)
        except Exception as exc:  # noqa: BLE001 — the session records the failure
            session = self.sessions.get(session_id)
            session["status"] = "failed"
            session["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        session["used"]["minutes"] = round(session["used"].get("minutes_before", 0.0) + (time.monotonic() - started) / 60, 2)
        return self.sessions.save(session)

    def message(self, session_id: str, text: str) -> dict[str, Any]:
        """A follow-up turn. In LLM mode the loop continues; in standard mode the project agent answers."""
        session = self.sessions.get(session_id)
        text = text.strip()[:4000]
        session["messages"].append({"role": "user", "content": text})
        session["final"] = None
        session["status"] = "queued"
        self.sessions.save(session)
        if session["mode"] == "llm":
            return self.run(session_id)
        started = time.monotonic()
        session["status"] = "running"
        session["used"]["minutes_before"] = session["used"]["minutes"]
        self.tracer = self._trace(session)
        if session.get("project_id"):
            result = self._step(session, "ask_project", {"project_id": session["project_id"], "question": text}, started)
            session["final"] = result.get("answer", "") + ("\n\nProof: " + ", ".join(result.get("proof", [])) if result.get("proof") else "")
        else:
            result = self._step(session, "search_evidence", {"query": text, "k": 4}, started)
            session["final"] = "Relevant evidence: " + "; ".join(f"{r['record_id']} — {r['title']}" for r in result.get("results", [])) or "Nothing matched."
        session["status"] = "completed"
        session["used"]["minutes"] = round(session["used"]["minutes_before"] + (time.monotonic() - started) / 60, 2)
        return self.sessions.save(session)

    # ------------------------------------------------------------------ shared
    def _task_message(self, session: dict[str, Any]) -> str:
        text = session["task"]
        if session.get("project_id"):
            text += f"\n\nWork in the existing project {session['project_id']} (call describe_data first)."
        return text + f"\n\nBudget: {session['budget']['max_steps']} tool calls, {session['budget']['max_minutes']} minutes."

    def _exhausted(self, session: dict[str, Any], started: float) -> str | None:
        if session["used"]["steps"] >= session["budget"]["max_steps"]:
            return f"the budget of {session['budget']['max_steps']} tool calls is used up"
        if session["used"].get("minutes_before", 0.0) + (time.monotonic() - started) / 60 >= session["budget"]["max_minutes"]:
            return f"the budget of {session['budget']['max_minutes']} minutes is used up"
        return None

    def _step(self, session: dict[str, Any], tool: str, arguments: dict[str, Any], started: float) -> Any:
        if self.tracer is not None:
            self.tracer.before()
        clock = time.perf_counter()
        result = self.toolbox.call(tool, arguments)
        step = {"n": len(session["steps"]) + 1, "tool": tool, "arguments": arguments if len(json.dumps(arguments, default=str)) <= 1200 else {"_truncated": truncate(arguments, 1200)},
                "summary": summarize(result), "ok": not (isinstance(result, dict) and "error" in result),
                "elapsed_seconds": round(time.perf_counter() - clock, 2), "at": now()}
        if tool == "create_project" and isinstance(result, dict) and result.get("project_id"):
            session["project_id"] = result["project_id"]
        self._record(tool, arguments, result, step["elapsed_seconds"], step["n"])
        session["steps"].append(step)
        session["used"]["steps"] += 1
        session["used"]["minutes"] = round(session["used"].get("minutes_before", 0.0) + (time.monotonic() - started) / 60, 2)
        self.sessions.save(session)
        return result

    def _record(self, tool: str, arguments: dict[str, Any], result: Any, seconds: float, n: int) -> None:
        """A standard-plan step's trace row: no model chose it, so it has no reply number."""
        if self.tracer is not None:
            self.tracer(tool, arguments, result, seconds, n=n)

    # ------------------------------------------------------------------ LLM mode
    def policy(self, session: dict[str, Any]) -> Policy:
        """The intern as a policy on the runtime: its instructions, the project tools and its own two, its budget."""
        budget = session["budget"]
        return Policy("intern", POLICY, tuple(self.toolbox.names()) + SESSION_TOOLS, max_steps=int(budget["max_steps"]),
                      max_seconds=float(budget["max_minutes"]) * 60, terminal=("finish",), result_chars=RESULT_CHARS)

    def _llm_loop(self, session: dict[str, Any], started: float) -> None:
        def on_step(step: Step) -> None:
            entry = {"n": len(session["steps"]) + 1, "tool": step.tool, "ok": step.ok, "elapsed_seconds": round(step.seconds, 2), "at": now()}
            if step.tool in SESSION_TOOLS:  # the plan and the report are on the session itself
                entry.update(arguments={}, summary=((session.get("plan") if step.tool == "write_plan" else session.get("final")) or "")[:200])
            else:
                arguments = step.arguments
                entry.update(arguments=arguments if len(json.dumps(arguments, default=str)) <= 1200 else {"_truncated": truncate(arguments, 1200)},
                             summary=summarize(step.result))
                if step.tool == "create_project" and isinstance(step.result, dict) and step.result.get("project_id"):
                    session["project_id"] = step.result["project_id"]
                self._verdict(step.tool, step.arguments, step.result)
            session["steps"].append(entry)
            if step.tool != "finish" or not step.ok:  # the budget counts tool calls; a report that ends the session is free
                session["used"]["steps"] += 1
            session["used"]["minutes"] = round(session["used"].get("minutes_before", 0.0) + (time.monotonic() - started) / 60, 2)
            self.sessions.save(session)

        try:
            out = run_policy(self.policy(session), _Model(self, session), self.toolbox.registry, session["messages"],
                             contexts={"project": self.toolbox, "session": InternTurn(self, session)}, on_step=on_step,
                             steps_used=session["used"]["steps"], started=started - session["used"].get("minutes_before", 0.0) * 60,
                             trace=self.tracer)
        except RuntimeError as error:
            if not str(error).startswith("BudgetExceeded"):
                raise
            session["status"] = "budget_exhausted"  # the same ending as the step and minute budgets
            session["final"] = f"Stopped: {str(error).split(': ', 1)[-1]}. " + self._progress_note(session)
            return
        if out.stopped in ("steps", "time"):
            budget = session["budget"]
            reason = (f"the budget of {budget['max_steps']} tool calls is used up" if out.stopped == "steps"
                      else f"the budget of {budget['max_minutes']} minutes is used up")
            session["status"] = "budget_exhausted"
            session["final"] = f"Stopped: {reason}. " + self._progress_note(session)
            return
        if out.stopped != "terminal":  # answered without finish: the answer is the report
            session["final"] = out.final or self._progress_note(session)
        session["status"] = "completed"
        self.sessions.save(session)

    def _verdict(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        """Tell the model gateway whether the model's tool call was usable (fixed reasons: never the model's text)."""
        report = getattr(self.client, "output", None)
        if not report:
            return
        if name not in self.toolbox.names() and name not in LEGACY_TOOLS:  # old names still work
            report(False, "called a tool that does not exist")
        elif "_raw" in arguments:
            report(False, "tool arguments were not valid JSON")
        elif isinstance(result, dict) and isinstance(result.get("verdict"), dict) and result["verdict"].get("status") == "blocked":
            report(False, "the workflow validator refused the move")
        elif isinstance(result, dict) and "error" in result:
            report(False, "the tool call was refused")
        else:
            report(True)

    def _progress_note(self, session: dict[str, Any]) -> str:
        pid = session.get("project_id")
        return (f"The work so far is in project {pid} (open it in the notebook: #project/{pid})." if pid else "No project was created.") + f" {len(session['steps'])} tool calls were made."

    # ------------------------------------------------------------------ standard mode
    def _standard_plan(self, session: dict[str, Any], started: float) -> None:
        task = session["task"].lower()
        pid = session.get("project_id")
        session["plan"] = ("Standard DCLab plan (no model configured, so the fixed order runs):\n"
                           "1. pick the dataset the task names\n2. create the project and load the data\n3. write the solution from the R&D audit and suggestion\n"
                           "4. run the five stages (quick mode)\n5. report the honest score, the leakage findings and the decisions with proof")
        self.sessions.save(session)
        if not pid:
            key = next((k for pattern, k in SAMPLE_HINTS if re.search(pattern, task)), None)
            samples = self._step(session, "list_samples", {}, started)
            keys = {s["key"]: s for s in samples.get("samples", [])}
            if key not in keys:
                session["status"] = "completed"
                session["final"] = ("I could not match the task to one of the studied datasets. Name one of: " + ", ".join(keys) +
                                    ", or create a project in the notebook, upload your table and hand it to me with the project id.")
                return
            created = self._step(session, "create_project", {"name": keys[key]["name"], "industry": keys[key]["industry"], "goal": session["task"][:300]}, started)
            pid = created["project_id"]
            self._step(session, "use_sample", {"project_id": pid, "key": key}, started)
        described = self._step(session, "describe_data", {"project_id": pid}, started)
        if "error" in described:
            session["status"] = "completed"
            session["final"] = described["error"]
            return
        if not described.get("solution"):
            suggestion = described.get("suggestion") or {}
            target = suggestion.get("target") or (described["target_candidates"][0] if described["target_candidates"] else None)
            if not target:
                session["status"] = "completed"
                session["final"] = "I could not find a target column. Set the solution in the notebook and hand the project back to me."
                return
            proposal = self._step(session, "propose_solution", {"project_id": pid, "target": target, **({"task": suggestion["task"]} if suggestion.get("task") else {})}, started)
            if "error" in proposal:
                session["status"] = "failed"
                session["final"] = proposal["error"]
                return
            known = {f["column"]: f for f in suggestion.get("forbidden", [])}
            for f in proposal.get("forbidden", []):
                if f["column"] not in known and any(p.startswith("LEAK-") for p in f.get("proof", [])):
                    known[f["column"]] = {"column": f["column"], "reason": f["reason"]}
            # A time, group or identifier column is already kept out of the model by its role; listing it as
            # forbidden too is a contradiction the solution validator refuses (the fraud sample's elapsed Time).
            roles = {suggestion.get("time_column"), suggestion.get("group_column"), *proposal.get("identifiers", []), *suggestion.get("identifiers", [])}
            known = {c: f for c, f in known.items() if c not in roles}
            solution = {"project_id": pid, "target": target, "task": proposal["task"],
                        "prediction_moment": suggestion.get("prediction_moment") or ("Predict at the moment the row is recorded; " + proposal["prediction_moment_hint"]),
                        "forbidden": list(known.values()), "identifiers": sorted(set(proposal.get("identifiers", [])) | set(suggestion.get("identifiers", []))),
                        "time_column": suggestion.get("time_column") or None, "group_column": suggestion.get("group_column") or None,
                        "text_columns": suggestion.get("text_columns") or [], "positive_label": suggestion.get("positive_label") or proposal.get("positive_label"),
                        "metric": suggestion.get("metric") or None}
            saved = self._step(session, "set_solution", {k: v for k, v in solution.items() if v not in (None, [], "")}, started)
            if "error" in saved:
                session["status"] = "failed"
                session["final"] = "The solution could not be saved: " + saved["error"]
                return
        if "full" in task or "all rows" in task:
            self._step(session, "set_settings", {"project_id": pid, "quick": False}, started)
        results = self._step(session, "run_all", {"project_id": pid}, started)
        self._step(session, "export_notebook", {"project_id": pid}, started)
        session["final"] = self._report(pid, results)
        session["status"] = "completed"

    def _report(self, pid: str, results: dict[str, Any]) -> str:
        lines = [f"Project #project/{pid} is ready in the notebook."]
        for stage, record in results.items():
            if not record or "error" in (record or {}):
                lines.append(f"- {stage}: {record.get('error') if record else 'not run'}")
                continue
            lines.append(f"- {record['title']}: {record['summary']}")
            for note in record.get("notes", [])[:2]:
                lines.append(f"  · {note['severity']}: {note['title']} (proof: {', '.join(note['proof'])})")
        final = results.get("final")
        if final and "error" not in final:
            metric = final["metric"]
            lo, hi = final["numbers"]["interval95"]
            lines.append(f"Honest score: {metric} {final['numbers']['holdout'][metric]:.4f} on the holdout (95% {lo:.4f}–{hi:.4f}), CV mean {final['numbers']['cv_mean']:.4f}. "
                         f"Tuning {'accepted' if final['numbers']['tuning_accepted'] else 'rejected'}. Not production-approved (DCLAB-R22).")
        lines.append("Next: open the project, review the notes with their proof, and change the solution if a flagged column is really unknown at prediction time.")
        return "\n".join(lines)


class _Model:
    """The gateway client as the runtime sees it, for one session: before each request it learns the session's project
    and euro limit (both can change mid-session), and after it the session counts the tokens and euros."""

    def __init__(self, intern: Intern, session: dict[str, Any]):
        self.intern, self.session = intern, session

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        client, session = self.intern.client, self.session
        if hasattr(client, "project_id"):  # the gateway counts each request against the session's project
            client.project_id = session.get("project_id")
            client.run_limit_eur = (session.get("budget") or {}).get("max_eur")  # an optional euro cap for this session
            client.spent_eur = float(session["used"].get("eur") or 0.0)  # across turns and restarts, not per client
        try:
            response = client.complete(messages, tools)
        finally:
            if hasattr(client, "spent_eur"):
                session["used"]["eur"] = round(float(client.spent_eur), 6)
        session["used"]["input_tokens"] += response["usage"]["input_tokens"]
        session["used"]["output_tokens"] += response["usage"]["output_tokens"]
        return response


class InternTurn:
    """What the session tools act on: the intern and the session it is running."""

    def __init__(self, intern: Intern, session: dict[str, Any]):
        self.intern, self.session = intern, session


def write_plan(turn: InternTurn, /, plan: str) -> dict[str, Any]:
    turn.session["plan"] = str(plan)[:2000]
    return {"ok": True}


def finish(turn: InternTurn, /, report: str) -> dict[str, Any]:
    turn.session["final"] = str(report).strip() or turn.intern._progress_note(turn.session)
    return {"ok": True}


class SessionGuard:
    """The session tools write the intern's own plan and report: no project or draft changes, so nothing for a validator
    to refuse. The guard exists so that every write in the registry passes one."""

    def __call__(self, tool: Tool, turn: InternTurn, arguments: dict[str, Any], proceed: Any) -> Any:
        return proceed()


def register(registry: Registry) -> None:
    """Register the intern's session tools (scope "session")."""
    registry.guards["session"] = SessionGuard()
    S = {"type": "string"}
    registry.register(Tool("write_plan", "Record your plan for the person (3-6 short steps) before acting.",
                           {"type": "object", "properties": {"plan": S}, "required": ["plan"]}, write_plan,
                           effect="write", scope="session", move=SESSION_MOVES["write_plan"], takes_context=True))
    registry.register(Tool("finish", "End the session with the report for the person.",
                           {"type": "object", "properties": {"report": S}, "required": ["report"]}, finish,
                           effect="write", scope="session", move=SESSION_MOVES["finish"], takes_context=True))
