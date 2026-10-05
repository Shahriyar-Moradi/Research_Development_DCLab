"""The intern's toolbox: evidence tools plus project tools, all with JSON schemas.

Every tool is deterministic and bounded. The model can look things up, create a
project, load a sample or describe uploaded data, propose and set a prediction
solution, run the five stages, approve a choice and export the notebook. It cannot
run arbitrary code, reach the network or touch files outside the project home.

Every project move the intern makes (run a stage, save a solution, approve a choice,
export) is checked by the workflow graph's validator as the actor "agent" and logged.
A blocked move comes back as an error with the verdict: which check failed, which rule
applies, and whether a person can unblock it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from dclab_rnd import tools as evidence_tools
from dclab_rnd.agents.registry import Registry, Tool
from dclab_rnd.studio import agent as studio_agent
from dclab_rnd.studio import solution as studio_solution
from dclab_rnd.studio import data as studio_data
from dclab_rnd.studio import engine as studio_engine
from dclab_rnd.studio import export as studio_export
from dclab_rnd.studio import graph as studio_graph
from dclab_rnd.studio.store import INDUSTRIES, STAGE_KEYS, ProjectStore

EVIDENCE_TOOLS = ("search_evidence", "get_record", "get_rules", "plan_next_stage", "review_code")
RESULT_CHARS = 6000


def compact_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the model needs from a stage record, without the fold-level bulk."""
    if not record:
        return None
    ev = record["evidence"]
    numbers: dict[str, Any] = {}
    stage = record["stage"]
    if stage == "data":
        numbers = {"train_rows": ev["train_rows"], "holdout_rows": ev["holdout_rows"], "usable_inputs": ev["usable_feature_count"],
                   "cv": ev["cv_protocol"], "risk_features": ev["risk_features"], "target_summary": ev.get("target_summary")}
    elif stage == "leakage":
        numbers = {"forbidden": ev["declared_leakage_features"], "apparent_lift": ev["apparent_lift"], "metric": ev["apparent_lift_metric"],
                   "review_candidates": [{"column": f["feature"], "reasons": f["reasons"]} for f in ev["heuristic_review_candidates"]],
                   "canary_passed": ev["detector_canary_passed"]}
    elif stage == "features":
        numbers = {"selected": ev["selected_recipe"], "best": ev["best_recipe"], "ladder": [{"recipe": r["recipe"], "features": r["feature_count_mean"],
                   "mean": r["metrics"][record["primary_metric"]]["mean"], "std": r["metrics"][record["primary_metric"]]["std"]} for r in ev["stage_results"]]}
    elif stage == "models":
        numbers = {"selected": ev["selected_model"], "ranking": [{"model": r["model"], "mean": r["mean"], "std": r["std"]} for r in ev["ranking"]]}
    elif stage == "final":
        numbers = {"model": ev["model"], "recipe": ev["feature_recipe"], "holdout": {record["primary_metric"]: ev["holdout_metrics"][record["primary_metric"]]},
                   "interval95": [ev["holdout_primary_metric_ci"]["low"], ev["holdout_primary_metric_ci"]["high"]], "cv_mean": ev["cv_selected_metric_mean"],
                   "tuning_accepted": ev["tuning_decision"]["accepted"], "holdout_uses": ev.get("holdout_uses_in_this_project"), "baselines": ev["baselines"]}
    decision = record.get("decision") or {}
    return {
        "stage": stage, "title": record["title"], "metric": record["primary_metric"], "summary": record["setup_summary"], "numbers": numbers,
        "claims": [f"[{c['kind']}] {c['statement']}" for c in record["claims"]],
        "notes": [{"severity": n["severity"], "title": n["title"], "text": n["text"], "proof": n["proof"]} for n in record.get("notes", []) if n["severity"] != "info"][:6],
        "decision": {k: decision[k] for k in ("kind", "selected", "chosen") if k in decision} | ({"options": [o["id"] for o in decision.get("options", [])]} if decision.get("options") else {}),
    }


# Tool names from before the prediction contract was renamed to the solution; old MCP and Chat UI configs still call them.
LEGACY_TOOLS = {"propose_contract": "propose_solution", "set_contract": "set_solution"}


def _expected(exc: Exception) -> dict[str, Any] | None:
    """The failures a model can act on, as tool results; anything else is a bug and raises."""
    if isinstance(exc, studio_graph.GraphBlocked):
        return blocked(exc.verdict)
    if isinstance(exc, KeyError):
        return {"error": f"Not found: {exc}"}
    if isinstance(exc, (ValueError, studio_data.DataError)):
        return {"error": str(exc)[:600]}
    return None


SELF_LOGGED = ("run_stage", "set_solution", "approve_stage", "capture")  # their handlers log the allowed move with its outcome


class ProjectGuard:
    """Every project write passes ``studio.graph.check`` as the actor "agent" before its handler runs (package A2.2).

    A refusal is logged as a transition and returned with the reason, the failed checks and the rules. An allowed move
    whose handler does not log itself is logged after it runs, with its outcome.
    """

    def __call__(self, tool: Tool, box: "Toolbox", arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        if tool.move == "create_project":
            project: dict[str, Any] = {}
            args = {"name": arguments.get("name"), "goal": arguments.get("goal")}
        else:
            project = box.projects.get(arguments["project_id"])
            args = {k: arguments[k] for k in ("stage", "choice", "gate") if arguments.get(k) is not None}
            if tool.name == "run_all":  # the first stage it would run; each later one is checked by the engine as it runs
                args["stage"] = next((s for s in STAGE_KEYS if project["stages"][s].get("status") not in studio_graph.DONE), None)
                if args["stage"] is None:
                    return proceed()  # nothing left to run: it only reads the records
        verdict = studio_graph.check(project, tool.move, "agent", **args)
        if not verdict.allowed:
            if project:
                studio_graph.log(box.projects, project["id"], verdict, project)
            # run_all answers one result per stage (the standard plan's report reads it that way)
            return {args["stage"]: blocked(verdict)} if tool.name == "run_all" else blocked(verdict)
        result = proceed()
        if tool.move not in SELF_LOGGED:
            project_id = project.get("id") or (result.get("project_id") if isinstance(result, dict) else None)
            if project_id:
                failed = isinstance(result, dict) and "error" in result
                studio_graph.log(box.projects, project_id, verdict, project or None,
                                 outcome=f"failed: {str(result['error'])[:200]}" if failed else "done")
        return result


def register(registry: Registry) -> None:
    """Register the evidence tools and the project tools (scope "project": the intern and MCP clients).

    Project tools take the Toolbox as their context: ``handler(toolbox, **arguments)``. Each write declares the
    workflow move it makes (``studio.graph.MOVES``); the project guard checks it before the handler runs.
    """
    registry.guards["project"] = ProjectGuard()
    for name in EVIDENCE_TOOLS:
        fn, schema = evidence_tools.TOOLS[name]
        registry.register(Tool(name, (fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "), schema, fn, errors=_expected))

    def add(name: str, properties: dict[str, Any], required: list[str], description: str, effect: str = "read",
            move: str | None = None, aliases: tuple[str, ...] = ()) -> None:
        registry.register(Tool(name, description, {"type": "object", "properties": properties, "required": required}, getattr(Toolbox, name),
                               effect=effect, move=move, aliases=aliases, takes_context=True, errors=_expected))

    S = {"type": "string"}
    pid = {"project_id": {**S, "description": "The project id returned by create_project."}}
    add("list_samples", {}, [], "List the datasets the R&D already studied, with their task type and the solution it wrote for them.")
    add("create_project", {"name": S, "industry": {**S, "enum": list(INDUSTRIES)}, "goal": S}, ["name", "goal"],
        "Create a new notebook project. Returns its project_id. New projects run in quick mode (3,000 rows) unless set_settings changes it.", "write", "create_project")
    add("use_sample", {**pid, "key": {**S, "description": "A key from list_samples."}}, ["project_id", "key"],
        "Load a sample dataset into the project. Returns the data profile and the solution suggestion the R&D wrote for it.", "write", "attach_data")
    add("describe_data", pid, ["project_id"], "Describe the project's table: rows, columns (kind, missing, unique, examples), candidate targets, time/identifier/text columns, and the solution suggestion if any.")
    add("propose_solution", {**pid, "target": S, "task": {**S, "enum": ["binary", "multiclass", "regression"]}}, ["project_id", "target"],
        "Audit the columns for a chosen target and propose a solution: task, forbidden columns with reasons and proof, identifiers, time/group/text candidates, metric.",
        "write", "propose_solution", aliases=tuple(k for k, v in LEGACY_TOOLS.items() if v == "propose_solution"))
    add("set_solution", {**pid, "target": S, "task": {**S, "enum": ["binary", "multiclass", "regression"]},
        "prediction_moment": {**S, "description": "When the prediction is made and what is known then (at least one sentence)."},
        "forbidden": {"type": "array", "items": {"type": "object", "properties": {"column": S, "reason": S}, "required": ["column"]}},
        "identifiers": {"type": "array", "items": S}, "time_column": S, "group_column": S, "text_columns": {"type": "array", "items": S},
        "positive_label": S, "metric": {**S, "enum": ["roc_auc", "average_precision", "macro_f1", "mae"]}},
        ["project_id", "target", "task", "prediction_moment"], "Save the solution. Saving clears any previous stage results.",
        "write", "set_solution", aliases=tuple(k for k, v in LEGACY_TOOLS.items() if v == "set_solution"))
    add("set_settings", {**pid, "quick": {"type": "boolean", "description": "True: 3,000 rows for a fast pass. False: up to max_rows."},
        "max_rows": {"type": "integer", "minimum": 200, "maximum": 200000}}, ["project_id"], "Change how many rows the stages use.", "write", "set_settings")
    add("run_stage", {**pid, "stage": {**S, "enum": list(STAGE_KEYS)}}, ["project_id", "stage"],
        "Run one stage (data, leakage, features, models, final) now. Stages run in order; each needs the previous one. Returns the stage record: summary, numbers, claims, notes with proof, decision.",
        "write", "run_stage")
    add("run_all", pid, ["project_id"], "Run every remaining stage in order and return all records. The final stage consumes the holdout once.", "write", "run_stage")
    add("approve_stage", {**pid, "stage": {**S, "enum": list(STAGE_KEYS)}, "choice": {**S, "description": "Optional: an option id from the stage's decision, to override the rule's choice (clears later stages)."}},
        ["project_id", "stage"], "Approve a completed stage, optionally choosing a different recipe or model than the rule picked.", "write", "approve_stage")
    add("get_results", {**pid, "stage": {**S, "enum": list(STAGE_KEYS)}}, ["project_id"], "Return the records of the finished stages (or one stage).")
    add("ask_project", {**pid, "question": S}, ["project_id", "question"], "Ask the deterministic project agent a question; it answers from the project's own results and the evidence index.")
    add("export_notebook", pid, ["project_id"], "Write the runnable scikit-learn notebook and the markdown report for the project; returns their download paths.", "write", "capture")
    add("get_graph", pid, ["project_id"],
        "The project's workflow graph: the ten steps WF-01…WF-10 with their state, the current step, which moves are allowed now and why not, the gates, and the last transitions.")
    add("check_move", {**pid, "move": {**S, "enum": [m for m in studio_graph.MOVES if m != "create_project"]}, "stage": {**S, "enum": list(STAGE_KEYS)},
        "choice": S, "gate": {**S, "enum": list(studio_graph.GATES)}}, ["project_id", "move"],
        "Ask the validator whether a move would be allowed right now, without doing it. Returns the checks, the rules and the side effects.")


class Toolbox:
    """The project tools bound to one project store. The tools themselves live in the shared registry."""

    def __init__(self, projects: ProjectStore, quick_default: bool = True, registry: Registry | None = None):
        self.projects = projects
        self.quick_default = quick_default
        if registry is None:
            from dclab_rnd.agents import default_registry
            registry = default_registry()
        self.registry = registry

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas("project")

    def names(self) -> list[str]:
        return self.registry.names("project")

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        # The intern's loop and MCP clients keep the handlers' clamping and coercion until A2.4 moves the intern onto run().
        return self.registry.call(name, arguments, context=self, scope="project", check="names")

    def list_samples(self) -> dict[str, Any]:
        return {"samples": [{k: s[k] for k in ("key", "name", "task", "goal", "industry", "source", "rows", "blocked")} for s in studio_data.sample_catalog() if s["available"]]}

    def create_project(self, name: str, goal: str, industry: str = "general") -> dict[str, Any]:
        project = self.projects.create(name, industry, goal)
        project["settings"]["quick"] = self.quick_default
        project["origin"] = "intern"
        self.projects.save(project)
        return {"project_id": project["id"], "name": project["name"], "url": f"#project/{project['id']}", "settings": project["settings"]}

    def use_sample(self, project_id: str, key: str) -> dict[str, Any]:
        project = studio_data.use_sample(self.projects, project_id, key)
        return self.describe_data(project_id) | {"suggestion": project.get("suggestion")}

    def describe_data(self, project_id: str) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if not project.get("data"):
            return {"error": "The project has no data yet. Call use_sample, or ask the person to upload a table in the notebook."}
        profile = project["data"]["profile"]
        return {"filename": project["data"]["filename"], "rows": project["data"]["rows"], "column_count": profile["column_count"],
                "columns": [{"name": c["name"], "kind": c["kind"], "missing": round(c["missing_rate"], 3), "unique": c["unique"], "examples": c["preview"][:3]} for c in profile["columns"]],
                "target_candidates": profile["target_candidates"], "time_candidates": profile["time_candidates"],
                "id_candidates": profile["id_candidates"], "text_candidates": profile["text_candidates"],
                "suggestion": project.get("suggestion"), "solution": project.get("solution")}

    def propose_solution(self, project_id: str, target: str, task: str | None = None) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if not project.get("data"):
            return {"error": "The project has no data yet."}
        frame = studio_data.load_table(self.projects.data_dir(project_id) / project["data"]["filename"])
        proposal = studio_solution.propose(frame, project["data"]["profile"], target, task)
        project["proposal"] = proposal
        self.projects.save(project)
        return {k: proposal[k] for k in ("target", "task", "detected", "positive_label", "forbidden", "identifiers", "time_candidates",
                                         "group_candidates", "text_columns", "metric", "metric_options", "prediction_moment_hint")}

    def set_solution(self, project_id: str, **fields: Any) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if not project.get("data"):
            return {"error": "The project has no data yet."}
        fields = {k: v for k, v in fields.items() if v not in (None, "", [])}
        solution = studio_solution.Solution(**fields)
        solution.check_columns(project["data"]["columns"])
        verdict = studio_graph.check(project, "set_solution", "agent", target=solution.target)
        studio_graph.log(self.projects, project_id, verdict, project)
        if not verdict.allowed:
            return blocked(verdict)
        project["solution"] = solution.model_dump()
        self.projects.save(project)
        self.projects.clear_stages(project_id)
        self.projects.log(project_id, "solution_saved", {"target": solution.target, "task": solution.task, "forbidden": [f.column for f in solution.forbidden], "by": "intern"})
        return {"saved": True, "solution": project["solution"], "next": "run_stage('data') or run_all"}

    def set_settings(self, project_id: str, quick: bool | None = None, max_rows: int | None = None) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if quick is not None:
            project["settings"]["quick"] = bool(quick)
        if max_rows is not None:
            project["settings"]["max_rows"] = max(200, min(int(max_rows), 200000))
        self.projects.save(project)
        return {"settings": project["settings"]}

    def run_stage(self, project_id: str, stage: str) -> dict[str, Any]:
        self.projects.get(project_id)
        try:
            record = studio_engine.execute(self.projects, project_id, stage, actor="agent")
        except studio_graph.GraphBlocked as exc:
            return blocked(exc.verdict)
        except Exception as exc:  # noqa: BLE001 — the failure is the tool result
            return {"error": f"{stage} failed: {type(exc).__name__}: {str(exc)[:400]}"}
        return compact_record(record)

    def run_all(self, project_id: str) -> dict[str, Any]:
        project = self.projects.get(project_id)
        out: dict[str, Any] = {}
        for stage in STAGE_KEYS:
            if project["stages"][stage].get("status") in ("completed", "approved"):
                out[stage] = compact_record(self.projects.read_stage(project_id, stage))
                continue
            result = self.run_stage(project_id, stage)
            out[stage] = result
            if "error" in result:
                break
        return out

    def approve_stage(self, project_id: str, stage: str, choice: str | None = None) -> dict[str, Any]:
        project = studio_engine.approve(self.projects, project_id, stage, choice, actor="agent")
        return {"stage": stage, "status": project["stages"][stage]["status"], "choice": choice, "stages": {k: v.get("status") for k, v in project["stages"].items()}}

    def get_results(self, project_id: str, stage: str | None = None) -> dict[str, Any]:
        self.projects.get(project_id)
        records = self.projects.records(project_id)
        if stage:
            return compact_record(records.get(stage)) or {"error": f"{stage} has not run"}
        return {s: compact_record(r) for s, r in records.items()} or {"error": "No stage has run yet"}

    def ask_project(self, project_id: str, question: str) -> dict[str, Any]:
        project = self.projects.get(project_id)
        records = self.projects.records(project_id)
        task_type = next((r["task_type"] for r in records.values()), None)
        answer = studio_agent.answer(question, project, records, task_type)
        return {"answer": answer["answer"], "proof": answer["proof"]}

    def export_notebook(self, project_id: str) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if not studio_graph.check(project, "capture", "agent").allowed:
            studio_graph.capture(self.projects, project_id, "agent")  # logs the blocked move and raises
        records = self.projects.records(project_id)
        folder = self.projects.export_dir(project_id)
        (folder / "notebook.ipynb").write_text(studio_export.dumps_notebook(studio_export.notebook(project, records)), encoding="utf-8")
        (folder / "report.md").write_text(studio_export.report(project, records), encoding="utf-8")
        studio_graph.capture(self.projects, project_id, "agent")
        return {"notebook": f"/api/projects/{project_id}/export/notebook", "report": f"/api/projects/{project_id}/export/report",
                "files": [str(folder / "notebook.ipynb"), str(folder / "report.md")], "project_url": f"#project/{project_id}"}


    def get_graph(self, project_id: str) -> dict[str, Any]:
        project = self.projects.get(project_id)
        view = studio_graph.describe(project, "agent")
        view["nodes"] = [{k: n[k] for k in ("id", "name", "state", "rules")} for n in view["nodes"]]
        view["recent_transitions"] = [{k: t.get(k) for k in ("at", "actor", "move", "args", "status", "message", "outcome")} for t in self.projects.transitions(project_id, 8)]
        return view

    def check_move(self, project_id: str, move: str, stage: str | None = None, choice: str | None = None, gate: str | None = None) -> dict[str, Any]:
        project = self.projects.get(project_id)
        return studio_graph.check(project, move, "agent", stage=stage, choice=choice, gate=gate).to_dict()


def blocked(verdict: "studio_graph.Verdict") -> dict[str, Any]:
    """A move the graph did not allow, in the shape the intern reads."""
    return {"error": verdict.message, "status": verdict.status,
            "failed_checks": [c for c in verdict.checks if not c["ok"]], "rules": verdict.rules, "evidence": verdict.evidence,
            "next": "Ask the owner: a person can approve this." if verdict.status == "needs_approval" else "Choose an allowed move (get_graph lists them)."}


def summarize(result: Any, chars: int = 360) -> str:
    """One line for the transcript."""
    if isinstance(result, dict) and "error" in result:
        return "error: " + str(result["error"])[:chars]
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= chars else text[:chars] + "…"


def truncate(result: Any, chars: int = RESULT_CHARS) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= chars else text[:chars] + f"… [{len(text) - chars} more characters omitted]"


__all__ = ["Toolbox", "compact_record", "summarize", "truncate", "time", "Path"]
