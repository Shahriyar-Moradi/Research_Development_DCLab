"""The intern's toolbox: evidence tools plus project tools, all with JSON schemas.

Every tool is deterministic and bounded. The model can look things up, create a
project, load a sample or describe uploaded data, propose and set a prediction
contract, run the five stages, approve a choice and export the notebook. It cannot
run arbitrary code, reach the network or touch files outside the project home.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from dclab_rnd import tools as evidence_tools
from dclab_rnd.studio import agent as studio_agent
from dclab_rnd.studio import contract as studio_contract
from dclab_rnd.studio import data as studio_data
from dclab_rnd.studio import engine as studio_engine
from dclab_rnd.studio import export as studio_export
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


class Toolbox:
    def __init__(self, projects: ProjectStore, quick_default: bool = True):
        self.projects = projects
        self.quick_default = quick_default
        self._tools: dict[str, tuple[Callable[..., Any], dict[str, Any], str]] = {}
        for name in EVIDENCE_TOOLS:
            fn, schema = evidence_tools.TOOLS[name]
            self._tools[name] = (fn, schema, (fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "))
        self._register()

    # ------------------------------------------------------------------ registry
    def _add(self, name: str, fn: Callable[..., Any], properties: dict[str, Any], required: list[str], description: str) -> None:
        self._tools[name] = (fn, {"type": "object", "properties": properties, "required": required}, description)

    def schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": name, "description": description, "parameters": schema}}
                for name, (_, schema, description) in self._tools.items()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if name not in self._tools:
            return {"error": f"Unknown tool {name!r}. Available: {', '.join(self._tools)}"}
        fn, schema, _ = self._tools[name]
        arguments = dict(arguments or {})
        unknown = set(arguments) - set(schema.get("properties", {}))
        if unknown:
            return {"error": f"Unknown argument(s) for {name}: {', '.join(sorted(unknown))}"}
        missing = [r for r in schema.get("required", []) if r not in arguments]
        if missing:
            return {"error": f"Missing required argument(s) for {name}: {', '.join(missing)}"}
        try:
            return fn(**arguments)
        except KeyError as exc:
            return {"error": f"Not found: {exc}"}
        except (ValueError, studio_data.DataError) as exc:
            return {"error": str(exc)[:600]}

    # ------------------------------------------------------------------ project tools
    def _register(self) -> None:
        S = {"type": "string"}
        pid = {"project_id": {**S, "description": "The project id returned by create_project."}}
        self._add("list_samples", self.list_samples, {}, [], "List the datasets the R&D already studied, with their task type and the contract it wrote for them.")
        self._add("create_project", self.create_project, {"name": S, "industry": {**S, "enum": list(INDUSTRIES)}, "goal": S}, ["name", "goal"],
                  "Create a new notebook project. Returns its project_id. New projects run in quick mode (3,000 rows) unless set_settings changes it.")
        self._add("use_sample", self.use_sample, {**pid, "key": {**S, "description": "A key from list_samples."}}, ["project_id", "key"],
                  "Load a sample dataset into the project. Returns the data profile and the contract suggestion the R&D wrote for it.")
        self._add("describe_data", self.describe_data, pid, ["project_id"], "Describe the project's table: rows, columns (kind, missing, unique, examples), candidate targets, time/identifier/text columns, and the contract suggestion if any.")
        self._add("propose_contract", self.propose_contract, {**pid, "target": S, "task": {**S, "enum": ["binary", "multiclass", "regression"]}}, ["project_id", "target"],
                  "Audit the columns for a chosen target and propose a prediction contract: task, forbidden columns with reasons and proof, identifiers, time/group/text candidates, metric.")
        self._add("set_contract", self.set_contract, {**pid, "target": S, "task": {**S, "enum": ["binary", "multiclass", "regression"]},
                  "prediction_moment": {**S, "description": "When the prediction is made and what is known then (at least one sentence)."},
                  "forbidden": {"type": "array", "items": {"type": "object", "properties": {"column": S, "reason": S}, "required": ["column"]}},
                  "identifiers": {"type": "array", "items": S}, "time_column": S, "group_column": S, "text_columns": {"type": "array", "items": S},
                  "positive_label": S, "metric": {**S, "enum": ["roc_auc", "average_precision", "macro_f1", "mae"]}},
                  ["project_id", "target", "task", "prediction_moment"], "Save the prediction contract. Saving clears any previous stage results.")
        self._add("set_settings", self.set_settings, {**pid, "quick": {"type": "boolean", "description": "True: 3,000 rows for a fast pass. False: up to max_rows."},
                  "max_rows": {"type": "integer", "minimum": 200, "maximum": 200000}}, ["project_id"], "Change how many rows the stages use.")
        self._add("run_stage", self.run_stage, {**pid, "stage": {**S, "enum": list(STAGE_KEYS)}}, ["project_id", "stage"],
                  "Run one stage (data, leakage, features, models, final) now. Stages run in order; each needs the previous one. Returns the stage record: summary, numbers, claims, notes with proof, decision.")
        self._add("run_all", self.run_all, pid, ["project_id"], "Run every remaining stage in order and return all records. The final stage consumes the holdout once.")
        self._add("approve_stage", self.approve_stage, {**pid, "stage": {**S, "enum": list(STAGE_KEYS)}, "choice": {**S, "description": "Optional: an option id from the stage's decision, to override the rule's choice (clears later stages)."}},
                  ["project_id", "stage"], "Approve a completed stage, optionally choosing a different recipe or model than the rule picked.")
        self._add("get_results", self.get_results, {**pid, "stage": {**S, "enum": list(STAGE_KEYS)}}, ["project_id"], "Return the records of the finished stages (or one stage).")
        self._add("ask_project", self.ask_project, {**pid, "question": S}, ["project_id", "question"], "Ask the deterministic project agent a question; it answers from the project's own results and the evidence index.")
        self._add("export_notebook", self.export_notebook, pid, ["project_id"], "Write the runnable scikit-learn notebook and the markdown report for the project; returns their download paths.")

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
                "suggestion": project.get("suggestion"), "contract": project.get("contract")}

    def propose_contract(self, project_id: str, target: str, task: str | None = None) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if not project.get("data"):
            return {"error": "The project has no data yet."}
        frame = studio_data.load_table(self.projects.data_dir(project_id) / project["data"]["filename"])
        proposal = studio_contract.propose(frame, project["data"]["profile"], target, task)
        project["proposal"] = proposal
        self.projects.save(project)
        return {k: proposal[k] for k in ("target", "task", "detected", "positive_label", "forbidden", "identifiers", "time_candidates",
                                         "group_candidates", "text_columns", "metric", "metric_options", "prediction_moment_hint")}

    def set_contract(self, project_id: str, **fields: Any) -> dict[str, Any]:
        project = self.projects.get(project_id)
        if not project.get("data"):
            return {"error": "The project has no data yet."}
        fields = {k: v for k, v in fields.items() if v not in (None, "", [])}
        contract = studio_contract.Contract(**fields)
        contract.check_columns(project["data"]["columns"])
        project["contract"] = contract.model_dump()
        self.projects.save(project)
        self.projects.clear_stages(project_id)
        self.projects.log(project_id, "contract_saved", {"target": contract.target, "task": contract.task, "forbidden": [f.column for f in contract.forbidden], "by": "intern"})
        return {"saved": True, "contract": project["contract"], "next": "run_stage('data') or run_all"}

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
            record = studio_engine.execute(self.projects, project_id, stage)
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
        project = studio_engine.approve(self.projects, project_id, stage, choice)
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
        if not project.get("contract"):
            return {"error": "Nothing to export: the project has no contract yet"}
        records = self.projects.records(project_id)
        folder = self.projects.directory(project_id) / "exports"
        folder.mkdir(exist_ok=True)
        (folder / "notebook.ipynb").write_text(studio_export.dumps_notebook(studio_export.notebook(project, records)), encoding="utf-8")
        (folder / "report.md").write_text(studio_export.report(project, records), encoding="utf-8")
        return {"notebook": f"/api/projects/{project_id}/export/notebook", "report": f"/api/projects/{project_id}/export/report",
                "files": [str(folder / "notebook.ipynb"), str(folder / "report.md")], "project_url": f"#project/{project_id}"}


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
