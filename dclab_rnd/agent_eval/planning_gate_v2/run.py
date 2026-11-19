"""One held-out case built end to end on the product's own unattended path, by one arm.

The order is the product's (Home's build, stage H5): the standard plan writes the solution, the plan is made and what it
catches is applied (``tasks.path.plan_project``), then the five stages run. The rules arm plans with no model; the model
arm first gets the rules' plan, as a build does, and then the model's plan with room, as the build's job does. The
built project is read by code into one row: what the plan caught and whether each catch was enforced, what was planted
and whether it was enforced, the forbidden columns before and after the plan, the events check 4 counts, and the plan's
seconds and euros.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

TRAP_NAME = {"late_column": "post_outcome_{column}", "group_split": "grouped_split_{column}", "time_split": "time_split_required"}


def request(case: Mapping[str, Any]) -> str:
    """The person's words: the case's brief, and the outcome when the case states it (a case that does not is a
    question the build must stop on)."""
    target = (case.get("known_facts") or {}).get("target")
    return str(case["brief"]) + (f" Predict `{target}`." if target else "")


def seed(projects: Any, case: Mapping[str, Any], frames: Mapping[str, Any]) -> str:
    """A project holding the case's tables: the first is the main table, the others attached by name."""
    from dclab_rnd.studio import data as sd

    project = projects.create(str(case["id"]), "general", request(case)[:300])
    pid = project["id"]
    for i, (name, frame) in enumerate(frames.items()):
        filename = name if name.endswith(".csv") else f"{name}.csv"
        frame.to_csv(projects.data_dir(pid) / filename, index=False)
        if i == 0:
            sd.attach_data(projects, pid, filename, source={"kind": "planning_gate_v2", "case": str(case["id"])}, actor="human")
        else:
            sd.attach_table(projects, pid, filename, name=Path(filename).stem, actor="human")
    return pid


def planted(case: Mapping[str, Any], solution: Mapping[str, Any], stages: Mapping[str, Any], columns: list[str]) -> list[dict[str, Any]]:
    """Each planted trap, read as a catch would be, and whether the built project enforced it. A time split counts only
    when it is ordered by the planted column."""
    from dclab_rnd.tasks import enforced

    out = []
    for trap in case.get("enforceable_traps") or []:
        name = TRAP_NAME[trap["kind"]].format(column=trap["column"])
        record = (enforced.obeyed([name], solution, stages, columns) or [{}])[0]
        done = bool(record.get("enforced"))
        if trap["kind"] == "time_split" and done:
            done = str(solution.get("time_column") or "") == str(trap["column"])
        out.append({"kind": trap["kind"], "column": trap["column"], "enforced": done})
    return out


class Watched:
    """The model's view with every failed request written down: a plan that falls back to the rules because the cap ran
    out, the credit ended or the key was refused must stop the run, never pass as the rules' plan (the review)."""

    def __init__(self, view: Any):
        self.view, self.errors = view, []

    def client(self, *args: Any, **kwargs: Any) -> Any:
        errors = self.errors
        try:
            inner = self.view.client(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - asking for a client failed: written down, raised as before
            errors.append(error)
            raise
        if inner is None:  # no model serves this purpose: the plan would quietly be the rules' (the second review)
            errors.append(RuntimeError(f"no model serves {args[0] if args else kwargs.get('purpose')}"))
            return None

        class Client:
            def __getattr__(self, name: str) -> Any:
                attr = getattr(inner, name)
                if not callable(attr):
                    return attr

                def call(*a: Any, **k: Any) -> Any:
                    try:
                        return attr(*a, **k)
                    except Exception as error:  # noqa: BLE001 - written down, then raised as before
                        errors.append(error)
                        raise
                return call
        return Client()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.view, name)


def _forbid_catches(plan: Mapping[str, Any], columns: list[str]) -> set[str]:
    from dclab_rnd.tasks import enforced

    return {str(enforced.column_of(t, columns)) for t in plan.get("caught_traps") or []
            if enforced.kind_of(str(t)) == "forbid" and enforced.column_of(t, columns)}


def build(case: Mapping[str, Any], frames: Mapping[str, Any], arm: str, folder: Path, gateway: Any = None,
          spent: Any = None) -> dict[str, Any]:
    """Build one case with one arm in its own workspace folder. ``gateway``: the model's view (the model arm), wrapped in
    ``Watched`` so a failed request is seen; ``spent``: a function returning the euros spent so far (to price the plan)."""
    from dclab_rnd.agent_eval.e2e.score import unsaid
    from dclab_rnd.intern import Intern, SessionStore
    from dclab_rnd.intern.tools import Toolbox
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import engine
    from dclab_rnd.tasks import enforced
    from dclab_rnd.tasks import path as task_path

    projects = open_stores(folder)[0]
    sessions = SessionStore(folder / "intern")
    pid = seed(projects, case, frames)
    columns = [c for f in frames.values() for c in f.columns]
    watched = Watched(gateway) if gateway is not None else None
    timing: dict[str, Any] = {"plan_seconds": None, "plan_eur": 0.0, "forbidden_before": None, "plan_status": None, "rules_catches": []}

    class Planned(Intern):
        """The standard plan, with the product's plan made between the solution and the stages. A plan that fails
        never stops the stages, as on Home's build and in its job."""

        def _step(self, session: dict[str, Any], tool: str, arguments: dict[str, Any], started: float) -> Any:
            if tool == "run_all" and timing["plan_status"] is None:
                try:
                    project = projects.get(pid)
                    timing["forbidden_before"] = sorted(enforced.forbidden_columns(project.get("solution") or {}))
                    clock = time.monotonic()
                    result = task_path.plan_project(projects, pid)  # the rules' plan, made at the build
                    timing["plan_seconds"] = round(time.monotonic() - clock, 3)
                    timing["rules_catches"] = sorted(_forbid_catches(task_path.stored(projects.get(pid)) or {}, columns))
                    if arm == "model":  # what the model adds, as the build's job adds it
                        euros, clock = (spent() if spent else 0.0), time.monotonic()
                        result = task_path.plan_project(projects, pid, gateway=watched, use_model=True)
                        timing.update(plan_seconds=round(time.monotonic() - clock, 2), plan_eur=round((spent() if spent else 0.0) - euros, 6))
                    timing["plan_status"] = {"status": result.get("status"), "routed_to": result.get("routed_to"),
                                             "applied": len(result.get("applied") or ()), "asks": len(result.get("asks") or ())}
                except Exception as error:  # noqa: BLE001
                    timing["plan_status"] = {"status": "error", "error": type(error).__name__}
            return super()._step(session, tool, arguments, started)

    from dclab_rnd.studio.agent import notes_off

    intern = Planned(sessions, Toolbox(projects), None)
    with notes_off():  # the stages' model-written notes would go past the run's cap and folder (the second review): the rules' notes stand
        session = intern.run(intern.start(request(case), {"max_steps": 40, "max_minutes": 30}, project_id=pid)["id"])
    project = projects.get(pid)
    stages = {s: projects.read_stage(pid, s) for s in engine.STAGE_KEYS}
    stages = {k: v for k, v in stages.items() if v is not None}
    solution = project.get("solution") or {}
    plan = task_path.stored(project) or {}
    caught = [c for c in enforced.obeyed(plan.get("caught_traps") or [], solution, stages, columns) if c["enforceable"]]
    # a catch code could not confirm is asked on the project's page: the plan's own way, by design (H5). Matched by its
    # kind and column (one question stands for every catch of that kind on that column; one for every time catch)
    asked = {(str(a.get("kind")), None if a.get("kind") == "time_split" else str(a.get("value"))) for a in (plan.get("enforcement") or {}).get("asked") or []}
    splits_on = {c for c in (solution.get("time_column"), solution.get("group_column")) if c}
    for c in caught:
        c["asked"] = not c.get("enforced") and (c["kind"], None if c["kind"] == "time_split" else str(c["column"])) in asked
        if c["kind"] == "forbid" and c["column"] in splits_on and not c.get("enforced"):
            c["by_role"] = True  # the split's own column: its role keeps it out of the features, the product's intended way
    step_error = any(isinstance(st.get("result"), dict) and "error" in st["result"] for st in session.get("steps") or [])
    stopped = ("final" not in stages and timing["plan_status"] is None and session.get("status") == "completed"
               and not solution.get("target") and not step_error)
    plants = planted(case, solution, stages, columns)
    final = str(session.get("final") or "")
    seen = [json.dumps(stages, default=str), json.dumps(solution, default=str), json.dumps(session.get("leak_evidence") or {}, default=str)]
    errors = [type(e).__name__ for e in (watched.errors if watched else [])]
    return {"case": str(case["id"]), "arm": arm, "project_id": pid, "session_status": session.get("status"),
            "stopped_with_question": bool(stopped), "step_error": step_error,
            "obeyed_read": len(caught), "obeyed_done": sum(bool(c.get("enforced") or c.get("asked") or c.get("by_role")) for c in caught),
            "asked_not_applied": sum(bool(c.get("asked")) for c in caught), "caught": caught,
            "routed_to": plan.get("routed_to"), "plan_status": timing["plan_status"], "model_errors": errors,
            "rules_catches": timing["rules_catches"], "plan_catches": sorted(_forbid_catches(plan, columns)),
            "forbidden_before": timing["forbidden_before"] if timing["forbidden_before"] is not None else sorted(enforced.forbidden_columns(solution)),
            "forbidden_after": sorted(enforced.forbidden_columns(solution)),
            "planted": len(plants), "planted_enforced": sum(p["enforced"] for p in plants), "planted_traps": plants,
            "holdout_read_twice": int((project.get("holdout_uses") or 0) > 1),
            "stages_past_a_needed_yes": int("final" in stages and any(a.get("needs_person") for a in (plan.get("enforcement") or {}).get("asked") or [])),
            "failed_with_nothing_built": int(not stopped and "final" not in stages),
            # the report names the project by its id (hex: a leading digit reads as a number): it is no number anyone claimed
            "numbers_nobody_gave": unsaid(final.replace(pid, ""), seen) if "final" in stages else 0,
            "plan_seconds": timing["plan_seconds"], "plan_eur": timing["plan_eur"], "holdout_split": enforced.split_of(stages)}


def load_held(folder: Optional[Path] = None) -> tuple[list[dict[str, Any]], list[int], Any]:
    """The held-out cases, their seeds and their tables' module, after checking the author's sha256 pins."""
    import hashlib
    import importlib.util

    folder = folder or Path(__file__).resolve().parent / "held"
    pins = {}
    for line in (folder / "cases.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split()
            pins[name] = sha
    for name in ("cases.json", "tables.py"):
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if pins.get(name) != actual:
            raise ValueError(f"{name} does not match its pin: the held-out cases changed after they were written")
    doc = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("planning_gate_v2_held_tables", folder / "tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return list(doc["cases"]), list(doc["seeds"]), module
