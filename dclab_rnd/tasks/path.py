"""The plan before the stages, on Home's build (stage H5 of Hierarchical Planning; the owner, 2026-11-16: "Yes, do it").

A project built from Home ran its five stages with no plan: the path was drawn afterwards, for display, with a warning
when the built model had not followed it. Now the plan is made first and what it catches is applied before the first
stage: a column known only after the outcome is kept out, a split the plan asks for is set. Every change makes the
evaluation stricter, goes through the move a person uses (``set_solution``: the graph's check, its log, the project's
memory, the audit trail, the ledger) and is written on the plan, so the page shows what was done and why. What code cannot
decide is kept as an open question on the plan; a solution a person signed is never changed here.

``plan_project`` is the one entry. With no model it is the rules' plan (instant, free). With ``use_model`` the model
plans with room (minutes: only ever inside a job, never a web request); a model's plan replaces the rules' plan as long
as no stage has run. After work began nothing is changed. Nobody is in the room when it runs, so only what code itself
confirms is applied (the review of 2026-11-16): the rest is asked.
"""

from __future__ import annotations

from typing import Any, Optional

KEY = "path_plan"  # the plan kept on the project: its breakdown, its path, and what was done about each catch


def _columns(project: dict[str, Any], tables: dict[str, list[str]]) -> list[str]:
    named = [str(c.get("name") if isinstance(c, dict) else c) for c in (project.get("data") or {}).get("columns") or []]
    return named or [c for cols in tables.values() for c in cols]


def stored(project: dict[str, Any]) -> Optional[dict[str, Any]]:
    plan = project.get(KEY)
    return plan if isinstance(plan, dict) and isinstance(plan.get("path"), dict) else None


def began(project: dict[str, Any]) -> bool:
    return any((meta or {}).get("status") != "pending" for meta in (project.get("stages") or {}).values())


def plannable(project: dict[str, Any]) -> bool:
    """One table, one outcome to predict: what the rules' plan and its checks are about. A programme of several parts, a
    solution map, or a kind with no outcome column is left to its own owner."""
    solution = project.get("solution") or {}
    return bool(solution.get("target")) and solution.get("task") in ("binary", "multiclass", "regression") and not any(
        project.get(key) for key in ("programme_plan", "programme", "solution_map"))


def _group_candidates(frames: dict[str, Any], columns: list[str], target: Any) -> list[str]:
    """Columns that code itself reads as an entity: text or whole numbers, many distinct values, and values that repeat.
    A model naming `plan_type` (three values) as the entity is not believed on its word."""
    out = []
    for frame in list((frames or {}).values())[:1]:
        rows = len(frame)
        for name in columns:
            if name == target or name not in getattr(frame, "columns", ()):
                continue
            series = frame[name].dropna()
            kind = getattr(series.dtype, "kind", "O")
            distinct = int(series.nunique())
            if kind in ("O", "i", "u", "U", "S") and 10 <= distinct <= 0.8 * max(rows, 1) and distinct < len(series):
                out.append(str(name))
    return out


def _confirmed(plan: dict[str, Any], main: str, column: str) -> Optional[bool]:
    """Did a check of this plan, on this very column of the main table, find by code that the column is known only after
    the outcome? None when no check of the plan is about this column at all (a catch about a related table's column,
    or about a column the table does not have, is never moved onto a column whose name merely ends the same)."""
    from dclab_rnd.tools import POST_OUTCOME_NAME

    seen = None
    for leg in (plan.get("path") or {}).get("legs") or []:
        for step in leg.get("steps") or []:
            if step.get("task") in ("leakage_probe", "profile_column") and step.get("on") in (column, f"{main}.{column}"):
                # confirmed: the check found it, and the name is one the product's own leakage audit keeps out
                # (``tools.POST_OUTCOME_NAME``). A snapshot column (`lifetime_orders`, `last_order_date`) may be honest
                # as of the prediction moment: that is the person's to say, so it is asked.
                seen = bool(seen) or (f"post_outcome_{column}" in (step.get("found") or []) and bool(POST_OUTCOME_NAME.search(column)))
    return seen


def plan_project(projects: Any, project_id: str, *, gateway: Any = None, use_model: bool = False, cap_eur: float = 0.25,
                 person_kept: Any = ()) -> dict[str, Any]:
    """Make the project's plan and apply what it catches, before any stage has run.

    ``{"status": "planned" | "kept" | "too_late" | "skipped" | "changed_meanwhile", "routed_to", "applied", "asks"}``.
    ``person_kept``: columns the person has said are known at the prediction moment; they are never kept out here.

    Nobody is in the room when this runs, so code applies only what code itself confirms, and asks about the rest:
    - a column is kept out only when a check of the plan, on that very column of the main table, found it by code;
    - a split by time only on a column code reads as a time; a group only on a column code reads as an entity;
    - a solution the person wrote or signed is never changed: every catch is a question;
    - a column the person said is known beforehand stays in.
    The model, when asked, is asked outside the project's lock (its answer takes minutes); its plan is applied only if
    the solution and the stages are as they were."""
    from dclab_core.task_fingerprints import metadata_fingerprint
    from dclab_rnd.studio.store import now
    from dclab_rnd.tasks import enforced
    from dclab_rnd.tasks.catalog import needs_model_planning, plan_with_model, project_catalog_inputs, rules_plan
    from dclab_rnd.tasks.dispatch import TaskDispatcher, change_solution

    def as_it_is(project: dict[str, Any]) -> str:
        return metadata_fingerprint({"solution": project.get("solution"), "data": (project.get("data") or {}).get("sha256"),
                                     "plan": (stored(project) or {}).get("made_at")})

    def kept_result(plan: dict[str, Any]) -> dict[str, Any]:
        return {"status": "kept", "routed_to": plan.get("routed_to"), "applied": [], "asks": list((plan.get("enforcement") or {}).get("asked") or [])}

    with projects.turn(project_id):
        project = projects.get(project_id)
        before = stored(project)
        if began(project):
            return {"status": "too_late", "routed_to": (before or {}).get("routed_to"), "applied": [], "asks": []}
        if not plannable(project):
            return {"status": "skipped", "routed_to": None, "applied": [], "asks": []}
        if before is not None and (not use_model or before.get("routed_to") == "model"):
            return kept_result(before)
        brief, tables, facts, router_tables, frames = project_catalog_inputs(projects, project)
        mark = as_it_is(project)
    plan = None
    if use_model and gateway is not None and needs_model_planning(brief, router_tables, facts):
        try:  # outside the lock: nothing of the project waits on a model's minutes
            plan = plan_with_model(brief, tables, facts, project_id=project_id, gateway=gateway, cap_eur=cap_eur, frames=frames)
        except Exception:  # noqa: BLE001 - a model that cannot plan never stops a build: the rules' plan stands
            plan = None
        if plan is not None and plan.get("routed_to") != "model":
            plan = None  # the model's answer was not usable
    if plan is None:
        if before is not None:
            return kept_result(before)
        plan = rules_plan(brief, tables, facts, frames=frames)
    with projects.turn(project_id):
        project = projects.get(project_id)
        if began(project):
            return {"status": "too_late", "routed_to": (before or {}).get("routed_to"), "applied": [], "asks": []}
        if as_it_is(project) != mark:  # the person changed the solution while the model thought: their solution stands
            return {"status": "changed_meanwhile", "routed_to": (stored(project) or {}).get("routed_to"), "applied": [], "asks": []}
        solution = project.get("solution") or {}
        columns = _columns(project, tables)
        main = next(iter(tables), "main")
        earlier = (before or {}).get("enforcement") or {}
        record = {"applied": list(earlier.get("applied") or []), "waived": list(earlier.get("waived") or []), "asked": [], "refused": [], "too_late": []}
        stays = {str(c) for c in person_kept or ()} | {str(w.get("value")) for w in record["waived"] if w.get("kind") == "forbid"}
        for column in sorted(stays - {str(w.get("value")) for w in record["waived"]}):
            record["waived"].append({"kind": "forbid", "value": column, "trap": f"post_outcome_{column}",
                                     "answer": "the person said this column is known at the prediction moment", "at": now()})
        by_person = project.get("solution_saved_by") == "human"
        times = TaskDispatcher._time_candidates(frames, columns)
        groups = _group_candidates(frames, columns, solution.get("target"))
        applied = []

        def ask(change: dict[str, Any], question: str, **more: Any) -> None:
            record["asked"].append({"trap": change["trap"], "kind": change["kind"], "field": change["field"], "value": change["value"], "question": question, **more})

        for change in enforced.changes(plan.get("caught_traps") or [], plan.get("calls") or [], solution, columns, times):
            field, value = change["field"], change["value"]
            if change["action"] == "none" or (field == "forbidden" and value in stays):
                continue
            found = _confirmed(plan, main, str(value)) if field == "forbidden" else True
            if found is None:
                continue  # no check of the plan is about this column of this table: not a catch about it, nor a question
            if change["action"] == "ask":
                ask(change, change["why"])
                continue
            if field == "forbidden":
                if not found:
                    ask(change, f"The plan suspects `{value}` is known only after the outcome, and code could not confirm it. Is `{value}` known at the prediction moment?")
                    continue
            if field == "time_column" and value not in times:
                ask(change, f"The plan asks for a split by time on `{value}`, which code does not read as a time. Which column holds the time of each row?")
                continue
            if field == "group_column" and value not in groups:
                ask(change, f"The plan asks to keep the rows of one `{value}` on one side of the split, and code does not read `{value}` as an entity "
                            "(many distinct values that repeat). Is it one?")
                continue
            if by_person:
                ask(change, f"The plan asks for a stricter check ({change['why']}), and you wrote this solution yourself. Change it?", needs_person=True)
                continue
            refused = change_solution(projects, project_id, [change], "agent")
            if refused is None:
                applied.append({"trap": change["trap"], "kind": change["kind"], "field": field, "value": value, "why": change["why"], "by": "agent", "at": now()})
            else:  # a signed solution, or one whose own rules refuse the change: said, never worked around
                ask(change, f"The plan asks for a stricter check ({change['why']}), and DCLab did not change the solution by itself: {refused[:160]} Change it?",
                    needs_person=True)
        record["applied"].extend(applied)
        project = projects.get(project_id)  # the solution as the changes left it
        project[KEY] = {**plan, "enforcement": record, "stored": True, "made_at": now(), "made_for": "home build"}
        projects.save(project)
        try:
            TaskDispatcher._record_catalog_ledger(project_id, project[KEY], tables)
        except Exception:  # noqa: BLE001 - capture is best effort
            pass
        from dclab_rnd import experience

        experience.record("agent.step", "system", {"agent": "task.path_plan", "n": len(record["applied"]), "state": str(plan.get("routed_to")),
                                                   "reply": None, "tool": "set_solution" if applied else None,
                                                   "verdict": (", ".join(f"{a['field']}={a['value']}" for a in applied) or "nothing to change")[:300],
                                                   "seconds": 0}, project=project_id)
        return {"status": "planned", "routed_to": plan.get("routed_to"), "applied": applied, "asks": record["asked"]}


def says(result: dict[str, Any]) -> Optional[str]:
    """One or two plain sentences for the person on what the plan did before the stages (for Home's reply)."""
    if result.get("status") != "planned":
        return None
    parts = []
    kept_out = [a["value"] for a in result.get("applied") or [] if a["field"] == "forbidden"]
    if kept_out:
        parts.append("Before the first stage, the plan kept out " + ", ".join(f"`{c}`" for c in kept_out) + " (known only after the outcome).")
    for a in result.get("applied") or []:
        if a["field"] == "time_column":
            parts.append(f"The split is by time, on `{a['value']}`.")
        if a["field"] == "group_column":
            parts.append(f"Rows of one `{a['value']}` stay on one side of the split.")
    if result.get("asks"):
        parts.append(f"The plan has {len(result['asks'])} open question(s) for you on the project's page.")
    return " ".join(parts) or None


def says_update(project: dict[str, Any]) -> Optional[str]:
    """What to tell the person when the model's plan replaced the rules' plan before the first stage. Live, 2026-11-16:
    the build message said "1 open question", the model's plan had none, and the chat was never corrected."""
    plan = stored(project)
    if plan is None or plan.get("routed_to") != "model":
        return None
    legs = (plan.get("path") or {}).get("legs") or []
    steps = sum(len(leg.get("steps") or []) for leg in legs)
    asks = (plan.get("enforcement") or {}).get("asked") or []
    return (f"Before the first stage, the model refined the plan: {len(legs)} sub-problem(s), {steps} step(s). "
            + (f"It has {len(asks)} open question(s) for you on the project's page; this replaces the count in my earlier message."
               if asks else "It has no open question for you; this replaces the count in my earlier message."))


__all__ = ["KEY", "began", "plan_project", "plannable", "says", "says_update", "stored"]
