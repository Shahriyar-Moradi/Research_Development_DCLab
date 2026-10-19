"""A project's Workflow Run and G4 (package 30.15): the framework's twelve steps read from the project's own records, each
with its check and the record path of its evidence, and G4 at the end of the final stage.

``g4_of`` runs when the final stage has read the holdout (both the tabular stages and a kind's own machine): the holdout
read once, the folds' score against the holdout's, a score too good to be honest, the signed brief's bar, the operating
rule and its calibration, the monitoring plan. Its ``GateResult`` is written into the final record, captured as the
ledger's ``workflow.g4`` decision, and a failure names the phase the work goes back to. ``build`` maps the steps; a
step the records do not show is missing, never assumed done. Nothing here reads the holdout.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core.workflow_run import STEPS, TITLES, GateResult, RunStep, WorkflowRun, g4


def _rule(p: Any, evidence: dict[str, Any]) -> Optional[str]:
    """The operating rule a person acts on, from the final evidence."""
    capacity = evidence.get("capacity") or {}
    if capacity.get("k"):
        return f"the top {capacity['k']} a {capacity.get('period') or 'period'} (the team's capacity)"
    threshold = ((evidence.get("task_assessment") or {}).get("threshold") or {})
    if threshold.get("value") is not None:
        return f"a score at or above {threshold['value']}: {threshold.get('chosen_by') or 'the operating point'}"
    if evidence.get("review_list") is not None:
        return "the review list of the budget's size"
    task = str(getattr(p, "task_type", "") or getattr(p, "task", ""))
    if "multiclass" in task:
        return "the most likely class for each row"
    from .engine import _regression

    if _regression(p):
        return "the predicted number for each row, with its interval when the assessment gives one"
    kind = getattr(p, "kind", None)
    if getattr(p, "machine", None) is not None and kind is not None:
        return f"the {kind.title.lower()}'s output, as its machine gives it"
    return None


def _calibration_said(evidence: dict[str, Any], task: str) -> Optional[str]:
    fit = evidence.get("calibration_fit")
    if fit and fit.get("calibrator"):
        c = fit["calibrator"]
        b = fit.get("holdout_brier") or {}
        return f"{c.get('method')} ({c.get('reason') or ''}); the holdout's Brier score {b.get('before')} before, {b.get('after')} after"
    if fit:
        return fit.get("reason") or "not fitted"
    return None if task in ("binary_imbalanced", "text_tabular_binary") else "not applicable: the model gives no probability of a yes or no"


def g4_of(p: Any, evidence: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    """G4 for the final stage just run, as a record (``GateResult``)."""
    metric = p.metric
    holdout = (evidence.get("holdout_metrics") or {}).get(metric)
    ci = evidence.get("holdout_primary_metric_ci") or {}
    interval = (float(ci["low"]), float(ci["high"])) if isinstance(ci.get("low"), (int, float)) and isinstance(ci.get("high"), (int, float)) else None
    higher = getattr(p.machine, "higher_is_better", True) if getattr(p, "machine", None) is not None else bool(p.bundle.spec.higher_is_better)
    folds = evidence.get("cv_selected_metric_mean")
    plan = evidence.get("monitoring") or {}
    task = str(getattr(p, "task_type", ""))
    chosen = next((r for r in evidence.get("configuration_results") or [] if r.get("config_id") == evidence.get("selected_optimization")), None)
    spread = ((chosen or {}).get("metrics") or {}).get(metric) or {}
    result = g4(holdout_reads=int(project.get("holdout_uses") or 0), metric=metric,
                holdout=float(holdout) if isinstance(holdout, (int, float)) else None,
                folds=float(folds) if isinstance(folds, (int, float)) else None, higher_is_better=bool(higher), interval=interval,
                folds_std=float(spread["std"]) if isinstance(spread.get("std"), (int, float)) else None,
                bar=evidence.get("success_bar"), rule=_rule(p, evidence), calibration=_calibration_said(evidence, task),
                monitoring=bool(plan.get("columns")), monitoring_why=str(plan.get("reason") or ""))
    return result.model_dump(mode="json")


def machine_monitoring(p: Any) -> dict[str, Any]:
    """A kind's own machine keeps no feature table of its training rows: the plan is the table's columns as the machine read
    them (its target, keys and forbidden columns left out), said so."""
    try:
        import pandas as pd

        from dclab_core import monitoring

        from dclab_core import sensitivity

        solution, project = p.solution, p.project
        training = getattr(p.machine, "training_rows", None)
        frame = training() if callable(training) else None
        on = "the training rows" if frame is not None else "every row"
        frame = p.frame if frame is None else frame
        profile = (project.get("data") or {}).get("profile") or {}
        # the review of 30.15: a person's columns (their phones) were kept by value, and the time column always looked moved
        personal = {c["name"] for c in profile.get("columns", []) if sensitivity.upgrade_column(c).get("sensitivity", "none") != "none"}
        times = {c["name"] for c in profile.get("columns", []) if str(c.get("kind")) in ("datetime", "date", "time")}
        times |= {v for k, v in (p.kind.fields(solution.kind_fields) or {}).items() if isinstance(v, str) and "time" in k} | {solution.time_column}
        out = {c for c in [solution.target, *(solution.identifiers or []), *(f.column for f in solution.forbidden or [])] if c} | personal | times
        columns = [c for c in frame.columns if c not in out]
        skip = {c: "unique on almost every row (an identifier or free text): its distribution says nothing"
                for c in columns if frame[c].nunique() > 0.9 * max(len(frame), 1)}
        built = monitoring.plan(frame[columns], skip=skip, built_on=f"{on} the kind's machine read ({len(frame)}): a baseline of the inputs, no score")
        left = [f"{c}: personal, never monitored by value" for c in sorted(personal & set(frame.columns))]
        left += [f"{c}: a time, which always moves" for c in sorted((times & set(frame.columns)) - personal)]
        return built.model_copy(update={"left_out": [*built.left_out, *left][:200]}).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        return {"reason": f"no monitoring plan ({type(exc).__name__})"}


def _stage(project: dict[str, Any], store: Any, stage: str) -> dict[str, Any]:
    record = (project.get("stages") or {}).get(stage) or {}
    if not record.get("evidence") and store is not None:
        record = store.read_stage(project["id"], stage) or record
    return record or {}


def _done(record: dict[str, Any]) -> bool:
    return record.get("status") in ("completed", "approved") or bool(record.get("evidence"))


def build(store: Any, project: dict[str, Any]) -> WorkflowRun:
    """The twelve steps from the project's records, and G4 when the final stage wrote one."""
    data, leak, feats, models, final = (_stage(project, store, s) for s in ("data", "leakage", "features", "models", "final"))
    ev = {k: (r.get("evidence") or {}) for k, r in (("data", data), ("features", feats), ("models", models), ("final", final))}
    machine = bool(final.get("machine") or data.get("machine"))
    exports = project.get("exports") or {}
    run_id = (final.get("run") or {}).get("run")
    packaged = [e for k, e in exports.items() if isinstance(e, dict) and (e.get("same") or (k == "service" and e.get("checked")))
                and e.get("final_run") == run_id]  # 30.15's review: an export of an earlier final never packages this one
    cleaning = ((project.get("draft") or {}).get("cleaning_log")) or []

    def step(sid: str, status: str, check: str, evidence: Optional[str] = None, why: str = "") -> RunStep:
        return RunStep(id=sid, status=status, check=check, evidence=evidence, why=why)

    final_done = _done(final)
    ranking = ev["models"].get("ranking") or []
    steps = [
        step("lock_split", "done" if _done(data) else "missing", "the holdout set aside before any stage reads a row of it",
             "data.provenance.holdout" if _done(data) else None, "" if _done(data) else "the data stage has not run"),
        step("clean", "done" if cleaning else ("partial" if _done(data) else "missing"),
             "the table's cleaning steps logged (names, types, duplicates) before the split is read",
             "project.draft.cleaning_log" if cleaning else ("data.evidence.risks" if _done(data) else None),
             "" if cleaning else "the table came without a cleaning log: the data stage profiled its risks, it changed nothing"),
        step("baseline", "done" if ev["final"].get("baselines") or ev["models"].get("dummy_cv") else ("missing" if not final_done else "partial"),
             "a simple model on the same rows: the number every candidate must beat",
             "models.evidence.dummy_cv" if ev["models"].get("dummy_cv") else ("final.evidence.baselines" if ev["final"].get("baselines") else None),
             "" if ev["models"].get("dummy_cv") else "the simple model is read on the holdout only, not on the folds"),
        step("features", "done" if _done(feats) else ("not_applicable" if machine and _done(data) else "missing"),
             "the feature recipe chosen by its tolerance on identical folds", "features.evidence.selected_recipe" if _done(feats) else None,
             "a kind's own machine builds its own inputs" if machine and not _done(feats) else ""),
        step("candidates", "done" if len(ranking) >= 2 else ("partial" if ranking else "missing"), "at least two families screened on the same folds",
             "models.evidence.ranking" if ranking else None),
        step("evaluate", "done" if ranking and all("mean" in r for r in ranking) else ("partial" if _done(models) else "missing"),
             "each candidate's mean and spread across the folds", "models.evidence.ranking" if ranking else None),
        step("select", "done" if ev["models"].get("selected_model") else "missing", "the rule's pick, written before the holdout is read",
             "models.evidence.selection_rule" if ev["models"].get("selected_model") else None),
        step("calibrate_rule", ("done" if (ev["final"].get("calibration_fit") or {}).get("calibrator") or (ev["final"].get("gate_g4") and _rule_ok(ev["final"]))
                                else ("partial" if final_done else "missing")),
             "the probabilities calibrated on the out-of-fold scores and the operating rule set on the training rows",
             "final.evidence.calibration_fit" if ev["final"].get("calibration_fit") else ("final.evidence.task_assessment.threshold" if ev["final"].get("task_assessment") else None)),
        step("holdout_once", ("done" if int(project.get("holdout_uses") or 0) == 1 else "partial") if final_done else "missing",
             "the holdout read once, after the choices", "final.evidence.holdout_uses_in_this_project" if final_done else None,
             f"read {int(project.get('holdout_uses') or 0)} times" if final_done and int(project.get("holdout_uses") or 0) != 1 else ""),
        step("package", "done" if packaged else ("partial" if exports else "missing"),
             "the artifact saved, reloaded and its predictions matched (or its code recomputing the holdout number)",
             "project.exports" if exports else None, "" if packaged else ("the export is of an earlier final, or did not match: export again"
                                                                           if exports else "export the project to package it")),
        step("report", "done" if final_done else "missing", "the report written from the records, every number traced",
             "final.evidence" if final_done else None),
        step("plan_monitoring", ("done" if (ev["final"].get("monitoring") or {}).get("score") or (ev["final"].get("monitoring") or {}).get("columns") and not machine
                                 else "partial" if (ev["final"].get("monitoring") or {}).get("columns") else "missing"),
             "the inputs' and the score's baselines with the alert lines, in the artifact", "final.evidence.monitoring" if ev["final"].get("monitoring") else None,
             "a kind's own machine: the inputs' baseline only, no score" if machine and (ev["final"].get("monitoring") or {}).get("columns") else ""),
    ]
    gate = ev["final"].get("gate_g4")
    return WorkflowRun(steps=steps, gate=GateResult.read(gate) if gate else None)


def _rule_ok(final_ev: dict[str, Any]) -> bool:
    return any(c.get("name") == "A rule to act on" and c.get("ok") for c in (final_ev.get("gate_g4") or {}).get("checks") or [])


def view(store: Any, project: dict[str, Any]) -> dict[str, Any]:
    run = build(store, project)
    return {"run": run.model_dump(mode="json"), "titles": TITLES, "order": list(STEPS),
            "done": sum(s.status == "done" for s in run.steps), "missing": [s.id for s in run.steps if s.status == "missing"]}


__all__ = ["g4_of", "machine_monitoring", "build", "view"]
