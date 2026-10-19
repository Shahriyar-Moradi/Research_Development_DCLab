"""The Workflow Run and G4 (package 30.15, the owner's framework, phase 4): the twelve fixed steps a solution goes through,
each with its check, its status and where its evidence is, and the gate that closes it: is the result honest and usable.

The twelve steps are fixed: lock the split, clean, a baseline, the features, the candidates, evaluate them on identical
folds, select by the rule, calibrate and set the operating rule, read the holdout once, package, report, and plan the
monitoring. A host maps them onto its own records (``RunStep.evidence`` is a record path); a step it does not have is
missing, never assumed done.

``GateResult`` is any gate's verdict with where a failure sends the work back (the problem, the data, the brief or the
workflow) and why; a host follows it. ``g4`` reads the facts of one run: the holdout read once, the gap between the
folds and the holdout, a score too good to be honest, the success bar of the signed brief, the operating rule and its
calibration, and the monitoring plan. G4 passes only when the result is honest (the holdout once, no gap the folds do
not explain, nothing too good) and usable (the bar met when one was signed, a rule to act on, a plan to watch it).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, MAX_TEXT, Contract, Versioned, _version

STEPS = ("lock_split", "clean", "baseline", "features", "candidates", "evaluate", "select", "calibrate_rule", "holdout_once",
         "package", "report", "plan_monitoring")
TITLES = {"lock_split": "Lock the split", "clean": "Clean", "baseline": "A baseline", "features": "The features", "candidates": "The candidates",
          "evaluate": "Evaluate on identical folds", "select": "Select by the rule", "calibrate_rule": "Calibrate and set the rule",
          "holdout_once": "Read the holdout once", "package": "Package", "report": "Report", "plan_monitoring": "Plan the monitoring"}
PHASES = ("problem", "data", "brief", "workflow")  # where a failed gate sends the work back: the framework's four phases
Status = Literal["done", "partial", "missing", "not_applicable"]
GAP_BOUND = 0.10  # the folds' score above the holdout's by more than this (and beyond the holdout's interval): not explained
UNIT_FLOOR = 0.1  # a metric on a 0 to 1 scale is read against at least this much of its value: AP 0.09 then 0.01 is not explained
UNIT = {"roc_auc", "average_precision", "accuracy", "macro_f1", "f1", "precision", "recall", "recall_at_budget", "precision_at_k", "r2",
        "ndcg_at_k", "incident_recall", "planted_recall_at_budget"}  # on a 0 to 1 scale: the gap is read as is; any other, relative to the folds'
TOO_GOOD = {"roc_auc": 0.995, "average_precision": 0.995, "accuracy": 0.999, "macro_f1": 0.995, "r2": 0.999}


class RunStep(Contract):
    id: Literal["lock_split", "clean", "baseline", "features", "candidates", "evaluate", "select", "calibrate_rule", "holdout_once",
                "package", "report", "plan_monitoring"]
    status: Status
    check: str = Field(max_length=MAX_TEXT, description="What was checked before the next step.")
    evidence: Optional[str] = Field(default=None, max_length=300, description="Where the step's evidence is: a record path.")
    why: str = Field(default="", max_length=MAX_TEXT)


class GateResult(Versioned):
    """A gate's verdict: passed, or where the work goes back and why."""

    schema_version: Literal["dclab_core.gate_result.v1"] = _version("gate_result")
    gate: Literal["G1", "G2", "G3", "G4"]
    passed: bool
    back_to: Optional[Literal["problem", "data", "brief", "workflow"]] = None
    reasons: list[str] = Field(default_factory=list, max_length=20)
    checks: list[dict[str, Any]] = Field(default_factory=list, max_length=20, description="Each check: name, ok, detail.")

    @model_validator(mode="after")
    def _back(self) -> "GateResult":
        if self.passed and self.back_to is not None:
            raise ValueError("a gate that passed sends nothing back")
        if not self.passed and self.back_to is None:
            raise ValueError("a gate that failed names the phase the work goes back to")
        return self


CONTRACTS["gate_result"] = GateResult


class WorkflowRun(Versioned):
    """The twelve steps of one run, in their fixed order, and G4."""

    schema_version: Literal["dclab_core.workflow_run.v1"] = _version("workflow_run")
    steps: list[RunStep] = Field(min_length=12, max_length=12)
    gate: Optional[GateResult] = None

    @model_validator(mode="after")
    def _order(self) -> "WorkflowRun":
        if tuple(s.id for s in self.steps) != STEPS:
            raise ValueError("the twelve steps, each once, in the framework's order")
        if self.gate is not None and self.gate.gate != "G4":
            raise ValueError("a run's gate is G4")
        return self


CONTRACTS["workflow_run"] = WorkflowRun


def _finite(value: Any) -> bool:
    import math

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail[:600]}


def g4(*, holdout_reads: int, metric: str, holdout: Optional[float], folds: Optional[float], higher_is_better: bool = True,
       interval: Optional[tuple[float, float]] = None, bar: Optional[dict[str, Any]] = None, rule: Optional[str] = None,
       calibration: Optional[str] = None, monitoring: bool = False, monitoring_why: str = "", gap_bound: float = GAP_BOUND,
       folds_std: Optional[float] = None) -> GateResult:
    """G4 on one run's facts. ``bar``: the brief's verdict (``met``, ``not_met``, ``cannot_judge``) when a brief was signed;
    ``rule``: the operating rule acted on, None when there is none; ``calibration``: what the step did (fitted, kept, not
    applicable); ``monitoring``: whether the artifact carries a monitoring plan (``monitoring_why`` when it cannot);
    ``folds_std``: the chosen model's spread across the folds, when the host has it.

    The folds explain the holdout when the gap is within ``gap_bound`` of the folds' value (on a 0 to 1 scale never below
    a tenth of the bound, so a score near zero is still read against something), within twice the folds' spread, or the
    folds' value lies inside the holdout's interval."""
    holdout = holdout if _finite(holdout) else None  # a value that is not a number is unread, never "too good"
    folds = folds if _finite(folds) else None
    checks, failed = [], []  # each failure: (phase, reason), the earliest phase wins

    def need(name: str, ok: bool, detail: str, phase: str) -> None:
        checks.append(_check(name, ok, detail))
        if not ok:
            failed.append((phase, f"{name}: {detail}"))

    need("The holdout read once", holdout_reads == 1, f"read {holdout_reads} time(s)" if holdout_reads != 1 else "read once", "workflow")
    if holdout is not None and folds is not None:
        gap = (folds - holdout) if higher_is_better else (holdout - folds)
        tolerance = max(gap_bound * max(abs(folds), UNIT_FLOOR if metric in UNIT else 0.0), 2 * float(folds_std) if _finite(folds_std) else 0.0)
        inside = interval is not None and (interval[0] <= folds <= interval[1])
        ok = gap <= tolerance or inside
        need("The folds explain the holdout", ok, f"the folds' {metric} {folds:.4g}, the holdout's {holdout:.4g}: a gap of {gap:.4g}"
             + ("" if ok else f", above what the folds allow ({tolerance:.4g})" + (" and outside the holdout's interval" if interval else "")
                + ": the split or the rows changed between them"), "workflow")
    ceiling = TOO_GOOD.get(metric)
    if holdout is not None and ceiling is not None and higher_is_better:
        need("Not too good to be honest", holdout < ceiling, f"the holdout's {metric} {holdout:.4g}" + ("" if holdout < ceiling else
             f" reaches {ceiling}: a column likely gives the answer away (leakage)"), "data")
    if bar is not None:
        verdict = str(bar.get("verdict") or "")
        need("The success bar", verdict == "met", str(bar.get("reason") or verdict)[:500], "brief")
    need("A rule to act on", bool(rule), rule or "no operating rule was set", "workflow")
    if calibration is not None:
        checks.append(_check("Calibration", True, calibration))
    need("A plan to watch it", monitoring, "the artifact carries the monitoring plan" if monitoring else (monitoring_why or "no monitoring plan"), "workflow")
    if not failed:
        return GateResult(gate="G4", passed=True, checks=checks)
    back = min((p for p, _ in failed), key=PHASES.index)
    return GateResult(gate="G4", passed=False, back_to=back, reasons=[r for _, r in failed][:20], checks=checks)


__all__ = ["STEPS", "TITLES", "PHASES", "GAP_BOUND", "UNIT", "TOO_GOOD", "RunStep", "GateResult", "WorkflowRun", "g4"]
