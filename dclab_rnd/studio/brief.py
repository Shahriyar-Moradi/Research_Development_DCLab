"""A project's Solution Brief (package 30.14): drafted by rules from what the project holds, edited by a person, signed
with the solution (``graph.approve_gate``: refused while G3 has problems), judged at the one holdout read
(``engine._success_bar``) and written into the report.

The draft reads the engine's own preparation (``engine.prepare``: the resolved recipe, the primary metric and its
direction, the split, the columns kept out), the solution's capacity, the problem card the project was built from (today's
process, the business number) and the plan's risks when there is a plan. Nothing here reads the holdout.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core.contracts import Risk
from dclab_core.solution_brief import SolutionBrief, fill, fingerprint, problems, signed

RULE_RISKS = [Risk(statement="a column written after the outcome would make the score look better than it can be (leakage)",
                   check="the leakage stage measures the model with and without each suspect column"),
              Risk(statement="the score may not hold on later rows", check="the holdout is the latest period when a column orders the rows; the "
                                                                           "report says the interval")]


def _card(project: dict[str, Any]) -> dict[str, Any]:
    return ((project.get("draft") or {}).get("card") or {}) if isinstance(project.get("draft"), dict) else {}


def draft(store: Any, project: dict[str, Any]) -> SolutionBrief:
    """A brief by rules for this project (not saved)."""
    from . import engine
    from .machine import MachinePrepared
    from dclab_core.contracts import SolutionPlan

    if not project.get("solution") or not project.get("data"):
        raise ValueError("a brief is drafted from the project's table and its solution: attach the data and save the solution first")
    p = engine.prepare(store, project)
    solution = p.solution
    recipe = getattr(p, "recipe", None) or {}
    parts = [x for x in [recipe.get("kind")] + list(recipe.get("modifiers") or []) if isinstance(x, str)]
    card = _card(project)
    today = ((card.get("baseline_today") or {}).get("value")) if isinstance(card.get("baseline_today"), dict) else None
    business = (((card.get("success") or {}).get("business") or {}).get("value")) if isinstance(card.get("success"), dict) else None
    risks = list(RULE_RISKS)
    if project.get("plan"):
        try:
            risks = list(SolutionPlan.read(project["plan"]).risks)[:12] or risks
        except Exception:  # noqa: BLE001 - a plan that cannot be read leaves the rules' risks
            pass
    capacity = {"k": solution.capacity_k, "period": solution.capacity_period} if solution.capacity_k else None
    excluded = {f.column: f.reason for f in solution.forbidden or []}
    if isinstance(p, MachinePrepared):  # a kind's own machine: its metric, its direction, its own first baseline
        kind = p.kind
        return fill(p.task_type or p.task, p.metric, bool(getattr(p.machine, "higher_is_better", True)), [f"{kind.key}"],
                    p.holdout_description, capacity=capacity, baseline_today=today, business=business, excluded=excluded, risks=risks,
                    simple=(list(kind.baselines) or [""])[0])
    spec = p.bundle.spec
    task = "binary" if p.task_type in ("binary", "binary_imbalanced", "text_tabular_binary") else ("multiclass" if "multiclass" in p.task_type else "regression")
    return fill(task, spec.primary_metric, bool(spec.higher_is_better), parts, str(spec.split_strategy), capacity=capacity, baseline_today=today,
                business=business, excluded=dict(spec.blocked_features or {}), risks=risks, budget=spec.settings.get("alert_budget"))


def view(store: Any, project: dict[str, Any]) -> dict[str, Any]:
    """The stored brief, or a draft when there is none, with what G3 still asks and whether it is signed."""
    stored = project.get("brief")
    brief = SolutionBrief.read(stored) if stored else draft(store, project)
    return {"brief": brief.model_dump(mode="json"), "stored": bool(stored), "problems": problems(brief), "signed": signed(brief),
            "fingerprint": fingerprint(brief), "holdout_read": int(project.get("holdout_uses") or 0) > 0}


def save(store: Any, project_id: str, doc: dict[str, Any], by: str = "owner") -> dict[str, Any]:
    """A person's brief, saved through the graph's ``set_brief`` move (refused while a stage runs, and after the holdout
    was read: a bar moved after the read would be chosen on it). An edit after signing unsigns it: the bar is frozen only
    at a signature, given by the solution gate, never written in an edit."""
    from . import graph

    project = store.get(project_id)
    brief = SolutionBrief.read({**dict(doc), "schema_version": SolutionBrief.model_fields["schema_version"].default})
    verdict = graph.check(project, "set_brief", "human", fingerprint=fingerprint(brief))
    graph.log(store, project_id, verdict, project)
    if not verdict.allowed:
        raise graph.GraphBlocked(verdict)
    before = SolutionBrief.read(project["brief"]) if project.get("brief") else None
    if brief.approval is not None and (before is None or before.approval != brief.approval or not signed(brief)):
        brief = brief.model_copy(update={"approval": None})
    project["brief"] = brief.model_dump(mode="json")
    store.save(project)
    from .. import audit

    audit.record_for(store, "brief", "human", by or "owner", project=project, move="set_brief", status="allowed",
                     args={"fingerprint": fingerprint(brief), "problems": len(problems(brief)), "signed": signed(brief)},
                     message="the Solution Brief saved")
    return view(store, store.get(project_id))


def verdict_of(project: dict[str, Any], store: Optional[Any] = None) -> Optional[dict[str, Any]]:
    """The final stage's verdict against the bar, when it ran with a brief."""
    final = (project.get("stages") or {}).get("final") or {}
    if store is not None and not final.get("evidence"):
        final = store.read_stage(project["id"], "final") or {}
    return (final.get("evidence") or {}).get("success_bar")


__all__ = ["draft", "view", "save", "verdict_of"]
