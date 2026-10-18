"""The debate (package 28.6, D5 of the phase 6 plan): two solution designers and a judge.

The first designer is the solution planner (``dclab_core.plan.SOLUTION_PLANNER``): it proposes changes to the rules' plan,
each citing measured findings. The second, the challenger, reads the same evidence and the first design, and proposes the
changes a careful second practitioner would make where the first missed something (an entity repeated across rows, a column
written after the prediction moment). A change both make is kept; a change only one makes, or two different choices for one
decision, go to the judge, who keeps one side or neither with a reason citing the findings. Every kept change passes the
plan's code checks again (the planner's guard and validator: an advisor may add a column to keep out, never bring one back;
a split the findings call for is never dropped), so no model writes the plan unchecked.

Without a judge (none configured, it abstained, it failed, or its answer was refused) every dispute keeps the first
designer's side: the debate is then the single designer, never worse for want of a model. Framework-free: the host gives the
three advisors; the record holds names, codes and the designers' reasons, never a row.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

SIDES = ("a", "b", "neither")
MAX_CHANGES = 13  # len(dclab_core.plan.PLANNABLE), a test keeps them equal: ``plan`` is read when used, never at import (the review of 28.6)


def _plan() -> Any:
    from dclab_core import plan as P

    return P


class Change(Contract):
    key: str = Field(max_length=80)
    choice: Any = None
    reason: str = Field(default="", max_length=400)
    cites: list[str] = Field(default_factory=list, max_length=10)


class Ruling(Contract):
    key: str = Field(max_length=80)
    keep: Literal["a", "b", "neither"]
    reason: str = Field(max_length=600)
    by: Literal["judge", "fallback"]


class Debate(Versioned):
    """One debate over a plan: what each designer changed, what they agreed on, how each dispute was ruled, what was kept."""

    schema_version: Literal["dclab_core.debate.v1"] = _version("debate")
    a: list[Change] = Field(default_factory=list, max_length=MAX_CHANGES)
    b: list[Change] = Field(default_factory=list, max_length=MAX_CHANGES)
    agreed: list[str] = Field(default_factory=list, max_length=MAX_CHANGES, description="Decision keys both designers changed the same way.")
    rulings: list[Ruling] = Field(default_factory=list, max_length=MAX_CHANGES)
    kept: list[Change] = Field(default_factory=list, max_length=MAX_CHANGES, description="The changes in the plan, after the code checks.")
    refused: list[dict[str, Any]] = Field(default_factory=list, max_length=4 * MAX_CHANGES)
    sources: dict[str, str] = Field(default_factory=dict, description="Each seat's source: advisor, or fallback with why.")


CONTRACTS["debate"] = Debate


def _same(x: Any, y: Any) -> bool:
    return json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)


# ------------------------------------------------------------------------------------------------ the challenger

def _summary(change: Mapping[str, Any]) -> dict[str, Any]:
    """A designer's change as the evidence carries it: the choice as compact text, as the rules' plan is (the evidence
    check refuses a choice nested as a table)."""
    P = _plan()
    return {"key": str(change.get("key")), "choice": P._compact(P._plain_choice(change.get("choice")), 300),
            "reason": str(change.get("reason") or "")[:300], "cites": [str(c) for c in change.get("cites") or []][:10]}


def _challenger_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {**_plan()._planner_evidence(inputs), "first_design": [_summary(c) for c in inputs.get("first") or []]}


def _challenger_role():
    from dclab_core.agents import Role

    return Role(name="solution_challenger",
                description="Reads the brief, the findings, the rules' plan and a first design, and proposes the changes the first "
                            "design missed, each citing measured findings; checked as the planner's are.",
                evidence=_challenger_evidence, parse=lambda answer, inputs: _plan()._planner_parse(answer, inputs),
                fallback=lambda inputs: {"plan": inputs["plan"], "kept": [], "refused": []})


SOLUTION_CHALLENGER = _challenger_role()


# ------------------------------------------------------------------------------------------------ the judge

def _judge_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    base = _plan()._planner_evidence(inputs)
    return {"brief": base["brief"], "task": base["task"], "findings": base["findings"], "rule_plan": base["rule_plan"],
            "disputes": [{"key": k, "a": _plan()._compact(_plan()._plain_choice(inputs["a"].get(k)), 300) if inputs["a"].get(k) is not None else None,
                          "b": _plan()._compact(_plan()._plain_choice(inputs["b"].get(k)), 300) if inputs["b"].get(k) is not None else None}
                         for k in inputs["disputed"]]}


def _judge_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[Ruling]:
    """Each ruling names a dispute, a side, a reason, and cites measured findings; a dispute left unruled keeps side a."""
    items = answer.get("rulings")
    if not isinstance(items, list):
        raise ValueError("rulings must be a list")
    ids = {f.id for f in inputs["findings"].findings}
    out: dict[str, Ruling] = {}
    for item in items:
        if not isinstance(item, Mapping) or item.get("key") not in inputs["disputed"] or item.get("key") in out:
            raise ValueError("a ruling names one dispute, once")
        side, reason, cites = item.get("keep"), str(item.get("reason") or "").strip(), item.get("cites") or []
        if side not in SIDES:
            raise ValueError(f"keep is one of {', '.join(SIDES)}")
        if side == "b" and inputs["b"].get(item["key"]) is None or side == "a" and inputs["a"].get(item["key"]) is None:
            raise ValueError("a ruling keeps a side that proposed this change")
        if len(reason) < 10 or not isinstance(cites, list) or not cites or not set(map(str, cites)) <= ids:
            raise ValueError("a ruling needs a reason and measured findings to cite")
        out[item["key"]] = Ruling(key=item["key"], keep=side, reason=reason[:600], by="judge")
    return [out.get(k) or Ruling(key=k, keep="a" if inputs["a"].get(k) is not None else "neither", reason="not ruled: the first design stands",
                                 by="fallback") for k in inputs["disputed"]]


def _judge_role():
    from dclab_core.agents import Role

    def fallback(inputs: Mapping[str, Any]) -> list[Ruling]:
        return [Ruling(key=k, keep="a" if inputs["a"].get(k) is not None else "neither", reason="no judge: the first design stands",
                       by="fallback") for k in inputs["disputed"]]

    return Role(name="debate_judge",
                description="Rules on each change the two designers dispute: keep the first's, the second's, or neither, citing the "
                            "measured findings; with no judge the first design stands.",
                evidence=_judge_evidence, parse=_judge_parse, fallback=fallback)


DEBATE_JUDGE = _judge_role()


# ------------------------------------------------------------------------------------------------ the debate

def debate(designer: Any, challenger: Any, judge: Any, brief: Any, findings: Any, columns: Any = None) -> dict[str, Any]:
    """The rules' plan, the two designs, the judge's rulings, and the plan with the kept changes after the code checks
    (``plan``), with the ``Debate`` record. Any advisor may be None: that seat falls back. ``columns``: the table's, when
    the host has them: a change naming another column is refused at every seat."""
    from dclab_core.agents import run_role

    P = _plan()
    PLANNABLE = P.PLANNABLE
    base = P.plan(brief, findings)
    inputs = {"brief": brief, "findings": findings, "plan": base, "columns": list(columns) if columns is not None else None}
    first = run_role(P.SOLUTION_PLANNER, designer, inputs)
    a = {c["key"]: c for c in first.output.get("kept") or []}
    second = run_role(SOLUTION_CHALLENGER, challenger, {**inputs, "first": list(a.values())})
    b = {c["key"]: c for c in second.output.get("kept") or []}
    # the challenger restates nothing it accepts (its prompt): a first-design change it left alone is agreed; a dispute is
    # where it chose otherwise, or changed what the first design left (the review of 28.6: silence was read as dispute)
    agreed = [k for k in PLANNABLE if k in a and (k not in b or _same(a[k]["choice"], b[k]["choice"]))]
    disputed = [k for k in PLANNABLE if k in b and k not in agreed]
    choices_a = {k: a[k]["choice"] for k in a}
    choices_b = {k: b[k]["choice"] for k in b}
    ruled = run_role(DEBATE_JUDGE, judge if disputed else None, {**inputs, "a": choices_a, "b": choices_b, "disputed": disputed})
    rulings = list(ruled.output)
    wanted = [a[k] for k in agreed] + [(a if r.keep == "a" else b)[r.key] for r in rulings if r.keep in ("a", "b")]
    order = {k: i for i, k in enumerate(PLANNABLE)}
    final = P._planner_parse({"decisions": sorted(wanted, key=lambda c: order[c["key"]])}, inputs)  # the code checks, again
    record = Debate(a=[Change(**c) for c in a.values()], b=[Change(**c) for c in b.values()], agreed=agreed, rulings=rulings,
                    kept=[Change(**c) for c in final["kept"]], refused=list(final["refused"])[:4 * MAX_CHANGES],
                    sources={"designer": first.source if first.source == "advisor" else f"fallback: {first.reason}"[:200],
                             "challenger": second.source if second.source == "advisor" else f"fallback: {second.reason}"[:200],
                             "judge": (ruled.source if ruled.source == "advisor" else f"fallback: {ruled.reason}"[:200]) if disputed else "no dispute"})
    return {"plan": final["plan"], "kept": final["kept"], "refused": final["refused"], "rule_plan": base, "debate": record,
            "single": first.output}  # ``single``: the first designer's own plan, the baseline the debate is measured against


__all__ = ["Change", "Ruling", "Debate", "SOLUTION_CHALLENGER", "DEBATE_JUDGE", "debate"]
