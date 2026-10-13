"""
workflow_map.py
===============

One unified workflow for every sample: the repository's WF-01…WF-10 graph (``dclab_rnd.studio.graph``).
Each stage of the factory is one step of it, every sample's input starts with its step, and every answer's NEXT
carries a ROUTE:

    continue → WF-xx      the step is done; go to the next one
    back → WF-xx          the evidence reopens an earlier step (only the moves the graph allows, with its reason)
    ask_owner → WF-xx     stop and ask; the step waits for the answer

The single-step samples keep their exact format (one decision, the reasoning, the rule, a confidence, the next
step); only the step header and the route are added. Trajectory generators compute their routes themselves with
the context they have (for example a leak the plan did not know); ``annotate`` routes the single-step samples.

    WF-01 Solution draft          problem_understanding
    WF-02 Source and lineage      contract_check
    WF-03 Split design            data_understanding (pre-split checks: duplicates, balance, size, missing values)
    WF-04 Train-only EDA          data_analysis, then solution_design (closes with the solution gate)
    WF-05 Leakage audit           leakage_audit
    WF-06 Feature ladder          feature_engineering
    WF-07 Algorithm screen        model_selection
    WF-08 Conservative optimization   calibration
    WF-09 Reliability challenge   final holdout, scored once (the holdout gate)
    WF-10 Knowledge capture       final report
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from dclab_rnd.studio import graph as G  # noqa: E402

NAME = {n["id"]: n["name"] for n in G.NODES}
REVISITS = dict(G.REVISITS)  # (from, to) -> why; plus any step -> WF-01 "reopen the solution"

STEP_OF_STAGE = {
    "problem_understanding": "WF-01",
    "contract_check": "WF-02",
    "data_understanding": "WF-03",
    "data_analysis": "WF-04",
    "solution_design": "WF-04",
    "leakage_audit": "WF-05",
    "feature_engineering": "WF-06",
    "feature_engineering_balance": "WF-06",
    "model_selection": "WF-07",
    "calibration": "WF-08",
    "holdout": "WF-09",
    "report": "WF-10",
}


def step_label(step: str) -> str:
    return f"{step} {NAME[step]}"


def header(step: str) -> str:
    return f"WORKFLOW STEP: {step_label(step)}\n"


def route(action: str, to: str, why: str = "") -> dict:
    assert action in ("continue", "back", "ask_owner"), action
    assert to in NAME, to
    if action == "back":
        frm_ok = to == "WF-01" or any(b == to for (_a, b) in REVISITS)
        assert frm_ok, f"the graph has no revisit to {to}"
    return {"action": action, "to": step_label(to), **({"why": why} if why else {})}


def route_text(r: dict) -> str:
    arrow = {"continue": "continue →", "back": "back →", "ask_owner": "ask_owner →"}[r["action"]]
    return f"[{arrow} {r['to'].split()[0]}]"


def _holdout_step(stage: str, user: str) -> bool:
    return stage == "optimization_reliability" and "final holdout" in user


def stage_step(sample: dict) -> tuple[str, str]:
    """(stage, step) for a single-step sample, telling calibration from the holdout check."""
    md = sample["metadata"]
    stage = md.get("stage", "leakage_audit")
    user = sample["messages"][1]["content"]
    if stage == "optimization_reliability":
        stage = "holdout" if _holdout_step(stage, user) else "calibration"
    return stage, STEP_OF_STAGE[stage]


def single_route(stage: str, decision: str, answer: dict | None = None) -> dict:
    """The route for a one-decision sample, from its stage and decision alone."""
    if stage == "problem_understanding":
        unknown = set((answer or {}).get("unknown", []))
        if {"target_column", "prediction_moment"} & unknown:
            return route("ask_owner", "WF-01", "the target or the prediction moment is unknown; the open questions must be answered first")
        return route("continue", "WF-02")
    if stage == "contract_check":
        return {"proceed": route("continue", "WF-03"),
                "fix_contract": route("back", "WF-01", "reopen the solution: the data corrects an inferred fact"),
                "ask_owner": route("ask_owner", "WF-02", "the data contradicts what the owner stated, or the target is unusable")}[decision]
    if stage == "data_understanding":
        return route("continue", "WF-04")
    if stage == "data_analysis":
        return route("continue", "WF-04", "the findings go to the solution design")
    if stage == "solution_design":
        return route("continue", "WF-05", "after the solution gate")
    if stage == "leakage_audit":
        return route("continue", "WF-05", "the audit continues with the next column")
    if stage in ("feature_engineering", "feature_engineering_balance"):
        return route("continue", "WF-07")
    if stage == "model_selection":
        return route("continue", "WF-08")
    if stage == "calibration":
        return route("continue", "WF-09")
    if stage == "holdout":
        if decision == "investigate_distribution_shift":
            return route("back", "WF-03", REVISITS[("WF-09", "WF-03")] + "; the spent holdout is reported, never reused or tuned on")
        return route("continue", "WF-10")
    raise ValueError(stage)


def with_route_text(answer: str, r: dict) -> str:
    """Put the route at the start of the NEXT line of a DECISION answer."""
    assert "\nNEXT: " in answer
    head, nxt = answer.rsplit("\nNEXT: ", 1)
    return f"{head}\nNEXT: {route_text(r)} {nxt}"


def with_route_json(answer: str, r: dict) -> str:
    obj = json.loads(answer)
    obj["route"] = r
    return json.dumps(obj, ensure_ascii=False, default=str)


CONNECTED_STAGES = {"connected_problem_to_plan": ["problem_understanding", "data_analysis", "solution_design"]}


def _route_answer(stage: str, ans: str) -> tuple[str, dict]:
    if ans.lstrip().startswith("{"):
        obj = json.loads(ans)
        r = single_route(stage, obj.get("decision", stage), obj)
        return with_route_json(ans, r), r
    r = single_route(stage, re.search(r"DECISION: (\S+)", ans).group(1))
    return with_route_text(ans, r), r


def annotate(sample: dict) -> dict:
    """Step header on every input, route on every answer. Single-step samples and the connected front-of-workflow
    conversations are handled here; trajectory samples (routed by their generator) are returned unchanged."""
    md = sample["metadata"]
    if md.get("routed"):
        return sample
    msgs = sample["messages"]
    if md.get("stage") in CONNECTED_STAGES:
        stages = CONNECTED_STAGES[md["stage"]]
    else:
        assert len(msgs) == 3, "a multi-turn sample needs its stage list"
        stages = [stage_step(sample)[0]]
    out, i, routes = [msgs[0]], 0, []
    for m in msgs[1:]:
        if m["role"] == "user":
            out.append({"role": "user", "content": header(STEP_OF_STAGE[stages[i]]) + m["content"]})
        else:
            ans, r = _route_answer(stages[i], m["content"])
            out.append({"role": "assistant", "content": ans})
            routes.append(r["action"])
            i += 1
    assert i == len(stages)
    return {"messages": out, "metadata": {**md, "step": STEP_OF_STAGE[stages[-1]], "route": routes[-1], "routed": True}}
