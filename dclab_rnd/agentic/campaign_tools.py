"""The research campaign's one write, an experiment, as a registered tool behind a guard (package A2.4).

The campaign agent (``engine.py``) is a LangGraph loop of NOOA specialists that answer in typed objects, not tool
calls, so its planning steps cannot run on ``agents.run()``. What it does to the world is one thing: run an
experiment. That move is a tool in the shared registry (scope "campaign"), and the campaign guard checks it before
it runs, as the graph validator checks a project write: the dataset is in the run's scope, every cited evidence id is
a successful trial, a paired reference is cited, and the same configuration was not tried before. Each phase and each
experiment is written to the agent trace.

This module imports no table library: the campaign runs in its own environment (``.venv-agent``, no pandas).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from dclab_rnd.agents.registry import Registry, Tool

from .schemas import Experiment


def signature(plan: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: plan[k] for k in ("dataset", "model", "parameters", "features", "drop_columns", "stress_columns")}, sort_keys=True).encode()).hexdigest()


def check_citations(ids: list[str], trials: list[dict[str, Any]], required: bool = False) -> None:
    allowed = {t["id"] for t in trials if t["status"] == "completed"}
    if (required and not ids) or not set(ids) <= allowed:
        raise ValueError("Agent output has missing or unrecognized successful evidence references")


@dataclass
class CampaignTurn:
    """What the experiment tool acts on: the campaign state, the worker that runs experiments, and the trial folder."""
    state: dict[str, Any]
    tool: Callable[[dict[str, Any]], Awaitable[Any]]
    directory: Path
    extra: dict[str, Any] = field(default_factory=dict)


class CampaignGuard:
    """Checks an experiment before it runs; a refusal is the error the trial records (the same words as before)."""

    def __call__(self, tool: Tool, turn: CampaignTurn, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        plan, trials = arguments, turn.state.get("trials", [])
        if plan["dataset"] not in turn.state["config"]["datasets"]:
            return {"error": "Dataset is outside this run's selected scope"}
        try:
            check_citations(plan["evidence_ids"], trials)
        except ValueError as error:
            return {"error": str(error)}
        if plan.get("reference_evidence_id") and plan["reference_evidence_id"] not in plan["evidence_ids"]:
            return {"error": "Paired reference must also appear in evidence_ids"}
        if any(signature(plan) == signature(t["plan"]) for t in trials):
            return {"error": "Duplicate experiment; propose a discriminating change"}
        return proceed()


async def run_experiment(turn: CampaignTurn, /, **plan: Any) -> dict[str, Any]:
    # A durable result is reused if the run was interrupted after the experiment but before LangGraph's checkpoint.
    completed = turn.directory / "result.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result["plan"] != plan:
            raise ValueError("Immutable trial conflict")
        return result
    return await turn.tool({"action": "experiment", "plan": plan, "max_rows": turn.state["config"]["max_rows"],
                            "repeats": turn.state["config"]["repeats"], "output": str(turn.directory)})


def _expected(error: Exception) -> dict[str, Any] | None:
    if isinstance(error, (ValueError, RuntimeError, asyncio.TimeoutError)):
        return {"error": str(error)[:2200]}
    return None


def register(registry: Registry) -> None:
    """Register the campaign's experiment tool (scope "campaign") and its guard."""
    registry.guards["campaign"] = CampaignGuard()
    registry.register(Tool("run_experiment", "Run one bounded experiment from a validated plan on the deterministic worker.",
                           Experiment.model_json_schema(), run_experiment, effect="write", scope="campaign", move="experiment",
                           takes_context=True, errors=_expected))
