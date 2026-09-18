"""Single LangGraph loop owner with durable checkpoints and bounded tools."""
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .catalog import ROOT
from .schemas import Experiment
from .store import Store
from .agents import ResearchPlanner, DataScientist, ExperimentDesigner, ResultsCritic, KnowledgeCurator, AuditedClient, ask

class ResearchState(TypedDict, total=False):
    config: dict
    profiles: list
    agenda: dict
    data_review: dict
    proposal: dict
    trials: list
    critiques: list
    synthesis: dict
    memory: list

def compact(trial):
    if trial["status"] != "completed": return trial
    r = trial["result"]
    return {"id": trial["id"], "status": trial["status"], "plan": r["plan"], "metrics": r["metrics"], "fold_standard_deviation": r["fold_standard_deviation"], "paired_comparison": trial.get("paired_comparison"), "input_sensitivity": r["input_sensitivity"][:8], "stress_tests": r["stress_tests"], "output_calibration": r["output_calibration"], "wall_seconds": r["wall_seconds"], "limitations": r["limitations"]}

def signature(plan):
    return hashlib.sha256(json.dumps({k: plan[k] for k in ("dataset", "model", "parameters", "features", "drop_columns", "stress_columns")}, sort_keys=True).encode()).hexdigest()

def check_citations(ids, trials, required=False):
    allowed = {t["id"] for t in trials if t["status"] == "completed"}
    if (required and not ids) or not set(ids) <= allowed:
        raise ValueError("Agent output has missing or unrecognized successful evidence references")

async def worker(request, timeout=300):
    python = Path(os.environ.get("DCLAB_ML_PYTHON", str(ROOT / ".venv/bin/python")))
    if not python.exists(): raise RuntimeError("ML Python is missing; set DCLAB_ML_PYTHON")
    # Whitelist child env; notably no OPENAI_API_KEY, proxy credentials, or tokens.
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TMPDIR", "LANG") if k in os.environ}
    env.update({"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    proc = await asyncio.create_subprocess_exec(str(python), "-m", "dclab_rnd.agentic.worker", cwd=ROOT, env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(json.dumps(request).encode()), timeout)
        if proc.returncode:
            # Worker is trusted deterministic code; no API credentials in its env.
            raise RuntimeError(stderr.decode(errors="replace")[-2000:])
        return json.loads(stdout)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

def build_graph(store, run_id, client, checkpointer, tool=worker, ask_fn=ask):
    async def phase(name, cls, method, context):
        store.update(run_id, phase=name)
        store.event(run_id, "phase", {"name": name})
        value = await ask_fn(cls, method, client, context)
        return value
    def context(state):
        trials = state.get("trials", [])
        # Most recent 12 plus best earlier candidate for each dataset keeps
        # context bounded without losing each dataset's reference candidate.
        keep = trials[-12:]
        for dataset in state["config"]["datasets"]:
            completed = [t for t in trials if t["status"] == "completed" and t["result"]["dataset"] == dataset]
            if completed:
                best = max(completed, key=lambda t: t["result"]["metrics"]["roc_auc"])
                if best not in keep: keep.insert(0, best)
        return {**{k: v for k, v in state.items() if k not in ("trials", "critiques", "proposal")}, "trials": [compact(t) for t in keep], "critique": state.get("critiques", [])[-1:], "attempted": len(trials), "remaining": state["config"]["max_experiments"] - len(trials), "successful_evidence_ids": [t["id"] for t in trials if t["status"] == "completed"]}
    async def profile(state):
        store.update(run_id, phase="Profile public data")
        profiles = await tool({"action": "profile", "datasets": state["config"]["datasets"], "max_rows": state["config"]["max_rows"]})
        store.event(run_id, "profiles", profiles)
        return {"profiles": profiles}
    async def plan(state):
        agenda = await phase("Research planner", ResearchPlanner, "plan", context(state))
        store.event(run_id, "agenda", agenda)
        return {"agenda": agenda}
    async def review(state):
        value = await phase("Data & leakage review", DataScientist, "review", context(state))
        store.event(run_id, "data_review", value)
        return {"data_review": value}
    async def propose(state):
        value = await phase("Design next experiment", ExperimentDesigner, "propose", context(state))
        Experiment.model_validate(value)
        store.event(run_id, "proposal", value)
        return {"proposal": value}
    async def execute(state):
        index = len(state.get("trials", [])) + 1
        evidence_id = f"{run_id}:trial-{index:03d}"
        plan = state["proposal"]
        directory = store.home / run_id / f"trial-{index:03d}"
        store.update(run_id, phase=f"Execute experiment {index}")
        store.event(run_id, "tool_start", {"id": evidence_id, "plan": plan})
        trial = {"id": evidence_id, "status": "failed", "plan": plan}
        try:
            if plan["dataset"] not in state["config"]["datasets"]:
                raise ValueError("Dataset is outside this run's selected scope")
            check_citations(plan["evidence_ids"], state.get("trials", []))
            if any(signature(plan) == signature(t["plan"]) for t in state.get("trials", [])):
                raise ValueError("Duplicate experiment; propose a discriminating change")
            # Durable tool result is idempotent if interruption occurred after
            # execution but before LangGraph committed the node checkpoint.
            completed = directory / "result.json"
            if completed.exists():
                result = json.loads(completed.read_text())
                if result["plan"] != plan: raise ValueError("Immutable trial conflict")
            else:
                result = await tool({"action": "experiment", "plan": plan, "max_rows": state["config"]["max_rows"], "repeats": state["config"]["repeats"], "output": str(directory)})
            trial.update(status="completed", result=result, artifact_directory=str(directory))
            references = [t for t in state.get("trials", []) if t["status"] == "completed" and t["result"]["dataset"] == result["dataset"] and t["result"]["split_hash"] == result["split_hash"] and t["result"]["data_hashes"] == result["data_hashes"]]
            if references:
                baseline = max(references, key=lambda t: t["result"]["metrics"]["roc_auc"])
                deltas = [a["roc_auc"] - b["roc_auc"] for a, b in zip(result["folds"], baseline["result"]["folds"])]
                trial["paired_comparison"] = {"reference_id": baseline["id"], "auc_deltas": deltas, "mean_auc_delta": sum(deltas) / len(deltas), "interpretation": "Descriptive paired development folds, not statistical significance; reference chosen adaptively."}
        except (ValueError, RuntimeError, asyncio.TimeoutError) as exc:
            trial["error"] = str(exc)[:2200]
        store.event(run_id, "trial", trial)
        return {"trials": state.get("trials", []) + [trial]}
    async def critique(state):
        value = await phase("Critique measured evidence", ResultsCritic, "critique", context(state))
        check_citations(value["evidence_ids"], state["trials"])
        store.event(run_id, "critique", value)
        return {"critiques": state.get("critiques", []) + [value]}
    def route(state):
        if len(state["trials"]) >= state["config"]["max_experiments"] or not state["critiques"][-1]["continue_research"]: return "synthesize"
        return "propose"
    async def synthesize(state):
        value = await phase("Curate research memory", KnowledgeCurator, "synthesize", context(state))
        for lesson in value["lessons"]: check_citations(lesson["evidence_ids"], state["trials"], required=True)
        store.event(run_id, "synthesis", value)
        directory = store.home / run_id
        directory.mkdir(exist_ok=True)
        (directory / "trajectory.json").write_text(json.dumps(store.export(run_id), indent=2))
        return {"synthesis": value}
    graph = StateGraph(ResearchState)
    for name, fn in [("profile", profile), ("plan", plan), ("review", review), ("propose", propose), ("execute", execute), ("critique", critique), ("synthesize", synthesize)]: graph.add_node(name, fn)
    for source, target in [(START, "profile"), ("profile", "plan"), ("plan", "review"), ("review", "propose"), ("propose", "execute"), ("execute", "critique"), ("synthesize", END)]: graph.add_edge(source, target)
    graph.add_conditional_edges("critique", route)
    return graph.compile(checkpointer=checkpointer)

async def run_research(store: Store, run_id: str, resume=False):
    run = store.get(run_id)
    started = time.monotonic()
    remaining = run["config"]["max_minutes"] * 60 - run["active_seconds"]
    if remaining <= 0:
        store.update(run_id, status="budget_exhausted", error="Active wall-time budget exhausted")
        return
    store.update(run_id, status="running", error=None)
    try:
        client = AuditedClient(run["config"]["model"], store, run_id)
        async with AsyncSqliteSaver.from_conn_string(str(store.home / "checkpoints.sqlite3")) as saver:
            graph = build_graph(store, run_id, client, saver)
            config = {"configurable": {"thread_id": run_id}, "recursion_limit": 250}
            snapshot = await graph.aget_state(config)
            initial = None if resume and snapshot.values else {"config": run["config"], "trials": [], "critiques": [], "memory": store.knowledge(run["config"]["datasets"])[:12]}
            await asyncio.wait_for(graph.ainvoke(initial, config), timeout=remaining)
        store.update(run_id, status="completed", phase="Research complete")
    except asyncio.CancelledError:
        store.update(run_id, status="paused", phase="Paused at durable checkpoint")
        raise
    except asyncio.TimeoutError:
        store.update(run_id, status="budget_exhausted", error="Active wall-time budget exhausted; evidence retained")
    except Exception as exc:
        # Provider exceptions can contain sensitive headers or requests: persist
        # only the exception type. Never serialize credential-bearing objects.
        store.update(run_id, status="failed", error=f"{type(exc).__name__}: agent request failed; check model access, API quota, connectivity or schema. Completed evidence and checkpoints retained.")
        store.event(run_id, "failure", {"type": type(exc).__name__})
    finally:
        store.update(run_id, active_seconds=run["active_seconds"] + time.monotonic() - started)
        directory = store.home / run_id
        directory.mkdir(exist_ok=True)
        (directory / "trajectory.json").write_text(json.dumps(store.export(run_id), indent=2))
