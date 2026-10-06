"""Single LangGraph loop owner with durable checkpoints and bounded tools."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .catalog import ROOT
from .projects import historical_context
from .schemas import Experiment
from .store import Store
from .agents import ResearchPlanner, DataScientist, ExperimentDesigner, ResultsCritic, KnowledgeCurator, AuditedClient, ask
from .campaign_tools import CampaignTurn, check_citations, signature  # noqa: F401 — signature and check_citations are imported from here too
from ..agents import registry_of
from ..agents.traces import Tracer, open_traces

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
    historical_evidence: list

def compact(trial):
    if trial["status"] != "completed": return trial
    r = trial["result"]
    return {"id": trial["id"], "status": trial["status"], "plan": r["plan"],
            "metrics": r["metrics"], "repeat_summary": r.get("repeat_summary", []),
            "fold_standard_deviation": r["fold_standard_deviation"],
            "paired_comparison": trial.get("paired_comparison"),
            "retained_columns": r.get("retained_columns", []),
            "pipeline_source_columns": r.get("pipeline_source_columns", []),
            "derived_feature_lineage": r.get("derived_feature_lineage", []),
            "split_hash": r.get("split_hash"), "data_hashes": r.get("data_hashes", {}),
            "input_sensitivity": r["input_sensitivity"][:8], "stress_tests": r["stress_tests"],
            "output_calibration": r["output_calibration"],
            "confusion_matrix_at_0_5_by_repeat": r.get("confusion_matrix_at_0_5_by_repeat", []),
            "evaluation": r.get("evaluation"), "wall_seconds": r["wall_seconds"],
            "limitations": r["limitations"]}

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

def build_graph(store, run_id, client, checkpointer, tool=worker, ask_fn=ask, traces=None):
    """The campaign as a LangGraph loop. Its specialists answer in typed objects (no tool calls), so their phases are
    not runtime replies; its one write, an experiment, is the registered tool ``run_experiment`` behind the campaign
    guard (package A2.4). Each phase and each experiment is a row of the agent trace (``traces``, under the run id)."""
    registry = registry_of("campaign")
    seen = {"trials": 0}  # the state a row records: how many trials the phase saw
    tracer = Tracer.resume(traces, run_id, "campaign", lambda: f"{seen['trials']} trials") if traces is not None else None

    def trace(name, arguments, value, started, trials):
        if tracer is not None:
            seen["trials"] = trials
            tracer(name, arguments, value, time.monotonic() - started)

    async def phase(name, cls, method, context):
        store.update(run_id, phase=name)
        store.event(run_id, "phase", {"name": name})
        started = time.monotonic()
        try:
            value = await ask_fn(cls, method, client, context)
        except Exception as exc:
            trace(method, {}, {"error": type(exc).__name__}, started, context.get("attempted", 0))
            raise
        trace(method, {}, value, started, context.get("attempted", 0))
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
        started = time.monotonic()
        try:
            # The registry checks the plan against its schema, the campaign guard checks scope, citations and
            # duplicates, then the experiment runs; a refusal or an expected failure comes back as an error.
            result = await registry.acall("run_experiment", plan, CampaignTurn(state, tool, directory), scope="campaign")
            trace("run_experiment", plan, result, started, len(state.get("trials", [])))
            if isinstance(result, dict) and set(result) <= {"error", "status"} and "error" in result:
                raise ValueError(result["error"])
            trial.update(status="completed", result=result, artifact_directory=str(directory))
            references = [t for t in state.get("trials", []) if t["status"] == "completed" and t["result"]["dataset"] == result["dataset"] and t["result"]["split_hash"] == result["split_hash"] and t["result"]["data_hashes"] == result["data_hashes"]]
            if references:
                explicit = [t for t in references if t["id"] == plan.get("reference_evidence_id")]
                cited = [t for t in references if t["id"] in plan["evidence_ids"]]
                if explicit:
                    baseline = explicit[0]
                elif cited:
                    # Backward-compatible fallback for plans made before the
                    # explicit reference field: isolate one variable where possible.
                    baseline = max(cited, key=lambda t: (t["plan"]["model"] == plan["model"], t["plan"]["parameters"] == plan["parameters"]))
                else:
                    baseline = max(references, key=lambda t: t["result"]["metrics"]["roc_auc"])
                metrics = result["metrics"]
                fold_deltas = {metric: [a[metric] - b[metric] for a, b in zip(result["folds"], baseline["result"]["folds"])] for metric in metrics}
                trial["paired_comparison"] = {"reference_id": baseline["id"], "fold_deltas": fold_deltas,
                    "mean_deltas": {metric: sum(values) / len(values) for metric, values in fold_deltas.items()},
                    "interpretation": "Descriptive paired development folds, not statistical significance; cited compatible reference preferred."}
        except (ValueError, RuntimeError, asyncio.TimeoutError) as exc:
            trial["error"] = str(exc)[:2200]
        # Durable META sidecar beside worker files (archive also rebuilds this).
        try:
            from .archive import write_trial_meta

            directory.mkdir(parents=True, exist_ok=True)
            write_trial_meta(directory, trial)
        except Exception:  # noqa: BLE001 — never fail the research loop on sidecar IO
            pass
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
        return {"synthesis": value}
    graph = StateGraph(ResearchState)
    for name, fn in [("profile", profile), ("plan", plan), ("review", review), ("propose", propose), ("execute", execute), ("critique", critique), ("synthesize", synthesize)]: graph.add_node(name, fn)
    for source, target in [(START, "profile"), ("profile", "plan"), ("plan", "review"), ("review", "propose"), ("propose", "execute"), ("execute", "critique"), ("synthesize", END)]: graph.add_edge(source, target)
    graph.add_conditional_edges("critique", route)
    return graph.compile(checkpointer=checkpointer)

def campaign_traces(home):
    """The trace store for campaigns: the database when one is configured and its driver is installed, otherwise a file
    beside the runs (the campaign's environment, .venv-agent, has no database libraries)."""
    try:
        return open_traces(home)
    except ImportError:
        from ..agents.traces import FileTraces
        return FileTraces(Path(home) / "agent_steps.jsonl")

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
            graph = build_graph(store, run_id, client, saver, traces=campaign_traces(store.home))
            config = {"configurable": {"thread_id": run_id}, "recursion_limit": 250}
            snapshot = await graph.aget_state(config)
            project = run["config"].get("project", "general")
            initial = None if resume and snapshot.values else {"config": run["config"], "trials": [], "critiques": [], "memory": store.knowledge(run["config"]["datasets"])[:12], "historical_evidence": historical_context(project)}
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
        store.finalize_archive(run_id)
