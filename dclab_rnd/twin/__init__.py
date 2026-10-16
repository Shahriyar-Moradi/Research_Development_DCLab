"""Twin runs (R&D only): the current system and an AI-first Claude agent on the same problem, at the same time,
everything from both kept, side by side.

A twin run takes one problem (a studied sample, a project, or a draft's chat), makes two lanes of it, and runs
both at once in two threads:

* **workflow**: the current system as it is: the rules, the graph validator, the five-stage engine, the standard
  plan (the intern without a model), with no model at all by default (``--workflow-models configured`` gives it the
  workspace's configured model, as the product has it);
* **ai_first**: Claude as the agent, with the same tools, the same validator and the same task: it reads the data,
  writes the solution, runs and approves the stages and reports; the stage notes, the leakage reviewer and the seven
  roles of dclab_core also run on Claude.

Each lane is a complete workspace folder of its own under ``<home>/twin/runs/<run_id>/``, so nothing of a twin run
touches the main workspace: project, stages, audit, moves, tool steps, the ledger of its decisions (19.7), its session
record (19.2), the intern's whole conversation, and for the AI-first lane every model request and answer verbatim
(``transcripts/``). The run folder holds ``run.json`` (the manifest), ``compare.json`` and ``COMPARE.md`` (the two
lanes side by side), ``steps.jsonl`` (the comparison as ``dclab_core.twin`` steps, judged where the truth is planted:
a studied sample's target and blocked columns) and a ``README.md`` that says what each file is.

Brought onto main by package 25.2 from the A6.6 branch (``sft-factory-all-stages``): the lanes, the caps and the
folder are kept; the A6.5 decision samples are the ledger here; Claude comes through the native Messages transport.

    python -m dclab_rnd.twin run --sample bank_marketing --yes --max-requests 60
    python -m dclab_rnd.twin run --project <id> --blind --yes --cap-eur 2
    python -m dclab_rnd.twin run --draft <id> --yes
    python -m dclab_rnd.twin list
    python -m dclab_rnd.twin compare <run_dir>

Nothing is sent to Claude without ``--yes``; ``ANTHROPIC_API_KEY`` names the key and ``DCLAB_TWIN_MODEL`` the model.
The sealed benchmark is never a seed. A twin run's files are research material: they hold prompts, answers and the
table's column names, so a run on a confidential table stays on the machine that made it.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import claude, compare as comparing
from .claude import ClaudeClient, NoModels, Transcript, TwinGateway
from .design import Designer
from .lane import TASK, Lane, run_both

__all__ = ["Designer", "TASK", "Lane", "ClaudeClient", "TwinGateway", "NoModels", "Transcript", "run", "list_runs", "compare_run", "twin_home", "runs_home"]

README = """# Twin run {run_id}

Two lanes ran the same problem at the same time. Each lane is a complete workspace folder; nothing is shared.

| Path | What it holds |
|---|---|
| `run.json` | the manifest: seed, task, model, caps, each lane's status, start and end |
| `compare.json`, `COMPARE.md` | the two lanes side by side: solution, stage choices, holdout, moves, tool calls, usage |
| `<lane>/projects/<id>/project.json` | the project: solution, settings, decisions, approvals, memory notes, suggestion |
| `<lane>/projects/<id>/stages/<stage>.json` | each stage's full record: evidence, claims, decision with its options, provenance |
| `<lane>/projects/<id>/transitions.jsonl` | every move checked by the workflow graph: allowed or refused, the rules, the state |
| `<lane>/projects/<id>/activity.jsonl` | the project's activity log |
| `<lane>/intern/<session>.json` | the agent's whole conversation: system prompt, task, every model reply, every tool result, the final report |
| `<lane>/agent_steps.jsonl` | one row per tool call: arguments, verdict, result summary, tokens, seconds |
| `<lane>/audit.jsonl` | the audit trail: moves, approvals, the solution saved, data imported |
| `<lane>/model_requests.jsonl` | one row per model request: purpose, model, tokens, seconds, cost, outcome |
| `<lane>/transcripts/model_calls.jsonl` | every model request and answer, verbatim (the AI-first lane; the workflow lane only when it ran with a model) |
| `<lane>/ledger/decisions/`, `ledger/labels/` | every decision the lane made (the rule's answer, the agent's, the value used), one line each, by day (19.7) |
| `<lane>/experience/` | the lane's session record: every event with its ids (19.2) |
| `steps.jsonl` | the comparison as twin steps (`dclab_core.twin`, 25.1): both lanes' answers per step, their differences, the verdict where the truth is planted |
| `expert_steps.jsonl` | each version of Claude's expert record against the current system's answers, as twin steps (25.3) |
| `ai_first/design/DESIGN.md` | how Claude, as the expert, understood the problem and the data, explored and prepared it, typed the problem, chose the algorithms and drew both workflows; each version, what changed and why, the timeline, plan against act (A6.7, 25.3) |
| `ai_first/design/expert.jsonl`, `design_vNN.json`, `.md` | each version as an `ExpertTrace`; `designs.jsonl` the raw answers with Claude's thinking; `timeline.jsonl` one row per tool call and design event |
| `ai_first/design/adherence.json`, `intents.jsonl`, `sft_candidates.jsonl` | the plan against the act; the record's decisions on the ledger's points; each version as a chat sample (unverified) |
| `ai_first/transcripts/thinking.jsonl` | Claude's thinking beside each reply that carried it (25.3) |

The workflow lane is the current system ({workflow_label}); the AI-first lane is {ai_label}.
These files hold prompts, model answers and the table's column names: research material, kept where it was made.
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def twin_home(home: Path | None = None) -> Path:
    """The workspace's twin folder: ``runs/`` (whole problems, 25.2) and ``sessions/`` (Home, live, 25.4)."""
    root = Path(__file__).resolve().parents[2]
    return Path(home or os.environ.get("DCLAB_AGENT_HOME") or root / "agent_runs") / "twin"


def runs_home(home: Path | None = None) -> Path:
    return twin_home(home) / "runs"


def _workflow_gateway(mode: str, lane_home: Path) -> Any:
    if mode == "configured":
        from ..models import for_workspace

        gateway = for_workspace(lane_home)
        gateway.shadows = None
        return gateway
    return NoModels()


ALL_ROLES = ("problem_analyst", "brief_checker", "data_investigator", "solution_planner", "feature_engineer", "critic", "reporter")


DEFAULT_CAP_EUR = 1.5  # the owner's cap for one run of the AI-first arm (2026-10-16); prices are entered in dollars as euros


def run(seed: str, **kwargs: Any) -> dict[str, Any]:
    """One twin run, as a session of the agent ``twin`` in the harness (its end is one ``run.recorded``). See ``_run``."""
    from ..agents import harness

    kwargs.setdefault("run_id", datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3))
    with harness.session("twin", kwargs["run_id"]):
        return _run(seed, **kwargs)


def _run(seed: str, *, home: Path | None = None, task: str = TASK, model: str | None = None, transport: Any = None, max_requests: int = 120,
        cap_eur: float | None = DEFAULT_CAP_EUR, max_steps: int = 30, max_minutes: int = 40, blind: bool = False, workflow_models: str = "off",
        quick: bool = True, roles: str = "all", capture_cells: bool = False, allow_private: bool = False, source_home: Path | None = None,
        run_id: str | None = None, design: bool = True, design_checkpoints: tuple[str, ...] | None = None, design_steer: bool = True,
        log=print) -> dict[str, Any]:
    """One twin run. ``seed`` is ``sample:<key>``, ``project:<id>`` or ``draft:<id>``; ``transport`` replaces the real
    Claude transport (tests pass a scripted one). ``roles`` is ``all`` (every role of dclab_core on, so the AI-first
    lane asks Claude for each of them; the workflow lane has no model, so its roles stay on their rules) or ``judged``
    (the product's own gating). ``design`` (A6.7, 25.3; a project or sample seed): Claude writes its expert record of
    the problem and the solution before the AI-first lane starts, revises it after each stage of ``design_checkpoints``
    (data, leakage, models by default) and once at the end, and the lane's ``design/`` folder keeps every version, the
    timeline of what the agent did against it and the adherence; ``design_steer`` hands each revision to the agent.
    Returns the manifest, written to ``<run>/run.json``."""
    kind, _, ref = seed.partition(":")
    if kind not in ("sample", "project", "draft") or not ref:
        raise ValueError("seed is sample:<key>, project:<id> or draft:<id>")
    if transport is None and os.environ.get("DCLAB_NO_LIVE_MODELS") == "1":
        raise RuntimeError("DCLAB_NO_LIVE_MODELS=1: no live model may be used here")
    from .design import CHECKPOINT_STAGES, CHECKPOINTS

    checkpoints = tuple(CHECKPOINTS if design_checkpoints is None else design_checkpoints)
    if any(c not in CHECKPOINT_STAGES for c in checkpoints):
        raise ValueError(f"a design checkpoint is one of {', '.join(CHECKPOINT_STAGES)}")
    # each lane is a folder of its own (file stores, ledger, log, audit, usage) even on a workspace kept in PostgreSQL; the
    # source project or draft is read through the workspace's own stores
    if kind == "project":  # refused before anything is written: a refused run leaves no folder
        from ..storage import open_stores

        src = open_stores(Path(source_home or twin_home(home).parent))[0].get(ref)
        if (src.get("settings") or {}).get("private") and not allow_private:
            raise RuntimeError("the source project is private (its data must not reach a model); pass allow_private to run a twin on it anyway")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    root = runs_home(home) / run_id
    root.mkdir(parents=True, exist_ok=False)
    model = model or claude.model_name()
    inner = transport or claude.transport(model)
    ai_gateway = TwinGateway(inner, Transcript(root / "ai_first" / "transcripts" / claude.TRANSCRIPT), max_requests=max_requests, cap_eur=cap_eur,
                             capture_cells=capture_cells, month_spent=_month_spent(twin_home(home).parent))
    workflow = Lane("workflow", root / "workflow", _workflow_gateway(workflow_models, root / "workflow"),
                    label="the current system: rules, validator, standard plan" + (", with its configured model" if workflow_models == "configured" else ", no model"))
    ai_first = Lane("ai_first", root / "ai_first", ai_gateway, label=f"Claude ({model}) as the agent, with the same tools and validator")
    from ..models.gateway import Gateway
    from ..models.usage import FileUsage

    ai_gateway.usage = Gateway(FileUsage(root / "ai_first" / "model_requests.jsonl"))  # the usage rows every client writes
    manifest: dict[str, Any] = {"run_id": run_id, "seed": seed, "task": task, "model": model, "started": now(), "finished": None,
                                "caps": {"max_requests": max_requests, "cap_eur": cap_eur, "max_steps": max_steps, "max_minutes": max_minutes},
                                "design": {"on": bool(design) and kind != "draft", "checkpoints": list(checkpoints), "steer": design_steer},
                                "thinking": getattr(inner, "thinking", None),
                                "blind": blind, "workflow_models": workflow_models, "quick": quick, "roles": roles, "capture_cells": capture_cells,
                                "lanes": {"workflow": {"label": workflow.label}, "ai_first": {"label": ai_first.label}}}
    try:
        from ..models import prices

        manifest["priced"] = prices.price_of(model) is not None
    except Exception:  # noqa: BLE001
        manifest["priced"] = False
    if cap_eur is not None and not manifest["priced"]:
        manifest["notes"] = [f"{model} has no price in the price table: the euro cap cannot be applied; the request cap ({max_requests}) is the limit"]
    (root / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "README.md").write_text(README.format(run_id=run_id, workflow_label=workflow.label, ai_label=ai_first.label), encoding="utf-8")
    log(f"twin run {run_id}: {seed} → {root}")

    # ---- seed both lanes with the same problem
    from .. import context

    if kind == "sample":
        goal = task[:300]
        for lane in (workflow, ai_first):
            with context.using(lane.services):  # the seed's own events (the upload, the audit) go to the lane's record
                lane.seed_sample(ref, goal, blind=blind)
    elif kind == "project":
        from ..storage import open_stores

        src_projects, _d, _s = open_stores(Path(source_home or twin_home(home).parent))
        for lane in (workflow, ai_first):
            with context.using(lane.services):
                lane.seed_project(src_projects, ref, blind=blind)
    else:
        from ..storage import open_stores

        _p, src_drafts, _s = open_stores(Path(source_home or twin_home(home).parent))
        turns = None
        for lane in (workflow, ai_first):
            with context.using(lane.services):
                _did, turns = lane.seed_draft(src_drafts, ref)
        manifest["draft_turns"] = len(turns or [])
    manifest["lanes"]["workflow"].update(project_id=workflow.project_id, draft_id=workflow.draft_id)
    manifest["lanes"]["ai_first"].update(project_id=ai_first.project_id, draft_id=ai_first.draft_id)

    # ---- run both at once: the engine's profile and the roles' gate are process-wide, so they are set once, around both lanes
    from contextlib import ExitStack
    from unittest import mock

    from ..expansion.runner import fast_profile

    if kind == "draft":
        wf_client = workflow.gateway.client("home_agent", draft_id=workflow.draft_id) if workflow_models == "configured" else None
        jobs = [(workflow, lambda: workflow.run_draft(turns or [], wf_client)),
                (ai_first, lambda: ai_first.run_draft(turns or [], ai_gateway.client("home_agent", draft_id=ai_first.draft_id)))]
    else:
        wf_client = workflow.gateway.client("intern", project_id=workflow.project_id) if workflow_models == "configured" else None
        designer = Designer(ai_first, ai_gateway, run_id=run_id, model=model, checkpoints=checkpoints, steer=design_steer) if design else None
        jobs = [(workflow, lambda: workflow.run_project(task, wf_client, max_steps, max_minutes)),
                (ai_first, lambda: ai_first.run_project(task, ai_gateway.client("intern", project_id=ai_first.project_id), max_steps, max_minutes,
                                                        designer=designer))]
    with ExitStack() as stack:
        if quick:
            stack.enter_context(fast_profile())
        if roles == "all":
            stack.enter_context(mock.patch.dict(os.environ, {"DCLAB_ROLES": ",".join(ALL_ROLES)}))
        from .. import roles as roles_module

        manifest["roles_on"] = [r["role"] for r in roles_module.status() if r.get("on")] if roles != "all" else list(ALL_ROLES)
        summaries = run_both(jobs)
    for lane, summary in zip((workflow, ai_first), summaries):
        (root / lane.name / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        manifest["lanes"][lane.name].update(status=lane.status, error=lane.error, started=lane.started, finished=lane.finished, session_id=lane.session_id)
    truth = comparing.planted(kind, ref, workflow) if blind else None
    if truth:
        manifest["truth"] = truth
    elif kind == "sample":
        manifest.setdefault("notes", []).append("the sample's known solution was given to both lanes (not --blind): nothing is judged against it")
    comparison = comparing.compare(summaries[0], summaries[1])
    steps = comparing.write_steps(root / "steps.jsonl", run_id, summaries[0], summaries[1], truth)
    comparison["judged"] = comparing.verdicts(steps)
    try:  # one record the steps cannot hold never costs the run its comparison
        comparison["expert_judged"] = comparing.verdicts(comparing.write_expert_steps(root / "expert_steps.jsonl", run_id, root / "ai_first" / "design",
                                                                                      summaries[0], truth))
    except Exception as exc:  # noqa: BLE001
        manifest.setdefault("notes", []).append(f"the expert steps could not be written: {type(exc).__name__}: {str(exc)[:200]}")
    (root / "compare.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (root / "COMPARE.md").write_text(comparing.markdown(comparison, manifest), encoding="utf-8")
    manifest["finished"] = now()
    manifest["model_usage"] = ai_gateway.totals()
    manifest["differences"] = {"solution": comparison["solution_differences"], "stages": comparison["stage_differences"]}
    (root / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"done: workflow {workflow.status}, ai_first {ai_first.status}; differences in solution {comparison['solution_differences']}, "
        f"stages {comparison['stage_differences']}; see {root / 'COMPARE.md'}")
    return manifest


def _month_spent(home: Path) -> float:
    """What the workspace's models spent this month (its usage log), so a twin run counts against the monthly cap too."""
    try:
        from ..models import open_usage
        from ..models.usage import FileUsage, month_start

        since = month_start()
        total = float(open_usage(Path(home)).spent(since))
        runs = runs_home(Path(home))
        for usage in sorted(runs.glob("*/ai_first/model_requests.jsonl")) if runs.is_dir() else []:  # earlier twin runs count too
            total += float(FileUsage(usage).spent(since))
        return total
    except Exception:  # noqa: BLE001 — no log yet: nothing spent
        return 0.0


def list_runs(home: Path | None = None) -> list[dict[str, Any]]:
    out = []
    base = runs_home(home)
    if not base.is_dir():
        return out
    for path in sorted(base.iterdir()):
        manifest = path / "run.json"
        if manifest.is_file():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({"run_id": m.get("run_id", path.name), "seed": m.get("seed"), "model": m.get("model"), "started": m.get("started"), "finished": m.get("finished"),
                        "status": {k: v.get("status") for k, v in (m.get("lanes") or {}).items()}, "differences": m.get("differences"), "path": str(path)})
    return out


def compare_run(run_dir: Path) -> dict[str, Any]:
    """Rebuild compare.json and COMPARE.md of a finished run from the lanes' own files."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summaries = []
    for name in ("workflow", "ai_first"):
        lane = Lane(name, run_dir / name, NoModels(), label=manifest["lanes"][name].get("label", ""))
        info = manifest["lanes"][name]
        lane.project_id, lane.draft_id, lane.session_id = info.get("project_id"), info.get("draft_id"), info.get("session_id")
        lane.status, lane.error, lane.started, lane.finished = info.get("status", "unknown"), info.get("error"), info.get("started"), info.get("finished")
        summary = lane.summary()
        if name == "ai_first" and manifest.get("model_usage"):
            summary["model"] = {**manifest["model_usage"], "refused_in_transcript": summary.get("model", {}).get("refused_in_transcript", 0)}
        summaries.append(summary)
    comparison = comparing.compare(summaries[0], summaries[1])
    (run_dir / "steps.jsonl").unlink(missing_ok=True)  # rebuilt from the lanes' files, never appended to
    steps = comparing.write_steps(run_dir / "steps.jsonl", manifest["run_id"], summaries[0], summaries[1], manifest.get("truth"))
    comparison["judged"] = comparing.verdicts(steps)
    try:
        comparison["expert_judged"] = comparing.verdicts(comparing.write_expert_steps(run_dir / "expert_steps.jsonl", manifest["run_id"],
                                                                                      run_dir / "ai_first" / "design", summaries[0], manifest.get("truth")))
    except Exception as exc:  # noqa: BLE001
        comparison["expert_steps_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    (run_dir / "compare.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (run_dir / "COMPARE.md").write_text(comparing.markdown(comparison, manifest), encoding="utf-8")
    return comparison
