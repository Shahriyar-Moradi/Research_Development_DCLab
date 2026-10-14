"""The kind acceptance harness and the capability matrix (package 16.2).

Each kind ships a replica case (seeded, invented data shaped like a real case, with that kind's traps planted; the
generators live in ``dclab_core.replicas``) and the decisions a careful data scientist reaches on it. :func:`run` builds
a project the way a person would (the data, the agent's proposal accepted, the prediction moment stated), runs the five
stages, and checks each decision. ``make kinds-check`` runs every kind and writes ``capability.json``; the matrix shown by
``/api/kinds`` and the Domain packs page is built from it, never written by hand: a kind is "supported" only when its
acceptance passes, otherwise "partial", with what is missing.

The records checked are the engine's own; nothing here calls a model. Replica scores say whether DCLab decides well,
never how a model would do on real data.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "capability.json"
STATUSES = ("supported", "partial")


@dataclass
class Case:
    """A replica as a person brings it: the table, what they state, and how the engine is run."""

    table: pd.DataFrame
    target: str | None  # None: a kind that reads no outcome (20.7)
    task: str
    moment: str
    solution: dict[str, Any] = field(default_factory=dict)  # what the person states beyond accepting the proposal
    pack: str | None = None
    labels: tuple[pd.DataFrame, str] | None = None  # (a label file, its target) joined like an upload (15.2)
    files: Callable[[Path], tuple[str, str | None]] | None = None  # writes the person's own files: (data file, label file)
    quick: bool = True
    facts: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] | None = None  # 16.15 C: related tables, attached by name beside the main one


@dataclass
class Run:
    """A finished replica project: its stage records and the solution it ran."""

    case: Case
    solution: dict[str, Any]
    records: dict[str, dict[str, Any]]
    labels: dict[str, Any] = field(default_factory=dict)  # the label file's join, when there is one (15.2)

    def evidence(self, stage: str) -> dict[str, Any]:
        return self.records[stage].get("evidence") or {}

    def role(self, column: str) -> str | None:
        return self.evidence("data").get("column_roles", {}).get(column)

    def holdout(self, metric: str) -> float | None:
        value = self.evidence("final").get("holdout_metrics", {}).get(metric)
        return float(value) if value is not None else None


def check(decision: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"decision": decision, "ok": bool(ok), "detail": detail}


def modelled_text(run: Run) -> set[str]:
    return set((run.evidence("data").get("text_stats") or {}).keys())


def kept_out(run: Run, column: str, why: str, roles: tuple[str, ...] = ("blocked", "identifier")) -> dict[str, Any]:
    """A column that must not reach the model (a leak, an identifier) is neither a feature nor a modelled text column;
    ``roles`` are the roles that count (a leak must be blocked, not merely set aside as a second date)."""
    role = run.role(column)
    ok = role in roles and column not in modelled_text(run)
    return check(f"`{column}` is kept out of the model ({why})", ok,
                 f"its role: {role}" + ("; but its text is modelled" if column in modelled_text(run) else ""))


def build(case: Case, root: Path) -> tuple[Any, str]:
    """The project a person would make from the case: the data, the label file, the proposal accepted."""
    from dclab_rnd.studio import ProjectStore, data as sd
    from dclab_rnd.studio.solution import Solution, propose

    store = ProjectStore(root)
    project = store.create("Replica", "general", case.moment)
    directory = store.data_dir(project["id"])
    label_bytes = None
    if case.files is not None:
        data_file, label_file = case.files(directory)
        if label_file is not None:  # attaching the data clears the folder: the label file is written again after it
            label_bytes = (directory / label_file).read_bytes()
    else:
        data_file, label_file = "table.csv", None
        case.table.to_csv(directory / data_file, index=False)
    sd.attach_data(store, project["id"], data_file)
    if label_bytes is not None:
        (directory / label_file).write_bytes(label_bytes)
    if case.labels is not None and label_file is None:
        label_file = "labels.csv"
        case.labels[0].to_csv(directory / label_file, index=False)
    if label_file is not None:
        project = sd.attach_labels(store, project["id"], label_file, target=case.labels[1] if case.labels else case.target)
    for name, frame in (case.tables or {}).items():  # after the main table and the labels: attaching those clears the folder
        frame.to_csv(directory / f"{name}.csv", index=False)
        sd.attach_table(store, project["id"], f"{name}.csv", name)
    project = store.get(project["id"])
    if case.pack:
        project["pack"] = case.pack
    if label_file is not None and project.get("suggestion"):  # a label file's join: its suggestion is the starting point
        proposal = {**project["suggestion"]}
    elif case.target is None:  # 20.7: no outcome to propose around; the person states the kind and its columns
        proposal = {}
    else:
        frame = sd.load_table(sd.data_path(store, project))
        proposal = propose(frame, project["data"]["profile"], case.target, case.task, case.pack)
    stated = {k: proposal.get(k) for k in Solution.model_fields if k in proposal and proposal.get(k) is not None}
    stated["forbidden"] = [{"column": f["column"], "reason": f.get("reason", "")[:600]} for f in proposal.get("forbidden") or []]
    # the person adopts the plan beside the proposal (the intern's adopt_plan move): what it forbids, and what it keeps
    # out until someone confirms when it is written (nobody confirms here). The moment rule's proposals are shown
    # unticked in the product, so they are not ticked here either.
    for decision in proposal.get("plan") or []:
        if decision.get("decision") in ("features.forbidden", "features.to_confirm"):
            for item in decision.get("choice") or []:
                if item.get("column") and item["column"] not in {f["column"] for f in stated["forbidden"]} and item["column"] != case.target:
                    stated["forbidden"].append({"column": item["column"], "reason": str(item.get("reason") or item.get("question") or decision["decision"])[:600]})
        choice = decision.get("choice")
        if (decision.get("decision") == "split.strategy" and isinstance(choice, dict) and choice.get("kind") == "spatial_block"
                and not {"time_column", "group_column", "spatial"} & set(case.solution)):  # 16.10: the plan's blocks of places, adopted
            stated["spatial"] = {"lat_column": choice["lat_column"], "lon_column": choice["lon_column"], "block_km": float(choice["block_km"])}
            stated["group_column"] = None
            if stated.get("kind") == case.task:  # the proposal's plain task kind was a suggestion: the pack's kind may now apply
                stated.pop("kind")
    stated.pop("metric", None)  # the kind's default decides, as for a person who leaves it
    stated.update({"target": case.target, "task": case.task, "prediction_moment": case.moment, **case.solution})
    if "time_column" not in case.solution and proposal.get("time_candidates") and not stated.get("spatial"):
        stated["time_column"] = proposal["time_candidates"][0]
    if stated.get("time_column"):  # the time column orders the split: the person unticks it under Forbidden, as the form asks
        stated["forbidden"] = [f for f in stated["forbidden"] if f["column"] != stated["time_column"]]
    stated["identifiers"] = [c for c in stated.get("identifiers") or [] if c not in {f["column"] for f in stated["forbidden"]}]
    project["solution"] = Solution(**stated).model_dump()
    project["settings"]["quick"] = case.quick
    store.save(project)
    from dclab_rnd.ledger import capture  # 19.7: the proposal a person accepts, as decisions (when a ledger keeps them)

    capture.solution_saved(proposal, project["data"]["profile"], project["solution"], "human", exposed=False, project=project["id"])  # a simulated person
    return store, project["id"]


def engine_run(case: Case, replica: str | None = None) -> Run:
    """The case as a project, every stage run by the engine with no model asked anything (a replica's records stay on the
    machine), and the finished records."""
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    with tempfile.TemporaryDirectory() as tmp:
        store, project_id = build(case, Path(tmp))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, project_id)
        from dclab_rnd.ledger import labels  # 19.8: the replica's planted traps label its decisions (when a ledger keeps them)

        labels.from_replica(project_id, case.facts, replica or case.target)
        records = {stage: store.read_stage(project_id, stage) for stage in engine.STAGE_KEYS}
        project = store.get(project_id)
        return Run(case, project["solution"], records, project.get("labels") or {})


def run(kind: Any, seed: int = 0) -> dict[str, Any]:
    """One kind's acceptance: {"kind", "status", "checks", "missing", "seconds"}."""
    case = kind.case(seed)
    if case is None:
        return {"kind": kind.key, "status": "partial", "checks": [], "missing": ["a replica case and its acceptance decisions"],
                "seconds": 0.0}
    started = time.perf_counter()
    own = kind.solve(case)  # a kind with its own machine (16.3 onward) solves its case itself
    if own is not None:
        checks = kind.accept(own)
        if kind.own_machine() is not None:  # 16.15: and the engine runs the same case, stage by stage, under its gates
            checks += kind.accept_engine(engine_run(case, kind.key))
        missing = [c["decision"] for c in checks if not c["ok"]] + ([] if kind.runs_in_engine else [ENGINE_MISSING])
        return {"kind": kind.key, "status": "supported" if checks and not missing else "partial", "checks": checks,
                "missing": missing, "seconds": round(time.perf_counter() - started, 1)}
    done = engine_run(case, kind.key)
    checks = kind.accept(done)
    computed = set((done.evidence("final").get("baselines") or {}).keys())
    checks.append(check("the baselines the kind declares are the ones the final stage computes", set(kind.baselines) <= computed,
                        f"declared {sorted(kind.baselines)}, computed {sorted(computed)}"))
    missing = [c["decision"] for c in checks if not c["ok"]]
    return {"kind": kind.key, "status": "supported" if checks and not missing else "partial", "checks": checks,
            "missing": missing, "seconds": round(time.perf_counter() - started, 1)}


def run_all(keys: list[str] | None = None) -> list[dict[str, Any]]:
    from dclab_rnd import kinds

    return [run(k) for k in kinds.all_kinds() if keys is None or k.key in keys]


def write(results: list[dict[str, Any]], path: Path = RESULTS) -> dict[str, Any]:
    """Store the results (``make kinds-check``): the matrix is read from this file."""
    doc = {"generated_by": "make kinds-check", "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
           "results": [{k: r[k] for k in ("kind", "status", "checks", "missing")} for r in results]}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def load(path: Path = RESULTS) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("results", [])
    except (OSError, ValueError):
        return []


ENGINE_MISSING = "the engine does not run it yet: it runs as a library (dclab_core) only"


def matrix(results: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One row per registered kind: supported only when its stored acceptance passed; partial otherwise, with what is
    missing (a kind never run says so)."""
    from dclab_rnd import kinds

    by_kind = {r["kind"]: r for r in (load() if results is None else results)}
    rows = []
    for kind in kinds.all_kinds():
        result = by_kind.get(kind.key)
        if result is None:
            rows.append({"kind": kind.key, "title": kind.title, "status": "partial", "passed": 0, "decisions": 0,
                         "missing": ["its acceptance has not been run (make kinds-check)"]})
            continue
        passed = sum(1 for c in result.get("checks", []) if c.get("ok"))
        ok = result.get("status") == "supported" and passed == len(result.get("checks", [])) and passed > 0 and kind.runs_in_engine
        missing = [] if ok else list(result.get("missing") or ([] if not kind.runs_in_engine else ["no decision checked"]))
        if not kind.runs_in_engine and ENGINE_MISSING not in missing:
            missing.append(ENGINE_MISSING)  # its decisions may all pass as a library; the product cannot run it yet
        rows.append({"kind": kind.key, "title": kind.title, "status": "supported" if ok else "partial", "passed": passed,
                     "decisions": len(result.get("checks", [])), "missing": missing})
    return rows
