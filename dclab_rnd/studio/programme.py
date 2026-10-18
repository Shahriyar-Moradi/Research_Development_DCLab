"""A project's programme (package 24.5): the solution map of a problem with several parts, run on the project's tables.

The draft that built the project kept its confirmed problem map, the routes and the solution map in ``project["programme_plan"]``. This module reads the
project's tables (the main table, the related tables attached by name, the label file of confirmed cases), runs ``dclab_core.programme`` on them and keeps the
result in ``project["programme"]``: the document (names, counts and scores, never a row) and the inputs it read (hashes, the budget, the columns kept out), so a later
change to any of them makes the programme stale. The holdout is read once, by the composition, after every leaf has run: that is the core's rule. The graph's moves
``run_programme`` and ``drop_leaf`` add the project's: once the holdout has been read by any programme of this project, a second read needs a person's written reason
and an agent may not make it, and a leaf can no longer be dropped. A replay or an export is refused when anything the run read has changed, since running it then would be
a second look at the holdout under another question.

The engine's five stages are not run for a programme: its leaves are the core's own runners, on the shared split of the solution map.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from dclab_core import leaves as core_leaves
from dclab_core import programme as core
from dclab_core.problem_map import ProblemMap
from dclab_core.solution_map import SolutionMap

from . import data as sd
from . import graph as studio_graph

SEED = 0
BOOTSTRAPS = 500
RUNNING = "programme"


class ProgrammeError(ValueError):
    """The programme cannot be run or changed now; the message says why."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- what the project holds

def plan_of(project: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The draft's confirmed plan for a problem of several parts, or None (a single problem has no programme)."""
    plan = project.get("programme_plan")
    return plan if isinstance(plan, dict) and plan.get("solution_map") else None


def maps(project: dict[str, Any]) -> tuple[ProblemMap, SolutionMap]:
    plan = plan_of(project)
    if plan is None:
        raise ProgrammeError("this project was not built from a problem with several parts: there is no programme to run")
    return ProblemMap.read(plan["problem_map"]), SolutionMap.read(plan["solution_map"])


def stored(project: dict[str, Any]) -> Optional[core.Programme]:
    doc = (project.get("programme") or {}).get("document")
    return core.Programme.read(doc) if doc else None


def reads_of(project: dict[str, Any]) -> int:
    """How many times a programme of this project read the holdout (the stored one, and the ones it replaced)."""
    run = project.get("programme") or {}
    return int(run.get("reads") or 0)


def budget_of(project: dict[str, Any]) -> Optional[float]:
    value = (project.get("suggestion") or {}).get("alert_budget")
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def excluded(project: dict[str, Any]) -> list[str]:
    """The columns no leaf reads: the identifiers and the columns the project forbids (the suggestion's and the saved solution's), the key that joins the labels (it names a
    case) and every column of the main table the data card marks as a leak suspect, an outcome or an identifier."""
    sug, sol = project.get("suggestion") or {}, project.get("solution") or {}
    names: list[Any] = [*(sug.get("identifiers") or []), *(sug.get("forbidden") or []), *(sol.get("identifiers") or []), *(sol.get("forbidden") or []),
                        *((project.get("labels") or {}).get("key") or []),
                        *(project.get("forbidden_by_person") or [])]  # 28.5: a person's own bans (the move forbid_columns), which no new table clears
    card = (project.get("data") or {}).get("card") or {}
    for table in card.get("tables") or []:
        if table.get("role", "main") == "main":
            names += [c.get("name") for c in table.get("columns") or [] if c.get("purpose") in ("leak_suspect", "outcome", "identifier")]
    named = [n.get("column") if isinstance(n, dict) else n for n in names]  # a forbidden column as the solution writes it: {"column", "reason"}
    return sorted({str(n) for n in named if n})


def inputs_of(project: dict[str, Any]) -> dict[str, Any]:
    """What a run reads, by hash, name and count: the main table, the related tables, the label file, the plan, the budget, the columns kept out and the choices that
    shape the run. A change to any of them makes the stored programme stale."""
    from dclab_core import __version__

    plan = plan_of(project) or {}
    return {"data_sha256": (project.get("data") or {}).get("sha256"),
            "tables": {n: t.get("sha256") for n, t in sorted((project.get("tables") or {}).items())},
            "labels_sha256": (project.get("labels") or {}).get("sha256"),
            "solution_map": (plan.get("solution_map") or {}).get("fingerprint"),
            "budget": budget_of(project), "excluded": excluded(project), "core_version": __version__, "seed": SEED, "bootstraps": BOOTSTRAPS}


def routing_inputs(project: dict[str, Any]) -> str:
    """What the routes are drawn from: the tables and the labels (28.5: the team routes again when one changes)."""
    return _hash({"data": (project.get("data") or {}).get("sha256"), "tables": {n: t.get("sha256") for n, t in sorted((project.get("tables") or {}).items())},
                  "labels": (project.get("labels") or {}).get("sha256")})


def audit_inputs(project: dict[str, Any]) -> str:
    """What the auditor screens and the person approves reading the holdout on: the tables, the labels, the columns kept out and the budget (28.5: a change to any of them
    makes the screen and the approval old)."""
    return _hash({"routing": routing_inputs(project), "excluded": excluded(project), "budget": budget_of(project)})


def load(store: Any, project: dict[str, Any]) -> dict[str, Any]:
    """The tables a programme reads, from the project's own files (each checked against the hash recorded when it was attached)."""
    if not project.get("data"):
        raise ProgrammeError("attach the main table first")
    frame = sd.load_table(sd.data_path(store, project))
    labels = known_at = None
    if project.get("labels"):
        from . import labels as label_file

        cases = sd.load_table(label_file.path(store, project))
        key = list(project["labels"]["key"])
        labels = core_leaves.mark_cases(frame, cases, key)
        column = core_leaves.confirmation_column(cases, key)
        known_at = core_leaves.case_dates(frame, cases, key, column) if column else None
    tables = sd.load_tables(store, project, list((project.get("tables") or {}).keys())) if project.get("tables") else {}
    return {"frame": frame, "labels": labels, "known_at": known_at, "tables": tables}


# ---------------------------------------------------------------------------------------------------------------- the plan follows the tables

def plan_card(store: Any, project: dict[str, Any]) -> Any:
    """The data card the router reads: the main table and every related one the project holds now (names, purposes and bands)."""
    from dclab_core import datacard

    name = str(project["data"].get("filename") or "table").rsplit(".", 1)[0]
    tables = {name: sd.load_table(sd.data_path(store, project))}
    if project.get("tables"):
        tables.update(sd.load_tables(store, project, list(project["tables"].keys())))
    return datacard.card_of(tables, roles={name: "main", **{k: "related" for k in tables if k != name}})


def refresh_plan(store: Any, project_id: str, advisor: Any = None, routes: Optional[dict[str, Any]] = None, record: bool = True) -> bool:
    """Draw the solution map and the routes again from the confirmed problem map and the tables the project holds now. The person confirmed the problem (its parts and how they
    combine); which approach reads which table follows from the data, so attaching the table of accounts is what lets the rings be routed. Returns whether the plan changed;
    a changed plan makes a stored programme stale (it ran another solution map). ``advisor``: the solution router when it serves (24.8), as it does on Home. ``routes``:
    the routes already chosen by the same router for every sub-problem (28.5's team, which asks it part by part in parallel). ``record`` false: the routes' decisions
    were recorded by whoever chose them (the team records each route it makes, with its shadow's answer: 28.10)."""
    from dclab_core import solution_map as sm_mod, solutions

    project = store.get(project_id)
    plan = plan_of(project)
    if plan is None or not project.get("data"):
        return False
    m = ProblemMap.read(plan["problem_map"])
    card = plan_card(store, project)
    if routes is None:  # 28.5: the solution team routes the parts in parallel and hands its routes here, the same router's
        routes = {s.id: solutions.route_with(s, card, m, None, advisor)[0] for s in m.sub_problems}
    sm = sm_mod.solution_map(m, routes, card)
    if sm.fingerprint == plan["solution_map"].get("fingerprint"):
        return False
    project["programme_plan"] = {**plan, "routes": {k: json.loads(v.model_dump_json()) for k, v in routes.items()}, "solution_map": json.loads(sm.model_dump_json())}
    store.save(project)
    store.log(project_id, "programme_plan_refreshed", {"was": plan["solution_map"].get("fingerprint"), "now": sm.fingerprint, "blockers": len(sm.blockers)})
    from ..experience import hooks
    from ..ledger import capture

    draft_id = (project.get("draft") or {}).get("id") or project_id
    for sid, route in (routes.items() if record else ()):  # an approach that changed with the tables is a decision again: recorded under the draft the project was built from
        was = (plan.get("routes") or {}).get(sid) or {}
        if was.get("chosen") != route.chosen or was.get("source") != route.source:
            sub = next(x for x in m.sub_problems if x.id == sid)
            capture.route_chosen(draft_id, sub, route, card, len(m.parts))
            hooks.route_made(draft_id, m.fingerprint, sub, route)
    return True


# ---------------------------------------------------------------------------------------------------------------- the view

def stale_reasons(project: dict[str, Any]) -> list[str]:
    """What the stored programme read that is not what the project holds now (empty: nothing changed)."""
    run = project.get("programme")
    if not run or not run.get("inputs"):
        return []
    then, now = run["inputs"], inputs_of(project)
    return [f"{k} changed since the programme ran" for k in sorted(set(then) | set(now)) if then.get(k) != now.get(k)]


def view(project: dict[str, Any]) -> dict[str, Any]:
    """The programme for the page and the API: the plan's blocks, its state, what blocks it, the report's sections and whether it is stale."""
    plan = plan_of(project)
    if plan is None:
        return {"available": False, "why": "this project was not built from a problem with several parts"}
    sm = SolutionMap.read(plan["solution_map"])
    p = stored(project)
    out: dict[str, Any] = {"available": True, "state": p.state if p else "not_run", "fingerprint": p.fingerprint if p else None,
                           "solution_map": sm.fingerprint, "map": plan["problem_map"].get("fingerprint"),
                           "blocks": [{"id": b.id, "title": b.title, "family": b.family, "approach": b.approach, "final": b.final} for b in sm.blocks],
                           "plan_blockers": list(sm.blockers), "split": json.loads(sm.holdout.split.model_dump_json()),
                           "stale": stale_reasons(project), "blockers": list(p.blockers) if p else [], "issues": core.open_issues(p) if p else [],
                           "tables": sorted((project.get("tables") or {}).keys()), "labels": bool(project.get("labels")), "sections": [],
                           "holdout_reads": reads_of(project)}
    if p:
        out.update({"sections": core.sections(p), "document": json.loads(p.model_dump_json()), "ran_at": project["programme"].get("ran_at"),
                    "validation": core.validate(p, sm if p.solution == sm.fingerprint else None), "earlier": project["programme"].get("earlier") or []})
    return out


# ---------------------------------------------------------------------------------------------------------------- run, drop, replay

def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _context(project: dict[str, Any], sm: SolutionMap, loaded: dict[str, Any], budget: Optional[float]) -> core_leaves.Context:
    frame = loaded["frame"]
    split = sm.holdout.split
    try:
        parts = core_leaves.partition(frame, moment=split.moment, keys=split.keys, share=split.share, seed=split.seed)
    except core_leaves.PartitionError as exc:
        raise ProgrammeError(str(exc)) from None
    return core_leaves.Context(frame=frame, parts=parts, time=split.moment, entities=list(split.keys), labels=loaded["labels"], known_at=loaded["known_at"],
                               tables=loaded["tables"], exclude=excluded(project), budget=budget, seed=SEED)


def execute(store: Any, project: dict[str, Any], *, prior: Optional[core.Programme] = None, bootstraps: int = BOOTSTRAPS, replaying: bool = False) -> core.Execution:
    """Run the programme on the project's tables, from a fresh plan that keeps what ``prior`` decided and saw (a person's drops, the count of reads of the holdout): the
    leaves are deterministic under the seed, so they run again to the same scores, the composition is chosen on validation and the holdout is read once when nothing blocks.
    Nothing is saved here."""
    m, sm = maps(project)
    loaded = load(store, project)
    budget = budget_of(project)
    ctx = _context(project, sm, loaded, budget)
    start = core.replan(sm, m, prior, budget=budget, replaying=replaying)
    return core.resume(core.Execution(start), sm, m, ctx, loaded["labels"], bootstraps=bootstraps)


def _release(store: Any, project_id: str) -> None:
    project = store.get(project_id)
    if project.get("running") == RUNNING:
        project["running"] = None
        store.save(project)


def run(store: Any, project_id: str, actor: str = "human", reuse_reason: Optional[str] = None, advisor: Any = None,
        routes: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The graph's move ``run_programme``, then the run. The project is marked running while it reads its files (no table can be attached under it), what it read is recorded
    from the project as it was when the run started, and the holdout is consumed only when the composition reads it. ``routes`` (28.5): the routes the solution team
    chose and its auditor screened the plan of, so the plan that runs is that one (none: the router is asked again here)."""
    project = store.get(project_id)
    verdict = studio_graph.check(project, "run_programme", actor, reuse_reason=reuse_reason)
    if not verdict.allowed:
        studio_graph.log(store, project_id, verdict, project)
        raise ProgrammeError(verdict.message)
    prior = stored(project)
    project["running"] = RUNNING
    store.save(project)
    studio_graph.log(store, project_id, verdict, store.get(project_id), outcome="started")
    try:
        refresh_plan(store, project_id, advisor, routes=routes)
        snapshot = store.get(project_id)
        inputs = inputs_of(snapshot)
        execution = execute(store, snapshot, prior=prior)
    except BaseException:
        _release(store, project_id)
        raise
    p = execution.programme
    project = store.get(project_id)
    read = p.holdout is not None
    run_ = project.get("programme") or {}
    earlier = list(run_.get("earlier") or [])
    if prior is not None and prior.holdout is not None:  # a second programme on the same holdout: the first stays in the record, with its numbers
        earlier.append({"fingerprint": prior.fingerprint, "ran_at": run_.get("ran_at"), "composed": prior.holdout.composed.value,
                        "difference": prior.holdout.difference.value, "reason": (reuse_reason or "")[:300] or None})
    project["programme"] = {"document": json.loads(p.model_dump_json()), "inputs": inputs, "ran_at": _now(), "earlier": earlier[-10:],
                            "reads": reads_of(project) + (1 if read else 0)}
    project["running"] = None
    if read:
        project["holdout_uses"] = int(project.get("holdout_uses", 0) or 0) + 1
        if actor == "agent":
            studio_graph.consume_holdout_approval(project)  # marks the owner's approval used, in place
    store.save(project)
    store.log(project_id, "programme_run", {"state": p.state, "fingerprint": p.fingerprint, "holdout_read": read, "blockers": len(p.blockers),
                                            "composed": p.holdout.chosen if p.holdout else None})
    from .. import audit
    from ..experience import hooks

    audit.record_for(store, "move", actor, project=project, move="run_programme", status="allowed", state=p.state, holdout_read=read, fingerprint=p.fingerprint,
                     reason=(reuse_reason or "")[:300] or None)
    hooks.programme_built(project_id, p, actor)
    from ..ledger import capture

    capture.composition_chosen(project_id, p)
    return view(store.get(project_id))


def drop(store: Any, project_id: str, block: str, reason: str, actor: str = "human") -> dict[str, Any]:
    """A person drops a leaf that cannot run (the composition goes on without it), through the graph's move ``drop_leaf``: refused for an agent, for an empty reason and
    for any project whose holdout a programme has read. It is recorded in the programme and the audit trail."""
    project = store.get(project_id)
    verdict = studio_graph.check(project, "drop_leaf", actor, reason=reason)
    if not verdict.allowed:
        studio_graph.log(store, project_id, verdict, project)
        raise ProgrammeError(verdict.message)
    p = stored(project)
    try:
        p2 = core.drop_leaf(p, block, reason, by="person")
    except ValueError as exc:
        raise ProgrammeError(str(exc)) from None
    project["programme"] = {**project["programme"], "document": json.loads(p2.model_dump_json())}
    store.save(project)
    studio_graph.log(store, project_id, verdict, store.get(project_id), outcome=f"done: {block} dropped")
    store.log(project_id, "programme_dropped", {"block": block, "reason": reason[:300]})
    from .. import audit

    audit.record_for(store, "move", "human", project=project, move="drop_leaf", status="allowed", block=block, reason=reason[:300])
    from ..experience import record

    record("problem.part_changed", "person", {"map": (plan_of(project) or {}).get("problem_map", {}).get("fingerprint"), "action": "drop_leaf", "part": block, "by": "person",
                                              "parts_before": sum(1 for l in p.leaves if l.status != "dropped"), "parts_after": sum(1 for l in p2.leaves if l.status != "dropped"),
                                              "reason": reason[:2000]}, project=project_id)
    return view(store.get(project_id))


def replay(store: Any, project_id: str) -> dict[str, Any]:
    """Run the stored programme again in a scratch copy (nothing is saved, the holdout is not consumed) and compare it with the record: the same inputs and seed must give
    the same document. Refused when anything the run read has changed, or when the stored programme never read the holdout: replaying then would look at the holdout
    under another question, with no reason asked and no count kept (that is a run, with the graph's rules)."""
    project = store.get(project_id)
    p = stored(project)
    if p is None:
        raise ProgrammeError("the programme has not run yet")
    changed = stale_reasons(project)
    if changed:
        raise ProgrammeError("a replay needs the same inputs as the run (" + "; ".join(changed) + "): run the programme again, with the graph's rules for a second read of the holdout")
    if p.holdout is None:
        raise ProgrammeError("the programme never read the holdout: there is nothing to replay, and running it is what reads it, once")
    again = execute(store, project, prior=p, replaying=True).programme
    same = again.fingerprint == p.fingerprint
    out: dict[str, Any] = {"replayed": True, "same": same, "fingerprint": p.fingerprint, "rerun_fingerprint": again.fingerprint, "stale": []}
    if not same:
        out["differ"] = [k for k in ("state", "leaves", "composition", "holdout", "blockers", "notes", "periods") if getattr(again, k) != getattr(p, k)]
    if again.holdout is not None:
        out["composed"] = p.holdout.composed.value
        out["rerun_composed"] = again.holdout.composed.value
    return out
