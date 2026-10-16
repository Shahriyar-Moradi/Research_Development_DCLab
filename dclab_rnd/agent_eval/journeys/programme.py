"""Journeys of a problem with several parts (package 24.6): a written person, from the first sentence to the one read of the holdout.

A journey (``journeys_programme_dev.json``) is a person, as in ``journeys/__init__.py``: the problem in their words, their answers keyed by the field a question asks,
and the replica of a programme (``dclab_core.replicas``) as the files they bring. :func:`play` runs it as the product does, with no model: a draft with the table and
the file of confirmed cases, the coordinator (21.2) walking the draft to a built project (it draws the map of the problem and the written person confirms it), the
related table attached by name, the programme run as a job would run it, and the pipeline read at the end. It is scored on four things, each against the replica's
planted truth:

- **card**: the problem was read as several parts (a complex map), with the labels where given;
- **workflow**: the map has the parts the journey expects, the router gave each its approach, nothing is left blocked once the related table is attached, and no move of
  the project's graph was refused;
- **pipeline**: every step behind the programme's numbers has its run and none is stale, and a replay in a scratch copy makes the same document;
- **holdout**: the composition chosen on validation beats the best leaf chosen on validation with the interval of the paired difference above zero, the holdout was read
  once, and the planted fraud it found is more than the best leaf's.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEV = HERE / "journeys_programme_dev.json"
PARTS = ("card", "workflow", "pipeline", "holdout")
REPLICAS = {"fraud_programme": "dclab_core.replicas.fraud_programme"}


def load(split: str = "dev") -> list[dict[str, Any]]:
    import json

    if split != "dev":
        raise ValueError("the programme journeys have a dev set only until 24.8 writes the sealed one")
    return json.loads(DEV.read_text(encoding="utf-8"))


def play(journey: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """One programme journey from its first sentence; its scores and what happened at each part."""
    import importlib

    from dclab_core import compose, programme as core
    from dclab_rnd.agent_eval.journeys import _Person, _upload, refused
    from dclab_rnd.draft.card import card_of
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import ProjectStore, data as sd, pipeline, programme

    started = time.perf_counter()
    replica_module = importlib.import_module(REPLICAS[journey["programme"]])
    replica = replica_module.generate(seed=seed)
    row: dict[str, Any] = {"id": journey["id"], "programme": journey["programme"], "parts": {}, "why": []}
    with tempfile.TemporaryDirectory() as tmp:
        home, mine = Path(tmp), Path(tmp) / "person"
        mine.mkdir()
        drafts, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        paths = replica_module.save(replica, mine)
        agent = HomeAgent(drafts, None)
        draft_id = drafts.create(journey["words"])["id"]
        agent.start(draft_id)
        for name in (replica_module.RIDE_FILE, replica_module.CASE_FILE):  # the table and the cases, as a person brings them; the accounts come when the project exists
            _upload(drafts, agent, draft_id, name, mine / name)
        person = _Person(journey["answers"])
        with private_scope(True):
            out = Coordinator(drafts, projects, agent, None).run(draft_id, person.person)
        understood = card_of(drafts.get(draft_id))
        row.update(asked=person.asked, questions=len(person.asked), steps=[s["action"] for s in out["steps"]], state=out["state"], said=out.get("says"))
        built = out.get("project_id")
        if built is None:
            row["parts"].update(card=False, workflow=False, pipeline=False, holdout=False)
            row["why"].append(f"not built: stopped at {out['state']}: {out.get('says')}")
            return _end(row, started)
        project = projects.get(built)
        plan = programme.plan_of(project)
        row["parts"]["card"] = plan is not None and plan["problem_map"]["size"] == "complex" and understood.labels.value == journey.get("labels", understood.labels.value)
        if not row["parts"]["card"]:
            row["why"].append(f"card: the problem was not read as several parts (plan {'kept' if plan else 'missing'}, labels {understood.labels.value!r})")
            row["parts"].update(workflow=False, pipeline=False, holdout=False)
            return _end(row, started)
        replica.accounts.to_csv(projects.data_dir(built) / "accounts.csv", index=False)  # the related table the rings read, attached by name
        sd.attach_table(projects, built, "accounts.csv", "accounts")
        programme.refresh_plan(projects, built)
        plan = programme.plan_of(projects.get(built))
        families = [s["family"] for s in plan["problem_map"]["sub_problems"]]
        chosen = {k: r["chosen"] for k, r in plan["routes"].items()}
        blockers = list(plan["solution_map"].get("blockers") or [])
        view = programme.run(projects, built)
        refused_moves = refused(projects, built)
        row["refused_moves"] = len(refused_moves)
        want = journey["expect"]
        problems = []
        if families != want["families"]:
            problems.append(f"parts {families}, expected {want['families']}")
        if list(chosen.values()) != want["approaches"]:
            problems.append(f"approaches {list(chosen.values())}, expected {want['approaches']}")
        if blockers:
            problems.append(f"blocked: {blockers[:2]}")
        row["parts"]["workflow"] = not problems and not refused_moves
        if problems or refused_moves:
            row["why"].append(f"workflow: {problems}; refused moves {[t.get('move') for t in refused_moves[:3]]}")
        project = projects.get(built)
        p = programme.stored(project)
        steps = pipeline.of(projects, built)
        stale = [f"{s.id} ({s.why})" for s in steps.steps if s.status == "stale"]
        missing = [f"{s.id} ({s.status})" for s in steps.steps if s.status not in ("done", "stale")]
        replay = pipeline.rerun(projects, built, "programme:holdout") if not missing else {"same": False}
        row["parts"]["pipeline"] = not stale and not missing and bool(replay.get("same"))
        if stale or missing or not replay.get("same"):
            row["why"].append(f"pipeline: stale {stale}, no run {missing}, replay same {replay.get('same')}")
        if p is None or p.holdout is None:
            row["parts"]["holdout"] = False
            row["why"].append(f"holdout: the programme is {view.get('state')}, " + "; ".join(view.get("blockers") or view.get("issues") or []))
            return _end(row, started)
        h = p.holdout
        run = programme.execute(projects, project, prior=None)  # the scores, for the truth check below: nothing is saved and the holdout is not consumed
        truth = run.programme  # the same document: checked above by the replay
        frame = sd.load_table(sd.data_path(projects, project))
        planted = frame["order_id"].isin(replica.truth["order_id"]).astype(int)
        hold = {b: s.scores["holdout"] for b, s in run.scores.items()}
        k = h.budget_rows
        at = lambda s: compose.recall_at(s.to_numpy(float), planted.reindex(s.index).to_numpy(), k)
        composed_found, best_found = at(compose.build(p.composition.chosen, hold, p.composition.weights)), at(hold[p.composition.best_leaf])
        failed = []
        if not h.beats_best_leaf:
            failed.append(f"the composition did not beat the best leaf: difference {h.difference.value} (interval {h.difference.low} to {h.difference.high})")
        if h.reads != 1 or int(project.get("holdout_uses", 0) or 0) != 1:
            failed.append(f"the holdout was read {h.reads} time(s) by the programme and {project.get('holdout_uses')} by the project")
        if composed_found <= best_found:
            failed.append(f"the planted fraud found {composed_found:.2f}, not more than the best leaf's {best_found:.2f}")
        if core.validate(truth, plan_solution(plan)):
            failed.append("the programme does not validate: " + "; ".join(core.validate(truth, plan_solution(plan))[:2]))
        row["parts"]["holdout"] = not failed
        row["holdout"] = {"composed": h.composed.value, "best_leaf": h.best_leaf_estimate.value, "difference": h.difference.value, "low": h.difference.low,
                          "high": h.difference.high, "planted_found": round(composed_found, 3), "planted_found_best_leaf": round(best_found, 3)}
        if failed:
            row["why"].append(f"holdout: {failed}")
    return _end(row, started)


def plan_solution(plan: dict[str, Any]) -> Any:
    from dclab_core.solution_map import SolutionMap

    return SolutionMap.read(plan["solution_map"])


def _end(row: dict[str, Any], started: float) -> dict[str, Any]:
    row["seconds"] = round(time.perf_counter() - started, 1)
    row["right"] = all(row["parts"].get(p) for p in PARTS)
    return row


__all__ = ["DEV", "PARTS", "load", "play"]
