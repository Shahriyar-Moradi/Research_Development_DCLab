"""The acceptance suite of conversations (package 20.10): a written person per family, from the first sentence to the
holdout number.

Each journey (``journeys_dev.json``, ``journeys_sealed.json``; the sealed file's SHA-256 pinned in ``sealed.sha256``)
is a person: the problem in their words, their answers keyed by the field a question asks, and the replica of a problem
kind (``dclab_core.replicas`` through ``dclab_rnd.kinds``) as the files they bring. :func:`play` runs it as the product
does, with no model: a draft per file the person brings (the cases apart from their table, as on the owner's morning),
the coordinator (21.2) walking the draft to a built project with the person answering, the tables the kind reads
attached by name, and every stage run by the engine. It is scored on four things, each against the replica's planted
truth:

- **card**: the problem card's family is the person's (one of those the journey accepts), with the labels where given;
- **workflow**: the draft's solution workflow validates and no move of the project's graph was refused;
- **pipeline**: every step behind the project's numbers has its run, none stale (20.8);
- **holdout**: the kind's acceptance decisions (``kinds.<kind>.accept``: the leak kept out, the right split, the
  holdout number above its planted floor and below its planted ceiling) all hold on the finished run.

A family the engine does not run yet is expected to stop at the build with its reason said (``"built": false``), and
is scored on the card and that refusal. The gate (``GATE``, fixed before any run) is judged on the sealed journeys,
run once; nothing is tuned on them.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "journeys_v1" / "results"
SEALED = {"sealed": "sealed.sha256"}
GATE = {
    "version": "journeys_gate_v1",
    "split": "sealed",
    "journey": "right on the card, the workflow, the pipeline and the holdout (or, for a family the engine does not run, "
               "on the card and a build refused with its reason)",
    "suite": "at least 85% of the sealed journeys right, the lower end of the 95% Wilson interval at least 0.6, every family "
             "with a journey, and no move the graph's validator refused in any journey",
    "seed": 1,  # the replicas' seed for the sealed run: not the tables the dev failures were fixed on (seed 0)
}
PARTS = ("card", "workflow", "pipeline", "holdout")
REFUSED = "family not built yet"  # dclab_core.coordinator.FAMILY_NOT_RUN: the build's rule for a family the engine does not run


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(split: str) -> list[dict[str, Any]]:
    path = HERE / f"journeys_{split}.json"
    if split in SEALED and fingerprint(path) != (HERE / SEALED[split]).read_text(encoding="utf-8").strip():
        raise ValueError("the sealed journeys changed since they were sealed: a sealed set is never edited")
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = right / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((centre - half) / (1 + z * z / n), 3), round((centre + half) / (1 + z * z / n), 3))


class _Person:
    """The written person: answers a question by the field it asks, approves every gate, and keeps what was asked."""

    def __init__(self, answers: dict[str, str]):
        from dclab_rnd.studio import team

        self.asked: list[str] = []
        self.person = team.Person(answer=self._answer, approve=lambda gate, summary: True, who="written person")
        self.answers = answers

    SAME = {"outcome": "target", "target_check": "target", "labels_check": "labels"}  # one question asked under two names

    def _answer(self, field: str, question: str) -> Optional[str]:
        self.asked.append(field)
        return self.answers.get(field) or self.answers.get(self.SAME.get(field, field))


def _upload(drafts: Any, agent: Any, draft_id: str, name: str, source: Path) -> None:
    from dclab_rnd.draft import pipeline

    (drafts.data_dir(draft_id) / name).write_bytes(source.read_bytes())
    asset = pipeline.new_asset(drafts, draft_id, "upload", name, name)
    pipeline.run(drafts, draft_id, asset["id"], agent=agent)


def _files(replica: Any, folder: Path) -> tuple[str, Optional[str]]:
    """The person's files on their disk: the table, and the label file when the replica has one apart."""
    if replica.files is not None:
        return replica.files(folder)
    replica.table.to_csv(folder / "table.csv", index=False)
    if replica.labels is not None:
        replica.labels[0].to_csv(folder / "labels.csv", index=False)
        return "table.csv", "labels.csv"
    return "table.csv", None


def play(journey: dict[str, Any], seed: int = 0, answerers: Any = None) -> dict[str, Any]:
    """One journey played from its first sentence; its scores and what happened at each part."""
    from dclab_rnd import kinds
    from dclab_rnd.draft import workflow as wf
    from dclab_rnd.draft.card import card_of
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds.acceptance import Run
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import ProjectStore, data as sd, engine, pipeline

    started = time.perf_counter()
    kind = kinds.get(journey["kind"])
    replica = kind.case(seed)
    row: dict[str, Any] = {"id": journey["id"], "kind": journey["kind"], "family": journey["family"], "parts": {}, "why": []}
    with tempfile.TemporaryDirectory() as tmp:
        home, mine = Path(tmp), Path(tmp) / "person"
        mine.mkdir()
        drafts, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        agent = HomeAgent(drafts, None)
        table, labels = _files(replica, mine)
        first = drafts.create(journey["words"])["id"]
        agent.start(first)
        start_at = first
        if labels is not None:  # the cases apart from their table, each in its own draft, as the owner brought them
            _upload(drafts, agent, first, labels, mine / labels)
            other = drafts.create(journey["words"])["id"]
            agent.start(other)
            _upload(drafts, agent, other, table, mine / table)
        else:
            _upload(drafts, agent, first, table, mine / table)
        person = _Person(journey["answers"])
        with private_scope(True):
            out = Coordinator(drafts, projects, agent).run(start_at, person.person)
        understood = card_of(drafts.get(start_at))
        card = understood.family.value
        accepted = journey["family"] if isinstance(journey["family"], list) else [journey["family"]]
        labels_ok = "labels" not in journey or understood.labels.value == journey["labels"]
        row.update(asked=person.asked, questions=len(person.asked), steps=[s["action"] for s in out["steps"]], state=out["state"],
                   said=out.get("says"), got={"family": card})
        row["parts"]["card"] = card in accepted and labels_ok
        if not row["parts"]["card"]:
            row["why"].append(f"card: {card!r}, the person's {accepted}" + ("" if labels_ok else "; the labels differ"))
        built = out.get("project_id")
        expect_built = journey.get("built", True)
        if not expect_built:  # right only when the build itself refused the family, never on another stop (the review's R2)
            said_no = built is None and out["next"] == "stop" and any(b.get("code") == REFUSED for b in out.get("blockers") or [])
            row["parts"].update(workflow=said_no, pipeline=said_no, holdout=said_no)
            if not said_no:
                row["why"].append("built: the engine does not run this family, and the build was not refused with a reason")
            return _end(row, started)
        if built is None:
            row["parts"].update(workflow=False, pipeline=False, holdout=False)
            row["why"].append(f"not built: stopped at {out['state']}: {out.get('says')}")
            return _end(row, started)
        draft = drafts.get(start_at)
        _, problems = wf.validate(draft.get("workflow") or {}, (draft.get("pack") or {}).get("key"))
        problems = list(problems) + _solution_workflow(understood, projects.get(built))
        for name, frame in (replica.tables or {}).items():  # the related tables the kind reads, attached by name on the project
            frame.to_csv(projects.data_dir(built) / f"{name}.csv", index=False)
            sd.attach_table(projects, built, f"{name}.csv", name)
        project = projects.get(built)
        refusal = None
        if not project.get("solution"):  # the person saves the Solution page: the proposal, and what they state on it
            refusal = _save(projects, project, journey.get("solution") or {})
        if refusal:
            row["parts"].update(workflow=False, pipeline=False, holdout=False)
            row["why"].append(f"solution refused: {refusal}")
            return _end(row, started)
        try:
            with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
                engine.run_all(projects, built)
            ran = None
        except Exception as error:  # noqa: BLE001 - a stage that fails is the journey's failure, said
            import traceback

            ran = f"{type(error).__name__}: {str(error)[:200]} at {traceback.extract_tb(error.__traceback__)[-1].name}"
        refused_moves = refused(projects, built)
        row["refused_moves"] = len(refused_moves)
        row["parts"]["workflow"] = not problems and not refused_moves
        if problems or refused_moves:
            row["why"].append(f"workflow: {problems[:2]}; refused moves {[t.get('move') for t in refused_moves[:3]]}")
        if ran is not None:
            row["parts"].update(pipeline=False, holdout=False)
            row["why"].append(f"engine: {ran}")
            return _end(row, started)
        with runner.fast_profile():  # the profile a run used is one of its inputs: read under the same one
            steps = pipeline.of(projects, built)
        stale = [f"{s.id} ({s.why})" for s in steps.steps if s.status == "stale"]
        own = kind.own_machine() is not None  # a kind with its own machine has no analysis of tabular training rows to cite
        missing = [f"{s.id} ({s.status})" for s in steps.steps if s.status not in ("done", "stale")  # every step run and recorded
                   and not (own and s.id == "analysis" and s.status == "not_recorded")]
        row["parts"]["pipeline"] = not stale and not missing
        if stale or missing:
            row["why"].append(f"pipeline: stale {stale}, no run {missing}")
        project = projects.get(built)
        records = {stage: projects.read_stage(built, stage) for stage in engine.STAGE_KEYS}
        done = Run(replica, project["solution"], records, project.get("labels") or {})
        checks = kind.accept_engine(done) if kind.own_machine() is not None else kind.accept(done)
        failed = [c["decision"] for c in checks if not c["ok"]]
        ran_as = kinds.resolve(project["solution"], project.get("pack")).key
        if ran_as != journey["kind"]:  # the problem ran as the kind of the person's problem (stated or proposed)
            failed.append(f"ran as {ran_as}, not {journey['kind']}")
        row["parts"]["holdout"] = bool(checks) and not failed
        row["holdout"] = (done.evidence("final").get("holdout_metrics") or {})
        row["got"].update(target=project["solution"].get("target"), kind=ran_as, kind_stated=bool((journey.get("solution") or {}).get("kind")))
        if failed or not checks:
            row["why"].append(f"holdout: {failed or 'no acceptance decision'}")
    return _end(row, started)


def refused(projects: Any, project_id: str) -> list[dict[str, Any]]:
    """The project's moves the graph's validator refused: its verdict is the entry's status (``allowed`` lists the moves
    open at that state)."""
    return [t for t in projects.transitions(project_id, limit=10_000) if t.get("status") == "blocked"]


def _solution_workflow(card: Any, project: dict[str, Any]) -> list[str]:
    """20.6's solution workflow from the problem card and the table's data card, validated on its content."""
    from dclab_core import solution_workflow as sw
    from dclab_core.datacard import DataCard

    try:
        data_card = (project.get("data") or {}).get("card")
        flow = sw.solution_workflow(card, DataCard.model_validate(data_card) if data_card else None)
    except Exception as error:  # noqa: BLE001 - no workflow for this card is the journey's failure, said
        return [f"no solution workflow: {str(error)[:200]}"]
    return [f"solution workflow: {p}" for p in sw.validate(flow)]


def _save(projects: Any, project: dict[str, Any], statements: dict[str, Any]) -> Optional[str]:
    """The person saves the Solution page through the same checks as the route: the graph's validator, then the
    columns; the refusal's words, or None when saved."""
    from dclab_rnd.kinds.acceptance import adopt
    from dclab_rnd.studio import graph
    from dclab_rnd.studio.solution import Solution
    from dclab_rnd.studio import data as sd
    from dclab_rnd.studio.solution import propose

    suggestion = project.get("suggestion") or {}
    target = statements.get("target", suggestion.get("target"))
    if project.get("labels") and suggestion:  # a label file's join: its suggestion is the starting point
        proposal = {**suggestion}
    elif target:
        frame = sd.load_table(sd.data_path(projects, project))
        proposal = propose(frame, project["data"]["profile"], target, statements.get("task"), project.get("pack"))
    else:
        proposal = {}
    task = statements.get("task") or proposal.get("task") or "binary"
    moment = suggestion.get("prediction_moment") or ((project.get("draft") or {}).get("understanding") or {}).get("prediction_moment") or ""
    verdict = graph.check(project, "set_solution", "human", target=target)
    graph.log(projects, project["id"], verdict, project, outcome="done: solution saved" if verdict.allowed else None)
    if not verdict.allowed:
        return verdict.message
    try:
        adopt(projects, project, proposal, target, task, moment, {k: v for k, v in statements.items() if k not in ("target", "task")})
        saved = projects.get(project["id"])
        label_target = (saved.get("labels") or {}).get("target")  # as the route: the label file's target joins the table
        Solution(**saved["solution"]).check_columns(list(saved["data"]["columns"]) + ([label_target] if label_target else []))
    except ValueError as error:
        return str(error).split("\n")[0][:300]
    return None


def _end(row: dict[str, Any], started: float) -> dict[str, Any]:
    row["right"] = all(row["parts"].get(p) for p in PARTS)
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def _play_one(args: tuple[dict[str, Any], int]) -> dict[str, Any]:
    journey, seed = args
    try:
        return play(journey, seed)
    except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason
        return {"id": journey["id"], "kind": journey["kind"], "family": journey["family"], "parts": {p: False for p in PARTS},
                "right": False, "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]}


def measure(split: str, seed: int = 0, workers: int = 1, only: Optional[list[str]] = None) -> list[dict[str, Any]]:
    journeys = [j for j in load(split) if not only or j["id"] in only or j["kind"] in only]
    if workers <= 1:
        return [_play_one((j, seed)) for j in journeys]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_play_one, [(j, seed) for j in journeys]))


def families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every family listed (one with no journey too, its interval [0, 1]), with its journeys right and the interval."""
    from dclab_core.card import FAMILIES

    out: dict[str, list[bool]] = {f: [] for f in FAMILIES}
    for r in rows:
        for family in (r["family"] if isinstance(r["family"], list) else [r["family"]])[:1]:
            out.setdefault(family, []).append(bool(r["right"]))
    return [{"family": f, "journeys": len(v), "right": sum(v), "interval": list(wilson(sum(v), len(v)))} for f, v in sorted(out.items())]


def verdict(rows: list[dict[str, Any]]) -> str:
    right, n = sum(r["right"] for r in rows), len(rows)
    low, _ = wilson(right, n)
    refused = sum(r.get("refused_moves") or 0 for r in rows)
    missing = [f["family"] for f in families(rows) if not f["journeys"]]
    ok = n > 0 and right / n >= 0.85 and low >= 0.6 and refused == 0 and not missing
    return (f"{'pass' if ok else 'fail'}: {right} of {n} journeys right (95% interval {low} to {wilson(right, n)[1]}), "
            f"{refused} moves refused by the validator" + (f", no journey for {', '.join(missing)}" if missing else ""))


def sealed_runs(split: str = "sealed") -> list[Path]:
    return sorted(RESULTS.glob(f"JRN-*_{split}.json")) if RESULTS.exists() else []


def store(rows: list[dict[str, Any]], split: str, result: str, arm: str = "rules", seed: int = 0) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("JRN-*.json")] or [0])
    ident = f"JRN-{n:03d}"
    doc = {"id": ident, "campaign_id": "journeys_v1", "kind": "acceptance_journeys", "split": split, "arm": arm,
           "journeys": fingerprint(HERE / f"journeys_{split}.json"), "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gate": GATE, "seed": seed, "spent_eur": 0.0, "rows": rows, "families": families(rows),
           "summary": {"journeys": len(rows), "right": sum(r["right"] for r in rows),
                       "by_part": {p: sum(bool(r["parts"].get(p)) for r in rows) for p in PARTS}},
           "verdict": result, "status": "completed", "experiment_id": ident,
           "rule": f"Scores on {len(rows)} written journeys over seeded replicas say how DCLab carries these people from their "
                   "first sentence to a holdout number; they are not a claim about any other problem or about production readiness."}
    path = RESULTS / f"{ident}_{split}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path
