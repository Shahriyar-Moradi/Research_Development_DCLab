"""Real problems solved by the teams (Phase 8: 30.2, 30.3, 30.4): ``python -m dclab_rnd.real_problems fraud|telco|hyperack``.

Each run is a person's morning in the product, played with written answers: a draft from their words, their files
uploaded as they are, the understanding walk (28.3) asking them and building the project, the Solution page saved from
the audit's proposal when the project solves one problem, and the solution team (28.5, 30.1) solving it: the auditor
before anything is read, the holdout once after the person approves. The workspace of the run is kept whole in
``agent_runs/real_problems/<date>_<name>/workspace`` (its drafts, project, boards, decisions and events), and the result
in ``evidence/campaigns/real_problems_v1/results/RPB-NNN_<name>.json``: what was asked and answered, the card, the plan,
the audit's verdicts, the holdout number with its interval, the cost, and the earlier solution beside it with the
difference of protocol said.

The owner's fraud data is private (its README: it must not be distributed): the whole run is in a private scope, so no
model is asked anything about it, not even a column's name. No run sends a request without ``--live`` and a cap.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evidence" / "campaigns" / "real_problems_v1" / "results"
RUNS = ROOT / "agent_runs" / "real_problems"
SNAP = ROOT.parent / "Snap_fruad"

PROBLEMS: dict[str, dict[str, Any]] = {
    "fraud": {
        "private": True,
        "files": [SNAP / "data" / "part2-dataset.csv", SNAP / "data" / "part2-fraud-sample.csv"],
        "case": "RG-01",  # the owner's morning (the harness suite's first case): their own words and answers
        "earlier": {"source": "Snap_fruad/complete_solution (the owner's full solution, its README)",
                    "numbers": {"random_forest_average_precision": 0.095, "pu_bagging_average_precision": 0.081, "recall_in_top_0_5_percent": 0.46},
                    "protocol": "biker-grouped cross-validation on the 133,138 delivered rides, 94 known frauds joined on order_request_id, a cluster bootstrap"},
    },
    "telco": {
        "private": False,
        "files": [ROOT / "data" / "project" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"],
        "words": "We are a telecom. We want to know which customers will leave us next month, so the retention team can call them first.",
        "answers": {"target": "Churn: Yes when the customer left", "prediction_moment": "At the start of each month, from the account as it stands then",
                    "entity": "Each customer, the customerID", "labels": "Churn is known for every customer in the export",
                    "budget": "About 200 calls a month", "action": "Retention calls the customers most likely to leave and offers a better contract",
                    "family": "Predict who will churn", "data_plan": "The customer export with their services, contract and charges"},
        "earlier": {"source": "research/churn-prediction (CHURN-014, logistic regression with charge features)",
                    "numbers": {"roc_auc": 0.8499, "average_precision": 0.6708}, "protocol": "the churn track's own split and folds"},
    },
    "hyperack": {
        "private": False,
        "files": [ROOT / "data" / "project" / "hyperack" / "hyper_ackt-dataset.csv"],
        "words": "We run deliveries by bike. When an order is created we want to know whether a biker will accept it quickly, so dispatch can raise the fare or reassign it in time.",
        "answers": {"target": "hyper_ack: 1 when the order was accepted", "prediction_moment": "When the order is created, before dispatch",
                    "entity": "Each delivery order", "labels": "hyper_ack is recorded for every order",
                    "budget": "Dispatch can act on a few hundred orders a day", "action": "Raise the fare or reassign the orders unlikely to be accepted",
                    "family": "Predict whether an order is accepted", "data_plan": "The orders export: time, category, distance, places and the fares"},
        # the person's Solution page: the category of a delivery is chosen when the order is created, so they untick it (the
        # proposal's audit forbade it as "separating the classes" at an AUC of 0.816: 30.6 measures that bar)
        "keep": {"deliverey_category_id": "The delivery's category is chosen when the order is created: it is known at the prediction moment."},
        "earlier": {"source": "research/tabular-classification (optimized_safe_model, exp 53: soft vote ET + LGBM + XGB)",
                    "numbers": {"roc_auc": 0.9455, "safe_baseline_roc_auc": 0.9450, "with_the_final_fares_roc_auc_inflated": 0.9793},
                    "protocol": "the tabular track's ladder, without the final fares (known only after the outcome)"},
    },
}


class Person:
    """The written person at every pause: answers by field (or by what a question's words ask), approves the map and the
    build, keeps a leak suspect out, approves reading the holdout, and drops a part the files cannot do, with the reason."""

    DROP = "The export I have holds no table for this part."

    def __init__(self, answers: dict[str, str]):
        from dclab_rnd.agent_eval.journeys import _Person

        self.inner = _Person(answers)
        self.person = self.inner.person
        self.gates: list[str] = []

    def ask(self, field: Any, question: str) -> Optional[str]:
        if str(field or "").startswith("drop:"):
            return self.DROP
        return self.inner._answer(field, question)

    def answer(self, field: Any, question: str) -> Optional[str]:
        return self.ask(field, question)

    def approve(self, gate: str, summary: Any) -> bool:
        self.gates.append(gate)
        return True


def _person(problem: dict[str, Any]) -> tuple[str, dict[str, str]]:
    if problem.get("case"):
        cases = json.loads((ROOT / "dclab_rnd" / "agent_eval" / "regression" / "cases.json").read_text(encoding="utf-8"))
        cases = cases if isinstance(cases, list) else cases.get("cases") or []
        case = next(c for c in cases if c["id"] == problem["case"])
        return case["words"], case["answers"]
    return problem["words"], problem["answers"]


def _keep(projects: Any, project_id: str, keep: dict[str, str]) -> list[str]:
    """The person's Solution page: columns the proposal forbade that they know are available at the prediction moment, taken
    out of the forbidden list through the move ``set_solution`` (a person's), their reasons in the log, their choice a decision."""
    from dclab_rnd.ledger import capture
    from dclab_rnd.studio import graph as studio_graph

    project = projects.get(project_id)
    solution = dict(project.get("solution") or {})
    before = [f.get("column") for f in solution.get("forbidden") or []]
    solution["forbidden"] = [f for f in solution.get("forbidden") or [] if f.get("column") not in keep]
    verdict = studio_graph.check(project, "set_solution", "human", target=solution.get("target"))
    if not verdict.allowed:
        raise SystemExit(f"the person's Solution page was refused: {verdict.message}")
    project["solution"] = solution
    projects.save(project)
    studio_graph.log(projects, project_id, verdict, projects.get(project_id), outcome="done: the person kept " + ", ".join(keep))
    projects.clear_stages(project_id)
    projects.log(project_id, "solution_saved", {"kept": {c: r[:300] for c, r in keep.items()}, "forbidden": [f.get("column") for f in solution["forbidden"]]})
    capture.solution_saved(project.get("proposal"), (project.get("data") or {}).get("profile"), solution, "human", project=project_id)  # the person's choice beside the proposal
    return [c for c in keep if c in before]


def _number(project: dict[str, Any], projects: Any) -> dict[str, Any]:
    """The holdout number, read from the project's own record: the programme's composed value, or the final stage's."""
    from dclab_rnd.studio import programme

    p = programme.stored(project)
    if p is not None and p.holdout is not None:
        h = p.holdout
        return {"kind": "programme", "metric": h.metric, "composed": h.composed.value, "best_leaf": h.best_leaf_estimate.value,
                "difference": h.difference.value, "interval": [h.difference.low, h.difference.high], "chosen": h.chosen,
                "leaves": [{"block": l.block, "approach": l.approach, "status": l.status, "chosen": l.chosen} for l in p.leaves]}
    final = projects.read_stage(project["id"], "final") or {}
    ev = final.get("evidence") or {}
    return {"kind": "stages", "metric": final.get("primary_metric"), "holdout": ev.get("holdout_metrics"), "interval": ev.get("holdout_primary_metric_ci"),
            "model": ev.get("model"), "feature_recipe": ev.get("feature_recipe"), "holdout_rows": ev.get("holdout_rows"), "train_rows": ev.get("train_rows"),
            "holdout_split": ev.get("holdout_split"), "cv_to_holdout_gap": ev.get("cv_to_holdout_gap"), "baselines": ev.get("baselines")}


def run(name: str, live: bool = False, cap_eur: Optional[float] = None) -> dict[str, Any]:
    """One real problem, end to end (see the module); returns the result stored."""
    import contextlib

    from dclab_rnd import experience, ledger
    from dclab_rnd.agent_eval.journeys import _upload
    from dclab_rnd.agent_eval.journeys import _save
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.orchestra import checkpoints
    from dclab_rnd.orchestra.solution import SolutionTeam, drive as drive_team
    from dclab_rnd.orchestra.walk import Walk, drive
    from dclab_rnd.studio import ProjectStore

    problem = PROBLEMS[name]
    missing = [str(f) for f in problem["files"] if not Path(f).is_file()]
    if missing:
        raise SystemExit(f"{name}: the files are not here: {missing}")
    if problem["private"] and live:
        raise SystemExit(f"{name} is private: no model may be asked about it (run it without --live)")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    folder = RUNS / f"{date.today().isoformat()}_{name}"
    if folder.exists() and live:  # the review: a paid run's records are kept for good, never deleted by the next run
        folder = RUNS / f"{date.today().isoformat()}_{name}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    elif folder.exists():
        shutil.rmtree(folder)
    home = folder / "workspace"
    home.mkdir(parents=True)
    words, answers = _person(problem)
    person = Person(answers)
    out: dict[str, Any] = {"problem": name, "private": problem["private"], "words": words, "answers_written": sorted(answers), "started": started,
                           "files": [Path(f).name for f in problem["files"]], "workspace": folder.relative_to(ROOT).as_posix(), "live": bool(live)}
    read = critic = advisor = None
    if live:  # the served model plays the roles, each a third of the cap, its requests in this run's own usage log
        import os

        from dotenv import load_dotenv

        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.draft.reader import reader
        from dclab_rnd.models.gateway import for_workspace

        load_dotenv(ROOT / ".env", override=False)  # the provider and its key, as the server reads them
        os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
        gateway = for_workspace(home)
        read = reader(gateway, cap_eur=cap_eur / 3, force=True, prebuild=True)  # the conversation reader beside the Home agent's rules
        critic = GatewayAdvisor(gateway, cap_eur=cap_eur / 3, max_tokens=1200)  # the understanding critic
        advisor = GatewayAdvisor(gateway, cap_eur=cap_eur / 3)  # the solution team's roles
        if read is None or critic._client("understanding_critic") is None:
            raise SystemExit("No model serves conversation_reader or understanding_critic: set the provider's key (and leave DCLAB_NO_LIVE_MODELS unset).")
    log = experience.FileLog(home / "experience", f"real-{name}")  # installed only once nothing can stop the run before its try
    book = ledger.FileLedger(home / "ledger", f"real-{name}")
    experience.install(log)
    ledger.install(book)
    scope = private_scope(True) if problem["private"] or not live else contextlib.nullcontext()
    from dclab_rnd.studio.agent import notes_off

    try:
        with scope, notes_off():  # the review: a stage's notes would ask the installed gateway, outside the run's caps and its spend
            drafts, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
            agent = HomeAgent(drafts, None, reader=read)
            draft_id = drafts.create(words)["id"]

            def whole(d: dict[str, Any]) -> None:  # a real run reads the whole table (the product's quick pass reads 3,000 rows)
                d["settings"] = {"where": "local", "budget": {"max_steps": 24, "max_minutes": 20, "eur": 0.0},  # as the settings route writes them
                                 **(d.get("settings") or {}), "quick": False, "max_rows": 200_000, "folds": 5}
            drafts.update(draft_id, whole)
            agent.start(draft_id)
            for f in problem["files"]:
                _upload(drafts, agent, draft_id, Path(f).name, Path(f))
            with checkpoints.open_checkpointer(home, database_url="") as saver:
                walked = drive(Walk(Coordinator(drafts, projects, agent), saver, home=home, critic_advisor=critic), draft_id, person)
            out["understanding"] = {"next": walked.get("next"), "says": walked.get("says"), "state": walked.get("state"),
                                    "asked": person.inner.asked, "gates": list(person.gates)}
            built = walked.get("project_id")
            if not built:
                out["result"] = "not built"
                return _store(name, _spent(out, live, read, critic, advisor, book, log))
            from dclab_rnd.intern.tools import Toolbox

            project = projects.get(built)
            if project.get("programme_plan") is None and not project.get("solution"):
                refusal = _save(projects, project, {})  # the Solution page, from the audit's proposal (the person accepts it)
                out["solution_saved"] = refusal or "saved"
            if problem.get("keep"):
                out["kept_by_person"] = _keep(projects, built, problem["keep"])
            Toolbox(projects).call("set_settings", {"project_id": built, "quick": False, "max_rows": 200_000})  # the whole table (after the save, which sets the quick pass)
            out["settings"] = {k: (projects.get(built).get("settings") or {}).get(k) for k in ("quick", "max_rows", "folds")}
            from dclab_rnd.studio import programme as sp

            out["plan"] = (sp.view(projects.get(built)) if sp.plan_of(projects.get(built)) else {"solution": {k: (projects.get(built).get("solution") or {}).get(k)
                                                                                                            for k in ("target", "task", "time_column", "group_column", "metric")}})
            with checkpoints.open_checkpointer(home, database_url="") as saver:
                solved = drive_team(SolutionTeam(projects, saver, home=home, advisor=advisor), built, person)
            project = projects.get(built)
            out["solution"] = {"next": solved.get("next"), "says": solved.get("says"), "nodes": [s.get("action") for s in solved.get("steps") or []],
                               "gates": list(person.gates), "holdout_uses": int(project.get("holdout_uses") or 0),
                               "audits": [h["verdict"] for h in (project.get("auditor") or {}).get("history") or []]}
            out["number"] = _number(project, projects)
            out["result"] = "solved" if solved.get("next") == "done" else "stopped"
    finally:
        experience.install(None)
        ledger.install(None)
    return _store(name, _spent(out, live, read, critic, advisor, book, log))


def _spent(out: dict[str, Any], live: bool, read: Any, critic: Any, advisor: Any, book: Any, log: Any) -> dict[str, Any]:
    """What the run spent and how many of its model requests failed, from its three capped advisors (and the reader's own
    count), however the run ended; its decisions and events."""
    if live:
        advisors = [x for x in (getattr(read, "advisor", None), critic, advisor) if x is not None]
        out["spent_eur"] = round(sum(float(x.spent_eur()) for x in advisors), 4)
        out["failed_requests"] = int(getattr(read, "failures", 0) or 0) + sum(int(getattr(x, "failures", 0) or 0) for x in advisors)
    out["decisions"] = len(book.records())
    out["events"] = len(log.read(limit=10**7))
    return out


def _store(name: str, out: dict[str, Any]) -> dict[str, Any]:
    problem = PROBLEMS[name]
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("RPB-*.json")] or [0])
    ident = f"RPB-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "real_problems_v1", "kind": "real_problem", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **out, "earlier": problem["earlier"],
           "rule": "One run on the problem's own data, played with written answers; its number is one holdout of DCLab's own split, set "
                   "beside an earlier solution measured on another protocol: evidence about this run, not production readiness."}
    path = RESULTS / f"{ident}_{name}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    doc["path"] = path.relative_to(ROOT).as_posix()
    return doc


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.real_problems", description=__doc__.split("\n")[0])
    parser.add_argument("problem", choices=tuple(PROBLEMS))
    parser.add_argument("--live", action="store_true", help="let the served model answer the understanding roles (never for a private problem)")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    if args.live and (args.cap_eur is None or args.cap_eur <= 0):
        print("refused: a live run needs --cap-eur above zero", file=sys.stderr)
        return 2
    doc = run(args.problem, args.live, args.cap_eur)
    print(json.dumps({k: doc.get(k) for k in ("id", "problem", "result", "understanding", "solution", "number", "decisions", "spent_eur", "path")}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
