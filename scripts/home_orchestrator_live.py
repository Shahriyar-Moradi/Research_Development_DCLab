"""A live run of the Home orchestrator on a real project, in a scratch copy (the owner, 2026-10-20: "test it really with real situation").

    .venv/bin/python scripts/home_orchestrator_live.py --project e15422617e62 --cap-eur 0.5 --yes

The project's document, its stage records and its table are copied from the source workspace (the dev database by default) into
a file workspace under ``--out``; the source is only read. A built chat is then played turn by turn with the configured model
(every turn capped, the whole run capped), the conversation the owner had first ("what are models you have traimed", "ok
improve the model for me", "ok do it", "ok", "build improvements") and the requests that followed. A job the Experimenter starts
is recorded, never run (the copy stays as it was). Each turn is scored against what it should have done, and everything is kept:
the orchestrator's own records, the chat, the score sheet and the spend. Without ``--yes`` it prints the plan and spends nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# what each message should make the team do: the tools that must be called, and those that must not
CONVERSATION = [
    ("what are models you have traimed", {"show_models"}, {"improve_model"}),
    ("which one is the best and why", {"show_models"}, {"improve_model"}),
    ("how can we improve it?", {"improvement_ideas"}, {"improve_model"}),
    ("ok do it", {"improve_model"}, set()),
    ("ok", set(), {"improve_model"}),
    ("build improvements", {"improve_model"}, set()),
    ("show me precision, recall, f1 and the confusion matrix", {"show_evaluation"}, {"improve_model"}),
    ("plot the accuracy of each cross validation fold", {"write_notebook_cell"}, {"improve_model"}),
    ("is there any leakage, and what does the data look like?", {"show_leakage", "show_data"}, {"improve_model"}),
]
# step 1 of connecting understanding (2026-10-21): the person changes the problem on the built chat (columns of the owner's lead table)
PROBLEM = [
    ("actually the outcome should be demo_requested, not converted_this_month", {"propose_problem_change"}, {"improve_model"}),
    ("yes", {"confirm_problem_change"}, {"improve_model"}),
    ("we should predict conversion within 60 days instead", {"propose_problem_change"}, {"improve_model"}),
    ("no, the prediction is made when the lead is created, before any email is sent", {"propose_problem_change"}, {"improve_model"}),
    ("this is the wrong problem", {"show_problem"}, {"improve_model", "confirm_problem_change"}),
    ("actually the target should be lead_id", {"propose_problem_change"}, {"improve_model"}),
    ("yes", set(), {"improve_model"}),
]
# step 2 (2026-10-21): a confirmed problem version planned and built as a new project version (the build is the copy's own)
VERSION = [
    ("actually the outcome should be demo_requested, not converted_this_month", {"propose_problem_change"}, {"improve_model"}),
    ("yes", {"confirm_problem_change"}, {"build_new_version"}),
    ("build it", {"plan_new_version"}, {"build_new_version"}),
    ("yes, build it", {"build_new_version"}, {"improve_model"}),
]
SETS = {"default": CONVERSATION, "problem": PROBLEM, "version": VERSION}
TURN_CAP_EUR = 0.06


class Recorded:
    """A job queue that records what would have run: the copy is never changed by a job."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def get(self, job_id):
        return self.jobs[job_id]

    def latest(self, kind, key):
        mine = [j for j in self.jobs.values() if j["kind"] == kind and j.get("key") == key]
        return mine[-1] if mine else None


class Services:
    def __init__(self, home: Path, projects, gateway):
        self.home, self.projects, self.gateway, self.job_store, self.submitted, self.built = home, projects, gateway, Recorded(), [], []
        self.store = type("S", (), {"home": home})()

    def build_from_chat(self, draft_id):
        """The copy's own build: a project created in the copy, recorded; no stage runs (the real path is checked in the browser)."""
        made = self.projects.create("Version (live test)", "general", "a version built from the chat")
        self.built.append({"draft": draft_id, "project": made["id"]})
        return {"project_id": made["id"], "name": made["name"], "started": False, "note": "recorded by the live test; no stage runs"}

    def submit(self, kind, key, payload, here=False, **_):
        job = {"id": f"live-{len(self.submitted) + 1}", "kind": kind, "key": key, "status": "queued", "payload": payload}
        self.submitted.append(job)
        self.job_store.jobs[job["id"]] = dict(job)
        return job


class Counting:
    """The gateway, with every client it hands out kept so the run's spend is the sum of theirs; each capped per turn."""

    def __init__(self, gateway):
        self.gateway, self.clients = gateway, []

    def client(self, purpose, **ids):
        c = self.gateway.client(purpose, **ids)
        if c is not None:
            c.run_limit_eur = TURN_CAP_EUR
            self.clients.append(c)
        return c

    def spent(self) -> float:
        return round(sum(float(getattr(c, "spent_eur", 0.0) or 0.0) for c in self.clients), 6)


def copy_project(project_id: str, source_url: str, source_home: Path, dest_home: Path):
    """The project's document, stage records and table, copied into a file workspace (the source is only read)."""
    os.environ["DCLAB_DATABASE_URL"] = source_url
    from dclab_rnd.storage.postgres import open_stores as pg_stores
    from dclab_rnd.studio import data as sd

    src, _, _ = pg_stores(source_home)
    doc, records = src.get(project_id), src.records(project_id)
    table = sd.data_path(src, doc) if (doc.get("data") or {}).get("filename") else None
    os.environ["DCLAB_DATABASE_URL"] = ""
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.studio.store import ProjectStore

    projects = ProjectStore(dest_home / "projects")
    drafts = DraftStore(dest_home / "drafts")
    projects.audit_path = drafts.audit_path = dest_home / "audit.jsonl"
    (projects.directory(project_id) / "data").mkdir(parents=True, exist_ok=True)
    (projects.directory(project_id) / "stages").mkdir(exist_ok=True)
    for key in ("improvements", "adopted_improvement", "notebook_agent_cells"):
        doc.pop(key, None)  # the run starts as the project was after its final stage
    doc["running"] = None
    draft = drafts.create(doc.get("goal") or doc.get("name") or "A built project")
    drafts.update(draft["id"], lambda d: d.update(status="built", project_id=project_id))
    doc["draft"] = {"id": draft["id"]}
    projects.save(doc)
    for stage, record in records.items():
        projects.write_stage(project_id, stage, record)
    if table is not None:
        shutil.copy2(table, projects.data_dir(project_id) / doc["data"]["filename"])
    return projects, drafts, draft["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", default="e15422617e62")
    ap.add_argument("--source-url", default="postgresql+psycopg://shahriar@/dclab_dev")
    ap.add_argument("--source-home", default=str(ROOT / "agent_runs"))
    ap.add_argument("--out", default=str(ROOT / "agent_runs" / "live_tests" / f"{datetime.now().strftime('%Y-%m-%d')}_home_orchestrator"))
    ap.add_argument("--cap-eur", type=float, default=0.5)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--rules", action="store_true", help="no model: the rules play every turn (spends nothing)")
    ap.add_argument("--set", default="default", choices=sorted(SETS), help="which conversation: the first one, or the problem changes (step 1)")
    args = ap.parse_args()
    conversation = SETS[args.set]
    print(f"Plan: {len(conversation)} turns on a copy of project {args.project}; each turn at most €{TURN_CAP_EUR}, the run at most €{args.cap_eur}.")
    if not args.yes and not args.rules:
        print("Add --yes to run it (it spends on the configured model).")
        return 0
    if args.yes:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)  # the product's model and key, as the app reads them
    out = Path(args.out)
    run = out / datetime.now(timezone.utc).strftime("run-%H%M%S")
    home = run / "workspace"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["DCLAB_AGENT_HOME"] = str(home)  # the run's records (ledger, experience, usage) stay in the copy's workspace
    projects, drafts, draft_id = copy_project(args.project, args.source_url, Path(args.source_home), home)
    source_solution = projects.get(args.project).get("solution")
    from dclab_rnd.draft import orchestrator as orch
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.models import installed

    gateway = Counting(installed())
    if args.rules:
        gateway.client = lambda purpose, **ids: None  # type: ignore[method-assign]
    elif gateway.gateway.client(orch.PURPOSE, project_id=args.project) is None:
        print("No model serves home_orchestrator here: nothing to measure.")
        return 1
    services = Services(home, projects, gateway)
    agent = HomeAgent(drafts, None)
    sheet = []
    for text, must, must_not in conversation:
        if gateway.spent() >= args.cap_eur - TURN_CAP_EUR:
            sheet.append({"text": text, "skipped": "the run's cap would be passed"})
            break
        before = gateway.spent()
        agent.reply_built(draft_id, text, orch.Orchestrator(services, agent), args.project)
        rows = [json.loads(l) for f in sorted((home / "home_orchestrator").glob("*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines()]
        rec = rows[-1] if rows and rows[-1].get("text") == text else {}
        called = {s.get("tool") for s in rec.get("steps") or [] if s.get("tool")}
        ok = must <= called and not (must_not & called)
        sheet.append({"text": text, "path": rec.get("path"), "called": sorted(called), "must": sorted(must), "must_not": sorted(must_not), "ok": ok,
                      "reply": rec.get("reply"), "dropped": rec.get("dropped"), "model_error": rec.get("model_error"), "eur": round(gateway.spent() - before, 6)})
        print(f"{'PASS' if ok else 'FAIL'}  {text!r:62} {rec.get('path')}: {sorted(called)}  €{gateway.spent() - before:.4f}")
    passed = sum(1 for r in sheet if r.get("ok"))
    draft = drafts.get(draft_id)
    summary = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "project": args.project, "set": args.set,
               "problem_versions": draft.get("problem_versions"), "problem_change": draft.get("problem_change"),
               "project_solution_unchanged": projects.get(args.project).get("solution") == source_solution, "turns": len(sheet), "passed": passed,
               "spent_eur": gateway.spent(), "turn_cap_eur": TURN_CAP_EUR, "cap_eur": args.cap_eur, "jobs_recorded": services.submitted, "versions_built": services.built, "sheet": sheet,
               "chat": drafts.get(draft_id)["messages"]}
    (run / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{passed}/{len(sheet)} turns did what they should; spent €{gateway.spent():.4f}. Kept in {run.relative_to(ROOT) if run.is_relative_to(ROOT) else run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
