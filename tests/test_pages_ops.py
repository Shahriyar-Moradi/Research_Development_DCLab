"""Compute & jobs and Home's Activity: /api/ops/jobs (one list over stage runs, intern sessions, draft data
pipelines and legacy research runs), /api/ops/jobs/{id} and /api/ops/evidence-recent. Fixtures are written straight
into the stores, so nothing trains and nothing calls a model."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agentic.pages import Context, ops  # noqa: E402
from dclab_rnd.agentic.store import Store  # noqa: E402
from dclab_rnd.storage import open_stores  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.intern import SessionStore  # noqa: E402
from dclab_rnd.studio import ProjectStore  # noqa: E402

SOLUTION = {"target": "Churn", "task": "binary", "positive_label": "True", "prediction_moment": "At a fixed snapshot each month",
            "forbidden": [], "identifiers": ["customerID"], "time_column": None, "group_column": None, "text_columns": [],
            "metric": "roc_auc", "notes": ""}
OLD = "2026-01-02T08:00:00+00:00"  # before any server in these tests started


def iso(dt):
    return dt.replace(microsecond=0).isoformat()


class Busy:
    """Stands in for an asyncio task that has not finished."""

    def done(self):
        return False


def make_fixtures(home: Path) -> dict:
    """One project (two failed data runs, one before it started, then a completed run with its record), one intern
    session, one draft with a ready asset and one stuck before the server started, and one legacy research run."""
    now = datetime.now(timezone.utc)
    t = lambda s: iso(now - timedelta(seconds=s))  # noqa: E731
    projects, drafts, sessions = open_stores(home)  # the app's own stores: files, or PostgreSQL when DCLAB_DATABASE_URL is set
    p = projects.create("Churn", goal="Predict churn")
    pid = p["id"]
    p["solution"] = SOLUTION
    p["data"] = {"filename": "telco.csv", "rows": 100, "columns": ["Churn"], "sha256": "ab" * 32}
    p["policy"] = {"require_solution_signoff": True}
    p["stages"]["data"] = {"status": "completed", "started": t(50), "finished": t(48), "elapsed_seconds": 1.75}
    projects.save(p)
    for at, kind, payload in [(t(120), "stage_started", {"stage": "data"}), (t(119), "stage_failed", {"stage": "data", "error": "TypeError: boom"}),
                              (t(50), "stage_started", {"stage": "data"}), (t(48), "stage_completed", {"stage": "data", "elapsed_seconds": 1.8, "summary": "Profiled 80 rows"})]:
        projects.log(pid, kind, payload, at=at)
    for at, actor, outcome in [(t(119), "human", "failed: TypeError: boom"), (t(80), "agent", "failed: DataError: unreadable file"), (t(48), "agent", "done: profiled")]:
        projects.transition(pid, {"at": at, "actor": actor, "move": "run_stage", "args": {"stage": "data"}, "status": "allowed", "outcome": outcome})
    projects.write_stage(pid, "data", {
        "stage": "data", "experiment_id": f"PRJ-{pid[:6]}-data", "status": "completed", "started_at": t(50), "completed_at": t(48),
        "elapsed_seconds": 1.75, "primary_metric": "roc_auc", "decision_time_rule": "At a fixed snapshot each month", "setup_summary": "Profiled 80 training rows",
        "evidence": {"train_rows": 80, "holdout_rows": 20}, "decision": None,
        "claims": [{"claim_id": "C1", "kind": "fact", "statement": "80 rows were profiled.", "limitations": ["Small sample."]}],
        "notes": [{"severity": "warning", "title": "Few rows", "text": "Only 100 rows.", "proof": ["DCLAB-R03"]}],
        "provenance": {"data_sha256": "ab" * 32, "filename": "telco.csv", "random_state": 42, "cv": "StratifiedKFold(3)", "holdout": "random 20 of 100",
                       "sampling": {"source_rows": 100, "rows_used": 100, "rule": "all rows"}}})

    s = sessions.create("Forecast daily bike rentals", mode="standard", model=None, budget={"max_steps": 12, "max_minutes": 5}, project_id=pid)
    s.update(status="completed", plan="1. pick the data", used={"steps": 2, "minutes": 0.5, "input_tokens": 0, "output_tokens": 0},
             steps=[{"n": 1, "tool": "list_samples", "arguments": {}, "summary": "16 samples", "ok": True, "elapsed_seconds": 0.1, "at": t(30)},
                    {"n": 2, "tool": "use_sample", "arguments": {"key": "bike", "project_id": pid}, "summary": "error: no such sample", "ok": False, "elapsed_seconds": 0.2, "at": t(29)}],
             final="Stopped: the sample does not exist.", created=t(31))
    sessions.save(s)

    d = drafts.create("Rank clients for the call campaign")
    ready = {"id": "a1111aaaa", "kind": "sample", "name": "bank_marketing", "filename": "bank_marketing.parquet", "status": "ready",
             "added": t(20), "synthetic": False, "rows": 45211, "columns": 17, "structure": {"format": "parquet", "parse_rate": 1.0}}
    stuck = {"id": "a2222bbbb", "kind": "upload", "name": "orders.csv", "filename": "orders.csv", "status": "structuring", "added": OLD, "synthetic": False}
    drafts.update(d["id"], lambda x: x["assets"].extend([ready, stuck]))
    drafts.emit(d["id"], "pipeline", {"asset": ready, "step": "queued", "text": "Received bank_marketing"})
    drafts.emit(d["id"], "chat", {"text": "hello"})
    drafts.emit(d["id"], "pipeline", {"asset": "a1111aaaa", "step": "cleaned", "text": "1 cleaning step applied", "log": [{"title": "Dropped duplicates", "detail": "3 rows"}]})
    drafts.emit(d["id"], "pipeline", {"asset": "a1111aaaa", "step": "ready", "text": "Ready", "rows": 45211, "columns": 17})

    store = Store(home)  # legacy research runs: written as rows, without the archive the engine also writes
    run = {"id": "f" * 32, "created": t(400), "updated": t(100), "status": "completed", "phase": "Research complete", "error": None, "llm_calls": 3,
           "usage": {"input_tokens": 1200, "output_tokens": 300}, "active_seconds": 250.0,
           "config": {"project": "general", "goal": "Compare two model families on bank marketing", "datasets": ["bank_marketing"], "model": "gpt-x",
                      "max_experiments": 2, "max_rows": 2000, "repeats": 1, "max_minutes": 5}}
    with sqlite3.connect(store.path) as db:
        db.execute("INSERT INTO runs VALUES (?, ?)", (run["id"], json.dumps(run)))
        for kind, payload in [("phase", {"name": "Research planner"}), ("llm_request", {"messages": []}), ("trial", {"status": "completed", "plan": {"title": "Logistic baseline"}})]:
            db.execute("INSERT INTO events(run_id,kind,created,payload) VALUES (?,?,?,?)", (run["id"], kind, t(300), json.dumps(payload)))
    return {"pid": pid, "sid": s["id"], "did": d["id"], "run": run["id"], "projects": projects, "sessions": sessions, "drafts": drafts, "store": store}


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        for key in ("DCLAB_STUDIO_HOME", "DCLAB_INTERN_HOME", "DCLAB_DRAFT_HOME"):
            os.environ.pop(key, None)
        self.home = Path(tempfile.mkdtemp())
        self.fx = make_fixtures(self.home)
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def jobs(self):
        r = self.client.get("/api/ops/jobs")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_one_list_over_every_kind_newest_first(self):
        d = self.jobs()
        by_id = {j["id"]: j for j in d["jobs"]}
        self.assertEqual({j["kind"] for j in d["jobs"]}, {"stage", "intern", "data", "research"})
        self.assertTrue(all(j["where"] == "this machine" and j["cost"] is None for j in d["jobs"]))
        starts = [j["started"] for j in d["jobs"] if j["started"]]
        self.assertEqual(starts, sorted(starts, reverse=True))
        pid, did = self.fx["pid"], self.fx["did"]
        # the data stage ran three times: a failure, a failure before it started (only the graph's move), then a completed run
        self.assertEqual(by_id[f"stage-{pid}-data-1"]["status"], "failed")
        self.assertEqual(by_id[f"stage-{pid}-data-1"]["by"], "human")
        done = by_id[f"stage-{pid}-data-2"]
        self.assertEqual((done["status"], done["current"], done["seconds"], done["by"]), ("done", True, 1.75, "agent"))
        early = [j for j in d["jobs"] if j["id"].startswith(f"stage-{pid}-data-m")]
        self.assertEqual(len(early), 1)
        self.assertIn("DataError", early[0]["detail"])
        self.assertEqual(by_id[f"intern-{self.fx['sid']}"]["project"], "Churn")
        self.assertEqual(by_id[f"intern-{self.fx['sid']}"]["seconds"], 30.0)
        self.assertEqual((by_id[f"data-{did}-a1111aaaa"]["status"], by_id[f"data-{did}-a1111aaaa"]["rows"]), ("done", 45211))
        stuck = by_id[f"data-{did}-a2222bbbb"]
        self.assertEqual((stuck["status"], stuck["label"]), ("failed", "interrupted"))  # it was mid-pipeline before this server started
        self.assertEqual(by_id[f"research-{self.fx['run']}"]["seconds"], 250.0)
        t = d["totals"]
        self.assertEqual((t["total"], t["running"], t["queued"], t["failed"]), (len(d["jobs"]), 0, 0, 3))
        self.assertEqual(set(t["seconds_this_month"]), {"stage", "intern", "data", "research"})
        self.assertFalse(d["spend"]["metered"])
        self.assertEqual([x["state"] for x in d["targets"]][1:], ["not switched on"] * 3)
        # the project's policy requires the solution sign-off and nobody signed: one approval waits
        self.assertEqual([(a["project_id"], a["gate"], a["page"]) for a in d["approvals"]], [(pid, "solution", "solution")])
        self.assertEqual(t["approvals"], 1)

    def test_job_details(self):
        pid, did = self.fx["pid"], self.fx["did"]
        stage = self.client.get(f"/api/ops/jobs/stage-{pid}-data-2").json()
        texts = " ".join(l["text"] for l in stage["logs"])
        self.assertIn("Few rows", texts)
        self.assertIn("80 rows were profiled", texts)
        self.assertEqual({a["href"] for a in stage["artifacts"]}, {f"/api/projects/{pid}/export/{k}" for k in ("notebook", "report", "sft")})
        repro = {r["label"]: r["value"] for r in stage["repro"]}
        self.assertEqual((repro["Seed"], repro["Code version"]), (42, "not recorded in the stage record"))
        self.assertIn("ab" * 32, repro["Data"])
        self.assertIsNone(stage["stop"])  # nothing can stop a stage run yet, so the page shows no Stop button
        failed = self.client.get(f"/api/ops/jobs/stage-{pid}-data-1").json()
        self.assertEqual((failed["artifacts"], failed["repro"]), ([], []))
        self.assertIn("failed", failed["repro_note"])
        intern = self.client.get(f"/api/ops/jobs/intern-{self.fx['sid']}").json()
        self.assertEqual([l["level"] for l in intern["logs"] if "list_samples" in l["text"] or "use_sample" in l["text"]], ["ok", "bad"])
        self.assertNotIn("project_id", " ".join(l["text"] for l in intern["logs"]))
        data = self.client.get(f"/api/ops/jobs/data-{did}-a1111aaaa").json()
        self.assertEqual([l["text"].split(" · ")[0] for l in data["logs"] if l["at"]], ["queued", "cleaned", "ready"])
        self.assertIn("Dropped duplicates", " ".join(l["text"] for l in data["logs"]))
        trial = Path(self.fx["store"].home) / self.fx["run"] / "trial-001"  # a trial's files (on the old UI's run page until 8.3)
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text("{}", encoding="utf-8")
        research = self.client.get(f"/api/ops/jobs/research-{self.fx['run']}").json()
        self.assertEqual([l["text"].split(" · ")[0] for l in research["logs"]], ["phase", "trial"])  # model requests are not shown
        self.assertEqual(research["artifacts"][0]["href"], f"/api/runs/{self.fx['run']}/export")
        self.assertEqual(research["artifacts"][1]["href"], f"/api/runs/{self.fx['run']}/trials/trial-001/result.json")
        self.assertEqual(self.client.get(research["artifacts"][1]["href"]).status_code, 200)
        self.assertIn("python -m dclab_rnd.agentic", research["repro_note"])  # how to act on it now that the old UI is gone
        for bad in ("stage-000000000000-data-1", "intern-000000000000", "data-zz-a1", "nope-1", "stage-..%2F..-x"):
            self.assertEqual(self.client.get(f"/api/ops/jobs/{bad}").status_code, 404, bad)

    def test_evidence_recent(self):
        d = self.client.get("/api/ops/evidence-recent").json()
        if not d["total"]:
            self.skipTest("evidence index not built")
        self.assertEqual(len(d["records"]), 3)
        dated = [r["date"] for r in d["records"] if r["date"]]
        self.assertEqual(dated, sorted(dated, reverse=True))
        ids = {json.loads(line)["record_id"] for line in ops.RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()}
        for r in d["records"]:
            self.assertIn(r["id"], ids)
            self.assertTrue(set(r["related"]) <= ids)
            self.assertIn(r["ordered_by"], ("date", "file order"))
        self.assertEqual(len(self.client.get("/api/ops/evidence-recent?limit=50").json()["records"]), 20)


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.fx = make_fixtures(self.home)

    def ctx(self, **jobs):
        return Context(store=self.fx["store"], projects=self.fx["projects"], drafts=self.fx["drafts"], intern_sessions=self.fx["sessions"], **jobs)

    def test_running_and_queued_need_a_live_job(self):
        projects, pid = self.fx["projects"], self.fx["pid"]
        p = projects.get(pid)
        p["stages"]["leakage"] = {"status": "running", "started": OLD}
        p["stages"]["features"] = {"status": "queued"}
        p["running"] = "leakage"
        projects.save(p)
        projects.log(pid, "stage_started", {"stage": "leakage"})
        # no job is alive and the server started after the run began: a leftover, and the queued stage waits for nothing
        jobs = ops.Jobs(self.ctx())
        jobs.started = datetime.now(timezone.utc) + timedelta(minutes=1)
        leak = [j for j in jobs.stage_runs(projects.get(pid)) if j["stage"] == "leakage"]
        self.assertEqual([(j["status"], j["label"]) for j in leak], [("failed", "interrupted")])
        self.assertFalse([j for j in jobs.stage_runs(projects.get(pid)) if j["status"] == "queued"])
        # a run this server saw begin counts as running even without a job of its own (the MCP tools run stages in-process)
        jobs.started = datetime.now(timezone.utc) - timedelta(minutes=1)
        leak = [j for j in jobs.stage_runs(projects.get(pid)) if j["stage"] == "leakage"]
        self.assertEqual([j["status"] for j in leak], ["running"])
        # with the project's job alive, the same state is running and the next stage is queued
        jobs = ops.Jobs(self.ctx(jobs={pid: Busy()}))
        jobs.started = datetime.now(timezone.utc) + timedelta(minutes=1)
        rows = jobs.stage_runs(projects.get(pid))
        self.assertEqual({(j["stage"], j["status"]) for j in rows if j["status"] in ("running", "queued")}, {("leakage", "running"), ("features", "queued")})
        listing = jobs.listing()
        self.assertEqual((listing["totals"]["running"], listing["totals"]["queued"]), (1, 1))
        self.assertEqual(listing["jobs"][0]["status"], "queued")  # waiting jobs lead the list

    def test_evidence_without_dates_keeps_file_order(self):
        path = self.home / "records.jsonl"
        rows = [{"record_id": f"DCLAB-R0{i}", "type": "rule", "title": f"Rule {i}", "text": "See DCLAB-R01 and EXP-999.", "citations": [], "metadata": {}} for i in (1, 2, 3)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        d = ops.recent_evidence(5, path)
        self.assertEqual([r["id"] for r in d["records"]], ["DCLAB-R03", "DCLAB-R02", "DCLAB-R01"])
        self.assertEqual({r["ordered_by"] for r in d["records"]}, {"file order"})
        self.assertEqual(d["records"][0]["related"], ["DCLAB-R01"])  # EXP-999 is not in this index, so it is dropped
        self.assertEqual(ops.recent_evidence(3, self.home / "missing.jsonl")["records"], [])


if __name__ == "__main__":
    unittest.main()
