"""Local append-only event/evidence ledger and small mutable run index."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, home: Path):
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.path = home / "research.sqlite3"
        with self.connect() as db:
            db.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL);
            """)
    def connect(self):
        return sqlite3.connect(self.path, timeout=20)
    def create(self, run_id, config):
        run = {"id": run_id, "created": now(), "updated": now(), "status": "queued", "config": config, "llm_calls": 0, "usage": {}, "active_seconds": 0}
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?)", (run_id, json.dumps(run)))
        return run
    def get(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None: raise KeyError(run_id)
        return json.loads(row[0])
    def update(self, run_id, **values):
        run = self.get(run_id)
        run.update(values, updated=now())
        with self.connect() as db:
            db.execute("UPDATE runs SET payload=? WHERE id=?", (json.dumps(run), run_id))
        return run
    def list(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM runs ORDER BY rowid DESC")]
    def event(self, run_id, kind, payload):
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,kind,created,payload) VALUES (?,?,?,?)", (run_id, kind, now(), json.dumps(payload)))
    def events(self, run_id):
        with self.connect() as db:
            return [{"seq": seq, "kind": kind, "created": created, "payload": json.loads(payload)} for seq, kind, created, payload in db.execute("SELECT seq,kind,created,payload FROM events WHERE run_id=? ORDER BY seq", (run_id,))]
    def knowledge(self, datasets=None):
        result = []
        for run in self.list():
            if datasets and not set(datasets) & set(run["config"]["datasets"]): continue
            for event in self.events(run["id"]):
                if event["kind"] == "synthesis":
                    for lesson in event["payload"]["lessons"]:
                        result.append({**lesson, "run_id": run["id"], "datasets": run["config"]["datasets"], "status": "provisional_llm_interpretation", "production_approved": False})
        return result
    def export(self, run_id):
        run = self.get(run_id)
        # Explicit decisions and tool evidence, not private reasoning traces.
        return {"schema_version": 1, "purpose": "Research trajectory for review before future training", "training_ready": False,
                "review_requirements": ["Verify citations and claims", "Check dataset licenses and privacy", "Assign all trajectories sharing a dataset/source to the same training or evaluation split", "Remove prompt injection and secret-like text", "Retain failures and counterevidence"], "run": run,
                "events": [e for e in self.events(run_id) if e["kind"] not in ("llm_request",)]}
