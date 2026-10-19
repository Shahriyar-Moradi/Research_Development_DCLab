"""Package 32.3: the keepers of what DCLab already records.

Every workspace under ``agent_runs`` is found (the development root, a live test's, a tester's as collected); each kind's
files are kept once, then as new versions: an append-only log by its new tail only, rebuilt whole and checked by its hash;
a workspace whose consent is ``operate`` keeps counts only; a table's rows are never read; the test runs are kept from
the repository; a live test's own logs and results are kept under the development workspace.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def runs_folder() -> Path:
    base = Path(tempfile.mkdtemp()) / "agent_runs"
    write(base / "consent.json", json.dumps({"train": True}))
    write(base / "experience/2026/10/20/v1.jsonl", '{"kind": "visit.started"}\n')
    write(base / "ledger/decisions/2026/10/20.jsonl", '{"point": "column.is_identifier"}\n')
    write(base / "audit.jsonl", '{"kind": "move"}\n')
    write(base / "model_requests.jsonl", '{"purpose": "home"}\n')
    write(base / "drafts/d1/events.jsonl", '{"kind": "chat.person"}\n')
    write(base / "drafts/d1/data/orders.csv", "email,amount\nQ7Kx@example.org,10\n")  # a table's rows: never read here
    write(base / "projects/p1/project.json", '{"id": "p1"}')
    write(base / "projects/p1/data/orders.csv", "id\n1\n")
    write(base / "live_tests/2026-10-20/run.log", "rules: 3 of 10\n")
    write(base / "live_tests/2026-10-20/ws/experience/2026/10/20/v2.jsonl", '{"kind": "visit.started"}\n')
    write(base / "testers/friend-1a2b/consent.json", json.dumps({"train": False, "improve": False}))
    write(base / "testers/friend-1a2b/audit.jsonl", '{"kind": "move"}\n')
    return base


@unittest.skipIf(MISSING, MISSING)
class KeepersTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store
        from dclab_rnd.store.keepers import standard

        self.base = runs_folder()
        self.test_runs = Path(tempfile.mkdtemp()) / "test_runs"
        write(self.test_runs / "runs.jsonl", '{"ok": true}\n')
        self.env = {"agent_runs": self.base, "test_runs": self.test_runs, "claude_projects": None}  # never this machine's sessions
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        self.store = Store(self.layout)
        self.keepers = standard.install({})
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)

    def run_once(self):
        from dclab_rnd.store import conductor

        return conductor.conduct(self.env, self.layout, self.keepers)

    def test_every_workspace_and_kind_is_kept_and_a_table_never_read(self):
        from dclab_rnd.store.keepers import workspaces

        names = {w.name: w for w in workspaces.discover(self.base)}
        self.assertEqual(set(names), {"dev", "live_tests__2026-10-20__ws", "testers__friend-1a2b"})
        self.assertEqual((names["dev"].consent, names["testers__friend-1a2b"].consent), ("train", "operate"))
        record = self.run_once()
        self.assertEqual(record["failed"], [])
        kept = {(k, e.workspace, e.source) for k in ("sessions", "decisions", "audit", "model_requests", "conversations", "workflows", "tests")
                for e in self.store.entries(k)}
        for want in [("sessions", "dev", "experience/2026/10/20/v1.jsonl"), ("decisions", "dev", "ledger/decisions/2026/10/20.jsonl"),
                     ("audit", "dev", "audit.jsonl"), ("model_requests", "dev", "model_requests.jsonl"),
                     ("conversations", "dev", "drafts/d1/events.jsonl"), ("workflows", "dev", "projects/p1/project.json"),
                     ("sessions", "live_tests__2026-10-20__ws", "experience/2026/10/20/v2.jsonl"),
                     ("tests", "dev", "live_tests/2026-10-20/run.log"), ("tests", "rnd", "test_runs/runs.jsonl")]:
            self.assertIn(want, kept)
        everything = b"".join(self.store.read(e) for k in ("sessions", "decisions", "audit", "model_requests", "conversations", "workflows", "tests")
                              for e in self.store.entries(k))
        self.assertNotIn(b"Q7Kx@example.org", everything)  # a table's rows: never read by these keepers
        tester = [e for e in self.store.entries("audit") if e.workspace == "testers__friend-1a2b"]
        self.assertEqual([e.source for e in tester], ["audit~counts.json"])  # no consent: counts only
        self.assertNotIn(b'"move"', self.store.read(tester[0]))

    def test_a_log_is_kept_by_its_new_tail_and_rebuilt_whole_and_a_rewrite_is_kept_whole(self):
        self.run_once()
        self.assertEqual(sum(s["unchanged"] for s in self.run_once()["steps"]), sum(s["kept"] + s["unchanged"] for s in self.run_once()["steps"]))
        log = self.base / "audit.jsonl"
        log.write_text(log.read_text() + '{"kind": "approval"}\n', encoding="utf-8")
        self.run_once()
        versions = [e for e in self.store.entries("audit") if e.workspace == "dev"]
        self.assertEqual([(e.version, e.offset > 0) for e in versions], [(1, False), (2, True)])
        self.assertEqual(self.store.read(versions[1]), b'{"kind": "approval"}\n')  # only the tail
        self.assertEqual(self.store.whole("audit", "dev", "audit.jsonl"), log.read_bytes())
        log.write_text('{"kind": "rewritten"}\n', encoding="utf-8")  # not an append: kept whole again
        self.run_once()
        last = [e for e in self.store.entries("audit") if e.workspace == "dev"][-1]
        self.assertEqual((last.offset, last.note), (0, "rewritten: kept whole again"))
        self.assertEqual(self.store.whole("audit", "dev", "audit.jsonl"), b'{"kind": "rewritten"}\n')
        self.assertEqual(len([e for e in self.store.entries("audit") if e.workspace == "dev"]), 3)  # the old versions stay


    def test_each_file_is_kept_once_at_any_depth_and_nothing_is_left_out(self):
        from dclab_rnd.store.keepers import standard

        deep = self.base / "live_tests/2026-10-20/journeys/arm/JD-01/workspace"
        write(deep / "projects/p9/project.json", '{"id": "p9"}')
        write(deep / "drafts/d9/events.jsonl", '{"kind": "chat"}\n')
        write(deep / "jobs.json", "{}")
        write(self.base / "real_problems/results/RPB-001.json", '{"id": "RPB-001"}')  # in no nested workspace: kept as other
        write(self.base / "live_tests/2026-10-20/driver/play.py", "print(1)\n")
        items = standard.inventory({"agent_runs": self.base})
        keys = [(i.kind, i.ws.name, i.rel) for i in items]
        self.assertEqual(len(keys), len(set(keys)))  # nothing twice
        on_disk = {p.resolve() for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(on_disk, {i.path.resolve() for i in items})  # nothing left out
        by_rel = {(i.ws.name, i.rel): i.kind for i in items}
        name = "live_tests__2026-10-20__journeys__arm__jd-01__workspace"
        self.assertEqual(by_rel[(name, "projects/p9/project.json")], "workflows")  # in its own workspace, its own kind
        self.assertEqual(by_rel[(name, "jobs.json")], "tasks")
        self.assertEqual(by_rel[("dev", "real_problems/results/RPB-001.json")], "other")
        self.assertEqual(by_rel[("dev", "live_tests/2026-10-20/driver/play.py")], "tests")

    def test_an_unreadable_file_is_kept_as_its_error_and_the_rest_still_run(self):
        from dclab_rnd.store import Store

        write(self.base / "drafts/a1/events.jsonl", '{"kind": "chat"}\n')
        write(self.base / "drafts/b2/events.jsonl", '{"kind": "chat"}\n')
        locked = self.base / "drafts/a1/events.jsonl"
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o644)
        record = self.run_once()
        step = next(s for s in record["steps"] if s["name"] == "conversations")
        self.assertEqual([e["source"] for e in step["errors"]], ["drafts/a1/events.jsonl"])
        kept = {e.source for e in Store(self.layout).entries("conversations")}
        self.assertIn("drafts/b2/events.jsonl", kept)  # the one after it is kept
        self.assertIn("drafts/a1/events.jsonl~error.json", kept)  # and the error itself

    def test_consent_off_counts_the_live_tests_too_and_tables_secrets_and_previews_never_leave(self):
        from dclab_rnd.store import Store

        write(self.base / "drafts/d1/draft.json", json.dumps({"id": "d1", "columns": [{"name": "email", "preview": ["Q7Kx@example.org"]}]}))
        write(self.base / "drafts/d1/events.jsonl", json.dumps({"kind": "data", "data": {"preview": ["Q7Kx@example.org"]}}) + "\n")
        write(self.base / "live_tests/2026-10-20/person/table.csv", "email\nQ7Kx@example.org\n")
        write(self.base / "live_tests/2026-10-20/api_key.txt", "sk-123")
        write(self.base / "projects/p1/export/pipeline.ipynb", json.dumps({"cells": [{"cell_type": "code", "source": "df.head()",
                                                                                         "outputs": [{"text": "Q7Kx@example.org"}]}]}))
        self.run_once()
        store = Store(self.layout)
        everything = b"".join(store.read(e) for k in ("conversations", "tests", "workflows", "other") for e in store.entries(k))
        self.assertNotIn(b"Q7Kx@example.org", everything)  # a preview, a table's file and a notebook's output never kept
        self.assertNotIn(b"sk-123", everything)  # a secret's file: counted only
        self.assertIn(b"df.head()", everything)  # the notebook's code is kept
        (self.base / "consent.json").write_text(json.dumps({"train": False, "improve": False}), encoding="utf-8")
        write(self.base / "live_tests/2026-10-20/later.log", "a later live test\n")
        self.run_once()
        later = [e for e in store.entries("tests") if "later" in e.source or e.source.endswith("~counts.json")]
        self.assertFalse(any(e.source == "live_tests/2026-10-20/later.log" for e in later))  # consent off: not its content
        self.assertTrue(any(e.source.endswith("~counts.json") for e in later))

    def test_the_worker_job_keeps_its_own_workspace(self):
        from types import SimpleNamespace
        from unittest import mock

        from dclab_rnd.jobs import handlers
        from dclab_rnd.store import Store

        services = SimpleNamespace(store=SimpleNamespace(home=self.base))
        with mock.patch.dict(os.environ, {"DCLAB_STORE": str(self.layout.root)}):
            handlers._store_run(services, {})
        self.assertIn("audit.jsonl", {e.source for e in Store(self.layout).entries("audit") if e.workspace == "dev"})


if __name__ == "__main__":
    unittest.main()
