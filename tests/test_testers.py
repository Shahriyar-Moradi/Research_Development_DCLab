"""Trusted testers on their own machines (package 25.11): the notice agreed, the switches set, what changed sent to the
owner's records repository (tables only when the tester said yes, never .env or a key), and the owner's collection
rebuilding each tester's workspace. Git runs against a local bare repository: nothing reaches a real remote."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import testers  # noqa: E402

GIT = shutil.which("git")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@unittest.skipIf(GIT is None, "git is not installed")
class TesterKitTest(unittest.TestCase):
    def setUp(self):
        self.env_patch = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""})  # a tester's workspace is files
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.remote = self.tmp / "records.git"
        git("init", "--bare", "--quiet", str(self.remote))
        seed = self.tmp / "seed"  # the owner makes the repository with a first commit
        git("clone", "--quiet", str(self.remote), str(seed))
        (seed / "README.md").write_text("records\n")
        git("add", "README.md", cwd=seed)
        git("-c", "user.name=o", "-c", "user.email=o@x", "commit", "--quiet", "-m", "start", cwd=seed)
        git("push", "--quiet", "origin", "HEAD", cwd=seed)
        self.home = self.tmp / "friend" / "agent_runs"
        self.env = self.tmp / "friend" / ".env"
        self.env.parent.mkdir(parents=True)
        self.env.write_text("ANTHROPIC_API_KEY=placeholder-not-a-key\n")

    def make_workspace(self):
        (self.home / "experience" / "2026" / "10" / "16").mkdir(parents=True, exist_ok=True)
        (self.home / "experience" / "2026" / "10" / "16" / "v1.jsonl").write_text('{"kind": "chat.person"}\n')
        (self.home / "projects" / "p1" / "data").mkdir(parents=True, exist_ok=True)
        (self.home / "projects" / "p1" / "project.json").write_text('{"id": "p1"}')
        (self.home / "projects" / "p1" / "data" / "orders.csv").write_text("a,b\n1,2\n")
        (self.home / "drafts" / "d1" / "data").mkdir(parents=True, exist_ok=True)
        (self.home / "drafts" / "d1" / "draft.json").write_text(json.dumps({"id": "d1", "analysis": {"profile": [{"name": "plan", "preview": ["Q7Kxgold"], "top": ["Q7Kxbasic"]}]}}))
        (self.home / "drafts" / "d1" / "data" / "private.csv").write_text("x\n1\n")
        (self.home / ".env").write_text("SECRET=1\n")  # a stray .env inside the workspace is never sent

    def test_setup_needs_the_agreement_and_sets_the_switches(self):
        with self.assertRaises(testers.TesterError):
            testers.setup(self.home, name="Ali", email="ali@example.org", repo=str(self.remote), agreed=False, log=lambda *a: None)
        doc = testers.setup(self.home, name="Ali", email="ali@example.org", repo=str(self.remote), agreed=True, env_file=self.env, log=lambda *a: None)
        from dclab_rnd import consent
        from dclab_rnd.twin import home as twin_home

        self.assertEqual(consent.workspace_consent(self.home), {"improve": True, "train": True})
        self.assertTrue(twin_home.settings(self.home)["on"])
        self.assertIn("DCLAB_WORKSPACE_MONTHLY_EUR=15", self.env.read_text())
        self.assertIn("ANTHROPIC_API_KEY=placeholder-not-a-key", self.env.read_text())  # their key untouched
        self.assertTrue((Path(doc["clone"]) / ".git").is_dir())
        self.assertTrue(doc["id"].startswith("ali-"))

    def test_share_sends_what_changed_and_only_agreed_tables_and_the_owner_collects_it(self):
        doc = testers.setup(self.home, name="Ali", email="ali@example.org", repo=str(self.remote), agreed=True, env_file=self.env, log=lambda *a: None)
        self.make_workspace()
        asked = []
        out = testers.share(self.home, ask=lambda rel, size: asked.append(rel) or rel.endswith("orders.csv"), log=lambda *a: None)
        self.assertEqual(sorted(asked), ["drafts/d1/data/private.csv", "projects/p1/data/orders.csv"])
        self.assertEqual(out["tables"], ["projects/p1/data/orders.csv"])
        manifest = json.loads((Path(out["bundle"]) / "manifest.json").read_text())
        sent = {f["path"] for f in manifest["files"]}
        self.assertIn("experience/2026/10/16/v1.jsonl", sent)
        self.assertNotIn("drafts/d1/data/private.csv", sent)  # the tester said no
        self.assertFalse(any(".env" in p or "tester-records" in p for p in sent))
        self.assertEqual(testers.share(self.home, ask=lambda *a: True, log=lambda *a: None)["files"], 0)  # nothing changed, nothing asked again
        (self.home / "experience" / "2026" / "10" / "16" / "v1.jsonl").write_text('{"kind": "chat.person"}\n{"kind": "chat.agent"}\n')
        self.assertEqual(testers.share(self.home, log=lambda *a: None)["files"], 1)
        owner = self.tmp / "owner"
        clone = owner / "records"
        git("clone", "--quiet", str(self.remote), str(clone))
        got = testers.collect(owner / "agent_runs", clone, log=lambda *a: None)
        folder = owner / "agent_runs" / "testers" / doc["id"]
        self.assertEqual(got[doc["id"]]["bundles"], 2)
        self.assertEqual((folder / "experience" / "2026" / "10" / "16" / "v1.jsonl").read_text().count("\n"), 2)  # the later copy won
        self.assertTrue((folder / "projects" / "p1" / "data" / "orders.csv").is_file())
        self.assertFalse((folder / "drafts" / "d1" / "data" / "private.csv").exists())
        self.assertEqual(json.loads((folder / "tester.json").read_text())["name"], "Ali")
        rows = testers.summary(owner / "agent_runs")
        self.assertEqual((rows[0]["tester"], rows[0]["projects"], rows[0]["drafts"]), (doc["id"], 1, 1))

    def test_a_bundle_cannot_write_outside_the_testers_folder_or_read_through_a_link(self):
        owner = self.tmp / "owner"
        clone = owner / "records"
        git("clone", "--quiet", str(self.remote), str(clone))
        bad = clone / "evil-000000" / "20261016T000000000000Z-aaaaaa"
        (bad / "files").mkdir(parents=True)
        payload = b"escaped\n"
        (owner / "agent_runs").mkdir(parents=True)
        escape = bad / "files" / ".." / ".." / ".." / "escape.txt"  # the file exists where the path leads, with the right hash
        escape.write_bytes(payload)
        secret = owner / "owner-secret.txt"
        secret.write_text("the owner's own file\n")
        (bad / "files" / "audit.jsonl").symlink_to(secret)
        files = [{"path": "../../../escape.txt", "sha256": hashlib.sha256(payload).hexdigest()},
                 {"path": "audit.jsonl", "sha256": hashlib.sha256(secret.read_bytes()).hexdigest()}]
        (bad / "manifest.json").write_text(json.dumps({"files": files}))
        broken = clone / "broken-000000" / "20261016T000000000000Z-bbbbbb"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{not json")  # one bad manifest stops nothing
        got = testers.collect(owner / "agent_runs", clone, pull=False, log=lambda *a: None)
        self.assertEqual(got["evil-000000"]["skipped_files"], 2)
        self.assertFalse((owner / "agent_runs" / "escape.txt").exists())
        self.assertFalse((owner / "agent_runs" / "testers" / "escape.txt").exists())
        self.assertFalse((owner / "agent_runs" / "testers" / "evil-000000" / "audit.jsonl").exists())
        self.assertIn("broken-000000", got)

    def test_cells_of_a_refused_table_never_leave_and_deletions_and_offline_bundles_reach_the_owner(self):
        doc = testers.setup(self.home, name="Ali", email="ali@example.org", repo=str(self.remote), agreed=True, env_file=self.env, log=lambda *a: None)
        self.make_workspace()
        out = testers.share(self.home, ask=lambda rel, size: False, push=False, log=lambda *a: None)  # offline: committed, not pushed
        draft = json.loads((Path(out["bundle"]) / "files" / "drafts" / "d1" / "draft.json").read_text())
        self.assertNotIn("Q7Kx", json.dumps(draft))  # the refused table's values are taken out of the record that quotes them
        self.assertEqual(draft["analysis"]["profile"][0]["name"], "plan")
        (self.home / "projects" / "p1" / "project.json").unlink()  # the tester deletes a project
        second = testers.share(self.home, log=lambda *a: None)  # online again: the offline bundle and this one both go
        self.assertIn("projects/p1/project.json", second["removed"])
        self.assertLess(Path(out["bundle"]).name, Path(second["bundle"]).name)  # in order of time
        owner = self.tmp / "owner"
        clone = owner / "records"
        git("clone", "--quiet", str(self.remote), str(clone))
        testers.collect(owner / "agent_runs", clone, log=lambda *a: None)
        folder = owner / "agent_runs" / "testers" / doc["id"]
        self.assertTrue((folder / "drafts" / "d1" / "draft.json").is_file())
        self.assertFalse((folder / "projects" / "p1" / "project.json").exists())  # deleted there, deleted here

    def test_a_changed_table_is_asked_about_again_and_the_cap_is_enforced(self):
        self.env.write_text("ANTHROPIC_API_KEY=placeholder-not-a-key\nDCLAB_WORKSPACE_MONTHLY_EUR=   # none\n")
        testers.setup(self.home, name="Ali", email="ali@example.org", repo=str(self.remote), agreed=True, env_file=self.env, log=lambda *a: None)
        self.assertIn("DCLAB_WORKSPACE_MONTHLY_EUR=15", self.env.read_text())
        self.assertIn("ANTHROPIC_API_KEY=placeholder-not-a-key", self.env.read_text())
        self.make_workspace()
        asked = []
        testers.share(self.home, ask=lambda rel, size: asked.append(rel) or False, log=lambda *a: None)
        (self.home / "projects" / "p1" / "data" / "orders.csv").write_text("a,b\n9,9\n")  # real data under the same name
        testers.share(self.home, ask=lambda rel, size: asked.append(rel) or False, log=lambda *a: None)
        self.assertEqual(asked.count("projects/p1/data/orders.csv"), 2)

    def test_a_workspace_on_postgresql_or_without_consent_shares_nothing(self):
        testers.setup(self.home, name="Ali", email="ali@example.org", repo=str(self.remote), agreed=True, env_file=self.env, log=lambda *a: None)
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": "postgresql+psycopg://x@/y"}):
            with self.assertRaises(testers.TesterError):
                testers.share(self.home, log=lambda *a: None)
        from dclab_rnd import consent

        consent.set_workspace_consent(self.home, "train", False)
        with self.assertRaises(testers.TesterError):
            testers.share(self.home, log=lambda *a: None)

    def test_the_evaluation_board_reads_every_collected_tester(self):
        from dclab_core import twin
        from dclab_rnd import evaluation

        owner = self.tmp / "owner" / "agent_runs"
        folder = owner / "testers" / "ali-000000" / "twin" / "sessions" / "2026" / "10" / "16" / "draft-d1"
        folder.mkdir(parents=True)
        st = twin.step("s1", "draft-d1", "home.pack", "2026-10-16T09:00:00+00:00", {"i": 1}, twin.ArmAnswer(arm="system", answerer="home", value={"pack": "tabular"}),
                       twin.ArmAnswer(arm="ai_first", answerer="ai_first@home", value={"pack": "fraud"}), truth=twin.Truth(value={"pack": "fraud"}, source="person_pick"))
        (folder / "steps.jsonl").write_text(st.model_dump_json() + "\n")
        result = evaluation.board(owner)
        self.assertEqual((result["sources"]["testers"], result["sources"]["judged_twin_steps"]), (1, 1))

if __name__ == "__main__":
    unittest.main()
