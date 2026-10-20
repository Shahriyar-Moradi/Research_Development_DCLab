"""Packages 32.15 and 32.16: a branch per tester, and what a tester shares.

Two testers set up and share at once, each to their own branch (``tester/<id>``) with no conflict, a branch that starts
empty (the owner's store never comes with it); a share refuses to commit anywhere but the tester's branch; the owner's
collect rebuilds both from their branches; the share holds the store's other kinds (the teams' boards, students, judgments,
model exchanges, MCP calls) and a tester's own Claude Code sessions only with their yes, keys removed. Git runs against a
local bare repository.
"""

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


def quiet(*a, **k):
    return None


@unittest.skipIf(GIT is None, "git is not installed")
class BranchesTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.remote = self.tmp / "records.git"
        git("init", "--bare", "--quiet", str(self.remote))
        seed = self.tmp / "seed"
        git("clone", "--quiet", str(self.remote), str(seed))
        (seed / "CATALOG.json").write_text('{"the owner": "store"}\n')
        git("add", "CATALOG.json", cwd=seed)
        git("-c", "user.name=o", "-c", "user.email=o@x", "commit", "--quiet", "-m", "owner's store", cwd=seed)
        git("push", "--quiet", "origin", "HEAD", cwd=seed)

    def make_tester(self, name, claude_code=False):
        home = self.tmp / name / "agent_runs"
        (home / "experience" / "2026" / "10" / "20").mkdir(parents=True, exist_ok=True)
        (home / "experience" / "2026" / "10" / "20" / "v1.jsonl").write_text(json.dumps({"kind": "chat.person", "by": name}) + "\n")
        (home / "orchestra").mkdir(parents=True, exist_ok=True)
        (home / "orchestra" / "walks.jsonl").write_text('{"walk": "w1"}\n')
        (home / "model_exchanges" / "2026" / "10").mkdir(parents=True, exist_ok=True)
        (home / "model_exchanges" / "2026" / "10" / "20.jsonl").write_text('{"purpose": "home_agent", "thinking": []}\n')
        (home / "mcp").mkdir(parents=True, exist_ok=True)
        (home / "mcp" / "calls.jsonl").write_text('{"tool": "search_evidence"}\n')
        doc = testers.setup(home, name=name, email=f"{name.lower()}@example.org", repo=str(self.remote), agreed=True, log=quiet, claude_code=claude_code)
        return home, doc

    def test_two_testers_share_to_their_own_branches_and_collect_rebuilds_both(self):
        (home_a, a), (home_b, b) = self.make_tester("Ali"), self.make_tester("Sara")
        self.assertEqual((a["branch"], b["branch"]), (f"tester/{a['id']}", f"tester/{b['id']}"))
        self.assertFalse((Path(a["clone"]) / "CATALOG.json").exists())  # the branch starts empty: the owner's store never comes with it
        testers.share(home_a, log=quiet)
        testers.share(home_b, log=quiet)
        branches = git("ls-remote", "--heads", str(self.remote)).stdout
        self.assertIn(f"refs/heads/tester/{a['id']}", branches)
        self.assertIn(f"refs/heads/tester/{b['id']}", branches)
        on_a = git("ls-tree", "-r", "--name-only", f"tester/{a['id']}", cwd=self.remote).stdout
        self.assertNotIn(b["id"], on_a)  # one tester's branch holds only their own folder
        self.assertIn(f"{a['id']}/", on_a)
        for kind in ("orchestra/walks.jsonl", "model_exchanges/2026/10/20.jsonl", "mcp/calls.jsonl"):
            self.assertIn(kind, on_a)  # 32.16: the store's other kinds go too
        owner = self.tmp / "owner"
        git("clone", "--quiet", str(self.remote), str(owner / "records"))
        got = testers.collect(owner / "agent_runs", owner / "records", log=quiet)
        self.assertEqual(set(got), {a["id"], b["id"]})
        self.assertIn("Sara", (owner / "agent_runs" / "testers" / b["id"] / "experience" / "2026" / "10" / "20" / "v1.jsonl").read_text())

    def test_a_share_never_commits_on_another_branch(self):
        home, doc = self.make_tester("Ali")
        clone = Path(doc["clone"])
        git("checkout", "--quiet", "-B", "tester/someone-else", cwd=clone)
        testers.share(home, log=quiet)  # it moves back to its own branch first
        self.assertEqual(git("rev-parse", "--abbrev-ref", "HEAD", cwd=clone).stdout.strip(), doc["branch"])
        self.assertNotIn("refs/heads/tester/someone-else", git("ls-remote", "--heads", str(self.remote)).stdout)

    def test_a_testers_claude_code_sessions_go_only_with_their_yes_and_keys_removed(self):
        projects = self.tmp / "claude_projects"
        from dclab_rnd.store.keepers import agents

        head = agents.encoded(ROOT)
        (projects / head).mkdir(parents=True)
        (projects / head / "s1.jsonl").write_text('{"text": "my key sk-ant-api03-' + "x" * 30 + '"}\n')
        (projects / head / "shot.png").write_bytes(b"\x89PNG")
        with mock.patch.object(agents, "PROJECTS", projects):
            home_no, doc_no = self.make_tester("Ali")
            testers.share(home_no, log=quiet)
            home_yes, doc_yes = self.make_tester("Sara", claude_code=True)
            testers.share(home_yes, log=quiet)
        on_no = git("ls-tree", "-r", "--name-only", doc_no["branch"], cwd=self.remote).stdout
        on_yes = git("ls-tree", "-r", "--name-only", doc_yes["branch"], cwd=self.remote).stdout
        self.assertNotIn("claude_code/", on_no)
        self.assertIn("claude_code/main/s1.jsonl", on_yes)
        self.assertNotIn("shot.png", on_yes)
        path = next(p for p in on_yes.splitlines() if p.endswith("claude_code/main/s1.jsonl"))
        text = git("show", f"{doc_yes['branch']}:{path}", cwd=self.remote).stdout
        self.assertNotIn("sk-ant-api03-", text)


if __name__ == "__main__":
    unittest.main()
