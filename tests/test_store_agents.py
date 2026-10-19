"""Package 32.7: other agents' sessions.

Every Claude Code session on this repository is kept (the main checkout's and each worktree's, never another repository's),
its transcript by its new lines, anything shaped like a key replaced by a marker; a tester's own sessions are kept from
their share, redacted the same way; an MCP client's call is kept whole as its own source, its cell-quoting keys removed.
"""

import json
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


@unittest.skipIf(MISSING, MISSING)
class AgentsTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout
        from dclab_rnd.store.keepers import agents

        self.repo = Path(tempfile.mkdtemp()) / "My R&D"
        self.repo.mkdir()
        self.projects = Path(tempfile.mkdtemp()) / "projects"
        head = agents.encoded(self.repo)
        self.main = write(self.projects / head / "s1.jsonl", json.dumps({"type": "user", "text": "use sk-ant-api03-" + "x" * 30}) + "\n")
        write(self.projects / head / "s1" / "subagents" / "a1.jsonl", '{"type": "assistant", "thinking": "plan"}\n')
        write(self.projects / f"{head}--claude-worktrees-brave-turing-1a2b" / "s2.jsonl", '{"type": "user"}\n')
        write(self.projects / "-Users-x-other-repo" / "s3.jsonl", '{"type": "user"}\n')  # another repository: never kept
        self.base = Path(tempfile.mkdtemp()) / "agent_runs"
        write(self.base / "consent.json", json.dumps({"train": True}))
        write(self.base / "testers/friend-1a2b/consent.json", json.dumps({"train": True}))
        write(self.base / "testers/friend-1a2b/claude_code/main/t1.jsonl", '{"text": "my token gho_' + "A" * 30 + '"}\n')
        write(self.base / "mcp/calls.jsonl", json.dumps({"tool": "profile", "result": {"columns": [{"name": "email", "preview": ["Q7Kx@example.org"]}]}}) + "\n")
        self.env = {"agent_runs": self.base, "claude_projects": self.projects, "repo": self.repo, "test_runs": Path(tempfile.mkdtemp())}
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)

    def conduct(self):
        from dclab_rnd.store import conductor
        from dclab_rnd.store.keepers import standard

        return conductor.conduct(self.env, self.layout, standard.install({}))

    def test_this_repositorys_sessions_are_kept_redacted_and_grow_by_their_new_lines(self):
        from dclab_rnd.store import Store

        self.assertEqual(self.conduct()["failed"], [])
        store = Store(self.layout)
        sources = {(e.workspace, e.source) for e in store.entries("agents")}
        self.assertIn(("rnd", "claude_code/main/s1.jsonl"), sources)
        self.assertIn(("rnd", "claude_code/main/s1/subagents/a1.jsonl"), sources)
        self.assertIn(("rnd", "claude_code/brave-turing-1a2b/s2.jsonl"), sources)
        self.assertFalse(any("s3.jsonl" in s for _, s in sources))  # another repository's sessions are not this one's
        kept = store.whole("agents", "rnd", "claude_code/main/s1.jsonl")
        self.assertNotIn(b"sk-ant-api03-", kept)
        self.assertIn(b"[SECRET REMOVED]", kept)
        with open(self.main, "a", encoding="utf-8") as fh:
            fh.write('{"type": "assistant", "text": "done"}\n')
        self.conduct()
        last = [e for e in store.entries("agents") if e.source == "claude_code/main/s1.jsonl"][-1]
        self.assertGreater(last.offset, 0)  # only the new line

    def test_a_testers_sessions_and_an_mcp_call_are_kept_the_same_way(self):
        from dclab_rnd.store import Store

        self.conduct()
        store = Store(self.layout)
        tester = [e for e in store.entries("agents") if e.workspace == "testers__friend-1a2b"]
        self.assertEqual([e.source for e in tester], ["claude_code/main/t1.jsonl"])
        self.assertNotIn(b"gho_", store.read(tester[0]))
        call = [e for e in store.entries("conversations") if e.source == "mcp/calls.jsonl"]
        self.assertEqual(len(call), 1)
        self.assertNotIn(b"Q7Kx@example.org", store.read(call[0]))  # the tool's result without its cell-quoting keys


    def test_on_a_testers_machine_this_machines_sessions_are_never_read_unless_named(self):
        import os
        from unittest import mock

        from dclab_rnd.store.keepers import agents, workspaces

        with mock.patch.object(workspaces, "AGENT_RUNS", self.base), mock.patch.dict(os.environ, {"DCLAB_STORE_CLAUDE_PROJECTS": ""}):
            (self.base / "tester.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(agents.projects_folder())
            with mock.patch.dict(os.environ, {"DCLAB_STORE_CLAUDE_PROJECTS": str(self.projects)}):
                self.assertEqual(agents.projects_folder(), self.projects)  # named explicitly: the owner's choice

    def test_screenshots_are_counted_never_kept(self):
        from dclab_rnd.store import Store
        from dclab_rnd.store.keepers import agents

        head = agents.encoded(self.repo)
        (self.projects / head / "tool-results").mkdir(parents=True, exist_ok=True)
        (self.projects / head / "tool-results" / "shot.jpg").write_bytes(b"\xff\xd8desktop")
        self.conduct()
        sources = {e.source for e in Store(self.layout).entries("agents")}
        self.assertNotIn("claude_code/main/tool-results/shot.jpg", sources)
        self.assertIn("claude_code/main/~images~counts.json", sources)


if __name__ == "__main__":
    unittest.main()
