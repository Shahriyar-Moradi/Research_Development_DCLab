"""Package 32.33: DCLab Code's own records.

The owner's developer app records each session (SDK messages whole, the person's permissions, answers and plan
approvals); the store keeps those records by their new lines and the sessions' index, anything shaped like a key replaced,
never on a tester's machine unless named, and never in a test that did not hand it a folder.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
class DclabCodeTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout

        self.home = Path(tempfile.mkdtemp()) / ".dclab-code"
        rec = {"v": "dclab_code.record.v1", "kind": "app", "type": "permission_decision", "reqId": "p_1", "decision": "deny"}
        sdk = {"v": "dclab_code.record.v1", "kind": "sdk", "message": {"type": "user", "content": "cat .env -> ANTHROPIC_API_KEY=sk-ant-api03-" + "y" * 30}}
        self.repo = Path(tempfile.mkdtemp()) / "My R&D"
        self.repo.mkdir()
        self.record = write(self.home / "records" / "s_abc123.jsonl", json.dumps(rec) + "\n" + json.dumps(sdk) + "\n")
        write(self.home / "records" / "s_copy01.jsonl", json.dumps(rec) + "\n")  # its own copy, under .claude/worktrees
        write(self.home / "records" / "s_other1.jsonl", json.dumps({"kind": "sdk", "message": {"content": "client rows 1,2,3"}}) + "\n")
        write(self.home / "records" / "s_orphan.jsonl", "{}\n")  # not in the index: counted only
        write(self.home / "sessions.json", json.dumps({"sessions": [
            {"id": "s_abc123", "projectPath": str(self.repo), "sdkSessionId": "u-1", "title": "fix", "worktree": None},
            {"id": "s_copy01", "projectPath": str(self.repo), "cwd": str(self.repo / ".claude/worktrees/dclab-copy01"), "title": "copy"},
            {"id": "s_other1", "projectPath": "/elsewhere/DCLab", "title": "the main product's secret plan"}]}))
        write(self.home / "config.json", "{}")  # settings, not a record: not kept
        (self.home / "attachments" / "s_abc123").mkdir(parents=True)
        (self.home / "attachments" / "s_abc123" / "0123456789abcdef.png").write_bytes(b"\x89PNG screenshot")
        (self.home / "attachments" / "s_other1").mkdir(parents=True)
        (self.home / "attachments" / "s_other1" / "fedcba9876543210.png").write_bytes(b"\x89PNG client screen")
        self.base = Path(tempfile.mkdtemp()) / "agent_runs"
        write(self.base / "consent.json", json.dumps({"train": True}))
        self.env = {"agent_runs": self.base, "claude_projects": None, "dclab_code_home": self.home, "repo": self.repo,
                    "test_runs": Path(tempfile.mkdtemp())}
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)

    def conduct(self, env=None):
        from dclab_rnd.store import conductor
        from dclab_rnd.store.keepers import standard

        return conductor.conduct(env or self.env, self.layout, standard.install({}))

    def test_records_and_the_index_are_kept_redacted_and_grow_by_new_lines(self):
        from dclab_rnd.store import Store

        self.assertEqual(self.conduct()["failed"], [])
        store = Store(self.layout)
        sources = {(e.workspace, e.source) for e in store.entries("agents")}
        self.assertIn(("rnd", "dclab_code/records/s_abc123.jsonl"), sources)
        self.assertIn(("rnd", "dclab_code/sessions.json"), sources)
        self.assertFalse(any(s.endswith("config.json") for _, s in sources))
        kept = store.whole("agents", "rnd", "dclab_code/records/s_abc123.jsonl")
        self.assertIn(b'"decision": "deny"', kept)  # the person's decision, whole
        self.assertNotIn(b"sk-ant-api03-", kept)
        self.assertIn(b"[SECRET REMOVED]", kept)
        with open(self.record, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "app", "type": "note", "text": "later"}) + "\n")
        self.conduct()
        last = [e for e in store.entries("agents") if e.source == "dclab_code/records/s_abc123.jsonl"][-1]
        self.assertGreater(last.offset, 0)  # only the new line

    def test_only_this_repositorys_sessions_are_kept_others_counted(self):
        from dclab_rnd.store import Store

        self.conduct()
        store = Store(self.layout)
        sources = {e.source for e in store.entries("agents")}
        self.assertIn("dclab_code/records/s_copy01.jsonl", sources)
        self.assertNotIn("dclab_code/records/s_other1.jsonl", sources)
        self.assertNotIn("dclab_code/records/s_orphan.jsonl", sources)
        counts = json.loads(store.whole("agents", "rnd", "dclab_code/~other-projects~counts.json"))
        self.assertEqual(sorted(c["source"] for c in counts), ["s_orphan.jsonl", "s_other1.jsonl"])
        images = json.loads(store.whole("agents", "rnd", "dclab_code/~images~counts.json"))
        self.assertEqual([c["source"] for c in images], ["0123456789abcdef.png"])  # this repository's image counted; another folder's not at all
        self.assertFalse(any("attachments" in s or s.endswith(".png") for s in sources))  # never an image kept
        index = store.whole("agents", "rnd", "dclab_code/sessions.json")
        self.assertIn(b"s_abc123", index)
        self.assertNotIn(b"secret plan", index)  # another folder's title is not kept either
        self.assertNotIn(b"client rows", b"".join(store.read(e) for e in store.entries("agents")))

    def test_shaped_lines_say_they_are_dclab_codes(self):
        from dclab_rnd.store import shapers

        self.conduct()
        shapers.shape(self.layout)
        bodies = [r.body for r in shapers.records(self.layout, "agents")]
        mine = [b for b in bodies if b.get("agent") == "dclab_code"]
        self.assertTrue(mine)
        self.assertFalse([b for b in bodies if b.get("agent") == "claude_code"])

    def test_nothing_is_read_unless_a_folder_is_given_or_it_is_the_owners_machine(self):
        from dclab_rnd.store.keepers import dclab_code, workspaces

        # A test or check that names the agents' folders but not this one reads nothing.
        self.assertIsNone(dclab_code.home_folder({"claude_projects": "/x"}))
        self.assertEqual(list(dclab_code.sources({**self.env, "dclab_code_home": None})), [])
        with mock.patch.dict(os.environ, {"DCLAB_STORE_DCLAB_CODE": str(self.home), "DCLAB_STORE_CLAUDE_PROJECTS": "off"}):
            self.assertIsNone(dclab_code.home_folder())  # the tests' switch wins over a named folder
        with mock.patch.dict(os.environ, {"DCLAB_STORE_DCLAB_CODE": "off", "DCLAB_STORE_CLAUDE_PROJECTS": ""}):
            self.assertIsNone(dclab_code.home_folder())
        with mock.patch.object(workspaces, "AGENT_RUNS", self.base), mock.patch.dict(os.environ, {"DCLAB_STORE_DCLAB_CODE": "", "DCLAB_STORE_CLAUDE_PROJECTS": ""}):
            self.assertEqual(dclab_code.home_folder(), dclab_code.HOME)  # the owner's machine
            (self.base / "tester.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(dclab_code.home_folder())  # a tester's machine
            with mock.patch.dict(os.environ, {"DCLAB_STORE_DCLAB_CODE": str(self.home)}):
                self.assertEqual(dclab_code.home_folder(), self.home)  # named explicitly

    def test_its_claim_does_not_overlap_the_agents_keeper(self):
        from dclab_rnd.store.keepers import standard

        reg = standard.install({})
        self.assertIn("dclab_code", reg)
        self.assertTrue(reg["dclab_code"].owns("agents", "dclab_code/records/s_1.jsonl"))
        self.assertFalse(reg["agents"].owns("agents", "dclab_code/records/s_1.jsonl"))


if __name__ == "__main__":
    unittest.main()
