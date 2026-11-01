"""2026-11-02: a sealed exam's vault is sealed and listed when the run ends (conversation_exam.vault): what it holds as counts, every
file with its SHA-256, the use eval_only until the owner releases it. The permanent archive keeps the vault whole; this lets anyone check
it is still what the run wrote. Counts and hashes only: no story is read."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.conversation_exam import vault  # noqa: E402


def make_vault(base: Path, name="CEX-099_sealed_v9") -> Path:
    folder = base / name
    for arm, pid in (("case", "s9-01"), ("plain", "s9-01")):
        (folder / "conversations" / arm / pid).mkdir(parents=True)
        (folder / "conversations" / arm / pid / "conversation.json").write_text(json.dumps({"id": pid, "arm": arm}))
        (folder / "conversations" / arm / pid / "agent_metrics.jsonl.lock").write_text("")
    (folder / "model_exchanges" / "2026" / "11").mkdir(parents=True)
    calls = [{"purpose": "case_agent", "whole": True, "thinking": [{"text": "t"}]}, {"purpose": "exam_judge", "whole": True, "thinking": []},
             {"purpose": "exam_person", "whole": False}]
    (folder / "model_exchanges" / "2026" / "11" / "02.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    (folder / "labels").mkdir()
    (folder / "labels" / "CEX-099_owner_todo.json").write_text("[]")
    return folder


class VaultTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, True)
        self.folder = make_vault(self.base)

    def test_sealing_lists_counts_and_hashes_and_marks_the_vault_eval_only(self):
        m = vault.seal(self.folder, "CEX-099", "sealed_v9", {"commit": "abc1234", "spent_eur": 4.0})
        self.assertEqual(m["counts"], {"conversations": 2, "model_calls": 3, "model_calls_whole": 2, "model_calls_with_thinking": 1})
        self.assertEqual((m["use"], m["released_to_training"], m["commit"]), ("eval_only", None, "abc1234"))
        paths = {f["path"] for f in m["files"]}
        self.assertIn("conversations/case/s9-01/conversation.json", paths)
        self.assertFalse(any(p.startswith("labels/") or p.endswith(".lock") or p == "MANIFEST.json" for p in paths))
        self.assertTrue(all(len(f["sha256"]) == 64 for f in m["files"]))
        self.assertEqual(vault.verify(self.folder), [])

    def test_a_changed_or_missing_file_is_reported_but_the_owners_labels_may_grow(self):
        vault.seal(self.folder, "CEX-099", "sealed_v9")
        (self.folder / "labels" / "CEX-099_owner_todo.json").write_text('[{"owner": "filled"}]')  # the owner labels after the run
        (self.folder / "labels" / "later.json").write_text("{}")
        self.assertEqual(vault.verify(self.folder), [])
        (self.folder / "conversations" / "case" / "s9-01" / "conversation.json").write_text("{}")
        (self.folder / "model_exchanges" / "2026" / "11" / "02.jsonl").unlink()
        problems = vault.verify(self.folder)
        self.assertTrue(any(p.startswith("changed:") and "conversation.json" in p for p in problems), problems)
        self.assertTrue(any(p.startswith("missing:") and "02.jsonl" in p for p in problems), problems)

    def test_sealing_twice_keeps_the_first_manifest(self):
        first = vault.seal(self.folder, "CEX-099", "sealed_v9", {"commit": "one"})
        (self.folder / "conversations" / "case" / "s9-01" / "conversation.json").write_text('{"changed": true}')
        again = vault.seal(self.folder, "CEX-099", "sealed_v9", {"commit": "two"})
        self.assertEqual(again, first)  # a sealed vault is never re-listed over a change
        self.assertTrue(vault.verify(self.folder))

    def test_the_all_command_seals_the_vaults_that_have_no_manifest_and_fails_on_a_changed_one(self):
        make_vault(self.base, "CEX-098_sealed_v8")
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(vault.main(["all"], root=self.base), 0)
        self.assertIn("sealed CEX-098_sealed_v8", out.getvalue())
        self.assertTrue((self.base / "CEX-098_sealed_v8" / vault.MANIFEST).exists())
        (self.base / "CEX-098_sealed_v8" / "conversations" / "case" / "s9-01" / "conversation.json").write_text("{}")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(vault.main(["verify"], root=self.base), 1)

    def test_a_run_that_ends_seals_its_vault_with_what_made_it_and_never_loses_its_result_when_sealing_fails(self):
        from unittest import mock

        from dclab_rnd.agent_eval import conversation_exam as X

        pin = self.base / "pin.sha256"
        pin.write_text("abc123\n")
        with mock.patch.dict(X.SEALED, {"sealed_v9": pin}), mock.patch.dict("os.environ", {"DCLAB_THINKING_EFFORT": "xhigh"}):
            m = X.seal_after_run(self.folder, "CEX-099", "sealed_v9", ["plain", "case"], 3.98765, 4.5, None)
        self.assertEqual((m["arms"], m["spent_eur"], m["cap_eur"], m["thinking_effort"], m["pinned_sha256"]), (["plain", "case"], 3.9876, 4.5, "xhigh", "abc123"))
        self.assertEqual(m["counts"]["conversations"], 2)
        err = io.StringIO()
        from contextlib import redirect_stderr

        with mock.patch.dict(X.SEALED, {}, clear=True), redirect_stderr(err):  # no pin for the split: the sealing fails, the run does not
            self.assertIsNone(X.seal_after_run(make_vault(self.base, "CEX-097_sealed_v7"), "CEX-097", "sealed_v7", ["case"], 1.0, 2.0, "cap reached"))
        self.assertIn("could not be sealed now", err.getvalue())

    def test_sealing_afterwards_records_the_commit_given_and_never_todays_head(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(vault.main(["seal", str(self.folder), "--commit", "3c13ef80", "--note", "spent 3.9 of 4.5"], root=self.base), 0)
        m = json.loads((self.folder / vault.MANIFEST).read_text())
        self.assertEqual((m["commit"], m["note"], m["sealed_after_the_run"]), ("3c13ef80", "spent 3.9 of 4.5", True))
        other = make_vault(self.base, "CEX-096_sealed_v6")
        with redirect_stdout(io.StringIO()):
            vault.main(["seal", str(other)], root=self.base)
        self.assertIsNone(json.loads((other / vault.MANIFEST).read_text())["commit"])  # unknown stays unknown

    def test_the_manifest_holds_no_story(self):
        (self.folder / "conversations" / "case" / "s9-01" / "conversation.json").write_text(json.dumps({"transcript": [{"text": "SECRET STORY WORDS"}]}))
        m = vault.seal(self.folder, "CEX-099", "sealed_v9")
        self.assertNotIn("SECRET STORY WORDS", json.dumps(m))
        self.assertNotIn("SECRET STORY WORDS", (self.folder / vault.MANIFEST).read_text())


if __name__ == "__main__":
    unittest.main()
