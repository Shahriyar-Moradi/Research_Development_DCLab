"""Package 32.6: fixes, linked.

A planted failure fixed by one commit yields one proposed fix: the failing test with its whole output, the run that passed
it next, and the commit between with its subject and the files it changed; a failure no later run passed stays open; a fix
in the working tree before any commit says so; nothing is accepted without a person's yes, and an accepted fix is kept
beside the development records with who accepted it.
"""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def run(runs: Path, ident: str, at: str, commit: str, units: list[tuple[str, bool, str]], pg: bool = False) -> None:
    day = runs / at[:10]
    day.mkdir(parents=True, exist_ok=True)
    with gzip.open(day / f"{ident}.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "test_run", "run": ident}) + "\n")
        for name, ok, output in units:
            fh.write(json.dumps({"kind": "unit", "run": ident, "file": name, "ok": ok, "output": output}) + "\n")
    with open(runs / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run": ident, "at": at, "commit": commit, "dirty": False, "pg": pg, "ok": all(u[1] for u in units),
                             "failing_files": [u[0] for u in units if not u[1]], "path": f"{at[:10]}/{ident}.jsonl.gz"}) + "\n")


class FixesTest(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp()) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "t@example.org")
        git(self.repo, "config", "user.name", "T")
        (self.repo / "a.py").write_text("x = 1\n")
        git(self.repo, "add", "a.py")
        git(self.repo, "commit", "-q", "-m", "start")
        self.broken = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "a.py").write_text("x = 2\n")
        git(self.repo, "commit", "-q", "-am", "Fix the value the test expects")
        self.fixed = git(self.repo, "rev-parse", "HEAD")
        self.runs = Path(tempfile.mkdtemp()) / "test_runs"
        failing = "F\nFAIL: test_value (test_a.ATest)\nAssertionError: 1 != 2\n"
        run(self.runs, "r1", "2026-10-20T10:00:00+00:00", self.broken, [("test_a.py", False, failing), ("test_b.py", True, "OK")])
        run(self.runs, "r2", "2026-10-20T11:00:00+00:00", self.fixed, [("test_a.py", True, "OK")])
        run(self.runs, "r3", "2026-10-20T12:00:00+00:00", self.fixed, [("test_c.py", False, "E\nERROR: test_x (test_c.CTest)\n")])

    def test_a_failure_fixed_by_a_commit_is_one_proposed_fix_with_its_commit_and_files(self):
        from dclab_rnd.store import fixes

        found = {f["test"]: f for f in fixes.link(self.runs, self.repo)}
        fix = found["test_a.py"]
        self.assertEqual((fix["status"], fix["failing_run"], fix["passing_run"]), ("proposed", "r1", "r2"))
        self.assertEqual(fix["tests"], ["test_value"])
        self.assertIn("AssertionError: 1 != 2", fix["failing_output"])  # whole
        self.assertEqual([(c["subject"], c["files"]) for c in fix["commits"]], [("Fix the value the test expects", ["a.py"])])
        self.assertFalse(fix["in_working_tree"])
        self.assertEqual(found["test_c.py"]["status"], "open")  # no later run passed it
        self.assertNotIn("test_b.py", found)

    def test_nothing_is_accepted_without_a_persons_yes_and_an_open_one_cannot_be(self):
        from dclab_rnd.store import fixes

        out = Path(tempfile.mkdtemp()) / "fixes"
        ident = next(f["id"] for f in fixes.link(self.runs, self.repo) if f["test"] == "test_a.py")
        self.assertFalse(out.exists())  # linking writes nothing there
        with self.assertRaises(ValueError):
            fixes.accept(next(f["id"] for f in fixes.link(self.runs, self.repo) if f["test"] == "test_c.py"), "owner", runs_dir=self.runs, out=out, repo=self.repo)
        path = fixes.accept(ident, "owner", note="right", runs_dir=self.runs, out=out, repo=self.repo)
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((doc["status"], doc["source"], doc["accepted_by"]), ("accepted", "person", "owner"))


if __name__ == "__main__":
    unittest.main()
