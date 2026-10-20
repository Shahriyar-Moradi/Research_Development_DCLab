"""Packages 32.21 and 32.23: code with its execution, and search and reading.

A run that fails and the next run of the same code that passes are one fix (both versions, the error, the passing output);
a notebook keeps its errors and what it printed, never a printed table; an evidence search keeps what it offered, and the
answer that cited some of it keeps what it cited and whether it passed; together they are one retrieval example.
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


@unittest.skipIf(MISSING, MISSING)
class SearchesAndCodeTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp()) / "agent_runs"
        self.home.mkdir()
        experience.install(experience.open_log(self.home))
        self.addCleanup(experience.install, None)

    def test_a_failed_run_and_its_fix_are_one_record_and_a_notebook_keeps_its_errors(self):
        from dclab_rnd import code_runs
        from dclab_rnd.store.keepers.standard import notebook

        code_runs.record("cell:1", "x = 1 / 0", stderr="ZeroDivisionError: division by zero", returncode=1, kind="cell")
        code_runs.record("cell:1", "x = 1 / 1", stdout="1.0", returncode=0, kind="cell")
        lines = [json.loads(line) for p in (self.home / "code_runs").rglob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
        fix = code_runs.fixes(lines)[0]
        self.assertEqual((fix["before"], fix["after"], fix["output"]), ("x = 1 / 0", "x = 1 / 1", "1.0"))
        self.assertIn("ZeroDivisionError", fix["error"])
        nb = {"cells": [{"cell_type": "code", "source": "df.head()", "outputs": [
            {"output_type": "execute_result", "data": {"text/plain": "email\nQ7Kx@example.org"}},
            {"output_type": "stream", "name": "stdout", "text": "email,amount\nQ7Kx@example.org,10\nfitted\n"},
            {"output_type": "error", "ename": "KeyError", "evalue": "'churn'", "traceback": ["KeyError: 'churn'"]}]}]}
        kept = json.loads(notebook(json.dumps(nb).encode()))
        outputs = kept["cells"][0]["outputs"]
        self.assertEqual([o["output_type"] for o in outputs], ["stream", "error"])
        self.assertNotIn("Q7Kx@example.org", json.dumps(kept))
        self.assertIn("fitted", outputs[0]["text"])
        self.assertEqual(outputs[1]["ename"], "KeyError")

    def test_a_search_and_the_answer_citing_it_are_one_retrieval_example(self):
        from dclab_rnd import cited, searches
        from dclab_rnd.store import Layout, Store, shapers
        from dclab_rnd.store.sets import others

        with experience.bind(run="run-1"):
            searches.offered("evidence", "duration leakage", ["LEAK-duration", "EXP-1"], "bm25")
            cited.check("The call's duration is known only after it ends [LEAK-duration].", {"LEAK-duration": "duration is known only after"})
        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(layout)
        path = next((self.home / "searches").rglob("*.jsonl"))
        store.keep_log("actions", "dev", path.relative_to(self.home).as_posix(), path, consent="train")
        shapers.shape(layout)
        found = [x for x in others.retrievals(layout) if x.query == "duration leakage"]
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].offered, found[0].cited, found[0].passed), (["LEAK-duration", "EXP-1"], ["LEAK-duration"], True))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(MISSING, MISSING)
class CommitsTest(unittest.TestCase):
    def test_each_commit_is_kept_once_as_its_patch_generated_files_left_out(self):
        import os
        import subprocess
        from unittest import mock

        from dclab_rnd.store import Layout, Store, conductor
        from dclab_rnd.store.keepers import commits

        repo = Path(tempfile.mkdtemp()) / "repo"
        repo.mkdir()
        run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)  # noqa: E731
        run("init", "-q")
        (repo / "a.py").write_text("x = 1\n")
        (repo / "docs").mkdir()
        (repo / "docs" / "r.pdf").write_text("generated")
        run("add", ".")
        run("-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", "Fix the value")
        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        with mock.patch.dict(os.environ, {"DCLAB_STORE": str(layout.root)}):
            conductor.conduct({"repo": repo, "agent_runs": Path(tempfile.mkdtemp())}, layout, {"commits": commits.KEEPER})
            conductor.conduct({"repo": repo, "agent_runs": Path(tempfile.mkdtemp())}, layout, {"commits": commits.KEEPER})
        kept = [e for e in Store(layout).entries("fixes") if e.source.startswith("git/")]
        self.assertEqual(len(kept), 1)  # once
        patch = Store(layout).read(kept[0]).decode()
        self.assertIn("Fix the value", patch)
        self.assertIn("a.py", patch)
        self.assertNotIn("r.pdf", patch)  # a generated file left out
