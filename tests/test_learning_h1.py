"""H1 of the harness that learns: the read-only layer and the refusal test.

A learned change touching the validator, the gateway, the loop, the grader, the generator, a prompt or the learning
harness itself is refused, and so is a path outside the repository; a skill's own folder is not. The refusal test
refuses every one of its bad moves on today's code, and reports a guard weakened on purpose: a test that cannot fail
proves nothing. Every tool entry point checks argument types as the loop does.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProtectedTests(unittest.TestCase):
    def test_a_learned_change_may_write_only_skills_and_never_the_layer(self):
        from dclab_rnd.learning.protected import Protected, check_diff, refused

        inside = ["dclab_rnd/studio/graph.py", "dclab_rnd/intern/tools.py", "dclab_rnd/draft/orchestrator.py", "dclab_rnd/models/settings.py",
                  "dclab_core/challenge_pack.py", "dclab_rnd/kinds/binary.py", "dclab_rnd/prompts/intern.md", "dclab_rnd/learning/refusal.py",
                  "dclab_rnd/agent_eval/gate.py", "evidence/campaigns/learning_v1/manifest.json", "tests/test_learning_h1.py", "Makefile",
                  ".git/hooks/pre-commit", str(ROOT / "dclab_core" / "variants.py"), "dclab_rnd/learning/../learning/protected.py"]
        self.assertEqual(refused(inside), inside)
        self.assertEqual(refused(["DCLAB_RND/Studio/GRAPH.py", "Dclab_Rnd/Models/Gateway.py"]), ["DCLAB_RND/Studio/GRAPH.py", "Dclab_Rnd/Models/Gateway.py"])
        self.assertEqual(refused(["/etc/passwd", "../outside.py", "skills/../Makefile"]), ["/etc/passwd", "../outside.py", "skills/../Makefile"])
        self.assertEqual(refused(["dclab_rnd/studio/report.py", "dclab_rnd/new_module.py"]), ["dclab_rnd/studio/report.py", "dclab_rnd/new_module.py"])  # outside the allow list
        allowed = ["skills/reconcile_sources/SKILL.md", "skills/reconcile_sources/run.py", "Skills/x/SKILL.md"]
        self.assertEqual(refused(allowed), [])
        with self.assertRaises(Protected):
            check_diff(["skills/a/SKILL.md", "dclab_rnd/agents/registry.py"])
        check_diff(allowed)  # no error

    def test_a_commit_range_counts_a_rename_as_both_paths_and_refuses_a_link(self):
        from dclab_rnd.learning import protected

        raw = (":000000 100644 0000000 1111111 A\tskills/a/SKILL.md\n"
               ":100644 000000 2222222 0000000 D\tdclab_core/challenge_pack.py\n"   # a rename listed as a deletion and an addition
               ":000000 100644 0000000 2222222 A\tskills/x/challenge_pack.py\n"
               ":000000 120000 0000000 3333333 A\tskills/b/link\n")
        fake = subprocess.CompletedProcess([], 0, stdout=raw, stderr="")
        with mock.patch.object(protected.subprocess, "run", return_value=fake) as ran:
            self.assertEqual(protected.check_git("abc"), ["dclab_core/challenge_pack.py", "skills/b/link"])
        self.assertIn("--no-renames", ran.call_args.args[0])


class RefusalTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_every_bad_move_is_refused_today(self):
        from dclab_rnd.learning import refusal

        report = refusal.run()
        self.assertTrue(report["passed"], [c for c in report["cases"] if not c["refused"]])
        self.assertGreaterEqual(report["of"], 11)

    def test_a_weakened_guard_is_reported(self):
        """The refusal test must be able to fail: each guard loosened on purpose shows up as a case allowed."""
        from dclab_rnd.learning import refusal
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.studio import graph

        real_check = graph.check

        def as_a_person(project, move, actor="human", **args):  # the review's attack: the guard's agent made a "human"
            return real_check(project, move, "human", **args)

        loosened = {
            "the holdout read twice by an agent": mock.patch.object(graph, "check", as_a_person),
            "a stage run before the one it needs": mock.patch.object(graph, "check", lambda *a, **k: graph.Verdict("run_stage", "agent", "allowed", None, None, "ok")),
            "a request past the run's cap": mock.patch.object(Gateway, "reserve", lambda self, *a, **k: (None, lambda: None)),
            "a learned change to the read-only layer": mock.patch("dclab_rnd.learning.protected.refused", lambda paths: []),
        }
        for case, patch in loosened.items():
            with patch:
                report = refusal.run()
            row = next(r for r in report["cases"] if r["case"] == case)
            self.assertFalse(row["refused"], case)
            self.assertFalse(report["passed"], case)

    def test_the_command_exits_non_zero_when_a_case_is_allowed(self):
        from dclab_rnd.learning import __main__ as cli, refusal

        with mock.patch.object(refusal, "run", return_value={"cases": [{"case": "x", "refused": False, "detail": "allowed"}], "refused": 0, "of": 1,
                                                             "passed": False}), mock.patch("builtins.print"):
            self.assertEqual(cli.main(["refusal"]), 1)


class OneValidationLevelTests(unittest.TestCase):
    def test_every_entry_point_checks_argument_types(self):
        import tempfile

        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore

        box = Toolbox(ProjectStore(Path(tempfile.mkdtemp()) / "projects"))
        self.assertIn("Invalid argument for describe_data: project_id", box.call("describe_data", {"project_id": 5})["error"])
        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Predict which customers cancel")
        self.assertIn("Invalid argument for get_profile: offset", HomeAgent(store, None).run_tool(draft["id"], "get_profile", {"offset": "x"})["error"])


if __name__ == "__main__":
    unittest.main()
