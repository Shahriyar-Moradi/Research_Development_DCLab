"""The owner's one-time exception (2026-11-11): a new model reruns a spent sealed set once, named apart and marked not
unseen; Claude cannot rerun it, a second rerun is refused, and without the exception the old refusal stands."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import sealed  # noqa: E402


class SealedRerunRuleTests(unittest.TestCase):
    def test_the_exception_is_asked_for_in_words_once_per_model_and_never_by_claude(self):
        self.assertIn("--model-rerun", sealed.refusal(None, "qwen/qwen3.8-flash", []))
        self.assertIn("not 'yes'", sealed.refusal("yes", "qwen/qwen3.8-flash", []))
        self.assertIn("Claude already took", sealed.refusal(sealed.EXCEPTION, "claude-sonnet-5-5", []))
        self.assertIn("Claude already took", sealed.refusal(sealed.EXCEPTION, "anthropic/claude-sonnet-5.5", []))  # Claude via OpenRouter
        self.assertIn("already reran", sealed.refusal(sealed.EXCEPTION, "qwen/qwen3.8-flash", [Path("CEX-050_sealed_v1_rerun_qwen3-8-flash.json")]))
        self.assertIsNone(sealed.refusal(sealed.EXCEPTION, "qwen/qwen3.8-flash", []))

    def test_a_rerun_is_named_apart_and_marked_not_unseen(self):
        self.assertEqual(sealed.suffix("qwen/qwen3.8-flash"), "rerun_qwen3-8-flash")
        stamp = sealed.stamp("qwen/qwen3.8-flash", ["CEX-013"])
        self.assertIs(stamp["unseen"], False)
        self.assertEqual(stamp["rerun"]["of"], ["CEX-013"])


class ConversationExamRerunTests(unittest.TestCase):
    def test_a_spent_sealed_set_is_still_refused_without_the_exception(self):
        from dclab_rnd.agent_eval import conversation_exam as X

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(X, "RESULTS", Path(tmp)), \
                mock.patch.object(X, "served_models", return_value={"arms": "qwen/qwen3.8-flash"}):
            (Path(tmp) / "CEX-013_sealed_v1.json").write_text("{}")
            err = io.StringIO()
            with redirect_stderr(err):
                code = X.main(["--split", "sealed_v1", "--arms", ",".join(X.SEALED_ARMS["sealed_v1"]), "--yes", "--cap-eur", "1"])
            self.assertEqual(code, 2)
            self.assertIn("--model-rerun", err.getvalue())
            (Path(tmp) / "CEX-050_sealed_v1_rerun_qwen3-8-flash.json").write_text("{}")
            err = io.StringIO()
            with redirect_stderr(err):
                code = X.main(["--split", "sealed_v1", "--arms", ",".join(X.SEALED_ARMS["sealed_v1"]), "--yes", "--cap-eur", "1",
                               "--model-rerun", sealed.EXCEPTION])
            self.assertEqual(code, 2)
            self.assertIn("already reran", err.getvalue())


class SuiteRerunGuardTests(unittest.TestCase):
    def test_journeys_refuse_a_rerun_where_no_model_plays(self):
        from dclab_rnd.agent_eval.journeys import __main__ as J

        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            code = J.main(["--split", "sealed_v2", "--model-rerun", sealed.EXCEPTION, "--yes"])
        self.assertEqual(code, 2)
        self.assertIn("rules alone are no model's rerun", out.getvalue())

    def test_home_refuses_the_exception_on_an_unspent_set_and_a_failed_rerun_never_counts(self):
        from dclab_rnd.agent_eval import home

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(home, "RESULTS", Path(tmp)):
            (Path(tmp) / "HOME-021_sealed_v5.json").write_text("{}")
            (Path(tmp) / "HOME-040_sealed_v5_rerun_qwen3-8-flash_unscored.json").write_text("{}")
            done = [p for p in home.sealed_runs("sealed_v5") if p.stem.endswith(sealed.suffix("qwen/qwen3.8-flash"))]
            self.assertEqual(done, [])  # the failed attempt does not use up the rerun


class SealedRunMatchingTests(unittest.TestCase):
    def test_a_rerun_counts_for_its_own_set_only(self):
        """"sealed" must never match sealed_v5's files, and a rerun of sealed_v5 counts for sealed_v5."""
        from dclab_rnd.agent_eval import home, journeys

        for module, prefix in ((home, "HOME"), (journeys, "JRN")):
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(module, "RESULTS", Path(tmp)):
                for name in (f"{prefix}-011_sealed.json", f"{prefix}-021_sealed_v5.json", f"{prefix}-030_sealed_v5_rerun_qwen3-8-flash.json"):
                    (Path(tmp) / name).write_text("{}")
                self.assertEqual([p.name for p in module.sealed_runs("sealed")], [f"{prefix}-011_sealed.json"], prefix)
                self.assertEqual(len(module.sealed_runs("sealed_v5")), 2, prefix)


if __name__ == "__main__":
    unittest.main()
