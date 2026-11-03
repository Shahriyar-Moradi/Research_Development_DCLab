"""The external layer (B8, 2026-11-03): what a run needs is checked and named, and a run is refused until all is here and a cap is given."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.bench import external  # noqa: E402


class ExternalTests(unittest.TestCase):
    def test_the_check_names_what_is_missing_and_never_a_key(self):
        with mock.patch.dict("os.environ", {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": "secret-value"}):
            state = external.check()
        self.assertIn("Kaggle key set", state["items"])
        self.assertNotIn("secret-value", str(state))

    def test_a_run_is_refused_until_ready_and_capped(self):
        with mock.patch.object(external, "check", return_value={"ready": False, "items": {"mlebench grader installed": False}}):
            with self.assertRaisesRegex(RuntimeError, "mlebench grader installed"):
                external.run(cap_eur=1, yes=True)
        with mock.patch.object(external, "check", return_value={"ready": True, "items": {}}):
            with self.assertRaisesRegex(RuntimeError, "--cap-eur"):
                external.run()

    def test_the_plan_lists_its_steps_and_limits(self):
        p = external.plan(["a-competition"])
        self.assertEqual(p["competitions"], ["a-competition"])
        self.assertTrue(p["steps"] and p["limits"])


if __name__ == "__main__":
    unittest.main()
