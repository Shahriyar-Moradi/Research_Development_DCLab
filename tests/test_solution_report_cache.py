"""solution_report.latest: one build per set of records, shared by the download and the chat's Reporter (2026-10-21)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.contracts import ReportSection, SolutionReport  # noqa: E402
from dclab_rnd.studio import ProjectStore, solution_report  # noqa: E402

REPORT = SolutionReport(title="r", sections=[ReportSection(title="Next steps", body="- monitor")])


class LatestTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        self.pid = self.store.create("Churn")["id"]
        self.records = {"final": {"evidence": {"holdout_metrics": {"roc_auc": 0.7}}}}
        for name, value in (("records", lambda pid: self.records),):
            patch = mock.patch.object(self.store, name, side_effect=value)
            patch.start()
            self.addCleanup(patch.stop)
        build = mock.patch.object(solution_report, "build", return_value=REPORT)
        self.build = build.start()
        self.addCleanup(build.stop)

    def test_the_same_records_are_built_once(self):
        first, again = solution_report.latest(self.store, self.pid), solution_report.latest(self.store, self.pid)
        self.assertEqual((first["cached"], again["cached"], self.build.call_count), (False, True, 1))
        self.assertEqual(again["report"].sections[0].title, "Next steps")

    def test_changed_records_a_private_project_and_an_old_cache_build_again(self):
        solution_report.latest(self.store, self.pid)
        self.records = {"final": {"evidence": {"holdout_metrics": {"roc_auc": 0.8}}}}
        self.assertFalse(solution_report.latest(self.store, self.pid)["cached"])
        p = self.store.get(self.pid)
        p["settings"]["private"] = True
        self.store.save(p)
        self.assertFalse(solution_report.latest(self.store, self.pid)["cached"])  # no prose from before it was made private
        self.store.update(self.pid, solution_report={"digest": self.store.get(self.pid)["solution_report"]["digest"], "markdown": "# old"})  # the shape before 2026-10-21
        self.assertFalse(solution_report.latest(self.store, self.pid)["cached"])
        self.assertEqual(self.build.call_count, 4)


if __name__ == "__main__":
    unittest.main()
