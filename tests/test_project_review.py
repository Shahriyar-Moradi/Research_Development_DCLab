"""GET /api/projects/{id}/review: the copilot reviews the project's exported notebook, read-only, in the demo's review shape."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOLUTION = {"task": "binary", "positive_label": None, "prediction_moment": "Right before the marketing call is placed.",
            "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": None, "notes": ""}


class ProjectReviewRouteTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.headers = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def new_project(self) -> str:
        c = self.client
        pid = c.post("/api/projects", json={"name": "Bank", "industry": "Banking", "goal": "who subscribes"}, headers=self.headers).json()["id"]
        sample = c.post(f"/api/projects/{pid}/data/sample", json={"key": "bank_marketing"}, headers=self.headers)
        self.assertEqual(sample.status_code, 200, sample.text)
        self.target = sample.json()["suggestion"]["target"]
        return pid

    def test_review_needs_a_solution(self):
        pid = self.new_project()
        self.assertEqual(self.client.get(f"/api/projects/{pid}/review").status_code, 409)
        self.assertEqual(self.client.get("/api/projects/nope/review").status_code, 404)

    def test_review_returns_summary_cells_and_findings_without_side_effects(self):
        c = self.client
        pid = self.new_project()
        saved = c.put(f"/api/projects/{pid}/solution", json={**SOLUTION, "target": self.target}, headers=self.headers)
        self.assertEqual(saved.status_code, 200, saved.text)
        before = c.get(f"/api/projects/{pid}").json()

        r = c.get(f"/api/projects/{pid}/review")
        self.assertEqual(r.status_code, 200, r.text)
        review = r.json()
        self.assertIn("findings", review["summary"])
        self.assertIn("by_severity", review["summary"])
        self.assertIsInstance(review["findings"], list)
        self.assertEqual(review["summary"]["findings"], len(review["findings"]))
        self.assertTrue(review["cells"] and {c["type"] for c in review["cells"]} <= {"markdown", "code"})
        for f in review["findings"]:
            self.assertEqual(set(f), {"detector", "severity", "cell", "line", "title", "message", "suggestion", "rules", "proof"})
            self.assertTrue(all(isinstance(p, str) for p in f["proof"]))
            self.assertLess(f["cell"], len(review["cells"]))

        after = c.get(f"/api/projects/{pid}").json()
        self.assertEqual(len(after["transitions"]), len(before["transitions"]))
        self.assertNotIn("captured", after)


if __name__ == "__main__":
    unittest.main()
