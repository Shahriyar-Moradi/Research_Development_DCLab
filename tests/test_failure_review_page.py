"""19.12: the failure candidates a build kept are listed for the owner with their reviews, and a verdict given on the
Admin page is written where the next build reads it (datasets/failures/reviews.json); only the owner reviews."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient

    MISSING = ""
except ImportError as error:  # the agent environment may lack the web stack
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class FailureReviewTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.agentic.server import create_app

        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        folder = self.home / "datasets" / "failures" / "benchmark_candidates"
        folder.mkdir(parents=True)
        rows = [{"id": "f1", "detectors": ["identifier_target"], "flags": [{"detector": "identifier_target", "severity": "high", "message": "an id as the outcome"}],
                 "problem": "Predict churn", "family": "binary", "outcome": "built", "review": "pending"},
                {"id": "f2", "detectors": ["rephrasing"], "flags": [], "problem": "Forecast sales", "family": "forecasting", "outcome": None, "review": "pending"}]
        (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_the_candidates_are_listed_and_a_verdict_is_kept_for_the_next_build(self):
        listed = self.client.get("/api/datasets/failures").json()
        self.assertEqual((listed["total"], listed["pending"]), (2, 2))
        done = self.client.post("/api/datasets/failures/f1/review", json={"verdict": "accept"}, headers=self.h)
        self.assertEqual(done.status_code, 200, done.text)
        kept = json.loads((self.home / "datasets" / "failures" / "reviews.json").read_text())
        self.assertEqual(kept["f1"]["verdict"], "accept")
        again = {i["id"]: i["review"] for i in self.client.get("/api/datasets/failures").json()["items"]}
        self.assertEqual(again, {"f1": "accept", "f2": "pending"})

    def test_a_wrong_verdict_or_an_unknown_candidate_is_refused(self):
        self.assertEqual(self.client.post("/api/datasets/failures/f1/review", json={"verdict": "maybe"}, headers=self.h).status_code, 422)
        self.assertEqual(self.client.post("/api/datasets/failures/nope/review", json={"verdict": "accept"}, headers=self.h).status_code, 404)

    def test_only_the_owner_reviews(self):
        from dclab_rnd.accounts import guard
        from dclab_rnd.accounts.principal import Principal

        self.assertEqual({guard.permission("GET", "/api/datasets/failures"), guard.permission("POST", "/api/datasets/failures/{candidate_id}/review")}, {"admin"})
        with mock.patch.object(guard, "current", return_value=Principal(role="reviewer", user_id="u1", via="session")):
            self.assertEqual(self.client.get("/api/datasets/failures").status_code, 403)


if __name__ == "__main__":
    unittest.main()
