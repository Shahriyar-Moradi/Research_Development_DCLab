"""GET /api/projects/{id}/review: the copilot reviews the project's exported notebook, read-only, in the demo's review shape."""
import os
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

    def test_fixes_are_written_on_request_checked_and_leave_the_review_unchanged(self):
        import json
        import re
        from unittest import mock

        c = self.client
        pid = self.new_project()
        c.put(f"/api/projects/{pid}/solution", json={**SOLUTION, "target": self.target}, headers=self.headers)
        review = c.get(f"/api/projects/{pid}/review").json()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_TIER_STANDARD_API_KEY": "", "DCLAB_TIER_STANDARD_LOCAL": ""}):
            self.assertFalse(c.get(f"/api/projects/{pid}/review").json()["fixes_available"])  # no model configured
            self.assertEqual(c.post(f"/api/projects/{pid}/review/fixes", headers=self.headers).status_code, 409)
        self.assertEqual(c.post(f"/api/projects/{pid}/review/fixes").status_code, 403)  # a model request needs the token like any write

        class Writer:
            """Answers from the prompt: for each finding, a sentence citing its first rule and its first pitfall or precedent."""
            def complete(self, messages, tools=None, max_tokens=None, **_):
                prompt, items = messages[-1]["content"], []
                for n, block in re.findall(r"Finding (\d+) \(.*?\n(.*?)(?=\n\nFinding |\Z)", prompt, re.S):
                    rule = re.search(r"\[(DCLAB-R\d+)\]", block)
                    other = re.search(r"\[((?:PIT|LEAK|FINDING)-[A-Za-z0-9_-]+)\]", block)
                    items.append({"finding": int(n), "fix": f"Change it [{rule.group(1)}]." + (f" The measured case shows why [{other.group(1)}]." if other else "")})
                text = json.dumps({"fixes": items})
                return {"content": text, "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": text}}

        keyed = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}})
        keyed.start()  # a configured tier; the transport below is scripted, so nothing leaves the machine
        self.addCleanup(keyed.stop)
        import threading
        import time

        gate = threading.Event()

        class Slow(Writer):
            def complete(self, messages, tools=None, max_tokens=None, **_):
                gate.wait(5)
                return super().complete(messages, tools, max_tokens)
        c.app.state.models.transport = lambda tier, purpose: Slow()
        first = {}
        running = threading.Thread(target=lambda: first.update(r=c.post(f"/api/projects/{pid}/review/fixes", headers=self.headers)))
        running.start()
        time.sleep(0.5)
        self.assertEqual(c.post(f"/api/projects/{pid}/review/fixes", headers=self.headers).status_code, 409)  # one request at a time per project
        gate.set()
        running.join(10)
        self.assertEqual(first["r"].status_code, 200)
        c.app.state.models.transport = lambda tier, purpose: Writer()
        self.assertTrue(c.get(f"/api/projects/{pid}/review").json()["fixes_available"])
        before = c.get(f"/api/projects/{pid}").json()
        out = c.post(f"/api/projects/{pid}/review/fixes", headers=self.headers)
        self.assertEqual(out.status_code, 200, out.text)
        got = out.json()
        self.assertEqual(got["findings"], len(review["findings"]))
        by_key = {f"{f['detector']}:{f['cell']}:{f['line']}": f for f in review["findings"]}
        self.assertEqual(sorted(got["fixes"]), sorted(by_key))  # a fix belongs to the finding with its key, not to a position
        for key, fix in got["fixes"].items():
            self.assertIn(by_key[key]["rules"][0], fix["cites"])  # the fix cites the finding's rule
            self.assertEqual(fix["written_by"], "model")
        self.assertEqual(c.get(f"/api/projects/{pid}/review").json()["findings"], review["findings"])  # the findings did not move
        self.assertEqual(len(c.get(f"/api/projects/{pid}").json()["transitions"]), len(before["transitions"]))


if __name__ == "__main__":
    unittest.main()
