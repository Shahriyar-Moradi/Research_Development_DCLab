"""The API's shape (package 10.1): server.py only builds the app, the OpenAPI schema describes every route, and a bad
request body is a 422 everywhere."""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.api_models import api_routes
        from dclab_rnd.agentic.server import create_app

        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        cls.routes = api_routes(cls.app)  # included routers too

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_server_only_builds_the_app(self):
        source = (ROOT / "dclab_rnd/agentic/server.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"@app\.(get|post|put|patch|delete)\(|add_api_route|os\.environ", source))
        modules = {r.endpoint.__module__ for r in self.routes}
        self.assertNotIn("dclab_rnd.agentic.server", modules)  # every route lives in a router
        self.assertGreaterEqual(len(self.routes), 90)  # all of them: the schema and the Platform page see the included routers

    def test_every_route_has_a_response_model_or_says_what_it_sends(self):
        from fastapi.responses import FileResponse, Response, StreamingResponse

        undocumented = [f"{sorted(r.methods)} {r.path}" for r in self.routes if r.include_in_schema and r.response_model is None
                        and r.response_class not in (Response, FileResponse, StreamingResponse)]
        self.assertEqual(undocumented, [])
        schema = self.app.openapi()
        for path, methods in schema["paths"].items():
            for method, op in methods.items():
                ok = {k: v for k, v in op["responses"].items() if k.startswith("2")}
                self.assertTrue(ok, f"{method} {path}")
                if method in ("post", "put", "patch") and "requestBody" not in op:
                    route = next(r for r in self.routes if r.path == path and method.upper() in r.methods)
                    self.assertFalse(route.dependant.body_params, f"{method} {path} reads a body the schema does not describe")

    def test_no_route_reads_its_body_by_hand(self):
        modules = [f"agentic/routers/{m}.py" for m in ("core", "runs", "projects", "intern")] + ["draft/api.py"]
        modules += [f"agentic/pages/{m}.py" for m in ("ops", "lab", "learn", "platform", "evidence")]
        for module in modules:
            source = (ROOT / "dclab_rnd" / module).read_text(encoding="utf-8")
            self.assertNotIn("await request.json()", source, module)

    def test_a_bad_body_is_a_422_everywhere(self):
        checked = []
        for r in self.routes:
            if not r.dependant.body_params:
                continue  # a route that takes no body, or raw bytes (the uploads)
            path = re.sub(r"\{[^}]+\}", "x", r.path)
            for method in sorted(r.methods - {"HEAD", "OPTIONS", "GET"}):
                response = self.client.request(method, path, content=b"{not json", headers={**self.h, "content-type": "application/json"})
                self.assertEqual(response.status_code, 422, f"{method} {r.path}: {response.status_code} {response.text[:200]}")
                checked.append(f"{method} {r.path}")
        self.assertGreater(len(checked), 25)


class PayloadTests(unittest.TestCase):
    """Every page's data comes back as before: no 5xx on any GET with real ids, and no key a response did not have."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.api_models import api_routes
        from dclab_rnd.agentic.server import create_app

        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        c = cls.client
        cls.h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
        cls.project = c.post("/api/projects", json={"name": "Sweep", "goal": "Predict churn"}, headers=cls.h).json()
        cls.draft = c.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=cls.h).json()
        cls.session = c.post("/api/intern/sessions?wait=true", json={"task": "Something the samples do not cover at all"}, headers=cls.h).json()
        cls.routes = api_routes(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_no_get_route_fails_on_real_ids(self):
        ids = {"project_id": self.project["id"], "draft_id": self.draft["id"], "session_id": self.session["id"], "record_id": "DCLAB-R01",
               "run_id": "missing", "job_id": f"project-{self.project['id']}", "trial_id": "trial-001", "filename": "result.json"}
        failed = []
        for r in self.routes:
            if "GET" not in r.methods or r.path.endswith("/events"):
                continue  # the event stream stays open; tests/test_draft_flow.py reads it
            path = re.sub(r"\{(\w+)\}", lambda m: ids.get(m.group(1), "x"), r.path)
            query = "?path=README.md" if r.path == "/api/research/file" else ""
            status = self.client.get(path + query).status_code
            if status >= 500:
                failed.append(f"{r.path}: {status}")
        self.assertEqual(failed, [])

    def test_a_response_has_the_keys_it_had_and_no_new_nulls(self):
        listed = next(p for p in self.client.get("/api/projects").json() if p["id"] == self.project["id"])
        for key in ("records", "activity", "graph", "transitions"):  # the list is light: the detail carries these
            self.assertNotIn(key, listed)
        self.assertNotIn("trace", self.session)  # the session's trace comes with GET /api/intern/sessions/{id} only
        self.assertIn("trace", self.client.get(f"/api/intern/sessions/{self.session['id']}").json())
        self.assertNotIn("events", self.client.get("/api/runs").json()[0] if self.client.get("/api/runs").json() else {})
        self.assertIsInstance(self.client.get("/api/knowledge").json(), list)

    def test_a_kaggle_search_returns_its_list(self):
        from unittest import mock
        with mock.patch("dclab_rnd.connectors.kaggle.search", return_value=[{"ref": "a/b", "title": "t"}]):
            r = self.client.post("/api/connectors/kaggle/search", json={"query": "churn"}, headers=self.h)
        self.assertEqual((r.status_code, r.json()), (200, [{"ref": "a/b", "title": "t"}]))

    def test_a_json_body_without_a_json_content_type_is_still_read(self):
        for kind in (None, "text/plain"):
            headers = {**self.h, **({"content-type": kind} if kind else {})}
            r = self.client.post(f"/api/projects/{self.project['id']}/ask", content=b'{"question": "Is there a leak?"}', headers=headers)
            self.assertEqual(r.status_code, 200, (kind, r.text[:200]))


if __name__ == "__main__":
    unittest.main()
