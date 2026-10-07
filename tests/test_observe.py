"""Logs, health and metrics (package 12.4): a request id the response carries, one JSON log line per request with
its route pattern and never a value, /healthz and /readyz, counters in Prometheus's format on a port of their own,
job failures counted and shown on the Compute page."""
import io
import json
import logging
import os
import socket
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ObserveTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.app = create_app(Path(tempfile.mkdtemp()))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_health_and_readiness(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
        ready = self.client.get("/readyz")
        self.assertEqual((ready.status_code, ready.json()["checks"]["files"]), (200, "ok"))
        from dclab_rnd.agentic.routers import core

        with mock.patch.object(core, "readiness", return_value={"database": False, "files": True}):
            down = self.client.get("/readyz")
        self.assertEqual((down.status_code, down.json()["checks"]), (503, {"database": "failed", "files": "ok"}))

    def test_a_load_balancer_probing_by_ip_reaches_the_probes_and_nothing_else(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        sys.path.insert(0, str(ROOT / "tests"))
        import pgtest

        url = pgtest.require()  # sign-in (required with a public host name) needs PostgreSQL
        with mock.patch.dict(os.environ, {"DCLAB_ALLOWED_HOSTS": "dclab.example.com", "DCLAB_AUTH": "password", "DCLAB_DATABASE_URL": url}), \
                TestClient(create_app(Path(tempfile.mkdtemp()))) as client:
            probe = {"Host": "10.40.10.7:8765"}  # how an ALB health check addresses a task
            self.assertEqual(client.get("/healthz", headers=probe).status_code, 200)
            self.assertIn(client.get("/readyz", headers=probe).status_code, (200, 503))
            self.assertEqual(client.get("/api/config", headers=probe).status_code, 400)  # everything else still checks the name
            self.assertIn(client.post("/healthz", headers=probe).status_code, (400, 403))  # only a GET or HEAD probe passes

    def test_a_request_id_and_one_log_line_with_the_route_never_its_values(self):
        from dclab_rnd import observe

        json_lines = mock.patch.dict(os.environ, {"DCLAB_LOG_FORMAT": "json"})
        json_lines.start()
        self.addCleanup(json_lines.stop)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(observe.JsonFormatter())
        observe.log.addHandler(handler)
        observe.log.setLevel(logging.INFO)
        self.addCleanup(observe.log.removeHandler, handler)
        pid = self.client.post("/api/projects", json={"name": "Logged", "goal": "g"}, headers=self.h).json()["id"]
        answer = self.client.get(f"/api/projects/{pid}?secret=hunter2", headers={"X-Request-ID": "trace-1234abcd"})
        self.assertEqual(answer.headers["X-Request-ID"], "trace-1234abcd")  # a proxy's id is kept
        self.assertNotEqual(self.client.get("/healthz", headers={"X-Request-ID": "bad id; drop table"}).headers["X-Request-ID"], "bad id; drop table")
        lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        mine = next(line for line in lines if line.get("request_id") == "trace-1234abcd")
        self.assertEqual((mine["message"], mine["route"], mine["status"], mine["method"]), ("request", "/api/projects/{project_id}", 200, "GET"))
        self.assertNotIn(pid, json.dumps(mine))
        self.assertNotIn("hunter2", stream.getvalue())  # never a query string

    def test_counters_on_a_port_of_their_own(self):
        from dclab_rnd import observe

        before = observe.METRICS.value("dclab_http_requests_total", method="GET", route="/api/projects", status=200)
        self.client.get("/api/projects")
        self.assertEqual(observe.METRICS.value("dclab_http_requests_total", method="GET", route="/api/projects", status=200), before + 1)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with mock.patch.dict(os.environ, {"DCLAB_METRICS_PORT": str(port)}), mock.patch.object(observe, "_SERVER", [None]):
            served = observe.serve_metrics()
            text = urllib.request.urlopen(f"http://127.0.0.1:{served}/metrics", timeout=5).read().decode()
            observe._SERVER[0].shutdown()
            observe._SERVER[0].server_close()
        self.assertIn("# TYPE dclab_http_requests_total counter", text)
        self.assertIn('dclab_http_requests_total{method="GET",route="/api/projects",status="200"}', text)
        self.assertNotIn("/metrics", [r.path for r in self.app.routes if hasattr(r, "path")])  # not on the public port

    def test_job_failures_are_counted_and_shown_on_the_compute_page(self):
        from dclab_rnd import observe
        from dclab_rnd.jobs import worker as jw

        jw.handler("t-observe", lambda env, payload: (_ for _ in ()).throw(ValueError("the table is broken")), lambda *a: None)
        self.addCleanup(jw.HANDLERS.pop, "t-observe", None)
        s = self.app.state.services
        before = observe.METRICS.value("dclab_job_failures_total", kind="t-observe", status="failed")
        job = s.job_store.enqueue("t-observe", "k1", {"stages": ["data"]}, worker=s.worker.id)
        s.worker.execute(job)
        self.assertEqual(observe.METRICS.value("dclab_job_failures_total", kind="t-observe", status="failed"), before + 1)
        failures = self.client.get("/api/ops/jobs").json()["failures"]
        mine = next(f for f in failures if f["id"] == job["id"])
        self.assertEqual((mine["status"], mine["retry"]), ("failed", f"/api/jobs/{job['id']}/retry"))
        self.assertIn("the table is broken", mine["error"])

    def test_model_requests_are_counted_where_the_gateway_makes_them(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_models_gateway import CLEAN, gateway

        from dclab_rnd import observe
        from dclab_rnd.models import FileUsage

        before = observe.METRICS.value("dclab_model_tokens_total", purpose="home_agent", direction="in")
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-test-not-real"}):
            gw, _ = gateway(FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl"))
            gw.client("home_agent").complete([{"role": "user", "content": "x"}])
        self.assertEqual(observe.METRICS.value("dclab_model_tokens_total", purpose="home_agent", direction="in"), before + 3)

    def test_a_made_up_method_is_one_label_and_an_error_carries_its_id(self):
        from dclab_rnd import observe

        for n in range(3):
            self.client.request(f"XMETHOD{n}", "/nothing")
        self.assertFalse([k for k in observe.METRICS._values if ("method", "XMETHOD0") in k[1]])
        self.assertGreaterEqual(observe.METRICS.value("dclab_http_requests_total", method="OTHER", route="unmatched", status=403), 3)

        def boom():
            raise ValueError("a secret value in the message")
        self.app.add_api_route("/api/boom-observe", boom)
        client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(self.app, raise_server_exceptions=False)
        answer = client.get("/api/boom-observe", headers={"X-Request-ID": "trace-err-0001"})
        self.assertEqual((answer.status_code, answer.headers.get("X-Request-ID"), answer.json()["request_id"]), (500, "trace-err-0001", "trace-err-0001"))
        self.assertNotIn("secret", answer.text)

    def test_a_busy_or_bad_metrics_port_leaves_the_server_running_and_counters_keep_every_digit(self):
        from dclab_rnd import observe

        taken = socket.socket()
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        self.addCleanup(taken.close)
        for value in (str(taken.getsockname()[1]), "abc"):
            with mock.patch.dict(os.environ, {"DCLAB_METRICS_PORT": value}), mock.patch.object(observe, "_SERVER", [None]):
                self.assertIsNone(observe.serve_metrics())
        metrics = observe.Metrics()
        metrics.add("dclab_model_tokens_total", 123456789, purpose="x", direction="in")
        metrics.add("dclab_job_seconds_sum", 0.25, mtype="summary", kind="k")
        text = metrics.render()
        self.assertIn('dclab_model_tokens_total{direction="in",purpose="x"} 123456789\n', text)
        self.assertIn('dclab_job_seconds_sum{kind="k"} 0.25', text)

    def test_a_job_a_dead_worker_left_is_counted_when_it_is_recovered(self):
        import time as clock

        from dclab_rnd import observe

        s = self.app.state.services
        before = observe.METRICS.value("dclab_job_failures_total", kind="stage", status="interrupted")
        dead = s.job_store.enqueue("stage", "dead-observe", {"project_id": "nothing", "stages": ["data"]}, worker="dead-host:1:x")
        s.job_store._change(dead["id"], lambda j: j.update(heartbeat=clock.time() - 3600) or True)
        self.assertIn(dead["id"], [j["id"] for j in s.worker.recover()])
        self.assertEqual(observe.METRICS.value("dclab_job_failures_total", kind="stage", status="interrupted"), before + 1)

    def test_plain_lines_carry_the_fields(self):
        from dclab_rnd import observe

        record = logging.LogRecord("dclab", logging.INFO, __file__, 1, "request", None, None)
        record.fields = {"route": "/api/projects", "status": 200, "user": None}
        line = observe.PlainFormatter().format(record)
        self.assertTrue(line.endswith("request route=/api/projects status=200"), line)


if __name__ == "__main__":
    unittest.main()
