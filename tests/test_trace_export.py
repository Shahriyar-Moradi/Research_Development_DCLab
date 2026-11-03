"""DCLab Watch B4 (2026-11-03): DCLab's spans sent as OpenTelemetry traces with their own ids, nothing sent by default, no
table cell or text ever, no sealed span, and a target that is down never raising into DCLab."""

import http.server
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import tracing  # noqa: E402

TRACE, SPAN, PARENT = "0af7651916cd43dd8448eb211c80319c", "b7ad6b7169203331", "00f067aa0ba902b7"


def reset():
    tracing._queue.clear()
    for k in tracing._counts:
        tracing._counts[k] = 0


class Sink(http.server.BaseHTTPRequestHandler):
    bodies: list = []
    headers_seen: list = []

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Sink.bodies.append((self.path, body))
        Sink.headers_seen.append(dict(self.headers))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class TracingTests(unittest.TestCase):
    def setUp(self):
        reset()

    def test_nothing_is_sent_by_default(self):
        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": ""}):
            self.assertFalse(tracing.record(TRACE, SPAN, None, "model_request", 1, 2, {"dclab.purpose": "intent"}))
        self.assertEqual(tracing.stats()["queued"], 0)

    def test_only_allowed_short_attributes_leave(self):
        got = tracing.clean({"dclab.purpose": "intent", "gen_ai.request.model": "claude-sonnet-5-5", "gen_ai.usage.input_tokens": 1200,
                             "dclab.cost_eur": 0.004, "dclab.ok": True,
                             "prompt": "the customer churned", "dclab.cell": "row 7: 1,655 customers\nchurned", "dclab.note": "x" * 500,
                             "gen_ai.prompt": "secret words", "dclab.nan": float("nan")})
        self.assertEqual(set(got), {"dclab.purpose", "gen_ai.request.model", "gen_ai.usage.input_tokens", "dclab.cost_eur", "dclab.ok"})

    def test_a_sealed_span_is_never_sent(self):
        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": "mlflow"}):
            self.assertFalse(tracing.record(TRACE, SPAN, None, "journey", 1, 2, {"dclab.split": "sealed_v6"}))
        self.assertEqual(tracing.stats()["refused"], 1)

    def test_the_encoded_span_keeps_dclabs_ids(self):
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

        body = tracing.encode([{"trace_id": TRACE, "span_id": SPAN, "parent": PARENT, "name": "agent_step", "start": 10, "end": 20,
                                "attributes": {"dclab.tool": "describe_data", "gen_ai.usage.output_tokens": 7}, "error": False}])
        req = ExportTraceServiceRequest.FromString(body)
        span = req.resource_spans[0].scope_spans[0].spans[0]
        self.assertEqual((span.trace_id.hex(), span.span_id.hex(), span.parent_span_id.hex(), span.name), (TRACE, SPAN, PARENT, "agent_step"))
        self.assertEqual({a.key for a in span.attributes}, {"dclab.tool", "gen_ai.usage.output_tokens"})

    def test_both_targets_receive_the_batch_and_langfuse_gets_its_keys(self):
        Sink.bodies, Sink.headers_seen = [], []
        server = http.server.HTTPServer(("127.0.0.1", 0), Sink)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        env = {"DCLAB_TRACE_EXPORT": "mlflow,langfuse", "DCLAB_MLFLOW_OTLP": base + "/v1/traces",
               "DCLAB_LANGFUSE_OTLP": base + "/api/public/otel/v1/traces", "LANGFUSE_PUBLIC_KEY": "pk-lf-test", "LANGFUSE_SECRET_KEY": "sk-lf-test"}
        try:
            with mock.patch.dict(os.environ, env), mock.patch.object(tracing, "_start"):
                self.assertTrue(tracing.record(TRACE, SPAN, None, "request", 1, 2, {"dclab.route": "/api/drafts"}))
                self.assertEqual(tracing.flush(), 1)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(sorted(p for p, _ in Sink.bodies), ["/api/public/otel/v1/traces", "/v1/traces"])
        lf = next(h for h in Sink.headers_seen if "Authorization" in h)
        self.assertTrue(lf["Authorization"].startswith("Basic "))
        self.assertEqual(tracing.stats()["sent"], 2)

    def test_a_target_that_is_down_is_counted_never_raised(self):
        env = {"DCLAB_TRACE_EXPORT": "mlflow", "DCLAB_MLFLOW_OTLP": "http://127.0.0.1:9/v1/traces"}
        with mock.patch.dict(os.environ, env), mock.patch.object(tracing, "_start"), mock.patch.object(tracing, "TIMEOUT", 0.5):
            tracing.record(TRACE, SPAN, None, "request", 1, 2, {})
            tracing.flush()
        self.assertEqual(tracing.stats()["failed"], 1)

    def test_a_full_queue_drops_and_counts(self):
        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": "mlflow"}), mock.patch.object(tracing, "_start"), \
                mock.patch.object(tracing, "QUEUE", 3):
            results = [tracing.record(TRACE, SPAN, None, "x", 1, 2, {}) for _ in range(5)]
        self.assertEqual(results, [True, True, True, False, False])
        self.assertEqual(tracing.stats()["dropped"], 2)


if __name__ == "__main__":
    unittest.main()


class SpanWiringTests(unittest.TestCase):
    """The spans DCLab already opened are what is sent: named, with their numbers, nested by their own ids."""

    def setUp(self):
        reset()

    def test_a_model_request_inside_an_agent_run_is_two_nested_spans_with_its_numbers(self):
        from dclab_rnd import observe

        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": "mlflow"}), mock.patch.object(tracing, "_start"):
            with observe.span(name="agent_run home", **{"dclab.agent": "home", "session.id": "draft-1"}):
                with observe.span(name="model_request"):
                    observe.annotate(**{"gen_ai.request.model": "claude-sonnet-5-5", "gen_ai.usage.input_tokens": 900,
                                        "dclab.prompt_text": "the full prompt\nwith a table cell 1,655"})
        spans = list(tracing._queue)
        self.assertEqual([s["name"] for s in spans], ["model_request", "agent_run home"])  # the inner one ends first
        inner, outer = spans
        self.assertEqual((inner["trace_id"], inner["parent"]), (outer["trace_id"], outer["span_id"]))
        self.assertEqual(inner["attributes"], {"gen_ai.request.model": "claude-sonnet-5-5", "gen_ai.usage.input_tokens": 900})
        self.assertEqual(outer["attributes"]["session.id"], "draft-1")

    def test_with_export_off_a_span_sends_nothing_and_an_error_still_raises(self):
        from dclab_rnd import observe

        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": ""}):
            with self.assertRaises(ValueError):
                with observe.span(name="x"):
                    raise ValueError("the work's own error")
        self.assertEqual(len(tracing._queue), 0)

    def test_an_api_request_is_a_root_span(self):
        import tempfile

        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": "mlflow"}), mock.patch.object(tracing, "_start"):
            client.get("/api/config")
        names = [s["name"] for s in tracing._queue]
        self.assertIn("GET /api/config", names)
        root = next(s for s in tracing._queue if s["name"] == "GET /api/config")
        self.assertEqual(root["attributes"]["dclab.status"], 200)
