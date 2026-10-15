"""23.11: one request, one trace. The edge accepts a W3C traceparent (or makes a trace from the request id) and answers with
one; every event, log line and model request of the request carries the trace and a span, a job it queued runs in the same
trace in a span of its own, and the record can be read by trace."""

import io
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import tracing
    from dclab_rnd import experience, observe
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT = "00f067aa0ba902b7"


@unittest.skipIf(MISSING, MISSING)
class CoreTest(unittest.TestCase):
    def test_a_traceparent_is_read_and_written_as_the_standard_says(self):
        got = tracing.parse_traceparent(f"00-{TRACE}-{PARENT}-01")
        self.assertEqual((got.trace_id, got.parent_span, got.sampled), (TRACE, PARENT, True))
        self.assertFalse(tracing.parse_traceparent(f"00-{TRACE}-{PARENT}-00").sampled)
        for bad in (None, "", "garbage", f"ff-{TRACE}-{PARENT}-01", f"00-{'0' * 32}-{PARENT}-01", f"00-{TRACE}-{'0' * 16}-01", f"00-{TRACE.upper()}-{PARENT}-01",
                    f"00-{TRACE[:30]}-{PARENT}-01", f"00-{TRACE}-{PARENT}-01-extra"):
            with self.subTest(bad=bad):
                self.assertIsNone(tracing.parse_traceparent(bad))
        self.assertEqual(tracing.parse_traceparent(f"01-{TRACE}-{PARENT}-01-future").trace_id, TRACE)  # a later version: its first four fields
        self.assertEqual(tracing.format_traceparent(TRACE, PARENT), f"00-{TRACE}-{PARENT}-01")
        with self.assertRaises(ValueError):
            tracing.format_traceparent("short", PARENT)

    def test_a_request_id_gives_one_trace_and_new_ids_are_well_formed(self):
        self.assertEqual(tracing.trace_id_for("abc12345"), tracing.trace_id_for("abc12345"))
        self.assertNotEqual(tracing.trace_id_for("abc12345"), tracing.trace_id_for("abc12346"))
        self.assertEqual(tracing.trace_id_for(TRACE), TRACE)  # already a trace id
        for _ in range(20):
            self.assertRegex(tracing.new_trace_id(), r"^[0-9a-f]{32}$")
            self.assertRegex(tracing.new_span_id(), r"^[0-9a-f]{16}$")


@unittest.skipIf(MISSING, MISSING)
class EdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.s = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_a_request_with_a_traceparent_continues_it_and_answers_with_the_same_trace_and_a_span_of_its_own(self):
        r = self.client.post("/api/drafts", json={"problem": "Which customers will leave us next month?"}, headers={**self.h, "traceparent": f"00-{TRACE}-{PARENT}-01"})
        self.assertEqual(r.status_code, 201, r.text)
        answered = tracing.parse_traceparent(r.headers["traceparent"])
        self.assertEqual(answered.trace_id, TRACE)
        self.assertNotEqual(answered.parent_span, PARENT)  # the response names this server's span, not the caller's
        self.assertEqual(r.headers["X-Request-ID"], TRACE[:16])  # no request id offered: the trace names it
        events = self.s.experience.read(trace=TRACE)
        self.assertTrue(events)
        self.assertEqual({e.ids.trace for e in events}, {TRACE})
        self.assertEqual({e.ids.span for e in events}, {answered.parent_span})
        self.assertEqual({e.ids.parent_span for e in events}, {PARENT})  # the caller's span is the request span's parent
        self.assertEqual({e.ids.request for e in events}, {TRACE[:16]})

    def test_a_request_without_one_gets_a_trace_derived_from_its_request_id(self):
        r = self.client.get("/api/projects", headers={**self.h, "X-Request-ID": "svc.7f3a9c01"})
        got = tracing.parse_traceparent(r.headers["traceparent"])
        self.assertEqual((got.trace_id, r.headers["X-Request-ID"]), (tracing.trace_id_for("svc.7f3a9c01"), "svc.7f3a9c01"))
        bare = self.client.get("/api/projects", headers=self.h)
        self.assertEqual(tracing.parse_traceparent(bare.headers["traceparent"]).trace_id, tracing.trace_id_for(bare.headers["X-Request-ID"]))
        bad = self.client.get("/api/projects", headers={**self.h, "traceparent": "garbage"})
        self.assertIsNotNone(tracing.parse_traceparent(bad.headers["traceparent"]))  # an invalid one is ignored, a fresh trace answers

    def test_a_job_the_request_queued_runs_in_its_trace_in_a_span_of_its_own(self):
        pid = self.client.post("/api/projects", json={"name": "Job", "goal": "g"}, headers=self.h).json()["id"]
        r = self.client.post(f"/api/projects/{pid}/pipeline/rerun?step=stage:models", headers={**self.h, "traceparent": f"00-{TRACE.replace('4', '5')}-{PARENT}-01"})
        self.assertEqual(r.status_code, 200, r.text)
        trace = TRACE.replace("4", "5")
        request_span = tracing.parse_traceparent(r.headers["traceparent"]).parent_span
        events = self.s.experience.read(trace=trace)
        job = [e for e in events if e.kind.startswith("job.")]
        self.assertTrue(job, "the job left no event in the request's trace")
        self.assertEqual({e.ids.trace for e in job}, {trace})
        self.assertTrue(all(e.ids.span != request_span for e in job))  # its own span
        self.assertEqual({e.ids.parent_span for e in job}, {request_span})  # inside the request's
        self.assertTrue(all(e.ids.job for e in job))

    def test_a_model_request_is_a_span_inside_the_one_running(self):
        from dclab_rnd.experience import hooks

        other = "a" * 32
        with observe.span(trace_id=other) as (trace, outer, _):
            with observe.span() as (_, inner, parent):
                hooks.from_model_request({"id": experience.new_model_request(), "purpose": "home_agent", "tier": "t", "model": "m", "outcome": "ok"})
        self.assertEqual((trace, parent), (other, outer))
        self.assertNotEqual(inner, outer)
        sent = [e for e in self.s.experience.read(trace=other) if e.kind == "model.request"]
        self.assertEqual([(e.ids.span, e.ids.parent_span) for e in sent], [(inner, outer)])  # the event carries the request's own span

    def test_the_log_lines_carry_the_ids_a_collector_joins_by(self):
        record = logging.LogRecord("dclab", logging.INFO, __file__, 1, "request", (), None)
        record.fields = {"status": 200}
        with observe.span(trace_id=TRACE, parent=PARENT) as (_, span, parent):
            line = json.loads(observe.JsonFormatter().format(record))
            plain = observe.PlainFormatter().format(record)
        self.assertEqual((line["trace_id"], line["span_id"], line["parent_span_id"]), (TRACE, span, PARENT))
        self.assertIn(f"trace_id={TRACE}", plain)
        outside = json.loads(observe.JsonFormatter().format(record))
        self.assertNotIn("trace_id", outside)

    def test_a_requests_log_line_and_its_events_carry_the_same_trace(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(observe.JsonFormatter())
        observe.log.addHandler(handler)
        previous = observe.log.level
        observe.log.setLevel(logging.INFO)
        self.addCleanup(lambda: (observe.log.removeHandler(handler), observe.log.setLevel(previous)))
        trace = "c" * 32
        with mock.patch.dict(os.environ, {"DCLAB_LOG_FORMAT": "json"}):
            r = self.client.post("/api/drafts", json={"problem": "Which customers will leave us next month?"}, headers={**self.h, "traceparent": f"00-{trace}-{PARENT}-01"})
        self.assertEqual(r.status_code, 201, r.text)
        lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.startswith("{")]
        requests = [line for line in lines if line.get("message") == "request" and line.get("trace_id") == trace]
        self.assertTrue(requests, f"no request line carried the trace: {lines[:3]}")
        events = self.s.experience.read(trace=trace)
        self.assertTrue(events)
        self.assertEqual({requests[0]["span_id"]}, {e.ids.span for e in events})  # the line and the events are one span of one trace

    def test_the_record_is_read_by_trace_on_files_and_the_route_and_the_command_line_ask_for_it(self):
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w1")
        with observe.span(trace_id=TRACE):
            experience.record("stage.finished", "system", {"stage": "data"}, log=log, visit="v1")
        with observe.span():
            experience.record("stage.finished", "system", {"stage": "final"}, log=log, visit="v1")
        self.assertEqual([e.payload["stage"] for e in log.read(trace=TRACE)], ["data"])
        found = self.client.get("/api/experience/search", params={"trace": TRACE})
        self.assertEqual(found.status_code, 200)
        self.assertEqual(self.client.get("/api/experience/search", params={"trace": "short"}).status_code, 422)
        self.assertEqual(self.client.get("/api/experience/search", params={"trace": TRACE.upper()}).status_code, 422)


class EveryTransportCall:
    """A transport with the three calls the gateway makes, answering at once."""

    def __init__(self, tier, purpose):
        self.tier, self.purpose = tier, purpose

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        return {"content": "ok", "tool_calls": [], "usage": {"input_tokens": 3, "output_tokens": 2}}

    def stream(self, messages, tools=None, max_tokens=1800, on_text=None):
        on_text and on_text("ok")
        return self.complete(messages)

    def embed(self, texts):
        return {"vectors": [[0.0] for _ in texts], "usage": {"input_tokens": 1, "output_tokens": 0}}


@unittest.skipIf(MISSING, MISSING)
class GatewayTest(unittest.TestCase):
    """A model request is a span of the trace it was made in, whichever call made it: a completion, a stream or an embedding."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model", "DCLAB_LLM_PROVIDER": "", "ANTHROPIC_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_complete_stream_and_embed_are_each_a_span_of_the_running_trace(self):
        from dclab_rnd.models import FileUsage, Gateway

        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl"), transport=lambda tier, purpose: EveryTransportCall(tier, purpose), sleep=lambda s: None)
        seen = []
        with mock.patch("dclab_rnd.experience.hooks.from_model_request", side_effect=lambda *a, **k: seen.append(observe.TRACE.get())):
            with observe.span(trace_id=TRACE) as (_, outer, _):
                client = gw.client("home_agent")
                client.complete([{"role": "user", "content": "x"}])
                client.stream([{"role": "user", "content": "x"}], on_text=lambda delta: None)
                client.stream([{"role": "user", "content": "y"}], on_text=lambda delta: None)
                gw.client("embedding").embed(["a", "b"])
        self.assertEqual(len(seen), 4, seen)
        self.assertEqual({trace for trace, _, _ in seen}, {TRACE})
        self.assertEqual({parent for _, _, parent in seen}, {outer})  # inside the span that was running
        self.assertEqual(len({span for _, span, _ in seen}), 4)  # four requests, four spans: none borrows the turn's own
        self.assertNotIn(outer, {span for _, span, _ in seen})


@unittest.skipIf(MISSING, MISSING)
class RunOutsideARequestTest(unittest.TestCase):
    """A script, the campaign or an evaluation runs agents with no request around them: each run is one trace, and what it asks joins it."""

    def setUp(self):
        self.log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w1")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def test_a_session_outside_a_request_is_one_trace_and_two_sessions_are_two(self):
        from dclab_rnd.agents import harness

        traces = []
        for _ in range(2):
            with harness.session("team") as current:
                current.step("ask", {"q": 1}, {"ok": True})
                current.step("answer", {}, {"ok": True})
            traces.append(current.run_id)
        events = [e for e in self.log.read() if e.ids.run in traces]
        self.assertTrue(events)
        by_run = {run: {e.ids.trace for e in events if e.ids.run == run} for run in traces}
        for run, found in by_run.items():
            self.assertEqual(len(found), 1, f"the steps of {run} are in {len(found)} traces")
            self.assertIsNotNone(next(iter(found)))
        self.assertNotEqual(*[next(iter(f)) for f in by_run.values()])  # two runs, two traces

    def test_a_model_loop_outside_a_request_is_one_trace_for_its_steps_and_its_end(self):
        from dclab_rnd.agents import Policy, Registry, Tool, harness, run
        from tests.test_agents_runtime import Scripted, add_numbers, obj, reply

        registry = Registry()
        registry.register(Tool("add", "Add.", obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]), add_numbers))
        found = []
        for _ in range(2):
            tracer = harness.tracer("team")
            run(Policy("team", "You add numbers.", ("add",), max_steps=4), Scripted(reply(calls=[("add", {"a": 1, "b": 2})]), reply("done")), registry,
                [{"role": "system", "content": "s"}], trace=tracer)
            found.append({e.ids.trace for e in self.log.read() if e.ids.run == tracer.run_id})
        self.assertTrue(all(len(traces) == 1 and None not in traces for traces in found), found)
        self.assertNotEqual(found[0], found[1])

    def test_a_session_inside_a_request_stays_in_its_trace_as_a_span_of_its_own(self):
        from dclab_rnd.agents import harness

        with observe.span(trace_id=TRACE) as (_, outer, _):
            with harness.session("team") as current:
                current.step("ask", {"q": 1}, {"ok": True})
        events = [e for e in self.log.read() if e.ids.run == current.run_id]
        self.assertTrue(events)
        self.assertEqual({e.ids.trace for e in events}, {TRACE})
        self.assertNotIn(outer, {e.ids.span for e in events if e.kind == "agent.step"})


@unittest.skipIf(MISSING, MISSING)
class AdapterTest(unittest.TestCase):
    def test_events_with_a_request_get_the_trace_that_request_would_have_and_the_others_stay(self):
        from dclab_core import experience as x
        from dclab_core.adapters import main_product_experience as mp

        a = x.make("stage.finished", "system", {"workspace": "w"}, {"stage": "data"}, id="ev000001", at="2026-10-16T10:00:00+00:00")
        b = x.make("stage.finished", "system", {"workspace": "w"}, {"stage": "final"}, id="ev000002", at="2026-10-16T10:00:01+00:00")
        out = mp.with_trace([a, b], lambda e: "svc.7f3a9c01" if e.id == "ev000001" else None)
        self.assertEqual((out[0].ids.request, out[0].ids.trace), ("svc.7f3a9c01", tracing.trace_id_for("svc.7f3a9c01")))
        self.assertEqual(out[1], b)
        again = mp.with_trace([a], lambda e: "svc.7f3a9c01")
        self.assertEqual(again[0].ids.trace, out[0].ids.trace)  # one request id, one trace, wherever it is read


if __name__ == "__main__":
    unittest.main()
