"""Tests for cancellable planning host in worker threads and background jobs (TP-09 / Step 2).

Verifies that model planning (decomposition and task planning) can run safely in worker threads
when hosted by a cancellable host, that cancellation cleanly stops planning without leaving
untracked requests or duplicate effects, that unhosted worker threads still safely refuse with
deadline_unavailable, and that jobs.worker Stopped exceptions propagate without being swallowed.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.tasks.decomposer import decompose, DecompositionBudget
from dclab_rnd.tasks.planner import plan_task, PlanningBudget
from dclab_rnd.tasks.host import (
    PlanningHost, PlanningCancelled, planning_host, current_host, is_cancellable
)
from tests.task_planning_cases import case
from tests.test_models_gateway import CLEAN
from tests.test_task_decomposer import boundaries


class TaskHostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {**CLEAN, "DCLAB_NO_LIVE_MODELS": "1", "DCLAB_THINKING": "",
                                "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "claude-sonnet-5-5",
                                "OPENAI_BASE_URL": "https://api.anthropic.com/v1"}).start()
        self.project, self.source, self.workflow, self.caps, self.graph, self.state = case()
        self.baseline = self.graph
        self.answer = boundaries(self.baseline)
        self.sent, self.reply = [], None
        outer = self

        class Transport:
            thinking = None

            def complete(self, messages, tools=None, max_tokens=None, **options):
                outer.sent.append(copy.deepcopy({"messages": messages, "tools": tools, "options": options}))
                if isinstance(outer.reply, Exception):
                    raise outer.reply
                # If decomposer:
                if "decomposer" in messages[0]["content"]:
                    return outer.reply or {
                        "content": json.dumps(outer.answer), "tool_calls": [], "finish_reason": "stop",
                        "thinking": [{"text": "Keep the existing final approval and group the preparation tasks."}],
                        "usage": {"input_tokens": 10, "output_tokens": 30}
                    }
                # If planner:
                baseline_spec = json.loads(messages[-1]["content"])["baseline"]
                proposal = {"version": "tp03.v1", "baseline_fingerprint": baseline_spec["fingerprint"],
                            **{k: baseline_spec[k] for k in ("actions", "budget", "policy")}}
                return outer.reply or {
                    "content": json.dumps(proposal), "tool_calls": [], "finish_reason": "stop",
                    "thinking": [{"text": "Keep one guarded stage and verify its declared evidence."}],
                    "usage": {"input_tokens": 10, "output_tokens": 30}
                }

        self.transport = Transport()
        self.gateway = Gateway(FileUsage(self.home / "usage.jsonl"), transport=lambda tier, purpose: self.transport,
                               sleep=lambda _: self.fail("must not retry"))
        patch("dclab_rnd.models.gateway._exchange_home", return_value=self.home).start()
        patch("dclab_rnd.experience.model_text_allowed", return_value=True).start()

    def run_decomposer(self, **changes):
        args = dict(capabilities=self.caps, deterministic_graph=self.baseline, confirmed_goal=self.project["goal"],
                    enabled=True, gateway=self.gateway)
        args.update(changes)
        return decompose(self.workflow, self.source, self.project, **args)

    def run_planner(self, task_id="data", **kwargs):
        remaining = {"max_tool_calls": 2, "max_seconds": 180, "max_cost_eur": 0.5}
        return plan_task(self.workflow, self.source, self.project, graph=self.graph, task_id=task_id,
                         observed_state=self.state, capabilities=self.caps, remaining_budget=remaining,
                         enabled=True, gateway=self.gateway, **kwargs)

    # -------------------------------------------------------------------------------- Host primitives
    def test_planning_host_context_manages_current_host_and_cancellability(self):
        self.assertIsNone(current_host())
        with planning_host(name="test_worker") as host:
            self.assertIs(current_host(), host)
            self.assertTrue(is_cancellable())
            self.assertFalse(host.is_cancelled())
            host.check_cancellation()  # does not raise

            host.cancel()
            self.assertTrue(host.is_cancelled())
            with self.assertRaises(PlanningCancelled):
                host.check_cancellation()
        self.assertIsNone(current_host())

    def test_planning_host_invokes_custom_cancel_callbacks(self):
        called = {"check": 0, "cancel": 0}
        def on_check(): called["check"] += 1
        def on_cancel(): called["cancel"] += 1

        with planning_host(check_cancel=on_check, on_cancel=on_cancel) as host:
            host.check_cancellation()
            self.assertEqual(called["check"], 1)
            self.assertEqual(called["cancel"], 0)
            host.cancel()
            self.assertEqual(called["cancel"], 1)

    # -------------------------------------------------------------------------------- Decomposer in worker thread
    def test_unhosted_worker_thread_refuses_with_deadline_unavailable(self):
        results = []
        thread = threading.Thread(target=lambda: results.append(self.run_decomposer()))
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0][0], self.baseline)
        self.assertEqual(results[0][1]["refusals"], ["deadline_unavailable"])
        self.assertFalse(self.sent)

    def test_hosted_worker_thread_allows_model_decomposition(self):
        results = []
        def worker():
            with planning_host(name="worker_thread"):
                results.append(self.run_decomposer())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        graph, report = results[0]
        self.assertEqual(report["status"], "proposed")
        self.assertEqual(report["model_calls"], 1)
        self.assertEqual(graph.provenance.method, "model")
        self.assertEqual(len(self.sent), 1)

    def test_hosted_worker_thread_cancelled_before_call_falls_back_safely(self):
        results = []
        def worker():
            with planning_host(name="worker_thread") as host:
                host.cancel()
                results.append(self.run_decomposer())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        graph, report = results[0]
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["status"], "fallback")
        self.assertEqual(report["refusals"], ["planning_cancelled"])
        self.assertFalse(self.sent)

    # -------------------------------------------------------------------------------- Planner in worker thread
    def test_hosted_worker_thread_allows_model_task_planning(self):
        results = []
        def worker():
            with planning_host(name="worker_thread"):
                results.append(self.run_planner())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        plan, report = results[0]
        self.assertEqual(report["status"], "proposed")
        self.assertEqual(report["model_calls"], 1)
        self.assertEqual(plan.provenance.method, "model")
        self.assertEqual(len(self.sent), 1)

    def test_hosted_worker_thread_cancelled_planner_falls_back_safely(self):
        results = []
        def worker():
            with planning_host(name="worker_thread") as host:
                host.cancel()
                results.append(self.run_planner())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        plan, report = results[0]
        self.assertEqual(report["status"], "fallback")
        self.assertIn(report["refusals"][0], ("proposal_refused", "planning_cancelled"))
        self.assertFalse(self.sent)

    # -------------------------------------------------------------------------------- Jobs.worker integration
    def test_job_worker_control_serves_as_cancellable_host_and_raises_stopped(self):
        from dclab_rnd.jobs.worker import Control, Stopped, _LOCAL

        class FakeWorker:
            id = "worker-1"
            stopping = False

        class FakeStore:
            def __init__(self):
                self.cancelled = False

            def get(self, job_id):
                return {"id": job_id, "status": "running", "worker": "worker-1", "cancel_requested": self.cancelled}

        store = FakeStore()
        control = Control(store, {"id": "job-101"}, FakeWorker())
        _LOCAL.control = control
        try:
            host = current_host()
            self.assertIsNotNone(host)
            self.assertTrue(is_cancellable())

            # While not cancelled, check passes
            host.check_cancellation()

            # When cancelled in job store, Stopped is raised
            store.cancelled = True
            with self.assertRaises(Stopped) as ctx:
                host.check_cancellation()
            self.assertEqual(ctx.exception.reason, "stopped")

            # Stopped is not swallowed by decompose or plan_task
            with self.assertRaises(Stopped):
                self.run_decomposer()
            with self.assertRaises(Stopped):
                self.run_planner()
        finally:
            _LOCAL.control = None

    def test_untracked_requests_prevented_on_failure(self):
        self.reply = RuntimeError("provider dropped connection")
        with planning_host():
            _, report = self.run_decomposer()
        self.assertEqual(report["status"], "fallback")
        self.assertEqual(report["refusals"], ["model_request_failed"])
        # Request ID is tracked in report even on failure
        self.assertIsNotNone(report["model_request"])


if __name__ == "__main__":
    unittest.main()
