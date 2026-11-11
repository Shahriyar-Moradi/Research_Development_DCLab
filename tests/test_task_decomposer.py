"""TP-02: scripted task-boundary proposals, never task execution or live model calls."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dclab_core.contracts import SolutionPlan
from dclab_core.solution_workflow import SolutionWorkflow
from dclab_core.task_validation import validate_graph
from dclab_rnd.agents import build_registry
from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.tasks.adapter import adapt_workflow, capability_snapshot
from dclab_rnd.tasks.decomposer import DecompositionBudget, decompose
from tests.test_models_gateway import CLEAN, transient

FIXTURE = Path(__file__).parent / "fixtures/tasks/telco_metadata.json"


def boundaries(graph, *, single=False):
    leaves = [n for n in graph.nodes if n.kind != "group"]
    chunks = [[n.id] for n in leaves] if single else [["data", "leakage", "features", "models"], ["holdout", "final", "report"]]
    return {"version": "tp02.v1", "baseline_fingerprint": graph.fingerprint,
            "groups": [{"id": f"group_{i}", "parent_id": None, "members": members} for i, members in enumerate(chunks)],
            "dependencies": [{"task_id": n.id, "requires": list(n.dependencies)} for n in leaves]}


class TaskDecomposerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "claude-sonnet-5-5",
                               "DCLAB_THINKING": "", "DCLAB_NO_LIVE_MODELS": "1"}).start()
        self.fixture = json.loads(FIXTURE.read_text())
        self.project = self.fixture["project"]
        self.plan = SolutionPlan.read(self.fixture["source_plan"])
        self.workflow = SolutionWorkflow.read(self.fixture["workflow"])
        self.registry = build_registry(("project",))
        self.caps = capability_snapshot(self.registry, allowed_tools={"run_stage", "export_notebook"})
        self.baseline, self.mapping = adapt_workflow(self.workflow, self.plan, self.project, capabilities=self.caps)
        self.sent = []
        self.answer = boundaries(self.baseline)
        self.reply = None
        outer = self

        class Transport:
            thinking = None

            def complete(self, messages, tools=None, max_tokens=None, **options):
                outer.sent.append(copy.deepcopy({"messages": messages, "tools": tools, "max_tokens": max_tokens, "options": options}))
                if isinstance(outer.reply, Exception):
                    raise outer.reply
                return outer.reply or {"content": json.dumps(outer.answer), "tool_calls": [], "finish_reason": "stop",
                                       "thinking": [{"text": "Keep the existing final approval and group the preparation tasks."}],
                                       "usage": {"input_tokens": 10, "output_tokens": 30}}

        self.transport = Transport()
        self.gateway = Gateway(FileUsage(self.home / "usage.jsonl"), transport=lambda tier, purpose: self.transport,
                               sleep=lambda _: self.fail("TP-02 must not retry"))
        patch("dclab_rnd.models.gateway._exchange_home", return_value=self.home).start()
        patch("dclab_rnd.experience.model_text_allowed", return_value=True).start()

    def run_decomposer(self, **changes):
        args = dict(capabilities=self.caps, deterministic_graph=self.baseline, confirmed_goal=self.project["goal"],
                    enabled=True, gateway=self.gateway)
        args.update(changes)
        return decompose(self.workflow, self.plan, self.project, **args)

    def test_model_groups_atomic_tasks_and_preserves_all_authority(self):
        graph, report = self.run_decomposer()
        self.assertEqual(report["status"], "proposed")
        self.assertFalse(report["execution_authorized"])
        self.assertEqual(report["model_calls"], 1)
        self.assertEqual(graph.revision, self.baseline.revision + 1)
        self.assertEqual(graph.source_refs, self.baseline.source_refs)
        self.assertEqual(graph.provenance.method, "model")
        for old in self.baseline.nodes:
            if old.kind == "group":
                continue
            new = next(n for n in graph.nodes if n.id == old.id)
            for key in type(old).model_fields:
                if key not in ("parent_id", "dependencies"):
                    self.assertEqual(getattr(old, key), getattr(new, key), key)
        self.assertEqual(validate_graph(graph, self.workflow, known_capabilities=self.caps,
                                       external_artifacts=self.mapping["external_artifacts"]), [])
        self.assertIsNone(self.sent[0]["tools"])

    def test_one_atomic_task_per_group(self):
        self.answer = boundaries(self.baseline, single=True)
        graph, report = self.run_decomposer()
        self.assertEqual(report["status"], "proposed")
        self.assertEqual(sum(n.kind != "group" for n in graph.nodes), 7)

    def test_current_baseline_revision_two_produces_revision_three(self):
        self.baseline, _ = adapt_workflow(self.workflow, self.plan, self.project, capabilities=self.caps, revision=2)
        self.answer = boundaries(self.baseline)
        graph, report = self.run_decomposer()
        self.assertEqual(report["status"], "proposed")
        self.assertEqual(graph.revision, 3)

    def test_invalid_model_response_falls_back_exactly(self):
        self.answer["groups"][0]["members"].remove("leakage")
        graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["status"], "fallback")
        self.assertTrue(report["refusals"])

    def test_disabled_mode_does_not_access_the_gateway(self):
        gateway = Mock(side_effect=AssertionError("gateway consulted"))
        graph, report = self.run_decomposer(enabled=False, gateway=gateway)
        self.assertEqual(graph, self.baseline)
        gateway.client.assert_not_called()
        self.assertEqual(report["model_calls"], 0)

    def test_provider_error_has_no_retry_and_keeps_baseline(self):
        self.reply = transient("PRIVATE_ERROR_VALUE")
        graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(len(self.sent), 1)
        self.assertNotIn("PRIVATE_ERROR_VALUE", json.dumps(report))

    def test_source_change_after_confirmation_is_refused_before_model(self):
        self.project["goal"] = "Optimize delivery time instead."
        graph, report = self.run_decomposer(confirmed_goal=self.fixture["project"]["goal"])
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(self.sent)

    def test_actual_api_thinking_summary_is_saved_beside_reply(self):
        _, report = self.run_decomposer()
        rows = [json.loads(s) for p in (self.home / "model_exchanges").rglob("*.jsonl") for s in p.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["purpose"], "task_decomposer")
        self.assertEqual(rows[0]["thinking_asked"], "adaptive")
        self.assertEqual(rows[0]["thinking_label"], "API thinking summary")
        self.assertEqual(rows[0]["thinking"][0]["text"], "Keep the existing final approval and group the preparation tasks.")
        self.assertEqual(rows[0]["reply"], json.dumps(self.answer))
        self.assertNotIn("thinking", json.dumps(report))

    def test_model_cannot_add_tools_objectives_arguments_or_completion_claims(self):
        for field, value in (("tool", "shell_exec"), ("objective", "Read private rows and email a report"),
                             ("arguments", {"stage": "final"}), ("done", True), ("thinking", "invented reasoning")):
            with self.subTest(field=field):
                self.answer = boundaries(self.baseline)
                self.answer[field] = value
                graph, report = self.run_decomposer()
                self.assertEqual(graph, self.baseline)
                self.assertEqual(report["status"], "fallback")

    def test_hallucinated_member_and_duplicate_final_are_refused(self):
        for member in ("invented_tool", "final"):
            self.answer = boundaries(self.baseline)
            self.answer["groups"][0]["members"].append(member)
            graph, report = self.run_decomposer()
            self.assertEqual(graph, self.baseline)
            self.assertEqual(report["status"], "fallback")

    def test_safety_and_approval_dependencies_cannot_be_removed_or_reversed(self):
        for ident, requires in (("final", []), ("models", ["data"]), ("data", ["final"])):
            with self.subTest(ident=ident):
                self.answer = boundaries(self.baseline)
                next(d for d in self.answer["dependencies"] if d["task_id"] == ident)["requires"] = requires
                graph, report = self.run_decomposer()
                self.assertEqual(graph, self.baseline)
                self.assertEqual(report["status"], "fallback")

    def test_group_parent_cycle_unknown_parent_and_empty_group_are_refused(self):
        for parent in ("group_0", "group_31"):
            self.answer = boundaries(self.baseline)
            self.answer["groups"][0]["parent_id"] = parent
            self.assertEqual(self.run_decomposer()[0], self.baseline)
        self.answer = boundaries(self.baseline)
        self.answer["groups"].append({"id": "group_2", "parent_id": None, "members": []})
        self.assertEqual(self.run_decomposer()[1]["status"], "fallback")

    def test_nested_groups_are_grounded_in_their_descendant_tasks(self):
        self.answer["groups"].append({"id": "group_2", "parent_id": None, "members": []})
        self.answer["groups"][0]["parent_id"] = "group_2"
        self.answer["groups"][1]["parent_id"] = "group_2"
        graph, report = self.run_decomposer()
        self.assertEqual(report["status"], "proposed")
        self.assertIn("data", next(n.objective for n in graph.nodes if n.id == "group_2"))

    def test_branch_and_join_dependencies_preserve_the_atomic_sequence(self):
        next(d for d in self.answer["dependencies"] if d["task_id"] == "models")["requires"].append("data")
        graph, report = self.run_decomposer()
        self.assertEqual(report["status"], "proposed")
        self.assertEqual(set(next(n.dependencies for n in graph.nodes if n.id == "models")), {"data", "features"})

    def test_missing_capability_returns_the_exact_blocked_adapter_graph(self):
        self.caps.pop("run_stage")
        self.baseline, _ = adapt_workflow(self.workflow, self.plan, self.project, capabilities=self.caps)
        graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["blockers"])
        self.assertFalse(self.sent)

    def test_missing_dataset_returns_none_and_a_reason_without_a_call(self):
        self.project.pop("data")
        self.baseline = None
        graph, report = self.run_decomposer()
        self.assertIsNone(graph)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["blockers"])
        self.assertFalse(self.sent)

    def test_unconfirmed_goal_and_unresolved_questions_block_the_request(self):
        for args in ({"confirmed_goal": "Other goal"}, {"unresolved_questions": ["Which outcome is confirmed?"]}):
            self.assertEqual(self.run_decomposer(**args)[1]["status"], "blocked")
        self.assertFalse(self.sent)

    def test_changed_solution_data_capability_or_signoff_is_stale(self):
        for change in ("solution", "data", "caps", "signoff"):
            with self.subTest(change=change):
                project, caps = copy.deepcopy(self.project), copy.deepcopy(self.caps)
                if change == "solution":
                    project["solution"]["prediction_moment"] = "After cancellation"
                elif change == "data":
                    project["data"]["sha256"] = "e" * 64
                elif change == "caps":
                    caps["run_stage"]["allowed"] = False
                else:
                    project.pop("signoff")
                graph, report = decompose(self.workflow, self.plan, project, deterministic_graph=self.baseline,
                                          capabilities=caps, confirmed_goal=project["goal"], enabled=True, gateway=self.gateway)
                self.assertEqual(graph, self.baseline)
                self.assertEqual(report["status"], "blocked")
        self.assertFalse(self.sent)

    def test_unavailable_or_private_model_uses_the_exact_fallback(self):
        for mode in ("unavailable", "private"):
            with self.subTest(mode=mode):
                if mode == "unavailable":
                    client_patch = patch.object(self.gateway, "client", return_value=None)
                else:
                    client_patch = patch.object(self.gateway, "project_private", return_value=True)
                with client_patch:
                    graph, report = self.run_decomposer()
                self.assertEqual(graph, self.baseline)
                self.assertEqual(report["refusals"], ["model_unavailable"])
        self.assertFalse(self.sent)

    def test_cut_refused_tool_and_non_json_replies_fall_back(self):
        for reply in ({"content": "{}", "finish_reason": "length"},
                      {"content": "no", "finish_reason": "refusal"},
                      {"content": json.dumps(self.answer), "tool_calls": [{"name": "run_stage"}]},
                      {"content": "Before the JSON: " + json.dumps(self.answer)},
                      {"content": "{\"version\":\"tp02.v1\",\"version\":\"tp02.v1\"}"},
                      {"content": "[]"}):
            with self.subTest(reply=reply):
                self.reply = reply
                self.assertEqual(self.run_decomposer()[0], self.baseline)

    def test_stale_model_echo_is_refused(self):
        self.answer["baseline_fingerprint"] = "f" * 64
        graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["stale_model_context"])

    def test_input_reply_nodes_and_depth_are_bounded(self):
        for changes in ({"max_input_bytes": 1000}, {"max_reply_bytes": 100}, {"max_nodes": 8}, {"max_depth": 2}):
            with self.subTest(changes=changes):
                graph, report = self.run_decomposer(budget=DecompositionBudget(**changes))
                self.assertEqual(graph, self.baseline)
                self.assertEqual(report["status"], "fallback")

    def test_invalid_budget_settings_are_rejected_without_a_call(self):
        for values in ({"cap_eur": -1.0}, {"max_seconds": float("inf")}, {"model_calls": 2}, {"max_nodes": 500}):
            with self.assertRaises(ValueError):
                self.run_decomposer(budget=values)
        self.assertFalse(self.sent)

    def test_spend_unknown_price_and_effective_token_limits_fail_before_send(self):
        for values in ({"cap_eur": 0.00000001}, {"cap_eur": 0.0}, {"max_total_tokens": 100}):
            self.assertEqual(self.run_decomposer(budget=DecompositionBudget(**values))[0], self.baseline)
        with patch.dict(os.environ, {"OPENAI_MODEL": "unpriced-task-model"}):
            self.assertEqual(self.run_decomposer()[0], self.baseline)
        self.assertFalse(self.sent)

    def test_timeout_option_is_forwarded_and_a_late_reply_is_discarded(self):
        _, report = self.run_decomposer()
        self.assertGreater(self.sent[0]["options"]["timeout"], 0)
        self.assertLessEqual(self.sent[0]["options"]["timeout"], 30)
        clock = [0.0]
        original = self.transport.complete

        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 31.0
            return result

        with patch("dclab_rnd.tasks.decomposer.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(self.transport, "complete", side_effect=late):
            graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["planning_timeout"])

    def test_sleeping_transport_is_interrupted_at_the_planning_deadline(self):
        import time

        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        ended = []

        def sleeping(*args, **kwargs):
            try:
                time.sleep(2)
            finally:
                ended.append(True)

        started = time.monotonic()
        with patch.object(self.transport, "complete", side_effect=sleeping) as request:
            graph, report = self.run_decomposer(budget=DecompositionBudget(max_seconds=0.1))
        request.assert_called_once()
        self.assertEqual(ended, [True])
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["planning_timeout"])

    def test_deadline_during_reservation_handoff_releases_every_hold(self):
        import time

        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        self.sent.clear()
        self.gateway.project_cap = lambda project: 0.2
        original = self.gateway.reserve

        def delayed(*args, **kwargs):
            reserved = original(*args, **kwargs)
            time.sleep(2)
            return reserved

        with patch.object(self.gateway, "reserve", side_effect=delayed):
            graph, report = self.run_decomposer(budget=DecompositionBudget(max_seconds=0.1))
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["planning_timeout"])
        self.assertTrue(all(value == 0 for value in self.gateway._held.values()), self.gateway._held)
        self.assertFalse(self.sent)
        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")

    def test_deadline_during_a_slow_cap_callback_has_no_held_money(self):
        import time

        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        self.sent.clear()
        self.gateway.project_cap = Mock(side_effect=lambda project: time.sleep(2) or 0.2)
        started = time.monotonic()
        graph, report = self.run_decomposer(budget=DecompositionBudget(max_seconds=0.1))
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(report["refusals"], ["planning_timeout"])
        self.assertEqual(self.gateway._held, {})
        self.assertFalse(self.sent)
        self.gateway.project_cap.assert_called_once()

    def test_alarm_during_atomic_budget_update_is_cleaned_up_after_handoff(self):
        import signal

        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        self.sent.clear()
        self.gateway.project_cap = lambda project: 0.2

        class InterruptingHolds(dict):
            fired = False

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if value > 0 and not self.fired:
                    self.fired = True
                    signal.raise_signal(signal.SIGALRM)

        self.gateway._held = InterruptingHolds()
        graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["planning_timeout"])
        self.assertTrue(all(value == 0 for value in self.gateway._held.values()))
        self.assertFalse(self.sent)

    def test_real_deadline_with_another_thread_cannot_interrupt_budget_bookkeeping(self):
        import threading
        import time

        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        self.sent.clear()
        self.gateway.project_cap = lambda project: 0.2
        stopped = threading.Event()
        helper = threading.Thread(target=lambda: stopped.wait(2))

        class PausingHolds(dict):
            paused = False

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if value > 0 and not self.paused:
                    self.paused = True
                    time.sleep(0.3)

        self.gateway._held = PausingHolds()
        helper.start()
        try:
            graph, report = self.run_decomposer(budget=DecompositionBudget(max_seconds=0.1))
        finally:
            stopped.set()
            helper.join(1)
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["planning_timeout"])
        self.assertTrue(all(value == 0 for value in self.gateway._held.values()), self.gateway._held)
        self.assertFalse(self.sent)
        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")

    def test_worker_thread_refuses_before_model_when_it_cannot_enforce_a_deadline(self):
        import threading

        results = []
        thread = threading.Thread(target=lambda: results.append(self.run_decomposer()))
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0][0], self.baseline)
        self.assertEqual(results[0][1]["refusals"], ["deadline_unavailable"])
        self.assertFalse(self.sent)

    def test_caller_alarm_and_handler_are_preserved(self):
        import signal

        handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.setitimer(signal.ITIMER_REAL, 10)
            graph, report = self.run_decomposer()
            self.assertEqual(graph, self.baseline)
            self.assertEqual(report["refusals"], ["deadline_unavailable"])
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        self.assertFalse(self.sent)

    def test_sensitive_metadata_is_not_sent_and_policy_stays_in_the_system_message(self):
        marker = "Q7KxprivateCell"
        self.project["data"]["profile"] = {"preview": [[marker]], "examples": [marker], "unknown": object()}
        self.project["data"]["rows"] = [[marker]]
        self.project["secret_key"] = marker
        self.run_decomposer()
        sent = self.sent[0]
        self.assertNotIn(marker, json.dumps(sent))
        evidence = json.loads(sent["messages"][1]["content"])
        self.assertIn("data_card_metadata", evidence)
        self.assertEqual(evidence["authorized_tool_names"], ["export_notebook", "run_stage"])
        self.assertEqual(len(sent["messages"]), 2)

    def test_planning_cannot_invoke_tools_guards_stage_writes_or_holdout_reads(self):
        from contextlib import ExitStack

        before = copy.deepcopy(self.project), self.workflow.model_dump_json(), self.baseline.model_dump_json()
        spies = []
        with ExitStack() as stack:
            for name in ("call", "acall", "check"):
                spies.append(stack.enter_context(patch.object(self.registry, name, side_effect=AssertionError("tool invoked"))))
            for target in ("dclab_rnd.studio.engine.prepare", "dclab_rnd.studio.engine.stage_final",
                           "dclab_rnd.studio.data.load_table", "dclab_rnd.studio.data.data_path",
                           "dclab_core.programme.read_holdout", "dclab_core.compose.read_holdout",
                           "dclab_rnd.studio.store.ProjectStore.save", "dclab_rnd.storage.postgres.PgProjects.save"):
                spies.append(stack.enter_context(patch(target, side_effect=AssertionError("runtime invoked"))))
            graph, report = self.run_decomposer()
        self.assertEqual(report["status"], "proposed")
        for spy in spies:
            spy.assert_not_called()
        self.assertEqual((self.project, self.workflow.model_dump_json(), self.baseline.model_dump_json()), before)

    def test_refusal_is_a_harness_step_without_source_or_model_text(self):
        self.answer["objective"] = "PRIVATE_UNGROUNDED_OBJECTIVE"
        with patch("dclab_rnd.agents.harness.role_step") as step:
            graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertIn("invalid_proposal_shape", step.call_args.args[2])
        self.assertNotIn("PRIVATE_UNGROUNDED_OBJECTIVE", str(step.call_args))
        self.assertEqual(step.call_args.kwargs["model_request"], report["model_request"])

    def test_a_shadow_route_cannot_make_an_extra_request(self):
        with patch.object(self.gateway, "route", return_value=("standard", "cheap")), \
                patch.object(self.gateway, "_tier_ready", return_value=True), \
                patch("dclab_rnd.models.gateway.Bound._shadow", side_effect=AssertionError("shadow requested")) as shadow:
            self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        shadow.assert_not_called()
        self.assertEqual(len(self.sent), 1)

    def test_twin_gateway_cannot_bypass_strict_request_bounds(self):
        from dclab_rnd.models.client import THINKING_REFUSED
        from dclab_rnd.twin.claude import Transcript, TwinGateway

        calls = []
        answer = self.answer

        class UnboundedTransport:
            model, base_url, thinking = "unpriced-twin-model", "https://api.anthropic.com", "adaptive"

            def effective_max_tokens(self, max_tokens, options=None):
                return 20_000

            def complete(self, messages, tools=None, max_tokens=None, **options):
                calls.append(options)
                if len(calls) == 1:
                    raise RuntimeError("BadRequestError: " + THINKING_REFUSED)
                return {"content": json.dumps(answer), "tool_calls": [], "finish_reason": "stop", "usage": {}}

        gateway = TwinGateway(UnboundedTransport(), Transcript(self.home / "twin-calls.jsonl"))
        graph, report = self.run_decomposer(gateway=gateway)
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["refusals"], ["bounded_client_unavailable"])
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(calls, [])

    def test_anthropic_requests_summarized_thinking_timeout_and_no_compatibility_retry(self):
        from types import SimpleNamespace as NS
        from dclab_rnd.models.anthropic_client import AnthropicClient
        from tests.test_anthropic_transport import FakeMessages, reply

        fake = FakeMessages(reply(NS(type="thinking", thinking="Preserve the approved tasks.", signature="sig"),
                                  NS(type="text", text=json.dumps(self.answer))))
        client = AnthropicClient(model="claude-sonnet-5-5", api_key="test-key", max_retries=0)
        client._client = NS(messages=fake)
        self.gateway.transport = lambda tier, purpose: client
        self.assertEqual(self.run_decomposer()[1]["status"], "proposed")
        self.assertEqual(fake.sent[0]["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertGreater(fake.sent[0]["timeout"], 0)
        self.assertEqual(fake.sent[0]["max_tokens"], 9800)
        self.assertNotIn("_single_attempt", fake.sent[0])

        class ThinkingRefused(Exception):
            status_code = 400
            body = {"error": {"type": "invalid_request_error", "message": "thinking is not supported for this model"}}

        fake.sent.clear()
        fake.error = ThinkingRefused()
        self.assertEqual(self.run_decomposer()[0], self.baseline)
        self.assertEqual(len(fake.sent), 1)

    def test_endpoint_token_floor_cannot_override_the_total_token_budget(self):
        from dclab_rnd.models.client import ChatClient

        transport = ChatClient(model="claude-sonnet-5-5", api_key="test-key", max_retries=0, max_tokens_floor=20_000)
        self.gateway.transport = lambda tier, purpose: transport
        graph, report = self.run_decomposer()
        self.assertEqual(graph, self.baseline)
        self.assertEqual(report["status"], "fallback")
        self.assertFalse(self.sent)

    def test_scripted_dev_gate_records_expected_outcomes_without_live_calls(self):
        from dclab_rnd.tasks.evaluate import evaluate

        result = evaluate()
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["summary"]["passed"], 9)
        self.assertEqual(result["summary"]["provider_calls"], 6)
        self.assertEqual(result["summary"]["deliberately_invalid_proposals"], 3)
        self.assertEqual(result["summary"]["live_calls"], 0)


if __name__ == "__main__":
    unittest.main()
