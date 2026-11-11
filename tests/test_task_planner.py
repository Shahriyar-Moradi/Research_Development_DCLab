"""TP-03 plans calls; these tests never execute them or contact a model service."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from dclab_core.task_fingerprints import plan_fingerprint
from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.tasks.planner import PlanningBudget, plan_task, check_plan, resolve_action
from tests.test_models_gateway import CLEAN
from tests.task_planning_cases import case, advance


class TaskPlannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {**CLEAN, "DCLAB_NO_LIVE_MODELS": "1", "DCLAB_THINKING": "",
                               "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "claude-sonnet-5-5",
                               "OPENAI_BASE_URL": "https://api.anthropic.com/v1"}).start()
        self.project, self.source, self.workflow, self.caps, self.graph, self.state = case()
        self.remaining = {"max_tool_calls": 2, "max_seconds": 180, "max_cost_eur": 0.5}
        self.sent, self.mutate, self.reply = [], None, None
        outer = self

        class Transport:
            thinking = None

            def complete(self, messages, tools=None, max_tokens=None, **options):
                outer.sent.append(copy.deepcopy({"messages": messages, "tools": tools, "options": options}))
                if isinstance(outer.reply, Exception):
                    raise outer.reply
                baseline = json.loads(messages[-1]["content"])["baseline"]
                proposal = {"version": "tp03.v1", "baseline_fingerprint": baseline["fingerprint"],
                            **{k: baseline[k] for k in ("actions", "budget", "policy")}}
                if outer.mutate:
                    outer.mutate(proposal)
                return outer.reply or {"content": json.dumps(proposal), "tool_calls": [], "finish_reason": "stop",
                    "thinking": [{"text": "Keep one guarded stage and verify its declared evidence."}],
                    "usage": {"input_tokens": 10, "output_tokens": 30}}

        self.gateway = Gateway(FileUsage(self.home / "usage.jsonl"), transport=lambda tier, purpose: Transport(),
                               sleep=lambda _: self.fail("planner must not retry"))
        patch("dclab_rnd.models.gateway._exchange_home", return_value=self.home).start()
        patch("dclab_rnd.experience.model_text_allowed", return_value=True).start()

    def plan(self, task_id="data", **kwargs):
        return plan_task(self.workflow, self.source, self.project, graph=self.graph, task_id=task_id,
                         observed_state=self.state, capabilities=self.caps, remaining_budget=self.remaining, **kwargs)

    def check(self, plan):
        return check_plan(plan, self.workflow, self.source, self.project, graph=self.graph,
                          observed_state=self.state, capabilities=self.caps, remaining_budget=self.remaining)

    def test_deterministic_plan_for_each_supported_stage(self):
        for ident in ("data", "leakage", "features", "models", "holdout", "final", "report"):
            with self.subTest(task=ident):
                plan, report = self.plan(ident)
                self.assertEqual(plan.status, "noop" if ident == "holdout" else "ready")
                self.assertFalse(report["execution_authorized"])
                self.assertEqual(report["model_calls"], 0)
                self.assertEqual(self.check(plan), [])
                if ident != "holdout":
                    self.assertEqual(len(plan.actions), 1)
                    self.assertEqual(plan.actions[0].capability_name, "export_notebook" if ident == "report" else "run_stage")
                self.state = advance(self.graph, self.state, ident)

    def test_dependency_not_ready_is_an_explicit_blocker(self):
        plan, report = self.plan("models")
        self.assertEqual(plan.status, "blocked")
        self.assertFalse(plan.actions)
        self.assertIn("task_not_ready", report["blockers"])

    def test_missing_tool_argument_is_refused(self):
        plan, _ = self.plan()
        body = plan.model_dump(mode="json")
        body["actions"][0]["arguments"] = body["actions"][0]["arguments"][:1]
        body["fingerprint"] = plan_fingerprint(body)
        self.assertTrue(self.check(body))

    def test_same_inputs_make_same_plan_without_mutating_host_state(self):
        original = copy.deepcopy((self.project, self.state, self.caps))
        first, _ = self.plan()
        second, _ = self.plan()
        self.assertEqual(first, second)
        self.assertEqual(original, (self.project, self.state, self.caps))

    def test_benign_source_and_atomic_set_ordering_keeps_plan_valid(self):
        original, _ = self.plan()
        body = self.graph.model_dump(mode="json")
        body["source_refs"].reverse()
        for node in body["nodes"]:
            for key in ("source_step_ids", "needs", "produces", "acceptance_checks"):
                node[key].reverse()
        from dclab_core.task_graph import TaskGraph
        self.graph = TaskGraph.read(body)
        result, report = self.plan()
        self.assertEqual(report["status"], "deterministic")
        self.assertEqual(result.fingerprint, original.fingerprint)

    def test_default_off_never_opens_a_gateway(self):
        gateway = Mock()
        plan, report = self.plan(gateway=gateway)
        self.assertEqual(plan.status, "ready")
        self.assertEqual(report["status"], "deterministic")
        gateway.client.assert_not_called()

    def test_valid_model_proposal_keeps_source_inputs_outputs_checks_and_caps(self):
        baseline, _ = self.plan()
        candidate, report = self.plan(enabled=True, gateway=self.gateway)
        self.assertEqual(report["status"], "proposed")
        self.assertEqual(candidate.provenance.method, "model")
        for field in ("expected_inputs", "declared_outputs", "acceptance_checks", "selected_capabilities"):
            self.assertEqual(getattr(candidate, field), getattr(baseline, field))
        self.assertEqual(self.check(candidate), [])
        self.assertIsNone(self.sent[0]["tools"])
        self.assertTrue(self.sent[0]["options"]["_single_attempt"])

    def test_hallucinated_tool_and_wrong_scope_fall_back_exactly(self):
        baseline, _ = self.plan()
        for field, value in (("capability_name", "arbitrary_python"), ("scope", "draft")):
            with self.subTest(field=field):
                self.mutate = lambda p: p["actions"][0].update({field: value})
                result, report = self.plan(enabled=True, gateway=self.gateway)
                self.assertEqual(result, baseline)
                self.assertEqual(report["status"], "fallback")

    def test_model_missing_arg_wrong_stage_and_invalid_bindings_fall_back(self):
        baseline, _ = self.plan()
        edits = [lambda p: p["actions"][0].update(arguments=[]),
                 lambda p: p["actions"][0]["arguments"][1]["value"].update(ref_id="stage_final"),
                 lambda p: p["actions"][0]["arguments"][1]["value"].update(kind="output", ref_id="data_record"),
                 lambda p: p["actions"][0]["arguments"][1]["value"].update(ref_id="nonexistent")]
        for edit in edits:
            self.mutate = edit
            result, report = self.plan(enabled=True, gateway=self.gateway)
            self.assertEqual(result, baseline)
            self.assertEqual(report["status"], "fallback")

    def test_duplicate_effect_and_automatic_retry_are_refused(self):
        baseline, _ = self.plan()
        for edit in (lambda p: p["actions"].append({**p["actions"][0], "id": "again"}),
                     lambda p: p["policy"].update(max_retries=1)):
            self.mutate = edit
            result, report = self.plan(enabled=True, gateway=self.gateway)
            self.assertEqual(result, baseline)
            self.assertEqual(report["status"], "fallback")

    def test_model_cannot_insert_literals_private_data_or_completion_claims(self):
        baseline, _ = self.plan()
        for field, value in (("done", True), ("reasoning", "PRIVATE_ROW"), ("goal", "another objective")):
            self.mutate = lambda p: p.update({field: value})
            result, report = self.plan(enabled=True, gateway=self.gateway)
            self.assertEqual(result, baseline)
            self.assertNotIn("PRIVATE_ROW", json.dumps(report))

    def test_capability_permission_scope_move_and_schema_changes_are_refused(self):
        plan, _ = self.plan()
        for field, value in (("allowed", False), ("scope", "draft"), ("move", None), ("guard_present", False)):
            with self.subTest(field=field), patch.dict(self.caps["run_stage"], {field: value}):
                self.assertTrue(self.check(plan))
        self.caps["run_stage"]["parameters"]["description"] = "Changed schema"
        self.assertTrue(self.check(plan))

    def test_static_arguments_must_fit_the_real_registry_schema(self):
        self.caps["run_stage"]["parameters"]["properties"]["project_id"]["pattern"] = "^another_project$"
        plan, report = self.plan(enabled=True, gateway=self.gateway)
        self.assertIsNone(plan)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(self.sent)

    def test_schema_references_cannot_access_a_remote_document(self):
        self.caps["run_stage"]["parameters"]["properties"]["project_id"]["$ref"] = "https://invalid.local/schema"
        self.assertEqual(self.plan()[1]["status"], "blocked")

    def test_source_data_goal_or_signoff_change_blocks_before_model(self):
        plan, _ = self.plan()
        for target, key, value in ((self.project, "goal", "Predict another outcome"),
                                  (self.project["data"], "sha256", "a" * 64),
                                  (self.project["signoff"], "by", "another-person")):
            with patch.dict(target, {key: value}):
                self.assertTrue(self.check(plan))
                self.assertEqual(self.plan(enabled=True, gateway=self.gateway)[1]["status"], "blocked")
        self.assertFalse(self.sent)

    def test_missing_and_changed_artifacts_block(self):
        del self.state["artifacts"]["table"]
        self.assertEqual(self.plan()[1]["status"], "blocked")
        self.state["artifacts"]["table"] = {"fingerprint": "a" * 64, "value_type": "artifact"}
        self.assertEqual(self.plan()[1]["status"], "blocked")

    def test_private_artifact_values_and_unrelated_artifacts_are_refused(self):
        for ident, meta in (("table", {"value": "PRIVATE_ROW"}), ("unrelated_table", {"value_type": "artifact", "fingerprint": "d" * 64})):
            old = copy.deepcopy(self.state)
            self.state["artifacts"].setdefault(ident, {}).update(meta)
            _, report = self.plan(enabled=True, gateway=self.gateway)
            self.assertEqual(report["status"], "blocked")
            self.assertNotIn("PRIVATE_ROW", json.dumps(report))
            self.assertFalse(self.sent)
            self.state = old

    def test_gate_group_and_unknown_task_never_become_completed_actions(self):
        for ident, expected in (("holdout", "noop"), ("root", "blocked"), ("invented", "blocked")):
            plan, report = self.plan(ident, enabled=True, gateway=self.gateway)
            self.assertEqual(report["status"], expected)
            self.assertFalse(plan and plan.actions)
            self.assertFalse(report["execution_authorized"])
        self.assertFalse(self.sent)

    def test_existing_outputs_and_failed_attempts_require_reconciliation(self):
        self.state["artifacts"]["data_record"] = {"value_type": "artifact", "fingerprint": "d" * 64, "producer_task": "data"}
        self.assertEqual(self.plan()[1]["status"], "blocked")
        self.state["artifacts"].pop("data_record")
        for failure in ("transient", "ambiguous_effect", "policy_refusal", "PRIVATE_ERROR"):
            plan, report = self.plan(past_failures=(failure,), enabled=True, gateway=self.gateway)
            self.assertEqual(plan.status, "blocked")
            self.assertNotIn("PRIVATE_ERROR", json.dumps(report))
        self.assertFalse(self.sent)

    def test_budget_exhaustion_and_model_overbudget(self):
        for key in ("max_tool_calls", "max_seconds"):
            with patch.dict(self.remaining, {key: 0}):
                self.assertEqual(self.plan()[0].status, "blocked")
        baseline, _ = self.plan()
        self.mutate = lambda p: p["budget"].update(max_cost_eur=0.6)
        self.assertEqual(self.plan(enabled=True, gateway=self.gateway)[0], baseline)
        self.mutate = lambda p: p["budget"].update(max_seconds=0)
        self.assertEqual(self.plan(enabled=True, gateway=self.gateway)[0], baseline)

    def test_zero_planning_cap_or_input_limit_makes_no_model_call(self):
        for budget in ({"cap_eur": 0.0}, {"max_input_bytes": 1000}):
            plan, report = self.plan(enabled=True, gateway=self.gateway, budget=budget)
            self.assertEqual(plan.status, "ready")
            self.assertEqual(report["status"], "fallback")
        self.assertFalse(self.sent)

    def test_model_unavailable_error_refusal_or_truncation_keeps_exact_fallback(self):
        baseline, _ = self.plan()
        unavailable = Mock()
        unavailable.client.return_value = None
        self.assertEqual(self.plan(enabled=True, gateway=unavailable)[0], baseline)
        for reply in (RuntimeError("PRIVATE_PROVIDER_ERROR"), {"content": "{}", "finish_reason": "refusal"},
                      {"content": "{}", "finish_reason": "length"}, {"content": "{}", "tool_calls": [{"name": "run_stage"}]}):
            self.reply = reply
            plan, report = self.plan(enabled=True, gateway=self.gateway)
            self.assertEqual(plan, baseline)
            self.assertNotIn("PRIVATE_PROVIDER_ERROR", json.dumps(report))

    def test_whole_call_timeout_stops_request_and_keeps_fallback(self):
        baseline, _ = self.plan()
        self.mutate = lambda p: time.sleep(0.15)
        began = time.monotonic()
        plan, report = self.plan(enabled=True, gateway=self.gateway, budget={"max_seconds": 0.06})
        self.assertEqual(plan, baseline)
        self.assertEqual(report["refusals"], ["planning_timeout"])
        self.assertLess(time.monotonic() - began, 0.4)

    def test_changed_prompt_or_output_schema_invalidates_a_stored_plan(self):
        plan, _ = self.plan()
        from dclab_rnd import prompts
        original = prompts.text("task_planner")
        with patch("dclab_rnd.prompts.text", return_value=original + "\nChanged policy"):
            self.assertTrue(self.check(plan))

    def test_resolved_arguments_are_rechecked_and_revocation_prevents_resolution(self):
        plan, _ = self.plan()
        kwargs = dict(graph=self.graph, observed_state=self.state, capabilities=self.caps, remaining_budget=self.remaining)
        self.assertEqual(resolve_action(plan, "call", self.workflow, self.source, self.project, **kwargs),
                         {"project_id": self.project["id"], "stage": "data"})
        self.caps["run_stage"]["allowed"] = False
        with self.assertRaises(ValueError):
            resolve_action(plan, "call", self.workflow, self.source, self.project, **kwargs)

    def test_planning_never_calls_tools_engine_or_holdout_guard(self):
        from dclab_rnd.agents.registry import Registry
        with patch.object(Registry, "call") as call, patch.object(Registry, "acall") as acall, \
                patch("dclab_rnd.studio.engine.execute") as execute, patch("dclab_rnd.studio.graph.check") as guard:
            plan, report = self.plan(enabled=True, gateway=self.gateway)
            self.assertEqual(report["status"], "proposed")
            for spy in (call, acall, execute, guard):
                spy.assert_not_called()
        self.assertEqual(self.state["holdout_uses"], 0)

    def test_actual_api_thinking_summary_is_saved_next_to_answer(self):
        _, report = self.plan(enabled=True, gateway=self.gateway)
        rows = [json.loads(line) for p in (self.home / "model_exchanges").rglob("*.jsonl") for line in p.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["purpose"], "task_planner")
        self.assertEqual(rows[0]["thinking_label"], "API thinking summary")
        self.assertEqual(rows[0]["thinking_asked"], "adaptive")
        self.assertEqual(rows[0]["thinking"][0]["text"], "Keep one guarded stage and verify its declared evidence.")
        self.assertTrue(rows[0]["reply"])
        self.assertNotIn("thinking", json.dumps(report))

    def test_recorded_dev_comparison_matches_current_prompt_code_and_fixture(self):
        from dclab_rnd.tasks.evaluate_plans import GATE, ROOT, SCENARIOS, fingerprints
        result = json.loads((ROOT / "evaluation/task_planning_tp03_dev.json").read_text())
        self.assertEqual(result["fingerprints"], fingerprints())
        self.assertEqual(result["gate"], GATE)
        self.assertEqual([(r["id"], r["task"], r["expected"]) for r in result["rows"]], list(SCENARIOS))
        self.assertTrue(all(r["passes"] for r in result["rows"]))
        self.assertEqual(result["verdict"], "pass")
        for key in ("live_calls", "cost_eur", "tool_calls", "extra_holdout_accesses"):
            self.assertEqual(result["summary"][key], 0)


if __name__ == "__main__":
    unittest.main()
