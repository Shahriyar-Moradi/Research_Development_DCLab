"""TP-01D: compile real workflow/registry metadata without running a tool or reading a table."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dclab_core.contracts import SolutionPlan
from dclab_core.solution_workflow import SolutionWorkflow, solution_workflow
from dclab_core.task_fingerprints import capability_fingerprint, graph_fingerprint
from dclab_core.task_validation import validate_graph
from dclab_rnd.agents import build_registry
from dclab_rnd.tasks.adapter import adapt_workflow, capability_snapshot

FIXTURE = Path(__file__).parent / "fixtures/tasks/telco_metadata.json"


class TaskAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text())
        self.project = self.fixture["project"]
        self.plan = SolutionPlan.read(self.fixture["source_plan"])
        self.workflow = solution_workflow({"family": "classification"})
        self.registry = build_registry(("project",))
        self.caps = capability_snapshot(self.registry, allowed_tools={"run_stage", "export_notebook"})

    def adapt(self, **kwargs):
        return adapt_workflow(self.workflow, self.plan, self.project, capabilities=self.caps, **kwargs)

    def test_real_workflow_and_registry_form_a_valid_nonexecuting_graph(self):
        graph, report = self.adapt()
        self.assertEqual(report["status"], "mapped")
        self.assertEqual(validate_graph(graph, self.workflow, known_capabilities=self.caps,
                                       external_artifacts=report["external_artifacts"]), [])
        self.assertEqual(graph.fingerprint, graph_fingerprint(graph))
        self.assertEqual(report["topological_order"], ["data", "leakage", "features", "models", "holdout", "final", "report"])
        self.assertEqual(self.workflow, SolutionWorkflow.read(self.fixture["workflow"]))

    def test_source_coverage_is_exact_and_physical_effects_are_not_duplicated(self):
        graph, report = self.adapt(); nodes = {node.id: node for node in graph.nodes}
        self.assertEqual(set(nodes["data"].source_step_ids), {"class_balance", "lock_holdout", "train_only_analysis"})
        self.assertEqual(set(nodes["final"].source_step_ids), {"threshold", "final_once"})
        self.assertEqual(nodes["features"].source_step_ids, ())
        self.assertEqual(nodes["holdout"].kind, "gate")
        self.assertIsNone(nodes["holdout"].capability)
        self.assertEqual({m["source_step_id"] for m in report["source_mapping"]}, {s.id for s in self.workflow.steps})
        self.assertEqual(sum(n.capability is not None and "final_once" in n.source_step_ids for n in graph.nodes), 1)

    def test_repeated_build_and_json_roundtrip_have_the_same_identity(self):
        first, a = self.adapt(); second, b = self.adapt()
        self.assertEqual(first, second); self.assertEqual(a, b)
        self.assertEqual(first.fingerprint, graph_fingerprint(json.loads(first.model_dump_json())))

    def test_no_tools_guards_storage_models_or_table_reads_are_called(self):
        project = copy.deepcopy(self.project); wf = self.workflow.model_dump_json(); plan = self.plan.model_dump_json()
        # Spy on the stored handlers too: they were bound when the registry was
        # built, so patching Toolbox methods alone would miss a direct call.
        spies = []
        for name, tool in list(self.registry.tools.items()):
            handler = Mock(name=name, side_effect=AssertionError("tool handler called"))
            self.registry.tools[name] = replace(tool, handler=handler)
            spies.append(handler)
        guard = Mock(side_effect=AssertionError("guard called during planning"))
        self.registry.guards["project"] = guard
        spies.append(guard)
        targets = [
            "dclab_rnd.studio.engine.prepare", "dclab_rnd.studio.engine.stage_final",
            "dclab_rnd.studio.data.data_path", "dclab_rnd.studio.data.load_table",
            "dclab_core.programme.read_holdout", "dclab_core.compose.read_holdout",
            "dclab_rnd.models.gateway.Gateway.client", "dclab_rnd.models.gateway.Bound.complete",
            "dclab_rnd.models.gateway.Bound.stream", "dclab_rnd.models.gateway.Bound.embed",
        ]
        for owner in ("dclab_rnd.studio.store.ProjectStore", "dclab_rnd.storage.postgres.PgProjects"):
            targets.extend(owner + "." + method for method in
                           ("create", "save", "update", "write_stage", "clear_stages", "delete", "log", "transition"))
        with ExitStack() as stack:
            spies.extend(stack.enter_context(patch.object(self.registry, method,
                         side_effect=AssertionError("registry invoked"))) for method in ("call", "acall", "check"))
            spies.extend(stack.enter_context(patch(target, side_effect=AssertionError("runtime invoked")))
                         for target in targets)
            caps = capability_snapshot(self.registry, allowed_tools={"run_stage", "export_notebook"})
            graph, report = adapt_workflow(self.workflow, self.plan, self.project, capabilities=caps)
            self.assertEqual(report["status"], "mapped")
            self.assertFalse(report["execution_authorized"])
            self.assertEqual(validate_graph(graph, self.workflow, known_capabilities=caps,
                                           external_artifacts=report["external_artifacts"]), [])
            # A caught exception or a blocked fallback must not hide an attempt.
            for spy in spies:
                spy.assert_not_called()
        self.assertEqual(self.project, project)
        self.assertEqual(self.workflow.model_dump_json(), wf); self.assertEqual(self.plan.model_dump_json(), plan)

    def test_missing_source_identities_block_without_fabricating_references(self):
        for field in ("solution", "data"):
            with self.subTest(field=field):
                project = copy.deepcopy(self.project); project.pop(field)
                graph, report = adapt_workflow(self.workflow, self.plan, project, capabilities=self.caps)
                self.assertIsNone(graph); self.assertEqual(report["status"], "blocked")
                self.assertTrue(report["blockers"])

    def test_unconfirmed_label_never_gets_an_action_binding(self):
        self.project.pop("signoff")
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))
        self.assertTrue(any(n.kind == "blocked" and n.blocker_reason for n in graph.nodes))

    def test_missing_capability_is_an_explicit_blocked_task(self):
        self.caps.pop("run_stage")
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(next(n.kind for n in graph.nodes if n.id == "models"), "blocked")
        self.assertFalse(any(n.capability and n.capability.name == "run_all" for n in graph.nodes))

    def test_invalid_source_workflow_is_refused_before_adaptation(self):
        self.workflow = self.workflow.model_copy(update={"steps": [s for s in self.workflow.steps if s.id != "leakage_audit"]})
        with self.assertRaisesRegex(ValueError, "source workflow"):
            self.adapt()

    def test_unknown_family_and_missing_source_plan_are_blocked(self):
        for workflow, plan in ((self.workflow.model_copy(update={"family": "unknown"}), self.plan), (self.workflow, None)):
            graph, report = adapt_workflow(workflow, plan, self.project, capabilities=self.caps)
            self.assertIsNone(graph); self.assertEqual(report["status"], "blocked")

    def test_changed_data_or_solution_changes_its_source_identity(self):
        from dclab_rnd.studio.graph import solution_hash
        original, report = self.adapt()
        self.project["data"]["sha256"] = "e" * 64
        changed, _ = self.adapt()
        self.assertNotEqual(changed.fingerprint, original.fingerprint)
        self.assertEqual(changed.solution_fingerprint, original.solution_fingerprint)
        self.project["solution"]["prediction_moment"] = "Before the next renewal using only current account fields."
        self.project["signoff"]["solution_hash"] = solution_hash(self.project["solution"])
        changed, _ = self.adapt()
        self.assertNotEqual(changed.solution_fingerprint, original.solution_fingerprint)

    def test_registry_membership_alone_does_not_grant_permission(self):
        caps = capability_snapshot(self.registry, allowed_tools=set())
        graph, report = adapt_workflow(self.workflow, self.plan, self.project, capabilities=caps)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_raw_profile_examples_and_free_text_are_not_copied_into_the_plan(self):
        sentinel = "PRIVATE_TABLE_VALUE_NEVER_COPY"
        self.project["data"]["profile"] = {"examples": [sentinel], "not_json": object()}
        self.project["goal"] = sentinel
        graph, report = self.adapt()
        self.assertNotIn(sentinel, graph.model_dump_json() + json.dumps(report))

    def test_different_saved_runtime_plan_blocks_even_with_signed_solution(self):
        saved = self.plan.model_dump(mode="json")
        saved["decisions"].append({"key": "threshold.policy", "choice": {"kind": "expected_cost", "costs": "10 per missed case"},
                                   "reason": "This saved runtime policy is not the supplied source plan"})
        self.project["plan"] = saved
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_runtime_plan_identity_is_preserved_when_present(self):
        self.project["plan"] = self.plan.model_dump(mode="json")
        first, report = self.adapt()
        self.assertEqual(report["status"], "mapped")
        ref = next(ref for ref in first.source_refs if ref.id == "runtime_plan")
        self.assertEqual(ref.fingerprint, next(ref.fingerprint for ref in first.source_refs if ref.id == "source_plan"))

    def test_each_source_logical_artifact_points_to_existing_owner_evidence(self):
        graph, report = self.adapt()
        by_id = {row["source_step_id"]: row for row in report["source_mapping"]}
        self.assertEqual(by_id["class_balance"]["output_artifact_refs"], {})
        self.assertEqual(by_id["threshold"]["output_artifact_refs"], {})
        self.assertEqual(by_id["final_once"]["input_artifact_refs"]["holdout"], "data_record")
        self.assertEqual(by_id["report_limits"]["input_artifact_refs"]["findings"], "data_record")

    def test_registry_snapshot_copies_the_actual_schema_and_detaches_it(self):
        for name, cap in self.caps.items():
            tool = self.registry.tools[name]
            self.assertEqual((cap["name"], cap["scope"], cap["effect"], cap["move"]),
                             (tool.name, tool.scope, tool.effect, tool.move))
            self.assertEqual(cap["parameters"], tool.parameters)
            self.assertIsNot(cap["parameters"], tool.parameters)
        graph, _ = self.adapt()
        for node in graph.nodes:
            if node.capability:
                self.assertEqual(node.capability.schema_fingerprint, capability_fingerprint(self.caps[node.capability.name]))
        self.caps["run_stage"]["parameters"]["properties"]["stage"]["enum"].clear()
        self.assertTrue(self.registry.tools["run_stage"].parameters["properties"]["stage"]["enum"])

    def test_missing_guard_or_host_allowlist_never_grants_a_write(self):
        self.registry.guards.pop("project")
        caps = capability_snapshot(self.registry, allowed_tools={"run_stage", "export_notebook"})
        self.assertTrue(all(cap["allowed"] is False for cap in caps.values()))
        graph, report = adapt_workflow(self.workflow, self.plan, self.project, capabilities=caps)
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))
        self.assertEqual(report["status"], "blocked")
        with self.assertRaises(TypeError): capability_snapshot(self.registry)

    def test_registry_schema_scope_and_move_changes_block_the_actual_operation(self):
        for field, value in (("scope", "draft"), ("effect", "read"), ("move", "capture"),
                             ("parameters", {"type": "object", "required": ["new_required_argument"]})):
            with self.subTest(field=field):
                caps = copy.deepcopy(self.caps); caps["run_stage"][field] = value
                graph, report = adapt_workflow(self.workflow, self.plan, self.project, capabilities=caps)
                self.assertEqual(report["status"], "blocked")
                self.assertEqual(next(n.kind for n in graph.nodes if n.id == "models"), "blocked")

    def test_schema_that_removes_final_does_not_produce_a_final_action(self):
        self.caps["run_stage"]["parameters"]["properties"]["stage"]["enum"].remove("final")
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(next(n.kind for n in graph.nodes if n.id == "final"), "blocked")

    def test_runtime_callbacks_in_supplied_snapshot_are_rejected_without_calling(self):
        self.caps["run_stage"]["handler"] = lambda: self.fail("runtime handler called")
        with self.assertRaisesRegex(ValueError, "JSON metadata"):
            self.adapt()

    def test_changed_plan_choice_and_unadopted_model_families_are_blocked(self):
        for change in ("metric", "families"):
            plan = self.plan.model_dump(mode="json")
            if change == "metric":
                next(d for d in plan["decisions"] if d["key"] == "metric.primary")["choice"] = "average_precision"
            else:
                plan["decisions"].append({"key": "model.families", "choice": ["invented_model"], "reason": "An unadopted model proposal must not become an execution promise"})
            graph, report = adapt_workflow(self.workflow, SolutionPlan.read(plan), self.project, capabilities=self.caps)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_missing_target_column_and_stale_human_signoff_block_all_actions(self):
        for change in ("column", "signoff"):
            project = copy.deepcopy(self.project)
            if change == "column": project["data"]["columns"].remove("Churn")
            else: project["signoff"]["solution_hash"] = "0" * 16
            graph, report = adapt_workflow(self.workflow, self.plan, project, capabilities=self.caps)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_source_plan_status_cannot_self_confirm_an_outcome(self):
        project = copy.deepcopy(self.project); project.pop("signoff")
        plan = self.plan.model_dump(mode="json")
        for decision in plan["decisions"]: decision.update(source="advisor", status="accepted")
        graph, report = adapt_workflow(self.workflow, SolutionPlan.read(plan), project, capabilities=self.caps)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_source_variants_without_a_proven_mapping_do_not_guess_a_stage(self):
        from dclab_core.solution_workflow import Step
        source = copy.deepcopy(self.workflow)
        source.steps.append(Step(id="special_report", block="WF-10", name="Extra report", purpose="PRIVATE_SOURCE_TEXT",
                                 inputs=["holdout score"], outputs=["report"], decision="risk.check", check="PRIVATE_SOURCE_TEXT", rule="Extra requirement"))
        graph, report = adapt_workflow(source, self.plan, self.project, capabilities=self.caps)
        self.assertIsNone(graph); self.assertEqual(report["status"], "blocked")
        self.assertNotIn("PRIVATE_SOURCE_TEXT", json.dumps(report))
        graph, report = adapt_workflow(solution_workflow({"family": "regression"}), self.plan, self.project, capabilities=self.caps)
        self.assertIsNone(graph); self.assertEqual(report["status"], "blocked")

    def test_nonbasic_solution_splits_are_blocked_instead_of_falsely_stratified(self):
        from dclab_rnd.studio.graph import solution_hash
        for field in ("time_column", "group_column"):
            project = copy.deepcopy(self.project); project["solution"][field] = "tenure"
            project["signoff"]["solution_hash"] = solution_hash(project["solution"])
            graph, report = adapt_workflow(self.workflow, self.plan, project, capabilities=self.caps)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_project_identity_and_revision_are_part_of_structural_identity(self):
        graph, report = self.adapt()
        revised, _ = self.adapt(revision=2)
        self.assertEqual(graph.revision, 1); self.assertEqual(revised.revision, 2)
        self.assertNotEqual(graph.fingerprint, revised.fingerprint)
        project = copy.deepcopy(self.project); project["id"] = "another_project"
        other, _ = adapt_workflow(self.workflow, self.plan, project, capabilities=self.caps)
        self.assertNotEqual(other.graph_id, graph.graph_id); self.assertNotEqual(other.fingerprint, graph.fingerprint)

    def test_mapping_never_creates_execution_approval_or_claims_completion(self):
        graph, report = self.adapt()
        self.assertIs(report["execution_authorized"], False)
        self.assertNotIn("approval", report)
        self.assertNotIn("holdout_approval", report)
        final = next(op for op in report["operations"] if op["task_id"] == "final")
        self.assertIn("evidence.task_assessment.threshold", final["evidence_paths"])
        self.assertNotIn("threshold_analysis", json.dumps(report))

    def test_schema_references_cannot_fetch_network_or_files_during_planning(self):
        for uri in ("https://example.invalid/forbidden-schema", "file:///forbidden-schema"):
            caps = copy.deepcopy(self.caps)
            caps["run_stage"]["parameters"]["properties"]["project_id"] = {"$ref": uri}
            with patch("urllib.request.urlopen", side_effect=AssertionError("schema resolution I/O")) as fetch:
                graph, report = adapt_workflow(self.workflow, self.plan, self.project, capabilities=caps)
            fetch.assert_not_called()
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(any(n.capability and n.capability.name == "run_stage" for n in graph.nodes))

    def test_execution_context_changes_are_bound_without_echoing_settings(self):
        original, _ = self.adapt()
        self.project["settings"] = {"quick": True, "max_rows": 500, "folds": 3, "private_note": "PRIVATE_SETTING_VALUE"}
        graph, report = self.adapt()
        self.assertEqual(report["status"], "mapped")
        self.assertNotEqual(graph.fingerprint, original.fingerprint)
        context = next(ref for ref in graph.source_refs if ref.id == "execution_context")
        self.assertEqual(context.fingerprint, report["external_artifacts"]["execution_context"]["fingerprint"])
        for node in graph.nodes:
            if node.kind in ("action", "gate"): self.assertIn("execution_context", node.needs)
        self.assertNotIn("PRIVATE_SETTING_VALUE", graph.model_dump_json() + json.dumps(report))

    def test_every_consumed_project_context_has_a_distinct_identity(self):
        original, _ = self.adapt()
        for key, value in (("pack", "custom_pack"), ("feature_programs", [{"op": "ratio", "name": "ratio"}]),
                           ("brief", {"changed": True}), ("exposure", {"seen": True}),
                           ("adopted_improvement", {"params": {"max_depth": 4}}),
                           ("name", "Changed project name"), ("proposal", {"target": "Churn", "modifiers": ["imbalance"]})):
            with self.subTest(key=key):
                project = copy.deepcopy(self.project); project[key] = value
                graph, _ = adapt_workflow(self.workflow, self.plan, project, capabilities=self.caps)
                self.assertNotEqual(graph.fingerprint, original.fingerprint)

    def test_effective_pack_cannot_silently_change_binary_to_rare_event(self):
        self.project["pack"] = "imbalanced"
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_problem_context_is_an_actual_problem_card(self):
        from dclab_core.card import ProblemCard, problem_card
        from dclab_rnd.core_adapter import brief_from_project
        from dclab_core.task_fingerprints import metadata_fingerprint
        graph, _ = self.adapt()
        card = problem_card(brief_from_project(self.project))
        self.assertEqual(ProblemCard.read(card.model_dump(mode="json")), card)
        self.assertEqual(next(r.fingerprint for r in graph.source_refs if r.id == "problem_context"), metadata_fingerprint(card.model_dump(mode="json")))

    def test_final_evidence_uses_existing_record_paths(self):
        graph, report = self.adapt()
        final = next(op for op in report["operations"] if op["task_id"] == "final")
        self.assertIn("evidence.holdout_metrics", final["evidence_paths"])
        self.assertIn("evidence.holdout_primary_metric_ci", final["evidence_paths"])
        self.assertNotIn("metrics", final["evidence_paths"])

    def test_programme_project_is_never_flattened_into_stage_actions(self):
        self.project["programme_plan"] = {"blocks": [{"id": "a"}, {"id": "b"}]}
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_categorical_and_explicit_sensitivity_metadata_change_identity(self):
        original, _ = self.adapt()
        self.project["data"]["categorical"] = ["tenure"]
        graph, report = self.adapt(); self.assertNotEqual(graph.fingerprint, original.fingerprint)
        self.project["data"]["profile"] = {"columns": [{"name": "tenure", "sensitivity": "sensitive", "preview": [object()]}]}
        changed, report = self.adapt()
        self.assertEqual(report["status"], "mapped")
        self.assertNotEqual(changed.fingerprint, graph.fingerprint)

    def test_legacy_profile_classification_does_not_inspect_private_previews(self):
        self.project["data"]["profile"] = {"columns": [{"name": "tenure", "preview": [object()]}]}
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))

    def test_draft_pack_fallback_and_explicit_binary_override_follow_current_owner(self):
        self.project["draft"] = {"pack": {"key": "imbalanced"}, "preview": object()}
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        from dclab_rnd.studio.graph import solution_hash
        self.project["solution"]["kind"] = "binary"
        self.project["signoff"]["solution_hash"] = solution_hash(self.project["solution"])
        graph, report = self.adapt()
        self.assertEqual(report["status"], "mapped")

    def test_verification_gaps_are_declared_instead_of_being_treated_as_success(self):
        graph, report = self.adapt()
        gaps = {row["source_step_id"]: row["limit"] for row in report["verification_limits"]}
        self.assertEqual(set(gaps), {"class_balance", "threshold", "report_limits"})
        self.assertIn("missing proof", gaps["threshold"])

    def test_planning_does_not_mutate_or_invoke_registry_handlers_and_guards(self):
        guard = Mock(side_effect=AssertionError("guard invoked"))
        self.registry.guards["project"] = guard
        with patch.object(type(self.registry), "check", side_effect=AssertionError("registry check")) as check, \
             patch("dclab_rnd.intern.tools.Toolbox.run_stage", side_effect=AssertionError("stage invoked")) as stage, \
             patch("dclab_rnd.intern.tools.Toolbox.export_notebook", side_effect=AssertionError("export invoked")) as export:
            caps = capability_snapshot(self.registry, allowed_tools={"run_stage", "export_notebook"})
            before = copy.deepcopy(caps)
            _, report = adapt_workflow(self.workflow, self.plan, self.project, capabilities=caps)
            self.assertEqual(report["status"], "mapped")
            self.assertEqual(caps, before)
            for spy in (guard, check, stage, export):
                spy.assert_not_called()

    def test_published_preview_matches_the_real_adapter_and_roundtrips(self):
        from dclab_core.task_graph import TaskGraph
        example = json.loads(FIXTURE.with_name("telco_task_preview.json").read_text())
        graph, report = self.adapt()
        self.assertEqual(TaskGraph.read(example["graph"]), graph)
        self.assertEqual(example["mapping_report"], report)

    def test_missing_execution_settings_do_not_silently_assume_defaults(self):
        self.project.pop("settings")
        graph, report = self.adapt()
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(any(n.kind == "action" for n in graph.nodes))


if __name__ == "__main__":
    unittest.main()
