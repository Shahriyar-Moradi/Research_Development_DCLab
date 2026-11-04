"""TP-01B: bounded plans contain typed references, never executable expressions."""
import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from dclab_core.task_execution_plan import TaskExecutionPlan

SCHEMA = "dclab_core.task_execution_plan.v1"
DIGEST = "a" * 64


def plan_document():
    binding = {"name": "run_stage", "scope": "project", "schema_fingerprint": DIGEST}
    return {
        "schema_version": SCHEMA, "graph_id": "churn", "graph_revision": 1, "graph_fingerprint": DIGEST,
        "task_id": "data", "plan_id": "data_plan", "revision": 1, "fingerprint": DIGEST,
        "status": "ready", "expected_inputs": [
            {"id": "project", "fingerprint": DIGEST, "value_type": "string"},
            {"id": "data_stage", "fingerprint": DIGEST, "value_type": "string"}],
        "selected_capabilities": [binding],
        "actions": [{"id": "run_data", "capability_name": "run_stage", "scope": "project", "arguments": [
            {"name": "project_id", "value": {"kind": "input", "ref_id": "project", "value_type": "string"}},
            {"name": "stage", "value": {"kind": "input", "ref_id": "data_stage", "value_type": "string"}}]}],
        "declared_outputs": [{"id": "data_record", "action_id": "run_data", "value_type": "artifact"}],
        "acceptance_checks": [{"id": "recorded", "kind": "lineage", "description": "Data record matches current inputs", "evidence_refs": ["data_record"]}],
        "budget": {"max_tool_calls": 1, "max_seconds": 120, "max_cost_eur": 0},
        "provenance": {"method": "deterministic", "reference_id": "planner", "producer_version": "v1"},
    }


class TaskExecutionPlanTests(unittest.TestCase):
    def test_minimal_capability_reference_plan_roundtrips(self):
        plan = TaskExecutionPlan.read(plan_document())
        self.assertEqual(TaskExecutionPlan.read(plan.model_dump_json()), plan)
        self.assertEqual(plan.actions[0].arguments[0].value.kind, "input")
        self.assertEqual(plan.policy.on_unknown, "block")

    def test_missing_or_unknown_versions_and_unknown_fields_are_refused(self):
        for version in (None, "dclab_core.task_execution_plan.v2"):
            data = plan_document()
            if version is None: del data["schema_version"]
            else: data["schema_version"] = version
            with self.subTest(version=version), self.assertRaises((ValueError, ValidationError)): TaskExecutionPlan.read(data)
            with self.assertRaises(ValidationError): TaskExecutionPlan.model_validate(data)
        for location in ("root", "action", "argument", "reference", "budget", "policy"):
            data = plan_document(); data["policy"] = {"on_unknown": "block"}
            obj = {"root": data, "action": data["actions"][0], "argument": data["actions"][0]["arguments"][0],
                   "reference": data["actions"][0]["arguments"][0]["value"], "budget": data["budget"], "policy": data["policy"]}[location]
            obj["execute"] = "arbitrary"
            with self.subTest(location=location), self.assertRaises(ValidationError): TaskExecutionPlan.read(data)

    def test_expressions_raw_literals_and_unresolved_references_are_refused(self):
        for value in ("__import__('os').system('anything')", "${project.id}",
                      {"kind": "expression", "expression": "read_secret()"}, {"kind": "literal", "value": "models"},
                      {"kind": "input", "ref_id": "../secret", "value_type": "string"},
                      {"kind": "input", "ref_id": "missing", "value_type": "string"},
                      {"kind": "input", "ref_id": "project", "value_type": "integer"}):
            data = plan_document(); data["actions"][0]["arguments"][0]["value"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError): TaskExecutionPlan.read(data)

    def test_deferred_reference_must_be_to_an_earlier_declared_output(self):
        data = plan_document()
        data["budget"]["max_tool_calls"] = 2
        data["actions"].append({"id": "inspect", "capability_name": "run_stage", "scope": "project", "arguments": [
            {"name": "record", "value": {"kind": "output", "ref_id": "data_record", "value_type": "artifact"}}]})
        self.assertEqual(len(TaskExecutionPlan.read(data).actions), 2)
        # Schema/authorization of 'record' for run_stage belongs to TP-03; this only validates reference shape/order.
        for ref in ("missing", "data_record"):
            broken = copy.deepcopy(data)
            broken["actions"][0]["arguments"][0]["value"] = {"kind": "output", "ref_id": ref, "value_type": "artifact"}
            with self.subTest(ref=ref), self.assertRaises(ValidationError): TaskExecutionPlan.read(broken)
        broken = copy.deepcopy(data); broken["declared_outputs"][0]["action_id"] = "missing"
        with self.assertRaises(ValidationError): TaskExecutionPlan.read(broken)

    def test_noop_and_blocked_are_distinct_nonexecuting_states(self):
        for status in ("noop", "blocked"):
            data = plan_document(); data.update(status=status, reason="Requires a person's current approval", actions=[], declared_outputs=[], selected_capabilities=[])
            data["budget"]["max_tool_calls"] = 0
            self.assertEqual(TaskExecutionPlan.read(data).status, status)
            for reason in (None, "", "  "):
                broken = copy.deepcopy(data); broken["reason"] = reason
                with self.subTest(status=status, reason=reason), self.assertRaises(ValidationError): TaskExecutionPlan.read(broken)
            data["actions"] = plan_document()["actions"]
            with self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        for status in ("completed", "verified", "success"):
            data = plan_document(); data["status"] = status
            with self.subTest(status=status), self.assertRaises(ValidationError): TaskExecutionPlan.read(data)

    def test_bounds_identity_duplicates_capability_and_budget_are_checked(self):
        for key, value in (("task_id", "http://tool"), ("revision", 0), ("revision", True), ("graph_revision", 0),
                           ("actions", plan_document()["actions"] * 33), ("expected_inputs", plan_document()["expected_inputs"] * 65),
                           ("actions", plan_document()["actions"] * 2), ("selected_capabilities", []),
                           ("expected_inputs", plan_document()["expected_inputs"] * 2),
                           ("declared_outputs", plan_document()["declared_outputs"] * 2), ("fingerprint", "bad")):
            data = plan_document(); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        # Each over-limit proposal is otherwise locally valid, so another error cannot mask the bound.
        data = plan_document()
        data["actions"] += [{**copy.deepcopy(data["actions"][0]), "id": f"action_{i}"} for i in range(32)]
        data["budget"]["max_tool_calls"] = 33
        with self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        data = plan_document()
        data["expected_inputs"] += [{"id": f"input_{i}", "fingerprint": DIGEST, "value_type": "string"} for i in range(127)]
        with self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        data = plan_document()
        data["actions"][0]["arguments"] = [{"name": f"arg_{i}", "value": {"kind": "input", "ref_id": "project", "value_type": "string"}} for i in range(33)]
        with self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        for policy in ({"on_unknown": "retry"}, {"compensation": "undo"}, {"max_retries": 4}):
            data = plan_document(); data["policy"] = policy
            with self.subTest(policy=policy), self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        for key, value in (("max_tool_calls", 0), ("max_tool_calls", 101), ("max_seconds", -1), ("max_cost_eur", float('inf'))):
            data = plan_document(); data["budget"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError): TaskExecutionPlan.read(data)
        data = plan_document(); data["actions"][0]["arguments"] *= 2
        with self.assertRaises(ValidationError): TaskExecutionPlan.read(data)

    def test_nested_collections_and_bindings_are_immutable(self):
        data = plan_document(); plan = TaskExecutionPlan.read(data)
        data["actions"][0]["arguments"].clear()
        self.assertEqual(len(plan.actions[0].arguments), 2)
        with self.assertRaises((AttributeError, TypeError)): plan.actions.append(plan.actions[0])
        with self.assertRaises(ValidationError): plan.actions[0].arguments[0].value.ref_id = "changed"

    def test_json_schema_refuses_expression_arguments_and_missing_version(self):
        import jsonschema
        from dclab_core import schemas
        validator = jsonschema.Draft202012Validator(schemas.generate()["task_execution_plan"])
        validator.validate(plan_document())
        missing = plan_document(); del missing["schema_version"]
        with self.assertRaises(jsonschema.ValidationError): validator.validate(missing)
        malicious = plan_document()
        malicious["actions"][0]["arguments"][0]["value"] = {"kind": "expression", "expression": "eval(user_text)"}
        with self.assertRaises(jsonschema.ValidationError): validator.validate(malicious)

    def test_json_schema_preserves_nonexecuting_status_and_ready_requirements(self):
        import jsonschema
        from dclab_core import schemas
        validator = jsonschema.Draft202012Validator(schemas.generate()["task_execution_plan"])
        mutations = []
        for status in ("blocked", "noop"):
            bad = plan_document(); bad["status"] = status; mutations.append(bad)
            bad = plan_document(); bad.update(status=status, actions=[], selected_capabilities=[], declared_outputs=[], reason=None)
            mutations.append(bad)
        for field in ("actions", "acceptance_checks"):
            bad = plan_document(); bad[field] = []; mutations.append(bad)
        for index, data in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(jsonschema.ValidationError): validator.validate(data)

    def test_schema_baseline_registration_and_adapter_conformance(self):
        from dclab_core import schemas
        from dclab_core.adapters import conformance
        schema = schemas.generate()["task_execution_plan"]
        directory = Path(schemas.SCHEMA_DIR)
        self.assertEqual(json.loads((directory / "task_execution_plan.json").read_text()), schema)
        self.assertEqual(json.loads((directory / "baseline" / (SCHEMA + ".json")).read_text()), schemas.normalize(schema))
        self.assertIn("schema_version", schema["required"])
        self.assertEqual(conformance.problems(), [])
        for name in ("task_graph", "task_execution_plan"):
            self.assertTrue(conformance.CONTRACT_ADAPTERS[name].startswith("library output"))
