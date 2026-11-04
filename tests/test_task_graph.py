"""TP-01B: structural task contracts, with no scheduler or execution authority."""
import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from dclab_core.task_graph import TaskGraph, TaskNode

SCHEMA = "dclab_core.task_graph.v1"
DIGEST = "a" * 64


def graph_document():
    return {
        "schema_version": SCHEMA, "graph_id": "churn", "revision": 1,
        "objective": "Understand churn before retention action",
        "source_refs": [{"kind": "problem_card", "id": "problem", "fingerprint": "b" * 16}],
        "workflow_fingerprint": "c" * 16, "solution_fingerprint": DIGEST,
        "fingerprint": DIGEST,
        "provenance": {"method": "deterministic", "reference_id": "adapter", "producer_version": "v1"},
        "nodes": [
            {"id": "prepare", "kind": "group", "title": "Understand data", "objective": "Describe available inputs"},
            {"id": "data", "parent_id": "prepare", "kind": "action", "title": "Run data stage",
             "objective": "Prepare the approved split and training summaries", "source_step_ids": ["lock_holdout", "train_only_analysis"],
             "needs": ["table"], "produces": ["data_record"], "scope": "project",
             "capability": {"name": "run_stage", "scope": "project", "schema_fingerprint": DIGEST},
             "acceptance_checks": [{"id": "data_exists", "kind": "artifact", "description": "Current data record exists", "evidence_refs": ["data_record"]}]},
            {"id": "copies", "kind": "blocked", "title": "Isolate near copies", "objective": "Keep copies on one side",
             "blocker_reason": "Existing detection does not prove isolation"},
        ],
        "edges": [{"source": "prepare", "target": "data", "kind": "grouping"}],
    }


class TaskGraphTests(unittest.TestCase):
    def test_minimal_example_and_unsupported_node_roundtrip(self):
        graph = TaskGraph.read(graph_document())
        self.assertEqual(TaskGraph.read(graph.model_dump_json()), graph)
        self.assertEqual(graph.nodes[2].kind, "blocked")
        self.assertIsNone(graph.nodes[2].capability)
        self.assertEqual(graph.edges[0].kind, "grouping")

    def test_missing_or_unknown_version_is_refused(self):
        for version in (None, "dclab_core.task_graph.v2"):
            data = graph_document()
            if version is None:
                del data["schema_version"]
            else:
                data["schema_version"] = version
            with self.subTest(version=version), self.assertRaises((ValueError, ValidationError)):
                TaskGraph.read(data)
            with self.assertRaises(ValidationError):
                TaskGraph.model_validate(data)

    def test_unknown_fields_are_refused_at_every_level(self):
        for location in ("root", "node", "capability", "check", "source", "provenance", "edge"):
            data = graph_document()
            obj = {"root": data, "node": data["nodes"][1], "capability": data["nodes"][1]["capability"],
                   "check": data["nodes"][1]["acceptance_checks"][0], "source": data["source_refs"][0],
                   "provenance": data["provenance"], "edge": data["edges"][0]}[location]
            obj["execute"] = "arbitrary expression"
            with self.subTest(location=location), self.assertRaises(ValidationError):
                TaskGraph.read(data)

    def test_bad_ids_enums_blank_text_and_bounds_are_refused(self):
        for path, value in (("graph_id", "https://example.com"), ("graph_id", "x" * 65), ("objective", "  "),
                            ("revision", True), ("revision", 0), ("revision", 1_000_001), ("fingerprint", "bad")):
            data = graph_document(); data[path] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValidationError):
                TaskGraph.read(data)
        for key, value in (("id", "../file"), ("kind", "python"), ("scope", "admin"), ("objective", "\n")):
            node = copy.deepcopy(graph_document()["nodes"][1]); node[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                TaskNode.model_validate(node)

    def test_duplicate_ids_and_overlong_collections_are_refused(self):
        for key, items in (("nodes", [graph_document()["nodes"][0]] * 2),
                           ("nodes", [graph_document()["nodes"][0]] * 257),
                           ("edges", graph_document()["edges"] * 2), ("edges", graph_document()["edges"] * 1025)):
            data = graph_document(); data[key] = items
            with self.subTest(key=key, count=len(items)), self.assertRaises(ValidationError):
                TaskGraph.read(data)
        # Unique objects ensure removing a size bound cannot be masked by duplicate-ID rejection.
        data = graph_document()
        data["nodes"] = [{"id": f"group_{i}", "kind": "group", "title": "Group", "objective": "Organize work"} for i in range(257)]
        with self.assertRaises(ValidationError): TaskGraph.read(data)
        data = graph_document()
        data["edges"] = [{"source": "prepare", "target": f"node_{i}", "kind": "dependency"} for i in range(1025)]
        with self.assertRaises(ValidationError): TaskGraph.read(data)
        node = copy.deepcopy(graph_document()["nodes"][1]); node["needs"] = [f"input_{i}" for i in range(33)]
        with self.assertRaises(ValidationError): TaskNode.model_validate(node)
        for key in ("needs", "produces", "dependencies", "source_step_ids"):
            node = copy.deepcopy(graph_document()["nodes"][1]); node[key] = ["same", "same"]
            with self.subTest(key=key), self.assertRaises(ValidationError):
                TaskNode.model_validate(node)

    def test_only_actions_can_bind_a_capability(self):
        for kind in ("group", "gate", "verification", "blocked"):
            node = copy.deepcopy(graph_document()["nodes"][1]); node["kind"] = kind
            node["blocker_reason"] = "Unsupported requirement" if kind == "blocked" else None
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                TaskNode.model_validate(node)
        for missing in ("capability", "scope", "acceptance_checks"):
            node = copy.deepcopy(graph_document()["nodes"][1]); node.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValidationError):
                TaskNode.model_validate(node)
        node = copy.deepcopy(graph_document()["nodes"][1]); node["scope"] = "draft"
        with self.assertRaises(ValidationError): TaskNode.model_validate(node)

    def test_gate_and_check_only_nodes_need_no_tool(self):
        for kind in ("gate", "verification"):
            node = TaskNode(id="check", kind=kind, title="Confirm current evidence", objective="Check approved source",
                            acceptance_checks=[{"id": "approved", "kind": "approval", "description": "Current human approval exists"}])
            self.assertIsNone(node.capability)
        for reason in (None, "", "  "):
            node = copy.deepcopy(graph_document()["nodes"][2]); node["blocker_reason"] = reason
            with self.subTest(reason=reason), self.assertRaises(ValidationError): TaskNode.model_validate(node)

    def test_nested_collections_are_immutable_and_detached_from_input(self):
        document = graph_document(); graph = TaskGraph.read(document)
        document["nodes"][1]["needs"].append("changed")
        self.assertEqual(graph.nodes[1].needs, ("table",))
        with self.assertRaises((AttributeError, TypeError)): graph.nodes.append(graph.nodes[0])
        with self.assertRaises(ValidationError): graph.nodes[1].title = "changed"
        with self.assertRaises((AttributeError, TypeError)): graph.nodes[1].needs.append("changed")

    def test_json_schema_accepts_example_and_refuses_missing_version_and_extras(self):
        import jsonschema
        from dclab_core import schemas
        validator = jsonschema.Draft202012Validator(schemas.generate()["task_graph"])
        validator.validate(graph_document())
        for key, value in (("schema_version", None), ("unexpected", True)):
            data = graph_document()
            if value is None: del data[key]
            else: data[key] = value
            with self.subTest(key=key), self.assertRaises(jsonschema.ValidationError): validator.validate(data)

    def test_example_action_names_existing_registry_capability_without_invoking_it(self):
        from dclab_rnd.agents import default_registry
        registry = default_registry()
        for node in TaskGraph.read(graph_document()).nodes:
            if node.capability is not None:
                tool = registry.get(node.capability.name)
                self.assertIsNotNone(tool)
                self.assertEqual(tool.scope, node.scope)

    def test_json_schema_preserves_nonblank_text_and_nonexecuting_kinds(self):
        import jsonschema
        from dclab_core import schemas
        validator = jsonschema.Draft202012Validator(schemas.generate()["task_graph"])
        bad = graph_document(); bad["objective"] = "   "
        mutations = [bad]
        for kind in ("group", "gate", "verification", "blocked"):
            bad = graph_document(); bad["nodes"][1]["kind"] = kind
            if kind == "blocked": bad["nodes"][1]["blocker_reason"] = "Unsupported"
            mutations.append(bad)
        for key, value in (("scope", "draft"), ("acceptance_checks", []), ("capability", None)):
            bad = graph_document(); bad["nodes"][1][key] = value; mutations.append(bad)
        bad = graph_document(); bad["nodes"][2]["blocker_reason"] = None; mutations.append(bad)
        for index, data in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(jsonschema.ValidationError): validator.validate(data)

    def test_fresh_process_import_orders_register_both_contracts(self):
        import subprocess
        import sys
        for first, second in (("task_graph", "task_execution_plan"), ("task_execution_plan", "task_graph")):
            code = (f"import dclab_core.{first}; import dclab_core.{second}; "
                    "from dclab_core.contracts import CONTRACTS; "
                    "assert {'task_graph', 'task_execution_plan'} <= CONTRACTS.keys()")
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_published_schema_and_frozen_baseline_are_fresh(self):
        from dclab_core import schemas
        schema = schemas.generate()["task_graph"]
        directory = Path(schemas.SCHEMA_DIR)
        self.assertEqual(json.loads((directory / "task_graph.json").read_text()), schema)
        baseline = json.loads((directory / "baseline" / (SCHEMA + ".json")).read_text())
        self.assertEqual(baseline, schemas.normalize(schema))
        self.assertIn("schema_version", schema["required"])
