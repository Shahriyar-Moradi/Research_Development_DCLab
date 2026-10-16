"""23.12: the conformance gate. Every contract, event kind, decision point and id has a mapping in an adapter or a listed reason; the
gate fails when a package adds one without either; every package of phase 1 has its integration section; and nothing in the library
imports the studio."""

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import experience as x, ledger as L
    from dclab_core.adapters import conformance, main_product_experience as mp
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class GateTest(unittest.TestCase):
    def test_the_adapters_are_conformant_at_the_commit_their_field_lists_were_read(self):
        self.assertEqual(conformance.problems(), [])
        from dclab_core.adapters.main_product import recorded

        self.assertRegex(recorded()["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(conformance.summary()["commit"], recorded()["commit"])

    def test_a_contract_with_no_adapter_and_no_reason_fails_the_gate(self):
        with mock.patch.dict(conformance.CONTRACT_ADAPTERS, {}, clear=False):
            conformance.CONTRACT_ADAPTERS.pop("outcome")
            self.assertTrue(any("contract outcome" in p for p in conformance.problems()))
        with mock.patch.dict(conformance.CONTRACT_ADAPTERS, {"outcome": "dclab_core.adapters.nowhere:nothing"}):
            self.assertTrue(any("adapter dclab_core.adapters.nowhere:nothing does not exist" in p for p in conformance.problems()))
        with mock.patch.dict(conformance.CONTRACT_ADAPTERS, {"a_contract_that_is_not_one": "library output"}):
            self.assertTrue(any("not a contract" in p for p in conformance.problems()))

    def test_a_value_that_is_neither_an_existing_adapter_nor_a_library_reason_fails_the_gate(self):
        for bad in ("TODO", "someday", "dclab_core.adapters.main_product_experience:with_trace_that_is_not_there"):
            with mock.patch.dict(conformance.CONTRACT_ADAPTERS, {"outcome": bad}):
                self.assertTrue(any("contract outcome" in p for p in conformance.problems()), bad)
        with mock.patch.object(conformance, "FROM_ROWS", conformance.FROM_ROWS | {"pipeline_not_a_contract"}):
            self.assertTrue(any("FROM_ROWS names pipeline_not_a_contract" in p for p in conformance.problems()))
        with mock.patch.object(conformance, "FROM_ROWS", conformance.FROM_ROWS | {"data_card"}):  # a library output has no adapter that reads rows
            self.assertTrue(any("FROM_ROWS names data_card" in p for p in conformance.problems()))

    def test_the_summary_says_which_adapters_read_rows_and_which_map_out(self):
        summary = conformance.summary()
        self.assertEqual(summary["contracts_read_from_rows"] + summary["contracts_mapped_out"] + summary["contracts_library_only"], summary["contracts"])
        self.assertEqual(summary["contracts_read_from_rows"], 6)
        self.assertEqual(conformance.CONTRACT_ADAPTERS["data_findings"], "dclab_core.adapters.main_product:problem_profile")  # the function that reads findings

    def test_a_kind_an_adapter_produces_is_not_counted_missing_and_a_point_read_as_a_record_needs_no_reason(self):
        self.assertIn("feedback.given", conformance.produced_kinds())
        self.assertEqual(conformance.summary()["kinds_missing"], len(set(mp.KINDS_MISSING) - conformance.produced_kinds()))
        from dclab_core.adapters.main_product_ledger import DECISION_TYPE_POINTS

        self.assertEqual(set(DECISION_TYPE_POINTS.values()) & set(conformance.POINTS_NOT_MAPPED), set())  # read by from_decision_record
        with mock.patch.dict(conformance.POINTS_NOT_MAPPED, {"model.promote": "no name"}):
            self.assertTrue(any("model.promote is both mapped and listed as not mapped" in p for p in conformance.problems()))

    def test_no_adapter_fills_the_parent_id_so_the_gate_lists_it(self):
        self.assertIn("parent", mp.IDS_MISSING)
        with mock.patch.dict(mp.IDS_MISSING, {}, clear=False):
            mp.IDS_MISSING.pop("parent")
            self.assertTrue(any("id parent" in p for p in conformance.problems()))

    def test_a_kind_a_point_and_an_id_with_neither_a_mapping_nor_a_reason_fail_it(self):
        with mock.patch.dict(x.KINDS, {"x.new_kind": {"a": "meta"}}):
            self.assertTrue(any("event kind x.new_kind" in p for p in conformance.problems()))
        with mock.patch.dict(mp.KINDS_MISSING, {"outcome.observed": " "}):
            self.assertTrue(any("event kind outcome.observed" in p for p in conformance.problems()))
        with mock.patch.dict(conformance.POINTS_NOT_MAPPED, {}, clear=False):
            conformance.POINTS_NOT_MAPPED.pop("report.claim")
            self.assertTrue(any("decision point report.claim" in p for p in conformance.problems()))
        with mock.patch.dict(mp.IDS_MISSING, {}, clear=False):
            mp.IDS_MISSING.pop("trace")
            self.assertTrue(any("id trace" in p for p in conformance.problems()))
        with mock.patch.dict(mp.KINDS_MISSING, {"not.a.kind": "reason"}):
            self.assertTrue(any("not a kind" in p for p in conformance.problems()))

    def test_every_finished_package_has_its_section_in_the_integration_guide(self):
        text = (ROOT / "docs" / "guides" / "CORE_INTEGRATION.md").read_text(encoding="utf-8")
        headings = {h for h in re.findall(r"^## .*\(((?:23|24|25)\.\d+)\)\s*$", text, re.M)}
        phase_one = {f"23.{n}" for n in range(1, 13)}
        self.assertTrue(phase_one <= headings, f"no section for {sorted(phase_one - headings)}")
        # a package of phase 2 is finished when the plan says "done" in its status: it has a section too
        plan = (ROOT / "docs" / "guides" / "CAPTURE_AND_PROBLEM_SOLVING_PLAN.md").read_text(encoding="utf-8")
        sections = re.split(r"^### (?=\d+\.\d+ )", plan, flags=re.M)[1:]
        finished = {section.split(" ", 1)[0] for section in sections if section.startswith("24.") and "**Status: done" in section.split("\n## ", 1)[0]}
        phase3 = (ROOT / "docs" / "guides" / "PHASE3_AI_FIRST_PLAN.md").read_text(encoding="utf-8")
        finished |= {section.split(" ", 1)[0] for section in re.split(r"^### (?=\d+\.\d+ )", phase3, flags=re.M)[1:]
                     if section.startswith("25.") and "**Status: done" in section.split("\n## ", 1)[0]}
        self.assertEqual(finished - headings, set(), f"finished packages with no section: {sorted(finished - headings)}")
        self.assertIn("24.1", finished)
        self.assertIn("25.1", finished)

    def test_the_library_imports_nothing_of_the_studio_and_the_wheel_check_exercises_the_new_modules(self):
        import ast

        offenders = []
        for path in (ROOT / "dclab_core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
                if any(n == "dclab_rnd" or n.startswith("dclab_rnd.") for n in names):
                    offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])
        wheel_check = (ROOT / "scripts" / "core_wheel_check.py").read_text(encoding="utf-8")
        for module in ("catalog", "errors", "feedback", "outcomes", "queries", "sensitivity", "steps", "tracing", "conformance"):
            self.assertIn(module, wheel_check, f"the wheel check does not exercise dclab_core.{module}")


if __name__ == "__main__":
    unittest.main()
