"""Unit tests for P2-P6 code-owned catalog, ProblemDecomposition, router, and bounded re-planning."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.task_catalog import (  # noqa: E402
    CATALOG,
    CatalogTaskCall,
    validate_catalog_plan,
    validate_decomposition,
)
from dclab_rnd.tasks.catalog import (  # noqa: E402
    compile_catalog_to_task_graph,
    deterministic_catalog_plan,
    execute_catalog_task,
    grounding_context,
    needs_model_planning,
    plan_with_model,
)


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, *args, **kwargs):
        txt = self._replies[self.calls]
        self.calls += 1
        return {"content": txt, "cost_eur": 0.002}


class FakeGateway:
    def __init__(self, client):
        self._client = client

    def client(self, purpose, project_id=None):
        return self._client


class TaskCatalogTests(unittest.TestCase):
    def test_catalog_rejects_hallucinated_columns_and_post_holdout_tasks(self):
        tables = {"subscribers.csv": ["subscriber_id", "signup_time", "churned"]}
        _, issues_col = validate_catalog_plan(
            [{"task_type": "leakage_probe", "params": {"table": "subscribers.csv", "column": "nonexistent_col"}}],
            tables,
        )
        self.assertTrue(any("nonexistent_col" in m for m in issues_col))

        _, issues_order = validate_catalog_plan(
            [
                {"task_type": "holdout", "params": {}},
                {"task_type": "time_split_check", "params": {"table": "subscribers.csv", "column": "signup_time"}},
            ],
            tables,
        )
        self.assertTrue(any("after holdout" in m for m in issues_order))

    def test_router_sends_standard_cases_to_deterministic_zero_cost(self):
        tables = {"listings.csv": ["listing_id", "bedrooms", "sqft", "nightly_price"]}
        facts = {"target": "nightly_price", "horizon_days": 0, "unit": "listing", "clean_single_table": True}
        self.assertFalse(needs_model_planning("Predict nightly price cleanly.", tables, facts))
        out = plan_with_model("Predict nightly price cleanly.", tables, facts, gateway=FakeGateway(FakeClient([])))
        self.assertEqual(out["routed_to"], "deterministic")
        self.assertEqual(out["extra_prompts"], 0)
        self.assertEqual(out["cost_usd"], 0.0)

    def test_bounded_replan_triggers_when_check_join_key_finds_multi_row_risk(self):
        tables = {
            "customers.csv": ["customer_id", "signup_date", "plan"],
            "addresses.csv": ["customer_id", "city"],
        }
        facts = {"target": "inactive_30d", "horizon_days": 30, "unit": "customer"}
        self.assertTrue(needs_model_planning("Predict customer inactivity across tables.", tables, facts))
        cites = grounding_context("Predict customer inactivity across tables.", tables)
        cite_ids = sorted(cites)[:1] or ["PIT-005"]

        first_reply = json.dumps({
            "version": "tp08.v1",
            "decomposition": {
                "schema_version": "dclab_core.problem_decomposition.v1",
                "decision": "Predict 30-day customer inactivity safely without join duplication",
                "target": "inactive_30d",
                "unit": "customer",
                "horizon_days": 30,
                "success_metric": "roc_auc",
                "subproblems": [{"id": "sub_1", "kind": "entity_tables", "objective": "Customer inactivity"}],
                "risks": [{"trap_kind": "many_to_one_join", "table": "addresses.csv", "column": "customer_id", "recommended_task": "check_join_key"}],
                "unknowns": [],
                "source": "model",
            },
            "calls": [
                {"task_type": "check_join_key", "params": {"table": "addresses.csv", "key": "customer_id", "parent_table": "customers.csv"}, "reason": "Check 1:many join risk", "citations": cite_ids},
            ],
            "caught_traps": ["addresses_one_to_many_row_duplication"],
        })
        second_reply = json.dumps({
            "version": "tp08.v1",
            "calls": [
                {"task_type": "check_join_key", "params": {"table": "addresses.csv", "key": "customer_id", "parent_table": "customers.csv"}, "reason": "Check join", "citations": cite_ids},
                {"task_type": "grouped_split", "params": {"table": "customers.csv", "entity": "customer_id"}, "reason": "Keep customer grouped across split", "citations": cite_ids},
            ],
            "caught_traps": ["entity_leak_across_random_rows"],
        })
        res = plan_with_model(
            "Predict customer inactivity across tables.",
            tables,
            facts,
            gateway=FakeGateway(FakeClient([first_reply, second_reply])),
        )
        self.assertEqual(res["routed_to"], "model")
        self.assertEqual(res["revisions"], 2)
        self.assertEqual(res["extra_prompts"], 2)
        self.assertIn("grouped_split", res["selected_tasks"])
        self.assertIn("check_join_key", res["selected_tasks"])
        self.assertIn("entity_leak_across_random_rows", res["caught_traps"])

    def test_schema_self_repair_and_task_graph_compilation(self):
        from dclab_core.solution_workflow import SolutionWorkflow, Step
        from dclab_core.task_fingerprints import graph_fingerprint

        tables = {"orders.csv": ["order_id", "ordered_at", "final_refund_amount", "returned"]}
        facts = {"target": "returned", "horizon_days": 30, "unit": "order", "time_column": "ordered_at"}
        bad_first = json.dumps({
            "version": "tp08.v1",
            "decomposition": {
                "schema_version": "dclab_core.problem_decomposition.v1",
                "decision": "Predict returns cleanly",
                "target": "returned",
                "unit": "order",
                "horizon_days": 30,
                "success_metric": "roc_auc",
                "subproblems": [{"id": "s1", "kind": "classification", "objective": "Predict order returns"}],
                "source": "model",
            },
            "calls": [{"task_type": "leakage_probe", "params": {"table": "orders.csv", "column": "missing_col"}}],
        })
        fixed_second = json.dumps({
            "version": "tp08.v1",
            "decomposition": {
                "schema_version": "dclab_core.problem_decomposition.v1",
                "decision": "Predict returns cleanly",
                "target": "returned",
                "unit": "order",
                "horizon_days": 30,
                "success_metric": "roc_auc",
                "subproblems": [{"id": "s1", "kind": "classification", "objective": "Predict order returns"}],
                "source": "model",
            },
            "calls": [
                {"task_type": "leakage_probe", "params": {"table": "orders.csv", "column": "final_refund_amount"}},
                {"task_type": "time_split_check", "params": {"table": "orders.csv", "column": "ordered_at"}},
            ],
            "caught_traps": ["post_outcome_final_refund_amount"],
        })
        res = plan_with_model("Predict returns.", tables, facts, gateway=FakeGateway(FakeClient([bad_first, fixed_second])))
        self.assertEqual(res["routed_to"], "model")
        self.assertEqual(res["revisions"], 2)
        wf = SolutionWorkflow(family="classification", steps=[Step(id="data", block="prepare", name="Data", purpose="p", check="c", rule="r")])
        tg = compile_catalog_to_task_graph("proj_1", wf, res["decomposition"], res["calls"])
        self.assertEqual(tg.fingerprint, graph_fingerprint(tg))
        self.assertGreaterEqual(len(tg.nodes), 3)

    def test_fallback_preserves_b1_trap_floor_and_null_coercion(self):
        from dclab_core.task_catalog import validate_decomposition
        from dclab_rnd.tasks.catalog import CatalogTaskCall, execute_catalog_task

        tables = {"customers.csv": ["customer_id", "signup_date", "last_order_date", "inactive_30d"]}
        facts = {"target": "inactive_30d", "horizon_days": 30, "unit": "customer", "time_column": "signup_date"}

        # Optional null metadata and overlong descriptive strings coerce/clip cleanly instead of refusing valid plans.
        long_cite = "KND-014 · Uplift: who responds to treatment vs propensity; target selection action goes where it matters most."
        decomp, issues = validate_decomposition({
            "decision": "Predict 30d inactivity safely",
            "target": "inactive_30d",
            "unit": "customer",
            "horizon_days": 30,
            "success_metric": None,
            "subproblems": [{"id": "s1", "kind": "classification", "objective": "Predict inactivity"}],
            "risks": [{"trap_kind": "end_of_period_last_order_date", "table": "customers.csv", "column": "last_order_date",
                       "evidence_citation": None, "recommended_task": "leakage_probe"},
                      {"trap_kind": "uplift_vs_propensity", "table": "customers.csv", "column": None,
                       "evidence_citation": long_cite, "recommended_task": "segment_subproblem"}],
            "constraints": None,
        }, tables)
        self.assertEqual(issues, [])
        self.assertEqual(decomp.risks[0].evidence_citation, "")
        self.assertLessEqual(len(decomp.risks[1].evidence_citation), 80)
        self.assertEqual(decomp.success_metric, "roc_auc")

        # Model exception/outage and refused schema both keep B1's caught traps and subproblems.
        out_fail = plan_with_model("Predict customer inactivity.", tables, facts, gateway=FakeGateway(FakeClient(["not json", "still bad"])))
        self.assertEqual(out_fail["routed_to"], "fallback_invalid_json")
        self.assertTrue(any("last_order_date" in t for t in out_fail["caught_traps"]))
        self.assertIn("time_split_required", out_fail["caught_traps"])
        ev = execute_catalog_task(CatalogTaskCall(task_type="time_split_check", params={"column": "signup_date"}), tables=tables, known_facts=facts)
        self.assertIn("temporal_drift_regime_risk", ev["traps"])
        from dclab_rnd.tasks.catalog import _derive_subproblems
        self.assertEqual(_derive_subproblems(decomp, [CatalogTaskCall(task_type="segment_subproblem", params={"kind": "rare_event"})]), ["classification", "rare_event"])


if __name__ == "__main__":
    unittest.main()
