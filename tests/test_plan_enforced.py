"""Hierarchical Planning, stage H0 (2026-11-16): did the engine obey the plan? A trap the plan catches and the model then
falls into is a failure, whatever the report says. Pure code on stored records; no model, no table."""
import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.tasks import enforced as E  # noqa: E402

COLUMNS = ["customer_id", "signup_date", "monthly_charge", "closure_recorded_after_review", "churn"]


def stages(split="stratified random 36 of 180 rows", features=("closure_recorded_after_review=1", "monthly_charge"), final=True):
    out = {"data": {"evidence": {"holdout_split": split}}}
    if final:
        out["final"] = {"evidence": {"configuration_results": [{"top_features": [{"feature": f} for f in features]}]}}
    return out


class ReadingTests(unittest.TestCase):
    def test_a_trap_is_read_by_its_kind_and_the_column_it_names(self):
        self.assertEqual((E.kind_of("post_outcome_closure_recorded_after_review"), E.column_of("post_outcome_closure_recorded_after_review", COLUMNS)),
                         ("forbid", "closure_recorded_after_review"))
        self.assertEqual((E.kind_of("grouped_split_customer_id"), E.column_of("grouped_split_customer_id", COLUMNS)), ("group", "customer_id"))
        self.assertEqual(E.kind_of("time_split_required"), "time_split")
        self.assertIsNone(E.kind_of("asymmetric_cost"))
        self.assertIsNone(E.column_of("post_outcome", COLUMNS))  # a trap that names no column cannot be enforced on one

    def test_the_final_models_columns_are_read_through_their_encoded_names(self):
        self.assertEqual(E.final_features(stages()), ["closure_recorded_after_review", "monthly_charge"])


class ObeyedTests(unittest.TestCase):
    def one(self, trap, solution=None, **kw):
        return E.obeyed([trap], solution or {}, stages(**kw), COLUMNS)[0]

    def test_a_column_known_after_the_outcome_must_be_forbidden_and_out_of_the_final_model(self):
        trap = "post_outcome_closure_recorded_after_review"
        self.assertFalse(self.one(trap)["enforced"])  # caught, and the top feature of the final model
        self.assertFalse(self.one(trap, {"forbidden": ["closure_recorded_after_review"]})["enforced"])  # forbidden on paper, still used
        self.assertTrue(self.one(trap, {"forbidden": ["closure_recorded_after_review"]}, features=("monthly_charge",))["enforced"])
        self.assertFalse(self.one(trap, {}, features=("monthly_charge",))["enforced"])  # absent by luck is not enforced: nothing forbids it

    def test_a_group_trap_needs_the_split_to_keep_the_group_on_one_side(self):
        self.assertFalse(self.one("grouped_split_customer_id")["enforced"])
        self.assertTrue(self.one("grouped_split_customer_id", {"group_column": "customer_id"})["enforced"])
        self.assertTrue(self.one("repeated_entity_leak_customer_id", split="group (declared group column customer_id)")["enforced"])

    def test_a_time_trap_needs_a_split_by_time(self):
        self.assertFalse(self.one("time_split_required")["enforced"])
        self.assertTrue(self.one("time_split_required", split="time (declared time column): the latest 36 rows")["enforced"])
        self.assertIsNone(E.obeyed(["time_split_required"], {}, {}, COLUMNS)[0]["enforced"])  # the data stage has not run: not read, never a pass

    def test_a_trap_code_cannot_enforce_is_never_counted_as_enforced(self):
        record = self.one("asymmetric_cost")
        self.assertEqual((record["enforceable"], record["enforced"]), (False, None))
        counted = E.summary(E.obeyed(["asymmetric_cost", "time_split_required", "post_outcome_closure_recorded_after_review"], {}, stages(), COLUMNS))
        self.assertEqual((counted["caught"], counted["enforceable"], counted["enforced"], counted["rate"]), (3, 2, 0, 0.0))
        self.assertEqual(counted["broken"], ["time_split_required", "post_outcome_closure_recorded_after_review"])


class StoredRunTests(unittest.TestCase):
    """The baseline, from a live run kept before this work (2026-11-13): every scenario "passed", and no trap the plan
    caught was enforced. Skipped where the kept run is not on disk (a clean checkout)."""

    RUN = ROOT / "agent_runs" / "live_tests" / "adaptive_studio_live_20261113_r4"

    @unittest.skipUnless((RUN / "summary.json").exists(), "the kept live run is not on this machine")
    def test_the_kept_run_caught_the_leak_and_used_it(self):
        scenario = next(s for s in json.loads((self.RUN / "summary.json").read_text())["scenarios"] if s["scenario_id"] == "studio_multitable_post_outcome")
        folder = self.RUN / "projects" / scenario["project_id"]
        solution = json.loads((folder / "project.json").read_text()).get("solution") or {}
        kept = {p.stem: json.loads(p.read_text()) for p in (folder / "stages").glob("*.json")}
        columns = [f["feature"] for f in kept["data"]["evidence"]["feature_profiles"]] + ["closure_recorded_after_review"]
        traps = scenario["caught_traps"] if isinstance(scenario["caught_traps"], list) else ast.literal_eval(scenario["caught_traps"])
        counted = E.summary(E.obeyed(traps, solution, kept, columns))
        self.assertTrue(scenario["passed"])
        self.assertEqual((counted["enforceable"], counted["enforced"]), (4, 0))
        self.assertIn("post_outcome_closure_recorded_after_review", counted["broken"])


class GateTests(unittest.TestCase):
    def project(self, read=2, enforced=2, stopped=False):
        return {"summary": {"read": read, "enforced": enforced}, "stopped_with_question": stopped}

    def test_the_gate_is_pinned_before_the_repair(self):
        from dclab_rnd.agent_eval.plan_enforced import gate

        self.assertIsNone(gate.pinned())
        self.assertEqual((gate.TERMS["obeyed"], gate.TERMS["people"], gate.TERMS["min_projects"]), (1.0, "dev3", 8))

    def test_every_caught_trap_must_be_enforced_or_the_build_must_have_stopped(self):
        from dclab_rnd.agent_eval.plan_enforced import gate

        self.assertTrue(gate.obeyed_check([self.project()] * 8)["ok"])
        self.assertFalse(gate.obeyed_check([self.project()] * 7 + [self.project(enforced=1)])["ok"])  # one trap not enforced in one project
        self.assertTrue(gate.obeyed_check([self.project()] * 7 + [self.project(enforced=0, stopped=True)])["ok"])  # it stopped and asked
        self.assertFalse(gate.obeyed_check([self.project()] * 4)["ok"])  # too few projects
        unread = gate.obeyed_check([self.project(read=0, enforced=0)] * 8)
        self.assertEqual((unread["ok"], unread["read"]), (False, False))  # nothing enforceable was read: never a pass

    def test_the_stored_baseline_fails_the_check_it_was_stored_for(self):
        path = ROOT / "evidence" / "campaigns" / "plan_enforced_v1" / "results" / "PE-001.json"
        doc = json.loads(path.read_text())
        self.assertEqual((doc["totals"]["reported_passed"], doc["totals"]["enforceable"], doc["totals"]["enforced"]), (5, 8, 0))
        self.assertFalse(doc["check_obeyed"]["ok"])
        self.assertEqual(doc["source"]["live_model_calls"], 0)
        paused = next(p for p in doc["projects"] if p["paused_with_question"])
        self.assertFalse(paused["stopped_with_question"])  # it asked, went on and built: that is not stopping


if __name__ == "__main__":
    unittest.main()
