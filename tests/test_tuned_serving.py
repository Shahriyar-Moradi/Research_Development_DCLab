"""Package 27.8: the owner's tuned model serves a role only the routing's own way: it shadows first (on this machine),
the plan says what is missing, promotion is refused without the harness suite and, for an understanding role, a
measurement showing it better on the same conversations; retiring it restores the role's own model; every change is audited."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import audit
    from dclab_rnd.models import routing, tuned
    from dclab_rnd.models.routing import RoutingError
    from tests.test_models_gateway import CLEAN
except ImportError as error:  # the agent environment has no pydantic contracts
    MISSING = str(error)
else:
    MISSING = ""

KEY = "sk-" + "test-" + "z" * 8
ENV = {"DCLAB_LLM_PROVIDER": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": KEY, "OPENAI_MODEL": "gpt-6-luna",
       "DCLAB_TIER_TUNED_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_TIER_TUNED_MODEL": "dclab-solver-8b", "DCLAB_DATABASE_URL": ""}


@unittest.skipIf(MISSING, MISSING)
class TunedServingTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, **ENV})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.audit = audit.open_audit(self.home)

    def suite(self, passed):
        return mock.patch("dclab_rnd.agent_eval.regression.passed_with", return_value={"id": "HRN-T"} if passed else None)

    def shadowed(self, n=3, purpose="reporter"):
        with (self.home / "model_shadow.jsonl").open("a") as f:
            for i in range(n):
                f.write(json.dumps({"at": f"2026-10-17T00:00:0{i}", "purpose": purpose, "shadow_model": "dclab-solver-8b", "shadow_tier": "tuned",
                                    "same_tool": i != 0, "same_arguments": True, "outcome": "ok"}) + "\n")

    def test_the_way_to_serving_shadow_then_suite_then_promote_then_retire(self):
        with self.suite(False):
            first = tuned.plan("reporter", self.home)
        self.assertFalse(first["ready"])
        self.assertTrue(any("shadow" in m for m in first["missing"]))
        tuned.shadow("reporter", self.home, self.audit)
        self.shadowed()
        with self.suite(False):
            mid = tuned.plan("reporter", self.home)
            self.assertEqual(mid["shadow_agreement"]["requests"], 3)
            self.assertEqual(len(mid["missing"]), 1)
            self.assertIn("harness suite", mid["missing"][0])
            with self.assertRaises(RoutingError):  # no suite passed with it: refused
                tuned.promote("reporter", self.home, "it agreed in shadow", audit_log=self.audit)
        with self.suite(True):
            self.assertTrue(tuned.plan("reporter", self.home)["ready"])
            doc = tuned.promote("reporter", self.home, "shadowed 3 requests, suite passed", audit_log=self.audit)
        self.assertEqual(routing.route(doc, "reporter")["serve"], "tuned")
        self.assertEqual(routing.route(doc, "reporter")["serve_model"], "dclab-solver-8b")
        doc = tuned.retire("reporter", self.home, "back to the standard model", self.audit)
        self.assertIsNone(routing.route(doc, "reporter")["serve"])
        moves = [r for r in self.audit.query(100, 0)[1] if r.get("move") == "set_routing"]
        self.assertEqual(len(moves), 3)  # shadow, promote, retire: each one audited row

    def test_an_understanding_role_needs_to_have_measured_better(self):
        tuned.shadow("conversation_reader", self.home, self.audit)
        self.shadowed(purpose="conversation_reader")
        with self.suite(True):
            plan = tuned.plan("conversation_reader", self.home)
            self.assertTrue(plan["ready"] and plan["needs_measurement"])
            self.assertEqual(plan["missing"], [])
            with self.assertRaises(RoutingError):
                tuned.promote("conversation_reader", self.home, "cheaper", audit_log=self.audit)
            with self.assertRaises(RoutingError):  # measured, but not better than the model serving it now
                tuned.promote("conversation_reader", self.home, "measured", {"dclab-solver-8b": 0.80, "gpt-6-luna": 0.83}, self.audit)
            doc = tuned.promote("conversation_reader", self.home, "measured on the dev conversations",
                                {"dclab-solver-8b": 0.87, "gpt-6-luna": 0.83}, self.audit)
        self.assertEqual(routing.route(doc, "conversation_reader")["serve"], "tuned")
        evidence = doc["history"][-1]["evidence"]
        self.assertEqual(evidence["measured"], {"dclab-solver-8b": 0.87, "gpt-6-luna": 0.83})

    def test_an_unconfigured_or_remote_tuned_tier_is_said_and_cannot_shadow(self):
        with mock.patch.dict(os.environ, {"DCLAB_TIER_TUNED_MODEL": ""}), self.suite(True):
            self.assertIn("not configured", tuned.plan("reporter", self.home)["missing"][0])
        with mock.patch.dict(os.environ, {"DCLAB_TIER_TUNED_BASE_URL": "https://tuned.example/v1", "DCLAB_TIER_TUNED_API_KEY": KEY}):
            with self.suite(True):
                self.assertIn("not on this machine", tuned.plan("reporter", self.home)["missing"][0])
            with self.assertRaises(RoutingError):
                tuned.shadow("reporter", self.home, self.audit)

    def test_promotion_without_a_shadow_is_refused(self):  # reviewer finding 1
        with self.suite(True):
            with self.assertRaises(RoutingError) as refused:
                tuned.promote("reporter", self.home, "never shadowed", audit_log=self.audit)
        self.assertIn("shadow", str(refused.exception))
        self.assertIsNone(routing.route(routing.open_routing(self.home).get(), "reporter")["serve"])

    def test_a_shadow_that_only_failed_is_not_ready(self):
        tuned.shadow("reporter", self.home, self.audit)
        with (self.home / "model_shadow.jsonl").open("a") as f:
            f.write(json.dumps({"at": "2026-10-17T00:00:00", "purpose": "reporter", "shadow_model": "dclab-solver-8b", "shadow_tier": "tuned",
                                "same_tool": None, "outcome": "error"}) + "\n")
        with self.suite(True):
            plan = tuned.plan("reporter", self.home)
        self.assertFalse(plan["ready"])
        self.assertTrue(any("failed" in m for m in plan["missing"]))

    def test_a_swapped_tuned_model_is_not_reported_as_serving(self):  # reviewer finding 2
        tuned.shadow("reporter", self.home, self.audit)
        self.shadowed()
        with self.suite(True):
            tuned.promote("reporter", self.home, "shadowed and suite passed", audit_log=self.audit)
            self.assertTrue(tuned.plan("reporter", self.home)["serving"])
            with mock.patch.dict(os.environ, {"DCLAB_TIER_TUNED_MODEL": "dclab-solver-8b-v2"}):
                plan = tuned.plan("reporter", self.home)
        self.assertFalse(plan["serving"] or plan["ready"])
        self.assertIn("approval was for dclab-solver-8b", plan["missing"][0])

    def test_retiring_puts_back_the_model_that_served_before(self):  # reviewer finding 3
        with mock.patch.dict(os.environ, {"DCLAB_TIER_STRONG_MODEL": "big-model", "DCLAB_TIER_STRONG_API_KEY": KEY}), self.suite(True):
            routing.switch(routing.open_routing(self.home), self.audit, "reporter", "serve", "strong", "reviewer", "writes better")
            tuned.shadow("reporter", self.home, self.audit)
            self.shadowed()
            tuned.promote("reporter", self.home, "shadowed and suite passed", audit_log=self.audit)
            doc = tuned.retire("reporter", self.home, "the tuned model wrote worse", self.audit)
        self.assertEqual((routing.route(doc, "reporter")["serve"], routing.route(doc, "reporter")["serve_model"]), ("strong", "big-model"))
        with self.suite(True):  # with no earlier model, the role's own tier
            doc = tuned.retire("reporter", self.home, "again", self.audit)
        self.assertEqual(routing.route(doc, "reporter")["serve"], "strong")  # not serving the tuned model: nothing to retire

    def test_a_malformed_measurement_is_refused_not_a_traceback(self):  # reviewer finding 5
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home)}), mock.patch("builtins.print") as out:
            self.assertEqual(tuned.main(["promote", "reporter", "--reason", "x", "--measured", "dclab-solver"]), 2)
        self.assertIn("model=score", out.call_args[0][0])

    def test_the_command_refuses_with_its_reason(self):
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home)}), self.suite(False), mock.patch("builtins.print") as out:
            self.assertEqual(tuned.main(["promote", "reporter", "--reason", "try"]), 2)
        self.assertIn("refused", out.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
