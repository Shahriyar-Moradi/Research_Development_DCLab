"""Step 3's measurement harness (dclab_rnd/agent_eval/roles): the gates and the planner and engineer paths with
scripted advisors. No model is called."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class RolesMeasureTest(unittest.TestCase):
    def test_the_planner_path_and_its_gate(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_rnd.agent_eval.roles import measure_planner, planner_verdict

        quiet = measure_planner(ScriptedAdvisor({"solution_planner": {"decisions": []}}))
        self.assertEqual(quiet["summary"]["planner"], quiet["summary"]["rules"])  # no change: the rules' score
        self.assertEqual(planner_verdict(quiet)["status"], "pass")
        self.assertEqual(planner_verdict(measure_planner(None))["status"], "not scored")

    def test_the_engineer_path_finds_the_shapes_and_measures_the_gain(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_rnd.agent_eval.roles import engineer_verdict, measure_engineer
        from dclab_rnd.expansion import runner

        from dclab_rnd.agent_eval.roles import _is_repeat, _is_short

        self.assertFalse(_is_short({"name": "x", "op": "count", "by": ["biker_id"], "of": {"below": [{"distance": ["source", "destination"]}, 50]}}))
        self.assertFalse(_is_repeat({"name": "x", "op": "count", "by": ["biker_id", {"place": "source"}]}))  # the pick-up, not the drop-off
        programs = [{"name": "drop_repeat", "op": "count", "by": ["biker_id", {"place": "destination"}]},
                    {"name": "short", "op": "mean", "of": {"below": [{"distance": ["source", "destination"]}, 60]}, "by": ["biker_id"]}]
        with runner.fast_profile():
            row = measure_engineer(ScriptedAdvisor({"feature_engineer": {"programs": programs}}))
        self.assertTrue(row["repeat_proposed"] and row["short_proposed"])
        self.assertGreater(row["gain_over_raw"], 0.05)
        self.assertEqual(engineer_verdict(row)["status"], "pass")
        self.assertEqual(engineer_verdict(measure_engineer(None))["status"], "not scored")

    def test_the_engineer_gate_v2_judges_the_gain_on_every_seed(self):
        from dclab_rnd.agent_eval.roles import GATES, engineer_v2_verdict

        self.assertEqual(GATES["engineer_v2"]["version"], "engineer_gate_v2")
        good = {"source": "advisor", "gain_over_raw": 0.3, "against_behaviour": 0.01}
        self.assertEqual(engineer_v2_verdict({"rows": [{**good, "seed": 1}, {**good, "seed": 2}]})["status"], "pass")
        below = {**good, "seed": 2, "against_behaviour": -0.02}  # worse than the behaviour recipe on one seed
        self.assertEqual(engineer_v2_verdict({"rows": [{**good, "seed": 1}, below]})["status"], "fail")
        self.assertEqual(engineer_v2_verdict({"rows": [{**good, "seed": 1, "source": "fallback"}]})["status"], "not scored")

    def test_the_critic_gate(self):
        from dclab_rnd.agent_eval.roles import critic_verdict

        row = {"critic_used": True, "worse": False, "worse_than_rules": False, "better": True, "kept_cite_weaknesses": True, "holdout_uses": 1}
        self.assertEqual(critic_verdict({"rows": [row, {**row, "better": False}]})["status"], "pass")
        self.assertEqual(critic_verdict({"rows": [row, {**row, "better": False, "worse": True}]})["status"], "fail")
        self.assertEqual(critic_verdict({"rows": [{**row, "worse_than_rules": True}]})["status"], "fail")  # must match the rules' loop
        self.assertEqual(critic_verdict({"rows": [{**row, "critic_used": False}]})["status"], "not scored")


if __name__ == "__main__":
    unittest.main()
