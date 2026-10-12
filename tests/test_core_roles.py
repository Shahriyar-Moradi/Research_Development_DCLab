"""Packages 18.4 and 18.6: the solution planner (dclab_core.plan.SOLUTION_PLANNER) and the critic
(dclab_core.review.CRITIC), each checked by code, with the rules as fallback. No model is called: advisors are scripted."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


def deliveries():
    from dclab_core.analyse import analyse
    from dclab_core.replicas import delivery_time

    frame = delivery_time.generate(seed=0, n=3000).orders
    return analyse({"orders": frame}, target="actual_minutes", task="regression")


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class PlannerTest(unittest.TestCase):
    def test_a_cited_change_is_kept_and_a_change_the_validator_refuses_is_recorded(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.plan import plan, plan_with

        findings = deliveries()
        base = plan(None, findings)
        leaks = [f.id for f in findings.findings if f.topic == "leakage" and f.severity == "critical"]
        self.assertTrue(leaks)
        target = next(f.id for f in findings.findings if f.topic == "target")
        advisor = ScriptedAdvisor({"solution_planner": {"decisions": [
            {"key": "metric.secondary", "choice": ["rmse", "mae_at_cap"], "reason": "The cap makes large errors matter separately.", "cites": [target]},
            {"key": "features.forbidden", "choice": [], "reason": "These columns look fine to me today.", "cites": leaks[:1]},   # keeps a leak in
            {"key": "baseline", "choice": "median", "reason": "A simple median baseline is enough.", "cites": []},           # cites nothing
            {"key": "holdout.size", "choice": 0.5, "reason": "A bigger holdout is more honest.", "cites": [target]}]}})  # not a decision key
        out = plan_with(advisor, None, findings)
        self.assertEqual([k["key"] for k in out["kept"]], ["metric.secondary"])
        self.assertEqual(out["plan"].decision("metric.secondary").choice, ["rmse", "mae_at_cap"])
        self.assertEqual(out["plan"].decision("metric.secondary").source, "advisor")
        reasons = {r["key"]: r["why"] for r in out["refused"]}
        self.assertIn("never remove", reasons["features.forbidden"])
        self.assertIn("cite", reasons["baseline"])
        self.assertIn("not a decision", reasons["holdout.size"])
        self.assertEqual(out["plan"].decision("features.forbidden").choice, base.decision("features.forbidden").choice)
        target_id = target
        unsafe = ScriptedAdvisor({"solution_planner": {"decisions": [
            {"key": "split.strategy", "choice": {"kind": "time", "time_column": "order_time"}, "reason": "Time matters more than restaurants.", "cites": [target_id]},
            {"key": "features.identifiers", "choice": [], "reason": "No column identifies a row really.", "cites": [target_id]},
            {"key": "task.framing", "choice": {"task": "binary"}, "reason": "Late or not is all that matters.", "cites": [target_id]},
            {"key": "model.families", "choice": ["magic_net"], "reason": "A better model family than these.", "cites": [target_id]},
            {"key": "threshold.policy", "choice": 0.5, "reason": "Half is a fine threshold to use.", "cites": [target_id]},
            "not a proposal"]}})
        refused = {r["key"]: r["why"] for r in plan_with(unsafe, None, findings)["refused"]}
        self.assertIn("grouped by restaurant_id", refused["split.strategy"])
        self.assertIn("never remove", refused["features.identifiers"])
        self.assertIn("not change it", refused["task.framing"])
        self.assertIn("known names", refused["model.families"])
        self.assertIn("kind is one of", refused["threshold.policy"])
        self.assertIn("not a proposal object", refused["not a proposal"])
        none = plan_with(None, None, findings)  # no model: the rule plan, unchanged
        self.assertEqual(none["plan"].model_dump(), base.model_dump())


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class CriticTest(unittest.TestCase):
    def review(self):
        from dclab_core.review import review

        rng = np.random.default_rng(0)
        base = rng.gamma(4, 4, 4000)
        y = base * np.exp(rng.normal(0, 0.5, 4000))  # multiplicative noise: errors grow with the prediction, and lean right
        return review("regression", y, base, metric="mae")

    def test_the_critic_ranks_checked_hypotheses_and_falls_back_to_the_rules(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.review import critique

        rv = self.review()
        rule_order = [e.name for e in rv.next_experiments]
        self.assertIn("target.log", rule_order)
        advisor = ScriptedAdvisor({"critic": {"hypotheses": [
            {"statement": "Errors grow with the prediction: a log target evens them.", "cites": ["F-weak-errors_grow"], "experiment": "target.log"},
            {"statement": "Something unmeasured.", "cites": ["F-weak-the-moon"], "experiment": "loss.median"},
            {"statement": "A new model.", "cites": ["F-weak-residual_skew"], "experiment": "deep.learning"}]}})
        out = critique(advisor, rv, "regression", metric="mae", model="ridge", score=8.4)
        self.assertEqual(out["order"][0], "target.log")
        self.assertEqual(sorted(out["order"]), sorted(rule_order))  # everything else follows, in the rules' order
        self.assertEqual(len(out["refused"]), 2)
        self.assertEqual(critique(None, rv, "regression")["order"], rule_order)
        self.assertEqual(critique(advisor, rv, "regression", score=float("nan"))["order"][0], "target.log")  # a missing score is no failure
        off_topic = ScriptedAdvisor({"critic": {"hypotheses": [
            {"statement": "Clip it.", "cites": ["F-weak-residual_skew"], "experiment": "predictions.clip"}]}})
        self.assertIn("does not follow", critique(off_topic, rv, "regression")["refused"][0]["why"])

    def test_the_engine_runs_the_critics_choice_and_records_it(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.replicas import delivery_time
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        frame = delivery_time.generate(seed=0, n=1200).orders.drop(columns=["delivered_at", "courier_rating"])
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Deliveries", "general", "Estimate delivery minutes")
        path = store.data_dir(project["id"]) / "data.csv"
        frame.to_csv(path, index=False)
        loaded = sd.load_table(path)
        project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path),
                           "profile": sd.profile_table(loaded)}
        project["solution"] = Solution(target="actual_minutes", task="regression", identifiers=["order_id"],
                                       prediction_moment="When the customer places the order.", group_column="restaurant_id").model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        critic = ScriptedAdvisor({"critic": lambda evidence: {"hypotheses": [
            {"statement": "A capped, skewed target is closer to even on a log scale.", "cites": [w.split(":")[0] for w in evidence["weaknesses"]][:1],
             "experiment": "target.log"}]}})  # cites a weakness a log target does not address
        with runner.fast_profile(), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=critic):
            engine.run_all(store, project["id"])
        rounds = store.read_stage(project["id"], "models")["evidence"]["review"]["rounds"]
        # a log target addresses errors that grow with the prediction, not a cap or a skew: refused, and the rules' choice runs
        self.assertEqual(rounds[0]["experiment"], "loss.median")
        self.assertEqual(rounds[0]["critic"]["source"], "advisor")
        self.assertIn("does not follow from the weaknesses cited", rounds[0]["critic"]["refused"][0]["why"])
        self.assertEqual(store.read_stage(project["id"], "final")["evidence"]["holdout_uses_in_this_project"], 1)


if __name__ == "__main__":
    unittest.main()
