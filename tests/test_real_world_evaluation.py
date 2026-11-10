"""Independent real-world evaluation and capability coverage suite (SCI-04, SCI-05)."""

from __future__ import annotations

import unittest
from pathlib import Path

from dclab_rnd import kinds
from dclab_rnd.kinds import UnsupportedTaskBlocker, validate_task_capability
from dclab_rnd.studio import data as studio_data, solution as studio_solution


class TestCapabilityAndRealWorldEvaluation(unittest.TestCase):
    def test_capability_validation_supported_tasks(self) -> None:
        # Standard base tasks
        self.assertIsNone(validate_task_capability("binary"))
        self.assertIsNone(validate_task_capability("multiclass"))
        self.assertIsNone(validate_task_capability("regression"))

        # Registered kinds
        self.assertIsNone(validate_task_capability("rare_event"))
        self.assertIsNone(validate_task_capability("uplift"))
        self.assertIsNone(validate_task_capability("segmentation"))
        self.assertIsNone(validate_task_capability("forecasting"))

    def test_capability_validation_unsupported_tasks_blockers(self) -> None:
        # Known unsupported tasks produce structured blockers
        blocker = validate_task_capability("reinforcement_learning")
        self.assertIsInstance(blocker, UnsupportedTaskBlocker)
        self.assertEqual(blocker.status, "unsupported")
        self.assertIn("environment_gym_adapter", blocker.missing_components)
        self.assertIn("uplift", blocker.supported_alternatives)

        t2i_blocker = validate_task_capability("text_to_image")
        self.assertIsInstance(t2i_blocker, UnsupportedTaskBlocker)
        self.assertIn("diffusion_pipeline", t2i_blocker.missing_components)

        # Arbitrary unregistered task
        unknown_blocker = validate_task_capability("quantum_annealing_optimization")
        self.assertIsInstance(unknown_blocker, UnsupportedTaskBlocker)
        self.assertIn("not registered", unknown_blocker.reason)

    def test_real_dataset_adult_census_evaluation(self) -> None:
        frame, suggestion = studio_data.load_sample("adult")
        self.assertGreater(len(frame), 1000)
        profile = studio_data.profile_table(frame)
        target = suggestion["target"]
        proposal = studio_solution.propose(frame, profile, target)

        self.assertEqual(proposal["task"], "binary")
        self.assertEqual(proposal["target"], target)
        self.assertEqual(proposal["metric"], "roc_auc")
        split_dec = next((p for p in proposal["plan"] if p["decision"] == "split.strategy"), None)
        self.assertIsNotNone(split_dec)
        self.assertEqual(split_dec["choice"]["kind"], "stratified")

    def test_real_dataset_bank_marketing_leakage_detection(self) -> None:
        frame, suggestion = studio_data.load_sample("bank_marketing")
        self.assertIn("duration", frame.columns)
        profile = studio_data.profile_table(frame)
        target = suggestion["target"]
        proposal = studio_solution.propose(frame, profile, target)

        # Call duration is post-outcome
        forbidden_cols = {f["column"] for f in proposal.get("forbidden", [])}
        self.assertIn("duration", forbidden_cols)


if __name__ == "__main__":
    unittest.main()
