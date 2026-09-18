from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dclab_rnd.analysis import build_evidence
from dclab_rnd.cycle import build_cycle_command
from dclab_rnd.registry import collect_registry
from dclab_rnd.report import render_outputs, sync_outputs


def _write(root: Path, relative: str, payload: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class RegistryTests(unittest.TestCase):
    def test_normalizes_suites_and_excludes_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "model_name": "lightgbm",
                "optimization": "optimized",
                "feature_count": 12,
                "metrics": {"roc_auc": 0.91, "f1": 0.8},
            }
            _write(root, "general_pipeline/results/safe_optimized_lightgbm.json", {**base, "mode": "safe"})
            _write(root, "general_pipeline/results/unsafe_optimized_lightgbm.json", {**base, "mode": "unsafe", "metrics": {"roc_auc": 0.99}})

            records, issues = collect_registry(root)
            self.assertEqual(2, len(records))
            self.assertFalse(issues)
            safe = next(record for record in records if record.mode == "safe")
            unsafe = next(record for record in records if record.mode == "unsafe")
            self.assertTrue(safe.deployment_eligible)
            self.assertFalse(unsafe.deployment_eligible)
            self.assertEqual("lightgbm", safe.model_family)

            evidence = build_evidence(root, records, issues)
            self.assertEqual(0.91, evidence["champions"][0]["roc_auc"])

    def test_detects_duplicate_canonical_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "dataset": "sample",
                "exp_id": 6,
                "exp_name": "full_fe",
                "mode": "safe",
                "has_leakage": False,
                "metrics": {"roc_auc": 0.8},
            }
            _write(root, "external_projects/sample_exp/results/ladder/06_safe_a.json", payload)
            _write(root, "external_projects/sample_exp/results/ladder/06_safe_b.json", payload)
            _, issues = collect_registry(root)
            self.assertTrue(any("duplicate canonical" in item["message"] for item in issues))

    def test_reports_out_of_range_metric_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "general_pipeline/results/safe_baseline_xgboost.json",
                {"model_name": "xgboost", "mode": "safe", "metrics": {"roc_auc": 1.2}},
            )
            _, issues = collect_registry(root)
            self.assertEqual("error", issues[0]["severity"])

    def test_clamps_machine_precision_metric_and_maps_playbook_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "external_projects/sample_exp/results/ladder/04_safe_fe_ratios.json",
                {
                    "dataset": "sample",
                    "exp_id": 4,
                    "exp_name": "fe_ratios",
                    "mode": "safe",
                    "metrics": {"roc_auc": 1.0000000000000002},
                },
            )
            records, issues = collect_registry(root)
            self.assertFalse(issues)
            self.assertEqual(1.0, records[0].roc_auc)
            self.assertEqual("lightgbm", records[0].model_family)


class EvidenceTests(unittest.TestCase):
    def test_cross_dataset_evidence_uses_one_family_result_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, lgbm, xgb in (("a", 0.90, 0.89), ("b", 0.85, 0.87)):
                for suffix, score in (("one", lgbm), ("two", lgbm - 0.01)):
                    _write(
                        root,
                        f"general_pipeline/results_external/{dataset}__{suffix}__lightgbm.json",
                        {"dataset": dataset, "model_name": "lightgbm", "metrics": {"roc_auc": score}},
                    )
                _write(
                    root,
                    f"general_pipeline/results_external/{dataset}__one__xgboost.json",
                    {"dataset": dataset, "model_name": "xgboost", "metrics": {"roc_auc": xgb}},
                )
            records, issues = collect_registry(root)
            evidence = build_evidence(root, records, issues)
            families = {row["model_family"]: row for row in evidence["model_evidence"]}
            self.assertEqual(2, families["lightgbm"]["dataset_count"])
            self.assertEqual(1.5, families["lightgbm"]["mean_rank"])
            self.assertEqual(1.5, families["xgboost"]["mean_rank"])

    def test_generated_outputs_are_deterministic_and_checkable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "general_pipeline/results/safe_baseline_logistic_regression.json",
                {"model_name": "logistic_regression", "mode": "safe", "metrics": {"roc_auc": 0.75}},
            )
            records, issues = collect_registry(root)
            evidence = build_evidence(root, records, issues)
            outputs = render_outputs(records, evidence)
            self.assertTrue(sync_outputs(root, outputs))
            self.assertFalse(sync_outputs(root, outputs, check=True))
            self.assertIn("Deployment-eligible champions", (root / "knowledge" / "KNOWLEDGE_BASE.md").read_text())


class CycleTests(unittest.TestCase):
    def test_builds_controlled_runner_commands(self) -> None:
        command = build_cycle_command(
            suite="external",
            dataset="adult",
            model="lightgbm",
            mode="safe",
            optimization="optimized",
        )
        self.assertEqual("general_pipeline/run_multi_dataset.py", command[1])
        self.assertEqual("adult", command[command.index("--dataset") + 1])
        self.assertNotIn("shell=True", command)

    def test_external_cycle_requires_dataset(self) -> None:
        with self.assertRaises(ValueError):
            build_cycle_command(
                suite="external",
                dataset=None,
                model="lightgbm",
                mode="safe",
                optimization="baseline",
            )


if __name__ == "__main__":
    unittest.main()
