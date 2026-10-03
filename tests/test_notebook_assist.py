import json
from pathlib import Path
import unittest

from dclab_rnd.notebook_assist import MANIFEST_CANDIDATES, code_facts, manifest_path, resolve_path, review_document
from dclab_rnd.notebook_assist_eval import evaluate


def reviewed(path):
    document = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    document["notebook_path"] = path
    return review_document(document)


class NotebookCompanionTests(unittest.TestCase):
    @unittest.skipIf(manifest_path() is None, f"frozen manifest not committed yet: {MANIFEST_CANDIDATES[0].name}")
    def test_frozen_development_suite_and_proof_rules(self):
        manifest = json.loads(manifest_path().read_text())
        score = evaluate(manifest)
        self.assertEqual((score["cases_passed"], score["cases_total"]), (7, 7))
        self.assertEqual((score["proof_rules_passed"], score["proof_rules_scored"]), (4, 4))
        self.assertTrue(score["availability_wording_safe"])
        self.assertFalse(score["overall_release_ready"])

    def test_hyperack_cross_cell_and_clean_eda(self):
        report = reviewed("part1_hyper_ack_classification.ipynb")
        cells = {c["code_cell_index"]: c for c in report["code_cells"]}
        self.assertEqual(cells[16]["findings"], [])
        self.assertEqual(cells[26]["findings"][0]["kind"], "cross_cell_preprocessing_review")
        self.assertEqual(cells[28]["findings"][0]["kind"], "preprocessing_before_split")
        self.assertIn("DCLAB-R04", [r["id"] for r in cells[26]["findings"][0]["proof"]["records"]])
        self.assertFalse(cells[26]["findings"][0]["proof"]["exact_analogous_experiment"])

    def test_churn_warning_is_conditional_and_citation_is_relevant(self):
        report = reviewed("research/agentic-ml-copilot/experiments/prototype_sept22/demo_churn_model.ipynb")
        self.assertIsNone(report["dataset_inferred"])  # Demo filename does not prove Telco source identity.
        first = {f["kind"]: f for f in report["code_cells"][0]["findings"]}
        self.assertEqual(set(first), {"preprocessing_before_split", "prediction_time_availability", "unseeded_split"})
        warning = first["prediction_time_availability"]
        self.assertIn("does not prove leakage", warning["explanation"])
        self.assertIn("DCLAB-R05", [r["id"] for r in warning["proof"]["records"]])
        self.assertFalse(warning["proof"]["exact_analogous_experiment"])
        self.assertEqual(report["code_cells"][1]["findings"][0]["kind"], "single_model_comparison")

    def test_direct_bank_duration_claim_is_only_used_for_bank(self):
        document = {"notebook_path": "bank_marketing_example.ipynb", "cells": [{"cell_type": "code",
                    "source": "features = ['duration', 'balance']\nX = df[features]\n"}]}
        finding = review_document(document)["code_cells"][0]["findings"][0]
        self.assertTrue(finding["proof"]["exact_analogous_experiment"])
        self.assertIn("EXP-007-C2", [r["id"] for r in finding["proof"]["records"]])

    def test_string_mentions_and_same_statement_do_not_claim_ordered_leakage(self):
        facts = code_facts("message = 'scaler.fit_transform(X); train_test_split(X)'\n")
        self.assertEqual(facts["operations"], [])
        facts = code_facts("X_train = train_test_split(scaler.fit_transform(X))\n")
        self.assertNotIn("fit", facts["operations"])
        self.assertNotIn("split", facts["operations"])
        report = review_document({"notebook_path": "demo.ipynb", "cells": [{"cell_type": "code", "source": "%matplotlib inline"}]})
        self.assertEqual(report["code_cells"][0]["review_status"], "unsupported_syntax")


if __name__ == "__main__":
    unittest.main()
