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
        # 22.8: DCLab's decision service also flags contract_duration from its name ("duration"): shown as its own
        # finding, likely a false alarm the person answers (the answer becomes a label), never folded into the review's
        self.assertEqual(set(first), {"preprocessing_before_split", "prediction_time_availability", "unseeded_split",
                                      "dclab_decision_leakage_suspect"})
        self.assertEqual(first["dclab_decision_leakage_suspect"]["decision"]["column"], "contract_duration")
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

    def test_review_asks_dclabs_leakage_decision_for_each_feature(self):
        """22.8: the review asks the decision service, so it and /api/decide give the same answer on a name."""
        from dclab_rnd.decide import ask
        from dclab_core.bands import name_tokens

        document = {"notebook_path": "orders.ipynb", "cells": [{"cell_type": "code",
                    "source": "features = ['refund_amount', 'final_fare', 'basket_size']\nX = df[features]\n"}]}
        report = review_document(document)
        findings = report["code_cells"][0]["findings"]
        by_column = {f["decision"]["column"]: f for f in findings if f.get("decision")}
        # refund_amount: only DCLab's decision flags it (the review's own list does not): its own kind, not the frozen one
        self.assertEqual(by_column["refund_amount"]["kind"], "dclab_decision_leakage_suspect")
        self.assertIn("name alone", by_column["refund_amount"]["explanation"])
        # final_fare: both flag it: one finding, the decision attached beside the review's own warning
        self.assertEqual(by_column["final_fare"]["kind"], "prediction_time_availability")
        self.assertNotIn("basket_size", by_column)
        for column, f in by_column.items():
            same = ask("feature.leakage_suspect", {"column": column, "name_tokens": name_tokens(column)})
            self.assertEqual(f["decision"]["answer"], same["answer"])
            self.assertEqual(f["decision"]["decided_by"], same["decided_by"])
            self.assertEqual(f["decision"]["rule"], "rule:bands-approx")
        self.assertTrue(any("name alone" in line for line in report["limitations"]))

    def test_review_does_not_ask_when_turned_off_or_when_the_service_fails(self):
        document = {"notebook_path": "orders.ipynb", "cells": [{"cell_type": "code",
                    "source": "features = ['refund_amount']\nX = df[features]\n"}]}
        self.assertEqual(review_document(document, ask_decisions=False)["code_cells"][0]["findings"], [])
        from unittest import mock
        with mock.patch("dclab_rnd.decide.ask", side_effect=RuntimeError("down")):
            report = review_document(document)
        self.assertEqual(report["code_cells"][0]["findings"], [])
        self.assertTrue(any("not asked" in line for line in report["limitations"]))

    def test_copilot_review_shows_the_decision_beside_its_column_warnings(self):
        from dclab_rnd.copilot import review_source
        from dclab_rnd.decide import ask
        from dclab_core.bands import name_tokens

        report = review_source("X = df[['refund_after_delivery', 'basket_size']]\ny = df['churned']\n")
        warned = [f for f in report["findings"] if f["detector"] in ("known_leakage_column", "suspicious_column_name")]
        self.assertTrue(warned)
        for f in warned:
            column = f["decision"]["column"]
            self.assertEqual(f["decision"]["answer"], ask("feature.leakage_suspect", {"column": column, "name_tokens": name_tokens(column)})["answer"])
        self.assertFalse(any(f.get("decision") for f in report["findings"] if f not in warned))  # never a finding of its own


if __name__ == "__main__":
    unittest.main()
