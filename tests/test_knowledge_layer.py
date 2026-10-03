"""Tests for the evidence index, critic gate, SFT v3 builder, notebook copilot and agent tools."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sft"))

from dclab_rnd import critic_gate, evidence_index, tools  # noqa: E402
from dclab_rnd.copilot import review_notebook, review_source  # noqa: E402
from dclab_rnd.copilot.render import annotate_notebook, to_html  # noqa: E402

EXAMPLES = ROOT / "dclab_rnd" / "copilot" / "examples"


class EvidenceIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = evidence_index.build_records(ROOT)
        cls.index = evidence_index.EvidenceIndex(cls.records)

    def test_build_is_deterministic_and_ids_unique(self):
        again = evidence_index.build_records(ROOT)
        self.assertEqual(evidence_index.render_index(self.records), evidence_index.render_index(again))
        ids = [r["record_id"] for r in self.records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_committed_index_is_current(self):
        self.assertTrue(evidence_index.write_index(ROOT, check=True), "run: python -m dclab_rnd.evidence_index build")

    def test_experiment_cards_are_self_contained(self):
        card = self.index.get("EXP-008")
        for needle in ("Predict term-deposit subscription", "Feature engineering ablation", "6 feature recipes", "ratios"):
            self.assertIn(needle, card["text"])

    def test_filters_narrow_before_ranking(self):
        hits = self.index.search("is call duration safe as a feature", dataset="bank_marketing", k=3)
        self.assertEqual(hits[0]["record_id"], "LEAK-bank_marketing")
        self.assertTrue(all(h["metadata"].get("dataset") == "bank_marketing" for h in hits))
        self.assertEqual([h["type"] for h in self.index.search("", type="rule", k=50)].count("rule"), 22)

    def test_no_provenance_noise_in_records(self):
        blob = evidence_index.render_index(self.records)
        self.assertNotIn("/Users/", blob)
        self.assertIsNone(re.search(r"\b[0-9a-f]{64}\b", blob))


class CriticGateTests(unittest.TestCase):
    def test_recorded_selections_follow_their_rules(self):
        checks = [g["rule_check"] for g in critic_gate.gate_campaigns(ROOT) if g["rule_check"]]
        self.assertGreaterEqual(len(checks), 30)
        self.assertTrue(all(c["consistent"] for c in checks))

    def test_known_false_critique_is_contradicted(self):
        result = json.loads(next((ROOT / "campaigns/model_building_50_v1/results").glob("EXP-008_*.json")).read_text())
        gated = critic_gate.gate_result(result)
        self.assertEqual(gated["rule_check"]["recomputed"], "ratios")
        self.assertGreaterEqual(gated["contradicted"], 1)

    def test_legitimate_scope_caveat_is_kept(self):
        challenge = {"issue": "The CV leader is not evidence that this family is best on unseen data.", "severity": "medium"}
        self.assertEqual(critic_gate.classify_challenge(challenge, {"consistent": True}), "unverified")


class SFTv3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_sft_dataset_v3 as builder

        cls.builder = builder
        cls.files, cls.manifest = builder.build(ROOT)

    def _rows(self, name):
        return [json.loads(line) for line in self.files[name].splitlines() if line.strip()]

    def test_validation_holds_out_whole_datasets(self):
        train = self._rows("train.chat.jsonl")
        val = self._rows("val.chat.jsonl")
        held = set(self.builder.VAL_DATASETS)
        self.assertFalse({r["metadata"].get("dataset") for r in train} & held)
        self.assertTrue({r["metadata"].get("dataset") for r in val} & held)

    def test_examples_are_clean_and_self_contained(self):
        for row in self._rows("train.chat.jsonl"):
            user, answer = row["messages"][1]["content"], row["messages"][2]["content"]
            self.assertNotRegex(user + answer, r"/Users/|\b[0-9a-f]{64}\b")
            if row["metadata"]["task"] == "explain_experiment":
                self.assertIn("### Dataset", user)
                self.assertIn("### What this run compared", user)
        self.assertEqual(self.manifest["quality_gates"]["contradicted_challenges_dropped"],
                         critic_gate.summary(critic_gate.gate_campaigns(ROOT))["contradicted_challenges"])

    def test_three_formats_have_equal_counts(self):
        counts = {fmt: len(self._rows(f"train.{fmt}.jsonl")) for fmt in ("chat", "alpaca", "sharegpt")}
        self.assertEqual(len(set(counts.values())), 1)

    def test_reference_answers_pass_the_evaluator(self):
        import eval_sft

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.chat.jsonl"
            path.write_text(self.files["val.chat.jsonl"])
            rows = eval_sft.load_val(path)
            result = eval_sft.score(rows, {r["id"]: r["messages"][2]["content"] for r in rows})
            self.assertEqual(result["overall"], 1.0)
            bad = eval_sft.score(rows, {r["id"]: "Looks fine." for r in rows})
            self.assertLess(bad["overall"], 0.5)

    def test_committed_corpus_is_current(self):
        stale = [n for n, text in self.files.items()
                 if not (self.builder.OUT_DIR / n).exists() or (self.builder.OUT_DIR / n).read_text() != text]
        self.assertEqual(stale, [], "run: python sft/build_sft_dataset_v3.py")


class CopilotTests(unittest.TestCase):
    def test_leaky_notebook_findings(self):
        report = review_notebook(EXAMPLES / "leaky_bank_marketing.ipynb")
        detectors = {f["detector"] for f in report["findings"]}
        for expected in ("known_leakage_column", "aggregate_before_split", "resample_before_split",
                         "preprocess_before_split", "holdout_reuse", "accuracy_only", "single_model_family"):
            self.assertIn(expected, detectors)
        for finding in report["findings"]:
            self.assertTrue(finding["proof"], finding["detector"])
            ids = [p["record_id"] for p in finding["proof"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_clean_notebook_has_no_medium_or_high(self):
        report = review_notebook(EXAMPLES / "clean_bank_marketing.ipynb")
        self.assertFalse([f for f in report["findings"] if f["severity"] in ("high", "medium")])

    def test_order_matters(self):
        before = review_source("Xs = StandardScaler().fit_transform(X)\nXtr, Xte, ytr, yte = train_test_split(Xs, y, random_state=0, stratify=y)")
        after = review_source("Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)\nsc = StandardScaler()\nXtr = sc.fit_transform(Xtr)\nXte = sc.transform(Xte)")
        self.assertIn("preprocess_before_split", {f["detector"] for f in before["findings"]})
        self.assertNotIn("preprocess_before_split", {f["detector"] for f in after["findings"]})

    def test_renderers_do_not_modify_original(self):
        path = EXAMPLES / "leaky_bank_marketing.ipynb"
        original = path.read_text()
        report = review_notebook(path)
        annotated = annotate_notebook(path, report)
        self.assertGreater(len(annotated["cells"]), len(json.loads(original)["cells"]))
        self.assertEqual(path.read_text(), original)
        page = to_html(report)
        self.assertIn("<title>", page)
        self.assertIn("Show proof", page)


class ToolTests(unittest.TestCase):
    def test_schemas_and_validation(self):
        names = {s["function"]["name"] for s in tools.tool_schemas()}
        self.assertTrue({"search_evidence", "plan_next_stage", "review_notebook", "audit_columns"} <= names)
        self.assertIn("input_schema", tools.tool_schemas("anthropic")[0])
        self.assertIn("error", tools.call_tool("search_evidence", {}))
        self.assertIn("error", tools.call_tool("search_evidence", {"query": "x", "bogus": 1}))

    def test_plan_next_stage_follows_workflow(self):
        plan = tools.call_tool("plan_next_stage", {"dataset": "adult", "completed_stages": ["data_understanding", "leakage_audit"]})
        self.assertEqual(plan["next_stage"], "feature_engineering")
        self.assertTrue(plan["rules"])

    def test_auditor_reconstructs_target_arithmetic(self):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"casual": rng.integers(0, 50, 300), "registered": rng.integers(0, 300, 300), "temp": rng.normal(size=300)})
        y = X["casual"] + X["registered"]
        report = tools._audit_frame(X, y, task="regression", blind=True)
        flagged = {r["column"] for r in report["flagged"]}
        self.assertTrue({"casual", "registered"} <= flagged)
        self.assertNotIn("temp", flagged)


if __name__ == "__main__":
    unittest.main()
