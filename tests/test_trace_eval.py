import json
from pathlib import Path
import tempfile
import unittest

from dclab_rnd.agentic.trace_eval import audit_trace


class HistoricalTraceEvaluationTests(unittest.TestCase):
    def test_missing_reference_and_failed_trial_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps({
                "run": {"id": "r1", "status": "completed", "config": {"datasets": ["bank_marketing"]}},
                "events": [
                    {"seq": 1, "kind": "trial", "payload": {"id": "r1:trial-001", "status": "completed"}},
                    {"seq": 2, "kind": "proposal", "payload": {
                        "dataset": "bank_marketing", "title": "Compare a second family",
                        "hypothesis": "A tree model might improve development ranking.",
                        "model": "random_forest", "parameters": {}, "features": [],
                        "drop_columns": [], "stress_columns": [],
                        "evidence_ids": ["r1:trial-001"], "reference_evidence_id": None,
                        "expected_learning": "Learn whether a tree helps"}},
                    {"seq": 3, "kind": "trial", "payload": {"id": "r1:trial-002", "status": "failed"}},
                    {"seq": 4, "kind": "critique", "payload": {
                        "observation": "The second attempt failed without a score.",
                        "interpretation": "No model conclusion", "limitations": ["No measurement"],
                        "next_question": "Repair validation", "evidence_ids": ["r1:trial-001"],
                        "continue_research": False}},
                ],
            }), encoding="utf-8")
            report = audit_trace(path)
            proposal = next(e for e in report["events"] if e["agent_stage"] == "proposal")
            critique = next(e for e in report["events"] if e["agent_stage"] == "critique")
            p = {c["name"]: c["pass"] for c in proposal["checks"]}
            c = {x["name"]: x["pass"] for x in critique["checks"]}
            self.assertTrue(p["citations_prior_success"])
            self.assertFalse(p["comparison_reference_explicit"])
            self.assertTrue(c["failed_trial_not_cited"])
            self.assertTrue(c["failure_acknowledged"])
