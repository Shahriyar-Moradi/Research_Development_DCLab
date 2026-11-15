"""Tests for Multi-Model AI Understanding & 5-Gate Scientific Harness on No-Data problems."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest import mock

from dclab_core import case as C
from dclab_rnd.agent_eval import no_data as no_data_eval
from dclab_rnd.agents import case_agent as A
from dclab_rnd.agents import specialists as S
from dclab_rnd.draft import no_data, no_data_harness


class _FakePurposeClient:
    def __init__(self, purpose: str, replies: list[str], model: str = "fake-ai-model"):
        self.purpose = purpose
        self.replies = list(replies)
        self.model = model
        self.usage: list[dict[str, Any]] = []
        self.calls = 0

    def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        idx = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        self.usage.append({"cost_eur": 0.002, "model": self.model})
        return {"content": self.replies[idx], "model": self.model, "cost_eur": 0.002}


class _FakeGateway:
    def __init__(self, by_purpose: dict[str, list[str]]):
        self.by_purpose = by_purpose
        self.clients: dict[str, _FakePurposeClient] = {}

    def available(self, purpose: str) -> bool:
        return True

    def client(self, purpose: str, **kwargs: Any) -> _FakePurposeClient:
        if purpose not in self.clients:
            replies = self.by_purpose.get(purpose) or ["{}"]
            self.clients[purpose] = _FakePurposeClient(purpose, replies)
        return self.clients[purpose]


class NoDataAIHarnessTests(unittest.TestCase):
    def test_epistemic_understanding_and_five_gates_offline_floor(self):
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            sol = no_data.solve_without_data(
                "Our hospital wants to predict at patient discharge whether a patient will be readmitted within 30 days. "
                "We cannot upload HIPAA records. 3,000 discharges/month, 12% readmission rate, $80 nurse follow-up cost, $1,600 penalty avoided."
            )
        und = sol["understanding"]
        self.assertTrue(und["restatement"])
        fields = und["fields"]
        for req_key in ("goal", "decision_action", "decision_unit", "prediction_moment", "outcome_and_horizon", "primary_metric"):
            self.assertIn(req_key, fields)
            self.assertIn(fields[req_key]["status"], ("user_stated", "ai_inferred"))
            self.assertTrue(fields[req_key]["value"])
        self.assertGreaterEqual(len(und["alternatives"]), 1)
        self.assertGreaterEqual(len(und["eda_plan"]), 2)
        self.assertTrue(all(p["inside_folds"] for p in und["preprocessing_plan"]))

        ds = sol["data_science"]
        self.assertGreater(len(ds["verified_feature_programs"]), 0)
        hrep = sol["harness_report"]
        self.assertTrue(hrep["all_gates_passed"])
        self.assertEqual(hrep["gates"]["gate_2_number_grounding"]["ungrounded_numbers_remaining"], 0)
        self.assertEqual(hrep["gates"]["gate_5_zero_auto_simulation_and_budget"]["auto_simulated_tables"], 0)

    def test_multi_model_harness_repairs_workflow_strips_hallucinated_numbers_and_excludes_leaks(self):
        # First reply from ai_first has a workflow ordering bug (final before leakage) AND an ungrounded number ("97.4% accuracy on 412,000 patients")
        bad_wf_trace = {
            "problem": {
                "restatement": "Predict 30-day hospital readmission at discharge with 97.4% accuracy on 412,000 patients.",
                "decision": "Schedule a post-discharge transition nurse call within 48 hours.",
                "unit": "hospital_admission",
                "moment": "At inpatient discharge sign-off before the patient leaves the ward",
                "outcome": "Unplanned inpatient readmission within 30 days",
                "outcome_known": "30 days after discharge date from EHR encounter log",
                "target": "readmitted_30d",
            },
            "problem_type": {
                "family": "classification",
                "kind": "binary",
                "task": "binary",
                "labels": "column",
                "why": "Binary outcome over a fixed 30-day post-discharge window.",
                "alternatives": [
                    {"family": "survival", "why_not": "Competing-risk hazard modeling is useful if exact day-to-readmission censorship varies."}
                ],
            },
            "data": [
                {
                    "name": "admissions_clinical_spine",
                    "grain": "One row per inpatient admission at discharge",
                    "columns": [
                        {"name": "admission_id", "role": "identifier", "why": "Unique encounter key"},
                        {"name": "discharged_at", "role": "time", "why": "Prediction cutoff timestamp"},
                        {"name": "length_of_stay_days", "role": "feature", "why": "Known at discharge"},
                        {"name": "prior_ed_visits_180d", "role": "feature", "why": "Historical utilization"},
                        {"name": "readmitted_30d", "role": "target", "why": "30-day readmission label"},
                        {"name": "post_discharge_billing_adjustment", "role": "leak", "why": "Coded weeks after readmission"},
                    ],
                    "identifiers": ["admission_id"],
                    "times": ["discharged_at"],
                    "groups": [],
                    "leak_suspects": [
                        {
                            "column": "post_discharge_billing_adjustment",
                            "why": "Finalized 45 days after discharge",
                            "test": "Compare billing timestamp against discharged_at",
                            "action": "exclude",
                        }
                    ],
                }
            ],
            "eda": [
                {"question": "Does readmission rate vary by ward?", "tool": "query", "conclusion": "Check ward stratification."}
            ],
            "preprocessing": [
                {"step": "Impute missing lab values", "columns": ["length_of_stay_days"], "inside_folds": True, "why": "Fold-safe median imputation."}
            ],
            "problem_workflow": {"size": "single", "parts": []},
            "algorithms": {
                "ranked": [{"family": "lightgbm", "rank": 1, "why": "Handles mixed clinical tabular signals."}],
                "baseline": "logistic_regression",
                "would_change": "Switch to regularized logistic regression if positive readmission count is small.",
            },
            "evaluation": {"split": "time", "split_column": "discharged_at", "split_why": "Prevent temporal leakage.", "metric": "pr_auc"},
            "solution_workflow": [
                {"id": "S1", "wf": "WF-09", "tool": "run_stage", "stage": "final", "does": "Read holdout first", "why": "bad order", "after": []},
                {"id": "S2", "wf": "WF-05", "tool": "run_stage", "stage": "leakage", "does": "Audit leakage", "why": "late", "after": ["S1"]},
            ],
            "routes": [],
            "risks": ["Post-discharge coding leakage"],
            "questions": ["Are transfers to skilled nursing facilities counted as readmissions?"],
            "confidence": "high",
        }
        repaired_trace = dict(bad_wf_trace)
        repaired_trace["solution_workflow"] = [
            {"id": "S1", "wf": "WF-04", "tool": "run_stage", "stage": "data", "does": "Profile data", "why": "split first", "after": []},
            {"id": "S2", "wf": "WF-05", "tool": "run_stage", "stage": "leakage", "does": "Audit leakage", "why": "before models", "after": ["S1"]},
            {"id": "S3", "wf": "WF-06", "tool": "run_stage", "stage": "features", "does": "Build features", "why": "fold-safe", "after": ["S2"]},
            {"id": "S4", "wf": "WF-07", "tool": "run_stage", "stage": "models", "does": "Screen models", "why": "on CV folds", "after": ["S3"]},
            {"id": "S5", "wf": "WF-09", "tool": "run_stage", "stage": "final", "does": "Read holdout once", "why": "final sign-off", "after": ["S4"]},
        ]

        fake_gw = _FakeGateway(
            {
                "ai_first": [json.dumps(bad_wf_trace), json.dumps(repaired_trace)],
                "synthetic_schema": [
                    json.dumps(
                        {
                            "name": "admissions_clinical_spine",
                            "columns": [
                                {"name": "charlson_comorbidity_index", "type": "integer"},
                                {"name": "discharge_disposition", "type": "categorical"},
                            ],
                        }
                    )
                ],
                "case_specialist": [
                    json.dumps(
                        {
                            "findings": [
                                {
                                    "kind": "risk",
                                    "text": "Exclude post_discharge_billing_adjustment because it is recorded after readmission; industry benchmark is 88.9% AUC on 999,999 rows.",
                                    "field": "data",
                                }
                            ]
                        }
                    )
                ],
                "solution_planner": [
                    json.dumps(
                        {
                            "changes": [
                                {
                                    "key": "split.strategy",
                                    "choice": "time",
                                    "reason": "Hospital discharge cohorts drift over months across F-time-01.",
                                    "cites": ["F-time-01"],
                                }
                            ]
                        }
                    )
                ],
                "solution_challenger": [
                    json.dumps(
                        {
                            "changes": [
                                {
                                    "key": "metric.primary",
                                    "choice": "average_precision",
                                    "reason": "Rare readmission outcome requires precision-recall focus per F-imb-01.",
                                    "cites": ["F-imb-01"],
                                }
                            ]
                        }
                    )
                ],
                "debate_judge": [
                    json.dumps(
                        {
                            "rulings": [
                                {
                                    "key": "metric.primary",
                                    "keep": "b",
                                    "reason": "Average precision directly tracks alert queue precision under F-imb-01.",
                                }
                            ]
                        }
                    )
                ],
            }
        )

        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "0"}):
            sol = no_data.solve_without_data(
                "Predict 30-day hospital readmission at discharge without uploading data. 3,000 discharges/month, 12% rate, $80 cost, $1,600 value.",
                gateway=fake_gw,
                models=fake_gw,
                cap_eur=0.25,
            )

        hrep = sol["harness_report"]
        self.assertEqual(hrep["mode"], "ai_harnessed")
        self.assertTrue(hrep["all_gates_passed"])
        # Self-repair turn was triggered on the bad workflow order
        self.assertGreaterEqual(len(hrep["repairs_applied"]), 1)
        self.assertEqual(fake_gw.clients["ai_first"].calls, 2)
        # Hallucinated numbers ("97.4", "412,000", "88.9", "999,999") were stripped by Gate 2!
        self.assertNotIn("412,000", sol["understanding"]["restatement"])
        self.assertNotIn("97.4", sol["understanding"]["restatement"])
        self.assertEqual(hrep["gates"]["gate_2_number_grounding"]["ungrounded_numbers_remaining"], 0)
        self.assertGreater(hrep["gates"]["gate_2_number_grounding"]["stripped_ungrounded_numbers"], 0)
        # Gate 3 enforced `post_discharge_billing_adjustment` into forbidden_leakage_columns and kept it out of tier1 features
        ds = sol["data_science"]
        self.assertIn("post_discharge_billing_adjustment", ds["forbidden_leakage_columns"])
        self.assertNotIn("post_discharge_billing_adjustment", ds["tier1_minimum_features"])
        # Domain columns from `ai_first` and `synthetic_schema` were merged
        vt_cols = ds["virtual_tables"]["admissions_clinical_spine"]
        self.assertIn("length_of_stay_days", vt_cols)
        self.assertIn("charlson_comorbidity_index", vt_cols)

    def test_specialists_allowed_on_virtual_schema_without_physical_table(self):
        case = C.new("draft-nodata-spec")
        case = C.hear(case, "We want to predict customer churn next month without uploading our data.")
        case = C.apply(case, "goal", C.CaseFact(value="predict customer churn", status="said", quote="predict customer churn"), by="agent", why="stated")

        # Without virtual_columns or blueprint, leakage_auditor is blocked
        self.assertIsNotNone(S.allowed("leakage_auditor", case, None, []))
        # With virtual_columns and has_blueprint=True, leakage_auditor, data_investigator, and solution_designer are allowed!
        self.assertIsNone(
            S.allowed(
                "leakage_auditor",
                case,
                None,
                [],
                virtual_columns=["customer_id", "snapshot_date", "churned_30d"],
                has_blueprint=True,
            )
        )
        self.assertIsNone(
            S.allowed(
                "solution_designer",
                case,
                None,
                [],
                virtual_columns=["customer_id", "snapshot_date", "churned_30d"],
                has_blueprint=True,
            )
        )

        # Verify CaseAgent._consult works on a no-data case after _design_solution_without_data
        state = A.CaseState(case=case)
        state.specialist_client = _FakePurposeClient(
            "case_specialist",
            [json.dumps({"findings": [{"kind": "risk", "text": "Ensure cancellation_reason is excluded at snapshot_date.", "field": "data"}]})],
        )
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            A._design_solution_without_data(state)
        out = A._consult(state, "leakage_auditor", "Check virtual schema for point-in-time leaks.")
        self.assertIn("findings", out)
        self.assertEqual(out["findings"][0]["kind"], "risk")

    def test_three_arm_no_data_evaluation_gate_passes(self):
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            res = no_data_eval.evaluate(split="dev")
        self.assertEqual(res["gate_id"], "no_data_gate_v1")
        self.assertTrue(res["passed"], f"Failed gate checks: {res['checks']}")
        self.assertGreater(
            res["summaries"]["ai_harnessed"]["mean_understanding_completeness"],
            res["summaries"]["rules"]["mean_understanding_completeness"],
        )
        self.assertEqual(res["summaries"]["ai_harnessed"]["ungrounded_number_rate"], 0.0)
        self.assertEqual(res["summaries"]["ai_harnessed"]["workflow_and_feature_safety_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
