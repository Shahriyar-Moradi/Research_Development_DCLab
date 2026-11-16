"""Tests for Multi-Model AI Understanding & 5-Gate Scientific Harness on No-Data problems."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest import mock

os.environ["DCLAB_NO_DATA_V2"] = "0"  # this file tests the first no-data design; the rebuilt one is Home's default since 2026-11-16

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
            self.assertIn(fields[req_key]["status"], ("user_stated", "rules_read", "dclab_default"))  # no model ran: nothing is "AI inferred"
            self.assertTrue(fields[req_key]["value"])
        self.assertNotIn("AI inferred", sol["markdown"])
        self.assertIn("DCLab default", sol["markdown"])
        self.assertGreaterEqual(len(und["alternatives"]), 1)
        self.assertGreaterEqual(len(und["eda_plan"]), 2)
        self.assertTrue(all(p["inside_folds"] for p in und["preprocessing_plan"]))

        ds = sol["data_science"]
        self.assertGreater(len(ds["verified_feature_programs"]), 0)
        hrep = sol["harness_report"]
        self.assertTrue(hrep["all_gates_passed"])
        self.assertEqual(hrep["mode"], "deterministic_floor")
        self.assertEqual(hrep["gates"]["gate_2_number_grounding"]["ungrounded_numbers_remaining"], 0)
        self.assertEqual(hrep["gates"]["gate_2_number_grounding"]["assumed_inputs"], ["share of actions that work"])
        self.assertEqual(set(hrep["gates"]), {"gate_1_epistemic_provenance", "gate_2_number_grounding", "gate_3_virtual_schema_and_features",
                                               "gate_4_workflow_and_debate", "gate_5_budget"})

    def test_every_gate_can_fail_and_the_card_says_which(self):
        """The review of 22ad1bf8: two gates were the constant True and a third could not fail; a failed gate changed nothing."""
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            sol = no_data.solve_without_data("No data yet. I want to predict which customers will churn next month.")
        report = sol["harness_report"]
        self.assertTrue(report["all_gates_passed"])
        # gate 2: a model-written sentence with a number nobody gave, or a claim of production readiness
        sol["business"]["domain_expert_findings"] = [{"kind": "fact", "text": "This will save 3600 a month and the model is production-ready.", "by": "domain_expert"}]
        report = no_data_harness.finish_gates(sol, sol["problem"])
        gate2 = report["gates"]["gate_2_number_grounding"]
        self.assertEqual((gate2["passed"], gate2["ungrounded_numbers_remaining"] > 0, gate2["overclaims"]), (False, True, 1))
        self.assertFalse(report["all_gates_passed"])
        self.assertIn("Gates passed: `4/5` · not passed: numbers", no_data.to_markdown(sol))
        # gate 5: the money cap (a role that cost more than the case had left)
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "0"}), \
                mock.patch.object(no_data_harness, "_call_ai_first_expert", lambda *a, **k: (None, [], 0.5, None)):
            over = no_data.solve_without_data("No data yet. I want to predict which customers will churn next month.", gateway=_FakeGateway({}),
                                              models=None, cap_eur=0.1)
        self.assertFalse(over["harness_report"]["gates"]["gate_5_budget"]["passed"])
        self.assertIn("not passed: budget", over["markdown"])

    def test_broken_model_replies_are_never_called_an_ai_result(self):
        """The review of 22ad1bf8: with every role returning malformed JSON the card said ai_harnessed, 5/5, and "AI inferred"."""
        broken = _FakeGateway({})
        for purpose in ("ai_first", "synthetic_schema", "case_specialist", "problem_analyst", "brief_checker", "understanding_critic", "feature_engineer",
                        "decomposer", "map_reviewer", "solution_router", "solution_planner", "solution_challenger", "debate_judge", "task_decomposer"):
            broken.by_purpose[purpose] = ["this is not JSON at all"]
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "0"}):
            sol = no_data.solve_without_data("No data yet. I want to predict which customers will churn next month.", gateway=broken, models=broken, cap_eur=0.25)
        report = sol["harness_report"]
        self.assertEqual(report["mode"], "deterministic_floor")  # models were asked; none gave a usable answer
        self.assertTrue(report["model_requests_made"])
        statuses = {f["status"] for f in sol["understanding"]["fields"].values()}
        self.assertNotIn("ai_inferred", statuses)
        self.assertNotIn("AI inferred", sol["markdown"])
        self.assertNotIn("Planner-Challenger-Judge Debate Verdict", sol["markdown"])

    def test_the_harness_never_switches_the_thinking_off_and_keeps_it(self):
        """The owner's rule (CLAUDE.md rule 8) and the review of 22ad1bf8: a retry sent reasoning off, the first call lowered
        the effort, and the words of these calls were not kept. Through the real gateway, with a scripted transport."""
        import shutil
        import tempfile
        from pathlib import Path

        from dclab_rnd import experience
        from dclab_rnd.models import gateway as gw
        from dclab_rnd.models.usage import FileUsage

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        seen: list[dict[str, Any]] = []

        class T:
            def __init__(self):
                self.thinking = {"type": "adaptive"}

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                seen.append({"max_tokens": max_tokens, **options})
                cut = len(seen) in (1, 3)  # the first answer of each role is cut before its JSON
                text = "" if cut else json.dumps({"columns": [{"name": "tenure_months"}], "problem": {"restatement": "Predict churn"}})
                return {"content": text, "tool_calls": [], "finish_reason": "length" if cut else "stop", "usage": {"input_tokens": 1, "output_tokens": 1},
                        "thinking": [{"text": "the summary"}], "assistant_message": {"role": "assistant", "content": text}}

        transport = T()
        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "DCLAB_NO_LIVE_MODELS",
                                 "DCLAB_THINKING", "OPENROUTER_API_KEY", "DCLAB_BLOCK_PROVIDERS")} | {"OPENAI_API_KEY": "k", "DCLAB_AGENT_HOME": str(home)}
        with mock.patch.object(gw, "_transport", lambda tier, purpose: transport), mock.patch.dict(os.environ, clean), \
                mock.patch.object(experience, "current_log", lambda: None):
            g = gw.Gateway(FileUsage(home / "usage.jsonl"), transport=gw._transport)
            g.home = home
            no_data_harness._call_ai_first_expert(g, "Predict churn with no data yet", family_key="churn", virtual_tables=[], as_said=[])
            no_data_harness._call_synthetic_schema_architect(g, "Predict churn with no data yet", family_key="churn",
                                                             prediction_moment="month start", target_name="churned", cap_eur=0.05)
        self.assertEqual(len(seen), 4)  # each role asked again once, with more room
        for call in seen:
            self.assertNotIn("reasoning", json.dumps(call.get("extra_body") or {}))  # never switched off, never lowered
        self.assertGreater(seen[1]["max_tokens"], seen[0]["max_tokens"])
        docs = [json.loads(line) for p in sorted(home.rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted({d["purpose"] for d in docs}), ["ai_first", "synthetic_schema"])
        self.assertTrue(all(d["thinking"] == [{"text": "the summary"}] for d in docs))
        self.assertTrue(all(d["whole"] is True for d in docs))  # no table exists: the words are kept whole

    def test_a_prompt_of_the_harness_is_a_file(self):
        from dclab_rnd import prompts

        from pathlib import Path

        source = Path(no_data_harness.__file__).read_text(encoding="utf-8")
        for name in ("no_data_expert_brief", "no_data_compact_json", "no_data_workflow_repair", "no_data_schema_brief"):
            self.assertIn(name, prompts.names())
            self.assertIn(f'"{name}"', source)
        self.assertNotIn("extra_body", source)
        self.assertNotIn("Output ONLY ONE compact", source)  # no instruction is built as a string in the code

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
        self.assertTrue(hrep["all_gates_passed"], hrep["gates"])
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
        # once a real table is shared, the blueprint no longer stands in for it: the specialists' own rules decide again
        late = A.CaseState(case=C.new("draft-late"), table=A.table_of("orders.csv", [{"name": "order_id", "kind": "text", "noticed": []}], 10))
        late.specialist_client = state.specialist_client
        late.no_data_solution = state.no_data_solution
        self.assertIn("error", A._consult(late, "leakage_auditor", "Check for leaks."))

    def test_the_evaluation_measures_every_arm_and_its_gate_can_fail(self):
        """The review of 22ad1bf8: the plain arm's scores were constants, the harness's safety read two constants, the domain
        words were the template's own columns, and nothing checked the family."""
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            res = no_data_eval.evaluate(split="dev")
            cases = no_data_eval.load_cases("dev")
            wrong = no_data_eval.evaluate(cases=[dict(c, expected_family="uplift", expected_inputs={"monthly_volume": 7}) for c in cases[:3]])
        self.assertEqual(res["gate_id"], "no_data_gate_v2")
        self.assertTrue(res["passed"], f"Failed gate checks: {res['checks']}")
        # with no model the plain arm is not measured: no constant, no canned reply, and nothing is compared with it
        plain = res["summaries"]["plain"]
        self.assertEqual((plain["measured"], plain["mean_domain_recall"], plain["ungrounded_number_rate"]), (0, None, None))
        self.assertIsNone(res["compare"]["domain_recall_minus_plain"])
        # every stated number of the cases is read, and the family is checked
        ai = res["summaries"]["ai_harnessed"]
        self.assertEqual(ai["economics_inputs_exact_rate"], 1.0)
        self.assertGreaterEqual(ai["family_right_rate"], 0.9)
        self.assertLess(ai["family_right_rate"], 1.0)  # ND-06 is read as a stream of sensor rows: reported, not hidden
        # a default is not understanding: with no model the harness's floor understands no more than the rules do
        self.assertEqual(res["compare"]["understanding_minus_rules"], 0.0)
        # the gate fails on wrong families and wrong numbers; an unmeasured check is None, never a pass
        self.assertFalse(wrong["passed"])
        self.assertEqual((wrong["checks"]["family_right"], wrong["checks"]["economics_inputs_exact"]), (False, False))
        self.assertFalse(no_data_eval.verdict({"ai_harnessed": {}})["passed"])
        # the domain words are written from the domain, never the generic template's own column names
        generic = {"entity", "decision", "tenure", "activity", "outcome", "timestamp", "rate", "record_id", "decision_timestamp", "segment", "historical", "settlement"}
        for case in cases:
            if case["id"] in ("ND-05", "ND-06", "ND-07", "ND-10"):
                self.assertFalse(generic & set(case["domain_keywords"]), case["id"])

    def test_the_plain_arm_is_scored_on_what_a_model_really_returned(self):
        case = no_data_eval.load_cases("dev")[0]
        gateway = _FakeGateway({"exam_plain": ["Collect tenure and usage per account, then compare churned and kept accounts. Expect 94.3% accuracy."]})
        row = no_data_eval._score_plain_arm(case, gateway=gateway)
        self.assertTrue(row["measured"])
        self.assertEqual(row["ungrounded_numbers"], 1)  # 94.3% is nobody's number
        self.assertGreater(row["domain_recall"], 0.0)
        self.assertIsNone(row["understanding_completeness"])  # free text has no provenance fields: not scored, not guessed
        silent = no_data_eval._score_plain_arm(case, gateway=_FakeGateway({"exam_plain": [""]}))
        self.assertFalse(silent["measured"])


if __name__ == "__main__":
    unittest.main()
