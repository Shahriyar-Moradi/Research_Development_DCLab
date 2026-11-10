import unittest
from unittest.mock import MagicMock
from dclab_core import case as C
from dclab_rnd.agents import case_agent as CA
from dclab_rnd.agents import case_session
from dclab_rnd.agents import runtime
from dclab_rnd.agents.registry import Registry, Tool
from dclab_rnd.settings import Settings


class TestGapA1ClearRequestTrap(unittest.TestCase):
    """Gap A1: Ensure Lead Agent acts first on clear requests and avoids clarifying questions on Turn 1."""

    def test_is_clear_on_arrival_with_table(self):
        table = {"table": "telco.csv", "rows": 1000, "columns": [{"name": "customer_id"}, {"name": "churn"}, {"name": "tenure"}]}
        case = C.new("c1", ["Predict churn on my customers before contract renewal"])
        self.assertTrue(C.is_clear_on_arrival(case, table))

    def test_is_clear_on_arrival_vague_request(self):
        table = {"table": "telco.csv", "rows": 1000, "columns": [{"name": "customer_id"}, {"name": "churn"}]}
        case = C.new("c2", ["Help me with my data"])
        self.assertFalse(C.is_clear_on_arrival(case, table))

    def test_is_clear_on_arrival_conceptual_advice(self):
        case = C.new("c3", ["Explain the difference between ROC AUC and PR AUC in theory"])
        self.assertTrue(C.is_clear_on_arrival(case, table=None))

    def test_readiness_may_propose_immediately_on_clear_arrival(self):
        table = {"table": "telco.csv", "rows": 1000, "columns": [{"name": "customer_id"}, {"name": "churn"}]}
        case = C.new("c4", ["Predict churn for our subscribers"])
        self.assertTrue(C.is_clear_on_arrival(case, table))
        r = C.readiness(case, table)
        self.assertTrue(r.may_propose)

    def test_clear_note_injected_on_turn_1_when_clear_on_arrival(self):
        table = {"table": "telco.csv", "rows": 1000, "columns": [{"name": "customer_id"}, {"name": "churn"}]}
        case = C.new("c5", [])
        state = CA.CaseState(case=case, table=table)
        client = MagicMock()
        client.complete.return_value = {
            "content": "Here are options",
            "tool_calls": [],
            "usage": {"input_tokens": 10, "output_tokens": 10},
            "assistant_message": {"role": "assistant", "content": "Here are options"}
        }
        CA.turn(state, client, "Predict churn on my customers before renewal")
        self.assertTrue(state.clear_on_arrival)
        built_messages = client.complete.call_args[0][0]
        sys_or_user = " ".join(str(m.get("content", "")) for m in built_messages)
        self.assertIn("crystal-clear on arrival", sys_or_user)
        self.assertIn("Do NOT ask clarifying questions", sys_or_user)

    def test_propose_allowed_on_turn_1_when_clear_on_arrival(self):
        table = {"table": "t", "rows": 100, "columns": [{"name": "x"}, {"name": "churn"}]}
        case = C.new("c6", ["Predict churn"])
        state = CA.CaseState(case=case, table=table, clear_on_arrival=True)
        options = [
            {"title": "Option 1", "what": "w", "gets": "g", "tradeoff": "t", "effort": "e", "track": "prediction", "tool": "ml_project"},
            {"title": "Option 2", "what": "w", "gets": "g", "tradeoff": "t", "effort": "e", "track": "analysis", "tool": "query"}
        ]
        out = CA._propose(state, options)
        self.assertIn("shown", out)
        self.assertEqual(len(out["shown"]), 2)


class TestGapA2RoutePrecedenceAndMissingData(unittest.TestCase):
    """Gap A2 & R-01: Verify route precedence, missing data handling, and chip clear/sync."""

    def test_data_work_recognized_without_table_when_track_named(self):
        case = CA.C.new("c7", ["I need to predict churn for next month"])
        state = CA.CaseState(case=case, table=None)
        self.assertTrue(CA.data_work(state))

    def test_data_work_recognized_when_chip_needs_table(self):
        case = CA.C.new("c8", ["Can you help us?"])
        state = CA.CaseState(case=case, table=None, chip="prediction")
        self.assertTrue(CA.data_work(state))

    def test_start_advice_refused_on_data_work(self):
        case = CA.C.new("c9", ["Build a predictive churn model"])
        state = CA.CaseState(case=case, table=None)
        out = CA._start(state, "advice", "cited_answer")
        self.assertIn("error", out)
        self.assertIn("data work", out["error"])

    def test_start_prediction_without_table_demands_table(self):
        case = CA.C.new("c10", ["Build a predictive churn model"])
        state = CA.CaseState(case=case, table=None)
        out = CA._start(state, "prediction", "ml_project")
        self.assertIn("error", out)
        self.assertIn("table", out["error"].lower())

    def test_chip_conflict_surfaced_once(self):
        case = CA.C.new("c11", ["Build a churn model"])
        state = CA.CaseState(case=case, chip="advice", chip_checked=False)
        note = CA.chip_note(state)
        self.assertIsNotNone(note)
        self.assertIn("propose", note.lower())

    def test_change_of_mind_on_turn_2_updates_route(self):
        case = CA.C.new("c12", ["I want advice on churn", "Actually, instead prepare and clean our customer table"])
        state = CA.CaseState(case=case, chip="advice")
        latest = CA.changed_mind(state)
        self.assertEqual(latest, "preparation")

    def test_case_session_clear_on_arrival_snapshot_restore(self):
        case = CA.C.new("c13", ["Predict churn"])
        state = CA.CaseState(case=case, clear_on_arrival=True)
        snap = case_session.snapshot(state)
        self.assertTrue(snap.get("clear_on_arrival"))
        restored_state = CA.CaseState(case=case)
        case_session.restore(restored_state, snap)
        self.assertTrue(restored_state.clear_on_arrival)


class TestGapA3PerformanceAndLoopGuard(unittest.TestCase):
    """Gap A3: Loop guard against max_output_tokens runaway and cap settings."""

    def test_cap_eur_headroom(self):
        self.assertEqual(case_session.CAP_EUR, 1.50)
        settings = Settings()
        self.assertEqual(settings.case_cap_eur, 1.50)

    def test_runtime_loop_guard_on_consecutive_length_hits(self):
        registry = Registry()
        registry.register(Tool(name="step_tool", description="d", parameters={"type": "object", "properties": {}}, handler=lambda: {"done": True}))
        policy = runtime.Policy(name="test_p", system="", tools=("step_tool",), max_steps=10, max_replies=5)

        class MockLengthClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, schemas=None):
                self.calls += 1
                return {
                    "content": f"Partial response #{self.calls}",
                    "tool_calls": [{"id": f"call_{self.calls}", "name": "step_tool", "arguments": {}}],
                    "finish_reason": "length",
                    "usage": {"input_tokens": 100, "output_tokens": 8000},
                    "assistant_message": {"role": "assistant", "content": f"Partial response #{self.calls}"}
                }

        client = MockLengthClient()
        messages = [{"role": "user", "content": "Tell me everything"}]
        result = runtime.run(policy, client, registry, messages)

        # Loop guard should stop after 2 consecutive length hits
        self.assertEqual(client.calls, 2)
        self.assertEqual(result.stopped, "max_output_tokens")
        self.assertIn("Partial response", result.final)


if __name__ == "__main__":
    unittest.main()
