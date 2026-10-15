"""Package 21.4: model management per role. Switching a role's model is one audited change naming the model, the prompt
version and the measurement; a role's monthly cap stops its requests; an understanding role moves to a stronger model
only when that model measured better on the same conversations (or the owner overrides, audited); the conversation
suite reports each role's score and cost by model; one page shows each role's model, prompt, score and cost."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import audit
    from dclab_rnd.agent_eval import conversation
    from dclab_rnd.models import routing, settings
    from dclab_rnd.models.usage import FileUsage, month_start
    from tests.test_models_gateway import CLEAN, gateway as scripted
except ImportError as error:  # the agent environment has no pydantic contracts
    MISSING = str(error)
else:
    MISSING = ""

KEY = "sk-" + "test-" + "y" * 8  # assembled: never a key-shaped literal
ENV = {"DCLAB_LLM_PROVIDER": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": KEY, "OPENAI_MODEL": "gpt-6-luna", "DCLAB_TIER_STRONG_MODEL": "big-model", "DCLAB_TIER_STRONG_API_KEY": KEY}


@unittest.skipIf(MISSING, MISSING)

def setUpModule():  # 21.5: these tests are about routing, not the harness suite: a suite run that passed is assumed
    from unittest import mock

    patcher = mock.patch("dclab_rnd.agent_eval.regression.passed_with", return_value={"id": "HRN-TEST"})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)

class SwitchTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, **ENV})
        env.start()
        self.addCleanup(env.stop)
        home = Path(tempfile.mkdtemp())
        self.store, self.audit = routing.FileRouting(home / "model_routing.json"), audit.open_audit(home)

    def rows(self):
        return [{**r, "args": (r.get("detail") or {}).get("args") or {}} for r in self.audit.query(100, 0)[1]]

    def test_switching_a_roles_model_is_one_audited_change(self):
        doc = routing.switch(self.store, self.audit, "reporter", "serve", "strong", "reviewer", "it writes better summaries")
        made = doc["history"][-1]
        self.assertEqual((made["purpose"], made["to"], made["model"]), ("reporter", "strong", "big-model"))
        self.assertEqual(made["prompt"], routing.prompt_of("reporter"))
        self.assertTrue(made["prompt"])  # the prompt version it was approved with
        row = next(r for r in self.rows() if r.get("move") == "set_routing")
        self.assertEqual((row["args"]["purpose"], row["args"]["model"], row["args"]["prompt"]), ("reporter", "big-model", made["prompt"]))
        routing.switch(self.store, self.audit, "reporter", "serve", "strong", "reviewer", "again")  # the same again: no change, no row
        self.assertEqual(sum(r.get("move") == "set_routing" for r in self.rows()), 1)
        with self.assertRaises(PermissionError):
            routing.switch(self.store, self.audit, "critic", "serve", "strong", "viewer", "why not")
        with self.assertRaises(routing.RoutingError):
            routing.switch(self.store, self.audit, "critic", "serve", "strong", "reviewer", "")

    def test_an_understanding_role_moves_to_a_stronger_model_only_when_it_measured_better(self):
        with self.assertRaises(routing.RoutingError):  # not measured
            routing.switch(self.store, self.audit, "problem_analyst", "serve", "strong", "reviewer", "it seems smarter")
        with self.assertRaises(routing.RoutingError):  # measured, not better
            routing.switch(self.store, self.audit, "problem_analyst", "serve", "strong", "reviewer", "close",
                           measured={"big-model": 0.8, "gpt-6-luna": 0.8})
        with self.assertRaises(routing.RoutingError):  # a reviewer cannot override
            routing.switch(self.store, self.audit, "problem_analyst", "serve", "strong", "reviewer", "trust me", override=True)
        doc = routing.switch(self.store, self.audit, "problem_analyst", "serve", "strong", "reviewer", "9 of 9 against 7 of 9",
                             measured={"big-model": 1.0, "gpt-6-luna": 0.78})
        self.assertEqual(doc["history"][-1]["evidence"], {"measured": {"big-model": 1.0, "gpt-6-luna": 0.78}, "override": False, "suite": "HRN-TEST"})
        doc = routing.switch(self.store, self.audit, "brief_checker", "serve", "strong", "owner", "the owner's call, measured later", override=True)
        self.assertTrue(doc["history"][-1]["evidence"]["override"])
        row = [r for r in self.rows() if r.get("move") == "set_routing"][0]
        self.assertIn("evidence", row["args"])
        routing.switch(self.store, self.audit, "reporter", "serve", "strong", "reviewer", "not an understanding role")  # no measurement needed

    def test_neither_a_step_down_nor_the_tuned_tier_gets_around_the_measurement(self):
        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_MODEL": "small-model", "DCLAB_TIER_CHEAP_API_KEY": KEY,
                                          "DCLAB_TIER_TUNED_MODEL": "any-model", "DCLAB_TIER_TUNED_API_KEY": KEY,
                                          "DCLAB_TIER_TUNED_BASE_URL": "https://tuned.example/v1"}):
            measured = {"big-model": 0.8, "gpt-6-luna": 0.9, "small-model": 0.7}
            routing.switch(self.store, self.audit, "conversation_reader", "serve", "cheap", "reviewer", "a step down needs no measure")
            with self.assertRaises(routing.RoutingError):  # the bar is its own tier's model, not the one it stepped down to
                routing.switch(self.store, self.audit, "conversation_reader", "serve", "strong", "reviewer", "up again", measured=measured)
            with self.assertRaises(routing.RoutingError):  # the tuned tier can be any model: measured like any other
                routing.switch(self.store, self.audit, "problem_analyst", "serve", "tuned", "reviewer", "tuned")
            routing.switch(self.store, self.audit, "problem_analyst", "serve", "tuned", "reviewer", "it measured better",
                           measured={"any-model": 0.95, "gpt-6-luna": 0.9})

    def test_a_cap_is_a_finite_number(self):
        for bad in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(routing.RoutingError):
                routing.switch(self.store, self.audit, "reporter", "cap", bad, "reviewer", "x")

    def test_a_roles_cap_stops_its_requests_and_not_the_others(self):
        usage = FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl")
        usage.record({"at": month_start(), "purpose": "reporter", "model": "gpt-6-luna", "cost_eur": 0.3, "outcome": "ok", "cost_basis": "price"})
        routing.switch(self.store, self.audit, "reporter", "cap", 0.3, "reviewer", "a month of summaries")
        self.assertEqual(routing.cap(self.store.get(), "reporter"), 0.3)
        gw, _ = scripted(usage, [{"content": "{}", "usage": {"input_tokens": 1, "output_tokens": 1}}] * 3)
        gw.routing = self.store
        with self.assertRaises(RuntimeError) as stopped:
            gw.client("reporter").complete([{"role": "user", "content": "summarise"}], None)
        self.assertIn("reporter has used its monthly budget", str(stopped.exception))
        self.assertEqual(gw.client("critic").complete([{"role": "user", "content": "review"}], None)["content"], "{}")
        with self.assertRaises(PermissionError):
            routing.set_cap(self.store, "reporter", 5.0, "editor", "more")
        self.assertEqual(usage.spent(month_start(), purpose="reporter"), 0.3)
        self.assertGreaterEqual(usage.totals(month_start())["by_purpose_model"]["reporter · gpt-6-luna"]["requests"], 1)


@unittest.skipIf(MISSING, MISSING)
class ConversationByModelTest(unittest.TestCase):
    def run_of(self, ident, split, model, right, prompt="p1", cost=0.01, which="nine"):
        return {"id": ident, "split": split, "arm": "reader", "summary": {"right": right, "conversations": 9}, "conversations": which,
                "roles": [{"role": "conversation_reader", "model": model, "tier": "standard", "prompt": prompt, "requests": 9, "cost_eur": cost}]}

    def test_models_are_compared_only_on_the_same_conversations_at_the_same_prompt(self):
        results = [self.run_of("C1", "dev", "small", 7), self.run_of("C2", "dev", "big", 9), self.run_of("C3", "sealed_v3", "big", 8),
                   self.run_of("C4", "dev", "big", 2, prompt="old"), self.run_of("C5", "dev", "small", 5, which="nine-older")]
        self.assertEqual(conversation.measured("conversation_reader", "p1", ["big", "small"], results), {"big": 1.0, "small": round(7 / 9, 4)})
        again = [self.run_of("C2", "dev", "big", 9), self.run_of("C6", "dev", "big", 5)]  # every run counts: a rerun cannot pick its luck
        self.assertEqual(conversation.measured("conversation_reader", "p1", ["big"], again), {"big": round((9 / 9 + 5 / 9) / 2, 4)})
        self.assertEqual(conversation.measured("conversation_reader", "p1", ["big", "other"], results), {})  # never measured together
        rows = conversation.by_model(results)
        self.assertEqual([(r["model"], r["right"], r["cost_eur"]) for r in rows][:2], [("small", 7, 0.01), ("big", 9, 0.01)])

    def test_a_stored_run_names_its_roles_and_its_conversations_and_the_report_reads_them(self):
        import io
        from contextlib import redirect_stdout

        from dclab_rnd.agent_eval.conversation import __main__ as cli

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(conversation, "RESULTS", Path(tmp)):
            roles = [{"role": "conversation_reader", "model": "big", "tier": "strong", "prompt": "p1", "requests": 3, "cost_eur": 0.002}]
            path = conversation.store([{"right": True, "questions": 2}], "dev", "reader", "pass", 0.002, roles)
            doc = json.loads(path.read_text())
            self.assertEqual((doc["roles"], doc["conversations"]), (roles, conversation.fingerprint(conversation.HERE / "conversations_dev.json")))
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--report"]), 0)
            self.assertIn("conversation_reader", out.getvalue())
            self.assertIn("big", out.getvalue())
            self.assertIn("€0.0020", out.getvalue())

    def test_the_advisor_reports_each_roles_model_prompt_and_cost(self):
        from dclab_rnd.core_advisor import GatewayAdvisor

        class Client:
            model, tier, spent_eur, run_limit_eur = "big", {"name": "strong"}, 0.002, None

            def complete(self, *a, **k):
                return {"content": '{"value": "binary"}'}

        class Gate:
            def client(self, purpose, **kw):
                self.asked = kw
                return Client()

        gate = Gate()
        advisor = GatewayAdvisor(gate, models={"conversation_reader": "big"})
        advisor.ask("conversation_reader", {"field": "family", "answer": "a yes or no"})
        self.assertEqual(gate.asked.get("model"), "big")  # the named model, for the measurement
        report = advisor.report()
        self.assertEqual((report[0]["role"], report[0]["model"], report[0]["tier"], report[0]["requests"], report[0]["cost_eur"]),
                         ("conversation_reader", "big", "strong", 1, 0.002))
        self.assertEqual(report[0]["prompt"], routing.prompt_of("conversation_reader"))


@unittest.skipIf(MISSING, MISSING)
class PageTest(unittest.TestCase):
    def test_one_page_shows_each_roles_model_prompt_score_and_cost_and_a_cap_is_one_audited_change(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        with mock.patch.dict(os.environ, {**CLEAN, **ENV}):
            client = TestClient(create_app(Path(tempfile.mkdtemp())))
            h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
            roles = {r["role"]: r for r in client.get("/api/models/roles").json()["roles"]}
            self.assertTrue({"problem_analyst", "conversation_reader", "reporter", "intern", "home_agent"} <= set(roles))
            reader = roles["conversation_reader"]
            self.assertEqual((reader["prompt"], reader["model"], reader["understanding"]), (routing.prompt_of("conversation_reader"), "gpt-6-luna", True))
            r = client.post("/api/models/routing", json={"purpose": "reporter", "setting": "cap", "cap_eur_month": 2.5, "reason": "a month"},
                            headers={**h, "X-DCLab-Role": "reviewer"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(client.get("/api/models/roles").json()["roles"][[*roles].index("reporter")]["cap_eur_month"], 2.5)
            r = client.post("/api/models/routing", json={"purpose": "problem_analyst", "setting": "serve", "tier": "strong", "reason": "bigger"},
                            headers={**h, "X-DCLab-Role": "reviewer"})
            self.assertEqual(r.status_code, 422)  # not measured better
            self.assertIn("measured better", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
