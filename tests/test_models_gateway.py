"""The model gateway (package A1.1): routing by purpose and tier, retries, the usage log, and no client outside it."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402
from dclab_rnd.models import FileUsage, Gateway, settings  # noqa: E402

CLEAN = {k: "" for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LOG_PROMPTS")} | {
    f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY")}


def transient(message="RateLimitError: the provider is limiting requests for now"):
    error = RuntimeError(message)
    error.transient = True
    return error


class Scripted:
    """A transport: plays a list of replies or errors, and remembers what tier and purpose it was built for."""
    built = []

    def __init__(self, plan, tier, purpose):
        self.plan, self.tier, self.purpose, self.calls = plan, tier, purpose, 0  # shared by every client of one gateway
        Scripted.built.append(self)

    def _next(self):
        self.calls += 1
        item = self.plan.pop(0) if self.plan else {"content": "ok", "tool_calls": [], "usage": {"input_tokens": 3, "output_tokens": 2}}
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        return self._next()

    def stream(self, messages, tools=None, max_tokens=1800, on_text=None):
        item = self.plan[0] if self.plan else None
        if isinstance(item, tuple):  # (text sent before failing, error)
            self.plan.pop(0)
            on_text(item[0])
            raise item[1]
        out = self._next()
        on_text(out.get("content") or "")
        return out


def gateway(usage, plan=()):
    sleeps, shared = [], list(plan)
    gw = Gateway(usage, transport=lambda tier, purpose: Scripted(shared, tier, purpose), sleep=sleeps.append)
    return gw, sleeps


class GatewayTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model"})
        env.start()
        self.addCleanup(env.stop)
        self.usage = FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl")
        Scripted.built = []

    def test_purposes_pick_their_tier_and_tiers_fall_back_to_standard(self):
        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_TIER_CHEAP_MODEL": "small", "DCLAB_TIER_CHEAP_API_KEY": "ollama"}):
            gw, _ = gateway(self.usage)
            gw.client("parse_pattern").complete([{"role": "user", "content": "x"}])
            gw.client("home_agent").complete([{"role": "user", "content": "x"}])
            self.assertEqual([(b.tier["name"], b.tier["model"]) for b in Scripted.built], [("cheap", "small"), ("standard", "std-model")])
            self.assertEqual(settings.tier("strong")["model"], "std-model")  # an unset tier is the standard one
            self.assertTrue(settings.public("cheap")["local"])
        with self.assertRaises(KeyError):
            gw.client("nonsense")

    def test_no_key_means_no_client_and_the_caller_takes_its_fallback(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            gw, _ = gateway(self.usage)
            self.assertIsNone(gw.client("home_agent"))
            self.assertFalse(gw.available("synthetic_schema"))
        self.assertEqual(self.usage.recent(), [])

    def test_every_request_is_logged_without_the_key_or_the_prompt(self):
        gw, _ = gateway(self.usage, [{"content": "a", "tool_calls": [], "usage": {"input_tokens": 120, "output_tokens": 30}}])
        gw.client("home_agent", draft_id="d1").complete([{"role": "user", "content": "my secret problem"}])
        entry = self.usage.recent()[0]
        self.assertEqual({k: entry[k] for k in ("purpose", "tier", "model", "draft_id", "input_tokens", "output_tokens", "attempts", "outcome")},
                         {"purpose": "home_agent", "tier": "standard", "model": "std-model", "draft_id": "d1", "input_tokens": 120, "output_tokens": 30, "attempts": 1, "outcome": "ok"})
        self.assertIsNone(entry["prompt"])
        everything = self.usage.path.read_text() + json.dumps(gw.summary())
        self.assertNotIn("sk-never-logged", everything)
        self.assertNotIn("my secret problem", everything)
        with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"}):
            gw.client("home_agent").complete([{"role": "user", "content": "logged on purpose"}])
        self.assertIn("logged on purpose", self.usage.recent()[0]["prompt"])

    def test_transient_failures_are_retried_with_backoff_and_counted_once(self):
        gw, sleeps = gateway(self.usage, [transient(), transient(), {"content": "fine", "tool_calls": [], "usage": {}}])
        self.assertEqual(gw.client("home_agent").complete([])["content"], "fine")
        self.assertEqual((len(self.usage.recent()), self.usage.recent()[0]["attempts"], self.usage.recent()[0]["outcome"]), (1, 3, "ok"))
        self.assertEqual(sleeps, [1.5, 3.0])

    def test_a_refusal_that_will_not_pass_is_not_retried(self):
        gw, sleeps = gateway(self.usage, [RuntimeError("RateLimitError: the model account has no credit left")])
        with self.assertRaises(RuntimeError):
            gw.client("home_agent").complete([])
        self.assertEqual((sleeps, self.usage.recent()[0]["attempts"], self.usage.recent()[0]["outcome"]), ([], 1, "the model account has no credit left"))
        gw, _ = gateway(self.usage, [transient(), transient(), transient()])
        with self.assertRaises(RuntimeError):
            gw.client("home_agent").complete([])
        self.assertEqual(self.usage.recent()[0]["attempts"], 3)  # the first try and two retries, then it stops

    def test_a_stream_is_retried_only_before_its_first_word(self):
        heard = []
        gw, _ = gateway(self.usage, [transient(), {"content": "hello", "tool_calls": [], "usage": {}}])
        gw.client("home_agent").stream([], on_text=heard.append)
        self.assertEqual(heard, ["hello"])
        heard.clear()
        gw, _ = gateway(self.usage, [("half a sent", transient())])
        with self.assertRaises(RuntimeError):
            gw.client("home_agent").stream([], on_text=heard.append)
        self.assertEqual(heard, ["half a sent"])  # never repeated to the user

    def test_totals_and_the_summary(self):
        gw, _ = gateway(self.usage, [{"content": "a", "usage": {"input_tokens": 10, "output_tokens": 5}}, RuntimeError("X: the provider refused the key")])
        gw.client("evidence_answer", project_id="p1").complete([])
        with self.assertRaises(RuntimeError):
            gw.client("home_agent").complete([])
        totals = self.usage.totals()
        self.assertEqual((totals["requests"], totals["failed"], totals["input_tokens"], totals["by_purpose"]["evidence_answer"]["output_tokens"]), (2, 1, 10, 5))
        self.assertEqual(self.usage.totals(project_id="p1")["requests"], 1)
        summary = gw.summary()
        self.assertEqual({p["purpose"] for p in summary["purposes"]}, set(settings.PURPOSES))
        self.assertTrue(all("may_see" in p and p["may_see"] for p in summary["purposes"]))
        self.assertTrue(all("api_key" not in t for t in summary["tiers"]))


class ReviewFindingsTests(unittest.TestCase):
    """One test per finding of the A1.1 review."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "DCLAB_INTERN_MODEL": "", "DCLAB_NO_LIVE_MODELS": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_the_old_single_model_setting_still_applies_to_every_tier(self):
        with mock.patch.dict(os.environ, {"DCLAB_LLM_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_INTERN_MODEL": "qwen2.5:7b"}):
            self.assertEqual({settings.tier(t)["model"] for t in settings.TIERS}, {"qwen2.5:7b"})
            self.assertEqual(settings.tier("standard")["api_key"], "local")  # a local server needs no key
        with mock.patch.dict(os.environ, {"DCLAB_INTERN_MODEL": "old", "DCLAB_TIER_CHEAP_MODEL": "new", "OPENAI_MODEL": "ignored"}):
            self.assertEqual((settings.tier("standard")["model"], settings.tier("cheap")["model"]), ("old", "new"))

    def test_a_key_follows_its_endpoint_and_never_goes_to_another_provider(self):
        with mock.patch.dict(os.environ, {"DCLAB_TIER_STANDARD_API_KEY": "std-key"}):
            self.assertEqual({settings.tier(t)["api_key"] for t in settings.TIERS}, {"std-key"})  # unset tiers inherit it
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai", "DCLAB_TIER_STRONG_BASE_URL": "https://router.huggingface.co/v1"}):
            self.assertEqual(settings.tier("strong")["api_key"], "")  # the OpenAI key is not sent to the router
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai", "DCLAB_TIER_STRONG_BASE_URL": "https://router.huggingface.co/v1", "DCLAB_TIER_STRONG_API_KEY": "hf_x"}):
            self.assertEqual(settings.tier("strong")["api_key"], "hf_x")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai", "DCLAB_TIER_CHEAP_BASE_URL": "https://api.openai.com/v1"}):
            self.assertEqual(settings.tier("cheap")["api_key"], "sk-openai")  # same host: the same key

    def test_tests_never_reach_a_live_model(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-real", "DCLAB_TIER_CHEAP_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_NO_LIVE_MODELS": "1"}):
            gw = Gateway(None)  # the real transport
            self.assertEqual([gw.client(p) for p in settings.PURPOSES], [None] * len(settings.PURPOSES))
            scripted = Gateway(None, transport=lambda tier, purpose: Scripted([], tier, purpose))
            self.assertIsNotNone(scripted.client("home_agent"))  # a scripted transport is allowed
        make = (ROOT / "Makefile").read_text()
        self.assertIn("rd-check test-pg product-e2e check-all: export DCLAB_NO_LIVE_MODELS = 1", make)
        self.assertIn('"DCLAB_NO_LIVE_MODELS": "1"', (ROOT / "scripts/product_e2e.py").read_text())

    def test_the_endpoint_never_shows_a_password(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", "DCLAB_LLM_BASE_URL": "https://user:pw@models.example.com:8443/v1"}):
            self.assertEqual(settings.public("standard")["endpoint"], "models.example.com:8443")
            usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
            Gateway(usage, transport=lambda tier, purpose: Scripted([], tier, purpose)).client("home_agent").complete([])
            self.assertNotIn("pw", usage.path.read_text())

    def test_the_token_limit_parameter_each_endpoint_accepts(self):
        from dclab_rnd.models.client import ChatClient
        self.assertEqual(ChatClient(model="m", base_url="https://api.openai.com/v1", api_key="k")._limit(400), {"max_completion_tokens": 400})
        self.assertEqual(ChatClient(model="m", base_url="http://127.0.0.1:11434/v1", api_key="k")._limit(400), {"max_tokens": 400})
        self.assertEqual(ChatClient(model="m", base_url="https://api.openai.com/v1", api_key="k")._limit(None), {})

    def test_failure_paths(self):
        class Broken:
            def record(self, entry):
                raise OSError("disk full")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            gw = Gateway(Broken(), transport=lambda tier, purpose: Scripted([{"content": "kept", "usage": {}}], tier, purpose))
            self.assertEqual(gw.client("home_agent").complete([])["content"], "kept")  # the answer survives a broken usage log
            paused = RuntimeError("ModelPaused: the model account has no credit left")  # no transient flag
            sleeps = []
            gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Scripted([paused], tier, purpose), sleep=sleeps.append)
            with self.assertRaises(RuntimeError):
                gw.client("home_agent").complete([])
            self.assertEqual(sleeps, [])

            class NoStream:
                def complete(self, messages, tools=None, max_tokens=1800, **options):
                    return {"content": "whole answer", "usage": {}}
            heard = []
            Gateway(None, transport=lambda tier, purpose: NoStream()).client("home_agent").stream([], on_text=heard.append)
            self.assertEqual(heard, ["whole answer"])

    def test_the_intern_counts_its_requests_against_the_sessions_project(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
            plan = [{"content": "", "tool_calls": [{"id": "c1", "name": "finish", "arguments": {"report": "done"}}], "usage": {"input_tokens": 5, "output_tokens": 1},
                     "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "finish", "arguments": "{\"report\": \"done\"}"}}]}}]
            gw = Gateway(usage, transport=lambda tier, purpose: Scripted(plan, tier, purpose))
            home = Path(tempfile.mkdtemp())
            projects = ProjectStore(home / "projects")
            pid = projects.create("P")["id"]
            intern = Intern(SessionStore(home / "intern"), Toolbox(projects), gw.client("intern"))
            session = intern.start("Finish at once", project_id=pid)
            intern.run(session["id"])
        self.assertEqual([(e["purpose"], e["project_id"]) for e in usage.recent()], [("intern", pid)])

    def test_every_noaa_client_reports_and_command_lines_install_a_logged_gateway(self):
        for rel in ("dclab_rnd/agentic/agents.py", "dclab_rnd/agentic/live_eval.py", "dclab_rnd/agentic/guide_review.py"):
            source = (ROOT / rel).read_text(encoding="utf-8")
            self.assertRegex(source, r"\(ResponsesClient\)")
            self.assertIn("installed().record(", source.replace(" ", ""), rel)
        for rel in ("dclab_rnd/agentic/__main__.py", "dclab_rnd/agentic/live_eval.py", "dclab_rnd/agentic/guide_review.py"):
            self.assertIn("install(for_workspace(", (ROOT / rel).read_text(encoding="utf-8"), rel)


class CallerPathTests(unittest.TestCase):
    """The callers that changed transport in A1.1, each through a scripted model."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "k", "DCLAB_INTERN_MODEL": ""})
        env.start()
        self.addCleanup(env.stop)
        self.usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
        self.seen = []
        outer = self

        class Recorder(Scripted):
            def complete(self, messages, tools=None, max_tokens=1800, **options):
                outer.seen.append((max_tokens, options))
                return super().complete(messages, tools, max_tokens, **options)
        self.gateway = Gateway(self.usage, transport=lambda tier, purpose: Recorder([{"content": json.dumps({"interpretation": "x", "overall_confidence": "low"}),
                                                                                     "usage": {"input_tokens": 9, "output_tokens": 4}}], tier, purpose))

    def test_stage_notes_and_project_answers_go_through_the_gateway(self):
        from dclab_rnd import models
        from dclab_rnd.studio import agent
        models.install(self.gateway)
        self.addCleanup(models.install, None)
        self.assertTrue(agent.llm_available())
        self.assertTrue(agent._chat("Explain the stage.", "stage_notes", "p1"))
        self.assertEqual([(e["purpose"], e["project_id"]) for e in self.usage.recent()], [("stage_notes", "p1")])

    def test_the_campaign_critic_keeps_its_options_and_has_no_token_cap(self):
        from dclab_rnd import llm_review
        root = Path(tempfile.mkdtemp())
        evidence = root / "evidence" / "r.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text(json.dumps({"experiment_id": "EXP-1", "metrics": {}}))
        client = self.gateway.client("campaign_review", model="critic-model")
        review = llm_review.review_one(client, root, {"evidence_path": "evidence/r.json", "experiment_id": "EXP-1"}, model="critic-model")
        self.assertEqual(self.seen, [(None, {"temperature": 0.2, "response_format": {"type": "json_object"}})])
        self.assertEqual((review["usage"]["prompt_tokens"], review["usage"]["total_tokens"]), (9, 13))
        self.assertEqual(self.usage.recent()[0]["model"], "critic-model")


class MoneyTests(unittest.TestCase):
    """Package A1.2: prices, costs and caps."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "k", "DCLAB_INTERN_MODEL": "", "OPENAI_MODEL": "priced-model",
                                           "DCLAB_WORKSPACE_MONTHLY_EUR": "", "DCLAB_PRICES_FILE": ""})
        env.start()
        self.addCleanup(env.stop)
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "prices.json").write_text(json.dumps({"priced-model": {"input": 2.0, "output": 8.0, "as_of": "2026-10-01", "source": "test"}}))
        self.usage = FileUsage(self.dir / "u.jsonl")
        self.sent = []

    def gateway(self, project_cap=None, tokens=(500_000, 100_000)):
        outer = self

        class Counting(Scripted):
            def complete(self, messages, tools=None, max_tokens=1800, **options):
                outer.sent.append(1)
                return {"content": "ok", "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}
        return Gateway(self.usage, transport=lambda tier, purpose: Counting([], tier, purpose), project_cap=project_cap)

    def test_a_price_is_never_guessed(self):
        from dclab_rnd.models import prices
        self.assertEqual(prices.cost("unknown-model", False, 1000, 1000), (None, "no price"))
        self.assertEqual(prices.cost("unknown-model", True, 1000, 1000), (0.0, "local"))
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            self.assertEqual(prices.cost("priced-model", False, 500_000, 100_000), (1.8, "price"))  # 0.5 M x 2 + 0.1 M x 8
        (self.dir / "bad.json").write_text(json.dumps({"m": {"input": -1, "output": 1, "as_of": "2026-10-01"}}))
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "bad.json")}), self.assertRaises(ValueError):
            prices.load()

    def test_costs_are_logged_and_unpriced_requests_are_counted_in_tokens(self):
        self.gateway().client("home_agent").complete([])
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            self.gateway().client("home_agent", project_id="p1").complete([])
        first, second = list(reversed(self.usage.recent()))
        self.assertEqual((first["cost_eur"], first["cost_basis"]), (None, "no price"))
        self.assertEqual((second["cost_eur"], second["cost_basis"]), (1.8, "price"))
        t = self.usage.totals()
        self.assertEqual((t["eur"], t["priced"], t["unpriced"], t["by_project_eur"]), (1.8, 1, 1, {"p1": 1.8}))

    def test_a_request_that_could_pass_the_cap_is_refused_before_sending(self):
        # price 2 / 8 euros per million tokens; 30,000 characters of prompt and max_tokens 1000:
        # upper bound (30,000 / 3) x 2 + 1,000 x 8 = 0.028 euros; the scripted reply costs 9,000 x 2 + 1,000 x 8 = 0.026
        prompt = [{"role": "user", "content": "x" * 30_000}]
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            gw = self.gateway(project_cap=lambda pid: 0.05 if pid == "p1" else None, tokens=(9_000, 1_000))
            gw.client("home_agent", project_id="p1").complete(prompt, max_tokens=1000)  # 0 + 0.028 <= 0.05: sent, costs 0.026
            with self.assertRaises(RuntimeError) as caught:
                gw.client("home_agent", project_id="p1").complete(prompt, max_tokens=1000)  # 0.026 + 0.028 > 0.05: never sent
            self.assertIn("this project has used its budget: €0.03 of €0.05 this month, €0.02 left", str(caught.exception))
            self.assertEqual(len(self.sent), 1)
            self.assertTrue(self.usage.recent()[0]["outcome"].startswith("refused: this project has used its budget"))
            gw.client("home_agent", project_id="p2").complete(prompt, max_tokens=1000)  # another project is not affected
            with mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": "0.06"}):
                with self.assertRaises(RuntimeError) as ws:
                    gw.client("home_agent", project_id="p2").complete(prompt, max_tokens=1000)
                self.assertIn("the workspace has used its monthly budget", str(ws.exception))

    def test_requests_sent_at_the_same_time_cannot_together_pass_a_cap(self):
        import threading
        prompt = [{"role": "user", "content": "x" * 30_000}]
        gate, outcomes = threading.Event(), []

        class Slow(Scripted):
            def complete(self, messages, tools=None, max_tokens=1800, **options):
                gate.wait(60)
                return {"content": "ok", "usage": {"input_tokens": 9_000, "output_tokens": 1_000}}
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            gw = Gateway(self.usage, transport=lambda tier, purpose: Slow([], tier, purpose), project_cap=lambda pid: 0.05)

            def one():
                try:
                    gw.client("home_agent", project_id="p1").complete(prompt, max_tokens=1000)
                    outcomes.append("sent")
                except RuntimeError:
                    outcomes.append("refused")
            threads = [threading.Thread(target=one) for _ in range(3)]
            [t.start() for t in threads]
            import time
            deadline = time.time() + 60  # the other two are refused while the first is held (a deadline, not a sleep: the suite runs in parallel)
            while time.time() < deadline and outcomes.count("refused") < 2:
                time.sleep(0.05)
            gate.set()
            [t.join(60) for t in threads]
        self.assertEqual(sorted(outcomes), ["refused", "refused", "sent"])  # one 0.028 hold fits under 0.05; a second does not
        self.assertEqual(gw._held, {"p:p1": 0.0})

    def test_a_run_cap_stops_one_session(self):
        prompt = [{"role": "user", "content": "x" * 30_000}]
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            client = self.gateway(tokens=(9_000, 1_000)).client("intern")
            client.run_limit_eur = 0.05
            client.complete(prompt, max_tokens=1000)
            with self.assertRaises(RuntimeError) as caught:
                client.complete(prompt, max_tokens=1000)
            self.assertIn("this run has used its budget", str(caught.exception))
            self.assertFalse(getattr(caught.exception, "transient", True))  # never retried
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            local = self.gateway().client("intern")  # a free (local) model is never stopped by a zero cap
            local.run_limit_eur = 0.0
            local.complete([{"role": "user", "content": "x" * 30_000}], max_tokens=1000)
            self.assertEqual(self.usage.recent()[0]["cost_basis"], "local")

    def test_a_broken_price_file_stops_before_sending_and_never_loses_an_answer(self):
        (self.dir / "broken.json").write_text(json.dumps({"priced-model": {"input": 1, "output": 1}}))  # no as_of
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "broken.json")}):
            with self.assertRaises(RuntimeError) as caught:
                self.gateway().client("home_agent").complete([])
            self.assertIn("DCLAB_PRICES_FILE is not valid", str(caught.exception))
            self.assertEqual(self.sent, [])
            eur = Gateway(self.usage).record("campaign", "priced-model", {"input_tokens": 1, "output_tokens": 1}, 0.1, 1, "ok")
            self.assertIsNone(eur)  # recording after a request never raises

    def test_local_means_this_machine(self):
        for url, local in (("http://127.0.0.1:11434/v1", True), ("http://localhost:1234/v1", True), ("http://[::1]:8000/v1", True),
                           ("https://api.example.com:11434/v1", False), ("https://localhost.example.com/v1", False),
                           ("https://proxy.example.com/v1?via=127.0.0.1", False)):
            self.assertEqual(settings.is_local(url), local, url)
        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_BASE_URL": "http://10.0.0.5:11434/v1", "DCLAB_TIER_CHEAP_LOCAL": "1"}):
            self.assertTrue(settings.public("cheap")["local"])  # a server on the private network, declared local

    def test_clients_the_gateway_does_not_send_for_are_stopped_by_a_reached_cap(self):
        from dclab_rnd import models
        from dclab_rnd.models.gateway import check_external
        from dclab_rnd.models.usage import month_start
        self.usage.record({"at": month_start(), "purpose": "campaign", "tier": "strong", "model": "m", "endpoint": "e", "input_tokens": 1,
                           "output_tokens": 1, "seconds": 0, "attempts": 1, "outcome": "ok", "cost_eur": 2.0, "cost_basis": "price"})
        models.install(Gateway(self.usage))
        self.addCleanup(models.install, None)
        check_external("campaign")  # no workspace cap: allowed
        with mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": "2"}), self.assertRaises(RuntimeError):
            check_external("campaign")
        Gateway(self.usage).record("campaign", "gpt-x", {"input_tokens": 1, "output_tokens": 1}, 0.1, 1, "ok", base_url="https://api.openai.com/v1")
        self.assertEqual((self.usage.recent()[0]["endpoint"], self.usage.recent()[0]["cost_basis"]), ("api.openai.com", "no price"))

    def test_the_intern_keeps_its_spend_across_turns_and_stops_as_budget_used_up(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore
        reply = {"content": "Not done yet.", "tool_calls": [], "usage": {"input_tokens": 9_000, "output_tokens": 1_000},
                 "assistant_message": {"role": "assistant", "content": "Not done yet."}}
        # a first turn looks at something before it answers (an answer with no tool call first is not a report: 8.1)
        look = {"content": "", "tool_calls": [{"id": "c1", "name": "list_samples", "arguments": {}}], "usage": {"input_tokens": 100, "output_tokens": 10},
                "assistant_message": {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "list_samples", "arguments": "{}"}}]}}
        with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(self.dir / "prices.json")}):
            gw = Gateway(self.usage, transport=lambda tier, purpose: Scripted([dict(look)] + [dict(reply) for _ in range(5)], tier, purpose))
            home = Path(tempfile.mkdtemp())
            sessions = SessionStore(home / "intern")
            s = Intern(sessions, Toolbox(ProjectStore(home / "projects")), gw.client("intern")).start("A task for the budget test", budget={"max_eur": 0.04})
            first = Intern(sessions, Toolbox(ProjectStore(home / "projects")), gw.client("intern")).run(s["id"])
            self.assertEqual(first["status"], "completed")
            self.assertGreater(first["used"]["eur"], 0)
            later = Intern(sessions, Toolbox(ProjectStore(home / "projects")), gw.client("intern")).message(s["id"], "x" * 60_000)  # a new client
            self.assertEqual(later["status"], "budget_exhausted")  # the earlier spend still counts
            self.assertIn("this run has used its budget", later["final"])
        for bad in ("abc", "nan", float("inf")):
            with self.assertRaises(ValueError):
                sessions.create("task", mode="llm", model=None, budget={"max_eur": bad})
        self.assertEqual(sessions.create("task", mode="llm", model=None, budget={"max_eur": 5000})["budget"]["max_eur"], 1000.0)

    def test_spent_is_the_same_on_both_backends(self):
        from dclab_rnd.models.usage import month_start
        entry = {"at": month_start(), "purpose": "home_agent", "tier": "standard", "model": "m", "endpoint": "e", "project_id": "p1",
                 "input_tokens": 1, "output_tokens": 1, "seconds": 0, "attempts": 1, "outcome": "ok", "cost_eur": 0.25, "cost_basis": "price"}
        self.usage.record(entry)
        self.usage.record({**entry, "project_id": "p2", "cost_eur": None, "cost_basis": "no price"})
        self.assertEqual((self.usage.spent(month_start()), self.usage.spent(month_start(), "p1"), self.usage.spent("2999-01")), (0.25, 0.25, 0.0))
        url = pgtest.url()
        if url:
            pgtest.empty(url)
            from dclab_rnd.models import PgUsage
            from dclab_rnd.storage import db
            pg = PgUsage(db.workspace("/m", "m", url), url)
            pg.record(entry)
            pg.record({**entry, "project_id": "p2", "cost_eur": None, "cost_basis": "no price"})
            self.assertEqual((pg.spent(month_start()), pg.spent(month_start(), "p1"), pg.totals()["unpriced"]), (0.25, 0.25, 1))


class OutputCheckTests(unittest.TestCase):
    """Package A1.3: a model that fails a purpose's output check twice is reported, with the purpose and tier."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "k", "OPENAI_MODEL": "m1", "DCLAB_INTERN_MODEL": ""})
        env.start()
        self.addCleanup(env.stop)

    def check_reporting(self, usage):
        gw = Gateway(usage, transport=lambda tier, purpose: Scripted([], tier, purpose))
        client = gw.client("parse_pattern")
        self.assertEqual(gw.summary()["failing"], [])
        client.output(True)
        client.output(False, "read only 10% of the sample lines")
        self.assertEqual(gw.summary()["failing"], [])  # one failure is not reported
        client.output(False, "not a valid regular expression")
        gw.client("home_agent").output(False, "the tool call was refused")  # another purpose, once
        failing = gw.summary()["failing"]
        self.assertEqual([(f["purpose"], f["tier"], f["model"], f["failed"], f["checked"]) for f in failing], [("parse_pattern", "cheap", "m1", 2, 3)])
        self.assertEqual(failing[0]["last_reason"], "not a valid regular expression")

    def test_failures_are_reported_from_two_on_files(self):
        self.check_reporting(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"))

    def test_failures_are_reported_from_two_in_postgresql(self):
        url = pgtest.require()
        pgtest.empty(url)
        from dclab_rnd.models import PgUsage
        from dclab_rnd.storage import db
        self.check_reporting(PgUsage(db.workspace("/checks", "c", url), url))

    def test_the_log_never_keeps_the_models_output(self):
        from dclab_rnd.draft import synthetic
        usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
        bad = json.dumps({"name": "t", "description": "d", "seed": 1, "time_order": False, "target": None,
                          "columns": [{"name": "patient_ssn_Q7Kxsecret", "type": "categorical", "description": "x", "null_rate": 0, "categories": ["a", "a"]}]})
        gw = Gateway(usage, transport=lambda tier, purpose: Scripted([{"content": bad, "usage": {}}, {"content": bad, "usage": {}}], tier, purpose))
        synthetic.spec_from_model(gw.client("synthetic_schema"), "x", None, 200)
        self.assertNotIn("Q7Kxsecret", usage.checks.read_text())
        self.assertEqual(gw.summary()["failing"][0]["purpose"], "synthetic_schema")

    def test_the_intern_reports_refused_and_unknown_tool_calls(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore

        def call(name, args, n):
            return {"content": "", "tool_calls": [{"id": f"c{n}", "name": name, "arguments": args}], "usage": {"input_tokens": 0, "output_tokens": 0},
                    "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}
        usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
        plan = [call("no_such_tool", {}, 1), call("get_results", {"project_id": "000000000000", "stage": "final"}, 2), call("list_samples", {}, 3), call("finish", {"report": "done"}, 4)]
        gw = Gateway(usage, transport=lambda tier, purpose: Scripted(plan, tier, purpose))
        home = Path(tempfile.mkdtemp())
        intern = Intern(SessionStore(home / "intern"), Toolbox(ProjectStore(home / "projects")), gw.client("intern"))
        intern.run(intern.start("Try some tools")["id"])
        rows = [json.loads(l) for l in usage.checks.read_text().splitlines()]
        self.assertEqual([(r["passed"], r["reason"]) for r in rows], [(False, "called a tool that does not exist"), (False, "the tool call was refused"), (True, None)])

    def test_the_critic_passes_only_a_json_object(self):
        from dclab_rnd import llm_review
        root = Path(tempfile.mkdtemp())
        (root / "r.json").write_text(json.dumps({"experiment_id": "E"}))
        usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
        gw = Gateway(usage, transport=lambda tier, purpose: Scripted([{"content": "[1, 2]", "usage": {}}], tier, purpose))
        with self.assertRaises(ValueError):
            llm_review.review_one(gw.client("campaign_review"), root, {"evidence_path": "r.json"}, model="m1")
        self.assertEqual(json.loads(usage.checks.read_text())["reason"], "the critique is not a JSON object")


class PgUsageTests(unittest.TestCase):
    def test_rows_in_postgresql_per_workspace(self):
        url = pgtest.require()
        pgtest.empty(url)
        from dclab_rnd.models import PgUsage
        from dclab_rnd.storage import db
        mine, theirs = PgUsage(db.workspace("/a", "a", url), url), PgUsage(db.workspace("/b", "b", url), url)
        mine.record({"at": "2026-10-06T10:00:00+00:00", "purpose": "home_agent", "tier": "standard", "model": "m", "endpoint": "api.openai.com",
                     "input_tokens": 7, "output_tokens": 3, "seconds": 0.4, "attempts": 1, "outcome": "ok"})
        self.assertEqual((mine.totals()["requests"], theirs.totals()["requests"]), (1, 0))
        self.assertEqual(mine.recent()[0]["input_tokens"], 7)
        self.assertEqual(mine.totals(since="2026-10-07")["requests"], 0)


class NoClientOutsideTheGatewayTests(unittest.TestCase):
    # The exceptions: three research tools run on NOOA's own HTTP client (ResponsesClient), which cannot be swapped;
    # each reports every request to the gateway's usage log instead (test_every_noaa_client_reports...).
    ALLOWED = ("dclab_rnd/models/", "dclab_rnd/agentic/agents.py", "dclab_rnd/agentic/live_eval.py", "dclab_rnd/agentic/guide_review.py")
    INTERN_LINE = "\x00"  # no line is excused any more
    PATTERN = re.compile(r"openai\.OpenAI\(|ChatClient\(|\.responses\.create\(|\bResponsesClient\b|from openai import|import openai")

    def test_only_the_gateway_makes_model_clients(self):
        found = []
        for path in sorted((ROOT / "dclab_rnd").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(self.ALLOWED):
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self.PATTERN.search(line) and self.INTERN_LINE not in line:
                    found.append(f"{rel}:{number}: {line.strip()[:100]}")
        self.assertEqual(found, [], "send model requests through dclab_rnd.models.Gateway")


class AdapterTests(unittest.TestCase):
    def test_the_campaign_adapter_reports_to_the_usage_log(self):
        source = (ROOT / "dclab_rnd/agentic/agents.py").read_text(encoding="utf-8")
        self.assertIn('installed().record("campaign"', source)  # NOOA runs only in .venv-agent; the reporting is checked here
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "k"}):
            usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
            Gateway(usage).record("campaign", "gpt-x", {"input_tokens": 11, "output_tokens": 4}, 0.5, 1, "ok")
        self.assertEqual({k: usage.recent()[0][k] for k in ("purpose", "tier", "model", "input_tokens")}, {"purpose": "campaign", "tier": "strong", "model": "gpt-x", "input_tokens": 11})

    def test_the_intern_and_the_critic_use_the_gateway(self):
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "k", "DCLAB_INTERN_MODEL": "intern-model"}):
            usage = FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")
            gw = Gateway(usage, transport=lambda tier, purpose: Scripted([], tier, purpose))
            client = gw.client("intern")
            self.assertEqual((client.model, client.p.tier), ("intern-model", "strong"))  # the intern's old setting still applies
            client.complete([{"role": "user", "content": "x"}], [])
            critic = gw.client("campaign_review", model="chosen-on-the-command-line")
            critic.complete([], max_tokens=10, temperature=0.2, response_format={"type": "json_object"})
        self.assertEqual([(e["purpose"], e["model"]) for e in reversed(usage.recent())], [("intern", "intern-model"), ("campaign_review", "chosen-on-the-command-line")])


class ApiTests(unittest.TestCase):
    """The product's own model uses, end to end, with a scripted model behind the real gateway."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-never-logged"})
        env.start()
        self.addCleanup(env.stop)
        self.app = create_app(Path(tempfile.mkdtemp()))
        self.app.state.models.transport = lambda tier, purpose: Scripted([], tier, purpose)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_the_home_flow_and_the_evidence_answers_are_counted_by_purpose(self):
        import time
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        deadline = time.time() + 60
        while time.time() < deadline and not c.get(f"/api/drafts/{d['id']}").json()["questions"]:
            time.sleep(0.2)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/messages", json={"text": "Customers who cancel within 30 days"}, headers=self.h).status_code, 202)
        c.post("/api/evidence/ask", json={"question": "Should I oversample before splitting?"}, headers=self.h)
        while time.time() < deadline and "home_agent" not in (c.get("/api/models").json()["usage"] or {}).get("by_purpose", {}):
            time.sleep(0.2)
        overview = c.get("/api/models").json()
        self.assertEqual(set(overview["usage"]["by_purpose"]), {"home_agent", "evidence_answer"})
        self.assertNotIn("sk-never-logged", json.dumps(overview))


if __name__ == "__main__":
    unittest.main()
