"""Shadow mode and a reviewed switch (package A6.4): a shadow answers beside the served model, is logged, never used."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import FileUsage, Gateway, settings  # noqa: E402
from dclab_rnd.models.routing import RoutingError, change, open_routing  # noqa: E402
from dclab_rnd.models.shadow import agreement, open_shadows  # noqa: E402

CLEAN = {k: "" for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LOG_PROMPTS")} | {
    f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.ROUTABLE for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}
TUNED = {"DCLAB_TIER_TUNED_BASE_URL": "http://127.0.0.1:8001/v1", "DCLAB_TIER_TUNED_MODEL": "dclab-policy-lora"}


def reply(content=None, calls=()):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": content, "tool_calls": calls, "usage": {"input_tokens": 10, "output_tokens": 3},
            "assistant_message": {"role": "assistant", "content": content,
                                  "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Router:
    """One scripted model per tier: the served one plays ``served``, the tuned one plays ``tuned`` (shared lists)."""

    def __init__(self, served, tuned):
        self.plans, self.seen = {"tuned": tuned}, []
        self.served = served

    def __call__(self, tier, purpose):
        router = self

        class Transport:
            def complete(self, messages, tools=None, max_tokens=None, **_):
                router.seen.append((tier["name"], [m.get("role") for m in messages]))
                plan = router.plans["tuned"] if tier["name"] == "tuned" else router.served
                item = plan.pop(0) if plan else reply("done")
                if isinstance(item, Exception):
                    raise item
                return item
        return Transport()


class Clock:
    """Seven days of traffic: each request is stamped one hour after the last."""

    def __init__(self):
        self.t = datetime(2026, 9, 30, 23, tzinfo=timezone.utc)  # the first request at midnight on day one

    def __call__(self):
        self.t += timedelta(hours=1)
        return self.t.isoformat(timespec="seconds")


class ShadowTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model", **TUNED})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.usage = FileUsage(self.home / "model_requests.jsonl")
        self.routing, self.shadows = open_routing(self.home), open_shadows(self.home)  # files, or PostgreSQL under test-pg

    def gateway(self, router, clock=None):
        return Gateway(self.usage, transport=router, routing=self.routing, shadows=self.shadows, shadow_runner=lambda fn: fn(), clock=clock)

    def test_a_week_of_scripted_traffic_logs_the_disagreements_and_returns_only_the_served_answers(self):
        week = 7 * 24
        served = [reply(calls=[("describe_data", {"project_id": "p1"})]) for _ in range(week)]
        tuned = [reply(calls=[("describe_data", {"project_id": "p1"})]) if i % 4 else reply(calls=[("set_solution", {"project_id": "p1", "target": "x"})])
                 for i in range(week)]
        tuned[5] = RuntimeError("the tuned server is down")
        router = Router(served, tuned)
        change(self.routing, "intern", "shadow", "tuned", None)
        client = self.gateway(router, Clock()).client("intern", project_id="p1")
        for _ in range(week):
            messages = [{"role": "user", "content": "Audit the data"}]
            out = client.complete(messages, tools=[{"type": "function", "function": {"name": "describe_data"}}])
            self.assertEqual(out["tool_calls"][0]["name"], "describe_data")  # always the served model's answer
            messages.append(out["assistant_message"])  # the caller's list grows after the answer; the shadow had a copy
        rows = self.shadows.rows()
        self.assertEqual(len(rows), week)
        self.assertEqual(len({r["at"][:10] for r in rows}), 7)  # a week
        a, = agreement(rows)
        self.assertEqual((a["purpose"], a["shadow_model"], a["requests"], a["errors"]), ("intern", "dclab-policy-lora", week, 1))
        self.assertEqual(a["disagreements"], 42)  # every fourth answer chose another tool
        self.assertEqual(len(a["by_day"]), 7)
        self.assertFalse(any("Audit the data" in json.dumps(r) or "target" in json.dumps(r) for r in rows))  # no prompt, no argument
        used = [json.loads(line) for line in (self.home / "model_requests.jsonl").read_text().splitlines()]
        self.assertEqual({(u["tier"], u["model"]) for u in used}, {("strong", "std-model")})  # no shadow request in usage or budgets
        self.assertEqual(len(used), week)
        self.assertTrue(all(roles == ["user"] for tier, roles in router.seen if tier == "tuned"))
        self.assertEqual({(r["primary_tool"], r["shadow_tool"]) for r in rows if r["same_tool"] is False}, {("describe_data", "other")})  # not offered: no name

    def test_the_validator_sees_only_the_served_models_moves(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore

        projects = ProjectStore(self.home / "projects")
        pid = projects.create("t", goal="Predict churn")["id"]
        served = [reply(calls=[("get_graph", {"project_id": pid})]), reply(calls=[("finish", {"report": "Looked."})])]
        tuned = [reply(calls=[("run_stage", {"project_id": pid, "stage": "data"})]), reply(calls=[("set_settings", {"project_id": pid, "quick": True})])]
        change(self.routing, "intern", "shadow", "tuned", None)
        gw = self.gateway(Router(served, tuned))
        intern = Intern(SessionStore(self.home / "intern"), Toolbox(projects), gw.client("intern"))
        done = intern.run(intern.start("Look at the project", project_id=pid)["id"])
        self.assertEqual([s["tool"] for s in done["steps"]], ["get_graph", "finish"])
        moves = [t["move"] for t in projects.transitions(pid, 100)]
        self.assertNotIn("run_stage", moves)  # the shadow proposed it; nothing ran it, nothing validated it
        self.assertFalse(projects.get(pid)["settings"]["quick"])
        self.assertEqual(len(self.shadows.rows()), 2)

    def test_moving_a_purpose_needs_a_reviewer_and_moving_it_back_is_one_setting(self):
        with self.assertRaises(PermissionError):
            change(self.routing, "intern", "serve", "tuned", None, "it agreed for a week")
        with self.assertRaises(RoutingError):
            change(self.routing, "intern", "serve", "tuned", "reviewer", "")  # a reason is required
        change(self.routing, "intern", "serve", "tuned", "reviewer", "It agreed 98% for a week.")
        router = Router([], [reply("from the tuned model")])
        gw = self.gateway(router)
        self.assertEqual(gw.client("intern").complete([{"role": "user", "content": "x"}])["content"], "from the tuned model")
        self.assertEqual(json.loads((self.home / "model_requests.jsonl").read_text().splitlines()[-1])["tier"], "tuned")
        change(self.routing, "intern", "serve", None, None)  # back, by anyone
        self.assertEqual(gw.route("intern"), ("strong", None))
        history = self.routing.get()["history"]
        self.assertEqual([(h["setting"], h["to"], h["by"]) for h in history], [("serve", "tuned", "reviewer"), ("serve", None, "human")])

    def test_an_approval_is_for_one_model(self):
        change(self.routing, "intern", "serve", "tuned", "reviewer", "lora-A agreed 98% for a week")
        gw = self.gateway(Router([reply("strong answer")], [reply("tuned answer")]))
        self.assertEqual(gw.route("intern")[0], "tuned")
        self.assertEqual(self.routing.get()["history"][-1]["model"], "dclab-policy-lora")  # the audit names the approved model
        with mock.patch.dict(os.environ, {"DCLAB_TIER_TUNED_MODEL": "lora-B-never-shadowed"}):
            self.assertEqual(gw.route("intern")[0], "strong")  # another model under the same tier name is not approved
            summary = next(p for p in gw.summary()["purposes"] if p["purpose"] == "intern")
            self.assertEqual((summary["serving"], summary["approval_holds"]), ("strong", False))
        with mock.patch.dict(os.environ, {"DCLAB_TIER_TUNED_BASE_URL": "", "DCLAB_TIER_TUNED_MODEL": ""}):
            with self.assertRaises(RoutingError):
                change(self.routing, "home_agent", "serve", "tuned", "reviewer", "no model to approve")

    def test_the_live_suite_plans_with_the_model_that_serves(self):
        from dclab_rnd.agent_eval import live
        from dclab_rnd.agent_eval.cases import CASES
        from dclab_rnd.models.gateway import for_workspace

        change(self.routing, "intern", "serve", "tuned", "reviewer", "approved for the benchmark")
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": ""}):  # plan() sends nothing; the real transport must count as able to answer
            p = live.plan(for_workspace(self.home), CASES[:1], 1, 4)
        self.assertEqual((p["tier"], p["model"], p["local"]), ("tuned", "dclab-policy-lora", True))  # not the intern's own strong tier

    def test_the_campaigns_purposes_are_not_routed(self):
        for purpose in ("campaign", "campaign_review"):
            with self.assertRaises(RoutingError):
                change(self.routing, purpose, "shadow", "tuned", None)  # their requests do not go through the gateway

    def test_the_default_runner_shadows_in_the_background(self):
        import time
        change(self.routing, "intern", "shadow", "tuned", None)
        gw = Gateway(self.usage, transport=Router([reply("served")], [reply("shadow")]), routing=self.routing, shadows=self.shadows)
        messages = [{"role": "user", "content": "x"}]
        self.assertEqual(gw.client("intern").complete(messages)["content"], "served")
        messages.append({"role": "assistant", "content": "served"})  # after the answer: the shadow's copy is unchanged
        deadline = time.time() + 30
        while time.time() < deadline and not self.shadows.rows():
            time.sleep(0.05)
        row, = self.shadows.rows()
        self.assertEqual((row["primary_tool"], row["shadow_tool"], row["outcome"]), ("text", "text", "ok"))

    def test_a_long_lived_client_follows_the_routing(self):
        gw = self.gateway(Router([reply("strong answer")], [reply("tuned answer")]))
        client = gw.client("intern")  # the server builds the intern's client once
        change(self.routing, "intern", "serve", "tuned", "owner", "approved")
        self.assertEqual(client.complete([{"role": "user", "content": "x"}])["content"], "tuned answer")

    def test_only_a_local_model_can_shadow_and_a_missing_tuned_model_is_never_used(self):
        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_BASE_URL": "https://api.example.com/v1", "DCLAB_TIER_CHEAP_API_KEY": "k"}):
            with self.assertRaises(RoutingError):
                change(self.routing, "intern", "shadow", "cheap", None)  # a paid shadow would spend money no cap sees
        change(self.routing, "intern", "serve", "tuned", "reviewer", "approved")
        with mock.patch.dict(os.environ, {"DCLAB_TIER_TUNED_BASE_URL": "", "DCLAB_TIER_TUNED_MODEL": ""}):
            self.assertEqual(settings.tier("tuned")["model"], "")  # unset is absent: never the standard model in disguise
            self.assertEqual(self.gateway(Router([], [])).route("intern")[0], "strong")  # the purpose keeps its own model

    def test_a_failing_shadow_never_reaches_the_caller(self):
        change(self.routing, "home_agent", "shadow", "tuned", None)
        gw = self.gateway(Router([reply("served")], [RuntimeError("boom")]))
        self.assertEqual(gw.client("home_agent").complete([{"role": "user", "content": "x"}])["content"], "served")
        self.assertEqual(self.shadows.rows()[-1]["outcome"], "error: RuntimeError")


class ApiTests(unittest.TestCase):
    def test_the_admin_routes_and_the_audit(self):
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.server import create_app

        env = mock.patch.dict(os.environ, {**CLEAN, **TUNED, "OPENAI_API_KEY": "sk-never-logged"})  # the transport below is scripted
        env.start()
        self.addCleanup(env.stop)
        with TestClient(create_app(Path(tempfile.mkdtemp()))) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            summary = c.get("/api/models").json()
            self.assertEqual((summary["tuned"]["model"], summary["tuned"]["local"]), ("dclab-policy-lora", True))
            r = c.post("/api/models/routing", json={"purpose": "intern", "setting": "serve", "tier": "tuned", "reason": "agreed"}, headers=h)
            self.assertEqual(r.status_code, 403)
            r = c.post("/api/models/routing", json={"purpose": "intern", "setting": "serve", "tier": "tuned", "reason": "agreed for a week"},
                       headers={**h, "X-DCLab-Role": "reviewer"})
            self.assertEqual(r.status_code, 200, r.text)
            intern = next(p for p in r.json()["purposes"] if p["purpose"] == "intern")
            self.assertEqual(intern["routing"]["serve"], "tuned")
            self.assertEqual(c.post("/api/models/routing", json={"purpose": "nope", "setting": "shadow", "tier": "tuned"}, headers=h).status_code, 422)
            self.assertEqual(c.post("/api/models/routing", json={"purpose": "intern", "setting": "serve", "tier": None}, headers=h).status_code, 200)
            c.post("/api/models/routing", json={"purpose": "intern", "setting": "shadow", "tier": "tuned"}, headers=h)
            gw = c.app.state.models
            gw.transport, gw.shadow_runner = Router([reply("served")], [reply("shadow")]), (lambda fn: fn())
            gw.client("intern").complete([{"role": "user", "content": "x"}])
            agree = c.get("/api/models").json()["agreement"]
            self.assertEqual((agree[0]["purpose"], agree[0]["requests"]), ("intern", 1))
            self.assertEqual(c.post("/api/models/routing", json={"purpose": "campaign", "setting": "shadow", "tier": "tuned"}, headers=h).status_code, 422)
            audit = c.get("/api/platform/audit").json()
            routed = [e for e in audit["items"] if e["kind"] == "routing"]
            self.assertEqual([(e["args"]["to"], e["who"]) for e in routed], [("tuned", "Human"), (None, "Human"), ("tuned", "Reviewer")])
            self.assertEqual(audit["counts"]["routing_changes"], 3)


if __name__ == "__main__":
    unittest.main()
