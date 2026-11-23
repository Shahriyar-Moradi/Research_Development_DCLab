"""B1 of the team workspace (2026-11-23): the brain switch.

Each team role runs with one brain, chosen by one audited switch (default: the rules). The rules arm is today's agents,
unchanged. A role switched to the helper or the agent arm asks its model once per turn, beside the rules and on the same
stored state; in B1 the rules always act, and a model that cannot answer (none configured, DCLAB_NO_LIVE_MODELS=1, a
timeout, an answer its check refuses) falls back to them with the reason recorded. Every turn that ran records which
brain answered (``team.brain.answered`` and the ledger point ``team.brain``), and every model call keeps its thinking.
On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that empties ``dclab_test``).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402
from test_teamspace_agents import _Contract  # noqa: E402  (the same workspace, project, problem and Business thread)

from dclab_core import team_brain as core  # noqa: E402
from dclab_rnd import experience  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402
from dclab_rnd.teamspace import brains  # noqa: E402

CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                         "DCLAB_THINKING", "DCLAB_TEAM_BRAIN")} | {"OPENAI_API_KEY": "k", "DCLAB_LOG_PROMPTS": "1"}


def said(content="", tool_calls=(), why="the task is open, so accepting it comes first"):
    """A scripted model reply with its thinking (the API's thinking summary)."""
    return {"content": content, "tool_calls": list(tool_calls), "usage": {"input_tokens": 3, "output_tokens": 2},
            "thinking": [{"text": why}], "finish_reason": "tool_calls" if tool_calls else "stop",
            "assistant_message": {"role": "assistant", "content": content}}


class Scripted:
    def __init__(self, *replies, delay=0.0):
        self.replies, self.thinking, self.seen, self.delay = list(replies), None, [], delay

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        self.seen.append({"messages": messages, "tools": tools, "options": options})
        if self.delay:
            import time

            time.sleep(self.delay)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def exchanges(home):
    return [json.loads(line) for p in sorted(Path(home).rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]


class CoreTests(unittest.TestCase):
    def test_the_arms_the_override_and_the_setting(self):
        self.assertEqual((core.ARMS, core.DEFAULT_ARM), (("rules", "helper", "agent"), "rules"))
        self.assertEqual(core.parse_override(""), {})
        self.assertEqual(set(core.parse_override("helper").values()), {"helper"})
        self.assertEqual(core.parse_override("lead=agent, sales=helper"), {"lead": "agent", "sales": "helper"})
        for bad in ("smart", "lead=smart", "home=agent"):
            with self.assertRaises(ValueError):
                core.parse_override(bad)
        stored = {"sales": {"arm": "helper"}}
        self.assertEqual(core.arm_of(stored, "sales"), ("helper", "stored"))
        self.assertEqual(core.arm_of(stored, "lead"), ("rules", "default"))
        self.assertEqual(core.arm_of(stored, "sales", {"sales": "agent"}), ("agent", "override"))
        self.assertEqual(core.arm_of(stored, "model", {"model": "agent"}), ("rules", "default"))  # a specialist is never switched
        s = core.BrainSetting(role="lead", arm="agent", by="owner", reason="compare it", since="2026-11-23T00:00:00+00:00")
        self.assertEqual(s.schema_version, "dclab_core.team_brain_setting.v1")
        self.assertEqual(core.BrainSetting.read(s.model_dump(mode="json")), s)
        with self.assertRaises(Exception):
            core.BrainSetting(role="lead", arm="agent", by="viewer", reason="x", since="t")


class _Brains(_Contract):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, CLEAN)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(brains.drain, 10)  # a failed assertion never leaves a request writing into the next test's home
        from dclab_rnd import audit
        from dclab_rnd.ledger import open_ledger
        from dclab_rnd.models.routing import open_routing

        self.services.ledger = open_ledger(self.home, self.services.workspace_id)
        self.routing = open_routing(self.home)
        self.audit = audit.open_audit(self.home)

    def scripted(self, *replies, delay=0.0):
        transport = Scripted(*replies, delay=delay)
        self.services.gateway = gw.Gateway(FileUsage(self.home / "model_requests.jsonl"), transport=lambda tier, purpose: transport,
                                           routing=self.routing)
        return transport

    def answered(self):
        return [e.payload for e in self.services.experience.read(kinds=["team.brain.answered"])]

    def decisions(self):
        return self.services.ledger.records(point="team.brain")

    def switch(self, role, arm, by="owner", reason="compare the arms on the dev set"):
        return brains.switch(self.routing, self.audit, role, arm, by, reason)

    # ------------------------------------------------------------------ the switch
    def test_every_role_starts_on_the_rules_and_the_switch_is_audited_with_a_reason(self):
        self.assertEqual({r["role"]: (r["arm"], r["source"]) for r in brains.overview(self.services)["roles"]},
                         {role: ("rules", "default") for role in core.ROLES})
        with self.assertRaises(PermissionError):
            self.switch("data_science", "helper", by="data_scientist")
        with self.assertRaises(brains.BrainSwitchError):
            self.switch("data_science", "helper", reason="  ")
        with self.assertRaises(brains.BrainSwitchError):
            self.switch("data_science", "clever")
        with self.assertRaises(brains.BrainSwitchError):
            self.switch("model", "agent")  # a specialist is not a switchable role
        self.assertEqual(self.audit.query(10, 0)[0], 0)  # nothing refused was written
        setting = self.switch("data_science", "helper")
        self.assertEqual((setting["arm"], setting["by"], setting["previous"]), ("helper", "owner", None))
        self.assertEqual(brains.arm_for(self.services, "data_science"), ("helper", "stored"))
        self.switch("data_science", "helper")  # the same arm again: no change, no second row
        total, rows = self.audit.query(10, 0)
        self.assertEqual(total, 1)
        self.assertEqual((rows[0]["kind"], rows[0]["move"], rows[0]["detail"]["args"]), ("routing", "set_team_brain",
                                                                                         {"purpose": "team_brain:data_science", "role": "data_science",
                                                                                          "from": "rules", "to": "helper"}))
        accepted = [e.payload for e in self.services.experience.read(kinds=["decision.accepted"])]  # the audit hook's event
        self.assertIn("routing:team_brain:data_science", [p.get("point") for p in accepted])
        self.assertIsNone(self.switch("data_science", "rules"))  # back to the default: the setting is removed
        self.assertEqual(brains.arm_for(self.services, "data_science"), ("rules", "default"))
        self.assertEqual(self.audit.query(10, 0)[0], 2)
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_BRAIN": "data_science=agent"}):  # an override wins, for tests and evaluations
            self.assertEqual(brains.arm_for(self.services, "data_science"), ("agent", "override"))
            self.assertEqual(brains.arm_for(self.services, "sales"), ("rules", "default"))
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_BRAIN": "nonsense"}):  # a broken override leaves the stored setting
            self.assertEqual(brains.arm_for(self.services, "data_science"), ("rules", "default"))

    # ------------------------------------------------------------------ the rules arm: today's agents, recorded
    def test_the_rules_arm_acts_as_before_and_each_turn_records_who_answered(self):
        self.open_renewal()
        self.pump()
        self.assertEqual([s.payload.get("tool") for s in self.steps("teamspace.data_science")], ["team_accept", "team_clarify", None])
        rows = self.answered()
        self.assertTrue(rows)
        self.assertTrue(all((r["arm"], r["source"], r["acted"]) == ("rules", "default", "rules") for r in rows), rows)
        self.assertTrue(all("fallback" not in r and "request" not in r for r in rows))  # no model was asked
        ds = [r for r in rows if r["agent"] == "data_science"]
        self.assertEqual((ds[0]["rule_tool"], ds[0]["task"], ds[0]["mode"]), ("team_accept", "T-renewal", "task"))
        records = self.decisions()
        self.assertEqual(len(records), len(rows))
        for r in records:
            self.assertEqual((r.used, r.value_used, r.level), ("rule", "rules", 0))
            self.assertEqual([(a.by, a.value) for a in r.answers], [("rule", "rules")])
            self.assertTrue(r.run)
        self.assertEqual(exchanges(self.home), [])

    # ------------------------------------------------------------------ a model arm with no model: the rules, and why
    def test_a_switched_role_with_no_model_falls_back_to_the_rules_and_says_why(self):
        self.switch("data_science", "helper")
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.open_renewal()
            self.pump()
        self.assertEqual([s.payload.get("tool") for s in self.steps("teamspace.data_science")], ["team_accept", "team_clarify", None])
        ds = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual({(r["arm"], r["source"], r["acted"], r["fallback"]) for r in ds}, {("helper", "stored", "rules", "no_live_models")})
        record = next(r for r in self.decisions() if r.state.get("team") == "data_science")
        self.assertEqual((record.used, record.value_used), ("rule", "rules"))
        model = next(a for a in record.answers if a.by == "model_role")
        self.assertEqual((model.answerer, model.value, model.reason), ("team_helper", None, "no_live_models"))
        self.assertEqual((record.state["brain_arm"], record.state["brain_fallback"], record.state["model_available"]),
                         ("helper", "no_live_models", False))
        self.assertTrue(all(r["arm"] == "rules" for r in self.answered() if r["agent"] == "business"))  # only the switched role

    # ------------------------------------------------------------------ the helper, beside the rules, thinking kept
    def test_the_helper_answers_beside_the_rules_its_thinking_is_kept_and_the_rules_still_act(self):
        self.switch("data_science", "helper")
        transport = self.scripted(said('{"tool": "team_accept"}'))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        self.assertTrue(brains.drain(10))  # sent after the rules' run, on its own thread
        self.assertEqual([s.payload.get("tool") for s in self.steps("teamspace.data_science")], ["team_accept", "team_clarify", None])
        [seen] = transport.seen  # one request for the one Data Science turn
        system, user = seen["messages"][0]["content"], seen["messages"][1]["content"]
        self.assertNotIn("<thinking>", system + user)
        for words in ("reasoning", "explain your", "your steps", "step by step"):
            self.assertNotIn(words, system.lower())  # the model's thinking comes from the API, never asked for in its words
        self.assertIn("The rules chose: team_accept", user)
        self.assertIn("Customers are leaving", user)
        self.assertEqual((seen["options"].get("_single_attempt"), seen["options"].get("timeout")), (True, 60.0))
        [ds] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((ds["arm"], ds["acted"], ds["model_tool"], ds["agreed"]), ("helper", "rules", "team_accept", True))
        self.assertNotIn("fallback", ds)
        [doc] = exchanges(self.home)
        self.assertEqual((doc["purpose"], doc["whole"], doc["model_request"]), ("team_helper", True, ds["request"]))
        self.assertEqual(doc["thinking"], [{"text": "the task is open, so accepting it comes first"}])
        self.assertEqual(doc["thinking_asked"], "adaptive")
        record = next(r for r in self.decisions() if r.state.get("team") == "data_science")
        model = next(a for a in record.answers if a.by == "model_role")
        self.assertEqual((model.value, model.raw, model.source, record.state["agreed"]), ("helper", "team_accept", ds["request"], True))
        requests = [e for e in self.services.experience.read(kinds=["model.request"])]
        self.assertEqual([e.ids.run for e in requests], [record.run])  # the model's request joins the turn's run
        [run] = [e for e in self.services.experience.read(kinds=["run.recorded"]) if e.ids.run == record.run]
        self.assertTrue(requests[0].ids.trace)
        self.assertEqual(requests[0].ids.trace, run.ids.trace)  # F4: and the turn's trace, though sent after the run ended

    def test_an_answer_the_check_refuses_or_a_timeout_falls_back_to_the_rules(self):
        self.switch("data_science", "helper")
        self.scripted(said('{"tool": "team_approve_everything"}'))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        self.assertTrue(brains.drain(10))
        [ds] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((ds["fallback"], ds.get("model_tool")), ("invalid_output", None))
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")  # the rules' turn stands
        self.assertEqual(len(exchanges(self.home)), 1)  # its exchange and thinking are kept all the same
        out = brains.HelperBrain().beside(self.turn("data_science", "T-renewal"), ("team_block", {}), ("team_block",), None)
        self.assertEqual(out["fallback"], "error")  # the script is spent: the transport raised, and the rules' turn would stand
        self.scripted(TimeoutError("late"))
        out = brains.HelperBrain().beside(self.turn("data_science", "T-renewal"), ("team_block", {}), ("team_block",), None)
        self.assertEqual((out["fallback"], out["tool"]), ("timeout", None))

    # ------------------------------------------------------------------ the agent arm, offered the agent's own tools
    def test_the_agent_arm_chooses_from_the_agents_own_tools_and_its_call_is_never_run(self):
        self.switch("data_science", "agent")
        transport = self.scripted(said(tool_calls=[{"id": "c1", "name": "team_block", "arguments": {"task_id": "T-renewal", "reason": "x"}}],
                                       why="a blocker would be quicker"))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        self.assertTrue(brains.drain(10))
        [seen] = transport.seen
        offered = sorted((t.get("function") or t)["name"] for t in seen["tools"])
        from dclab_rnd.teamspace.agents import config_of

        self.assertEqual(offered, sorted(config_of("data_science").policy_tools))
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")  # the rules acted; the model's block did not run
        [ds] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((ds["arm"], ds["acted"], ds["model_tool"], ds["agreed"]), ("agent", "rules", "team_block", False))
        [doc] = exchanges(self.home)
        self.assertEqual((doc["purpose"], doc["whole"], doc["thinking"]), ("team_agent", True, [{"text": "a blocker would be quicker"}]))
        self.assertIn("team_block", doc["tool_names"])

    # ------------------------------------------------------------------ the review of B1: never in the delivery's way
    def test_a_slow_model_never_fails_repeats_or_delays_a_delivery(self):
        """F1: a model slower than the delivery's time limit. The rules' turn is not held: the row is delivered on its first
        attempt, the rules run once, the model is asked once (after the rules), and its answer is still recorded."""
        self.switch("data_science", "helper")
        # the limit leaves the rules' own turn room on a loaded machine (0.4 s failed on PostgreSQL during a gate); the model
        # is twice as slow as the limit, so a delivery that waited on it would time out
        transport = self.scripted(said('{"tool": "team_accept"}'), delay=4.0)
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_DELIVERY_TIMEOUT_SECONDS": "2"}):
            self.open_renewal()
            self.pump(only={"T-renewal"})
        rows = [r for r in self.work.deliveries(project_id=self.pid) if r["task_id"] == "T-renewal" and r["to_team"] == "data_science"]
        self.assertTrue(rows)
        self.assertTrue(all(r["status"] == "delivered" and not r.get("attempts") for r in rows), rows)
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")
        self.assertTrue(brains.drain(10))
        runs = [e for e in self.services.experience.read(kinds=["run.recorded"]) if e.payload.get("name") == "teamspace.data_science"]
        self.assertEqual(len(runs), 1)  # exactly one rules turn
        self.assertEqual(len(transport.seen), 1)  # and one model request, never a second
        [ds] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((ds["model_tool"], ds["acted"]), ("team_accept", "rules"))

    def test_a_turn_that_raises_is_still_recorded_and_asks_no_model(self):
        """F3: a Stop at a checkpoint or a failing tool ends the rules' run with an error: the turn's brain is recorded all
        the same, as ``stopped`` for a model arm, and no model is asked for a turn that did not complete."""
        from dclab_rnd.agents import runtime
        from dclab_rnd.teamspace import agents

        self.switch("data_science", "helper")
        transport = self.scripted(said('{"tool": "team_accept"}'))
        self.open_renewal()  # Business's turn: on the rules
        with mock.patch.object(runtime, "run", side_effect=RuntimeError("stopped at a checkpoint")):
            with self.assertRaises(RuntimeError):
                agents.run_turn(self.turn("data_science", "T-renewal", row={"kind": "task"}))
            self.switch("data_science", "rules")
            with self.assertRaises(RuntimeError):  # the rules arm too
                agents.run_turn(self.turn("data_science", "T-renewal", row={"kind": "task"}))
        self.assertTrue(brains.drain(10))
        self.assertEqual(transport.seen, [])
        helper, rules = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((helper["arm"], helper["fallback"], helper["acted"]), ("helper", "stopped", "rules"))
        self.assertEqual((rules["arm"], rules.get("fallback")), ("rules", None))
        states = [r.state.get("brain_fallback") for r in self.decisions() if r.state.get("team") == "data_science"]
        self.assertEqual(sorted(states, key=str), sorted(["stopped", None], key=str))

    def test_a_burst_past_the_cap_asks_no_model_and_is_recorded_busy(self):
        """Shadow requests out at once are capped: a turn past ``MAX_LIVE`` starts no thread and asks no model."""
        from dclab_rnd.teamspace import agents

        self.switch("data_science", "helper")
        transport = self.scripted(said('{"tool": "team_accept"}'))
        self.open_renewal()
        held = [object() for _ in range(brains.MAX_LIVE)]
        with brains._LIVE_LOCK:
            brains._LIVE.update(held)
        try:
            agents.run_turn(self.turn("data_science", "T-renewal", row={"kind": "task"}))
        finally:
            with brains._LIVE_LOCK:
                brains._LIVE.difference_update(held)
        self.assertTrue(brains.drain(10))
        self.assertEqual(transport.seen, [])
        [helper] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((helper["arm"], helper["fallback"], helper["acted"]), ("helper", "busy", "rules"))

    def test_a_thread_that_cannot_start_holds_no_slot_and_the_turn_is_recorded(self):
        """Re-review: a failed ``start()`` leaves no slot taken for good (every later turn would be ``busy``)."""
        import threading

        from dclab_rnd.teamspace import agents

        self.switch("data_science", "helper")
        transport = self.scripted(said('{"tool": "team_accept"}'))
        self.open_renewal()
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("can't start new thread")):
            agents.run_turn(self.turn("data_science", "T-renewal", row={"kind": "task"}))
        self.assertEqual(len(brains._LIVE), 0)
        self.assertTrue(brains.drain(1))
        self.assertEqual(transport.seen, [])
        [helper] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((helper["arm"], helper["fallback"]), ("helper", "error"))

    def test_a_store_error_while_preparing_never_fails_the_rules_turn(self):
        """Re-review: the client lookup reads whether the project is private; a store error there is the ``error``
        fallback, and the rules' turn runs as it is."""
        from dclab_rnd.teamspace import agents

        self.switch("data_science", "helper")
        self.scripted(said('{"tool": "team_accept"}'))
        self.open_renewal()
        with mock.patch.object(brains, "_gateway", side_effect=OSError("the store is away")):
            out = agents.run_turn(self.turn("data_science", "T-renewal", row={"kind": "task"}))
        self.assertFalse(out["skipped"])
        self.assertEqual(out["refused"], [])
        [helper] = [r for r in self.answered() if r["agent"] == "data_science"]
        self.assertEqual((helper["arm"], helper["fallback"]), ("helper", "error"))

    def test_a_record_that_cannot_read_the_store_never_fails_the_turn(self):
        """F3: the record reads the task for its problem id; a store that fails there is reported, the event is still
        written (without the problem), and a ledger that fails never fails the turn whose rules already acted."""
        from types import SimpleNamespace

        from dclab_rnd.ledger import capture

        def broken(*_):
            raise OSError("the store is away")
        turn = SimpleNamespace(work=SimpleNamespace(task=broken), project_id=self.pid, task_id="T-x", agent="sales", row=None, extra={})
        brains.record(turn, "rules", "default", ("team_accept", {}), None, "teamspace-0123")
        [row] = [r for r in self.answered() if r["task"] == "T-x"]
        self.assertNotIn("problem", row)
        with mock.patch.object(capture, "team_brain_answered", side_effect=RuntimeError("the ledger is away")):
            out = self.open_renewal()
        self.assertEqual(out["refused"], [])
        self.assertFalse(out["skipped"])

    def test_the_route_reads_every_role_and_only_the_owner_switches_one(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        home = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_AGENT_HOME": str(home)}), TestClient(create_app(home)) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            self.assertEqual({r["arm"] for r in c.get("/api/team-brains").json()["roles"]}, {"rules"})
            self.assertEqual(c.put("/api/team-brains/lead", json={"arm": "agent"}, headers=h).status_code, 422)  # no reason
            self.assertEqual(c.put("/api/team-brains/model", json={"arm": "agent", "reason": "x"}, headers=h).status_code, 422)
            out = c.put("/api/team-brains/lead", json={"arm": "agent", "reason": "the Lead plans first in B3"}, headers=h)
            self.assertEqual(out.status_code, 200, out.text)
            self.assertEqual((out.json()["arm"], out.json()["source"]), ("agent", "stored"))
            self.assertEqual(c.get("/api/team-brains").json()["history"][-1]["to"], "agent")
            rows = c.app.state.services.audit.query(10, 0)[1]
            self.assertEqual([r["move"] for r in rows if r["kind"] == "routing"], ["set_team_brain"])


for _name in [n for n in dir(_Contract) if n.startswith("test_")]:  # the fixture only: test_teamspace_agents runs its own tests
    setattr(_Brains, _name, None)


class FileTeamBrainsTests(_Brains, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgTeamBrainsTests(_Brains, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


class ThinkingKeptTests(unittest.TestCase):
    """Rule 8: a new purpose's thinking is stored, whole: the team purposes are shown the team's records, never cells."""

    def test_team_helper_and_team_agent_keep_their_thinking(self):
        for purpose in ("team_helper", "team_agent"):
            home = Path(tempfile.mkdtemp())
            transport = Scripted(said('{"tool": "none"}', why=f"the {purpose} thought"))
            with mock.patch.dict(os.environ, CLEAN), mock.patch.object(experience, "current_log", lambda: None):
                g = gw.Gateway(FileUsage(home / "model_requests.jsonl"), transport=lambda tier, p: transport)
                g.client(purpose).complete([{"role": "user", "content": "the turn"}], None)
            [doc] = exchanges(home)
            self.assertEqual((doc["purpose"], doc["whole"], doc["thinking_asked"]), (purpose, True, "adaptive"))
            self.assertEqual(doc["thinking"], [{"text": f"the {purpose} thought"}])
            self.assertFalse(doc["may_leave"])  # team notes stay in the workspace


if __name__ == "__main__":
    unittest.main()
