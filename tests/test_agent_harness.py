"""Package 21.1: one harness for every agent. Every model purpose is an agent's, a role's or a plain model use; every
entry point opens the harness; no tool is called outside the modules it records; and each agent, run on a scripted
model, leaves a recorded step for every move it made (its replies without a tool call too), joined to its run."""

import ast
import asyncio
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
    from dclab_rnd import experience
    from dclab_rnd.agents import harness
    from dclab_rnd.models.settings import PURPOSES
except ImportError as error:  # the agent environment has no pydantic contracts for the record
    MISSING = str(error)
else:
    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"

# Each agent's probe: a test here (a method of ProbeTest) or, for an agent that needs another environment or a long
# fixture, the test that checks its steps there. An agent added to AGENTS without one fails the enumeration.
PROBES = {
    "home": "test_home_with_a_model",
    "home.script": "test_home_without_a_model",
    "intern": "test_intern_with_a_model",
    "intern.standard": "test_intern_standard_plan",
    "mcp": "test_an_mcp_clients_calls",
    "team": "tests/test_team.py::test_the_teams_moves_are_steps_of_its_run",
    "campaign": "tests/test_agentic.py::test_the_campaign_runs_in_the_harness_and_its_requests_are_recorded",
    "coordinator": "tests/test_coordinator.py::test_the_mornings_walk_is_a_run_of_the_coordinator",
    "twin": "tests/test_twin_runs.py::test_the_run_is_a_session_of_the_twin_agent",
}

# The only functions that call a registered tool, each recording the call as a step (the loop, the standard plan, the
# Home agent's code-made call, an MCP client's call, the campaign's experiment) or being the registry itself. A call by
# reference counts (``asyncio.to_thread(toolbox.call, ...)``), whatever the receiver is named.
TOOL_CALLERS = {("dclab_rnd/agents/runtime.py", "_run"), ("dclab_rnd/agents/registry.py", "Registry.acall"),
                ("dclab_rnd/intern/loop.py", "Intern._step"), ("dclab_rnd/intern/tools.py", "Toolbox.call"),
                ("dclab_rnd/draft/chat.py", "HomeAgent.run_tool"), ("dclab_rnd/mcp_server.py", "build_server.call_tool"),
                ("dclab_rnd/agentic/engine.py", "build_graph.execute")}
SKIP = ("dclab_rnd/agent_eval/",)  # measurement harnesses: they drive the agents, they are not entry points


def _sources():
    for path in sorted((ROOT / "dclab_rnd").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith(SKIP):
            yield rel, ast.parse(path.read_text(encoding="utf-8"))


@unittest.skipIf(MISSING, MISSING)
class EnumerationTest(unittest.TestCase):
    def test_every_purpose_is_an_agents_a_roles_or_a_model_use_and_only_one(self):
        named = [p for a in harness.AGENTS.values() for p in a.purposes] + list(harness.ROLES) + list(harness.MODEL_USES)
        self.assertEqual(len(named), len(set(named)), "a purpose classified twice")
        self.assertEqual(set(named), set(PURPOSES), "every purpose of models/settings.py, and no other")

    def test_every_agent_has_a_probe_and_its_entry_opens_the_harness(self):
        self.assertEqual(set(PROBES), set(harness.AGENTS))
        for name, agent in harness.AGENTS.items():
            source = _source_of(agent.entry)  # read, not imported: the campaign's module needs the agent environment
            need = {"loop": "run_policy(", "session": "harness.session("}
            self.assertIn(need[agent.runs], source, f"{name}: {agent.entry} does not run in the harness")
            probe = PROBES[name]
            if "::" in probe:
                file, test = probe.split("::")
                self.assertIn(f"def {test}(", (ROOT / file).read_text(encoding="utf-8"), f"{name}: its probe {probe} is missing")
            else:
                self.assertTrue(hasattr(ProbeTest, probe), f"{name}: its probe {probe} is missing")

    def test_every_model_client_is_for_a_known_purpose(self):
        for rel, tree in _sources():
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("client", "advisor")
                        and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    purpose = node.args[0].value
                    if node.func.attr == "client" and purpose in ("s3", "ecs", "cloudwatch"):  # boto3, not a model
                        continue
                    self.assertIn(purpose, harness.purposes(), f"{rel}:{node.lineno}: {purpose!r} is not classified in the harness")

    def test_no_tool_is_called_outside_the_functions_that_record_it(self):
        def walk(rel, node, where):
            for child in ast.iter_child_nodes(node):
                inner = f"{where}.{child.name}" if where and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else (
                    child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else where)
                if isinstance(child, ast.Attribute) and child.attr in ("call", "acall"):
                    owner = child.value
                    mine = (isinstance(owner, ast.Name) and owner.id == "self") or (
                        isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super")
                    if not mine:  # self.call is a class's own method; super().acall a model client's request
                        self.assertIn((rel, where), TOOL_CALLERS, f"{rel}:{child.lineno} ({where}): a tool called outside the harness")
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "handler":
                    self.assertEqual(rel, "dclab_rnd/agents/registry.py", f"{rel}:{child.lineno}: a tool's handler called directly")
                walk(rel, child, inner)

        for rel, tree in _sources():
            walk(rel, tree, "")


def _source_of(entry):
    module, _, path = entry.partition(":")
    file = ROOT / (module.replace(".", "/") + ".py")
    if not file.is_file():  # a package: its __init__
        file = ROOT / module.replace(".", "/") / "__init__.py"
    text = file.read_text(encoding="utf-8")
    scope = ast.parse(text).body
    node = None
    for part in path.split("."):
        node = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part)
        scope = getattr(node, "body", [])
    return ast.get_source_segment(text, node)


def _log():
    home = Path(tempfile.mkdtemp())
    return home, experience.FileLog(home / "experience", "wtest")


def _steps_cover(test, events, run=None):
    """Every tool call has its step, in the same run; return the steps."""
    steps = [e for e in events if e.kind == "agent.step" and (run is None or e.ids.run == run)]
    calls = [e for e in events if e.kind == "tool.call" and (run is None or e.ids.run == run)]
    test.assertTrue(steps, "no step recorded")
    stepped = {(e.ids.run, e.payload.get("tool")) for e in steps}
    for call in calls:
        test.assertIn((call.ids.run, call.payload["tool"]), stepped)
    return steps


@unittest.skipIf(MISSING, MISSING)
class ProbeTest(unittest.TestCase):
    def setUp(self):
        from tests.test_models_gateway import CLEAN

        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-" + "test-" + "x" * 8, "OPENAI_MODEL": "std-model"})
        env.start()
        self.addCleanup(env.stop)
        self.home, self.log = _log()
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def events(self, **match):
        return self.log.read(**match)

    def test_home_with_a_model(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.models.usage import FileUsage
        from tests.test_models_gateway import gateway as scripted

        call = {"id": "c1", "name": "ask_user", "arguments": {"field": "target", "question": "Which rows are fake?", "why": "the outcome",
                                                               "options": ["Confirmed", "None"]}}
        plan = [{"content": "I will ask.", "tool_calls": [call], "usage": {"input_tokens": 5, "output_tokens": 3},
                 "assistant_message": {"role": "assistant", "content": "I will ask.", "tool_calls": [
                     {"id": "c1", "type": "function", "function": {"name": "ask_user", "arguments": '{"field": "target"}'}}]}}]
        gw, _ = scripted(FileUsage(self.home / "usage.jsonl"), plan)
        store = DraftStore(self.home / "drafts")
        with experience.bind(visit="v00000000000000c1"):
            d = store.create("find fake deliveries")
            HomeAgent(store, gw.client("home_agent", draft_id=d["id"])).reply(d["id"], "it is anomaly detection")
        events = self.events(visit="v00000000000000c1")
        steps = [s for s in _steps_cover(self, events, run=d["id"]) if s.payload["agent"] == "home"]
        self.assertEqual([s.payload.get("tool") for s in steps], ["ask_user"])
        self.assertIsNotNone(steps[0].ids.model_request)
        self.assertTrue(any(e.kind == "run.recorded" and e.payload.get("run_kind") == "agent" and e.ids.run == d["id"] for e in events))

    def test_home_without_a_model(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(self.home / "drafts")
        with experience.bind(visit="v00000000000000c2"):
            d = store.create("predict which customers will cancel")
            home = HomeAgent(store, None)
            home.start(d["id"])
            home.reply(d["id"], "whether they cancel within a month")
        events = self.events(visit="v00000000000000c2")
        steps = [s for s in _steps_cover(self, events, run=d["id"]) if s.payload["agent"] == "home.script"]
        self.assertEqual([s.payload.get("tool") for s in steps][:2], ["ask_user", "take_answer"])  # the opening question, then the answer read
        runs = [e for e in events if e.kind == "run.recorded" and e.ids.run == d["id"]]
        self.assertTrue(runs and runs[-1].payload["name"] == "home.script" and runs[-1].payload["status"] == "completed")

    def test_intern_with_a_model(self):
        from dclab_rnd.intern.loop import Intern
        from dclab_rnd.intern.tools import Toolbox
        from tests.test_intern import ScriptedModel, make

        class Talks(ScriptedModel):  # a plan, then an answer in words: both are steps
            def complete(self, messages, tools):
                if self.turn >= len(self.script):
                    self.turn += 1
                    return {"content": "I looked; nothing to build yet.", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
                            "assistant_message": {"role": "assistant", "content": "I looked; nothing to build yet."}}
                return super().complete(messages, tools)

        projects, sessions = make(str(self.home / "w"))
        intern = Intern(sessions, Toolbox(projects), Talks([("write_plan", {"plan": "look around"})]))
        session = intern.run(intern.start("Look around first", {"max_steps": 5, "max_minutes": 5})["id"])
        steps = _steps_cover(self, self.events(), run=session["id"])
        self.assertEqual([s.payload.get("tool") for s in steps], ["write_plan", None])  # the answer without a tool call is a step
        self.assertEqual(steps[-1].payload.get("reply"), "I looked; nothing to build yet.")

    def test_intern_standard_plan(self):
        from dclab_rnd.intern.loop import Intern
        from dclab_rnd.intern.tools import Toolbox
        from tests.test_intern import make

        projects, sessions = make(str(self.home / "w"))
        intern = Intern(sessions, Toolbox(projects))
        session = intern.run(intern.start("Model the quarterly widget pipeline of my company.", {"max_steps": 5, "max_minutes": 1})["id"])
        events = self.events()
        steps = _steps_cover(self, events, run=session["id"])
        self.assertEqual(len(steps), len(session["steps"]))
        run = [e for e in events if e.kind == "run.recorded" and e.ids.run == session["id"]]
        self.assertEqual((run[-1].payload["name"], run[-1].payload["status"]), ("intern.standard", "completed"))

    def test_an_mcp_clients_calls(self):
        try:
            import mcp.types as types
        except ImportError:
            self.skipTest("mcp is not installed")
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.mcp_server import build_server
        from tests.test_intern import make

        projects, _ = make(str(self.home / "w"))
        server = build_server(Toolbox(projects))
        handler = server.request_handlers[types.CallToolRequest]
        for name, args in (("search_evidence", {"query": "leakage", "k": 1}), ("create_project", {"name": "via mcp", "goal": "probe"})):
            asyncio.run(handler(types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name=name, arguments=args))))
        box = Toolbox(projects)
        with mock.patch.object(box, "call", side_effect=KeyError("broken")):  # a tool that raises is a step too (MCP answers an error)
            asyncio.run(build_server(box).request_handlers[types.CallToolRequest](
                types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name="list_samples", arguments={}))))
        events = self.events()
        steps = _steps_cover(self, events)
        self.assertEqual([(s.payload["agent"], s.payload.get("tool"), s.payload["verdict"]) for s in steps],
                         [("mcp", "search_evidence", "ok"), ("mcp", "create_project", "ok"), ("mcp", "list_samples", "error")])
        self.assertEqual(len({s.ids.run for s in steps}), 3)  # one run per call: callers are never one trajectory
        ended = {e.ids.run: e.payload["status"] for e in events if e.kind == "run.recorded"}
        self.assertEqual([ended[s.ids.run] for s in steps], ["completed", "completed", "failed"])

    def test_a_role_is_a_step_of_what_asked_it(self):
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.models.usage import FileUsage
        from tests.test_models_gateway import gateway as scripted

        gw, _ = scripted(FileUsage(self.home / "usage.jsonl"), [{"content": '{"family": "binary"}', "usage": {"input_tokens": 3, "output_tokens": 2}},
                                                                {"content": "no object here", "usage": {"input_tokens": 3, "output_tokens": 2}}])
        advisor = GatewayAdvisor(gw, project_id="p1")
        with experience.bind(run="team-x"):
            self.assertEqual(advisor.ask("problem_analyst", {"problem": "churn"}), {"family": "binary"})
            with self.assertRaises(ValueError):
                advisor.ask("problem_analyst", {"problem": "churn"})
        steps = [e for e in self.events() if e.kind == "agent.step"]
        self.assertEqual([(s.payload["agent"], s.payload["verdict"], s.ids.run, s.ids.project) for s in steps],
                         [("role.problem_analyst", "ok", "team-x", "p1"), ("role.problem_analyst", "error", "team-x", "p1")])
        self.assertTrue(all(s.ids.model_request for s in steps))
        self.assertTrue(all(s.payload.get("reply") is None for s in steps))  # a role's JSON may name columns: it stays in its request

    def test_a_role_asked_inside_a_run_is_a_step_of_that_run_and_names_only_its_own_request(self):
        from dclab_rnd.core_advisor import GatewayAdvisor

        class Breaks:  # raises before any request is recorded
            spent_eur = 0.0

            def complete(self, *a, **k):
                raise ValueError("transport broke")

        class Gate:
            def client(self, purpose, **kw):
                return Breaks()

        experience.new_model_request()  # an earlier request, of something else
        with harness.session("team") as current:
            with self.assertRaises(ValueError):
                GatewayAdvisor(Gate(), project_id="p1").ask("critic", {"x": 1})
        step = next(e for e in self.events() if e.kind == "agent.step")
        self.assertEqual((step.payload["agent"], step.ids.run, step.ids.model_request), ("role.critic", current.run_id, None))

    def test_a_session_ends_as_its_caller_says(self):
        with harness.session("team") as current:
            current.status = "stopped"  # a declined gate: it returned, but did not complete
        with self.assertRaises(TimeoutError):
            with harness.session("campaign") as timed_out:
                timed_out.status = "stopped"
                raise TimeoutError()
        ended = {e.ids.run: e.payload["status"] for e in self.events() if e.kind == "run.recorded"}
        self.assertEqual((ended[current.run_id], ended[timed_out.run_id]), ("stopped", "stopped"))

    def test_tool_calls_written_as_text_keep_the_step_not_the_words(self):
        from dclab_rnd.agents import Policy, Registry, Tool, run

        registry = Registry()
        registry.register(Tool("record", "Record a field.", {"type": "object", "properties": {"value": {"type": "string"}}},
                               lambda **a: {"ok": True}, scope="draft"))
        text = '{"name": "record", "arguments": {"value": "a person\'s address"}}'

        class Writes:
            def complete(self, messages, tools):
                return {"content": text, "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": text}}

        run(Policy("home", "s", ("record",)), Writes(), registry, [{"role": "user", "content": "hi"}])
        steps = [e for e in self.events() if e.kind == "agent.step"]
        self.assertTrue(steps and all(e.payload["verdict"] == "no_call" and e.payload.get("reply") is None for e in steps))

    def test_a_run_without_a_tracer_of_its_own_is_still_recorded(self):
        from dclab_rnd.agents import Policy, Registry, run

        class Says:
            def complete(self, messages, tools):
                return {"content": "Nothing to do.", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
                        "assistant_message": {"role": "assistant", "content": "Nothing to do."}}

        run(Policy("home", "s", ()), Says(), Registry(), [{"role": "user", "content": "hi"}])
        events = self.events()
        step = next(e for e in events if e.kind == "agent.step")
        self.assertEqual((step.payload.get("tool"), step.payload.get("reply"), step.payload["verdict"]), (None, "Nothing to do.", "answered"))
        self.assertTrue(any(e.kind == "run.recorded" and e.ids.run == step.ids.run for e in events))

    def test_a_session_that_fails_or_is_stopped_says_so(self):
        for exc, status in ((ValueError("broken"), "failed"), (KeyboardInterrupt(), "stopped")):
            with self.assertRaises(type(exc)):
                with harness.session("team") as current:
                    current.step("planner", {}, {"ok": True})
                    raise exc
            done = [e for e in self.events() if e.kind == "run.recorded" and e.ids.run == current.run_id]
            self.assertEqual(done[-1].payload["status"], status)
        with self.assertRaises(KeyError):
            with harness.session("not-an-agent"):
                pass


if __name__ == "__main__":
    unittest.main()
