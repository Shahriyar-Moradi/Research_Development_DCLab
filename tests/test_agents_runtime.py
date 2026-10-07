"""The agent runtime and the tool registry (package A2.1): one registration, every reader lists it; bad calls come
back as errors a model can read; the loop stops on an answer, a terminal tool, or its step or time budget."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agents import EFFECTS, SCOPES, Policy, Registry, Tool, build_registry, default_registry, run  # noqa: E402
from dclab_rnd.draft.chat import DRAFT_MOVES, HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.intern.loop import SESSION_MOVES, Intern  # noqa: E402
from dclab_rnd.intern.sessions import SessionStore  # noqa: E402
from dclab_rnd.intern.tools import LEGACY_TOOLS, Toolbox  # noqa: E402
from dclab_rnd.studio import ProjectStore  # noqa: E402
from dclab_rnd.studio import graph as studio_graph  # noqa: E402


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required)}


def reply(content=None, calls=()):
    """One scripted model reply in the gateway client's shape."""
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": content, "tool_calls": calls, "usage": {"input_tokens": 10, "output_tokens": 3},
            "assistant_message": {"role": "assistant", "content": content,
                                  "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Scripted:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, **_):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": [t["function"]["name"] for t in tools or []]})
        return self.replies.pop(0) if self.replies else reply("done")


def add_numbers(a, b):
    return {"sum": a + b}


class Boom(Exception):
    pass


def explode():
    raise Boom("not expected")


def refuse():
    raise ValueError("not like that")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.r = Registry()
        self.r.register(Tool("add", "Add two integers.", obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]), add_numbers, aliases=("plus",)))
        self.r.register(Tool("explode", "Fails.", obj(), explode))
        self.r.register(Tool("refuse", "Fails as expected.", obj(), refuse, errors=lambda e: {"error": str(e)} if isinstance(e, ValueError) else None))

    def test_a_good_call_runs_and_an_alias_still_works(self):
        self.assertEqual(self.r.call("add", {"a": 2, "b": 3}), {"sum": 5})
        self.assertEqual(self.r.call("plus", {"a": 2, "b": 3}), {"sum": 5})

    def test_bad_calls_come_back_as_errors_not_exceptions(self):
        cases = {
            "unknown tool": self.r.call("nope", {}),
            "unknown argument": self.r.call("add", {"a": 1, "b": 2, "c": 3}),
            "missing argument": self.r.call("add", {"a": 1}),
            "wrong type": self.r.call("add", {"a": "1", "b": 2}),
            "arguments not JSON": self.r.call("add", {"_raw": "{a: 1"}),
            "expected failure": self.r.call("refuse", {}),
        }
        for case, result in cases.items():
            with self.subTest(case):
                self.assertIn("error", result)
        self.assertIn("Available: add, explode, refuse", cases["unknown tool"]["error"])
        self.assertIn("c", cases["unknown argument"]["error"])
        self.assertIn("b", cases["missing argument"]["error"])
        self.assertIn("a:", cases["wrong type"]["error"])
        self.assertEqual(cases["expected failure"], {"error": "not like that"})

    def test_a_null_counts_as_not_given(self):
        self.r.register(Tool("opt", "Optional b.", obj({"a": {"type": "integer"}, "b": {"type": "string"}}, ["a"]), lambda a, b=None: {"a": a, "b": b}))
        self.assertEqual(self.r.call("opt", {"a": 1, "b": None}), {"a": 1, "b": None})
        self.assertIn("Missing", self.r.call("opt", {"a": None})["error"])
        self.assertIn("Unknown", self.r.call("opt", {"a": 1, "c": None})["error"])  # the handler could not take it

    def test_lighter_check_levels(self):
        self.assertIn("error", self.r.call("add", {"a": 1}, check="names"))
        with self.assertRaises(TypeError):  # the type is the handler's to handle at this level
            self.r.call("add", {"a": "1", "b": 2}, check="names")
        with self.assertRaises(TypeError):  # "none" leaves even a missing argument to the handler
            self.r.call("add", {"a": 1}, check="none")

    def test_an_unexpected_failure_is_a_bug_and_raises(self):
        with self.assertRaises(Boom):
            self.r.call("explode", {})

    def test_registration_rules(self):
        with self.assertRaises(ValueError):
            self.r.register(Tool("add", "again", obj(), add_numbers))
        with self.assertRaises(ValueError):
            self.r.register(Tool("other", "takes an alias", obj(), add_numbers, aliases=("plus",)))
        with self.assertRaises(ValueError):
            self.r.register(Tool("x", "bad scope", obj(), add_numbers, scope="everywhere"))
        with self.assertRaises(ValueError):
            self.r.register(Tool("y", "bad effect", obj(), add_numbers, effect="delete"))
        with self.assertRaises(ValueError):
            self.r.register(Tool("z", "not an object", {"type": "string"}, add_numbers))

    def test_scopes_keep_tools_apart(self):
        self.r.register(Tool("draft_only", "A draft tool.", obj(), lambda: {"ok": True}, scope="draft"))
        self.assertNotIn("draft_only", self.r.names("project"))
        self.assertEqual(self.r.names("draft"), ["draft_only"])
        self.assertIn("error", self.r.call("draft_only", {}, scope="project"))
        self.assertEqual(self.r.call("draft_only", {}, scope="draft"), {"ok": True})


class DclabRegistryTests(unittest.TestCase):
    """The real tools: registered once, every one well formed."""

    def test_every_tool_is_registered_once_and_well_formed(self):
        import jsonschema

        registry = default_registry()
        self.assertEqual(len(registry.names("project")), 21)
        self.assertEqual(set(registry.names("draft")), {"ask_user", "record", "set_pack", "propose_workflow", "request_data", "get_profile", "get_analysis", "simulate_data"})
        self.assertEqual(set(registry.names("session")), {"write_plan", "finish"})
        self.assertEqual(registry.names("campaign"), ["run_experiment"])
        for tool in registry.tools.values():
            with self.subTest(tool.name):
                jsonschema.Draft202012Validator.check_schema(tool.parameters)
                self.assertIn(tool.effect, EFFECTS)
                self.assertIn(tool.scope, SCOPES)
                self.assertTrue(tool.description)
                if tool.move is not None:
                    moves = {"project": studio_graph.MOVES, "draft": DRAFT_MOVES.values(), "session": SESSION_MOVES.values(),
                             "campaign": ("experiment",)}[tool.scope]
                    self.assertIn(tool.move, moves)
                    self.assertEqual(tool.effect, "write")
        for old, new in LEGACY_TOOLS.items():
            self.assertEqual(registry.get(old).name, new)

    def test_the_toolbox_reads_the_shared_registry(self):
        box = Toolbox(ProjectStore(Path(tempfile.mkdtemp())))
        self.assertIs(box.registry, default_registry())
        self.assertEqual(box.names(), default_registry().names("project"))
        self.assertIn("error", box.call("set_settings", {"project_id": "p", "max_rows": "many"}))  # checked against the schema first


class UnchangedBehaviourTests(unittest.TestCase):
    """Calls that worked before the registry still work through the toolbox (its handlers clamp and coerce)."""

    def test_the_toolbox_keeps_its_handlers_tolerance(self):
        box = Toolbox(ProjectStore(Path(tempfile.mkdtemp())))
        made = box.call("create_project", {"name": "x", "goal": "Predict churn", "industry": "retail"})
        pid = made["project_id"]
        self.assertEqual(box.projects.get(pid)["industry"], "general")
        self.assertEqual(box.call("set_settings", {"project_id": pid, "max_rows": 100})["settings"]["max_rows"], 200)
        self.assertEqual(box.call("set_settings", {"project_id": pid, "max_rows": "5000"})["settings"]["max_rows"], 5000)
        self.assertIn("error", box.call("get_results", {"project_id": pid, "stage": None}))  # "No stage has run yet", as before
        self.assertIn("results", box.call("search_evidence", {"query": "duration leakage", "k": 50}))

    def test_a_home_handler_ignores_a_stray_turn_argument(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Predict which customers cancel")
        self.assertIn("error", HomeAgent(store, None).run_tool(draft["id"], "get_profile", {"turn": 1}))


class OneRegistrationTests(unittest.TestCase):
    """Adding a tool is one registration: the intern, MCP clients and (for draft tools) the Home agent list it."""

    def setUp(self):
        self.registry = build_registry()
        self.registry.register(Tool("count_projects", "How many projects exist.", obj(), lambda box: {"projects": len(box.projects.list())}, takes_context=True))
        self.registry.register(Tool("note_for_owner", "Leave a note on the draft.", obj({"text": {"type": "string"}}, ["text"]),
                                    lambda turn, text: {"noted": turn.draft_id}, scope="draft", takes_context=True))
        self.home = Path(tempfile.mkdtemp())
        self.toolbox = Toolbox(ProjectStore(self.home / "projects"), registry=self.registry)

    def test_the_intern_lists_and_calls_it(self):
        self.assertIn("count_projects", self.toolbox.names())
        self.assertEqual(self.toolbox.call("count_projects", {}), {"projects": 0})
        model = Scripted(reply(calls=[("finish", {"report": "done"})]))
        intern = Intern(SessionStore(self.home / "intern"), self.toolbox, model)
        intern.run(intern.start("Count the projects")["id"])
        self.assertIn("count_projects", model.seen[0]["tools"])
        self.assertNotIn("note_for_owner", model.seen[0]["tools"])

    def test_mcp_lists_it(self):
        try:
            import mcp.types as types
            from dclab_rnd.mcp_server import build_server
        except ImportError as error:
            self.skipTest(f"mcp not installed: {error}")
        server = build_server(self.toolbox)
        result = asyncio.run(server.request_handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list")))
        names = {t.name for t in result.root.tools}
        self.assertIn("count_projects", names)
        self.assertNotIn("note_for_owner", names)

    def test_the_home_agent_lists_its_draft_tools(self):
        store = DraftStore(self.home / "drafts")
        draft = store.create("Predict which customers cancel")
        model = Scripted(reply("Hello."))
        agent = HomeAgent(store, model, registry=self.registry)
        agent.complete(draft["id"], [])
        self.assertIn("note_for_owner", model.seen[0]["tools"])
        self.assertNotIn("count_projects", model.seen[0]["tools"])
        self.assertEqual(agent.run_tool(draft["id"], "note_for_owner", {"text": "x"}), {"noted": draft["id"]})
        self.assertIn("error", agent.run_tool(draft["id"], "count_projects", {}))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.r = Registry()
        self.r.register(Tool("add", "Add two integers.", obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]), add_numbers))
        self.r.register(Tool("hidden", "Not in the policy.", obj(), lambda: {"secret": 1}))
        self.r.register(Tool("ask", "Ask the person.", obj({"q": {"type": "string"}}, ["q"]), lambda q: {"asked": q}))
        self.r.register(Tool("finish", "End the run.", obj({"report": {"type": "string"}}, ["report"]), lambda report: {"report": report}))
        self.policy = Policy("test", "You add numbers.", ("add", "ask", "finish"), max_steps=6, terminal=("finish",), stop_after=("ask",))

    def test_tool_calls_written_as_text_are_not_an_answer(self):
        """8.1: a small model in Ollama wrote its calls as JSON in its answer; nothing ran, yet the run ended "answered"."""
        as_text = '```json\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n```'
        model = Scripted(reply(as_text), reply(calls=[("add", {"a": 2, "b": 3})]), reply("The sum is 5."))
        out = run(self.policy, model, self.r, [{"role": "system", "content": "s"}])
        self.assertEqual((out.stopped, out.final, [s.tool for s in out.steps]), ("answered", "The sum is 5.", ["add"]))  # told once, it called
        self.assertIn("wrote tool calls as text", model.seen[1]["messages"][-1]["content"])
        model = Scripted(reply(as_text), reply(as_text))
        out = run(self.policy, model, self.r, [{"role": "system", "content": "s"}])
        self.assertEqual((out.stopped, out.steps), ("tools_as_text", []))  # twice: the run ends, nothing done in its name
        out = run(self.policy, Scripted(reply('Use {"name": "Ada", "arguments": "none"} as the example.')), self.r, [])
        self.assertEqual(out.stopped, "answered")  # JSON about something that is not one of its tools is just text
        tags = '<tools>\n {"type": "function", "function": {add a b}}\n</tools>'  # another shape the same model used
        self.assertEqual(run(self.policy, Scripted(reply(tags), reply(tags)), self.r, []).stopped, "tools_as_text")
        out = run(self.policy, Scripted(reply(as_text), reply("I cannot run tools directly.")), self.r, [])
        self.assertEqual(out.stopped, "tools_as_text")  # told once, it answered in prose without a call: still nothing ran
        report = 'I called add. To repeat it: ```json {"name": "add", "arguments": {"a": 2, "b": 3}}``` The sum is 5.'
        out = run(self.policy, Scripted(reply(calls=[("add", {"a": 2, "b": 3})]), reply(report)), self.r, [])
        self.assertEqual((out.stopped, out.final), ("answered", report))  # after real work, a report that quotes a call is an answer

    def test_a_bad_call_is_returned_to_the_model_which_corrects_it(self):
        model = Scripted(reply(calls=[("add", {"a": "two", "b": 3})]), reply(calls=[("add", {"a": 2, "b": 3})]), reply("The sum is 5."))
        steps = []
        out = run(self.policy, model, self.r, [{"role": "system", "content": "s"}], on_step=steps.append)
        self.assertEqual((out.stopped, out.final), ("answered", "The sum is 5."))
        self.assertEqual([s.ok for s in out.steps], [False, True])
        self.assertEqual(steps, out.steps)
        error = json.loads(model.seen[1]["messages"][-1]["content"])
        self.assertIn("Invalid argument for add: a", error["error"])
        self.assertEqual(json.loads(model.seen[2]["messages"][-1]["content"]), {"sum": 5})
        self.assertEqual(out.usage, {"input_tokens": 30, "output_tokens": 9})
        self.assertEqual(model.seen[0]["tools"], ["add", "ask", "finish"])

    def test_a_tool_outside_the_policy_is_refused(self):
        model = Scripted(reply(calls=[("hidden", {})]), reply("ok"))
        out = run(self.policy, model, self.r, [])
        self.assertFalse(out.steps[0].ok)
        self.assertIn("not available here", out.steps[0].result["error"])

    def test_terminal_and_stop_after_tools_end_the_run(self):
        out = run(self.policy, Scripted(reply(calls=[("add", {"a": 1, "b": 1}), ("finish", {"report": "r"})])), self.r, [])
        self.assertEqual((out.stopped, len(out.steps)), ("terminal", 2))
        out = run(self.policy, Scripted(reply(calls=[("ask", {"q": "Which target?"})])), self.r, [])
        self.assertEqual(out.stopped, "stop_after")

    def test_the_step_budget_stops_a_model_that_never_answers(self):
        forever = Scripted(*[reply(calls=[("add", {"a": 1, "b": 1})]) for _ in range(20)])
        out = run(self.policy, forever, self.r, [])
        self.assertEqual((out.stopped, len(out.steps)), ("steps", 6))
        out = run(self.policy, Scripted(reply(calls=[("add", {"a": 1, "b": 1})])), self.r, [], steps_used=6)
        self.assertEqual((out.stopped, out.steps), ("steps", []))

    def test_the_budget_holds_inside_one_reply(self):
        ran = []
        self.r.register(Tool("tick", "Count.", obj(), lambda: ran.append(1) or {"n": len(ran)}))
        model = Scripted(reply(calls=[("tick", {})] * 5))
        out = run(Policy("tight", "s", ("tick",), max_steps=2), model, self.r, [])
        self.assertEqual((out.stopped, len(out.steps), len(ran)), ("steps", 2, 2))
        answers = [m for m in out.messages if m["role"] == "tool"]
        self.assertEqual(len(answers), 5)  # every call is answered, the last three with "not run"
        self.assertIn("Not run", answers[-1]["content"])

    def test_a_used_up_budget_still_lets_the_run_end_in_the_same_reply(self):
        model = Scripted(reply(calls=[("add", {"a": 1, "b": 1})] * 4 + [("finish", {"report": "my report"})]))
        out = run(Policy("tight", "s", ("add", "finish"), max_steps=3, terminal=("finish",)), model, self.r, [])
        self.assertEqual((out.stopped, [s.tool for s in out.steps]), ("terminal", ["add", "add", "add", "finish"]))
        self.assertEqual(out.steps[-1].result, {"report": "my report"})

    def test_a_refused_finish_or_question_goes_back_to_the_model(self):
        model = Scripted(reply(calls=[("finish", {})]), reply(calls=[("ask", {"q": 3})]), reply(calls=[("finish", {"report": "r"})]))
        out = run(self.policy, model, self.r, [])
        self.assertEqual([(s.tool, s.ok) for s in out.steps], [("finish", False), ("ask", False), ("finish", True)])
        self.assertEqual(out.stopped, "terminal")

    def test_nothing_runs_after_a_terminal_tool(self):
        ran = []
        self.r.register(Tool("tock", "Count.", obj(), lambda: ran.append(1) or {}, aliases=("done_alias",)))
        policy = Policy("t", "s", ("add", "finish", "tock"), terminal=("finish",))
        out = run(policy, Scripted(reply(calls=[("finish", {"report": "r"}), ("tock", {})])), self.r, [])
        self.assertEqual((out.stopped, ran), ("terminal", []))
        policy = Policy("alias", "s", ("tock",), terminal=("done_alias",))  # a terminal tool may be named by its alias
        self.assertEqual(run(policy, Scripted(reply(calls=[("tock", {})])), self.r, []).stopped, "terminal")

    def test_the_time_budget_stops_the_run(self):
        import time

        policy = Policy("slow", "s", ("add",), max_seconds=5)
        model = Scripted(reply(calls=[("add", {"a": 1, "b": 1})]))
        out = run(policy, model, self.r, [], started=time.monotonic() - 10)
        self.assertEqual((out.stopped, model.seen), ("time", []))

    def test_a_policy_naming_an_unregistered_tool_is_a_bug(self):
        with self.assertRaises(ValueError):
            run(Policy("broken", "s", ("add", "missing")), Scripted(), self.r, [])

    def test_a_streaming_client_streams(self):
        class Streaming(Scripted):
            def stream(self, messages, tools=None, on_text=None, **_):
                on_text("Hel")
                on_text("lo")
                return self.complete(messages, tools)

        parts = []
        out = run(self.policy, Streaming(reply("Hello")), self.r, [], stream=parts.append)
        self.assertEqual((out.final, parts), ("Hello", ["Hel", "lo"]))


if __name__ == "__main__":
    unittest.main()
