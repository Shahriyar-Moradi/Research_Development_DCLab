"""Traces (package A2.3): a row per agent step, in files or PostgreSQL; a finished run replays to the same moves; the
intern writes its trace and the Intern page receives it; no cell value is stored."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402
from dclab_rnd.agents import FileTraces, Policy, Registry, Tool, Tracer, replay, run  # noqa: E402
from dclab_rnd.agents.traces import ARGUMENT_CHARS, FIELDS  # noqa: E402
from dclab_rnd.intern.loop import Intern  # noqa: E402
from dclab_rnd.intern.sessions import SessionStore  # noqa: E402
from dclab_rnd.intern.tools import Toolbox  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required)}


def reply(content=None, calls=(), tokens=(10, 3)):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": content, "tool_calls": calls, "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            "assistant_message": {"role": "assistant", "content": content,
                                  "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)

    def complete(self, messages, tools=None, **_):
        return self.replies.pop(0) if self.replies else reply("done")


def registry():
    r = Registry()
    r.register(Tool("add", "Add.", obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]), lambda a, b: {"sum": a + b}))
    r.register(Tool("gate", "Refused by a validator.", obj(), lambda: {"error": "Blocked: the owner must sign", "status": "needs_approval", "rules": ["DCLAB-R01"]}))
    r.register(Tool("ask", "Ask the person.", obj({"q": {"type": "string"}}, ["q"]), lambda q: {"asked": q}))
    r.register(Tool("finish", "End.", obj({"report": {"type": "string"}}, ["report"]), lambda report: {"ok": True}))
    return r


POLICY = Policy("test", "You add.", ("add", "gate", "ask", "finish"), max_steps=10, terminal=("finish",), stop_after=("ask",))


def traced_run(store, run_id="run1", model=None):
    tracer = Tracer(store, run_id, "test", state=lambda: "c---------")
    model = model or Scripted(
        reply(calls=[("add", {"a": "two", "b": 3}), ("add", {"a": 2, "b": 3})], tokens=(100, 20)),  # a bad call, then a good one
        reply(calls=[("gate", {})], tokens=(120, 5)),
        reply(calls=[("finish", {"report": "r"}), ("add", {"a": 1, "b": 1})], tokens=(130, 9)))  # nothing runs after finish
    out = run(POLICY, model, registry(), [{"role": "system", "content": "s"}], trace=tracer)
    return out, store.steps(run_id)


class TraceRowTests(unittest.TestCase):
    def setUp(self):
        self.store = FileTraces(Path(tempfile.mkdtemp()) / "agent_steps.jsonl")

    def test_a_row_per_step_with_verdict_reply_and_tokens(self):
        out, rows = traced_run(self.store)
        self.assertEqual(out.stopped, "terminal")
        self.assertEqual([(r["n"], r["reply"], r["tool"], r["verdict"]) for r in rows],
                         [(1, 0, "add", "error"), (2, 0, "add", "ok"), (3, 1, "gate", "needs_approval"), (4, 2, "finish", "ok")])
        self.assertEqual([(r["input_tokens"], r["output_tokens"]) for r in rows], [(100, 20), (0, 0), (120, 5), (130, 9)])  # once per reply
        self.assertTrue(all(set(r) == set(FIELDS) and r["state"] == "c---------" and r["agent"] == "test" for r in rows))
        self.assertEqual(sum(r["input_tokens"] for r in rows), out.usage["input_tokens"])

    def test_long_arguments_and_results_are_cut(self):
        r = Registry()
        r.register(Tool("echo", "Echo.", obj({"text": {"type": "string"}}, ["text"]), lambda text: {"echo": text}))
        store = self.store
        run(Policy("p", "s", ("echo",)), Scripted(reply(calls=[("echo", {"text": "x" * 20000})])), r, [], trace=Tracer(store, "long", "p"))
        row = store.steps("long")[0]
        self.assertLessEqual(len(json.dumps(row["arguments"])), ARGUMENT_CHARS + 40)
        self.assertLessEqual(len(row["result"]), 361)

    def test_a_broken_store_never_stops_the_run(self):
        class Broken:
            def record(self, row):
                raise OSError("disk full")

            def steps(self, run_id):
                raise OSError("disk full")

        out = run(POLICY, Scripted(reply(calls=[("add", {"a": 1, "b": 2})])), registry(), [], trace=Tracer.resume(Broken(), "x", "test"))
        self.assertEqual((out.stopped, len(out.steps)), ("answered", 1))

    def test_resume_continues_the_numbers_and_delete_forgets(self):
        traced_run(self.store)
        tracer = Tracer.resume(self.store, "run1", "test")
        run(POLICY, Scripted(reply(calls=[("add", {"a": 1, "b": 1})])), registry(), [], trace=tracer)
        rows = self.store.steps("run1")
        self.assertEqual([(r["n"], r["reply"]) for r in rows][-1], (5, 3))
        traced_run(self.store, "other")
        self.store.delete("run1")
        self.assertEqual(self.store.steps("run1"), [])
        self.assertEqual(len(self.store.steps("other")), 4)


class ReviewFindingTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = FileTraces(self.home / "agent_steps.jsonl")

    def test_no_cell_value_reaches_a_row(self):
        box = Toolbox(ProjectStore(self.home / "projects"))
        pid = box.call("create_project", {"name": "t", "goal": "Predict churn"})["project_id"]
        tracer = Tracer(self.store, "cells", "test")
        for tool, args in (("use_sample", {"project_id": pid, "key": "telco_churn"}), ("describe_data", {"project_id": pid})):
            tracer(tool, args, box.call(tool, args), 0)
        values = [str(v) for c in box.call("describe_data", {"project_id": pid})["columns"] for v in c["examples"] if str(v).strip()]
        self.assertTrue(values)  # the tool does return cell values to the model...
        stored = json.dumps(self.store.steps("cells"))
        leaked = [v for v in values if len(v) > 3 and v in stored]
        self.assertEqual(leaked, [])  # ...but the trace keeps none of them

    def test_the_state_is_what_the_agent_saw_before_its_move(self):
        state = {"n": 0}
        r = Registry()
        r.register(Tool("bump", "Change the state.", obj(), lambda: state.update(n=state["n"] + 1) or {"n": state["n"]}))
        tracer = Tracer(self.store, "before", "test", state=lambda: f"n={state['n']}")
        run(Policy("p", "s", ("bump",)), Scripted(reply(calls=[("bump", {}), ("bump", {})])), r, [], trace=tracer)
        self.assertEqual([row["state"] for row in self.store.steps("before")], ["n=0", "n=1"])

    def test_the_intern_trace_numbers_follow_the_session_steps(self):
        sessions = SessionStore(self.home / "intern")
        intern = Intern(sessions, Toolbox(ProjectStore(self.home / "projects")), None, traces=self.store)
        s = intern.start("Something the samples do not cover at all")
        s["steps"] = [{"n": 1, "tool": "list_samples", "arguments": {}, "summary": "", "ok": True}]  # a step from before traces
        s["used"]["steps"] = 1
        sessions.save(s)
        done = intern.message(s["id"], "What is leakage?")
        self.assertEqual([r["n"] for r in self.store.steps(s["id"])], [done["steps"][-1]["n"]])
        self.assertEqual(done["steps"][-1]["n"], 2)

    def test_the_standard_plan_records_the_state_before_creating_a_project(self):
        intern = Intern(SessionStore(self.home / "intern"), Toolbox(ProjectStore(self.home / "projects")), None, traces=self.store)
        s = intern.start("Predict which telco customers churn")
        intern.run(s["id"])
        rows = {r["tool"]: r for r in self.store.steps(s["id"])}
        self.assertEqual(rows["create_project"]["state"], "no project")
        self.assertEqual(rows["use_sample"]["state"], "c---------")

    def test_a_call_by_its_old_name_replays(self):
        r = registry()
        r.register(Tool("plus", "Add, new name.", obj({"a": {"type": "integer"}}, ["a"]), lambda a: {"a": a}, aliases=("old_plus",)))
        policy = Policy("alias", "s", ("plus",))
        run(policy, Scripted(reply(calls=[("old_plus", {"a": 1})])), r, [], trace=Tracer(self.store, "alias", "test"))
        self.assertTrue(replay(self.store.steps("alias"), policy, r)["same"])

    def test_cut_arguments_are_refused_by_replay_not_misreported(self):
        r = Registry()
        r.register(Tool("echo", "Echo.", obj({"text": {"type": "string"}}, ["text"]), lambda text: {"n": len(text)}))
        policy = Policy("p", "s", ("echo",))
        run(policy, Scripted(reply(calls=[("echo", {"text": "x" * 20000})])), r, [], trace=Tracer(self.store, "cut", "p"))
        with self.assertRaises(ValueError):
            replay(self.store.steps("cut"), policy, r)

    def test_a_torn_line_loses_nothing_and_delete_still_works(self):
        traced_run(self.store, "a")
        with self.store.path.open("a", encoding="utf-8") as handle:
            handle.write('{"run_id": "b", "n"')  # a crash in the middle of an append
        traced_run(self.store, "c")
        self.assertEqual(len(self.store.steps("c")), 4)
        self.store.delete("a")
        self.assertEqual((self.store.steps("a"), len(self.store.steps("c"))), ([], 4))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.store = FileTraces(Path(tempfile.mkdtemp()) / "agent_steps.jsonl")

    def test_a_finished_run_replays_to_the_same_moves(self):
        _, rows = traced_run(self.store)
        again = replay(rows, POLICY, registry())
        self.assertTrue(again["same"], again)
        self.assertEqual(again["replayed"], [("add", "error"), ("add", "ok"), ("gate", "needs_approval"), ("finish", "ok")])

    def test_a_continued_run_replays_across_its_stops(self):
        tracer = Tracer(self.store, "cont", "test")
        run(POLICY, Scripted(reply(calls=[("ask", {"q": "Which target?"})])), registry(), [], trace=tracer)
        tracer.next_call()
        run(POLICY, Scripted(reply(calls=[("add", {"a": 1, "b": 1})]), reply("Done.")), registry(), [], trace=tracer)
        rows = self.store.steps("cont")
        self.assertEqual([r["reply"] for r in rows], [0, 1])
        self.assertTrue(replay(rows, POLICY, registry())["same"])

    def test_replay_notices_when_the_runtime_would_decide_differently(self):
        _, rows = traced_run(self.store)
        tighter = Policy("test", "You add.", ("add", "gate", "ask", "finish"), max_steps=2, terminal=("finish",))
        again = replay(rows, tighter, registry())
        self.assertFalse(again["same"])
        self.assertEqual(again["replayed"], [("add", "error"), ("add", "ok")])

    def test_a_trace_without_reply_numbers_cannot_be_replayed(self):
        with self.assertRaises(ValueError):
            replay([{"tool": "add", "verdict": "ok", "reply": None, "arguments": {}, "result": "", "n": 1}], POLICY, registry())


class PgTraceTests(unittest.TestCase):
    def test_rows_in_postgresql_per_workspace(self):
        url = pgtest.require()
        pgtest.empty(url)
        from dclab_rnd.agents import PgTraces
        from dclab_rnd.storage import db

        mine, theirs = PgTraces(db.workspace("/a", "a", url), url), PgTraces(db.workspace("/b", "b", url), url)
        _, rows = traced_run(mine)
        self.assertEqual([r["verdict"] for r in rows], ["error", "ok", "needs_approval", "ok"])
        self.assertEqual(theirs.steps("run1"), [])
        self.assertTrue(replay(rows, POLICY, registry())["same"])
        mine.delete("run1")
        self.assertEqual(mine.steps("run1"), [])


class InternTraceTests(unittest.TestCase):
    def test_the_intern_traces_every_step_with_its_reply_and_state(self):
        home = Path(tempfile.mkdtemp())
        traces = FileTraces(home / "agent_steps.jsonl")
        projects = ProjectStore(home / "projects")
        pid = projects.create("t", goal="Predict churn")["id"]
        model = Scripted(
            reply(calls=[("write_plan", {"plan": "1. look"}), ("get_graph", {"project_id": pid})], tokens=(50, 7)),
            reply(calls=[("run_stage", {"project_id": pid, "stage": "data"})], tokens=(60, 4)),  # no solution yet: refused
            reply(calls=[("finish", {"report": "Done."})]))
        intern = Intern(SessionStore(home / "intern"), Toolbox(projects), model, traces=traces)
        s = intern.start("Look at the project", project_id=pid)
        done = intern.run(s["id"])
        rows = traces.steps(s["id"])
        self.assertEqual([(r["tool"], r["reply"], r["verdict"]) for r in rows],
                         [("write_plan", 0, "ok"), ("get_graph", 0, "ok"), ("run_stage", 1, "blocked"), ("finish", 2, "ok")])
        self.assertEqual(len(rows), len(done["steps"]))  # finish is a step of the session too, so a replay checks it
        self.assertEqual(done["final"], "Done.")
        self.assertEqual([r["input_tokens"] for r in rows][:3], [50, 0, 60])
        self.assertEqual(rows[1]["state"], "c---------")  # the project's graph state as the intern saw it
        again = replay(rows, intern.policy(done), intern.toolbox.registry)  # the intern runs on run(), so it replays
        self.assertTrue(again["same"], again)

    def test_the_standard_plan_traces_without_reply_numbers(self):
        home = Path(tempfile.mkdtemp())
        traces = FileTraces(home / "agent_steps.jsonl")
        intern = Intern(SessionStore(home / "intern"), Toolbox(ProjectStore(home / "projects")), None, traces=traces)
        s = intern.start("Something the samples do not cover at all")
        intern.run(s["id"])
        rows = traces.steps(s["id"])
        self.assertEqual([(r["tool"], r["reply"]) for r in rows], [("list_samples", None)])

    def test_the_api_returns_the_trace_and_forgets_it_with_the_session(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"dependencies not installed: {error}")
        with TestClient(create_app(Path(tempfile.mkdtemp()))) as client:
            headers = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
            s = client.post("/api/intern/sessions?wait=true", json={"task": "Something the samples do not cover at all"}, headers=headers).json()
            got = client.get(f"/api/intern/sessions/{s['id']}").json()
            self.assertEqual([r["tool"] for r in got["trace"]], ["list_samples"])
            self.assertEqual(client.delete(f"/api/intern/sessions/{s['id']}", headers=headers).status_code, 204)


if __name__ == "__main__":
    unittest.main()
