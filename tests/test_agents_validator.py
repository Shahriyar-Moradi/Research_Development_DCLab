"""The validator in front of every write tool (package A2.2).

Every project write tool declares the workflow move it makes; the runtime checks that move with the actor "agent"
before the handler runs, logs a refusal as a transition, and returns the reason and the rules. The Home agent's
writes change a draft, so they have their own checks: workflow validation, the pack list, one simulation at a time.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agents import Registry, Tool, build_registry, default_registry  # noqa: E402
from dclab_rnd.draft import chat as home  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.intern.tools import Toolbox  # noqa: E402
from dclab_rnd.studio import graph  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402

SOLUTION = {"target": "y", "task": "binary", "prediction_moment": "Before the call is placed.", "forbidden": [{"column": "duration", "reason": "after the call"}]}


def ready(store, *done, **extra):
    """A project with data and a solution, and the given stages completed."""
    p = store.create("validator test", goal="test")
    p["data"] = {"filename": "t.csv", "sha256": "abc", "rows": 10, "columns": ["y", "x", "duration"]}
    p["solution"] = dict(SOLUTION)
    for stage in done:
        p["stages"][stage] = {"status": "completed"}
    p.update(extra)
    return store.save(p)


def bare(store):
    return store.create("empty", goal="test")


def signed(store):
    p = ready(store, policy={"require_solution_signoff": True})
    return graph.approve_gate(store, p["id"], "solution", reason="read it")


# One state per project write tool in which the validator refuses the agent's move. A new write tool without a case
# here fails test_every_project_write_tool_has_a_refusal_case.
REFUSALS = {
    "use_sample": (signed, {"key": "telco_churn"}),  # replacing the data would drop a solution the owner signed
    "set_settings": (lambda s: ready(s, running="data"), {"max_rows": 500}),  # a stage is running
    "propose_solution": (bare, {"target": "y"}),  # no data
    "set_solution": (bare, {"target": "y", "task": "binary", "prediction_moment": "Before the call is placed."}),  # no data
    "run_stage": (bare, {"stage": "data"}),  # no solution
    "run_all": (bare, {}),  # no solution
    "approve_stage": (lambda s: ready(s, "data"), {"stage": "models"}),  # models has not run
    "export_notebook": (bare, {}),  # nothing to capture
}


class DeclaredMoveTests(unittest.TestCase):
    def test_every_write_tool_declares_its_move(self):
        for tool in default_registry().tools.values():
            if tool.effect != "write":
                continue
            with self.subTest(tool.name):
                self.assertIsNotNone(tool.move)
                if tool.scope == "project":
                    self.assertIn(tool.move, graph.MOVES)
                else:
                    self.assertIn(tool.move, home.DRAFT_MOVES.values())

    def test_every_project_write_tool_has_a_refusal_case(self):
        writes = {t.name for t in default_registry().tools.values() if t.scope == "project" and t.effect == "write"}
        self.assertEqual(writes - {"create_project"}, set(REFUSALS))

    def test_a_write_tool_without_a_move_cannot_be_registered(self):
        with self.assertRaises(ValueError):
            Registry().register(Tool("w", "writes", {"type": "object", "properties": {}}, lambda: None, effect="write"))

    def test_a_write_tool_in_a_scope_without_a_validator_never_runs(self):
        ran = []
        r = Registry()
        r.register(Tool("w", "writes", {"type": "object", "properties": {}}, lambda: ran.append(1), effect="write", move="run_stage"))
        with self.assertRaises(RuntimeError):
            r.call("w", {})
        self.assertEqual(ran, [])


class ProjectRefusalTests(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(Path(tempfile.mkdtemp()))
        self.box = Toolbox(self.store)

    def test_every_project_write_is_refused_with_its_reason_and_logged(self):
        for name, (make, args) in REFUSALS.items():
            with self.subTest(name):
                project = make(self.store)
                before = json.dumps(self.store.get(project["id"]), sort_keys=True, default=str)
                logged = len(self.store.transitions(project["id"]))
                result = self.box.call(name, {"project_id": project["id"], **args})
                if name == "run_all":  # run_all answers per stage
                    result = result["data"]
                self.assertIn("error", result)
                self.assertTrue(result["failed_checks"], result)
                self.assertIn("rules", result)
                self.assertIn(result["status"], ("blocked", "needs_approval"))
                self.assertEqual(json.dumps(self.store.get(project["id"]), sort_keys=True, default=str), before)  # the handler did not run
                entries = self.store.transitions(project["id"])
                self.assertEqual(len(entries), logged + 1)
                self.assertEqual((entries[-1]["actor"], entries[-1]["move"], entries[-1]["status"]),
                                 ("agent", default_registry().get(name).move, result["status"]))

    def test_check_move_does_not_offer_creating_a_project(self):
        enum = default_registry().get("check_move").parameters["properties"]["move"]["enum"]
        self.assertNotIn("create_project", enum)

    def test_creating_a_project_needs_a_name_and_a_goal(self):
        result = self.box.call("create_project", {"name": "x", "goal": "   "})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.store.list(), [])
        made = self.box.call("create_project", {"name": "x", "goal": "Predict churn"})
        first = self.store.transitions(made["project_id"])
        self.assertEqual([(t["move"], t["status"], t["outcome"]) for t in first], [("create_project", "allowed", "done")])
        self.assertEqual(first[0]["args"], {"name": "x"})  # the goal (the user's words) is not logged

    def test_an_allowed_move_without_its_own_log_is_logged_with_its_outcome(self):
        p = ready(self.store)
        self.assertEqual(self.box.call("set_settings", {"project_id": p["id"], "max_rows": 500})["settings"]["max_rows"], 500)
        last = self.store.transitions(p["id"])[-1]
        self.assertEqual((last["move"], last["status"], last["outcome"]), ("set_settings", "allowed", "done"))

    def test_an_allowed_move_is_logged_once(self):
        p = ready(self.store, "data")
        self.box.call("approve_stage", {"project_id": p["id"], "stage": "data"})
        self.assertEqual([t["move"] for t in self.store.transitions(p["id"])], ["approve_stage"])

    def test_an_unknown_project_is_not_found(self):
        self.assertIn("Not found", self.box.call("run_stage", {"project_id": "nope", "stage": "data"})["error"])

    def test_the_runtime_checks_even_a_handler_that_forgets(self):
        """A new write tool whose handler does no check of its own is still refused by the runtime."""
        ran = []
        registry = build_registry()
        registry.register(Tool("careless_rerun", "Reruns a stage without checking.", {"type": "object", "properties": {"project_id": {"type": "string"}, "stage": {"type": "string"}},
                                                                                        "required": ["project_id", "stage"]},
                               lambda box, project_id, stage: ran.append(stage), effect="write", move="run_stage", takes_context=True))
        p = bare(self.store)
        result = Toolbox(self.store, registry=registry).call("careless_rerun", {"project_id": p["id"], "stage": "data"})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(ran, [])
        self.assertEqual(self.store.transitions(p["id"])[-1]["move"], "run_stage")


class RunAllShapeTests(unittest.TestCase):
    def test_the_standard_plan_reports_a_refused_run_all(self):
        """The intern's standard plan reads run_all per stage; a refusal must keep that shape, not crash its report."""
        from dclab_rnd.intern.loop import Intern
        from dclab_rnd.intern.sessions import SessionStore

        home = Path(tempfile.mkdtemp())
        store = ProjectStore(home / "projects")
        p = ready(store, policy={"require_solution_signoff": True})  # a solution nobody signed yet
        p["data"]["profile"] = {"column_count": 3, "columns": [], "target_candidates": ["y"], "time_candidates": [], "id_candidates": [], "text_candidates": []}
        store.save(p)
        intern = Intern(SessionStore(home / "intern"), Toolbox(store), None)
        s = intern.start("Finish this project", project_id=p["id"])
        done = intern.run(s["id"])
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertIn("Needs a person", done["final"])


class DraftCheckTests(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.requests = []
        self.agent = HomeAgent(self.store, None, on_request=lambda d, what, args: self.requests.append(what))
        self.draft = self.store.create("Predict which customers cancel")
        self.agent.start(self.draft["id"])

    def test_the_pack_must_be_on_the_list_and_the_users_choice_stands(self):
        self.assertIn("error", self.agent.run_tool(self.draft["id"], "set_pack", {"key": "astrology", "why": "x"}))
        self.store.update(self.draft["id"], lambda d: d.update(pack={"key": "tabular", "source": "user", "why": "picked"}))
        refused = self.agent.run_tool(self.draft["id"], "set_pack", {"key": "timeseries", "why": "x"})
        self.assertIn("error", refused)
        self.assertEqual(self.store.get(self.draft["id"])["pack"]["key"], "tabular")

    def test_the_handlers_still_default_what_a_small_model_leaves_out(self):
        self.assertEqual(self.agent.run_tool(self.draft["id"], "request_data", {}), {"offered": True})
        self.assertEqual(self.agent.run_tool(self.draft["id"], "set_pack", {"key": "timeseries"}), {"pack": "timeseries"})
        self.assertIn("asked", self.agent.run_tool(self.draft["id"], "ask_user", {"question": "When is it predicted?"}))

    def test_one_simulation_at_a_time(self):
        self.store.update(self.draft["id"], lambda d: d["assets"].append({"id": "a1", "status": "cleaning"}))
        self.assertIn("error", self.agent.run_tool(self.draft["id"], "simulate_data", {"description": "churn table"}))
        self.assertEqual(self.requests, [])

    def test_an_invalid_workflow_is_refused_and_the_current_one_kept(self):
        before = self.store.get(self.draft["id"])["workflow"]
        result = self.agent.run_tool(self.draft["id"], "propose_workflow", {"nodes": [{"wf": "WF-07", "label": "models first"}]})
        self.assertIn("error", result)
        self.assertTrue(result["problems"])
        self.assertEqual(self.store.get(self.draft["id"])["workflow"]["version"], before["version"])
        rejected = [e for e in self.store.events(self.draft["id"]) if e["kind"] == "workflow" and e["data"].get("rejected")]
        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
