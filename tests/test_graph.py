"""The workflow graph: node states, the validator's verdicts, gates, the transition log, and its wiring into tools and API."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio import graph  # noqa: E402
from dclab_rnd.studio.store import STAGE_KEYS, ProjectStore  # noqa: E402

CONTRACT = {"target": "y", "task": "binary", "prediction_moment": "Before the call is placed.", "forbidden": [{"column": "duration", "reason": "after the call"}]}


def ready(store, *done, uses=0, **extra):
    """A project with data and a contract, and the given stages completed."""
    p = store.create("graph test", goal="test")
    p["data"] = {"filename": "t.csv", "sha256": "abc", "rows": 10, "columns": ["y", "x", "duration"]}
    p["contract"] = dict(CONTRACT)
    for stage in done:
        p["stages"][stage] = {"status": "completed"}
    p["holdout_uses"] = uses
    p.update(extra)
    return store.save(p)


class NodeStateTests(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(Path(tempfile.mkdtemp()))

    def test_empty_project_starts_at_the_contract(self):
        p = self.store.create("x", goal="y")
        self.assertEqual(graph.current_node(p), "WF-01")
        self.assertEqual(graph.state_string(p), "c---------")

    def test_states_follow_the_stages(self):
        p = ready(self.store, "data", "leakage")
        states = graph.node_states(p)
        self.assertEqual([states[n] for n in ("WF-01", "WF-02", "WF-03", "WF-04", "WF-05")], ["done"] * 5)
        self.assertEqual(states["WF-06"], "current")
        p = ready(self.store, *STAGE_KEYS)
        self.assertEqual(graph.current_node(p), "WF-10")  # export closes the loop
        p["captured"] = graph.now()
        self.assertIsNone(graph.current_node(p))

    def test_unsigned_contract_waits_when_the_policy_asks_for_a_signature(self):
        p = ready(self.store, policy={"require_contract_signoff": True})
        self.assertEqual(graph.node_states(p)["WF-01"], "waiting")


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(Path(tempfile.mkdtemp()))

    def test_skipping_a_step_is_not_an_edge(self):
        v = graph.check(ready(self.store), "run_stage", "agent", stage="leakage")
        self.assertEqual(v.status, "blocked")
        self.assertIn("Edge exists", [c["name"] for c in v.checks if not c["ok"]])
        self.assertIn("DCLAB-R04", v.rules)

    def test_next_step_is_allowed_and_names_its_side_effects(self):
        v = graph.check(ready(self.store, "data", "leakage", "features", "models"), "run_stage", "agent", stage="final")
        self.assertTrue(v.allowed)
        self.assertIn("Consumes the holdout once.", v.side_effects)
        self.assertEqual(v.node_to, "WF-09")

    def test_holdout_reuse_needs_a_person_with_a_reason_and_never_the_agent(self):
        p = ready(self.store, "data", "leakage", "features", "models", uses=1)
        self.assertEqual(graph.check(p, "run_stage", "human", stage="final").status, "needs_approval")
        self.assertTrue(graph.check(p, "run_stage", "human", stage="final", reuse_reason="new model chosen by the owner").allowed)
        agent = graph.check(p, "run_stage", "agent", stage="final", reuse_reason="anything")
        self.assertEqual(agent.status, "blocked")
        self.assertIn("PIT-006", agent.evidence)

    def test_contract_gate(self):
        p = ready(self.store, policy={"require_contract_signoff": True})
        self.assertEqual(graph.check(p, "run_stage", "agent", stage="data").status, "needs_approval")
        p = graph.approve_gate(self.store, p["id"], "contract", reason="read it")
        self.assertTrue(graph.check(p, "run_stage", "agent", stage="data").allowed)
        self.assertEqual(graph.check(p, "set_contract", "agent").status, "needs_approval")  # only a person changes a signed contract
        p["contract"] = {**CONTRACT, "prediction_moment": "Changed later."}
        self.assertFalse(graph.contract_signed(p))  # a new version needs a new signature

    def test_holdout_gate_for_the_agent(self):
        p = ready(self.store, "data", "leakage", "features", "models", policy={"require_holdout_approval": True})
        self.assertEqual(graph.check(p, "run_stage", "agent", stage="final").status, "needs_approval")
        self.assertTrue(graph.check(p, "run_stage", "human", stage="final").allowed)  # the owner running it is the approval
        p = graph.approve_gate(self.store, p["id"], "holdout")
        self.assertTrue(graph.check(p, "run_stage", "agent", stage="final").allowed)
        graph.consume_holdout_approval(p)
        self.assertEqual(graph.check(p, "run_stage", "agent", stage="final").status, "needs_approval")

    def test_only_a_person_approves_a_gate(self):
        v = graph.check(ready(self.store), "approve_gate", "agent", gate="holdout")
        self.assertEqual(v.status, "blocked")
        self.assertEqual(graph.check(ready(self.store), "approve_gate", "human", gate="nope").status, "blocked")

    def test_changing_a_choice_after_the_holdout(self):
        p = ready(self.store, *STAGE_KEYS, uses=1)
        self.assertEqual(graph.check(p, "approve_stage", "agent", stage="models", choice="lightgbm").status, "blocked")
        human = graph.check(p, "approve_stage", "human", stage="models", choice="lightgbm")
        self.assertTrue(human.allowed)
        self.assertTrue(any("cannot be judged honestly" in s for s in human.side_effects))

    def test_unknown_moves_and_stages(self):
        p = ready(self.store)
        self.assertEqual(graph.check(p, "delete_data").status, "blocked")
        self.assertEqual(graph.check(p, "run_stage", stage="deploy").status, "blocked")

    def test_every_verdict_can_be_logged_as_a_trajectory_step(self):
        p = ready(self.store, "data")
        v = graph.check(p, "run_stage", "agent", stage="models")
        entry = graph.log(self.store, p["id"], v, p)
        self.assertEqual(entry["state"], "dddd c-----".replace(" ", ""))
        self.assertEqual(self.store.transitions(p["id"])[-1]["status"], "blocked")
        self.assertEqual(entry["failed_checks"], ["Edge exists"])

    def test_describe_lists_steps_moves_and_gates(self):
        view = graph.describe(ready(self.store, "data"))
        self.assertEqual(len(view["nodes"]), 10)
        self.assertEqual(view["current"], "WF-05")
        moves = {m["stage"]: m["status"] for m in view["moves"] if m["move"] == "run_stage"}
        self.assertEqual(moves["leakage"], "allowed")
        self.assertEqual(moves["models"], "blocked")
        self.assertEqual({g["gate"] for g in view["gates"]}, {"contract", "holdout"})


class WiringTests(unittest.TestCase):
    def test_intern_tools_go_through_the_validator(self):
        from dclab_rnd.intern.tools import Toolbox

        store = ProjectStore(Path(tempfile.mkdtemp()))
        box = Toolbox(store)
        self.assertTrue({"get_graph", "check_move"} <= set(box.names()))
        pid = box.call("create_project", {"name": "x", "goal": "y"})["project_id"]
        result = box.call("run_stage", {"project_id": pid, "stage": "data"})
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Contract", [c["name"] for c in result["failed_checks"]])
        self.assertEqual(box.call("get_graph", {"project_id": pid})["current"], "WF-01")
        self.assertEqual(box.call("check_move", {"project_id": pid, "move": "approve_gate", "gate": "contract"})["status"], "blocked")
        self.assertEqual(store.transitions(pid)[-1]["actor"], "agent")

    def test_api_routes(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"dependencies not installed: {error}")
        with TestClient(create_app(Path(tempfile.mkdtemp()))) as client:
            token = client.get("/api/config").json()["csrf"]
            headers = {"X-DCLab-Token": token}
            pid = client.post("/api/projects", json={"name": "g", "goal": "g"}, headers=headers).json()["id"]
            view = client.get(f"/api/projects/{pid}/graph").json()
            self.assertEqual(view["current"], "WF-01")
            verdict = client.post(f"/api/projects/{pid}/graph/check", json={"move": "run_stage", "stage": "data", "actor": "agent"}, headers=headers).json()
            self.assertEqual(verdict["status"], "blocked")
            self.assertEqual(client.post(f"/api/projects/{pid}/approvals", json={"gate": "contract"}, headers=headers).status_code, 409)
            self.assertIn("graph", client.get(f"/api/projects/{pid}").json())
            policy = client.patch(f"/api/projects/{pid}", json={"policy": {"require_contract_signoff": True, "unknown": True}}, headers=headers).json()["graph"]["policy"]
            self.assertEqual(policy, {"require_contract_signoff": True, "require_holdout_approval": False})


if __name__ == "__main__":
    unittest.main()
