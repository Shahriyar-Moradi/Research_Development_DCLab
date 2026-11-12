"""Offline unit & integration tests for Arm C (model-proposed + guarded TaskDispatcher + SolutionTeam)."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dclab_rnd import experience, ledger
from dclab_rnd.agent_eval.task_planning.gate import Case
from dclab_rnd.agent_eval.task_planning.probe import offline_environment
from dclab_rnd.agent_eval.task_planning.runner import Audit, _owner_complete, make_project
from dclab_rnd.agents.registry import Registry
from dclab_rnd.expansion import runner
from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.orchestra.checkpoints import open_checkpointer
from dclab_rnd.orchestra.solution import SolutionTeam
from dclab_rnd.studio import ProjectStore
from dclab_rnd.tasks.dispatch import TaskDispatcher
from dclab_rnd.tasks.host import planning_host


class _ScriptedProposalServer:
    thinking = None

    def __init__(self, *, corrupt_after=None):
        self.calls = []
        self.corrupt_after = corrupt_after

    def complete(self, messages, tools=None, max_tokens=None, **options):
        body = json.loads(messages[-1]["content"])
        baseline = body.get("baseline") or body["deterministic_graph"]
        idx = len(self.calls)
        self.calls.append("decompose" if "deterministic_graph" in body else f"plan:{baseline.get('task_id')}")
        if self.corrupt_after is not None and idx >= self.corrupt_after:
            answer = {"version": "tp03.v1", "unexpected": "arbitrary_tool_injection"}
        elif "deterministic_graph" in body:
            leaves = [node for node in baseline["nodes"] if node["kind"] != "group"]
            answer = {
                "version": "tp02.v1",
                "baseline_fingerprint": baseline["fingerprint"],
                "groups": [{"id": "group_0", "parent_id": None, "members": [node["id"] for node in leaves]}],
                "dependencies": [{"task_id": node["id"], "requires": node["dependencies"]} for node in leaves],
            }
        else:
            answer = {
                "version": "tp03.v1",
                "baseline_fingerprint": baseline["fingerprint"],
                **{key: baseline[key] for key in ("actions", "budget", "policy")},
            }
        return {
            "content": json.dumps(answer),
            "finish_reason": "stop",
            "tool_calls": [],
            "thinking": [{"text": "SCRIPTED: preserve immutable leaf tasks and gates."}],
            "usage": {"input_tokens": 12, "output_tokens": 36},
        }


class ArmCRuntimeTests(unittest.TestCase):
    def test_arm_c_solution_team_decomposes_plans_and_completes_verified_workflow(self):
        server = _ScriptedProposalServer()
        case = Case("arm_c_smoke", "fresh_binary", "execution", "normal", "complete", "prediction",
                    "Predict subscriber churn before retention outreach at the monthly cutoff.")
        with tempfile.TemporaryDirectory(prefix="arm-c-test-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "arm-c-test"))
            ledger.install(None)
            stack.callback(experience.install, prev_log)
            stack.callback(ledger.install, prev_ledger)
            stack.enter_context(runner.fast_profile())
            stack.enter_context(patch("dclab_rnd.experience.model_text_allowed", return_value=True))
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = make_project(store, case, 72001)
            audit = Audit(store, pid, "normal")
            audit.install(stack)
            gateway = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)

            with planning_host(name="arm-c-smoke"):
                team = SolutionTeam(store, saver, home=home, task_execution=True, admit_proposals=True, gateway=gateway)
                res = team.run(pid)
                self.assertEqual(res.get("status"), "paused")
                self.assertEqual(res.get("asks", {}).get("gate"), "holdout")
                res = team.run(pid, resume=True)

            self.assertEqual(res.get("next"), "done")
            self.assertTrue(_owner_complete(store, pid))
            self.assertEqual(store.get(pid).get("holdout_uses"), 1)
            self.assertEqual(audit.unauthorized, 0)
            self.assertEqual(audit.leakage_bypasses, 0)
            state = TaskDispatcher(store, saver).state(pid)
            self.assertEqual(state["graph"]["provenance"]["method"], "model")
            self.assertIn("admission", state)
            self.assertEqual(len(state["attempts"]), 6)
            self.assertTrue(all(a.get("planning", {}).get("status") == "proposed" for a in state["attempts"]))
            self.assertEqual(server.calls[0], "decompose")

    def test_arm_c_recovers_from_two_transient_validation_faults_via_replan(self):
        server = _ScriptedProposalServer()
        case = Case("arm_c_dual_fault", "recovery", "execution", "transient", "complete", "prediction",
                    "Predict monthly account closure while recovering cleanly from validation timeouts.")
        with tempfile.TemporaryDirectory(prefix="arm-c-fault-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "arm-c-test"))
            ledger.install(None)
            stack.callback(experience.install, prev_log)
            stack.callback(ledger.install, prev_ledger)
            stack.enter_context(runner.fast_profile())
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = make_project(store, case, 72003)

            original_check = Registry.check
            fault_stages = {"data": 0, "models": 0}

            def two_stage_fault(registry, tool, arguments, level="schema"):
                stage = (arguments or {}).get("stage")
                if tool.handler is not None and tool.name == "run_stage" and stage in fault_stages and fault_stages[stage] == 0:
                    fault_stages[stage] += 1
                    raise TimeoutError(f"scripted_validation_fault_{stage}")
                return original_check(registry, tool, arguments, level)

            stack.enter_context(patch.object(Registry, "check", new=two_stage_fault))
            gateway = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)

            with planning_host(name="arm-c-fault"):
                team = SolutionTeam(store, saver, home=home, task_execution=True, admit_proposals=True, gateway=gateway)
                res = team.run(pid)
                self.assertEqual(res.get("status"), "paused")
                res = team.run(pid, resume=True)

            self.assertEqual(res.get("next"), "done")
            self.assertTrue(_owner_complete(store, pid))
            state = TaskDispatcher(store, saver).state(pid)
            self.assertEqual(len(state["replans"]), 2)
            self.assertEqual([r["task"] for r in state["replans"]], ["data", "models"])

    def test_arm_c_falls_back_safely_on_invalid_model_proposal(self):
        server = _ScriptedProposalServer(corrupt_after=1)
        case = Case("arm_c_fallback", "fresh_binary", "execution", "normal", "complete", "prediction",
                    "Predict monthly subscriber churn with deterministic fallback on bad model plans.")
        with tempfile.TemporaryDirectory(prefix="arm-c-fb-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "arm-c-test"))
            ledger.install(None)
            stack.callback(experience.install, prev_log)
            stack.callback(ledger.install, prev_ledger)
            stack.enter_context(runner.fast_profile())
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = make_project(store, case, 72007)
            gateway = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)

            with planning_host(name="arm-c-fb"):
                team = SolutionTeam(store, saver, home=home, task_execution=True, admit_proposals=True, gateway=gateway)
                res = team.run(pid)
                if res.get("status") == "paused":
                    res = team.run(pid, resume=True)

            self.assertEqual(res.get("next"), "done")
            self.assertTrue(_owner_complete(store, pid))
            state = TaskDispatcher(store, saver).state(pid)
            self.assertTrue(all(a.get("planning", {}).get("status") == "fallback" for a in state["attempts"]))


if __name__ == "__main__":
    unittest.main()
