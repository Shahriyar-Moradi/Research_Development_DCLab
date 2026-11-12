"""TP-08 (graph admission): a validated model-proposed hierarchy executes through the existing guarded owners.

Real tools on invented Telco-shaped tables; the proposal is compiled by TP-02's own compiler from a scripted boundary
reply, never a live model or a sealed set. Every refusal must leave zero effects and keep the deterministic fallback.
"""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dclab_rnd import experience
from dclab_rnd.expansion import runner
from dclab_rnd.orchestra.checkpoints import open_checkpointer
from dclab_rnd.studio import ProjectStore, graph
from dclab_rnd.tasks.dispatch import TaskDispatcher
from tests.test_task_decomposer import boundaries
from tests.test_task_dispatch import make_project

STAGES = ("data", "leakage", "features", "models")


def sources_of(store, pid, caps):
    from dclab_core.card import problem_card
    from dclab_core.solution_workflow import solution_workflow
    from dclab_rnd.core_adapter import brief_from_project, plan_from_project
    from dclab_rnd.tasks.adapter import adapt_workflow

    project = store.get(pid)
    source = plan_from_project(project)
    workflow = solution_workflow(problem_card(brief_from_project(project)))
    baseline, mapping = adapt_workflow(workflow, source, project, capabilities=caps)
    return workflow, baseline, mapping


def proposed(store, pid, caps, *, answer=None):
    """What decompose() returns for a scripted, valid boundary reply: TP-02's own compiler, no model call."""
    from dclab_rnd.tasks.decomposer import DecompositionBudget, _compile

    workflow, baseline, mapping = sources_of(store, pid, caps)
    return _compile(answer(baseline) if answer else boundaries(baseline), baseline, workflow, caps,
                    mapping["external_artifacts"], DecompositionBudget(), "proposal_fixture")


def resealed(candidate, change):
    from dclab_core.task_fingerprints import graph_fingerprint
    from dclab_core.task_graph import TaskGraph

    body = candidate.model_dump(mode="json")
    change(body)
    body["fingerprint"] = graph_fingerprint(body)
    return TaskGraph.read(body)


class TaskAdmissionTests(unittest.TestCase):
    database_url = ""

    def make_store(self):
        return ProjectStore(self.home / "projects")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.store = self.make_store()
        experience.install(experience.FileLog(self.home / "experience", "task-test"))
        self.addCleanup(experience.install, None)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.saver = self.stack.enter_context(open_checkpointer(self.home, database_url=self.database_url))
        self.stack.enter_context(runner.fast_profile())
        self.pid = make_project(self.store)

    def dispatcher(self, **kwargs):
        kwargs.setdefault("admit_proposals", True)
        return TaskDispatcher(self.store, self.saver, enabled=True, **kwargs)

    def caps(self, d):
        from dclab_rnd.tasks.adapter import capability_snapshot
        return capability_snapshot(d.registry, allowed_tools=d.allowed)

    def candidate(self, d, **kwargs):
        return proposed(self.store, self.pid, self.caps(d), **kwargs)

    def assert_untouched(self, d):
        self.assertIsNone(d.state(self.pid))
        for stage in (*STAGES, "final"):
            self.assertIsNone(self.store.read_stage(self.pid, stage))
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 0)

    def run_through_models(self, d):
        for task in STAGES:
            result = d.dispatch(self.pid, task)
            self.assertEqual(result["status"], "completed", result)

    # ------------------------------------------------------------------------------------------- the main path
    def test_admitted_hierarchy_runs_every_task_once_through_the_guarded_owners(self):
        d = self.dispatcher()
        candidate = self.candidate(d)
        admitted = d.admit(self.pid, candidate)
        self.assertEqual(admitted["status"], "admitted", admitted)
        self.assertEqual(admitted["graph_fingerprint"], candidate.fingerprint)
        state = d.state(self.pid)
        self.assertEqual(state["graph"], candidate.model_dump(mode="json"))
        self.assertEqual(state["graph"]["provenance"]["method"], "model")
        self.assertEqual(state["admission"]["reference_id"], "proposal_fixture")
        self.assertEqual(state["attempts"], [])
        self.run_through_models(d)
        refused = d.dispatch(self.pid, "final")
        self.assertEqual(refused["status"], "blocked", refused)
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 0)
        graph.approve_gate(self.store, self.pid, "holdout", by="fixture-owner", reason="One authored final read")
        for task in ("final", "report"):
            result = d.dispatch(self.pid, task)
            self.assertEqual(result["status"], "completed", result)
        state = d.state(self.pid)
        self.assertEqual([a["task"] for a in state["attempts"]], [*STAGES, "final", "report"])
        self.assertTrue(all(a["acceptance_verification"] == "verified" for a in state["attempts"]))
        self.assertTrue(all(a["graph_fingerprint"] == candidate.fingerprint for a in state["attempts"]))
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        self.assertEqual(d.dispatch(self.pid, "final")["status"], "already_completed")
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)

    def test_nested_groups_with_redundant_dependencies_are_admitted(self):
        def nested(baseline):
            answer = boundaries(baseline, single=True)
            answer["groups"].append({"id": "group_9", "parent_id": None, "members": []})
            for group in answer["groups"][:-1]:
                group["parent_id"] = "group_9"
            for dep in answer["dependencies"]:
                if dep["task_id"] == "report":
                    dep["requires"] = ["final", "models", "data"]
            return answer

        d = self.dispatcher()
        result = d.admit(self.pid, self.candidate(d, answer=nested))
        self.assertEqual(result["status"], "admitted", result)
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")

    # ------------------------------------------------------------------------------------------- opt-in
    def test_admission_is_opt_in_twice_and_default_off_writes_nothing(self):
        plain = TaskDispatcher(self.store, self.saver, enabled=True)
        candidate = self.candidate(plain)
        self.assertEqual(plain.admit(self.pid, candidate)["reason"], "proposal_admission_disabled")
        off = TaskDispatcher(self.store, self.saver, admit_proposals=True)
        self.assertEqual(off.admit(self.pid, candidate)["status"], "disabled")
        self.assert_untouched(plain)

    def test_turning_admission_off_blocks_effects_instead_of_switching_graphs(self):
        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, self.candidate(d))["status"], "admitted")
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        plain = TaskDispatcher(self.store, self.saver, enabled=True)
        result = plain.dispatch(self.pid, "leakage")
        self.assertEqual(result["reason"], "proposed_graph_admission_disabled", result)
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))
        # Read-only inspection of the admitted graph still works without the opt-in.
        self.assertEqual(plain.verify(self.pid, "data")["status"], "verified")
        self.assertIsNone(plain.preflight(self.pid))

    # ------------------------------------------------------------------------------------------- refusals
    def test_tampered_or_weakened_proposals_have_zero_effects(self):
        d = self.dispatcher()
        candidate = self.candidate(d)

        def retitle(body):
            next(n for n in body["nodes"] if n["id"] == "final")["title"] = "Read the holdout freely"

        def drop_gate(body):
            final = next(n for n in body["nodes"] if n["id"] == "final")
            final["dependencies"] = ["models"]
            body["edges"] = [e for e in body["edges"] if not (e["kind"] == "dependency" and e["target"] == "final")]
            body["edges"].append({"kind": "dependency", "source": "models", "target": "final"})

        def drop_check(body):
            node = next(n for n in body["nodes"] if n["id"] == "leakage")
            node["acceptance_checks"] = node["acceptance_checks"][:-1] if len(node["acceptance_checks"]) > 1 else node["acceptance_checks"]
            node["title"] = node["title"] + " (lighter)"

        def deterministic_claim(body):
            body["provenance"] = {"method": "deterministic", "reference_id": "binary_workflow_adapter", "producer_version": "tp01d.v1"}

        def other_revision(body):
            body["revision"] = body["revision"] + 1

        for change in (retitle, drop_gate, drop_check, deterministic_claim, other_revision):
            with self.subTest(change=change.__name__):
                try:
                    tampered = resealed(candidate, change)
                except Exception:
                    continue  # the frozen contract itself refused it
                result = d.admit(self.pid, tampered)
                self.assertEqual(result["status"], "blocked", result)
                self.assert_untouched(d)
        self.assertEqual(d.admit(self.pid, {"nodes": []})["reason"], "proposal_invalid")
        self.assert_untouched(d)

    def test_proposal_for_a_changed_source_is_stale_and_refused(self):
        d = self.dispatcher()
        candidate = self.candidate(d)
        self.store.update(self.pid, goal="Another confirmed goal is needed")
        result = d.admit(self.pid, candidate)
        self.assertEqual(result["status"], "blocked", result)
        self.assert_untouched(d)

    def test_unknown_capability_in_the_registry_refuses_admission(self):
        d = self.dispatcher()
        candidate = self.candidate(d)
        d.allowed.remove("export_notebook")
        result = d.admit(self.pid, candidate)
        self.assertEqual(result["status"], "blocked", result)
        self.assert_untouched(d)

    def test_malformed_proposal_keeps_the_exact_deterministic_fallback(self):
        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, "not a graph")["status"], "blocked")
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        state = d.state(self.pid)
        self.assertEqual(state["graph"]["provenance"]["method"], "deterministic")
        self.assertNotIn("admission", state)

    def test_admission_after_execution_began_is_refused(self):
        d = self.dispatcher()
        candidate = self.candidate(d)
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        before = d.state(self.pid)
        result = d.admit(self.pid, candidate)
        self.assertEqual(result["reason"], "execution_already_initialized", result)
        self.assertEqual(d.state(self.pid)["graph"], before["graph"])

    # ------------------------------------------------------------------------------------------- after admission
    def test_source_change_after_admission_refuses_further_effects(self):
        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, self.candidate(d))["status"], "admitted")
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        self.store.update(self.pid, goal="Another confirmed goal is needed")
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_saved_admitted_graph_cannot_be_rewritten_in_the_checkpoint(self):
        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, self.candidate(d))["status"], "admitted")
        state = d.state(self.pid)
        node = next(n for n in state["graph"]["nodes"] if n["id"] == "group_1")
        node["title"] = "Coordinate holdout"
        from dclab_core.task_fingerprints import graph_fingerprint
        state["graph"]["fingerprint"] = graph_fingerprint(state["graph"])
        d._save(self.pid, state, "tampered")
        result = d.dispatch(self.pid, "data")
        self.assertEqual(result["status"], "blocked", result)
        self.assertIsNone(self.store.read_stage(self.pid, "data"))

    def test_crash_after_effect_reconciles_without_repeating_or_rereading_the_holdout(self):
        class Crash(BaseException):
            pass

        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, self.candidate(d))["status"], "admitted")
        self.run_through_models(d)
        graph.approve_gate(self.store, self.pid, "holdout", by="fixture-owner", reason="One authored final read")
        save = TaskDispatcher._save

        def interrupted(dispatcher, pid, state, kind, task=None):
            if kind == "completed" and task == "final":
                raise Crash()
            return save(dispatcher, pid, state, kind, task)

        with patch.object(TaskDispatcher, "_save", new=interrupted), self.assertRaises(Crash):
            d.dispatch(self.pid, "final")
        record = self.store.read_stage(self.pid, "final")
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        result = self.dispatcher().dispatch(self.pid, "final")
        self.assertEqual(result["status"], "reconciled", result)
        self.assertEqual(record, self.store.read_stage(self.pid, "final"))
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        self.assertEqual([a["task"] for a in d.state(self.pid)["attempts"]].count("final"), 1)

    def test_source_revision_keeps_the_admitted_structure_and_invalidates_work(self):
        d = self.dispatcher(budget={"max_tool_calls": 8, "max_seconds": 3600, "max_cost_eur": 0.0})
        candidate = self.candidate(d)
        self.assertEqual(d.admit(self.pid, candidate)["status"], "admitted")
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        project = self.store.get(self.pid)
        solution = copy.deepcopy(project["solution"])
        solution["prediction_moment"] += " A newly confirmed constraint."
        self.store.update(self.pid, solution=solution)
        self.store.clear_stages(self.pid)
        graph.approve_gate(self.store, self.pid, "solution", by="owner", reason="Task source revision " + d.source_revision(self.pid))
        result = d.adopt_sources(self.pid)
        self.assertEqual(result["status"], "ready", result)
        self.assertIn("data", result["invalidated"])
        state = d.state(self.pid)
        self.assertEqual(state["graph"]["provenance"]["method"], "model")
        self.assertGreater(state["graph"]["revision"], candidate.revision)
        groups = lambda g: sorted((n["id"], n["parent_id"]) for n in g["nodes"] if n["kind"] == "group")
        self.assertEqual(groups(state["graph"]), groups(candidate.model_dump(mode="json")))
        self.assertEqual(d.dispatch(self.pid, "data").get("acceptance_verification"), "verified")

    def test_solution_team_completes_an_admitted_hierarchy_with_one_human_holdout_gate(self):
        from dclab_rnd.orchestra.solution import SolutionTeam
        d = self.dispatcher()
        candidate = self.candidate(d)
        self.assertEqual(d.admit(self.pid, candidate)["status"], "admitted")
        first = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True, admit_proposals=True).run(self.pid)
        self.assertEqual(first["status"], "paused", first)
        self.assertEqual(first["asks"]["gate"], "holdout")
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 0)
        done = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True, admit_proposals=True).run(self.pid, resume=True)
        self.assertEqual(done["next"], "done", done)
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        state = d.state(self.pid)
        self.assertEqual(state["graph"]["fingerprint"], candidate.fingerprint)
        self.assertEqual(len(state["attempts"]), 6)

    def test_adopt_sources_refuses_reconstructed_graph_with_dangling_dependency(self):
        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, self.candidate(d))["status"], "admitted")
        state = d.state(self.pid)
        feat = next(n for n in state["graph"]["nodes"] if n["id"] == "features")
        feat["dependencies"] = ["nonexistent_stage"]
        from dclab_core.task_fingerprints import graph_fingerprint
        state["graph"]["fingerprint"] = graph_fingerprint(state["graph"])
        state["admission"]["fingerprint"] = state["graph"]["fingerprint"]
        d._save(self.pid, state, "tampered_dep")

        project = self.store.get(self.pid)
        solution = copy.deepcopy(project["solution"])
        solution["prediction_moment"] += " Updated."
        self.store.update(self.pid, solution=solution)
        self.store.clear_stages(self.pid)
        graph.approve_gate(self.store, self.pid, "solution", by="owner", reason="Task source revision " + d.source_revision(self.pid))
        result = d.adopt_sources(self.pid)
        self.assertEqual(result["status"], "blocked")

    def test_unrooted_groups_or_unparented_leaves_are_refused(self):
        d = self.dispatcher()
        candidate = self.candidate(d)
        def orphan_group(body):
            body["nodes"].append({"id": "orphan", "kind": "group", "parent_id": None, "title": "Orphan", "objective": "Orphan"})
        self.assertEqual(d.admit(self.pid, resealed(candidate, orphan_group))["status"], "blocked")

        def unparented_leaf(body):
            data_node = next(n for n in body["nodes"] if n["id"] == "data")
            data_node["parent_id"] = None
            body["edges"] = [e for e in body["edges"] if not (e["target"] == "data" and e["kind"] == "grouping")]
        self.assertEqual(d.admit(self.pid, resealed(candidate, unparented_leaf))["status"], "blocked")

    def test_admit_on_corrupted_checkpoint_returns_blocked(self):
        d = self.dispatcher()
        from langgraph.checkpoint.base import empty_checkpoint
        cp = empty_checkpoint()
        cp["channel_values"] = {"task_state": "corrupted"}
        cp["channel_versions"] = {"task_state": 1}
        self.saver.put(d.config(self.pid), cp, {}, {})
        res = d.admit(self.pid, self.candidate(d))
        self.assertEqual(res["status"], "blocked")
        self.assertEqual(res["reason"], "invalid_task_checkpoint")

    def test_tampered_provenance_in_checkpoint_refuses_dispatch(self):
        d = self.dispatcher()
        self.assertEqual(d.admit(self.pid, self.candidate(d))["status"], "admitted")
        state = d.state(self.pid)
        state["graph"]["provenance"]["reference_id"] = "forged_reference"
        d._save(self.pid, state, "tampered_provenance")
        result = d.dispatch(self.pid, "data")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "source_or_capability_changed")


class PostgresTaskAdmissionTests(TaskAdmissionTests):
    def setUp(self):
        from tests import pgtest
        self.database_url = pgtest.require()
        pgtest.empty(self.database_url)
        super().setUp()

    def make_store(self):
        from dclab_rnd.storage.postgres import open_stores
        return open_stores(self.home, self.database_url)[0]


if __name__ == "__main__":
    unittest.main()
