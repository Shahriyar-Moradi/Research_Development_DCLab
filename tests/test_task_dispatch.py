"""TP-04: real guarded tools on invented Telco-shaped tables, never a live model or sealed set."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
import contextvars
import unittest
from unittest.mock import patch

from dclab_rnd import experience
from dclab_rnd.expansion import runner
from dclab_rnd.orchestra.checkpoints import open_checkpointer
from dclab_rnd.studio import ProjectStore, data, graph
from dclab_rnd.tasks.dispatch import TaskDispatcher


def make_project(store):
    import numpy as np
    import pandas as pd

    fixture = json.loads((Path(__file__).parent / "fixtures/tasks/telco_metadata.json").read_text())
    p = store.create("Invented Telco task execution", "telecom", fixture["project"]["goal"])
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({"customerID": [f"customer_{i}" for i in range(240)],
                          "tenure": rng.integers(1, 72, 240), "MonthlyCharges": rng.uniform(10, 100, 240),
                          "Churn": np.where(rng.random(240) < 0.35, "Yes", "No")})
    frame.to_csv(store.data_dir(p["id"]) / "invented.csv", index=False)
    data.attach_data(store, p["id"], "invented.csv")
    p = store.get(p["id"])
    p["solution"] = copy.deepcopy(fixture["project"]["solution"])
    p["settings"].update(quick=True, max_rows=500)
    store.save(p)
    graph.approve_gate(store, p["id"], "solution", by="fixture-owner", reason="Authored integration fixture")
    return p["id"]


class TaskDispatchTests(unittest.TestCase):
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
        return TaskDispatcher(self.store, self.saver, enabled=True, **kwargs)

    def test_real_data_stage_leaves_evidence_and_durable_attempt_without_repeat(self):
        d = self.dispatcher()
        result = d.dispatch(self.pid, "data")
        self.assertEqual(result["status"], "completed", result)
        record = self.store.read_stage(self.pid, "data")
        self.assertEqual(record["run"]["parent"], result["attempt_id"])
        again = self.dispatcher().dispatch(self.pid, "data")
        self.assertEqual(again["status"], "already_completed")
        self.assertEqual(record, self.store.read_stage(self.pid, "data"))
        self.assertEqual(len(d.state(self.pid)["attempts"]), 1)

    def test_unready_task_is_blocked_before_any_effect(self):
        result = self.dispatcher().dispatch(self.pid, "models")
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "models"))
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 0)

    def test_all_stages_and_export_run_once_with_real_owner_permission(self):
        d = self.dispatcher()
        for task in ("data", "leakage", "features", "models"):
            result = d.dispatch(self.pid, task)
            self.assertEqual(result["status"], "completed", result)
        before = d.dispatch(self.pid, "final")
        self.assertEqual(before["status"], "blocked")
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 0)
        graph.approve_gate(self.store, self.pid, "holdout", by="fixture-owner", reason="One authored final read")
        for task in ("final", "report"):
            result = d.dispatch(self.pid, task)
            self.assertEqual(result["status"], "completed", result)
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        state = d.state(self.pid)
        self.assertEqual([a["task"] for a in state["attempts"]], ["data", "leakage", "features", "models", "final", "report"])
        self.assertTrue(all(a["acceptance_verification"] == "verified" for a in state["attempts"]))
        self.assertEqual(d.dispatch(self.pid, "final")["status"], "already_completed")

    def test_default_off_creates_no_attempt_or_effect(self):
        d = TaskDispatcher(self.store, self.saver)
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "disabled")
        self.assertIsNone(d.state(self.pid))
        self.assertIsNone(self.store.read_stage(self.pid, "data"))

    def test_completed_effect_reconciles_after_a_crash_before_checkpoint(self):
        class Crash(BaseException): pass
        d = self.dispatcher()
        save = d._save

        def interrupted(pid, state, kind, task=None):
            if kind == "completed": raise Crash()
            return save(pid, state, kind, task)

        with patch.object(d, "_save", side_effect=interrupted), self.assertRaises(Crash):
            d.dispatch(self.pid, "data")
        before = self.store.read_stage(self.pid, "data")
        self.assertEqual(d.state(self.pid)["attempts"][0]["status"], "started")
        result = self.dispatcher().dispatch(self.pid, "data")
        self.assertEqual(result["status"], "reconciled", result)
        self.assertEqual(before, self.store.read_stage(self.pid, "data"))
        self.assertEqual(len(d.state(self.pid)["attempts"]), 1)

    def test_crash_before_effect_is_unknown_and_never_retried(self):
        class Crash(BaseException): pass
        d = self.dispatcher()
        with patch.object(d, "_invoke", side_effect=Crash()), self.assertRaises(Crash):
            d.dispatch(self.pid, "data")
        for _ in range(2):
            result = self.dispatcher().dispatch(self.pid, "data")
            self.assertEqual(result["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "data"))
        self.assertEqual(len(d.state(self.pid)["attempts"]), 1)

    def test_completed_record_without_owner_metadata_stops_without_repeating(self):
        class Crash(BaseException): pass
        d = self.dispatcher()
        write = self.store.write_stage

        def partial(pid, stage, record):
            write(pid, stage, record)
            raise Crash()

        with patch.object(self.store, "write_stage", side_effect=partial), self.assertRaises(Crash):
            d.dispatch(self.pid, "data")
        record = self.store.read_stage(self.pid, "data")
        result = self.dispatcher().dispatch(self.pid, "data")
        self.assertEqual(result["reason"], "owner_metadata_incomplete", result)
        self.assertEqual(record, self.store.read_stage(self.pid, "data"))

    def test_changed_source_schema_permissions_or_actual_bytes_refuse_dispatch(self):
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        self.store.update(self.pid, goal="Another confirmed goal is needed")
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_changed_actual_table_bytes_are_not_used(self):
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        path = data.data_path(self.store, self.store.get(self.pid))
        with path.open("a") as out:
            out.write("\n")
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_registry_guard_denial_is_recorded_without_changing_domain_state(self):
        d = self.dispatcher()
        before = self.store.get(self.pid)
        check = graph.check
        with patch("dclab_rnd.studio.graph.check", side_effect=lambda p, move, actor, **args:
                   check({**p, "running": "models"}, move, actor, **args)):
            result = d.dispatch(self.pid, "data")
        self.assertEqual(result["reason"], "guard_refused", result)
        after = self.store.get(self.pid)
        for key in ("stages", "solution", "data", "holdout_uses", "approvals"):
            self.assertEqual(before[key], after[key])
        self.assertTrue(any(entry["status"] != "allowed" for entry in self.store.transitions(self.pid)))

    def test_two_workers_cannot_repeat_the_same_effect(self):
        results = []
        contexts = [contextvars.copy_context(), contextvars.copy_context()]
        threads = [threading.Thread(target=lambda c=c: results.append(c.run(self.dispatcher().dispatch, self.pid, "data"))) for c in contexts]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(sorted(r["status"] for r in results), ["already_completed", "completed"])
        self.assertEqual(len(self.dispatcher().state(self.pid)["attempts"]), 1)

    def test_job_stop_before_call_and_after_completion_does_not_repeat(self):
        from dclab_rnd.jobs import Stopped
        d = self.dispatcher()
        with patch("dclab_rnd.jobs.checkpoint", side_effect=Stopped("stopped")), self.assertRaises(Stopped):
            d.dispatch(self.pid, "data")
        self.assertIsNone(d.state(self.pid))
        with patch("dclab_rnd.jobs.checkpoint", side_effect=[None, Stopped("stopped")]), self.assertRaises(Stopped):
            d.dispatch(self.pid, "data")
        record = self.store.read_stage(self.pid, "data")
        self.assertEqual(self.dispatcher().dispatch(self.pid, "data")["status"], "already_completed")
        self.assertEqual(record, self.store.read_stage(self.pid, "data"))

    def test_durable_tool_budget_cannot_reset_on_resume(self):
        d = self.dispatcher(budget={"max_tool_calls": 1, "max_seconds": 3600, "max_cost_eur": 0.0})
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        result = self.dispatcher().dispatch(self.pid, "leakage")
        self.assertEqual(result["reason"], "execution_budget_exhausted", result)
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_solution_team_runs_pauses_and_resumes_in_existing_checkpointer(self):
        from dclab_rnd.orchestra.solution import SolutionTeam
        first = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True).run(self.pid)
        self.assertEqual(first["status"], "paused", first)
        self.assertEqual(first["asks"]["gate"], "holdout")
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 0)
        result = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True).run(self.pid, resume=True)
        self.assertEqual(result["next"], "done", result)
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        attempts = self.dispatcher().state(self.pid)["attempts"]
        self.assertEqual(len(attempts), 6)
        legacy = SolutionTeam(self.store, self.saver, home=self.home).run(self.pid)
        self.assertEqual(legacy["next"], "stop")
        self.assertEqual(self.dispatcher().state(self.pid)["attempts"], attempts)

    def test_solution_team_crash_resume_keeps_attempt_and_does_not_repeat_data(self):
        from dclab_rnd.orchestra.solution import SolutionTeam
        class Crash(BaseException): pass
        save = TaskDispatcher._save

        def crash(d, pid, state, kind, task=None):
            if kind == "completed" and task == "data": raise Crash()
            return save(d, pid, state, kind, task)

        with patch.object(TaskDispatcher, "_save", new=crash), self.assertRaises(Crash):
            SolutionTeam(self.store, self.saver, home=self.home, task_execution=True).run(self.pid)
        record = self.store.read_stage(self.pid, "data")
        result = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True).run(self.pid)
        self.assertEqual(result["status"], "paused", result)
        self.assertEqual(result["asks"]["gate"], "holdout")
        self.assertEqual(record, self.store.read_stage(self.pid, "data"))
        attempts = self.dispatcher().state(self.pid)["attempts"]
        self.assertEqual([a["task"] for a in attempts], ["data", "leakage", "features", "models"])
        self.assertEqual(attempts[0]["status"], "reconciled")

    def test_schema_or_permission_change_refuses_an_existing_graph(self):
        from dataclasses import replace
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        original = d.registry.tools["run_stage"]
        schema = copy.deepcopy(original.parameters)
        schema["description"] = "changed definition"
        d.registry.tools["run_stage"] = replace(original, parameters=schema)
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        d.registry.tools["run_stage"] = original
        d.allowed.remove("run_stage")
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_changed_owner_output_cannot_satisfy_a_completed_receipt(self):
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        record = self.store.read_stage(self.pid, "data")
        record["evidence"]["tampered"] = True
        self.store.write_stage(self.pid, "data", record)
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_corrupt_progress_cannot_reclaim_budget_or_repeat_work(self):
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        state = d.state(self.pid)
        state["tool_calls"] = 0
        d._save(self.pid, state, "corrupt_fixture")
        self.assertEqual(d.dispatch(self.pid, "leakage")["status"], "blocked")
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_attempt_parent_does_not_change_scientific_lineage_fingerprint(self):
        from dclab_rnd.studio import lineage
        from dclab_core.lineage import fingerprint
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        run = self.store.read_stage(self.pid, "data")["run"]
        self.assertTrue(run["parent"].startswith("task_"))
        self.assertEqual(run["fingerprint"], fingerprint(run["inputs"], run["params"]))
        self.assertNotIn("task_attempt", run["inputs"]["refs"])
        self.assertIsNone(lineage.current_task_attempt())

    def test_attempt_links_actual_owner_and_guard_decision_ids(self):
        from dclab_rnd import ledger
        book = ledger.FileLedger(self.home / "ledger", "task-test")
        ledger.install(book)
        self.addCleanup(ledger.install, None)
        d = self.dispatcher()
        result = d.dispatch(self.pid, "data")
        self.assertEqual(result["status"], "completed", result)
        attempt = d.state(self.pid)["attempts"][0]
        actual = {r.id for run in (attempt["owner_run"], attempt["id"]) for r in book.records(project=self.pid, run=run)}
        self.assertTrue(actual)
        self.assertEqual(set(attempt["decision_ids"]), actual)
        self.assertEqual(attempt["decision_capture"], "linked")

    def test_known_time_overrun_cannot_disappear_after_crash_reconciliation(self):
        class Crash(BaseException): pass
        d = self.dispatcher(budget={"max_tool_calls": 6, "max_seconds": 1, "max_cost_eur": 0.0})
        save = d._save

        def crash(pid, state, kind, task=None):
            if kind == "completed": raise Crash()
            return save(pid, state, kind, task)

        with patch.object(d, "_save", side_effect=crash), self.assertRaises(Crash):
            d.dispatch(self.pid, "data")
        record = self.store.read_stage(self.pid, "data")
        record["run"]["seconds"] = 2.0  # authored over-limit owner duration, without a slow test
        self.store.write_stage(self.pid, "data", record)
        self.assertEqual(self.dispatcher().dispatch(self.pid, "data")["status"], "reconciled")
        self.assertTrue(self.dispatcher().state(self.pid)["budget_overrun"])
        result = self.dispatcher().dispatch(self.pid, "leakage")
        self.assertEqual(result["reason"], "execution_budget_exceeded", result)
        self.assertIsNone(self.store.read_stage(self.pid, "leakage"))

    def test_checkpoints_keep_source_hashes_without_private_class_values(self):
        import pandas as pd
        project = self.store.get(self.pid)
        solution = copy.deepcopy(project["solution"])
        path = data.data_path(self.store, project)
        frame = pd.read_csv(path)
        marker = "PRIVATE_POSITIVE_CLASS"
        frame["Churn"] = frame["Churn"].replace({"Yes": marker})
        frame.to_csv(path, index=False)
        data.attach_data(self.store, self.pid, path.name)
        project = self.store.get(self.pid)
        solution["positive_label"] = marker
        project["solution"] = solution
        self.store.save(project)
        graph.approve_gate(self.store, self.pid, "solution", by="fixture-owner", reason="Current invented class")
        d = self.dispatcher()
        result = d.dispatch(self.pid, "data")
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(marker not in json.dumps(d.state(self.pid)), "Private class value entered task checkpoint")


class PostgresTaskDispatchTests(TaskDispatchTests):
    def setUp(self):
        from tests import pgtest
        self.database_url = pgtest.require()
        pgtest.empty(self.database_url)
        super().setUp()

    def make_store(self):
        from dclab_rnd.storage.postgres import open_stores
        return open_stores(self.home, self.database_url)[0]

    def test_other_workspace_cannot_read_task_state_or_a_paused_question(self):
        from dclab_rnd.storage.postgres import open_stores
        from dclab_rnd.orchestra.solution import SolutionTeam
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, "data")["status"], "completed")
        other = open_stores(self.home / "other-workspace", self.database_url)[0]
        with self.assertRaises(KeyError):
            TaskDispatcher(other, self.saver).state(self.pid)
        with self.assertRaises(KeyError):
            SolutionTeam(other, self.saver, home=self.home, task_execution=True).run(self.pid)

    def test_advisory_owner_and_waiter_do_not_exhaust_a_single_connection_pool(self):
        import sqlalchemy as sa
        one = copy.copy(self.store)
        one.engine = sa.create_engine(self.database_url, pool_size=1, max_overflow=0, pool_timeout=0.2)
        self.addCleanup(one.engine.dispose)
        other = copy.copy(one)
        errors, waiting = [], threading.Event()

        def waiter():
            waiting.set()
            try:
                with other.turn(self.pid):
                    other.get(self.pid)
            except Exception as error:
                errors.append(type(error).__name__)

        with one.turn(self.pid):
            with other.turn(self.pid):  # cross-instance reentrancy in this same thread
                one.save(one.get(self.pid))
            thread = threading.Thread(target=waiter)
            thread.start()
            self.assertTrue(waiting.wait(1))
            one.update(self.pid, goal=one.get(self.pid)["goal"])
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
