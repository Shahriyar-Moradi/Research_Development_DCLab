"""Hierarchical Planning, stage H0 (2026-11-16): did the engine obey the plan? A trap the plan catches and the model then
falls into is a failure, whatever the report says. Pure code on stored records; no model, no table."""
import ast
import json
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.tasks import enforced as E  # noqa: E402

COLUMNS = ["customer_id", "signup_date", "monthly_charge", "closure_recorded_after_review", "churn"]


def stages(split="stratified random 36 of 180 rows", features=("closure_recorded_after_review=1", "monthly_charge"), final=True):
    out = {"data": {"evidence": {"holdout_split": split}}}
    if final:
        out["final"] = {"evidence": {"configuration_results": [{"top_features": [{"feature": f} for f in features]}]}}
    return out


class ReadingTests(unittest.TestCase):
    def test_a_trap_is_read_by_its_kind_and_the_column_it_names(self):
        self.assertEqual((E.kind_of("post_outcome_closure_recorded_after_review"), E.column_of("post_outcome_closure_recorded_after_review", COLUMNS)),
                         ("forbid", "closure_recorded_after_review"))
        self.assertEqual((E.kind_of("grouped_split_customer_id"), E.column_of("grouped_split_customer_id", COLUMNS)), ("group", "customer_id"))
        self.assertEqual(E.kind_of("time_split_required"), "time_split")
        self.assertIsNone(E.kind_of("asymmetric_cost"))
        self.assertIsNone(E.column_of("post_outcome", COLUMNS))  # a trap that names no column cannot be enforced on one

    def test_the_final_models_columns_are_read_through_their_encoded_names(self):
        self.assertEqual(E.final_features(stages()), ["closure_recorded_after_review", "monthly_charge"])


class ObeyedTests(unittest.TestCase):
    def one(self, trap, solution=None, **kw):
        return E.obeyed([trap], solution or {}, stages(**kw), COLUMNS)[0]

    def test_a_column_known_after_the_outcome_must_be_forbidden_and_out_of_the_final_model(self):
        trap = "post_outcome_closure_recorded_after_review"
        self.assertFalse(self.one(trap)["enforced"])  # caught, and the top feature of the final model
        self.assertFalse(self.one(trap, {"forbidden": ["closure_recorded_after_review"]})["enforced"])  # forbidden on paper, still used
        self.assertTrue(self.one(trap, {"forbidden": ["closure_recorded_after_review"]}, features=("monthly_charge",))["enforced"])
        self.assertFalse(self.one(trap, {}, features=("monthly_charge",))["enforced"])  # absent by luck is not enforced: nothing forbids it

    def test_a_group_trap_needs_the_split_to_keep_the_group_on_one_side(self):
        self.assertFalse(self.one("grouped_split_customer_id")["enforced"])
        self.assertTrue(self.one("grouped_split_customer_id", {"group_column": "customer_id"})["enforced"])
        self.assertTrue(self.one("repeated_entity_leak_customer_id", split="group (declared group column customer_id)")["enforced"])

    def test_a_time_trap_needs_a_split_by_time(self):
        self.assertFalse(self.one("time_split_required")["enforced"])
        self.assertTrue(self.one("time_split_required", split="time (declared time column): the latest 36 rows")["enforced"])
        self.assertIsNone(E.obeyed(["time_split_required"], {}, {}, COLUMNS)[0]["enforced"])  # the data stage has not run: not read, never a pass

    def test_a_split_ordered_by_the_solutions_time_column_is_a_split_by_time(self):
        said = {"data": {"evidence": {"holdout_split": "last 120 of 600 rows by `signup_date`"}}}
        self.assertTrue(E.obeyed(["time_split_required"], {"time_column": "signup_date"}, said, ["signup_date"])[0]["enforced"])
        self.assertFalse(E.obeyed(["time_split_required"], {}, said, ["signup_date"])[0]["enforced"])  # no time column named: not known to be by time

    def test_a_trap_code_cannot_enforce_is_never_counted_as_enforced(self):
        record = self.one("asymmetric_cost")
        self.assertEqual((record["enforceable"], record["enforced"]), (False, None))
        counted = E.summary(E.obeyed(["asymmetric_cost", "time_split_required", "post_outcome_closure_recorded_after_review"], {}, stages(), COLUMNS))
        self.assertEqual((counted["caught"], counted["enforceable"], counted["enforced"], counted["rate"]), (3, 2, 0, 0.0))
        self.assertEqual(counted["broken"], ["time_split_required", "post_outcome_closure_recorded_after_review"])


class ChangesTests(unittest.TestCase):
    """Stage H2: what a plan's catches ask of the solution before the first stage runs."""

    def of(self, traps, solution=None, calls=(), time_candidates=()):
        return {(c["kind"], c["action"], c["field"], c["value"]) for c in E.changes(traps, calls, {"target": "churn", **(solution or {})}, COLUMNS, time_candidates)}

    def test_each_catch_becomes_one_stricter_change(self):
        got = self.of(["post_outcome_closure_recorded_after_review", "grouped_split_customer_id", "repeated_entity_leak_customer_id", "time_split_required"],
                      calls=[{"task_type": "time_split_check", "params": {"column": "signup_date"}}])
        self.assertEqual(got, {("forbid", "apply", "forbidden", "closure_recorded_after_review"), ("group", "apply", "group_column", "customer_id"),
                               ("time_split", "apply", "time_column", "signup_date")})

    def test_what_is_already_so_changes_nothing(self):
        solution = {"forbidden": [{"column": "closure_recorded_after_review", "reason": "known after"}], "group_column": "customer_id", "time_column": "signup_date"}
        got = E.changes(["post_outcome_closure_recorded_after_review", "grouped_split_customer_id", "time_split_required"], [], {"target": "churn", **solution}, COLUMNS)
        self.assertEqual({c["action"] for c in got}, {"none"})

    def test_what_code_cannot_tell_is_a_question_never_a_guess(self):
        asks = [c for c in E.changes(["time_split_required"], [], {"target": "churn"}, COLUMNS, time_candidates=["signup_date", "last_seen"])]
        self.assertEqual((asks[0]["action"], asks[0]["field"]), ("ask", "time_column"))
        self.assertIn("Which column holds the time", asks[0]["why"])
        self.assertEqual(self.of(["time_split_required"], time_candidates=["signup_date"]), {("time_split", "apply", "time_column", "signup_date")})  # one candidate: it is the one
        other = E.changes(["grouped_split_customer_id"], [], {"target": "churn", "group_column": "signup_date"}, COLUMNS)[0]
        self.assertEqual(other["action"], "ask")  # a group is already set to another column: the person chooses

    def test_the_outcome_a_missing_column_and_an_unenforceable_trap_are_never_touched(self):
        self.assertEqual(E.changes(["post_outcome_churn", "post_outcome_no_such_column", "asymmetric_cost"], [], {"target": "churn"}, COLUMNS), [])

    def test_the_plans_word_keeps_out_only_a_share_of_the_table(self):
        columns = ["a", "b", "c", "d", "e", "f", "y"]
        got = E.changes([f"post_outcome_{c}" for c in "abcd"], [], {"target": "y"}, columns)
        self.assertEqual([c["action"] for c in got], ["apply", "apply", "ask", "ask"])  # two of seven on its own word, then the person is asked

    def test_a_column_the_split_is_made_on_is_never_also_kept_out(self):
        cols = ["signup_date", "closure", "churn", "a", "b", "c", "d"]
        out = E.changes(["time_split_required", "post_outcome_signup_date", "post_outcome_closure"], [], {"target": "churn"}, cols, ["signup_date"])
        self.assertEqual([(c["field"], c["value"]) for c in out], [("time_column", "signup_date"), ("forbidden", "closure")])  # the solution refuses a time column that is also forbidden
        grouped = E.changes(["post_outcome_a"], [], {"target": "churn", "group_column": "a"}, cols)
        self.assertEqual(grouped, [])

    def test_the_outcome_is_never_a_catch_about_itself(self):
        records = E.obeyed(["post_outcome_returned"], {"target": "returned"}, {}, ["returned", "x"])
        self.assertFalse(records[0]["enforceable"])  # before: a clean project with a target named `returned` was shown as not following its plan

    def test_a_forbidden_list_of_columns_with_reasons_is_read(self):
        solution = {"forbidden": [{"column": "closure_recorded_after_review", "reason": "x"}]}
        record = E.obeyed(["post_outcome_closure_recorded_after_review"], solution, stages(features=("monthly_charge",)), COLUMNS)[0]
        self.assertTrue(record["enforced"])


def _runtime():
    """The scripted planner and the test project of the runtime tests, with a first plan that catches the leak."""
    import test_task_catalog_runtime as rt

    class Catching(rt._HybridCatalogServer):
        def complete(self, messages, tools=None, max_tokens=None, **options):
            out = super().complete(messages, tools, max_tokens, **options)
            answer = json.loads(out["content"])
            if self.calls and self.calls[-1] == "catalog_plan":
                answer["caught_traps"] = ["temporal_drift", "time_split_required", "post_outcome_closure_recorded_after_review", "asymmetric_cost"]
                out["content"] = json.dumps(answer)
            return out

    return rt, Catching()


class EngineObeysPlanTests(unittest.TestCase):
    """Stage H2 on a real project store: what the plan catches changes the solution before any stage runs. The task path
    runs only on a solution a person signed, and an agent may not change a signed solution: so the person is asked, and
    their yes makes the change theirs."""

    def run_case(self, body):
        import tempfile
        from contextlib import ExitStack

        from dclab_rnd import experience, ledger
        from dclab_rnd.agent_eval.task_planning.probe import offline_environment
        from dclab_rnd.models import FileUsage, Gateway
        from dclab_rnd.orchestra.checkpoints import open_checkpointer
        from dclab_rnd.studio import ProjectStore
        from dclab_rnd.tasks.dispatch import TaskDispatcher
        from dclab_rnd.tasks.host import planning_host

        rt, server = _runtime()
        with tempfile.TemporaryDirectory(prefix="plan-enf-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            previous = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "plan-enf"))
            ledger.install(None)
            stack.callback(experience.install, previous[0])
            stack.callback(ledger.install, previous[1])
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = rt._make_runtime_project(store, include_unresolved_leak=True)
            gateway = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)
            dispatcher = TaskDispatcher(store, saver, gateway=gateway, enabled=True, admit_proposals=True)
            with planning_host(name="plan-enf"):
                self.assertEqual(dispatcher.prepare_catalog_plan(pid)["status"], "planned")
                body(store, dispatcher, pid)

    def solution(self, store, pid):
        return store.get(pid)["solution"]

    def test_a_signed_solution_is_never_changed_by_the_agent_and_the_persons_yes_makes_the_change_theirs(self):
        def body(store, dispatcher, pid):
            before = dict(self.solution(store, pid))
            out = dispatcher.enforce_plan(pid)
            self.assertEqual((out["status"], out["applied"]), ("asked", []))
            self.assertEqual(self.solution(store, pid), before)  # untouched: only a person may change a signed solution
            self.assertTrue(all("may not change this solution by itself" in a["question"] for a in out["asks"]))
            self.assertEqual(len(dispatcher.plan_unenforced(pid)), 2)  # and the final stage would wait for the answer
            out = dispatcher.enforce_plan(pid, reply=True)
            self.assertEqual({(a["field"], a["value"], a["by"]) for a in out["applied"]},
                             {("forbidden", "closure_recorded_after_review", "human"), ("time_column", "signup_date", "human")})
            self.assertTrue(out["needs_signoff"])  # the changed solution is theirs to sign again: no approval is written for them
            self.assertEqual(dispatcher.plan_unenforced(pid), [])
            solution = self.solution(store, pid)
            self.assertEqual(solution["time_column"], "signup_date")
            self.assertIn("closure_recorded_after_review", E.forbidden_columns(solution))
            self.assertEqual(dispatcher.preflight(pid), "solution_changed_sign_it_again")  # no stage runs on an unsigned change
            self.assertEqual(dispatcher.enforce_plan(pid)["status"], "nothing")  # nothing is asked or applied twice
            kept = dispatcher.state(pid)["catalog_plan"]
            self.assertEqual(len(kept["enforcement"]["applied"]), 2)
            self.assertIn("post_outcome_closure_recorded_after_review", kept["caught_traps"])  # the plan itself is kept
            # The review of 2026-11-16: once the person signs again, the reason is the real one. A split by time is not a
            # workflow the task path can run, so the team hands the project to the stage path; before, it said "sign it
            # again" for ever.
            from dclab_rnd.studio import graph as studio_graph

            studio_graph.approve_gate(store, pid, "solution", by="the-person", reason="signed again after the plan's change")
            self.assertEqual(dispatcher.preflight(pid), "source_or_capability_blocked")
            state = dispatcher.state(pid)
            self.assertTrue(state.get("rebase") and not state["attempts"])  # what ``orchestra.service`` reads to walk the stages directly

        self.run_case(body)

    def test_a_reply_that_is_neither_yes_nor_no_settles_nothing(self):
        def body(store, dispatcher, pid):
            before = dict(self.solution(store, pid))
            dispatcher.enforce_plan(pid)
            out = dispatcher.enforce_plan(pid, reply="what does that column hold?")
            self.assertEqual(out["status"], "asked")  # before: any reply that was not a yes was stored as the person's waiver
            self.assertEqual(self.solution(store, pid), before)
            self.assertEqual(dispatcher.state(pid)["catalog_plan"]["enforcement"]["waived"], [])
            self.assertEqual(len(dispatcher.plan_unenforced(pid)), 2)  # the final stage still waits

        self.run_case(body)

    def test_a_change_the_solution_refuses_never_sinks_the_others_nor_counts_as_the_persons_choice(self):
        def body(store, dispatcher, pid):
            real = dispatcher._change_solution

            def picky(project_id, changes, actor):  # the solution's own rules refuse the time change, and only that one
                return "the time column cannot be used this way" if changes[0]["field"] == "time_column" else real(project_id, changes, actor)

            dispatcher.enforce_plan(pid)
            with mock.patch.object(dispatcher, "_change_solution", side_effect=picky):
                out = dispatcher.enforce_plan(pid, reply=True)
            self.assertEqual([(a["field"], a["value"]) for a in out["applied"]], [("forbidden", "closure_recorded_after_review")])  # the leak is kept out
            record = dispatcher.state(pid)["catalog_plan"]["enforcement"]
            self.assertEqual((record["waived"], [r["value"] for r in record["refused"]]), ([], ["signup_date"]))
            self.assertIn("own rules refuse this change", out["asks"][0]["question"])
            self.assertEqual([c["field"] for c in dispatcher.plan_unenforced(pid)], ["time_column"])  # still unenforced: the final stage waits

        self.run_case(body)

    def test_a_person_who_says_no_is_recorded_and_never_asked_again(self):
        def body(store, dispatcher, pid):
            before = dict(self.solution(store, pid))
            dispatcher.enforce_plan(pid)
            out = dispatcher.enforce_plan(pid, reply=False)
            self.assertEqual((out["applied"], out["asks"]), ([], []))
            self.assertEqual(self.solution(store, pid), before)
            waived = dispatcher.state(pid)["catalog_plan"]["enforcement"]["waived"]
            self.assertEqual({w["value"] for w in waived}, {"closure_recorded_after_review", "signup_date"})
            self.assertEqual(dispatcher.plan_unenforced(pid), [])  # their decision: the final stage may run
            self.assertEqual(dispatcher.enforce_plan(pid)["asks"], [])

        self.run_case(body)

    def test_after_work_began_nothing_changes_behind_the_persons_back(self):
        def body(store, dispatcher, pid):
            project = store.get(pid)
            project["stages"]["data"]["status"] = "completed"
            store.save(project)
            before = dict(self.solution(store, pid))
            out = dispatcher.enforce_plan(pid)
            self.assertEqual(out["status"], "asked")
            self.assertTrue(all(a["question"].startswith("After work began") for a in out["asks"]))
            stop = dispatcher.enforce_plan(pid, reply=True)
            self.assertEqual(stop["status"], "stop")
            self.assertIn("run the stages again", stop["says"])
            self.assertEqual(self.solution(store, pid), before)
            self.assertEqual(dispatcher.enforce_plan(pid, reply="no, keep it as it is")["status"], "waived")
            self.assertEqual(dispatcher.plan_unenforced(pid), [])

        self.run_case(body)


class StoredRunTests(unittest.TestCase):
    """The baseline, from a live run kept before this work (2026-11-13): every scenario "passed", and no trap the plan
    caught was enforced. Skipped where the kept run is not on disk (a clean checkout)."""

    RUN = ROOT / "agent_runs" / "live_tests" / "adaptive_studio_live_20261113_r4"

    @unittest.skipUnless((RUN / "summary.json").exists(), "the kept live run is not on this machine")
    def test_the_kept_run_caught_the_leak_and_used_it(self):
        scenario = next(s for s in json.loads((self.RUN / "summary.json").read_text())["scenarios"] if s["scenario_id"] == "studio_multitable_post_outcome")
        folder = self.RUN / "projects" / scenario["project_id"]
        solution = json.loads((folder / "project.json").read_text()).get("solution") or {}
        kept = {p.stem: json.loads(p.read_text()) for p in (folder / "stages").glob("*.json")}
        columns = [f["feature"] for f in kept["data"]["evidence"]["feature_profiles"]] + ["closure_recorded_after_review"]
        traps = scenario["caught_traps"] if isinstance(scenario["caught_traps"], list) else ast.literal_eval(scenario["caught_traps"])
        counted = E.summary(E.obeyed(traps, solution, kept, columns))
        self.assertTrue(scenario["passed"])
        self.assertEqual((counted["enforceable"], counted["enforced"]), (4, 0))
        self.assertIn("post_outcome_closure_recorded_after_review", counted["broken"])


class GateTests(unittest.TestCase):
    def project(self, read=2, enforced=2, stopped=False):
        return {"summary": {"read": read, "enforced": enforced}, "stopped_with_question": stopped}

    def test_the_gate_is_pinned_before_the_repair(self):
        from dclab_rnd.agent_eval.plan_enforced import gate

        self.assertIsNone(gate.pinned())
        self.assertEqual((gate.TERMS["obeyed"], gate.TERMS["people"], gate.TERMS["min_projects"]), (1.0, "dev3", 8))

    def test_every_caught_trap_must_be_enforced_or_the_build_must_have_stopped(self):
        from dclab_rnd.agent_eval.plan_enforced import gate

        self.assertTrue(gate.obeyed_check([self.project()] * 8)["ok"])
        self.assertFalse(gate.obeyed_check([self.project()] * 7 + [self.project(enforced=1)])["ok"])  # one trap not enforced in one project
        self.assertTrue(gate.obeyed_check([self.project()] * 7 + [self.project(enforced=0, stopped=True)])["ok"])  # it stopped and asked
        self.assertFalse(gate.obeyed_check([self.project()] * 4)["ok"])  # too few projects
        unread = gate.obeyed_check([self.project(read=0, enforced=0)] * 8)
        self.assertEqual((unread["ok"], unread["read"]), (False, False))  # nothing enforceable was read: never a pass

    def test_the_stored_baseline_fails_the_check_it_was_stored_for(self):
        path = ROOT / "evidence" / "campaigns" / "plan_enforced_v1" / "results" / "PE-001.json"
        doc = json.loads(path.read_text())
        self.assertEqual((doc["totals"]["reported_passed"], doc["totals"]["enforceable"], doc["totals"]["enforced"]), (5, 8, 0))
        self.assertFalse(doc["check_obeyed"]["ok"])
        self.assertEqual(doc["source"]["live_model_calls"], 0)
        paused = next(p for p in doc["projects"] if p["paused_with_question"])
        self.assertFalse(paused["stopped_with_question"])  # it asked, went on and built: that is not stopping


if __name__ == "__main__":
    unittest.main()
