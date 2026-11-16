"""Stage H5 of Hierarchical Planning (the owner, 2026-11-16: "Yes, do it"): a project built from Home gets its plan, and
what the plan catches is applied, before its first stage runs. Scripted models only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dclab_rnd import experience  # noqa: E402
from dclab_rnd.studio import ProjectStore, data, graph  # noqa: E402
from dclab_rnd.tasks import enforced  # noqa: E402
from dclab_rnd.tasks import path as P  # noqa: E402
from dclab_rnd.tasks import view  # noqa: E402

LEAK = "closure_recorded_after_review"


def home_project(store: ProjectStore, signed: bool = False) -> str:
    """A project as Home's build leaves it: a solution accepted from the audit, not signed, no stage run. Its table holds
    a column recorded after the outcome that the solution does not keep out."""
    project = store.create("Churn from Home", "telecom", "Predict which customers leave next month.")
    pid = project["id"]
    rng = np.random.default_rng(17)
    n = 160
    churn = (rng.random(n) < 0.3).astype(int)
    frame = pd.DataFrame({"customer_id": [f"c_{i}" for i in range(n)], "tenure": rng.integers(1, 72, n), "monthly_charge": rng.uniform(15, 110, n),
                          "signup_date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
                          LEAK: np.where(rng.random(n) < np.where(churn == 1, 0.78, 0.22), 1, 0), "churn": np.where(churn == 1, "yes", "no")})
    frame.to_csv(store.data_dir(pid) / "customers.csv", index=False)
    data.attach_data(store, pid, "customers.csv")
    project = store.get(pid)
    project["solution"] = {"target": "churn", "task": "binary", "positive_label": "yes", "prediction_moment": "At the monthly review, before outreach.",
                           "forbidden": [], "identifiers": ["customer_id"], "metric": "roc_auc", "labels": "complete"}
    project["settings"].update(quick=True, max_rows=500, private=False)
    store.save(project)
    if signed:
        graph.approve_gate(store, pid, "solution", by="the-person", reason="signed in the wizard")
    return pid


class Planner:
    def __init__(self, answer):
        self.answer, self.calls, self.spent_eur = answer, 0, 0.0

    def complete(self, messages, tools=None, max_tokens=None, **options):
        self.calls += 1
        return {"content": json.dumps(self.answer) if self.answer is not None else "not json"}


class Gateway:
    def __init__(self, client):
        self._client = client

    def client(self, purpose, **kwargs):
        return self._client

    def available(self, purpose):
        return True


class Case(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="path-plan-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        previous = experience.current_log()
        experience.install(experience.FileLog(self.home / "experience", "path-plan"))
        self.addCleanup(experience.install, previous)
        self.store = ProjectStore(self.home / "projects")

    def kept_out(self, pid):
        return enforced.forbidden_columns(self.store.get(pid)["solution"])


class PlanBeforeStagesTests(Case):
    def test_the_plan_is_made_and_its_catch_applied_before_any_stage(self):
        pid = home_project(self.store)
        self.assertNotIn(LEAK, self.kept_out(pid))  # as Home's build left it: the leak would have been a feature
        out = P.plan_project(self.store, pid)
        self.assertEqual((out["status"], out["routed_to"]), ("planned", "rules"))
        self.assertEqual([(a["field"], a["value"], a["by"]) for a in out["applied"]], [("forbidden", LEAK, "agent")])
        self.assertIn(LEAK, self.kept_out(pid))
        kept = P.stored(self.store.get(pid))
        self.assertEqual(kept["enforcement"]["applied"][0]["trap"], f"post_outcome_{LEAK}")
        self.assertEqual([s["task"] for leg in kept["path"]["legs"] for s in leg["steps"]], ["leakage_probe", "baseline"])
        self.assertIn(f"`{LEAK}`", P.says(out))  # the person is told, in the build's own message
        self.assertTrue(any(t.get("move") == "set_solution" and t.get("actor") == "agent" and t.get("status") == "allowed"
                            for t in self.store.transitions(pid)))  # through the validated move, on the record
        again = P.plan_project(self.store, pid)
        self.assertEqual((again["status"], again["applied"]), ("kept", []))  # made once

    def test_a_signed_solution_is_never_changed_and_the_catch_is_an_open_question(self):
        pid = home_project(self.store, signed=True)
        before = dict(self.store.get(pid)["solution"])
        out = P.plan_project(self.store, pid)
        self.assertEqual((out["status"], out["applied"]), ("planned", []))
        self.assertEqual(self.store.get(pid)["solution"], before)
        self.assertIn("did not change the solution by itself", out["asks"][0]["question"])
        self.assertIn("open question", P.says(out))

    def test_after_a_stage_ran_nothing_is_changed(self):
        pid = home_project(self.store)
        project = self.store.get(pid)
        project["stages"]["data"]["status"] = "completed"
        self.store.save(project)
        out = P.plan_project(self.store, pid)
        self.assertEqual(out["status"], "too_late")
        self.assertNotIn(LEAK, self.kept_out(pid))
        self.assertIsNone(P.stored(self.store.get(pid)))
        self.assertIsNone(P.says(out))

    def test_a_models_plan_replaces_the_rules_plan_while_no_stage_has_run(self):
        pid = home_project(self.store)
        P.plan_project(self.store, pid)
        answer = {"decomposition": {"decision": "Which customers to call before they leave", "target": "churn", "unit": "customer", "horizon_days": 30,
                                    "success_metric": "roc_auc",
                                    "subproblems": [{"id": "moment", "kind": "classification", "objective": "Fix what is known at the monthly review"},
                                                    {"id": "score", "kind": "classification", "objective": "Score each customer", "dependency_ids": ["moment"]}]},
                  "calls": [{"task_type": "time_split_check", "params": {"column": "signup_date"}, "purpose": "Later customers must be the test", "subproblem": "moment"},
                            {"task_type": "leakage_probe", "params": {"table": "customers", "column": LEAK}, "purpose": "Recorded after the outcome", "subproblem": "score"}],
                  "caught_traps": [f"post_outcome_{LEAK}", "time_split_required"]}
        client = Planner(answer)
        out = P.plan_project(self.store, pid, gateway=Gateway(client), use_model=True)
        self.assertEqual((out["status"], out["routed_to"]), ("planned", "model"))
        self.assertEqual([(a["field"], a["value"]) for a in out["applied"]], [("time_column", "signup_date")])  # the leak was kept out already
        solution = self.store.get(pid)["solution"]
        self.assertEqual(solution["time_column"], "signup_date")
        kept = P.stored(self.store.get(pid))
        self.assertEqual([leg["id"] for leg in kept["path"]["legs"]], ["moment", "score"])
        self.assertEqual({a["field"] for a in kept["enforcement"]["applied"]}, {"forbidden", "time_column"})  # what the rules' plan did is kept on the record
        self.assertEqual(P.plan_project(self.store, pid, gateway=Gateway(client), use_model=True)["status"], "kept")  # the model is asked once

    def test_a_model_that_cannot_plan_never_stops_the_build(self):
        pid = home_project(self.store)
        out = P.plan_project(self.store, pid, gateway=Gateway(Planner(None)), use_model=True)
        self.assertEqual((out["status"], out["routed_to"]), ("planned", "rules"))
        self.assertIn(LEAK, self.kept_out(pid))


def model_answer(calls, traps):
    return {"decomposition": {"decision": "Which customers to call before they leave", "target": "churn", "unit": "customer", "horizon_days": 30,
                              "success_metric": "roc_auc", "subproblems": [{"id": "score", "kind": "classification", "objective": "Score each customer"}]},
            "calls": calls, "caught_traps": traps}


class OnlyWhatCodeConfirmsTests(Case):
    """The review of 2026-11-16: nobody is in the room when the plan is applied on Home's build, so a model's word alone
    changes nothing, a catch is never moved onto another column, and the person's own decisions stand."""

    def test_a_catch_about_a_related_tables_column_is_never_moved_onto_the_main_table(self):
        pid = home_project(self.store)
        project = self.store.get(pid)
        frame = pd.read_csv(data.data_path(self.store, project))
        frame["status"], frame["amount"] = "open", 10.0
        frame.to_csv(self.store.data_dir(pid) / "customers.csv", index=False)
        data.attach_data(self.store, pid, "customers.csv")
        project = self.store.get(pid)
        project["tables"] = {"orders": {"columns": ["customer_id", "closed_status", "final_amount"]}}
        project["solution"] = {"target": "churn", "task": "binary", "positive_label": "yes", "prediction_moment": "At the monthly review, before outreach.",
                               "forbidden": [], "identifiers": ["customer_id"], "metric": "roc_auc", "labels": "complete"}
        self.store.save(project)
        out = P.plan_project(self.store, pid)
        kept_out = self.kept_out(pid)
        self.assertNotIn("status", kept_out)  # before: `orders.closed_status` kept out the main table's `status`
        self.assertNotIn("amount", kept_out)
        self.assertEqual([a["value"] for a in out["applied"]], [LEAK])
        self.assertFalse(any(a.get("value") in ("status", "amount") for a in out["asks"]))

    def test_a_models_word_alone_sets_no_split_and_keeps_no_column_out(self):
        pid = home_project(self.store)
        answer = model_answer([{"task_type": "time_split_check", "params": {"column": "tenure"}, "purpose": "Order by tenure"},
                               {"task_type": "grouped_split", "params": {"entity": "monthly_charge"}, "purpose": "Group by charge"},
                               {"task_type": "leakage_probe", "params": {"table": "customers", "column": "monthly_charge"}, "purpose": "Looks late"},
                               {"task_type": "leakage_probe", "params": {"table": "customers", "column": LEAK}, "purpose": "Recorded after the outcome"}],
                              ["post_outcome_monthly_charge", f"post_outcome_{LEAK}", "time_split_required"])
        out = P.plan_project(self.store, pid, gateway=Gateway(Planner(answer)), use_model=True)
        solution = self.store.get(pid)["solution"]
        self.assertEqual([(a["field"], a["value"]) for a in out["applied"]], [("forbidden", LEAK)])  # the one catch code itself confirms
        self.assertIsNone(solution.get("time_column"))  # `tenure` is not a time: before, the holdout became the longest-tenure customers
        self.assertIsNone(solution.get("group_column"))
        self.assertNotIn("monthly_charge", self.kept_out(pid))
        asked = {(a["field"], a["value"]) for a in out["asks"]}
        self.assertEqual(asked, {("time_column", "tenure"), ("group_column", "monthly_charge"), ("forbidden", "monthly_charge")})

    def test_a_snapshot_column_that_may_be_honest_is_asked_about_never_kept_out(self):
        pid = home_project(self.store)
        project = self.store.get(pid)
        frame = pd.read_csv(data.data_path(self.store, project))
        frame["lifetime_orders"] = 3
        frame.to_csv(self.store.data_dir(pid) / "customers.csv", index=False)
        data.attach_data(self.store, pid, "customers.csv")
        project = self.store.get(pid)
        project["solution"] = {"target": "churn", "task": "binary", "positive_label": "yes", "prediction_moment": "At the monthly review, before outreach.",
                               "forbidden": [], "identifiers": ["customer_id"], "metric": "roc_auc", "labels": "complete"}
        self.store.save(project)
        out = P.plan_project(self.store, pid)
        self.assertNotIn("lifetime_orders", self.kept_out(pid))  # known as of the review or as of the export? only the person can say
        self.assertIn(("forbidden", "lifetime_orders"), {(a["field"], a["value"]) for a in out["asks"]})
        self.assertIn(LEAK, self.kept_out(pid))  # a name the product's own audit keeps out is kept out

    def test_a_solution_the_person_wrote_is_never_changed(self):
        pid = home_project(self.store)
        project = self.store.get(pid)
        project["solution_saved_by"] = "human"
        self.store.save(project)
        before = dict(project["solution"])
        out = P.plan_project(self.store, pid)
        self.assertEqual((out["applied"], self.store.get(pid)["solution"]), ([], before))
        self.assertIn("you wrote this solution yourself", out["asks"][0]["question"])

    def test_a_column_the_person_said_is_known_stays_in_now_and_in_the_models_later_plan(self):
        pid = home_project(self.store)
        out = P.plan_project(self.store, pid, person_kept=[LEAK])
        self.assertEqual((out["applied"], out["asks"]), ([], []))
        self.assertNotIn(LEAK, self.kept_out(pid))
        self.assertEqual(P.stored(self.store.get(pid))["enforcement"]["waived"][0]["value"], LEAK)
        answer = model_answer([{"task_type": "leakage_probe", "params": {"table": "customers", "column": LEAK}, "purpose": "Recorded after the outcome"}],
                              [f"post_outcome_{LEAK}"])
        later = P.plan_project(self.store, pid, gateway=Gateway(Planner(answer)), use_model=True)  # the job's pass, with no one to pass the answer again
        self.assertEqual((later["routed_to"], later["applied"]), ("model", []))
        self.assertNotIn(LEAK, self.kept_out(pid))

    def test_a_solution_changed_while_the_model_thought_is_left_as_the_person_made_it(self):
        pid = home_project(self.store)
        store = self.store

        class Slow(Planner):
            def complete(self, messages, tools=None, max_tokens=None, **options):
                project = store.get(pid)
                project["solution"]["metric"] = "pr_auc"  # the person edits the solution during the model's minutes
                store.save(project)
                return super().complete(messages, tools, max_tokens, **options)

        answer = model_answer([{"task_type": "leakage_probe", "params": {"table": "customers", "column": LEAK}, "purpose": "Recorded after the outcome"}],
                              [f"post_outcome_{LEAK}"])
        out = P.plan_project(self.store, pid, gateway=Gateway(Slow(answer)), use_model=True)
        self.assertEqual((out["status"], out["applied"]), ("changed_meanwhile", []))
        self.assertNotIn(LEAK, self.kept_out(pid))

    def test_a_project_of_several_parts_or_with_no_outcome_is_left_alone(self):
        pid = home_project(self.store)
        project = self.store.get(pid)
        project["solution_map"] = {"blocks": [{"sub_problem": "a"}]}
        self.store.save(project)
        self.assertEqual(P.plan_project(self.store, pid)["status"], "skipped")
        self.assertNotIn(LEAK, self.kept_out(pid))


class ShownTests(Case):
    def test_the_page_shows_the_stored_plan_with_what_was_done_and_that_it_is_followed(self):
        pid = home_project(self.store)
        P.plan_project(self.store, pid)
        shown = view.task_plan_view(self.store, self.home, pid, enabled=True)["catalog_plan"]
        steps = {s["task"]: s for leg in shown["path"]["legs"] for s in leg["steps"]}
        self.assertEqual(steps["leakage_probe"]["did"], f"enforced: forbidden = {LEAK}")
        self.assertEqual((shown["followed"]["checked"], shown["followed"]["followed"], shown["followed"]["not_followed"]), (1, 1, []))
        self.assertEqual(shown["mode"], "made before the stages")
        self.assertEqual(shown["path"]["next"]["who"], "DCLab")
        self.assertEqual(shown["path"]["goal"], "Predict which customers leave next month.")  # a kept plan starts from the person's own goal

    def test_an_open_question_of_the_plan_is_listed_on_the_page(self):
        pid = home_project(self.store, signed=True)
        P.plan_project(self.store, pid)
        shown = view.task_plan_view(self.store, self.home, pid, enabled=True)["catalog_plan"]
        self.assertIn("did not change the solution by itself", shown["clarifying_questions"][0]["question"])
        self.assertEqual(len(shown["followed"]["not_followed"]), 1)  # and the page says the solution does not follow its plan


    def test_a_question_the_person_settled_on_the_solution_is_no_longer_open(self):
        pid = home_project(self.store, signed=True)
        P.plan_project(self.store, pid)
        project = self.store.get(pid)
        project["solution"]["forbidden"] = [{"column": LEAK, "reason": "the person kept it out"}]
        self.store.save(project)
        shown = view.task_plan_view(self.store, self.home, pid, enabled=True)["catalog_plan"]
        self.assertEqual(shown["clarifying_questions"], [])
        self.assertEqual(shown["followed"]["not_followed"], [])


    def test_after_the_stages_ran_the_page_says_what_ran_what_is_followed_and_what_is_still_a_question(self):
        """A real Home build (2026-11-16, no model): five finished stages were drawn as "blocked", a split by the time column
        was read as not by time, an open question was shown as a broken plan, and the next step named a tool."""
        pid = home_project(self.store)
        project = self.store.get(pid)
        frame = pd.read_csv(data.data_path(self.store, project))
        frame["lifetime_orders"] = 3
        frame.to_csv(self.store.data_dir(pid) / "customers.csv", index=False)
        data.attach_data(self.store, pid, "customers.csv")
        project = self.store.get(pid)
        project["solution"] = {"target": "churn", "task": "binary", "positive_label": "yes", "prediction_moment": "At the monthly review, before outreach.",
                               "forbidden": [], "identifiers": ["customer_id"], "metric": "roc_auc", "labels": "complete", "time_column": "signup_date"}
        self.store.save(project)
        P.plan_project(self.store, pid)
        before = view.task_plan_view(self.store, self.home, pid, enabled=True)["catalog_plan"]
        self.assertIn("Is `lifetime_orders` known at the prediction moment?", before["path"]["next"]["what"])  # the question, not "leakage_probe"
        project = self.store.get(pid)
        for name in project["stages"]:
            project["stages"][name] = {"status": "completed", "run": f"run_{name}"}
        self.store.save(project)
        self.store.write_stage(pid, "data", {"stage": "data", "status": "completed", "evidence": {"holdout_split": "last 32 of 160 rows by `signup_date`"}})
        shown = view.task_plan_view(self.store, self.home, pid, enabled=True)["catalog_plan"]
        self.assertEqual({b["status"] for b in shown["path"]["build"]}, {"completed"})
        self.assertEqual(shown["followed"]["not_followed"], [])  # the split is by time, and the open question is not a broken plan
        self.assertEqual(shown["followed"]["open"], 1)
        self.assertEqual(shown["path"]["next"]["who"], "you")  # what is left is the person's question


class JobTests(Case):
    def services(self, pid, admit=True):
        return SimpleNamespace(projects=self.store, store=SimpleNamespace(home=self.home), gateway=Gateway(Planner(None)),
                               intern_sessions=SimpleNamespace(get=lambda session_id: {"project_id": pid} if session_id == "s1" else {}))

    def test_the_stages_job_makes_the_plan_first(self):
        from dclab_rnd.jobs import handlers

        pid = home_project(self.store)
        handlers._plan_first(self.services(pid), "s1")
        self.assertIn(LEAK, self.kept_out(pid))
        self.assertIsNotNone(P.stored(self.store.get(pid)))
        handlers._plan_first(self.services(pid), "another session with no project")  # nothing to plan: no error, nothing changed

    def test_only_the_builds_own_job_plans_first(self):
        from dclab_rnd.jobs import handlers

        pid = home_project(self.store)
        services = self.services(pid)
        services.intern = lambda: SimpleNamespace(run=lambda session_id: None, message=lambda *a, **k: None)
        handlers._intern_run(services, {"session_id": "s1", "action": "run"})  # a session a person started on their own project
        self.assertNotIn(LEAK, self.kept_out(pid))
        self.assertIsNone(P.stored(self.store.get(pid)))
        handlers._intern_run(services, {"session_id": "s1", "action": "run", "plan_first": True})  # Home's build
        self.assertIn(LEAK, self.kept_out(pid))

    def test_the_build_message_says_what_the_plan_did(self):
        source = (ROOT / "dclab_rnd" / "draft" / "chat.py").read_text(encoding="utf-8")
        self.assertIn('if done.get("plan_note"):', source)
        api = (ROOT / "dclab_rnd" / "draft" / "api.py").read_text(encoding="utf-8")
        self.assertIn("task_path.plan_project(projects, project[\"id\"], person_kept=kept)", api)
        self.assertIn('{"action": "run", "plan_first": True}', api)


if __name__ == "__main__":
    unittest.main()
