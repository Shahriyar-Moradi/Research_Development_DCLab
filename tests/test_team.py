"""Package 18.8: the roles as one team on the workflow graph. The fraud, churn and delivery replicas run end to end with
every role asked (a scripted advisor: no model is called), every move validated, the person asked only what is still
open and their answers used, the gates signed by the person; a gate the person declines stops the run there."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.agents import ScriptedAdvisor
    from dclab_rnd.agent_eval import team as harness
    from dclab_rnd.studio import ProjectStore, data as sd, engine, team
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""


def echo_reporter(evidence):
    return {"summaries": [{"title": s["title"], "text": list(s["lines"].values())[0] + " [L1]"} for s in evidence["sections"]]}


def advisor():
    """Every role asked: the reporter echoes a line (kept by the checks); the others abstain, so their rules stand."""
    return ScriptedAdvisor({"reporter": echo_reporter})


def person(case, approve=True, **override):
    answers = {"target_column": case.target, "prediction_moment": case.moment, "budget": str(case.solution.get("alert_budget") or ""), **override}
    gates = approve if isinstance(approve, dict) else {"solution": approve, "holdout": approve}
    return team.Person(answer=lambda field, question: answers.get(field), approve=lambda gate, summary: gates.get(gate, True),
                       who="scripted person")


@unittest.skipIf(MISSING, MISSING)
class TeamTest(unittest.TestCase):
    def run_team(self, key, adv=None, **kw):
        from dclab_rnd.expansion import runner

        store, pid, case = harness._project(key)
        adv = adv if adv is not None else advisor()
        with runner.fast_profile():
            out = team.run(store, pid, harness.DESCRIPTIONS[key], person(case, **kw), adv, quick=case.quick)
        return store, pid, case, out, adv

    def test_three_replicas_end_to_end_every_role_asked_every_move_validated(self):
        for key in harness.KINDS:
            with self.subTest(kind=key):
                store, pid, case, out, adv = self.run_team(key)
                self.assertEqual(out["status"], "completed", out)
                self.assertEqual([s["role"] for s in out["steps"]],
                                 ["problem_analyst", "data_investigator", "solution_planner", "feature_engineer", "critic", "reporter"])
                self.assertEqual(out["steps"][-1]["source"], "advisor")  # the reporter's summaries were kept
                self.assertEqual({s["source"] for s in out["steps"][:-1]}, {"fallback"})  # the others were asked and abstained: said so
                asked = {p for p, _ in adv.requests}
                self.assertTrue({"problem_analyst", "brief_checker", "data_investigator", "solution_planner", "feature_engineer",
                                 "critic", "reporter"} <= asked, asked)  # the critic through the team's own advisor
                moves = store.transitions(pid, limit=2000)
                self.assertTrue(harness._approvals_came(moves))
                approvals = [m for m in moves if m["move"] == "approve_gate" and m["status"] == "allowed"]
                self.assertEqual({m.get("args", {}).get("gate") for m in approvals}, {"solution", "holdout"})
                self.assertEqual(store.read_stage(pid, "final")["evidence"]["holdout_uses_in_this_project"], 1)
                self.assertFalse(store.get(pid)["settings"].get("critic"))  # the critic was the run's, not left on
                self.assertEqual(len([e for e in store.activity(pid, limit=500) if e["kind"] == "team_step"]), len(out["steps"]))

    def test_a_planned_grouped_forward_split_the_engine_cannot_test_is_not_saved(self):  # 28.6: the debate's live runs found it
        import numpy as np
        import pandas as pd

        from dclab_rnd.expansion import runner

        rng = np.random.default_rng(0)
        n = 1500
        frame = pd.DataFrame({"created_at": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
                              "customer_id": rng.integers(0, 120, n), "amount": rng.gamma(2, 20, n).round(2),  # customers recur over time
                              "visits": rng.integers(1, 9, n), "churned": (rng.random(n) < 0.3).astype(int)})
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("x", "general", "Predict which customers churn next month.")["id"]
        frame.to_csv(store.data_dir(pid) / "customers.csv", index=False)
        sd.attach_data(store, pid, "customers.csv")

        def planner(evidence):  # a split labelled a group that still carries the time order: the validator lets it through
            cite = next(f.split(" ")[0] for f in evidence["findings"] if "customer_id" in f)
            return {"decisions": [{"key": "split.strategy", "choice": {"kind": "group", "group_column": "customer_id", "time_column": "created_at",
                                                                      "holdout": "locked first, used once, after the choice"},
                                   "reason": "the same customer appears on many rows", "cites": [cite]}]}
        adv = ScriptedAdvisor({"reporter": echo_reporter, "solution_planner": planner})
        said = {"target_column": "churned", "prediction_moment": "At the start of each month, before the renewal is due"}
        who = team.Person(answer=lambda field, question: said.get(field), approve=lambda gate, summary: True, who="scripted person")
        with runner.fast_profile():
            out = team.run(store, pid, "Predict which customers churn next month.", who, adv, quick=True)
        self.assertEqual(out["status"], "completed", out)  # the engine ran to its holdout
        solution = store.get(pid)["solution"]
        self.assertEqual((solution.get("time_column"), solution.get("group_column")), ("created_at", None))  # the time order kept
        why = [s for s in out["steps"] if s["role"] == "solution_planner" and "recurs" in s["reason"]]
        self.assertTrue(why, out["steps"])  # said, as a step of the run

    def test_the_teams_moves_are_steps_of_its_run(self):
        # 21.1: the team runs in the agent harness: each role's move is a step of one run, which ends recorded
        from dclab_rnd import experience

        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        try:
            store, pid, case, out, _ = self.run_team("binary")
        finally:
            experience.install(None)
        events = log.read()
        steps = [e for e in events if e.kind == "agent.step" and e.payload["agent"] == "team"]
        self.assertEqual([e.payload.get("tool") for e in steps], [s["role"] for s in out["steps"]])
        self.assertTrue(all(e.ids.project == pid for e in steps))
        self.assertEqual(len({e.ids.run for e in steps}), 1)
        done = [e for e in events if e.kind == "run.recorded" and e.ids.run == steps[0].ids.run]
        self.assertEqual((done[-1].payload["name"], done[-1].payload["status"]), ("team", "completed"))

    def test_a_kept_measurement_leaves_its_runs_where_the_store_collects_them(self):
        """2026-10-30: every team measurement ran in a temporary folder, so the store had no team run: a kept one leaves its
        projects, its session log (the team's run.recorded) and its ledger in a workspace folder marked scripted."""
        from unittest import mock

        from dclab_rnd import experience

        keep = Path(tempfile.mkdtemp()) / "team-run"
        with mock.patch.object(harness, "KINDS", ("binary",)):
            result = harness.measure(advisor(), keep)
        self.assertEqual(result["rows"][0]["status"], "completed")
        self.assertTrue((keep / "scripted.json").is_file() and any((keep / "projects").iterdir()))
        events = experience.open_log(keep).read()
        self.assertIn(("team", "completed"), [(e.payload.get("name"), e.payload.get("status")) for e in events if e.kind == "run.recorded"])
        self.assertIsNone(experience._INSTALLED.get("log"))  # the process's log put back

    def test_only_what_is_open_is_asked_and_the_answers_are_used(self):
        store, pid, case, out, _ = self.run_team("rare_event")
        fields = [q["field"] for q in out["asked"]]
        self.assertNotIn("target_column", fields)  # the label file names the target
        self.assertNotIn("task", fields)  # the data settles it
        self.assertIn("budget", fields)  # the rare-event kind needs it, and nothing else says it
        solution = store.get(pid)["solution"]
        self.assertEqual((solution["labels"], solution["alert_budget"]), ("partial", 0.005))
        self.assertEqual(store.get(pid)["pack"], "imbalanced")
        plan = store.get(pid)["plan"]
        self.assertNotIn("is assumed, not stated", str(plan.get("would_change")))  # the person's moment reached the planner

    def test_loose_words_name_the_target_and_a_bare_half_is_not_a_budget(self):
        self.assertEqual(team._column("the churned column", ["churned", "plan"]), "churned")
        self.assertEqual(team._column("Churned", ["churned", "plan"]), "churned")
        self.assertIsNone(team._column("the outcome", ["churned", "plan"]))
        self.assertEqual(team._share("0.5%", 1000, None), 0.005)
        self.assertEqual(team._share(".5%", 1000, None), 0.005)
        self.assertEqual(team._share("0.005", 1000, None), 0.005)
        self.assertIsNone(team._share("0.5", 1000, None))  # half the rows? ambiguous: asked again
        self.assertAlmostEqual(team._share("70 a day", 3000, 30.0), 0.7)

    def test_a_declined_gate_stops_the_run_there(self):
        from dclab_rnd import experience

        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        try:
            store, pid, case, out, _ = self.run_team("binary", approve={"solution": True, "holdout": False})
        finally:
            experience.install(None)
        self.assertEqual((out["status"], out["at"]), ("stopped", "holdout gate"))
        ended = [e for e in log.read() if e.kind == "run.recorded" and e.payload["name"] == "team"]
        self.assertEqual(ended[-1].payload["status"], "stopped")  # 21.1: the harness run says so, not "completed"
        self.assertTrue(store.has_stage(pid, "models"))
        self.assertFalse(store.has_stage(pid, "final"))  # the holdout stays sealed
        store, pid, case, out, _ = self.run_team("binary", approve=False)
        self.assertEqual((out["status"], out["at"]), ("stopped", "solution gate"))
        self.assertFalse(store.has_stage(pid, "data"))

    def test_a_private_project_asks_no_role_anything_even_with_the_critic_on(self):
        from dclab_rnd.expansion import runner

        store, pid, case = harness._project("binary")
        p = store.get(pid)
        p["settings"].update(private=True, critic=True)
        store.save(p)
        adv = advisor()
        with runner.fast_profile():
            out = team.run(store, pid, harness.DESCRIPTIONS["binary"], person(case), adv, quick=True)
            with engine.advisor_scope(adv):
                engine.execute(store, pid, "models", actor="agent")  # the critic's own path refuses a private project too
        self.assertEqual(out["status"], "completed")
        self.assertEqual(adv.requests, [])

    def test_the_cap_is_read_again_before_every_request(self):
        from dclab_rnd.core_advisor import GatewayAdvisor

        class Client:
            def __init__(self):
                self.spent_eur, self.run_limit_eur = 0.0, None

            def complete(self, messages, tools=None, **kw):
                self.spent_eur += 0.004
                return {"content": "{}"}

        class Gate:
            def client(self, purpose, **kw):
                return Client()

        adv = GatewayAdvisor(Gate(), cap_eur=0.01)
        adv.ask("problem_analyst", {"x": 1})
        adv.ask("reporter", {"x": 1})
        adv.ask("problem_analyst", {"x": 1})
        self.assertAlmostEqual(adv._clients["problem_analyst"].run_limit_eur, 0.004 + (0.01 - 0.008))  # what is left of the shared cap
        self.assertEqual(adv.calls, {"problem_analyst": 2, "reporter": 1})


if __name__ == "__main__":
    unittest.main()
