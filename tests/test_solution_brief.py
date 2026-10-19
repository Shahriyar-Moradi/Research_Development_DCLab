"""Package 30.14: the Solution Brief and G3.

A brief by rules has a bar, a baseline and an alternative; G3 refuses one without them; a person signs its fingerprint
with the solution gate and an edit unsigns it; the one holdout read is judged against the frozen bar (met, not met, or
cannot be judged with why) and the report says the verdict; a signed brief covers the holdout approval (D3), and after
the holdout was read the brief is closed.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    import pydantic  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def brief(**changes):
    from dclab_core.solution_brief import fill

    made = fill("binary", "average_precision", True, ["rare_event@1"], "time", baseline_today="two analysts read the flagged list")
    return made.model_copy(update=changes) if changes else made


@unittest.skipIf(MISSING, MISSING)
class CoreTests(unittest.TestCase):
    def test_a_brief_by_rules_can_be_signed_and_g3_names_what_is_missing(self):
        from dclab_core.solution_brief import SolutionBrief, SuccessBar, problems

        self.assertEqual(problems(brief()), [])
        self.assertEqual([b.name for b in brief().baselines], ["today", "majority_class"])
        self.assertEqual(SolutionBrief.read(brief().model_dump(mode="json")), brief())  # the contract reads its own dump
        cases = {"no success bar": brief(success=None), "no baseline": brief(baselines=[]), "no alternative": brief(alternatives=[]),
                 "has no target": brief(success=SuccessBar(metric="average_precision")),
                 "not one of the brief's baselines": brief(success=SuccessBar(metric="average_precision", relative_to="last_year", factor=1.2))}
        for said, case in cases.items():
            self.assertTrue(any(said in p for p in problems(case)), (said, problems(case)))

    def test_a_signature_covers_the_brief_and_an_edit_unsigns_it(self):
        from dclab_core.solution_brief import sign, signed

        with self.assertRaisesRegex(ValueError, "no alternative"):
            sign(brief(alternatives=[]), by="owner", at="2026-10-19T09:00:00")
        done = sign(brief(), by="owner", at="2026-10-19T09:00:00")
        self.assertTrue(signed(done))
        self.assertFalse(signed(brief()))
        self.assertFalse(signed(done.model_copy(update={"decision_rule": "every case above 0.1"})))

    def test_the_holdout_is_judged_against_the_frozen_bar(self):
        from dclab_core.solution_brief import Guard, SuccessBar, judge, sign

        signed_brief = sign(brief(), by="owner", at="2026-10-19T09:00:00")
        baselines = {"majority_class": {"accuracy": 0.97, "roc_auc": 0.5, "average_precision_random_ranking": 0.03}}
        met = judge(signed_brief, {"average_precision": 0.21}, baselines)  # the random ranking's rate, under the engine's name
        self.assertEqual((met.verdict, met.baseline_value, met.target), ("met", 0.03, 0.03))
        self.assertEqual(judge(signed_brief, {"average_precision": 0.02}, baselines).verdict, "not_met")
        self.assertIn("not signed", judge(brief(), {"average_precision": 0.21}, baselines).reason)
        self.assertIn("has no average_precision", judge(signed_brief, {"roc_auc": 0.8}, baselines).reason)
        self.assertEqual(judge(signed_brief, {"average_precision": 0.21}, {}).verdict, "cannot_judge")
        guarded = sign(brief(success=SuccessBar(metric="average_precision", op=">", value=0.1,
                                                guards=[Guard(metric="roc_auc", op=">=", value=0.7)])), by="owner", at="2026-10-19T09:00:00")
        verdict = judge(guarded, {"average_precision": 0.3, "roc_auc": 0.6}, baselines)
        self.assertEqual(verdict.verdict, "not_met")
        self.assertIn("roc_auc", verdict.reason)  # which guard failed is said, not only that the bar was missed
        from dclab_core.solution_brief import fill

        regression = fill("regression", "mae", False, [], "random")
        lower = sign(regression.model_copy(update={"success": SuccessBar(metric="mae", op="<", relative_to="training_mean", factor=0.8)}),
                     by="owner", at="2026-10-19T09:00:00")
        self.assertEqual(judge(lower, {"mae": 7.0}, {"training_mean": {"mae": 10.0}}).verdict, "met")
        self.assertEqual(judge(lower, {"mae": 9.0}, {"training_mean": {"mae": 10.0}}).verdict, "not_met")
        self.assertEqual(judge(signed_brief, {"average_precision": float("nan")}, baselines).verdict, "cannot_judge")  # never a crash

    def test_g3_refuses_bars_that_could_never_be_judged_or_go_the_wrong_way(self):
        from dclab_core.solution_brief import Baseline, Guard, SuccessBar, fill, judge, problems, sign

        regression = fill("regression", "mae", False, [], "random")
        cases = {
            "lower is better": regression.model_copy(update={"success": SuccessBar(metric="mae", op=">", relative_to="training_mean", margin=0.0)}),
            "reports": brief(success=SuccessBar(metric="mae", op="<", relative_to="majority_class", factor=0.8)),
            "no number is given": brief(success=SuccessBar(metric="average_precision", op=">", relative_to="today", factor=1.1)),
            "the guard on rmse has no target": regression.model_copy(update={"success": SuccessBar(metric="mae", op="<", value=5.0, guards=[Guard(metric="rmse", op="<=")])}),
            "the guard on rmse is measured against nowhere": regression.model_copy(update={"success": SuccessBar(
                metric="mae", op="<", value=5.0, guards=[Guard(metric="rmse", op="<=", relative_to="nowhere")])}),
            "the guard on rmse says >=": regression.model_copy(update={"success": SuccessBar(
                metric="mae", op="<", value=5.0, guards=[Guard(metric="rmse", relative_to="training_mean")])}),
        }
        for said, case in cases.items():
            self.assertTrue(any(w in p for p in problems(case) for w in [said.split(" is measured")[0], said]), (said, problems(case)))
        today = brief(baselines=[Baseline(name="today", kind="today", description="two analysts", value=0.1), brief().baselines[1]],
                      success=SuccessBar(metric="average_precision", op=">", relative_to="today", factor=1.5))
        self.assertEqual(problems(today), [])  # today's number given: a bar against it can be judged
        self.assertEqual(judge(sign(today, by="owner", at="2026-10-19T09:00:00"), {"average_precision": 0.2}, {}).target, 0.15000000000000002)
        budget = fill("binary", "recall_at_budget", True, [], "time", budget=0.02)
        self.assertEqual((budget.success.value, budget.success.relative_to), (0.02, None))  # reviewing 2% at random finds 2% of the cases
        self.assertEqual(judge(sign(budget, by="owner", at="2026-10-19T09:00:00"), {"recall_at_budget": 0.3}, {}).verdict, "met")
        none = fill("segmentation", "silhouette", True, ["segmentation"], "random", simple="")
        self.assertIsNone(none.success)  # a kind with no simple model: a person writes the bar (G3 asks)
        self.assertTrue(any("no success bar" in p for p in problems(none)))


@unittest.skipIf(MISSING, MISSING)
class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.replicas import ticket_routing
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd
        from dclab_rnd.studio.solution import Solution

        cls.runner = runner
        cls.store = store = ProjectStore(Path(tempfile.mkdtemp()))
        tickets = ticket_routing.generate(seed=0, n=1200).tickets.drop(columns=["resolved_by", "first_response_minutes"])
        project = store.create("Ticket routing", "general", "Route each support ticket to the right team when it is created.")
        path = store.data_dir(project["id"]) / "data.csv"
        tickets.to_csv(path, index=False)
        loaded = sd.load_table(path)
        project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path),
                           "profile": sd.profile_table(loaded)}
        project["solution"] = Solution(target="assigned_team", task="multiclass", identifiers=["ticket_id"], time_column="created_at",
                                       prediction_moment="When the ticket is created, from its subject, body and plan.").model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        cls.project_id = project["id"]

    def test_a_brief_is_drafted_signed_with_the_solution_judged_once_and_closed_after(self):
        from dclab_rnd.studio import brief as studio_brief, engine, graph, solution_report
        from dclab_core import report as rp

        store, pid = self.store, self.project_id
        with self.runner.fast_profile():
            view = studio_brief.view(store, store.get(pid))
            self.assertFalse(view["stored"])
            self.assertEqual(view["problems"], [])
            self.assertEqual(view["brief"]["baselines"][0]["name"], "majority_class")
            incomplete = {**view["brief"], "alternatives": []}
            saved = studio_brief.save(store, pid, incomplete, "owner")
            self.assertTrue(saved["stored"] and saved["problems"])
            with self.assertRaises(graph.GraphBlocked) as refused:
                graph.approve_gate(store, pid, "solution")
            self.assertIn("Brief complete", [c["name"] for c in refused.exception.verdict.checks if not c["ok"]])
            studio_brief.save(store, pid, view["brief"], "owner")
            graph.approve_gate(store, pid, "solution")
            project = store.get(pid)
            self.assertTrue(graph.brief_signed(project))
            self.assertEqual(project["signoff"]["brief_hash"], project["brief"]["approval"]["fingerprint"])
            forged = {**project["brief"], "approval": {**project["brief"]["approval"], "fingerprint": "0" * 16}}
            self.assertFalse(studio_brief.save(store, pid, forged, "owner")["signed"])  # a signature is the gate's, never written in an edit
            graph.approve_gate(store, pid, "solution")
            project = store.get(pid)
            project["policy"] = {**(project.get("policy") or {}), "require_holdout_approval": True}
            for stage in ("data", "leakage", "features", "models"):
                project.setdefault("stages", {})[stage] = {"status": "approved"}
            verdict = graph.check(project, "run_stage", "agent", stage="final")
            self.assertTrue(next(c for c in verdict.checks if c["name"] == "Owner approval")["ok"])  # D3: the signed brief covers the read
            owner = lambda p: next(c for c in graph.check(p, "run_stage", "agent", stage="final").checks if c["name"] == "Owner approval")["ok"]  # noqa: E731
            self.assertFalse(owner({**project, "brief": {**project["brief"], "decision_rule": "changed after signing"}}))
            self.assertFalse(owner({**project, "solution": {**project["solution"], "prediction_moment": "Changed after signing."}}))  # D3 needs both
            running = {**project, "running": "final"}
            self.assertFalse(graph.check(running, "set_brief", "human").allowed)  # the final reads the brief: never changed under it
            self.assertFalse(graph.check(project, "set_brief", "agent").allowed)  # the owner's: an agent may draft, never save
            engine.run_all(store, pid)
        project = store.get(pid)
        bar = studio_brief.verdict_of(project, store)
        self.assertIn(bar["verdict"], ("met", "not_met"), bar)
        self.assertEqual(bar["brief"], project["brief"]["approval"]["fingerprint"])
        self.assertEqual(bar["baseline"], "majority_class")
        inputs = solution_report.inputs(store, pid, None)
        self.assertIn("success bar signed in the Solution Brief", rp.to_markdown(rp.build(**inputs)))
        with self.assertRaisesRegex(ValueError, "new holdout"):
            studio_brief.save(store, pid, project["brief"], "owner")
        self.assertTrue(studio_brief.view(store, project)["holdout_read"])
        moved = {**project["brief"], "approval": None, "success": {**project["brief"]["success"], "margin": -0.5}}
        project["brief"] = moved  # a bar written in after the read (as a race with the final could leave it)
        store.save(project)
        graph.approve_gate(store, pid, "solution")
        self.assertFalse(graph.brief_signed(store.get(pid)))  # never signed after the holdout was read

    def test_a_kind_with_its_own_machine_is_drafted_and_judged_against_its_own_baseline(self):
        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio import brief as studio_brief, engine, graph

        case = kinds.get("anomaly").case(0)
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        view = studio_brief.view(store, store.get(pid))
        self.assertEqual((view["brief"]["success"]["relative_to"], view["problems"]), ("rules", []))
        studio_brief.save(store, pid, view["brief"], "owner")
        graph.approve_gate(store, pid, "solution")
        with self.runner.fast_profile():
            engine.run_all(store, pid)
        bar = studio_brief.verdict_of(store.get(pid), store)
        self.assertIn(bar["verdict"], ("met", "not_met"), bar)  # the machine's holdout against its rules' baseline
        self.assertEqual(bar["baseline"], "rules")


@unittest.skipIf(MISSING, MISSING)
class RouteTests(unittest.TestCase):
    def test_the_routes_draft_save_and_refuse_a_malformed_brief(self):
        import os

        from unittest import mock

        from fastapi.testclient import TestClient

        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            from dclab_rnd.agentic.server import create_app

            with TestClient(create_app(Path(tempfile.mkdtemp()))) as client:
                client.headers["X-DCLab-Token"] = client.get("/api/config").json()["csrf"]
                self.routes(client)

    def routes(self, client):
        created = client.post("/api/projects", json={"name": "Empty", "pack": "general", "goal": "Predict churn next month"})
        self.assertEqual(created.status_code, 201, created.text)
        pid = created.json()["id"]
        self.assertEqual(client.get(f"/api/projects/{pid}/brief").status_code, 409)  # nothing to draft from: no table, no solution
        body = brief().model_dump(mode="json")
        saved = client.put(f"/api/projects/{pid}/brief", json=body)
        services = client.app.state.services
        rows = services.audit.query(100, 0, kind="brief")[1]
        self.assertEqual([(r["move"], r["detail"]["args"]["fingerprint"]) for r in rows], [("set_brief", saved.json()["fingerprint"])])
        events = [e for e in services.experience.read() if e.kind == "decision.accepted" and e.payload.get("point") == "solution.brief"]
        self.assertEqual([e.payload.get("choice") for e in events], [saved.json()["fingerprint"]])  # its capture, read back
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual((saved.json()["stored"], saved.json()["problems"]), (True, []))
        self.assertEqual(client.get(f"/api/projects/{pid}/brief").json()["fingerprint"], saved.json()["fingerprint"])
        self.assertEqual(client.put(f"/api/projects/{pid}/brief", json={**body, "approach_level": "magic"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
