"""23.5: an outcome is what followed a decision — the final stage's holdout, a promotion, a business number — judged
against its reference; every decision of the project gets its row, the Decisions tab counts them per point, the
decision sets carry them, and deleting the project removes them. An outcome is never a label."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import ledger as L, outcomes as oc
    from dclab_rnd import context, experience, ledger
    from dclab_rnd.ledger import outcomes
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class ContractTest(unittest.TestCase):
    def test_the_verdict_holds_the_reference_within_a_tolerance(self):
        self.assertEqual(oc.judge(0.84, 0.85, True), "well")       # a holdout within 0.02 of its CV estimate
        self.assertEqual(oc.judge(0.80, 0.85, True), "badly")
        self.assertEqual(oc.judge(12.6, 12.0, False), "well")      # a lower-is-better metric within 5% of its reference
        self.assertEqual(oc.judge(13.0, 12.0, False), "badly")
        self.assertEqual(oc.judge(0.9, None, True), "unknown")
        self.assertEqual(oc.judge(0.9, 0.8, None), "unknown")
        self.assertEqual(oc.judge("x", 0.8, True), "unknown")

    def test_an_outcome_round_trips_and_the_newest_per_record_is_current(self):
        a = oc.Outcome(id="do000001", record="dr0001", point="split.strategy", metric="roc_auc", value=0.84, interval=(0.80, 0.88), reference=0.85,
                       higher_is_better=True, observed="holdout", verdict="well", by="final_stage", at="2026-10-15T10:00:00+00:00")
        self.assertEqual(oc.Outcome.read(a.model_dump_json()), a)
        b = a.model_copy(update={"id": "do000002", "observed": "business", "verdict": "badly", "at": "2026-10-16T10:00:00+00:00"})
        records = [L.DecisionRecord(id="dr0001", point="split.strategy", subject=L.Subject(kind="project", key="p"), at=a.at, used="rule",
                                    answers=[L.Answer(by="rule", answerer="r", value="time")], value_used="time")]
        self.assertEqual(oc.join(records, [a, b])["dr0001"].id, "do000002")
        self.assertEqual(oc.by_point(records, [a, b]), {"split.strategy": {"well": 0, "badly": 1, "unknown": 0, "none": 0}})
        self.assertEqual(oc.by_point(records, []), {"split.strategy": {"well": 0, "badly": 0, "unknown": 0, "none": 1}})
        with self.assertRaises(Exception):
            oc.Outcome(id="do000003", record="dr0001", point="p", metric="m", value=1.0, observed="guess", verdict="well", by="x", at=a.at)
        with self.assertRaises(Exception):  # a metric is a slug: a person's words never travel in it
            oc.Outcome(id="do000004", record="dr0001", point="p", metric="retention of ACME Corp", value=1.0, observed="business", verdict="well", by="x", at=a.at)

    def test_a_later_unjudged_number_or_a_promotion_never_erases_the_holdouts_verdict(self):
        base = dict(record="dr0001", point="split.strategy", metric="roc_auc", value=0.7, reference=0.85, higher_is_better=True, by="x")
        holdout = oc.Outcome(id="do000010", observed="holdout", verdict="badly", at="2026-10-15T10:00:00+00:00", **base)
        promotion = oc.Outcome(id="do000011", observed="promotion", verdict="badly", at="2026-10-15T11:00:00+00:00", **base)
        typed = oc.Outcome(id="do000012", observed="business", verdict="unknown", at="2026-10-16T10:00:00+00:00", **{**base, "reference": None})
        judged = oc.Outcome(id="do000013", observed="business", verdict="well", at="2026-10-14T10:00:00+00:00", **base)
        rec = [L.DecisionRecord(id="dr0001", point="split.strategy", subject=L.Subject(kind="project", key="p"), at=holdout.at, used="rule",
                                answers=[L.Answer(by="rule", answerer="r", value="time")], value_used="time")]
        self.assertEqual(oc.join(rec, [holdout, promotion, typed])["dr0001"].id, "do000011")  # the unjudged number is ignored; the newest judged stands
        self.assertEqual(oc.join(rec, [holdout, promotion, typed, judged])["dr0001"].id, "do000013")  # a judged business number comes first
        from dclab_core import datasets as builders

        docs = [o.model_dump(mode="json") for o in (holdout, promotion, typed)]
        self.assertEqual(builders._current_outcome(docs)["observed"], "promotion")


@unittest.skipIf(MISSING, MISSING)
class ProductTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.home = Path(tempfile.mkdtemp())
        cls.client = TestClient(create_app(cls.home))
        cls.client.__enter__()
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def project_with_decisions(self, name):
        pid = self.services.projects.create(name, "general", "Predict churn")["id"]
        with context.using(self.services):
            for point, value in (("split.strategy", "time"), ("metric.primary", "roc_auc"), ("workflow.stage_approval", True)):
                ledger.decide(point, ("project", pid), answers=[ledger.answer("rule", "rule:test", value)], value_used=value, used="rule", project=pid)
        return pid

    def test_the_final_stage_writes_an_outcome_of_every_decision_and_the_tab_counts_them(self):
        pid = self.project_with_decisions("Churn A")
        record = {"primary_metric": "roc_auc", "evidence": {"holdout_metrics": {"roc_auc": 0.84}, "holdout_primary_metric_ci": [0.80, 0.88],
                                                             "cv_selected_metric_mean": 0.85}}
        with context.using(self.services):
            out = outcomes.from_final(pid, record, "run-1", True)
        self.assertEqual((out["decisions"], out["verdict"]), (3, "well"))
        rows = self.services.ledger.outcomes(project=pid)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r.record for r in rows}, {r.id for r in self.services.ledger.records(project=pid)})
        self.assertEqual((rows[0].metric, rows[0].value, rows[0].interval, rows[0].reference, rows[0].observed, rows[0].run), ("roc_auc", 0.84, (0.8, 0.88), 0.85, "holdout", "run-1"))
        events = [e for e in self.services.experience.read(project=pid) if e.kind == "outcome.observed"]
        self.assertEqual((events[-1].payload["decisions"], events[-1].payload["verdict"], events[-1].payload["observed"]), (3, "well", "holdout"))
        summary = self.client.get("/api/ledger/summary").json()
        self.assertGreaterEqual(summary["outcomes"]["split.strategy"]["well"], 1)  # the workspace's count: other projects add theirs
        self.assertEqual(outcomes.by_point(self.services.ledger, project=pid)["split.strategy"], {"well": 1, "badly": 0, "unknown": 0, "none": 0})
        # a promotion is judged as the holdout was; a bad holdout is not turned good by being approved
        bad = {"primary_metric": "roc_auc", "evidence": {"holdout_metrics": {"roc_auc": 0.70}, "cv_selected_metric_mean": 0.85}, "run": {"run": "run-2"}}
        pid_b = self.project_with_decisions("Churn bad")
        with context.using(self.services):
            outcomes.from_final(pid_b, bad, "run-2", True)
            outcomes.from_approval(pid_b, bad, "human")
        self.assertEqual(self.services.ledger.outcomes(project=pid_b, point="metric.primary")[-1].observed, "promotion")
        self.assertEqual(self.services.ledger.outcomes(project=pid_b, point="metric.primary")[-1].run, "run-2")  # the stage record's run id
        self.assertEqual(outcomes.by_point(self.services.ledger, project=pid_b)["metric.primary"], {"well": 0, "badly": 1, "unknown": 0, "none": 0})
        with context.using(self.services):
            outcomes.from_approval(pid, {**record, "run": {"run": "run-1"}}, "human")
        self.assertEqual(self.services.ledger.outcomes(project=pid, point="metric.primary")[-1].observed, "promotion")
        self.assertEqual(self.services.ledger.outcomes(project=pid, point="metric.primary")[-1].verdict, "well")
        self.services.projects.save({**self.services.projects.get(pid), "stages": {**self.services.projects.get(pid)["stages"], "final": {"status": "approved"}}})
        typed = self.client.post(f"/api/projects/{pid}/outcome", json={"metric": "retained_customers", "value": 120, "reference": 150,
                                                                       "higher_is_better": True, "note": "after the March campaign"}, headers=self.h)
        self.assertEqual(typed.status_code, 200, typed.text)
        self.assertEqual((typed.json()["decisions"], typed.json()["verdict"]), (3, "badly"))
        current = oc.join(self.services.ledger.records(project=pid), self.services.ledger.outcomes(project=pid))
        self.assertEqual({o.verdict for o in current.values()}, {"badly"})  # the newest observation is the current one
        self.assertEqual(outcomes.by_point(self.services.ledger, project=pid)["split.strategy"], {"well": 0, "badly": 1, "unknown": 0, "none": 0})
        post = lambda body: self.client.post(f"/api/projects/{pid}/outcome", json=body, headers=self.h).status_code  # noqa: E731
        self.assertEqual(post({"metric": "x", "value": "many"}), 422)
        self.assertEqual(post({"value": 1}), 422)
        self.assertEqual(post({"metric": "x", "value": "nan"}), 422)
        self.assertEqual(post({"metric": "x", "value": 1, "reference": "inf"}), 422)
        self.assertEqual(post({"metric": "x", "value": 1, "reference": 2, "higher_is_better": "false"}), 422)  # a JSON boolean, not a string
        self.assertEqual(post({"metric": "retention of ACME Corp", "value": 1}), 422)  # a name, not a sentence
        fresh = self.services.projects.create("Not run yet", "general", "Predict churn")["id"]
        self.assertEqual(self.client.post(f"/api/projects/{fresh}/outcome", json={"metric": "x", "value": 1}, headers=self.h).status_code, 409)
        from dclab_rnd.accounts.guard import permission

        self.assertEqual(permission("POST", "/api/projects/{project_id}/outcome"), "write")  # owner and ML engineer
        before = len(self.services.ledger.outcomes(project=pid))
        self.assertEqual(post({"metric": "revenue", "value": 1200}), 200)  # no reference: recorded, unknown
        self.assertEqual(outcomes.by_point(self.services.ledger, project=pid)["split.strategy"], {"well": 0, "badly": 1, "unknown": 0, "none": 0})  # still the judged number

    def test_a_replay_writes_nothing_and_a_machines_direction_is_its_own(self):
        pid = self.project_with_decisions("Churn replay")
        record = {"primary_metric": "mae", "evidence": {"holdout_metrics": {"mae": 14.0}, "cv_selected_metric_mean": 12.0}}
        with context.using(self.services), experience.quiet():
            self.assertEqual(outcomes.from_final(pid, record, "scratch", False)["decisions"], 0)
        self.assertEqual(self.services.ledger.outcomes(project=pid), [])  # a replay on a scratch copy did not happen to the project
        self.assertEqual([e for e in self.services.experience.read(project=pid) if e.kind == "outcome.observed"], [])
        from dclab_rnd.studio import engine

        class Machine:
            higher_is_better = False

        class Prepared:  # what the engine prepares for a kind with its own machine: no bundle
            machine = Machine()

        seen = {}
        original = outcomes.from_final
        try:
            outcomes.from_final = lambda project_id, rec, run_id, higher: seen.update(higher=higher)  # noqa: E731
            engine._after_final(pid, record, "run-3", Prepared())
        finally:
            outcomes.from_final = original
        self.assertEqual(seen, {"higher": False})
        with context.using(self.services):
            out = outcomes.from_final(pid, record, "run-3", Machine.higher_is_better)
        self.assertEqual(out["verdict"], "badly")  # 14 against 12, lower is better, beyond 5%

    def test_the_decisions_of_the_drafts_that_became_the_project_get_its_outcome(self):
        pid = self.project_with_decisions("Churn draft")
        draft = self.services.drafts.create("Predict which customers churn next month")["id"]
        self.services.drafts.update(draft, lambda d: d.update(project_id=pid))
        with context.using(self.services):
            ledger.decide("problem.family", ("draft", draft), answers=[ledger.answer("rule", "rule:test", "classification")],
                          value_used="classification", used="rule", draft=draft)
            out = outcomes.observe(pid, metric="roc_auc", value=0.8, observed="holdout", by="final_stage", reference=0.82, higher_is_better=True)
        self.assertEqual(out["decisions"], 4)
        family = self.services.ledger.outcomes(point="problem.family", project=pid)
        self.assertEqual([(o.verdict, o.observed) for o in family], [("well", "holdout")])  # the draft's decision follows the project it became

    def test_the_decision_sets_carry_the_outcome_and_a_deleted_project_loses_its_rows(self):
        from dclab_core import datasets as builders

        from dclab_rnd import consent
        from dclab_rnd.experience import retention

        switches = {"improve": True, "train": True}
        pid = self.project_with_decisions("Churn B")  # written before any consent: stamped operate, never leaves
        with context.using(self.services):
            outcomes.observe(pid, metric="roc_auc", value=0.7, observed="holdout", by="final_stage", reference=0.8, higher_is_better=True)
        projects = {pid: self.services.projects.get(pid)}
        self.assertEqual(consent.export_decisions(self.services.ledger.records(project=pid), [], switches, projects,
                                                  outcomes=self.services.ledger.outcomes(project=pid)), [])
        consent.set_workspace_consent(self.home, "improve", True)
        consent.set_workspace_consent(self.home, "train", True)
        project = self.services.projects.get(pid)
        project["settings"] = {**(project.get("settings") or {}), "consent": {"improve": True, "train": True}}
        self.services.projects.save(project)
        consent.invalidate(pid)
        pid2 = self.services.projects.create("Churn C", "general", "Predict churn")["id"]  # consents first, then decides
        project = self.services.projects.get(pid2)
        project["settings"] = {**(project.get("settings") or {}), "consent": {"improve": True, "train": True}}
        self.services.projects.save(project)
        consent.invalidate(pid2)
        with context.using(self.services):
            for point, value in (("split.strategy", "time"), ("metric.primary", "roc_auc"), ("workflow.stage_approval", True)):
                ledger.decide(point, ("project", pid2), answers=[ledger.answer("rule", "rule:test", value)], value_used=value, used="rule", project=pid2)
            out = outcomes.observe(pid2, metric="roc_auc", value=0.7, observed="holdout", by="final_stage", reference=0.8, higher_is_better=True,
                                   note="my words about the number")
        projects = {pid2: self.services.projects.get(pid2)}
        docs = consent.export_decisions(self.services.ledger.records(project=pid2), [], switches, projects, outcomes=self.services.ledger.outcomes(project=pid2))
        self.assertEqual((out["decisions"], len(docs)), (3, 3))
        self.assertTrue(all(d["outcomes"] and d["outcomes"][-1]["verdict"] == "badly" and "note" not in d["outcomes"][-1] for d in docs))
        rows, _ = builders.decisions(docs, "split.strategy", workspace="w")
        self.assertEqual([r["outcome"]["verdict"] for r in rows], ["badly"])
        self.assertEqual([r["label"] for r in rows], [None])  # an outcome is never a label
        labelled, m2 = builders.outcome_labels(docs, workspace="w")
        self.assertEqual((len(labelled), m2.counts["verdicts"], m2.builder), (3, {"badly": 3}, "outcomes"))
        self.assertNotIn("my words", str(docs))
        gone = retention.delete_decisions(self.services.ledger, project=pid2)
        self.assertEqual((gone, self.services.ledger.outcomes(project=pid2)), (3, []))
        self.assertEqual(len(self.services.ledger.outcomes(project=pid)), 3)  # the other project's rows stand


@unittest.skipIf(MISSING, MISSING)
class MainProductTest(unittest.TestCase):
    def test_an_experiments_result_is_the_outcome_of_the_decisions_linked_to_it(self):
        from dclab_core.adapters import main_product_outcomes as mpo

        rule = dict(used="rule", answers=[L.Answer(by="rule", answerer="r", value="time")], value_used="time")
        decisions = [L.DecisionRecord(id="dr000a", point="split.strategy", subject=L.Subject(kind="project", key="p"), at="2026-10-15T10:00:00+00:00",
                                      experiment="exp-1", **rule),
                     L.DecisionRecord(id="dr000b", point="split.strategy", subject=L.Subject(kind="project", key="p"), at="2026-10-15T10:00:00+00:00",
                                      experiment="exp-2", **rule)]
        row = {"id": "exp-1", "workspace_id": "w", "project_id": "p", "result": {"roc_auc": 0.81, "n": 3}, "status": "COMPLETED", "ended_at": "2026-10-15T11:00:00+00:00"}
        out = mpo.from_experiment(row, decisions, "roc_auc", higher_is_better=True, reference=0.85)
        self.assertEqual([(o.record, o.verdict, o.run, o.observed) for o in out], [("dr000a", "badly", "exp-1", "holdout")])
        self.assertEqual(mpo.from_experiment({**row, "result": {"roc_auc": None}}, decisions, "roc_auc"), [])
        self.assertIn("model_versions", mpo.FIELDS_MISSING)


if __name__ == "__main__":
    unittest.main()
