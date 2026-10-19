"""Package 30.15 (part A): the Workflow Run and G4.

A calibrator chosen and fitted on out-of-fold scores lowers the Brier score of miscalibrated scores and leaves good ones
alone; the monitoring plan keeps the training rows' bins and ``monitor`` flags a planted drift (a column, a category, the
flagged share) and nothing on rows like the training ones; G4 sends a failed run back to the right phase; and a project
run end to end records the twelve steps, its calibrator, its monitoring plan and G4.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
    import pydantic  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class CalibrationTests(unittest.TestCase):
    def test_miscalibrated_scores_are_calibrated_on_their_folds_and_good_ones_are_left_alone(self):
        from dclab_core import calibration

        rng = np.random.default_rng(0)
        truth = rng.uniform(0.02, 0.6, 4000)
        y = (rng.uniform(size=4000) < truth).astype(int)
        squeezed = truth ** 0.3  # every score far too high
        fitted = calibration.fit(squeezed[:3000], y[:3000])
        self.assertIn(fitted.method, ("sigmoid", "isotonic"))
        before, after = calibration.brier(y[3000:], squeezed[3000:]), calibration.brier(y[3000:], calibration.apply(fitted, squeezed[3000:]))
        self.assertLess(after, before)  # on rows it never saw
        honest = calibration.fit(truth[:3000], y[:3000])
        self.assertLessEqual(calibration.brier(y[3000:], calibration.apply(honest, truth[3000:])), calibration.brier(y[3000:], truth[3000:]) + 0.002)
        self.assertEqual(calibration.fit(truth[:50], y[:50]).method, "none")  # too few rows: said, never fitted
        self.assertIn("not probabilities", calibration.fit(truth * 3, y).reason)
        self.assertEqual(calibration.Calibrator.read(fitted.model_dump(mode="json")), fitted)


@unittest.skipIf(MISSING, MISSING)
class MonitoringTests(unittest.TestCase):
    def frame(self, n, shift=0.0, mix=(0.5, 0.3, 0.2), seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"amount": rng.normal(100 + shift, 15, n), "channel": rng.choice(["web", "store", "phone"], n, p=list(mix)),
                             "name": [f"customer {i}" for i in range(n)]})

    def test_a_planted_drift_is_flagged_and_rows_like_the_training_ones_are_not(self):
        from dclab_core import monitoring

        train = self.frame(5000)
        scores = np.random.default_rng(1).uniform(size=5000)
        plan = monitoring.plan(train, scores=scores, threshold=0.9, rule="the top 10%", personal=["name"])
        self.assertEqual([c.column for c in plan.columns], ["amount", "channel"])
        self.assertTrue(any("name: personal" in w for w in plan.left_out))  # never monitored by value
        self.assertAlmostEqual(plan.flagged_share, 0.1, delta=0.02)
        same = monitoring.monitor(plan, self.frame(1500, seed=2), np.random.default_rng(3).uniform(size=1500))
        self.assertEqual((same.status, same.alerts), ("stable", []))
        moved = monitoring.monitor(plan, self.frame(1500, shift=20, seed=4), np.random.default_rng(5).uniform(size=1500))
        self.assertEqual(moved.status, "alert")
        self.assertEqual([r.column for r in moved.columns if r.status == "alert"], ["amount"])
        mixed = monitoring.monitor(plan, self.frame(1500, mix=(0.1, 0.3, 0.6), seed=6), np.random.default_rng(7).uniform(size=1500))
        self.assertEqual([r.column for r in mixed.columns if r.status == "alert"], ["channel"])
        flagged = monitoring.monitor(plan, self.frame(1500, seed=8), np.random.default_rng(9).uniform(0.5, 1.0, 1500))
        self.assertEqual(flagged.flagged_status, "above")  # the rule flags far more than it did: the score's world changed
        self.assertTrue(any("flags" in a for a in flagged.alerts))
        few = monitoring.monitor(plan, self.frame(50, shift=40, seed=10))
        self.assertEqual((few.status, few.alerts), ("too_few_rows", []))  # read and said, never alerted on
        gone = monitoring.monitor(plan, self.frame(1500, seed=11).drop(columns=["channel"]))
        self.assertTrue(any("channel is missing" in w for w in gone.warnings))
        self.assertEqual(monitoring.MonitoringPlan.read(plan.model_dump(mode="json")), plan)


@unittest.skipIf(MISSING, MISSING)
class GateTests(unittest.TestCase):
    def test_g4_passes_an_honest_usable_run_and_sends_a_failed_one_back_to_its_phase(self):
        from dclab_core.workflow_run import STEPS, GateResult, RunStep, WorkflowRun, g4

        good = dict(holdout_reads=1, metric="average_precision", holdout=0.42, folds=0.45, interval=(0.35, 0.5), rule="the top 80 a day",
                    monitoring=True, bar={"verdict": "met"})
        self.assertTrue(g4(**good).passed)
        cases = {"workflow": {**good, "holdout_reads": 2}, "data": {**good, "holdout": 0.999, "folds": 0.999},
                 "brief": {**good, "bar": {"verdict": "not_met", "reason": "below the simple model"}}}
        for phase, facts in cases.items():
            gate = g4(**facts)
            self.assertEqual((gate.passed, gate.back_to), (False, phase), facts)
        near_zero = g4(**{**good, "folds": 0.09, "holdout": 0.01, "interval": None})
        self.assertEqual(near_zero.back_to, "workflow")  # the review: a score near zero falling to nothing is not explained
        self.assertTrue(g4(**{**good, "folds": 0.48, "holdout": 0.40, "interval": None, "folds_std": 0.05}).passed)  # within the folds' spread
        self.assertNotIn("interval", g4(**{**good, "folds": 0.8, "holdout": 0.4, "interval": None}).reasons[0])  # no interval: none said
        self.assertNotEqual(g4(**{**good, "holdout": float("nan")}).back_to, "data")  # unread, never "too good"
        gap = g4(**{**good, "folds": 0.80, "interval": (0.35, 0.5)})
        self.assertEqual(gap.back_to, "workflow")  # the folds' score far above the holdout's interval: the split or the rows changed
        both = g4(**{**good, "holdout": 0.999, "folds": 0.999, "bar": {"verdict": "not_met"}})
        self.assertEqual(both.back_to, "data")  # the earliest phase a failure names
        self.assertFalse(g4(**{**good, "monitoring": False}).passed)
        self.assertTrue(g4(**{**good, "metric": "mae", "holdout": 7.0, "folds": 6.5, "higher_is_better": False, "interval": None}).passed)
        self.assertEqual(g4(**{**good, "metric": "mae", "holdout": 9.0, "folds": 6.5, "higher_is_better": False, "interval": None}).back_to,
                         "workflow")  # an error a third above the folds' own: not explained
        with self.assertRaises(pydantic.ValidationError):
            GateResult(gate="G4", passed=False)  # a failed gate names where the work goes back
        steps = [RunStep(id=s, status="done", check="c") for s in STEPS]
        self.assertEqual(len(WorkflowRun(steps=steps, gate=g4(**good)).steps), 12)
        with self.assertRaises(pydantic.ValidationError):
            WorkflowRun(steps=list(reversed(steps)))  # the framework's order, each step once


@unittest.skipIf(MISSING, MISSING)
class StudioTests(unittest.TestCase):
    def test_a_project_run_end_to_end_records_its_twelve_steps_calibrator_monitoring_plan_and_g4(self):
        from dclab_core import monitoring
        from dclab_core.replicas import delivery_fraud as dfr
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine, workflow_run
        from dclab_rnd.studio.solution import Solution

        from dclab_core.ledger import check_state
        from dclab_rnd import ledger
        from dclab_rnd.studio import service

        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        ledger.install(book)
        self.addCleanup(ledger.install, None)
        store = ProjectStore(Path(tempfile.mkdtemp()))
        replica = dfr.generate(seed=0)
        project = store.create("Rides", "delivery", "Find fraudulent activity of bikers and customers")
        paths = dfr.save(replica, Path(tempfile.mkdtemp()))
        directory = store.data_dir(project["id"])
        (directory / dfr.RIDE_FILE).write_bytes(paths["rides"].read_bytes())
        sd.attach_data(store, project["id"], dfr.RIDE_FILE)
        (directory / dfr.SAMPLE_FILE).write_bytes(paths["sample"].read_bytes())
        project = sd.attach_labels(store, project["id"], dfr.SAMPLE_FILE, target="is_fraud")
        project["solution"] = Solution(**{**project["suggestion"], "group_column": "biker_id", "alert_budget": 0.005,
                                          "prediction_moment": "After delivery, in a daily audit of the day's rides."}).model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile():
            engine.run_all(store, project["id"])
        project = store.get(project["id"])
        final = store.read_stage(project["id"], "final")["evidence"]
        self.assertIn(final["calibration_fit"]["calibrator"]["method"], ("none", "sigmoid", "isotonic"))
        self.assertIn("out-of-fold", final["calibration_fit"]["calibrator"]["chosen_on"])
        plan = monitoring.MonitoringPlan.read(final["monitoring"])
        self.assertTrue(plan.columns and plan.score is not None and plan.threshold is not None)
        models = store.read_stage(project["id"], "models")["evidence"]
        self.assertEqual(models["dummy_cv"]["model"], "majority_class")  # the simple model on the screen's own folds
        gate = final["gate_g4"]
        self.assertEqual(gate["gate"], "G4")
        self.assertEqual(gate["passed"], gate["back_to"] is None)
        run = workflow_run.build(store, project)
        status = {s.id: s.status for s in run.steps}
        for sid in ("lock_split", "baseline", "candidates", "evaluate", "select", "calibrate_rule", "holdout_once", "report", "plan_monitoring"):
            self.assertEqual(status[sid], "done", sid)
        self.assertEqual(status["package"], "missing")  # not exported yet: never assumed done
        self.assertEqual(run.gate.gate, "G4")
        recorded = [r for r in book.records() if r.point == "workflow.g4"]
        self.assertEqual([r.answer("rule").value for r in recorded], ["passed" if gate["passed"] else gate["back_to"]])
        self.assertEqual(check_state(dict(recorded[0].state)), [])  # bands and codes, never a score or a row
        engine.approve(store, project["id"], "final")
        out = Path(tempfile.mkdtemp()) / "svc"
        with runner.fast_profile():
            described = service.export(store, project["id"], out)
        self.assertTrue(described["checked"]["checked"])  # saved, reloaded in an isolated Python, predicted, matched
        self.assertTrue((out / "monitoring.json").is_file())
        shipped = __import__("json").loads((out / "monitoring.json").read_text())
        self.assertTrue(shipped.pop("hashed_levels"))
        shipped = monitoring.MonitoringPlan.read(shipped)
        for kept, sent in zip(plan.columns, shipped.columns):
            self.assertEqual((kept.column, kept.edges, kept.shares), (sent.column, sent.edges, sent.shares))
            if kept.kind == "category":  # every level under the export key's hash, as the model's own levels are
                self.assertTrue(all(c.startswith("h:") or c in (monitoring.OTHER, monitoring.MISSING) for c in sent.categories), sent.categories)
        self.assertEqual({s.id: s.status for s in workflow_run.build(store, store.get(project["id"])).steps}["package"], "done")
        blob = b"".join(f.read_bytes() for f in out.rglob("*") if f.is_file() and "vendor" not in f.parts)
        for column in ("order_id", "biker_id", "sender_phone", "status"):  # the review: no value of the table, every one checked
            for value in set(replica.rides[column].dropna().astype(str)):
                self.assertNotIn(f'"{value}"'.encode(), blob, (column, value))
        p = engine.prepare(store, store.get(project["id"]))
        watched = {c.column for c in plan.columns}
        self.assertFalse(watched & {"Unnamed: 0", "order_id", "biker_id", "created_date"})  # the review: no identifier, date or group column
        self.assertTrue(watched <= {c for c, role in p.roles.items() if role == "feature"})  # only what the model reads
        report = monitoring.monitor(plan, p.bundle.X_test, None)
        self.assertEqual([r.column for r in report.columns if r.status == "alert"], ["status"])  # a real shift: CANCELLED only in the holdout

    def test_g4_passes_on_the_regression_replica_and_each_service_kind_reloads_and_matches(self):
        from dclab_rnd import kinds
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio import engine, service

        for key in ("regression", "binary", "multiclass"):
            with self.subTest(key):
                store, pid = acceptance.build(kinds.get(key).case(0), Path(tempfile.mkdtemp()))
                with runner.fast_profile():
                    engine.run_all(store, pid)
                final = store.read_stage(pid, "final")["evidence"]
                rule = next(c for c in final["gate_g4"]["checks"] if c["name"] == "A rule to act on")
                self.assertTrue(rule["ok"], final["gate_g4"])  # the review: a regression's task type is timeseries_regression
                if key == "regression":
                    self.assertTrue(final["gate_g4"]["passed"], final["gate_g4"])
                engine.approve(store, pid, "final")
                with runner.fast_profile():
                    done = service.export(store, pid, Path(tempfile.mkdtemp()) / "svc")
                self.assertTrue(done["checked"]["checked"])  # saved, reloaded in an isolated Python, the refit's rows predicted, matched

    def test_the_team_follows_a_failed_g4_to_the_phase_it_names_through_its_run(self):
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.solution import SolutionTeam
        from dclab_rnd.studio import ProjectStore

        home = Path(tempfile.mkdtemp())
        store = ProjectStore(home / "projects")
        project = store.create("Churn", "general", "Predict churn")
        project["solution"] = {"target": "y", "task": "binary", "prediction_moment": "Before the call."}
        project["stages"] = {k: {"status": "completed"} for k in ("data", "leakage", "features", "models", "final")}
        store.save(project)
        store.write_stage(project["id"], "final", {"stage": "final", "status": "completed", "summary": "read once",
                                                   "evidence": {"gate_g4": {"gate": "G4", "passed": False, "back_to": "brief",
                                                                            "reasons": ["The success bar: below the simple model"]}}})
        with checkpoints.open_checkpointer(home, database_url="") as saver:
            out = SolutionTeam(store, saver, home=home).run(project["id"])  # the review: LangGraph dropped a key its state did not name
        self.assertEqual((out["next"], out.get("back_to")), ("stop", "brief"))
        self.assertIn("back to the brief", out["says"])

if __name__ == "__main__":
    unittest.main()
