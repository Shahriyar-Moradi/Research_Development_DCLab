"""23.8: the steps inside a stage. The core records them as a context (free when nothing collects), the engine's stages
collect them into the run record with a digest of what each produced, a replay reports the first step whose digest
differs, and a decision traces to the steps that carry the numbers it was made on."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import experience as x, lineage as l, steps as S
    from dclab_rnd import experience
    from dclab_rnd.studio import lineage
except ImportError as error:  # the agent environment may lack pandas
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class CoreTest(unittest.TestCase):
    def test_a_step_is_free_when_nothing_collects_and_recorded_when_something_does(self):
        self.assertFalse(S.active())
        with S.step("alone", "other") as s:
            s.out(value=1.0)  # no collector: nothing happens, nothing fails
        with S.collect() as log:
            self.assertTrue(S.active())
            with S.step("load", "load", params={"rows": 10, "bad": [1, 2], "ok": True}, shape_in=(10, 3)) as s:
                s.out(shape=(10, 3), produced=np.arange(30.0).reshape(10, 3))
            with S.step("fold 1", "fold", fold=1, shape_in=(8,)) as s:
                s.out(metric="roc_auc", value=0.81)
        self.assertFalse(S.active())
        first, second = log.steps
        self.assertEqual((first.index, first.params, first.rows_in, first.cols_out, len(first.digest)), (0, {"rows": 10, "ok": True}, 10, 3, 16))
        self.assertEqual((second.index, second.fold, second.metric, second.value), (1, 1, "roc_auc", 0.81))
        self.assertEqual(second.digest, S.digest_of(0.81))  # a metric step with no array is digested by its value

    def test_a_digest_is_the_same_for_the_same_numbers_and_not_for_others(self):
        a = np.linspace(0, 1, 50).reshape(10, 5)
        self.assertEqual(S.digest_of(a), S.digest_of(a.copy()))
        self.assertNotEqual(S.digest_of(a), S.digest_of(a + 1e-6))
        self.assertEqual(S.digest_of(a), S.digest_of(a + 1e-12))  # rounded to six decimals: a float's last bits are not a difference
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        self.assertEqual(S.digest_of(frame), S.digest_of(frame.copy()))
        self.assertNotEqual(S.digest_of(frame), S.digest_of(frame.iloc[:1]))  # its shape is part of it
        from scipy import sparse

        self.assertEqual(S.digest_of(sparse.csr_matrix(a)), S.digest_of(sparse.csr_matrix(a.copy())))

    def test_a_step_that_raises_is_kept_and_the_error_goes_on_and_a_callback_never_stops_a_run(self):
        seen = []

        def broken(step):
            seen.append(step.name)
            raise RuntimeError("a host's event failed")

        with S.collect(broken, broken) as log:
            with self.assertRaises(ValueError):
                with S.step("fits", "fit"):
                    raise ValueError("the fit failed")
            with S.step("after", "other"):
                pass
        self.assertEqual([s.name for s in log.steps], ["fits", "after"])
        self.assertEqual(seen, ["fits", "fits", "after", "after"])  # started and finished, both called, both survived

    def test_the_log_keeps_so_many_steps_and_counts_the_rest(self):
        with mock.patch.object(S, "MAX_STEPS", 5), S.collect() as log:
            for i in range(8):
                with S.step(f"s{i}", "other"):
                    pass
        self.assertEqual((len(log.steps), log.dropped), (5, 3))

    def test_compare_names_the_first_step_that_differs_by_name_fold_or_digest_and_ignores_time(self):
        def make(digests, seconds=0.1):
            with S.collect() as log:
                for i, d in enumerate(digests):
                    with S.step(f"s{i}", "fold", fold=i) as s:
                        s.digest = d
            return [st.model_copy(update={"seconds": seconds}) for st in log.steps]

        base = make(["a", "b", "c"])
        self.assertTrue(S.compare(base, make(["a", "b", "c"], seconds=9.9))["identical"])
        out = S.compare(base, make(["a", "X", "c"]))
        self.assertEqual((out["identical"], out["first_difference"]["position"], out["first_difference"]["replayed"]["digest"]), (False, 1, "X"))
        longer = S.compare(base, make(["a", "b", "c", "d"]))
        self.assertEqual((longer["first_difference"]["position"], longer["first_difference"]["recorded"]), (3, None))

    def test_a_v1_run_record_reads_as_v2_without_steps(self):
        record = l.RunRecord(run="run-1", kind="stage", name="data", status="completed", started="2026-10-16T10:00:00+00:00")
        self.assertEqual((record.schema_version, record.steps, record.steps_dropped), ("dclab_core.run_record.v2", [], 0))
        old = record.model_dump(mode="json")
        old["schema_version"] = "dclab_core.run_record.v1"
        old.pop("steps"), old.pop("steps_dropped")
        self.assertEqual(l.RunRecord.model_validate(old).schema_version, "dclab_core.run_record.v2")
        with_steps = l.RunRecord(run="run-2", kind="stage", name="models", status="completed", started="2026-10-16T10:00:00+00:00",
                                 steps=[S.Step(index=0, name="fit", kind="fit", digest="abc")])
        self.assertEqual(l.event_payload(with_steps)["steps"], 1)
        again = l.compare(with_steps, with_steps)
        self.assertEqual((again["same_steps"], again["steps_compared"]), (True, 1))
        self.assertIsNone(l.compare(record, with_steps)["same_steps"])  # one side recorded none: nothing to compare


@unittest.skipIf(MISSING, MISSING)
class StudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.replicas import delivery_time
        from dclab_rnd import ledger
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        cls.home = Path(tempfile.mkdtemp())
        cls.store = store = ProjectStore(cls.home / "projects")
        cls.log = experience.FileLog(cls.home / "experience", "wtest")
        cls.book = ledger.FileLedger(cls.home / "ledger", "wtest")
        cls.log.home = cls.book.home = cls.home
        experience.install(cls.log)
        ledger.install(cls.book)
        project = store.create("Delivery time", "general", "Predict delivery minutes when the customer orders.")
        path = store.data_dir(project["id"]) / "data.csv"
        delivery_time.generate(seed=0, n=600).orders.drop(columns=["delivered_at", "courier_rating"]).to_csv(path, index=False)
        loaded = sd.load_table(path)
        project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path),
                           "profile": sd.profile_table(loaded)}
        project["solution"] = Solution(target="actual_minutes", task="regression", identifiers=["order_id"],
                                       prediction_moment="When the customer places the order.").model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        cls.pid = project["id"]
        with runner.fast_profile():
            engine.run_all(store, cls.pid)

    @classmethod
    def tearDownClass(cls):
        from dclab_rnd import ledger

        experience.install(None)
        ledger.install(None)

    def test_every_stage_run_records_its_steps_and_the_models_stage_has_a_fold_for_every_fit(self):
        for stage in ("data", "leakage", "features", "models", "final"):
            with self.subTest(stage=stage):
                got = lineage.stage_steps(self.store, self.pid, stage)
                self.assertTrue(got["steps"], f"{stage} took no recorded step")
                self.assertEqual([s["index"] for s in got["steps"]], list(range(len(got["steps"]))))
        steps = lineage.stage_steps(self.store, self.pid, "models")["steps"]
        kinds = [s["kind"] for s in steps]
        self.assertEqual(kinds.count("fit"), kinds.count("predict"))
        self.assertGreaterEqual(kinds.count("fold"), 2)
        folds = [s for s in steps if s["kind"] == "metric" and s.get("fold")]
        self.assertTrue(folds and all(s["metric"] and s["value"] is not None for s in folds))
        self.assertTrue(all(s.get("digest") for s in steps if s["kind"] in ("transform", "predict")))
        final = lineage.stage_steps(self.store, self.pid, "final")["steps"]
        self.assertEqual([s["name"] for s in final if s["name"].startswith("holdout")], ["holdout: metrics", "holdout: bootstrap interval"])
        holdout = next(s for s in final if s["name"] == "holdout: metrics")
        self.assertEqual(holdout["value"], self.store.records(self.pid)["final"]["evidence"]["holdout_metrics"][holdout["metric"]])
        with self.assertRaises(KeyError):
            lineage.stage_steps(self.store, self.pid, "nowhere")

    def test_no_step_holds_a_value_of_the_table(self):
        import json

        blob = json.dumps([lineage.stage_steps(self.store, self.pid, s)["steps"] for s in ("data", "models", "final")])
        table = pd.read_csv(self.store.data_dir(self.pid) / "data.csv")
        for value in table["order_id"].astype(str).head(20):
            self.assertNotIn(f'"{value}"', blob)

    def test_the_run_lists_count_steps_and_the_events_say_each_one_began_and_ended(self):
        runs = lineage.runs(self.store, self.pid)
        self.assertGreater(runs["models"]["steps"], 5)
        events = [e for e in self.log.read(project=self.pid) if e.kind.startswith("pipeline.step_")]
        by_run = {}
        for e in events:
            by_run.setdefault(e.payload["run"], []).append(e.kind)
        models_run = self.store.records(self.pid)["models"]["run"]["run"]
        steps = lineage.stage_steps(self.store, self.pid, "models")["steps"]
        self.assertEqual(by_run[models_run].count("pipeline.step_finished"), runs["models"]["steps"])  # every step ends with an event
        self.assertEqual(by_run[models_run].count("pipeline.step_started"), sum(1 for s in steps if s["kind"] not in ("fit", "predict", "metric")))  # a unit of work is announced
        finished = [e for e in events if e.kind == "pipeline.step_finished" and e.payload["run"] == models_run]
        self.assertTrue(any(e.payload.get("digest") for e in finished))
        self.assertEqual({e.ids.run for e in events if e.payload["run"] == models_run}, {models_run})

    def test_a_replay_repeats_every_step_and_pins_a_change_to_the_first_step_that_differs(self):
        from dclab_rnd.expansion import runner

        with runner.fast_profile():
            out = lineage.replay(self.store, self.pid, "features")
        self.assertTrue(out["replayed"], out)
        self.assertEqual((out["same_steps"], out["first_step_difference"]), (True, None))
        self.assertGreater(out["steps_compared"], 3)
        original = l.RunRecord.model_validate(self.store.records(self.pid)["features"]["run"])
        moved = original.model_copy(update={"steps": [s.model_copy(update={"digest": "0" * 16}) if s.index == 2 else s for s in original.steps]})
        walked = l.compare(original, moved)
        self.assertEqual((walked["same_steps"], walked["first_step_difference"]["position"]), (False, 2))

    def test_a_real_change_in_a_metric_is_pinned_to_the_first_metric_step_that_differs(self):
        from dclab_rnd.expansion import runner

        real = runner.compute_metrics

        def nudged(*args, **kwargs):  # the code behind the metric changed: every fold scores a little higher
            out = real(*args, **kwargs)
            return {k: (v + 0.01 if isinstance(v, float) else v) for k, v in out.items()}

        with runner.fast_profile(), mock.patch.object(runner, "compute_metrics", nudged):
            out = lineage.replay(self.store, self.pid, "features")
        self.assertTrue(out["replayed"], out)
        self.assertFalse(out["same_steps"])
        first = out["first_step_difference"]
        self.assertEqual(first["recorded"]["kind"], "metric")  # the transforms and fits before it were the same
        self.assertNotEqual(first["recorded"]["digest"], first["replayed"]["digest"])
        self.assertGreater(first["position"], 2)

    def test_the_replay_command_fails_when_a_step_differs(self):
        import contextlib
        import io

        from dclab_rnd.experience import __main__ as cli

        out = {"replayed": True, "same_inputs": True, "same_value": True, "same_numbers": True, "same_steps": False}
        with mock.patch.object(lineage, "replay", return_value=out), mock.patch.object(cli, "_projects", return_value=self.store), \
                mock.patch.object(cli, "_workspace_log", contextlib.nullcontext), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["replay", self.pid, "features"]), 1)
            out["same_steps"] = True
            self.assertEqual(cli.main(["replay", self.pid, "features"]), 0)

    def test_the_steps_route_answers_for_a_stage_a_decision_and_refuses_what_it_cannot(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(self.home))
        client.__enter__()
        try:
            got = client.get(f"/api/projects/{self.pid}/steps", params={"stage": "models"})
            self.assertEqual((got.status_code, got.json()["stage"]), (200, "models"))
            self.assertTrue(got.json()["steps"])
            self.assertEqual(client.get(f"/api/projects/{self.pid}/steps").status_code, 422)  # name a stage or a decision
            self.assertEqual(client.get(f"/api/projects/{self.pid}/steps", params={"stage": "models", "decision": "x"}).status_code, 422)
            self.assertEqual(client.get(f"/api/projects/{self.pid}/steps", params={"stage": "nowhere"}).status_code, 404)
            self.assertEqual(client.get(f"/api/projects/{self.pid}/steps", params={"decision": "dr-nothing-0001"}).status_code, 404)
            trace = client.get(f"/api/projects/{self.pid}/lineage", params={"path": "final.evidence.holdout_metrics"}).json()
            self.assertIsInstance(trace["run"]["steps"], int)  # a clicked number carries the count, not the list
        finally:
            client.__exit__(None, None, None)

    def test_a_decision_of_a_stage_that_ran_again_is_read_from_the_events_of_its_run(self):
        models_run = self.store.records(self.pid)["models"]["run"]["run"]
        record = next(r for r in self.book.records(project=self.pid) if r.run == models_run)
        older = record.model_copy(update={"id": "dr0older0001", "run": "run-superseded"})
        self.book.add(older)
        experience.record("pipeline.step_finished", "system", {"run": "run-superseded", "step": "model ridge: fit", "kind": "fit", "index": 1, "digest": "ab"},
                          log=self.log, project=self.pid)
        experience.record("pipeline.step_finished", "system", {"run": "run-superseded", "step": "fold 1: metrics", "kind": "metric", "index": 2, "metric": "mae",
                                                               "value": 7.5, "fold": 1, "digest": "cd"}, log=self.log, project=self.pid)
        got = lineage.decision_steps(self.store, self.book, self.pid, older.id)
        self.assertEqual((got.get("from_events"), [s["name"] for s in got["steps"]]), (True, ["model ridge: fit", "fold 1: metrics"]))
        self.assertIn("run again", got["note"])

    def test_a_decision_traces_to_the_steps_that_carry_its_numbers(self):
        from dclab_rnd import ledger

        models_run = self.store.records(self.pid)["models"]["run"]["run"]
        record = next((r for r in self.book.records(project=self.pid) if r.run == models_run), None)
        self.assertIsNotNone(record, "the models stage made no decision inside its run")
        got = lineage.decision_steps(self.store, self.book, self.pid, record.id)
        self.assertEqual((got["decision"], got["run"], got["stage"]), (record.id, models_run, "models"))
        self.assertTrue(got["steps"], got)
        self.assertTrue(all(s.get("metric") is not None or s["kind"] in ("fit", "transform") for s in got["steps"]))  # the numbers, not the plumbing
        self.assertTrue(any(s.get("metric") and s.get("value") is not None for s in got["steps"]))
        early = next(r for r in self.book.records(project=self.pid) if r.run and r.run != models_run and r.run == self.store.records(self.pid)["data"]["run"]["run"])
        self.assertEqual(lineage.decision_steps(self.store, self.book, self.pid, early.id)["stage"], "data")  # a decision of the data stage: that stage's steps
        with self.assertRaises(KeyError):
            lineage.decision_steps(self.store, self.book, self.pid, "dr-nothing-0001")
        bare = lineage.decision_steps(self.store, self.book, self.pid, next(r for r in self.book.records(project=self.pid) if not r.run).id) \
            if any(not r.run for r in self.book.records(project=self.pid)) else {"steps": []}
        self.assertEqual(bare["steps"], [])


@unittest.skipIf(MISSING, MISSING)
class MainProductTest(unittest.TestCase):
    def test_its_fold_and_candidate_rows_are_steps_in_order_and_claim_no_digest(self):
        from dclab_core.adapters import main_product_experience as mp

        columns = {c: None for c in mp.recorded()["experience_tables"]["ml_run_events"]}
        rows = [{**columns, "id": "e3", "workspace_id": "w", "workflow_run_id": "run-1", "experiment_id": "x", "sequence": 3, "stage": "train",
                 "event_type": "cv_fold_completed", "status": "completed", "duration_ms": 1500.0, "payload": {"fold": 2, "metric": "roc_auc", "value": 0.8}},
                {**columns, "id": "e1", "workspace_id": "w", "workflow_run_id": "run-1", "experiment_id": "x", "sequence": 1, "stage": "train",
                 "event_type": "train_started", "status": "running"},
                {**columns, "id": "e2", "workspace_id": "w", "workflow_run_id": "run-1", "experiment_id": "x", "sequence": 2, "stage": "train",
                 "event_type": "candidate_scored", "status": "completed", "duration_ms": 200, "payload": {"score": 0.7}},
                {**columns, "id": "e9", "workspace_id": "w", "workflow_run_id": "run-2", "experiment_id": "y", "sequence": 1, "stage": "train",
                 "event_type": "cv_fold_completed", "status": "completed"}]
        out = mp.steps_from_run_events(rows, run_id="run-1")
        self.assertEqual([(s.index, s.name, s.kind, s.fold, s.value, s.seconds, s.digest) for s in out],
                         [(0, "candidate_scored", "fit", None, 0.7, 0.2, None), (1, "cv_fold_completed", "fold", 2, 0.8, 1.5, None)])
        self.assertEqual(len(mp.steps_from_run_events(rows)), 3)  # both runs when none is named
        self.assertIn("pipeline.step_finished", mp.KINDS_MISSING)


if __name__ == "__main__":
    unittest.main()
