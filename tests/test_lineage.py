"""Package 19.5: the lineage of every stage run. Each stage record carries its run (what it read, what it was told,
what it made, the code that ran it); every event inside the run carries its id; a report's number is traced to its
run in one call; a stage rerun from its record in a scratch copy reproduces its number, and is refused when what it
read has changed. The main product's runs (experiments, stage runs, agent runs) map onto the same record. Studio
tests run on PostgreSQL too under ``make test-pg``."""

import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import experience as x
    from dclab_core import lineage as l
    from dclab_core.adapters import main_product_experience as mpe
    from dclab_core.adapters import main_product_lineage as mpl
    from dclab_core.adapters.main_product import recorded
except ImportError as error:  # the agent environment has no numpy
    CORE_MISSING = str(error)
else:
    CORE_MISSING = ""

try:
    import pandas  # noqa: F401

    from dclab_rnd import experience
    from dclab_rnd.studio import lineage
except ImportError as error:
    STUDIO_MISSING = str(error)
else:
    STUDIO_MISSING = ""

T0 = "2026-10-14T08:00:00+00:00"


def record(**over):
    base = dict(run="run1", kind="stage", name="final", status="completed", started=T0,
                inputs=l.RunInputs(data_sha256="ab" * 32, data_rows=100, seed=42, decisions={"models": "ridge"}, upstream=["runA"]),
                params={"model": "ridge"}, outputs=l.RunOutputs(metric="mae", value=12.1, low=11.0, high=13.2))
    return l.RunRecord(**{**base, **over})


@unittest.skipIf(CORE_MISSING, CORE_MISSING)
class RunRecordTest(unittest.TestCase):
    def test_the_fingerprint_is_the_inputs_and_parameters(self):
        a = record()
        self.assertEqual(a.fingerprint, record(run="run2", started="2026-10-15T00:00:00+00:00").fingerprint)  # not when, not which run
        self.assertEqual(a.fingerprint, record(inputs=a.inputs.model_copy(update={"upstream": ["runB"]})).fingerprint)  # upstream ids differ on a rerun
        self.assertNotEqual(a.fingerprint, record(params={"model": "lasso"}).fingerprint)
        self.assertNotEqual(a.fingerprint, record(inputs=a.inputs.model_copy(update={"data_sha256": "cd" * 32})).fingerprint)
        with self.assertRaises(ValueError):  # a record edited after its run no longer matches its fingerprint
            l.RunRecord.model_validate({**a.model_dump(), "params": {"model": "lasso"}})
        self.assertEqual(l.RunRecord.model_validate(json.loads(a.model_dump_json())), a)

    def test_numbers_by_path(self):
        found = l.numbers({"holdout_metrics": {"mae": 12.1, "ok": True, "nan": float("nan")}, "folds": [1, 2.5], "name": "x"}, "evidence")
        self.assertEqual(found, {"evidence.holdout_metrics.mae": 12.1, "evidence.folds.0": 1.0, "evidence.folds.1": 2.5})
        self.assertEqual(len(l.numbers({"a": list(range(1000))}, limit=7)), 7)

    def test_a_report_path_names_its_run_and_a_rerun_is_compared(self):
        final = record()
        self.assertIs(l.run_for_path({"final": final}, "final.evidence.holdout_metrics.mae"), final)
        self.assertIsNone(l.run_for_path({"final": final}, "findings.F-cap.statement@0"))
        same = l.compare(final, record(run="run9", inputs=final.inputs.model_copy(update={"upstream": ["runZ"]})))
        self.assertTrue(same["same_inputs"] and same["same_value"] and same["same_code"])
        moved = l.compare(final, record(run="run9", params={"model": "lasso"}, outputs=l.RunOutputs(metric="mae", value=12.4)))
        self.assertEqual((moved["same_inputs"], moved["changed_params"], moved["same_value"]), (False, ["model"], False))

    def test_the_session_record_carries_a_run_as_v2_and_reads_v1(self):
        payload = l.event_payload(record(parent="run0"))
        event = x.make("run.recorded", "system", {"workspace": "w1", "run": "run1"}, payload, id="ev0000001", at=T0)
        self.assertEqual(event.schema_version, "dclab_core.experience_event.v3")
        self.assertEqual((event.payload["parent"], event.payload["outputs"]["value"]), ("run0", 12.1))
        self.assertTrue(all(c in ("meta", "count") for c in event.classes.values()))  # a run's lineage is never a table's values
        old = x.make("job.started", "system", {"workspace": "w1"}, {"kind": "stage"}, id="ev0000002", at=T0).model_dump(mode="json")
        old["schema_version"] = "dclab_core.experience_event.v1"
        self.assertEqual(x.ExperienceEvent.read(old).schema_version, "dclab_core.experience_event.v3")
        with self.assertRaises(ValueError):
            x.ExperienceEvent.read({**old, "schema_version": "dclab_core.experience_event.v0"})


def uid():
    return str(uuid.uuid4())


@unittest.skipIf(CORE_MISSING, CORE_MISSING)
class MainProductTest(unittest.TestCase):
    """Rows shaped like the main product's tables at the recorded commit."""

    def setUp(self):
        self.columns = recorded()["experience_tables"]
        self.ws, self.project = uid(), uid()

    def row(self, table, **values):
        return {c: None for c in self.columns[table]} | {"id": uid(), "workspace_id": self.ws, "project_id": self.project} | values

    def test_an_experiment_its_stages_and_an_agent_run(self):
        experiment = self.row("experiments", status="COMPLETED", pipeline_name="tabular", intent="Use the customer Jane Roe's rows", dataset_id=uid(), split_plan_id=uid(), seed=7,
                              git_commit="abc123", config={"model": {"family": "ridge"}, "folds": 5}, result={"metrics": {"mae": 3.2}},
                              started_at=T0, ended_at="2026-10-14T08:05:00+00:00")
        run = mpl.from_experiment(experiment)
        self.assertEqual((run.kind, run.name, run.status, run.seconds, run.code.commit, run.inputs.seed), ("pipeline", "tabular", "completed", 300.0, "abc123", 7))
        self.assertEqual(run.inputs.settings, {"model.family": "ridge", "folds": 5})
        self.assertEqual(run.outputs.numbers, {"metrics.mae": 3.2})
        self.assertEqual(set(run.inputs.refs), {"dataset", "split_plan"})
        self.assertNotIn("Jane Roe", run.model_dump_json())  # the person's intent is never a parameter
        stage = mpl.from_stage_run(self.row("pipeline_stage_runs", pipeline_run_id=experiment["id"], stage_key="train", status="completed",
                                            started_at=T0, duration_ms=1500, output_summary={"rows": 900},
                                            input_summary={"column": "customer_name", "rows": 1000}))
        self.assertEqual(stage.params, {"rows": 1000.0})  # only the summary's numbers: a name or a value never travels
        self.assertEqual((stage.parent, stage.seconds, stage.outputs.numbers), (experiment["id"], 1.5, {"rows": 900.0}))
        agent = mpl.from_agent_run(self.row("agent_runs", kind="specialist", agent_key="experiment_critic", agent_version="3", status="completed",
                                            context_digest="d1", policy_digest="d2", tool_catalog_digest="d3", runtime="harness",
                                            runtime_version="2", prompt_release_id=uid(), started_at=T0, finished_at="2026-10-14T08:00:30+00:00"))
        self.assertEqual(agent.inputs.refs, {"context": "d1", "policy": "d2", "tool_catalog": "d3"})
        self.assertEqual((agent.code.runner, agent.seconds), ("experiment_critic@3 harness@2", 30.0))

    def test_live_rows_are_not_runs_and_every_status_is_known(self):
        vocab = recorded()["experience_vocabularies"]["agent_runs.status"]
        live = {"queued", "running", "waiting_user"}
        self.assertEqual(set(mpl.AGENT_STATUS) | live, set(vocab))
        for status in live:
            self.assertIsNone(mpl.from_agent_run(self.row("agent_runs", status=status, kind="lead", started_at=T0)))
        self.assertIsNone(mpl.from_experiment(self.row("experiments", status="CREATED", created_at=T0)))

    def test_a_renamed_column_fails_loudly(self):
        with self.assertRaises(ValueError):
            mpl.from_experiment(self.row("experiments", status="COMPLETED", started_at=T0, commit_sha="x"))

    def test_runs_reach_the_session_record(self):
        agent_run = self.row("agent_runs", kind="lead", agent_key="assistant", status="closed", started_at=T0)
        events = mpe.events(runs={agent_run["id"]: agent_run},
                            experiments=[self.row("experiments", status="failed", failure_reason="OOM", started_at=T0)])
        recorded_runs = [e for e in events if e.kind == "run.recorded"]
        self.assertEqual(sorted(e.payload["status"] for e in recorded_runs), ["completed", "failed"])


@unittest.skipIf(STUDIO_MISSING or CORE_MISSING, STUDIO_MISSING or CORE_MISSING)
class StudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.replicas import delivery_time
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        cls.home = Path(tempfile.mkdtemp())
        cls.store = store = ProjectStore(cls.home / "projects")
        if os.environ.get("DCLAB_DATABASE_URL"):  # make test-pg: the record in PostgreSQL, as the server keeps it
            from dclab_rnd.storage import db

            db.upgrade()
            cls.log = experience.PgLog(db.workspace_for(cls.home))
        else:
            cls.log = experience.FileLog(cls.home / "experience", "wtest")
        experience.install(cls.log)
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
        cls.records = store.records(cls.pid)

    @classmethod
    def tearDownClass(cls):
        experience.install(None)

    def replay(self, stage):
        from dclab_rnd.expansion import runner

        with runner.fast_profile():  # the profile the project ran with: another one is another run (refused)
            return lineage.replay(self.store, self.pid, stage)

    def test_a_replay_under_another_engine_profile_is_refused(self):
        out = lineage.replay(self.store, self.pid, "features")  # outside fast_profile: the full model sizes
        self.assertFalse(out["replayed"])
        self.assertIn("settings", out["why"][0])

    def test_every_stage_has_its_run_chained_to_the_ones_it_read(self):
        from dclab_rnd.studio.store import STAGE_KEYS

        runs = {k: l.RunRecord.model_validate(r["run"]) for k, r in self.records.items()}
        self.assertEqual(list(runs), list(STAGE_KEYS))
        for i, stage in enumerate(STAGE_KEYS):
            self.assertEqual(runs[stage].inputs.upstream, [runs[k].run for k in STAGE_KEYS[:i]], stage)
            self.assertEqual(self.store.get(self.pid)["stages"][stage]["run"], runs[stage].run)
        final = runs["final"]
        project = self.store.get(self.pid)
        self.assertEqual(final.inputs.data_sha256, project["data"]["sha256"])
        self.assertEqual(final.inputs.decisions["models"], str(self.records["models"]["decision"]["chosen"]))
        self.assertEqual(final.params["model"], self.records["models"]["decision"]["chosen"])
        self.assertEqual(final.outputs.value, self.records["final"]["evidence"]["holdout_metrics"][final.outputs.metric])
        self.assertLessEqual(final.outputs.low, final.outputs.value)
        self.assertEqual(final.code.core, __import__("dclab_core").__version__)
        self.assertTrue(final.code.prompts)
        self.assertRegex(final.code.commit or "", r"^[0-9a-f]{40}$")

    def test_the_session_record_holds_each_run_and_its_events_carry_its_id(self):
        events = self.log.read(project=self.pid)
        final_run = self.records["final"]["run"]["run"]
        mine = [e for e in events if e.kind == "run.recorded" and e.ids.run == final_run]  # a replay records its own run besides
        self.assertEqual([(e.payload["name"], e.payload["status"]) for e in mine], [("final", "completed")])
        self.assertEqual(mine[0].payload["fingerprint"], self.records["final"]["run"]["fingerprint"])
        self.assertEqual({e.payload["name"] for e in events if e.kind == "run.recorded" and not e.payload.get("parent")},
                         set(self.records))  # one run per stage, besides replays
        finished = [e for e in events if e.kind == "stage.finished" and e.payload.get("stage") == "final"]
        self.assertEqual([e.ids.run for e in finished], [final_run])

    def test_every_number_in_the_report_is_traced(self):
        from dclab_rnd.studio import solution_report

        built = solution_report.build(self.store, self.pid)
        self.assertGreater(len(built.numbers), 30)
        for path, value in built.numbers.items():
            with self.subTest(path=path):
                found = lineage.trace(self.store, self.pid, path)
                self.assertAlmostEqual(found["value"], value, places=6)
                if found["stage"] is not None:
                    self.assertEqual(found["run"]["run"], self.records[found["stage"]]["run"]["run"])
                else:
                    self.assertIn("report", found["note"])

    def test_a_number_in_the_report_is_one_call_from_its_run(self):
        metric = self.records["final"]["primary_metric"]
        path = f"final.evidence.holdout_metrics.{metric}"
        found = lineage.trace(self.store, self.pid, path)
        self.assertEqual(found["value"], self.records["final"]["evidence"]["holdout_metrics"][metric])
        self.assertEqual(found["run"]["run"], self.records["final"]["run"]["run"])
        self.assertIn(path.split(".", 1)[1], found["run"]["outputs"]["numbers"])
        with self.assertRaises(KeyError):
            lineage.trace(self.store, self.pid, "final.evidence.nothing_here")

    def test_a_replay_reproduces_the_number_and_leaves_the_project_alone(self):
        from dclab_rnd.studio import engine

        before = (self.store.get(self.pid), self.store.records(self.pid))
        events_before = len(self.log.read(project=self.pid))
        for stage in ("models", "final"):
            engine._BUNDLES.clear()  # the replay reads the table again, as a later process would
            out = self.replay( stage)
            self.assertTrue(out["replayed"], out)
            self.assertTrue(out["same_inputs"] and out["same_value"] and out["same_choice"] and out["same_numbers"], out)
            self.assertGreater(out["numbers_compared"], 10)
        after = (self.store.get(self.pid), self.store.records(self.pid))
        self.assertEqual(before[1], after[1])
        self.assertEqual({k: v for k, v in before[0].items() if k != "updated"}, {k: v for k, v in after[0].items() if k != "updated"})
        new = self.log.read(project=self.pid)[events_before:]
        self.assertEqual([e.kind for e in new], ["run.recorded", "run.recorded"])  # the replays' own steps are not the project's
        self.assertEqual(new[-1].payload["parent"], self.records["final"]["run"]["run"])

    def test_a_replay_is_refused_when_what_it_read_has_changed(self):
        project = self.store.get(self.pid)
        cases = {
            "solution_hash": {**project, "solution": {**project["solution"], "prediction_moment": "Later."}},
            "settings": {**project, "settings": {**project["settings"], "quick": False, "max_rows": 300}},
            "refs": {**project, "feature_programs": [{"name": "planted"}]},
            "data_sha256": {**project, "data": {**project["data"], "sha256": "0" * 64}},
        }
        for field, changed in cases.items():
            with self.subTest(field=field), mock.patch.object(self.store, "get", return_value=changed):
                out = self.replay( "final")
                self.assertFalse(out["replayed"], out)
                self.assertTrue(field in out["why"][0] or "cannot be read" in out["why"][0], out)
        models = dict(self.records["models"], decision={**self.records["models"]["decision"], "chosen": "planted_family"})
        real = self.store.read_stage
        with mock.patch.object(self.store, "read_stage", side_effect=lambda pid, k: models if k == "models" else real(pid, k)):
            out = self.replay( "final")
        self.assertFalse(out["replayed"])
        self.assertIn("decisions", out["why"][0])

    def test_a_setting_that_changes_no_run_changes_no_replay(self):
        project = self.store.get(self.pid)
        opted = {**project, "settings": {**project["settings"], "share_for_training": not project["settings"].get("share_for_training")}}
        with mock.patch.object(self.store, "get", return_value=opted):
            out = self.replay( "features")
        self.assertTrue(out["replayed"] and out["same_inputs"], out)

    def test_the_final_stage_is_replayed_only_on_its_own_code(self):
        other = l.RunCode(commit="0" * 40, core="9.9.9")
        with mock.patch.object(lineage, "code", return_value=other):
            out = self.replay( "final")
        self.assertFalse(out["replayed"])
        self.assertIn("holdout", out["why"][0])

    def test_a_changed_number_is_seen_even_when_nothing_is_chosen(self):
        from dclab_rnd.studio import engine

        real = engine.stage_leakage

        def planted(p, eid):
            evidence, claims, summary = real(p, eid)
            evidence = dict(evidence, planted_number=999.0)
            return evidence, claims, summary

        with mock.patch.object(engine, "stage_leakage", side_effect=planted):
            out = self.replay( "leakage")
        self.assertTrue(out["replayed"] and out["same_value"], out)  # leakage chooses nothing: no value to differ
        self.assertFalse(out["same_numbers"])
        self.assertIn("evidence.planted_number", out["changed_numbers"])

    def test_a_failed_stage_is_a_failed_run(self):
        from dclab_rnd.studio import engine

        other = self.store.create("Copy", "general", "")
        copy = {**self.store.get(self.pid), "id": other["id"], "stages": other["stages"]}
        self.store.save(copy)
        source = self.store.data_dir(self.pid) / "data.csv"
        (self.store.data_dir(other["id"]) / "data.csv").write_bytes(source.read_bytes())
        with mock.patch.object(engine, "stage_data", side_effect=RuntimeError("planted " + "x" * 900)), self.assertRaises(RuntimeError):
            engine.execute(self.store, other["id"], "data")  # a long message: the record keeps its first 600 characters
        failed = [e for e in self.log.read(project=other["id"]) if e.kind == "run.recorded"]
        self.assertEqual([(e.payload["status"], e.payload["name"]) for e in failed], [("failed", "data")])
        self.assertEqual(self.store.get(other["id"])["stages"]["data"]["run"], failed[0].ids.run)

        class Stopped(BaseException):  # as jobs.worker.Stopped: a person's Stop between steps
            pass

        with mock.patch.object(engine, "stage_data", side_effect=Stopped()), self.assertRaises(Stopped):
            engine.execute(self.store, other["id"], "data")
        last = [e for e in self.log.read(project=other["id"]) if e.kind == "run.recorded"][-1]
        self.assertEqual(last.payload["status"], "stopped")

    def test_the_route_answers_where_a_number_comes_from(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        store = client.app.state.services.projects
        project = store.create("Traced", "general", "")
        store.write_stage(project["id"], "final", self.records["final"])
        metric = self.records["final"]["primary_metric"]
        r = client.get(f"/api/projects/{project['id']}/lineage", params={"path": f"final.evidence.holdout_metrics.{metric}"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["run"]["run"], self.records["final"]["run"]["run"])
        runs = client.get(f"/api/projects/{project['id']}/lineage").json()["runs"]
        self.assertEqual(list(runs), ["final"])
        self.assertEqual(client.get(f"/api/projects/{project['id']}/lineage", params={"path": "data.evidence"}).status_code, 404)
        self.assertEqual(client.get(f"/api/projects/{project['id']}/lineage", params={"path": "final.x" * 60}).status_code, 422)

    def test_a_stage_from_before_lineage_says_so(self):
        record_ = dict(self.records["data"])
        record_.pop("run")
        with mock.patch.object(self.store, "read_stage", return_value=record_):
            self.assertIsNotNone(lineage.trace(self.store, self.pid, "data.evidence")["note"])
            self.assertFalse(self.replay( "data")["replayed"])


@unittest.skipIf(STUDIO_MISSING or CORE_MISSING, STUDIO_MISSING or CORE_MISSING)
class OtherRunsTest(unittest.TestCase):
    """An agent's run and a draft's data pipeline are runs too: each in the session record with its own id."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.log = experience.FileLog(self.home / "experience", "wtest")
        experience.install(self.log)

    def tearDown(self):
        experience.install(None)

    def test_an_agent_run(self):
        from dclab_rnd.agents import Policy, Registry, Tool, run
        from dclab_rnd.agents.traces import Tracer
        from tests.test_agents_runtime import obj, reply

        class Scripted:
            def __init__(self, *replies):
                self.replies = list(replies)

            def complete(self, messages, tools=None, **_):
                return self.replies.pop(0) if self.replies else reply("done")

        registry = Registry()
        registry.register(Tool("add", "Add.", obj({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]), lambda a, b: {"sum": a + b}))
        policy = Policy(name="adder", system="Add.", tools=("add",), max_steps=3)
        trace = Tracer(None, "conv1", "adder")
        run(policy, Scripted(reply(calls=[("add", {"a": 1, "b": 2})]), reply("3")), registry, [{"role": "user", "content": "1+2"}], trace=trace)
        run(policy, Scripted(reply("3")), registry, [{"role": "user", "content": "again"}], trace=trace)  # a second turn
        events = [e for e in self.log.read() if e.kind == "run.recorded"]
        self.assertEqual(len(events), 2)
        e = events[0]
        self.assertEqual((e.payload["run_kind"], e.payload["name"], e.payload["status"], e.payload["parent"], e.ids.run),
                         ("agent", "adder", "completed", "conv1", "conv1"))
        self.assertEqual(e.payload["outputs"]["chosen"], "answered")
        self.assertEqual(e.payload["code"]["runner"], "dclab_rnd.agents.runtime")
        self.assertNotEqual(events[0].payload["run"], events[1].payload["run"])  # each turn its own run, in one conversation

        class Broken:
            def complete(self, *a, **k):
                raise RuntimeError("model down")

        with self.assertRaises(RuntimeError):
            run(policy, Broken(), registry, [{"role": "user", "content": "1+2"}])
        failed = [e for e in self.log.read() if e.kind == "run.recorded"][-1]
        self.assertEqual(failed.payload["status"], "failed")

        class Stopped(BaseException):  # as jobs.worker.Stopped: a person pressed Stop
            pass

        class Stopping:
            def complete(self, *a, **k):
                raise Stopped()

        with self.assertRaises(Stopped):
            run(policy, Stopping(), registry, [{"role": "user", "content": "1+2"}])
        self.assertEqual([e for e in self.log.read() if e.kind == "run.recorded"][-1].payload["status"], "stopped")

    def test_a_draft_pipeline_run(self):
        import hashlib

        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.store import DraftStore

        drafts = DraftStore(self.home / "drafts")
        draft = drafts.create("Predict late orders")
        path = drafts.data_dir(draft["id"]) / "orders.csv"
        path.write_text("order_id,distance,late\n" + "".join(f"{i},{i % 7},{i % 2}\n" for i in range(60)))
        asset = pipeline.new_asset(drafts, draft["id"], "upload", "orders.csv", "orders.csv")
        final = pipeline.run(drafts, draft["id"], asset["id"])
        self.assertEqual(final["status"], "ready")
        mine = [e for e in self.log.read() if e.kind == "run.recorded"]
        self.assertEqual(len(mine), 1)
        e = mine[0]
        self.assertEqual((e.payload["run_kind"], e.payload["status"], e.ids.conversation), ("pipeline", "completed", draft["id"]))
        self.assertEqual(e.payload["inputs"]["data_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        inside = [x for x in self.log.read() if x.ids.run == e.ids.run and x.kind != "run.recorded"]
        self.assertTrue(inside, "the pipeline's own events carry its run id")


if __name__ == "__main__":
    unittest.main()
