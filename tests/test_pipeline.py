"""Package 20.8, the pipeline of workflows: every step behind a project's numbers as one object with its lineage. A
project's pipeline lists every step that produced its numbers, marks a step stale when what it read has changed (and
every later step with it), says where to resume, and reruns one step from its record: a stage replayed in a scratch
copy, the report rebuilt and compared number by number."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import pipeline as core  # noqa: E402

try:
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine, pipeline
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


def chain(*statuses):
    ids = ["a", "b", "c", "d"][:len(statuses)]
    return core.Pipeline(steps=[{"id": i, "kind": "stage", "name": i, "status": st, "depends_on": [ids[n - 1]] if n else [],
                                 "numbers": [f"{i}.value"]} for n, (i, st) in enumerate(zip(ids, statuses))])


class PipelineContractTest(unittest.TestCase):
    def test_a_step_comes_after_what_it_reads(self):
        with self.assertRaises(ValueError):
            core.Pipeline(steps=[{"id": "b", "kind": "stage", "name": "b", "status": "done", "depends_on": ["a"]},
                                 {"id": "a", "kind": "stage", "name": "a", "status": "done"}])
        with self.assertRaises(ValueError):
            core.Pipeline(steps=[{"id": "a", "kind": "stage", "name": "a", "status": "done"}] * 2)

    def test_a_stale_step_makes_every_later_step_stale_and_is_where_to_resume(self):
        p = core.stale_from(chain("done", "stale", "done", "done"))
        self.assertEqual([s.status for s in p.steps], ["done", "stale", "stale", "stale"])
        self.assertIn("b", p.steps[2].why)
        self.assertEqual(core.resume(chain("done", "stale", "done", "done")), "b")
        self.assertEqual(core.resume(chain("done", "done", "not_run", "not_run")), "c")
        self.assertIsNone(core.resume(chain("done", "done", "done")))

    def test_a_number_is_traced_to_every_step_behind_it(self):
        p = chain("done", "done", "done", "done")
        self.assertEqual(core.produced(p, "c.value"), ["a", "b", "c"])
        self.assertEqual(core.downstream(p, "b"), ["c", "d"])
        self.assertEqual(core.produced(p, "nowhere.value"), [])

    def test_the_main_products_digest_of_a_pipeline_version(self):
        from dclab_core.adapters import main_product_pipeline

        p = chain("done", "done")
        row = main_product_pipeline.pipeline_version(p, {"seed": 0})
        expected = hashlib.sha256(json.dumps({"config": {"seed": 0}, "graph_definition": row["graph_definition"]}, sort_keys=True,
                                             separators=(",", ":"), default=str).encode()).hexdigest()
        self.assertEqual(row["content_digest"], expected)
        self.assertEqual(row["graph_definition"]["edges"], [{"from": "a", "to": "b"}])
        self.assertNotIn("numbers", json.dumps(row["graph_definition"]))  # what a step made belongs to its run, not the graph
        fields = json.loads((ROOT / "dclab_core/adapters/main_product_fields.json").read_text())["pipeline_version_fields"]
        self.assertTrue(set(main_product_pipeline.FIELDS) <= set(fields))


@unittest.skipIf(MISSING, MISSING)
class ProjectPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(cls.store, cls.pid)
        cls.pipe = pipeline.of(cls.store, cls.pid)

    def test_every_step_that_produced_the_numbers_is_listed_with_its_run(self):
        ids = [s.id for s in self.pipe.steps]
        self.assertEqual(ids, ["problem", "data", "solution", "workflow", "stage:data", "stage:leakage", "stage:features",
                               "stage:models", "stage:final", "analysis", "report"])
        for s in self.pipe.steps:
            if s.kind == "stage":
                self.assertEqual(s.status, "done")
                self.assertTrue(s.run and s.fingerprint and s.numbers, s.id)
        self.assertEqual(self.pipe.step("problem").status, "not_recorded")  # made without a draft: said, not guessed
        holdout = "final.evidence.holdout_metrics.roc_auc"
        self.assertIn(holdout, self.pipe.step("stage:final").numbers)
        self.assertEqual(core.produced(self.pipe, holdout)[-1], "stage:final")
        self.assertIn("stage:data", core.produced(self.pipe, holdout))
        self.assertIsNone(core.summary(self.pipe)["resume_from"])

    def test_every_number_of_the_report_is_traced_to_the_step_that_wrote_it(self):
        from dclab_rnd.studio import solution_report

        report = solution_report.build(self.store, self.pid, analyse=True)
        self.assertTrue(any(p.startswith("findings") for p in report.numbers))  # the analysis is cited, and listed
        for path in report.numbers:
            chain = core.produced(self.pipe, path)
            self.assertTrue(chain, path)
            root = path.split(".", 1)[0].split("@")[0]
            if root in ("data", "leakage", "features", "models", "final"):
                self.assertEqual(chain[-1], f"stage:{root}", path)  # a stage's text number is the stage's, not the report's
            if root.startswith("findings"):
                self.assertEqual(chain[-1], "analysis", path)
        self.assertEqual(self.pipe.step("analysis").status, "done")

    def test_the_graph_says_what_the_pipeline_is_not_how_far_it_ran(self):
        from dclab_core.adapters import main_product_pipeline

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        before = main_product_pipeline.pipeline_version(pipeline.of(store, pid))["content_digest"]
        self.assertEqual(before, main_product_pipeline.pipeline_version(self.pipe)["content_digest"])
        nodes = core.by_node(self.pipe)
        self.assertEqual(nodes["WF-01"], "done")
        self.assertEqual(self.pipe.step("stage:final").nodes, ["WF-08", "WF-09"])

    def test_a_record_edited_after_its_run_is_stale_and_the_report_says_so(self):
        import copy

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        record = copy.deepcopy(store.read_stage(pid, "final"))
        metric = record["primary_metric"]
        record["evidence"]["holdout_metrics"][metric] = 0.123
        store.write_stage(pid, "final", record)
        p = pipeline.of(store, pid)
        self.assertEqual(p.step("stage:final").status, "stale")
        self.assertIn("no longer says", p.step("stage:final").why)
        out = pipeline.rerun(store, pid, "report")
        self.assertFalse(out["same"], out)
        self.assertIn(f"final.evidence.holdout_metrics.{metric}", out["differ"])

    def test_a_persons_choice_is_shown_beside_the_rules(self):
        import copy

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        record = copy.deepcopy(store.read_stage(pid, "models"))
        other = next(o["id"] for o in record["decision"]["options"] if o["id"] != record["decision"]["chosen"])
        record["decision"]["chosen"] = other  # what an approval with another choice leaves
        store.write_stage(pid, "models", record)
        step = pipeline.of(store, pid).step("stage:models")
        self.assertEqual((step.status, step.outputs["chosen"], step.outputs["overridden"]), ("done", other, True))

    def test_a_stage_from_before_run_records_cannot_vouch_for_what_read_it(self):
        p = core.from_steps(None, [{"id": "a", "kind": "stage", "name": "a", "status": "not_recorded"},
                                   {"id": "b", "kind": "stage", "name": "b", "status": "done", "depends_on": ["a"]}])
        self.assertEqual(p.step("b").status, "stale")
        q = core.from_steps(None, [{"id": "a", "kind": "problem", "name": "a", "status": "not_recorded"},
                                   {"id": "b", "kind": "stage", "name": "b", "status": "done", "depends_on": ["a"]}])
        self.assertEqual(q.step("b").status, "done")  # a problem stated in the solution is not a stage to check

    def test_one_stage_is_rerun_from_its_record_and_makes_the_same_numbers(self):
        out = pipeline.rerun(self.store, self.pid, "stage:models")
        self.assertTrue(out["replayed"], out)
        self.assertTrue(out["same_numbers"] and out["same_choice"], out)

    def test_the_report_is_rebuilt_and_every_number_matches_its_record(self):
        out = pipeline.rerun(self.store, self.pid, "report")
        self.assertTrue(out["same"], out)
        self.assertGreater(out["numbers"], 10)
        self.assertGreater(out["compared_with_runs"], 0)

    def test_a_report_number_that_differs_from_its_record_is_found(self):
        from dclab_rnd.studio import solution_report

        real = solution_report.build

        def altered(*args, **kwargs):
            report = real(*args, **kwargs)
            path = next(p for p in report.numbers if p.startswith("final.evidence.holdout_metrics."))  # a number its run recorded
            return report.model_copy(update={"numbers": {**report.numbers, path: report.numbers[path] + 0.5}})

        with mock.patch.object(pipeline, "_report", lambda store, pid: altered(store, pid, analyse=True)):
            out = pipeline.rerun(self.store, self.pid, "report")
        self.assertFalse(out["same"], out)
        self.assertEqual(len(out["differ"]), 1)

    def test_what_a_person_attached_is_not_rerun(self):
        self.assertFalse(pipeline.rerun(self.store, self.pid, "data")["replayed"])
        with self.assertRaises(KeyError):
            pipeline.rerun(self.store, self.pid, "stage:nowhere")

    def test_a_changed_input_makes_its_stage_and_everything_after_stale(self):
        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        project = store.get(pid)
        project["settings"] = {**project.get("settings", {}), "folds": 4}  # a setting every stage's run read
        store.save(project)
        p = pipeline.of(store, pid)
        self.assertEqual(p.step("stage:data").status, "stale")
        self.assertIn("settings", p.step("stage:data").why)
        self.assertEqual({p.step(f"stage:{s}").status for s in ("leakage", "features", "models", "final")}, {"stale"})
        self.assertEqual(p.step("report").status, "stale")
        self.assertEqual(core.resume(p), "stage:data")

    def test_a_draft_built_project_records_its_problem_card(self):
        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        project = store.get(pid)
        project["draft"] = {"card": {"family": "binary", "problem": "x"}, "workflow": {"steps": ["a"]}}
        store.save(project)
        p = pipeline.of(store, pid)
        self.assertEqual((p.step("problem").status, p.step("workflow").status), ("done", "done"))
        self.assertEqual(p.step("problem").outputs["family"], "binary")
        self.assertEqual(p.step("stage:data").status, "not_run")
        self.assertEqual(core.resume(p), "stage:data")

    def test_the_routes(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "P", "goal": "g"}, headers=h).json()["id"]
        r = client.get(f"/api/projects/{pid}/pipeline", headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["resume_from"], "data")  # nothing attached: the pipeline starts with the person's table
        self.assertEqual(r.json()["steps"][1]["status"], "not_run")
        r = client.post(f"/api/projects/{pid}/pipeline/rerun?step=stage:nowhere", headers=h)
        self.assertEqual(r.status_code, 404, r.text)
        r = client.post(f"/api/projects/{pid}/pipeline/rerun?step=stage:models", headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()["replayed"])
        self.assertEqual(r.json()["job"]["status"], "done")  # a job: the limits on jobs at once and Stop apply


if __name__ == "__main__":
    unittest.main()
