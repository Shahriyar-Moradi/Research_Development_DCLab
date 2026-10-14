"""Decision samples (package A6.5): every typed decision the studio makes is written with the evidence the rule saw;
what a person does to it is written as a label; the export joins them, split by dataset, with ids masked; no cell
value and no holdout number ever enters a row; a write that fails never fails the work."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402
from dclab_rnd import decisions  # noqa: E402
from dclab_rnd.decisions import extract, labels, schema, store as dstore  # noqa: E402
from dclab_rnd.studio import ProjectStore, engine, graph, solution as sc  # noqa: E402
from test_studio import CHURN_CONTRACT, churn_frame, new_project  # noqa: E402


class SchemaTests(unittest.TestCase):
    def test_a_sample_scrubs_cells_and_reads_as_one_line(self):
        row = schema.sample("column.leakage", "fee", {"missing_rate": 0.1234567, "preview": ["secret"], "reasons": ["suspicious_name"],
                                                       "time": {"start": "2021-01-01", "span_days": 3}},
                            "review_flag", "studio.leakage@0.1.0", project_id="p1", dataset="abc")
        self.assertEqual(row["key"], "column.leakage:fee")
        self.assertNotIn("preview", row["features"])
        self.assertEqual(row["features"]["time"], {"span_days": 3})  # a row's date is a cell
        with self.assertRaises(ValueError):
            schema.label(None, "k", "p", "x", "human_exposed", "me", reason="because Finance said so")  # never a person's words
        self.assertEqual(row["features"]["missing_rate"], 0.123457)
        self.assertTrue(row["state_text"].startswith("column.leakage column=fee | missing_rate=0.1235"))
        self.assertTrue(row["sample_id"].startswith("ds_") and len(row["sample_id"]) == 23)
        with self.assertRaises(ValueError):
            schema.sample("not.a.point", None, {}, 1, "v")
        with self.assertRaises(ValueError):
            schema.sample("column.leakage", "x", {}, "clear", "v", partition="holdout")

    def test_ids_sort_by_time(self):
        a, b = schema.new_id("ds_"), schema.new_id("ds_")
        self.assertLessEqual(a[:13], b[:13])


class ExtractTests(unittest.TestCase):
    def test_the_brief_gives_one_sample_per_contract_field(self):
        from dclab_core.understand import understand

        brief = understand("Predict whether a customer churns before renewal. The target column is churned.", {"t": ["customer_id", "churned", "age"]})
        rows = extract.from_brief(brief, draft_id="d1", columns=["customer_id", "churned", "age"])
        self.assertEqual({r["point"] for r in rows}, set(extract.BRIEF_POINTS.values()))
        target = next(r for r in rows if r["point"] == "contract.target_column")
        expected = brief.target_column.value if brief.target_column.status != "unknown" else "unknown"
        self.assertEqual(target["rule_answer"], expected)  # the rule's own reading, whatever it is: a sample records, never corrects
        self.assertEqual(target["features"]["status"], brief.target_column.status)
        self.assertEqual(target["evidence_partition"], "metadata")
        self.assertEqual(target["features"]["columns"], ["age", "churned", "customer_id"])
        moment = next(r for r in rows if r["point"] == "contract.prediction_moment")
        self.assertIn(moment["rule_answer"], ("stated", "inferred", "unknown"))
        text = json.dumps(rows)
        self.assertNotIn("renewal", text)  # the person's words never enter a sample
        self.assertNotIn("quote\"", text.replace('"quoted"', "").replace('"quote_words"', ""))

    def test_the_core_plan_gives_column_and_plan_samples(self):
        frame = churn_frame(400)
        from dclab_rnd.studio import data as sd

        profile = sd.profile_table(frame)
        findings, plan = sc.core_plan(frame.drop(columns=["churned"]), frame["churned"], "churned", "binary", "signup_date")
        rows = extract.from_core(findings, plan, columns=[str(c) for c in frame.columns], profile=profile, target="churned", project_id="p1", dataset="d")
        by_key = {r["key"]: r for r in rows}
        self.assertEqual(by_key["column.forbidden:final_refund_amount"]["rule_answer"], "exclude")
        self.assertEqual(by_key["column.forbidden:age"]["rule_answer"], "clear")
        self.assertEqual(by_key["column.identifier:customer_id"]["rule_answer"], "identifier")
        self.assertEqual(by_key["column.identifier:age"]["rule_answer"], "feature")
        self.assertIn("unique_ratio", by_key["column.identifier:customer_id"]["features"])  # the profile's aggregates travel
        self.assertNotIn("column.forbidden:churned", by_key)  # never the target
        self.assertIn("split.strategy:-", by_key)
        self.assertIn(by_key["split.strategy:-"]["rule_answer"], schema.POINTS["split.strategy"]["answers"])
        self.assertIn("model.families:-", by_key)
        for r in rows:
            self.assertNotIn("preview", json.dumps(r))


class RecorderTests(unittest.TestCase):
    """One engine run on the churn table: the samples the stages write, the labels an approval and a solution write,
    and the export that joins them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        home = Path(cls.tmp.name)
        cls.store = ProjectStore(home / "projects")
        cls.store.audit_path = home / "audit.jsonl"  # a workspace: where the samples go too
        cls.project = new_project(cls.store, churn_frame(), CHURN_CONTRACT)
        cls.records = {stage: engine.execute(cls.store, cls.project["id"], stage) for stage in ("data", "leakage", "features", "models", "final")}
        cls.samples = dstore.FileSamples(home)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_every_stage_wrote_its_decisions_with_the_evidence_the_rule_saw(self):
        rows = self.samples.samples(self.project["id"])
        points = {r["point"] for r in rows}
        self.assertTrue({"column.role", "column.risk", "column.leakage", "feature.recipe", "model.family", "optimization.config"} <= points, points)
        role = next(r for r in rows if r["key"] == "column.role:customer_id")
        self.assertEqual(role["rule_answer"], "identifier")
        self.assertIn("unique_ratio", role["features"])
        self.assertEqual(role["dataset_ref"], self.project["data"]["sha256"][:16])
        leak = next(r for r in rows if r["key"] == "column.leakage:final_refund_amount")
        self.assertEqual(leak["rule_answer"], "exclude")
        family = next(r for r in rows if r["point"] == "model.family")
        self.assertEqual(family["rule_answer"], self.records["models"]["evidence"]["selected_model"])
        self.assertIn(family["rule_answer"], family["alternatives"])
        self.assertTrue(all(o.get("mean") is not None for o in family["features"]["options"]))
        for r in rows:
            self.assertEqual(r["evidence_partition"], "train")
            self.assertEqual(r["schema_version"], 1)

    def test_2_no_holdout_number_and_no_cell_enters_a_sample(self):
        final = self.records["final"]["evidence"]
        holdout_value = final["holdout_metrics"][self.records["final"]["primary_metric"]]
        text = "\n".join(json.dumps(r, default=str) for r in self.samples.samples(self.project["id"]))
        self.assertNotIn(f"{holdout_value:.6f}"[:8], text)
        self.assertNotIn("C00001", text)  # a customer id is a cell
        self.assertNotIn("preview", text)

    def test_3_approval_and_override_write_labels_that_join_at_export(self):
        pid = self.project["id"]
        options = [o["id"] for o in self.records["models"]["decision"]["options"]]
        other = next(o for o in options if o != self.records["models"]["evidence"]["selected_model"])
        engine.approve(self.store, pid, "models", choice=other)  # the person prefers another family (clears the final stage)
        engine.approve(self.store, pid, "data")  # the rule's roles stand
        labelled = self.samples.labels(pid)
        self.assertTrue(any(r["key"] == "column.role:customer_id" and r["label"] == "identifier" for r in labelled))
        chosen = next(r for r in labelled if r["key"] == "model.family:-")
        self.assertEqual((chosen["label"], chosen["source"], chosen["move"]), (other, "human_exposed", "approve_stage"))
        self.assertIsNotNone(chosen["sample_id"])
        out = Path(self.tmp.name) / "export"
        manifest = decisions.export(Path(self.tmp.name), out, self.samples)
        rows = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
        exported = next(r for r in rows if r["key"] == "model.family:-")
        self.assertEqual(exported["label"], other)
        self.assertEqual(exported["label_source"], "human_exposed")
        self.assertNotEqual(exported["project_id"], pid)  # masked
        self.assertIsNone(exported["user_id"])
        self.assertIn(exported["split"], ("train", "val", "sealed"))
        self.assertEqual(len({r["split"] for r in rows}), 1)  # one dataset, one split
        self.assertEqual(manifest["rule_agreement"]["model.family"], {"agree": 0, "differ": 1})
        self.assertGreater(manifest["labelled"], 1)

    def test_4_a_saved_solution_labels_the_column_judgments_and_a_deleted_project_is_forgotten(self):
        pid = self.project["id"]
        project = self.store.get(pid)
        before = dict(project["solution"])
        after = {**before, "forbidden": before["forbidden"] + [{"column": "support_calls", "reason": "counted after the decision"}]}
        n = decisions.label_solution(self.store, project, before, after, "human")
        self.assertEqual(n, 1)  # only what changed
        row = self.samples.labels(pid)[-1]
        self.assertEqual((row["key"], row["label"], row["move"], row["reason"]), ("column.forbidden:support_calls", "exclude", "set_solution", "excluded by the person"))
        self.assertNotIn("counted after", json.dumps(row))  # the person's reason stays in the audit
        first = labels.solution_saved(None, project, None, after, "human")
        self.assertIn("contract.target_column:-", {r["key"] for r in first})  # a first save labels everything it states
        self.assertIn("split.strategy:-", {r["key"] for r in first})
        self.assertEqual(next(r["label"] for r in first if r["key"] == "split.strategy:-"), "time")
        self.assertEqual(next(r["label"] for r in first if r["key"] == "contract.prediction_moment:-"), "stated")
        gate_rows = labels.gate_approved(None, project, "solution", "owner", "fine")
        self.assertTrue(all(r["move"] == "approve_gate" and r["reason"] == "gate approved" for r in gate_rows))
        agent = labels.stage_approved(None, project, "data", self.records["data"], None, actor="agent")
        self.assertTrue(agent and all(r["source"] == "agent" for r in agent))  # an agent's approval is kept, never as a person's
        decisions.forget(self.store, pid)
        self.assertEqual(self.samples.samples(pid), [])
        self.assertEqual(self.samples.labels(pid), [])


class StoreTests(unittest.TestCase):
    def test_file_store_is_torn_line_safe_and_a_bare_store_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            s = dstore.FileSamples(home)
            s.record([schema.sample("column.role", "a", {"x": 1}, "feature", "v", project_id="p")])
            with (home / dstore.SAMPLES_FILE).open("a") as f:
                f.write('{"sample_id": "torn')  # a crash mid-write
            s.record([schema.sample("column.role", "b", {"x": 2}, "feature", "v", project_id="p")])
            self.assertEqual([r["subject"] for r in s.samples("p")], ["a", "b"])
            self.assertEqual(s.latest("p", "column.role:b")["subject"], "b")
            self.assertIsNone(dstore.for_store(ProjectStore(home / "bare")))
            self.assertEqual(decisions.record_stage(ProjectStore(home / "bare"), {"id": "p"}, {"stage": "data", "evidence": {}}), 0)

    def test_labels_written_before_their_sample_join_by_key(self):
        s = [schema.sample("column.forbidden", "fee", {}, "clear", "v", project_id="p", at="2026-01-02T00:00:00+00:00")]
        early = schema.label(None, "column.forbidden:fee", "p", "exclude", "human_exposed", "me", at="2026-01-01T00:00:00+00:00")
        joined = decisions.join(s, [early])
        self.assertEqual(joined[0]["label"], "exclude")
        self.assertEqual(decisions.split_of("abc", None), decisions.split_of("abc", "other"))
        self.assertNotEqual(decisions.split_of(None, None, "d1"), "none")


class PgStoreTests(unittest.TestCase):
    def setUp(self):
        self.url = pgtest.require()
        pgtest.empty(self.url)

    def test_rows_round_trip_and_delete_by_project(self):
        from dclab_rnd.storage import db

        wid = db.workspace_for(Path(tempfile.mkdtemp()), self.url)
        s = dstore.PgSamples(wid, self.url)
        self.assertIsInstance(dstore.for_store(type("S", (), {"workspace_id": wid, "engine": db.engine(self.url)})()), dstore.PgSamples)
        s.record([schema.sample("column.role", "a", {"x": 1.5, "names": ["b"]}, "feature", "v", project_id="p1"),
                  schema.sample("column.role", "a", {"x": 2}, "identifier", "v", project_id="p1")])
        s.label([schema.label(s.latest("p1", "column.role:a"), "column.role:a", "p1", "identifier", "human_exposed", "me")])
        rows = s.samples("p1")
        self.assertEqual([r["rule_answer"] for r in rows], ["feature", "identifier"])
        self.assertEqual(rows[0]["features"], {"x": 1.5, "names": ["b"]})
        self.assertEqual(s.latest("p1", "column.role:a")["rule_answer"], "identifier")
        self.assertEqual(len(s.labels("p1")), 1)
        s.delete("p1")
        self.assertEqual((s.samples("p1"), s.labels("p1")), ([], []))


if __name__ == "__main__":
    unittest.main()
