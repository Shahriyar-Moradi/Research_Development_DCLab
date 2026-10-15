"""Package 19.12: training and evaluation sets from real use. `make datasets` builds every set the consent allows and
nothing else; each manifest names every source; the decision set of column.is_identifier keeps its benchmark,
measured and person labels apart, with the label that counts, and its sealed partition in a file of its own."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import datasets as builders
    from dclab_rnd import consent, datasets, experience, ledger
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""

EMAIL = "jane.roe" + "@" + "example.com"


@unittest.skipIf(MISSING, MISSING)
class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.agentic.server import create_app

        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.s = self.client.app.state.services
        self.home = Path(self.s.store.home)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        for use in ("improve", "train"):  # 23.1: a record carries the level it was written under; the switches are on before the writes
            consent.set_workspace_consent(self.home, use, True)
        self.p1 = self.client.post("/api/projects", json={"name": "Consented", "goal": "g"}, headers=self.h).json()["id"]
        self.p2 = self.client.post("/api/projects", json={"name": "Not consented", "goal": "g"}, headers=self.h).json()["id"]
        self.client.patch(f"/api/projects/{self.p1}", json={"settings": {"consent": {"train": True, "improve": True}}}, headers=self.h)
        book, log = self.s.ledger, self.s.experience
        self.recs = []
        def named(split):  # a dataset whose rows of this point land in ``split``
            return next(f"ds_{i}" for i in range(1000) if builders.split_of(f"dataset:ds_{i}") == split)

        for i, (project, dataset) in enumerate([(self.p1, named("train")), (self.p1, named("sealed")), (self.p1, named("dev")), (self.p2, "ds_x")]):
            rec = ledger.decide("column.is_identifier", ("column", f"col{i}"), answers=[ledger.answer("rule", "id_like", False)], value_used=False,
                                state={"column": f"col{i}", "dtype": "integer"}, ledger=book, project=project, dataset=dataset)
            self.recs.append(rec)
        first = self.recs[0]
        self.labels = [ledger.label(first, True, "benchmark", "replica:delivery_fraud", ledger=book),
                       ledger.label(first, True, "measured", "measured:uniqueness", ledger=book),
                       ledger.label(first, True, "user_decision_exposed", "person:u_owner1", ledger=book)]
        overridden = ledger.decide("column.semantic_role", ("column", "amount"), answers=[ledger.answer("rule", "roles", "identifier"),
                                   ledger.answer("person", "owner", "numeric")], value_used="numeric", used="person", ledger=book, project=self.p1)
        self.override = overridden
        for i, (kind, actor, payload) in enumerate([("draft.created", "person", {"problem": f"Predict late orders, write to {EMAIL}"}),
                                                    ("chat.agent", "agent", {"text": "When is the prediction made?", "question_field": "prediction_moment"}),
                                                    ("chat.person", "person", {"text": "When the order is placed"}),
                                                    ("solution.saved", "person", {"target": "customer_id", "task": "multiclass"}),
                                                    ("draft.built", "person", {"project": self.p1})]):
            experience.record(kind, actor, payload, log=log, at=f"2026-10-13T09:0{i}:00+00:00", visit="v1", conversation="d_conv", project=self.p1)
        for i in range(3):
            experience.record("agent.step", "agent", {"agent": "intern", "n": i, "tool": "describe_data", "verdict": "ok"}, log=log,
                              at=f"2026-10-13T10:0{i}:00+00:00", visit="v2", run="run_1", project=self.p1)
        experience.record("chat.person", "person", {"text": "secret plans"}, log=log, visit="v3", conversation="d_other", project=self.p2)

    def built(self, out):
        return {(b["builder"], b["name"]): b for b in out["built"]}

    def test_nothing_without_the_workspaces_switch(self):
        for use in ("improve", "train"):
            consent.set_workspace_consent(self.home, use, False)
        out = datasets.build(self.s)
        self.assertEqual(out["built"], [])
        self.assertEqual({x["use"] for x in out["not_built"]}, {"train", "improve"})
        self.assertFalse((self.home / datasets.FOLDER / "decisions").exists())

    def test_every_set_counts_its_rows_by_the_level_they_were_written_under(self):
        from dclab_rnd.experience import catalog

        out = self.built(datasets.build(self.s))
        manifest = json.loads((Path(out["decisions", "column.is_identifier"]["folder"]) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "dclab_core.dataset_manifest.v4")
        self.assertEqual(set(manifest["levels"]), {"train"})
        self.assertTrue(any("rows by level" in line for line in catalog.lines(catalog.catalog(self.home))))

    def test_the_identifier_set_keeps_its_labels_apart(self):
        consent.set_workspace_consent(self.home, "train", True)
        out = self.built(datasets.build(self.s))
        folder = Path(out["decisions", "column.is_identifier"]["folder"])
        rows = [json.loads(x) for f in ("rows.jsonl", "sealed.jsonl") if (folder / f).exists() for x in (folder / f).read_text().splitlines()]
        self.assertEqual({r["record"] for r in rows}, {r.id for r in self.recs[:3]})  # p2 did not consent
        first = next(r for r in rows if r["record"] == self.recs[0].id)
        self.assertEqual(first["labels"], {"benchmark": True, "measured": True, "user_decision_exposed": True, "blind": None, "model": None})
        self.assertEqual(first["label"], {"value": True, "source": "benchmark"})
        sealed = (folder / "sealed.jsonl").read_text().splitlines() if (folder / "sealed.jsonl").exists() else []
        self.assertTrue(all(json.loads(x)["split"] == "sealed" for x in sealed))
        self.assertFalse(any(json.loads(x)["split"] == "sealed" for x in (folder / "rows.jsonl").read_text().splitlines()))
        manifest = builders.DatasetManifest.read((folder / "manifest.json").read_text())
        named = {(s.kind, s.id) for s in manifest.sources}
        self.assertEqual(named, {("decision", r.id) for r in self.recs[:3]} | {("label", x.id) for x in self.labels})  # every source
        self.assertEqual(manifest.counts["labels_by_source"]["benchmark"], 1)
        self.assertNotIn("u_owner1", (folder / "rows.jsonl").read_text() + (folder / "manifest.json").read_text())

    def test_every_set_the_consent_allows(self):
        consent.set_workspace_consent(self.home, "train", True)
        consent.set_workspace_consent(self.home, "improve", True)
        out = self.built(datasets.build(self.s))
        self.assertIn(("decisions", "column.semantic_role"), out)
        folder = Path(out["conversations", "problem_understanding"]["folder"])
        convo = [json.loads(x) for f in ("rows.jsonl", "sealed.jsonl") if (folder / f).exists() for x in (folder / f).read_text().splitlines()]
        everything = "".join(p.read_text() for p in (self.home / datasets.FOLDER).rglob("*.jsonl"))
        self.assertNotIn(EMAIL, everything)
        self.assertNotIn("secret plans", everything)  # p2's conversation
        self.assertEqual(out["conversations", "problem_understanding"]["rows"], 1)
        self.assertEqual([m["role"] for m in convo[0]["messages"]], ["system", "user", "assistant", "user"][:3])
        self.assertEqual(out["preferences", "corrections"]["rows"], 1)
        self.assertEqual(out["trajectories", "agent_runs"]["rows"], 1)
        failures = out["failures", "benchmark_candidates"]
        self.assertGreaterEqual(failures["rows"], 1)  # the visit that built a model of customer_id is flagged
        candidate = json.loads((Path(failures["folder"]) / "rows.jsonl").read_text().splitlines()[0])
        self.assertEqual(candidate["review"], "pending")
        (self.home / datasets.FOLDER / "failures" / "reviews.json").write_text(json.dumps({candidate["id"]: {"verdict": "accept", "by": "reviewer"}}))
        datasets.build(self.s)
        accepted = (Path(failures["folder"]) / "accepted.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(x)["id"] for x in accepted], [candidate["id"]])
        for manifest in (self.home / datasets.FOLDER).rglob("manifest.json"):
            m = builders.DatasetManifest.read(manifest.read_text())
            self.assertTrue(m.sources or m.rows == 0, manifest)

    def test_a_workflow_leaves_without_a_file_name_or_the_persons_words(self):
        import pandas as pd

        from dclab_core.datacard import card_of

        consent.set_workspace_consent(self.home, "train", True)
        frame = pd.DataFrame({"late": [0, 1] * 30, "minutes": range(60)})
        for project, name in ((self.p1, "jane_roe_payroll.csv"), (self.p2, "other_secret_file.csv")):
            d = self.s.drafts.create("Predict which orders arrive late")["id"]
            card = card_of({name: frame}).model_dump(mode="json")
            self.s.drafts.update(d, lambda x, project=project, card=card: x.update(
                project_id=project, data_card=card,
                workflow={"steps": [{"title": "WF-01", "detail": "Predict late, when John Smith calls +44 7700 900123."}]}))
            p = self.s.projects.get(project)
            p["solution"] = {"target": "late", "task": "binary"}
            self.s.projects.save(p)
        out = self.built(datasets.build(self.s))
        text = (Path(out["workflows", "problem_to_pipeline"]["folder"]) / "rows.jsonl").read_text() \
            if (Path(out["workflows", "problem_to_pipeline"]["folder"]) / "rows.jsonl").exists() else \
            (Path(out["workflows", "problem_to_pipeline"]["folder"]) / "sealed.jsonl").read_text()
        self.assertEqual(out["workflows", "problem_to_pipeline"]["rows"], 1)  # p2 did not consent
        for gone in ("jane_roe", "John Smith", "7700", "other_secret"):
            self.assertNotIn(gone, text)
        self.assertIn("table_1.csv", text)

    def test_a_visit_that_also_touched_a_project_without_consent(self):
        consent.set_workspace_consent(self.home, "improve", True)
        log = self.s.experience
        experience.record("chat.person", "person", {"text": "Hello"}, log=log, at="2026-10-13T11:00:00+00:00", visit="vmix", project=self.p1)
        experience.record("solution.saved", "person", {"target": "falcon_merger_id", "task": "binary"}, log=log, at="2026-10-13T11:01:00+00:00",
                          visit="vmix", project=self.p2)
        experience.record("error.raised", "system", {"where": "x", "code": "E_X"}, log=log, at="2026-10-13T11:02:00+00:00", visit="vmix", project=self.p2)
        out = self.built(datasets.build(self.s))
        text = "".join(f.read_text() for f in Path(out["failures", "benchmark_candidates"]["folder"]).glob("*"))
        self.assertNotIn("falcon", text)
        self.assertNotIn("vmix", text)  # with p2's events set aside, nothing in that visit is flagged

    def test_a_withdrawn_consent_takes_its_sets_away(self):
        consent.set_workspace_consent(self.home, "train", True)
        first = self.built(datasets.build(self.s))
        self.assertIn(("decisions", "column.is_identifier"), first)
        self.client.patch(f"/api/projects/{self.p1}", json={"settings": {"consent": {"train": False}}}, headers=self.h)
        again = datasets.build(self.s)
        self.assertIn("decisions/column.is_identifier", again["removed"])
        self.assertFalse((self.home / datasets.FOLDER / "decisions" / "column.is_identifier").exists())
        consent.set_workspace_consent(self.home, "train", False)
        consent.set_workspace_consent(self.home, "improve", False)  # both withdrawn: every set goes
        datasets.build(self.s)
        self.assertEqual([p for p in (self.home / datasets.FOLDER).glob("*/*") if p.is_dir()], [])

    def test_a_draft_and_the_project_it_became_are_one_group(self):
        drafts = {"d1": "p9"}
        on_draft = {"id": "r1", "point": "column.is_identifier", "draft": "d1"}
        on_project = {"id": "r2", "point": "column.is_identifier", "project": "p9"}
        self.assertEqual(builders._group(on_draft, drafts), builders._group(on_project, drafts))
        self.assertEqual(builders.split_of(builders._group(on_draft, drafts)), builders.split_of("project:p9"))

    def test_a_row_still_holding_a_path_is_refused_and_not_named(self):
        consent.set_workspace_consent(self.home, "train", True)
        log = self.s.experience
        for i, (kind, actor, payload) in enumerate([("draft.created", "person", {"problem": "Predict churn from /Users/someone/data.csv"}),
                                                    ("chat.agent", "agent", {"text": "Which column?", "question_field": "target"}),
                                                    ("draft.built", "person", {"project": self.p1})]):
            experience.record(kind, actor, payload, log=log, at=f"2026-10-13T12:0{i}:00+00:00", visit="vpath", conversation="d_path", project=self.p1)
        out = self.built(datasets.build(self.s))
        convo = out["conversations", "problem_understanding"]
        self.assertEqual((convo["rows"], convo["refused"]), (1, 1))  # the clean one stays, the one with a path goes
        manifest = builders.DatasetManifest.read((Path(convo["folder"]) / "manifest.json").read_text())
        self.assertNotIn(("conversation", "d_path"), {(x.kind, x.id) for x in manifest.sources})
        self.assertEqual(sum(manifest.splits.values()), 1)

    def test_development_records_are_listed(self):
        out = datasets.development(self.home / datasets.FOLDER)
        self.assertGreater(out["rows"], 100)


@unittest.skipIf(MISSING, MISSING)
class SplitTest(unittest.TestCase):
    def test_a_group_always_lands_in_one_split(self):
        splits = {builders.split_of(f"dataset:{i}", "p") for i in range(200)}
        self.assertEqual(splits, {"train", "dev", "sealed"})
        self.assertEqual(builders.split_of("dataset:x", "p"), builders.split_of("dataset:x", "p"))


if __name__ == "__main__":
    unittest.main()
