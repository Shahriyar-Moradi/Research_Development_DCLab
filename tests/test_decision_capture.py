"""Package 19.7: every decision is captured where it is made. One project, from its first sentence on Home to its
report, leaves a decision record for every decision of both kinds it made, each with its state and its answers; the
list of points is complete (each is captured or said not to be decided here, with why); no state holds a table value;
a refused record never stops the work. The same test runs on PostgreSQL under ``make test-pg``."""

import json
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
    from fastapi.testclient import TestClient

    from dclab_core import ledger as L
    from dclab_rnd import ledger
    from dclab_rnd.agentic.server import create_app
    from tests.test_draft_flow import wait
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""

PROBLEM = "Predict which orders arrive late, at the moment the order is placed"
SECRET = "Jane Roe of 12 Main Street"  # a value in the table: it must never reach a decision's state


def orders(n=400, seed=0):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.5, 12, n)
    return pd.DataFrame({"order_id": np.arange(10_000, 10_000 + n), "created_at": pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                         "customer": [SECRET if i % 7 == 0 else f"c{i % 40}" for i in range(n)], "distance_km": distance.round(2),
                         "zone": [f"z{i % 4}" for i in range(n)], "fare": (20 + distance * 3 + rng.normal(0, 2, n)).round(1),
                         "late": (rng.random(n) < 0.15 + 0.04 * distance).astype(int)})


@unittest.skipIf(MISSING, MISSING)
class HomeToReportTest(unittest.TestCase):
    """The whole path through the real API: the Home agent with no model (its standard turns), the wizard, the build,
    every stage, an approval with an override, the report."""

    @classmethod
    def setUpClass(cls):
        cls.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        cls.env.start()
        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        c, h = cls.client, {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        d = c.post("/api/drafts", json={"problem": PROBLEM}, headers=h).json()
        cls.draft = d["id"]
        c.put(f"/api/drafts/{d['id']}/data?filename=orders.csv", content=orders().to_csv(index=False).encode(), headers=h)
        wait(lambda: any(a["status"] == "ready" for a in c.get(f"/api/drafts/{d['id']}").json()["assets"]))
        proposal = c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": "late"}, headers=h).json()
        solution = {"target": "late", "task": "binary", "positive_label": "1", "prediction_moment": "When the order is placed, before dispatch",
                    "forbidden": [{"column": "fare", "reason": "the fare is settled after delivery"}], "identifiers": ["order_id"],
                    "time_column": "created_at", "group_column": None, "text_columns": [], "metric": proposal["metric"], "notes": ""}
        saved = c.put(f"/api/drafts/{d['id']}/solution", json=solution, headers=h)
        assert saved.status_code == 200, saved.text
        c.put(f"/api/drafts/{d['id']}/settings", json={"split": "time", "folds": 3, "quick": True, "max_rows": 400, "budget": {}}, headers=h)
        built = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=h)
        assert built.status_code == 201, built.text
        cls.project = built.json()["id"]
        for stage in ("data", "leakage", "features", "models", "final"):
            ran = c.post(f"/api/projects/{cls.project}/stages/{stage}/run?wait=true", headers=h)
            assert ran.status_code == 200, (stage, ran.text)
        record = c.get(f"/api/projects/{cls.project}").json()
        options = [o["id"] for o in ((record.get("records") or {}).get("models") or {}).get("decision", {}).get("options", [])]
        cls.override = next((o for o in options if o != record["records"]["models"]["decision"]["selected"]), None)
        overridden = c.post(f"/api/projects/{cls.project}/stages/models/approve", json={"choice": cls.override}, headers=h)
        cls.override_status = overridden.status_code  # before data's approval: approving a stage clears the ones after it
        approved = c.post(f"/api/projects/{cls.project}/stages/data/approve", json={}, headers=h)
        assert approved.status_code == 200, approved.text
        cls.again = c.post(f"/api/projects/{cls.project}/stages/final/run?wait=true&reuse_reason=checking%20a%20replay", headers=h).status_code
        cls.revisited = c.post(f"/api/projects/{cls.project}/stages/data/run?wait=true", headers=h).status_code  # a completed stage again
        cls.services = cls.client.app.state.services
        cls.records = cls.services.ledger.records()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.env.stop()

    def points(self, **filters):
        return {r.point for r in self.records if all(getattr(r, k) == v for k, v in filters.items())}

    def test_the_final_stage_and_its_approval_leave_an_outcome_for_the_decisions_it_followed(self):
        """23.5 through the real engine: the holdout is the outcome of every decision of the project, and a replay adds none."""
        rows = self.services.ledger.outcomes(project=self.project)
        self.assertTrue(rows, "no outcome after the final stage")
        self.assertEqual({o.observed for o in rows} - {"promotion"}, {"holdout"})  # the final stage, and its approval when the test made one
        holdouts = [o for o in rows if o.observed == "holdout"]
        self.assertTrue(all(o.verdict in ("well", "badly") and o.higher_is_better is not None and o.reference is not None for o in holdouts))
        decided = {r.id for r in self.services.ledger.records(project=self.project) if r.at <= max(o.at for o in holdouts)}
        self.assertTrue(decided <= {o.record for o in holdouts} | {o.record for o in rows}, "a decision of the project has no outcome")

    def test_every_point_is_captured_or_said_not_decided_here(self):
        self.assertEqual(set(ledger.CAPTURED) | set(ledger.NOT_DECIDED), set(L.POINTS))
        self.assertFalse(set(ledger.CAPTURED) & set(ledger.NOT_DECIDED))

    def test_the_path_from_home_to_the_stages_leaves_its_decisions(self):
        on_home = {"problem.pack", "workflow.template", "workflow.next_question", "data.cleaning_step", "target.column", "task.framing",
                   "column.is_identifier", "feature.leakage_suspect", "column.time", "problem.family", "problem.ready"}
        in_project = {"workflow.move", "split.strategy", "split.folds", "metric.primary", "data.sampling", "column.semantic_role",
                      "features.recipe", "model.family", "tuning.accept", "workflow.stage_approval"}
        found = {r.point for r in self.records}
        self.assertEqual(on_home - found, set(), "decided on Home without a record")
        self.assertEqual(in_project - found, set(), "decided in the project without a record")
        self.assertTrue({"technical", "workflow"} <= {r.kind for r in self.records})

    def test_each_record_has_its_state_its_answers_and_its_place(self):
        identifier = next(r for r in self.records if r.point == "column.is_identifier" and r.subject.key == "order_id")
        self.assertEqual((identifier.value_used, identifier.used, identifier.draft), (True, "person", self.draft))
        self.assertEqual(identifier.answer("rule").value, True)  # the proposal said so too: rule and person agree
        self.assertIn(identifier.state["dtype"], L.DTYPE_KINDS)
        self.assertIn(identifier.state["uniqueness"], L.RATIO_BANDS)
        fare = next(r for r in self.records if r.point == "feature.leakage_suspect" and r.subject.key == "fare")
        self.assertEqual((fare.value_used, fare.used), (True, "person"))  # the person forbade it, whatever the audit said
        model = next(r for r in self.records if r.point == "model.family")
        self.assertEqual((model.project, model.used), (self.project, "rule"))
        self.assertTrue(model.options)
        moves = [r for r in self.records if r.point == "workflow.move"]
        self.assertTrue(all(r.project == self.project and r.answer("rule") is not None for r in moves))
        self.assertTrue(any(r.state.get("stage") == "final" for r in moves))

    def test_every_answer_is_attributed_to_who_gave_it(self):
        metric = next(r for r in self.records if r.point == "metric.primary")
        self.assertEqual(metric.used, "person")  # the wizard saved the metric
        folds = next(r for r in self.records if r.point == "split.folds")
        self.assertEqual((folds.used, folds.value_used), ("person", 3))
        self.assertEqual(self.override_status, 200)
        if self.override:
            chosen = [r for r in self.records if r.point == "model.family" and r.used == "person"]
            self.assertEqual([r.value_used for r in chosen], [self.override])
            self.assertNotEqual(chosen[0].answer("rule").value, self.override)
        approvals = [r for r in self.records if r.point == "workflow.stage_approval"]
        self.assertTrue(approvals and all(r.value_used is True for r in approvals))
        reuse = [r for r in self.records if r.point == "holdout.reuse"]
        self.assertTrue(reuse, f"final run again: {self.again}")
        self.assertEqual((reuse[0].answer("person").value, reuse[0].state["reason_given"]), (True, True))
        self.assertTrue(any(r.point == "workflow.revisit" for r in self.records))

    def test_a_state_never_holds_its_own_answer(self):
        for point, field in (("task.framing", "task"), ("metric.primary", "metric"), ("workflow.move", "gate_required"), ("lesson.accept", "kind")):
            for record in (r for r in self.records if r.point == point):
                self.assertNotIn(field, record.state, point) if point != "lesson.accept" else None

    def test_the_leakage_stage_records_its_measurement(self):
        fare = [r for r in self.records if r.point == "feature.leakage_suspect" and r.subject.key == "fare" and r.project == self.project]
        self.assertTrue(fare)
        self.assertIsInstance(fare[0].state.get("apparent_lift"), float)

    def test_a_question_is_recorded_only_when_asked(self):
        draft = self.services.drafts.get(self.draft)
        asked = draft["questions"]  # every question asked, a later one superseding an earlier included
        recorded = [r for r in self.records if r.point == "workflow.next_question" and r.value_used != "none"]
        self.assertLessEqual(len(recorded), len(asked))

    def test_the_workflow_is_recorded_when_it_changes(self):
        drawn = [r for r in self.records if r.point == "workflow.template" and r.draft == self.draft]
        self.assertTrue(drawn)
        self.assertEqual(len(drawn), len({(r.value_used, r.used) for r in drawn}) + 0 if len(drawn) == 1 else len(drawn))
        self.assertLessEqual(len(drawn), 2)  # the first drawing, and at most a pack change

    def test_no_state_or_answer_holds_a_table_value(self):
        for record in self.records:
            text = record.model_dump_json()
            self.assertNotIn(SECRET, text, record.point)
            self.assertNotIn("Main Street", text, record.point)

    def test_a_refused_record_never_stops_the_work(self):
        before = ledger.DROPPED["count"]
        record = ledger.decide("column.is_identifier", ("column", "x"), answers=[ledger.answer("rule", "r", True)], value_used="Jane Roe",
                               ledger=self.services.ledger)
        self.assertIsNone(record)
        self.assertEqual(ledger.DROPPED["count"], before + 1)


@unittest.skipIf(MISSING, MISSING)
class CaptureTest(unittest.TestCase):
    """The capture functions on what their sites pass, one case each."""

    def setUp(self):
        self.store = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "wtest")
        ledger.install(self.store)

    def tearDown(self):
        ledger.install(None)

    PROFILE = {"rows": 500, "columns": [{"name": n, "kind": k, "unique_ratio": u, "missing_rate": 0.0, "unique": int(500 * u)}
                                       for n, k, u in (("order_id", "numeric", 1.0), ("fare", "numeric", 0.6), ("zone", "categorical", 0.01),
                                                       ("late", "numeric", 0.004))]}

    def test_a_studied_samples_solution_is_truth_not_an_answer(self):
        from dclab_rnd.ledger import capture

        proposal = {"target": "late", "detected": {"task": "binary"}, "identifiers": ["order_id", "zone"],  # zone: from the written solution
                    "forbidden": [{"column": "fare", "proof": ["DCLAB-R01"]}], "rule": {"identifiers": ["order_id"], "forbidden": []}}
        truth = {"identifiers": ["order_id", "zone"], "forbidden": [{"column": "fare"}], "key": "rides"}
        solution = {"target": "late", "task": "binary", "identifiers": ["order_id", "zone"], "forbidden": [{"column": "fare"}]}
        capture.solution_saved(proposal, self.PROFILE, solution, "human", truth, project="p1")
        zone = next(r for r in self.store.records(point="column.is_identifier") if r.subject.key == "zone")
        self.assertEqual((zone.answer("rule").value, zone.answer("person").value), (False, True))  # the rule did not say zone
        labels = {x.record: x for x in self.store.labels()}
        self.assertEqual((labels[zone.id].source, labels[zone.id].value), ("benchmark", True))
        fare = next(r for r in self.store.records(point="feature.leakage_suspect") if r.subject.key == "fare")
        self.assertEqual(fare.answer("rule").value, False)  # the written solution's forbidden column is not the audit's

    def test_a_proposal_for_another_target_answers_nothing(self):
        from dclab_rnd.ledger import capture

        capture.solution_saved({"target": "fare", "identifiers": ["order_id"], "rule": {"identifiers": ["order_id"]}}, self.PROFILE,
                               {"target": "late", "task": "binary", "identifiers": ["order_id"], "forbidden": []}, "human", project="p1")
        ident = next(r for r in self.store.records(point="column.is_identifier") if r.subject.key == "order_id")
        self.assertIsNone(ident.answer("rule"))
        self.assertNotIn("task", next(r for r in self.store.records(point="task.framing")).state)

    def test_who_set_a_pack_and_why_a_run_ended(self):
        from types import SimpleNamespace

        from dclab_rnd.ledger import capture

        capture.pack("d0000001", "fraud", "agent")
        self.assertEqual(self.store.records(point="problem.pack")[0].used, "model_role")
        policy = SimpleNamespace(name="intern", max_steps=10)
        capture.stopped(policy, None, None, False)  # a run that raised
        capture.stopped(policy, SimpleNamespace(stopped="answered", steps=[1, 2]), None)
        self.assertEqual([r.used for r in self.store.records(point="workflow.stop")], ["rule", "model_role"])

    def test_review_rounds_and_the_leakage_measurement(self):
        from dclab_rnd.ledger import capture

        owner = {"id": "p1", "data": {"rows": 500, "columns": ["a", "b"]}, "settings": {}}
        models = {"task_type": "binary", "decision": {"selected": "lightgbm", "chosen": "lightgbm", "options": [{"id": "lightgbm"}, {"id": "ridge"}]},
                  "evidence": {"review": {"rounds": [
                      {"round": 1, "experiment": "monotone", "rule_first": "segment_weights", "options": ["segment_weights", "monotone"], "adopted": True,
                       "critic": {"order": ["monotone", "segment_weights"]}},
                      {"round": 2, "experiment": "segment_weights", "rule_first": "segment_weights", "options": ["segment_weights"], "adopted": False}]}}}
        capture.stage_decisions("models", models, None, owner, project="p1")
        nexts = self.store.records(point="experiment.next")
        self.assertEqual([(r.value_used, r.used) for r in nexts], [("monotone", "model_role"), ("segment_weights", "rule")])
        self.assertEqual(nexts[0].answer("model_role").value, "monotone")
        self.assertEqual([r.value_used for r in self.store.records(point="experiment.accept")], [True, False])
        self.assertEqual(self.store.records(point="model.review")[0].value_used, "change")
        leakage = {"task_type": "binary", "evidence": {"declared_leakage_features": ["fare"], "apparent_lift": 0.21,
                                                       "heuristic_review_candidates": [{"feature": "fare"}, {"feature": "zone"}]}}
        capture.stage_decisions("leakage", leakage, None, owner, project="p1")
        fare, zone = sorted(self.store.records(point="feature.leakage_suspect"), key=lambda r: r.subject.key)
        self.assertEqual((fare.value_used, fare.state["apparent_lift"], zone.value_used), (True, 0.21, False))
        measured = self.store.labels(record=fare.id)
        self.assertEqual([(x.source, x.value, x.evidence.kind) for x in measured], [("measured", True, "apparent_lift")])
        self.assertEqual(self.store.labels(record=zone.id), [])

    def test_a_question_the_agent_could_not_ask_is_none(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        drafts = DraftStore(Path(tempfile.mkdtemp()) / "drafts")
        draft = drafts.create("Predict which orders arrive late")
        agent = HomeAgent(drafts, None)
        with mock.patch.object(agent, "next_question", return_value=("target", "What is the outcome?", [], "why")), \
                mock.patch.object(agent, "ask", return_value=None), mock.patch.object(agent, "summarize"):
            agent.next_turn(draft["id"])
        self.assertEqual([r.value_used for r in self.store.records(point="workflow.next_question")], ["none"])
        with mock.patch.object(agent, "next_question", return_value=("target", "What is the outcome?", [], "why")), \
                mock.patch.object(agent, "ask", return_value=True):
            agent.next_turn(draft["id"])
        self.assertEqual([r.value_used for r in self.store.records(point="workflow.next_question")], ["none", "target"])

    def test_a_saved_solution_is_one_write(self):
        from dclab_rnd.ledger import capture

        with mock.patch.object(self.store, "add", side_effect=AssertionError("one record at a time")):
            written = capture.solution_saved({}, self.PROFILE, {"target": "late", "task": "binary", "identifiers": [], "forbidden": []}, "human", project="p1")
        self.assertGreater(written, 5)
        self.assertEqual(len(self.store.records()), written)


@unittest.skipIf(MISSING, MISSING)
class StoreTest(unittest.TestCase):
    def store(self):
        home = Path(tempfile.mkdtemp())
        if os.environ.get("DCLAB_DATABASE_URL"):
            from dclab_rnd.storage import db

            db.upgrade()
            return ledger.PgLedger(db.workspace_for(home))
        return ledger.FileLedger(home / "ledger", "wtest")

    def test_records_and_labels_are_kept_and_found(self):
        store = self.store()
        rec = ledger.decide("split.strategy", ("project", "p1"), answers=[ledger.answer("rule", "rule:kind_split", "time")], value_used="time",
                            used="rule", state={"has_time": True}, ledger=store, project="p1")
        self.assertIsNotNone(rec)
        lab = ledger.label(rec, "time", "measured", "optimism_gap", evidence={"kind": "optimism_gap", "value": 0.04}, ledger=store)
        self.assertEqual([r.id for r in store.records(project="p1")], [rec.id])
        self.assertEqual([x.id for x in store.labels(record=rec.id)], [lab.id])
        self.assertEqual(store.records(point="model.family"), [])

    def test_a_replay_records_no_decision(self):
        from dclab_rnd import experience

        store = self.store()
        with experience.quiet():
            self.assertIsNone(ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge",
                                            ledger=store, project="p1"))
        self.assertEqual(store.records(), [])

    def test_a_column_state_is_bands_only(self):
        state = ledger.column_state({"name": "customerID_v2", "kind": "categorical", "missing_rate": 0.031, "unique": 40, "unique_ratio": 0.1,
                                     "preview": ["Jane Roe"]})
        self.assertEqual(state, {"column": "customerID_v2", "name_tokens": ["customer", "id", "v2"], "dtype": "categorical",
                                 "uniqueness": "from_10_to_50pct", "nulls": "from_1_to_10pct", "cardinality": "medium"})
        self.assertNotIn("Jane", json.dumps(state))


if __name__ == "__main__":
    unittest.main()
